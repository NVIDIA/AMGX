/* Copyright (c) 2011-2017, NVIDIA CORPORATION. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *  * Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *  * Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *  * Neither the name of NVIDIA CORPORATION nor the names of its
 *    contributors may be used to endorse or promote products derived
 *    from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
 * EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 * PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
 * CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 * EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 * PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
 * OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#include <thrust/count.h>
#include <thrust/functional.h>
#include <thrust/inner_product.h>
#include <thrust/sort.h>
#include <thrust/transform.h>
#include <thrust/transform_scan.h>
#include <thrust_wrapper.h>

#include <energymin/interpolators/em.h>
#include <energymin/interpolators/common.h>
#include <basic_types.h>
#include <types.h>
#include <cutil.h>
#include <fstream>
#include <set>
#include <vector>
#include <algorithm>
#include <cmath> // std::signbit
#include <assert.h>

#include "util.h"

#include "amgx_cublas.h"

namespace amgx
{
namespace energymin
{


/*************************************************************************
* create the interpolation matrix P by computing p and using energy minimization
************************************************************************/


/*************************************************************************
* create the interpolation matrix P on GPU
************************************************************************/

template <typename IndexType, typename ValueType>
__global__
void count_PnnzPerCol_kernel( const IndexType *ArowOffsets, const IndexType *AcolInd, const ValueType *Avalues,
                              const ValueType *Adiag, const IndexType numCoarse,
                              const int *cf_map, const int *fullToCoarseMap, const int *coarse_idx_ptr,
                              int *PnnzPerCol, ValueType *thresh_per_row )
{
    for (int tIdx = threadIdx.x + blockDim.x * blockIdx.x; tIdx < numCoarse; tIdx += blockDim.x * gridDim.x)
    {
        // each thread works on one "coarse" row of A at a time
        const int row_idx = coarse_idx_ptr[tIdx];
        const int ArowBegin = ArowOffsets[row_idx];
        const int ArowEnd   = ArowOffsets[row_idx + 1];
        // Matlab pseudo-code:
        // p = sparse(n, nc);
        // p(coarse,:) = speye(nc);
        // fAj = AD(j,fine)/AD(j,j) < -.1;  // fine neighbors of j with oppositive sign from AD_jj
        // become the non-zeros in initial P.
        ValueType minVal(0), maxVal(0), diag_value(0);
        ValueType fine_avg_thresh = 0.;
        const ValueType thresh = (ValueType)0.25;

        for (int j = ArowBegin; j < ArowEnd; j++)
        {
            ValueType aValue = Avalues[j];

            if ( j != tIdx )
            {
                minVal = min( minVal, aValue );
                maxVal = max( maxVal, aValue );
            }
            else
            {
                diag_value = aValue;
            }
        }

        if ( diag_value < 0.0)
        {
            fine_avg_thresh = thresh * maxVal;
        }
        else
        {
            fine_avg_thresh = thresh * minVal;
        }

        thresh_per_row[tIdx] = fine_avg_thresh;
        int p_col_nnz = 0;

        for (int j = ArowBegin; j < ArowEnd; j++)
        {
            int AjColIndex = AcolInd[j];

            if (AjColIndex == tIdx)
            {
                p_col_nnz++;
                continue;
            }

            if (cf_map[AjColIndex] == FINE)
            {
                if (diag_value * Avalues[j] < diag_value * fine_avg_thresh)
                {
                    p_col_nnz++;
                }
            }
        }

        PnnzPerCol[tIdx] = p_col_nnz;
    }
}


template <typename IndexType, typename ValueType>
__global__
void init_ProwInd_greedy_aggregation_kernel(const IndexType *ArowOffsets, const IndexType *AcolInd, const ValueType *Avalues,
        const ValueType *Adiag, const IndexType AnumRows,
        const int *cf_map, const int *fullToCoarseMap,
        const IndexType *PcolOffsets, IndexType *ProwInd,
        bool *Ma_nzDiagRows)
{
    // for each coarse point form aggregates
    /* Matlab-like pseudo-code
    p = sparse(n, nc);
    p(coarse,:) = speye(nc);
    for (int j : coarse)              // coarse point with FULL index j
      fAj = AD(j,fine)/AD(j,j) < -.1; // fine neighbors of j with oppositive sign from AD_jj
      P(fine(fAj),cj)=1;              // become the non-zeros in initial P.
    end
    */

    // size of P and its column offsets were precomputed.
    for (int tIdx = threadIdx.x + blockDim.x * blockIdx.x; tIdx < AnumRows; tIdx += blockDim.x * gridDim.x)
    {
        // each thread works on one "coarse" row of A at a time (builds a column of P)
        const int ArowBegin = ArowOffsets[tIdx];
        const int ArowEnd   = ArowOffsets[tIdx + 1];

        if (cf_map[tIdx] == COARSE)     // "coarse" row of A
        {
            // P is CSC format since each thread works on each coarse point
            // and is therefore building a single column of P.
            // There are numCoarse columns of P
            // Columns of P correspond to rows of A (A is assumed symmetric)
            bool PcolCoarseDiagUnset = 1;
            const int PcolIndex = fullToCoarseMap[tIdx];
            int PentryNum = PcolOffsets[PcolIndex];  // starting location of the current column of P

            for (int j = ArowBegin; j < ArowEnd; j++)
            {
                // go thru each "fine" column of A
                int AjColIndex = AcolInd[j];

                if (cf_map[AjColIndex] == FINE && Avalues[j] / Adiag[tIdx] < -.1)
                {
                    //p(AjColIndex, fullToCoarseMap(j))=1;
                    if (AjColIndex > tIdx && PcolCoarseDiagUnset)     // COARSE diagonal of P is identity
                    {
                        // set COARSE diagonal of P
                        ProwInd[PentryNum] = tIdx; // set COARSE diagonal entry of P to 1.
                        PentryNum++;
                        PcolCoarseDiagUnset = 0;
                        Ma_nzDiagRows[tIdx] = 1;    // record the newly added row index of P
                    }

                    ProwInd[PentryNum] = AjColIndex; // columns of A become rows in P (since A is assumed symmetric)
                    PentryNum++;
                    Ma_nzDiagRows[AjColIndex] = 1;    // record the newly added row index of P
                }
            }

            if (PcolCoarseDiagUnset)     // COARSE diagonal of P is identity
            {
                // set COARSE diagonal of P
                ProwInd[PentryNum] = tIdx; // columns of A become rows in P (since A is assumed symmetric)
                PentryNum++;
                PcolCoarseDiagUnset = 0;
                Ma_nzDiagRows[tIdx] = 1;    // record the newly added row index of P
            }
        }
    }
}


template <typename IndexType, typename ValueType>
__global__
void  extract_dense_Aijs_col_major_kernel(const IndexType *ArowOffsets, const IndexType *AcolInd, const ValueType *Avalues,
        const IndexType *PcolOffsets, const IndexType *ProwInd, const IndexType PnumCols,
        const IndexType *AijOffsets, ValueType *dense_Aijs)
{
    /* Matlab-like pseudo-code
    for (int j=1; j<=nc; j++)
    {
      // Find the nz pattern in the j-th column of P
      ij = find(p(:,j));              // col vec of subindices
      Aij{j} = A(ij,ij);
    }
    */

    // each thread builds one Aij
    for (unsigned int tIdx = threadIdx.x + blockDim.x * blockIdx.x; tIdx < PnumCols; tIdx += blockDim.x * gridDim.x)
    {
        // each thread works with one column of P
        const unsigned int PcolBegin  = PcolOffsets[tIdx];
        const unsigned int PcolEnd    = PcolOffsets[tIdx + 1];
        const unsigned int AijNumRows = PcolEnd - PcolBegin;
        // pointer to the beginning of the specific Aij submatrix
        ValueType *dense_Aij = dense_Aijs + AijOffsets[tIdx];

        for (unsigned int i = PcolBegin; i < PcolEnd; i++)
        {
            // go thru each non-zero row of the given column of P
            // and scan the corresponding rows of A
            unsigned int PiRowIndex = ProwInd[i];
            const unsigned int ArowBegin = ArowOffsets[PiRowIndex];
            const unsigned int ArowEnd   = ArowOffsets[PiRowIndex + 1];

            for (unsigned int j = ArowBegin; j < ArowEnd; j++)
            {
                // scan the current row of A and pick the elts based on the sparsity pattern of P
                unsigned int AjColIndex = AcolInd[j];

                // if AjColIndex is an index of a non-zero entry in the current column of P, take it.
                for (unsigned int ii = PcolBegin; ii < PcolEnd; ii++)
                {
                    if (ProwInd[ii] == AjColIndex)
                    {
                        // add elt to Aij in column-major format
                        unsigned int AijRowIndex = i - PcolBegin;
                        unsigned int AijColIndex = ii - PcolBegin;
                        dense_Aij[AijColIndex * AijNumRows + AijRowIndex] = Avalues[j]; //wrong
                        break;
                    }
                }
            }
        }
    }
}


// kernel to initialize invAij's to dense identity matrices
template <typename IndexType, typename ValueType>
__global__
void  init_dense_invAijs_col_major_kernel(const IndexType *PcolOffsets, const IndexType PnumCols,
        const IndexType *AijOffsets, ValueType *dense_invAijs)
{
    for (unsigned int tIdx = threadIdx.x + blockDim.x * blockIdx.x; tIdx < PnumCols; tIdx += blockDim.x * gridDim.x)
    {
        // each thread initializes one invAij
        const unsigned int PcolBegin  = PcolOffsets[tIdx];
        const unsigned int PcolEnd    = PcolOffsets[tIdx + 1];
        const unsigned int AijNumRows = PcolEnd - PcolBegin;
        // pointer to the beginning of the specific invAij submatrix
        ValueType *dense_invAij = dense_invAijs + AijOffsets[tIdx];

        for (unsigned int i = 0; i < AijNumRows; i++)
        {
            dense_invAij[i * AijNumRows + i] = 1;
        }
    }
}




__global__
void  count_Ma_nnz_per_row_kernel(const int *PcolOffsets, const int *ProwInd, const int PnumCols,
                                  int *Ma_nnzPerRow, const int AnumRows, const bool *Ma_nzDiagRows)
{
    for (int tIdx = threadIdx.x + blockDim.x * blockIdx.x; tIdx < AnumRows; tIdx += blockDim.x * gridDim.x)
    {
        // every thread works on a single row of Ma (figures out nnz in that row)
        bool rowNotFilled = 1;          // marks whether the Ma row nnz count has been initialized
        int nnzPerRow = 0;

        for (int j = 0; j < PnumCols; j++)
        {
            // each thread goes thru all columns of P
            const int PjColBegin  = PcolOffsets[j];
            const int PjColEnd    = PcolOffsets[j + 1];
            //const int PjCol_nnz   = PjColEnd - PjColBegin;

            // check that the column is relevant (i.e. has a non-zero in the tIdx row)
            if (tIdx < ProwInd[PjColBegin] || ProwInd[PjColEnd - 1] < tIdx) { continue; }

            // Make sure the column is actually relevant (i.e. has a non-zero in the tIdx row)
            bool relevantCol = 0;

            for (int i = PjColBegin; i < PjColEnd; i++)
            {
                if (ProwInd[i] == tIdx) { relevantCol = 1; break; }
            }

            if (!relevantCol) { continue; }

            if (rowNotFilled)
            {
                nnzPerRow = PjColEnd - PjColBegin;  // initial nnz per row of Ma
                rowNotFilled = 0;
                continue;
            }

            // if the row nnz count has been initialized, we need to figure out which elts haven't already appeared
            for (int i = PjColBegin; i < PjColEnd; i++)
            {
                // grab a row in the current column j of P
                const int PiRowIndex = ProwInd[i];
                bool rowPrevAppeared = 0;

                for (int jj = j - 1; jj >= 0; jj--) // scan columns of P backwards since it's more likely that the same row entry
                {
                    // appeared in the direct neighbors of the current column
                    const int PjjColBegin  = PcolOffsets[jj];
                    const int PjjColEnd    = PcolOffsets[jj + 1];

                    // check that the column is relevant (i.e. has a non-zero in the tIdx row)
                    if (tIdx < ProwInd[PjjColBegin] || ProwInd[PjjColEnd - 1] < tIdx) { continue; }

                    if (PiRowIndex < ProwInd[PjjColBegin] || ProwInd[PjjColEnd - 1] < PiRowIndex) { continue; }

                    // make sure the column is actually relevant (i.e. has a non-zero in the tIdx row)
                    bool relevantPcol = 0;

                    for (int ii = PjjColBegin; ii < PjjColEnd; ii++)
                    {
                        if (ProwInd[ii] == tIdx) { relevantPcol = 1; break; }
                    }

                    if (!relevantPcol) { continue; }

                    // see if PiRowIndex has already appeared previously
                    for (int ii = PjjColBegin; ii < PjjColEnd; ii++)
                    {
                        if (ProwInd[ii] == PiRowIndex) { rowPrevAppeared = 1; break; }
                    }

                    if (rowPrevAppeared) { break; } // stop scanning columns of P, since we already accounted for PiRowIndex previously
                }

                if (!rowPrevAppeared)  // if we haven't come across the P column with the nz in the PiRowIndex,
                {
                    // then we can increment the row nnz count
                    nnzPerRow++;
                }
            }
        }

        if (nnzPerRow == 0)           { Ma_nnzPerRow[tIdx] = 1; }
        else if (Ma_nzDiagRows[tIdx]) { Ma_nnzPerRow[tIdx] = nnzPerRow; }
        else                          { Ma_nnzPerRow[tIdx] = nnzPerRow + 1; }
    }
}


/*
template <typename IndexType, typename ValueType>
__global__
void  find_Ma_nz_diag_rows_kernel(const IndexType *PcolOffsets, const IndexType *ProwInd,
                                  const IndexType PnumCols, bool *Ma_nzDiagRows)
{
  for (unsigned int tIdx = threadIdx.x + blockDim.x*blockIdx.x; tIdx < PnumCols; tIdx += blockDim.x*gridDim.x)
  { // each thread goes thru a single column of P

    const unsigned int PcolBegin = PcolOffsets[tIdx];
    const unsigned int PcolEnd   = PcolOffsets[tIdx+1];

    for (unsigned int i=PcolBegin; i<PcolEnd; i++)
    {
      if (tIdx == ProwInd[i])
      {
        Ma_nzDiagRows[tIdx] = 1;  // Mark the non-zero diagonal rows of Ma
        break;
      }
    }

  }
}
*/

__global__
void  get_unsorted_MaColInd_kernel( const int *PcolOffsets, const int *ProwInd, const int PnumCols,
                                    const int *MaRowOffsets, int *MaColInd, const int AnumRows,
                                    const bool *Ma_nzDiagRows )
{
    for (int tIdx = threadIdx.x + blockDim.x * blockIdx.x; tIdx < AnumRows; tIdx += blockDim.x * gridDim.x)
    {
        // each thread works on a single row of Ma (figures out column indices for that row)
        int rowIsFilled = 0;         // indicates that MaColInd has not been initialized
        const int MaRowBegin = MaRowOffsets[tIdx];
        const int MaRowEnd   = MaRowOffsets[tIdx + 1];
        const int MaRowSize  = MaRowEnd - MaRowBegin;

        // Skip empty rows of Ma (should never happen if Ma_nzDiagRows is initialized ??)

        if (MaRowSize <= 1)
        {
            if (MaRowSize == 0)
            {
                printf("\nWARNING: found zero row of Ma in row %d. Ma_nzDiagRows[%d] = %d \n",
                       tIdx, tIdx, Ma_nzDiagRows[tIdx]);
            }
            else { MaColInd[MaRowBegin] = tIdx; }   // diagonal entry is the only one.

            continue;
        }

        int MaIndex = MaRowBegin;    // keeps count of the number of assigned indices to the Ma row

        for (int j = 0; j < PnumCols; j++)
        {
            // thread goes thru all columns of P
            const int PjColBegin = PcolOffsets[j];
            const int PjColEnd   = PcolOffsets[j + 1];

            // Check that the column is relevant (i.e. has a non-zero in the tIdx row)
            if (tIdx < ProwInd[PjColBegin] || ProwInd[PjColEnd - 1] < tIdx) { continue; }

            // Make sure the column is actually relevant (i.e. has a non-zero in the tIdx row)
            bool relevantCol = 0;

            for (int i = PjColBegin; i < PjColEnd; i++)
            {
                if (ProwInd[i] == tIdx) { relevantCol = 1; break; }
            }

            if (!relevantCol) { continue; }

            if (rowIsFilled == 0)  // Simply copy the row indices in the current P column
            {
                // to the column indices of the Ma row
                for (int i = PjColBegin; i < PjColEnd; i++)
                {
                    MaColInd[MaIndex] = ProwInd[i];
                    MaIndex++;          // Increment the number of assigned indices to the current row.
                }

                rowIsFilled = 1;      // indicates that Ma row is initialized but not completely filled
                continue;
            }

            // If the row has been initialized, we need to take
            // only those row indices of P that haven't already appeared
            for (int i = PjColBegin; i < PjColEnd; i++)
            {
                // grab the next row index of the current column j of P
                const int PiRowIndex = ProwInd[i];
                bool indexPrevAppeared = 0;   // indicate that PiRowIndex haven't been previously assigned to the row of Ma

                for (int jj = MaRowBegin; jj < MaIndex; jj++) // scan the accumulated Ma column indices and check whether
                {
                    // the current P row index already exists
                    if (MaColInd[jj] == PiRowIndex)
                    {
                        // the index already exists, so skip it
                        indexPrevAppeared = 1;
                        break;
                    }
                }

                if (!indexPrevAppeared)   // if the index is new to the MaColInd array, add it to the end (i.e. non-sorted)
                {
                    MaColInd[MaIndex] = PiRowIndex;
                    MaIndex++;
                }

                if (MaIndex == MaRowEnd)  // stop searching, since we filled the entire Ma row.
                {
                    rowIsFilled = 2;        // indicates that Ma row is completely filled with column indices
                    break;
                }
            }

            if (rowIsFilled == 2) { break; } // stop searching, since we filled the entire Ma row.
        }

        if (!Ma_nzDiagRows[tIdx]) { MaColInd[MaIndex] = tIdx; }
    } // end for (int tIdx ...)
}





template <typename IndexType, typename ValueType>
__global__
void  add_invAijs_to_Ma_col_major_kernel( const IndexType *PcolOffsets, const IndexType *ProwInd, const IndexType PnumCols,
        const IndexType *AijOffsets, const ValueType *dense_invAijs,
        const IndexType *MaRowOffsets, const IndexType *MaColInd,
        ValueType *MaValues )
{
    // go thru all invAij matrices
    for (unsigned int j = 0; j < PnumCols; j++)
    {
        // every thread runs through every invAij matrix and works on a single row of each invAij
        const int AijOffset = AijOffsets[j];
        const ValueType *invAij = dense_invAijs + AijOffset;
        const unsigned int PcolBegin = PcolOffsets[j];
        // dimension of the current Aij within dense_Aijs
        const unsigned int AijNumRows = PcolOffsets[j + 1] - PcolOffsets[j];
        const unsigned int lda = AijNumRows;  // leading dim of the current Aij (col-major)

        for (unsigned int tIdx = threadIdx.x + blockDim.x * blockIdx.x; tIdx < AijNumRows; tIdx += blockDim.x * gridDim.x)
        {
            // each thread works on a single row of the current invAij
            const unsigned int MaRowIndex = ProwInd[PcolBegin + tIdx];
            // get the bounds of each thread's Ma row
            const unsigned int MaRowBegin = MaRowOffsets[MaRowIndex];
            const unsigned int MaRowEnd   = MaRowOffsets[MaRowIndex + 1];
            unsigned int MaEntryIndex = MaRowBegin;

            for (unsigned int i = 0; i < AijNumRows; i++)
            {
                // go through each element in the given row of invAij and put this element
                // into its corresponding ("spattered") location in Ma
                const unsigned int MaColIndex = ProwInd[PcolBegin + i];

                // there are potentially more nz entries in a given row of Ma
                // than the dimension of the current invAij, hence need this loop
                // (note that we do not restart count since the col indices are stored in order)
                for ( ; MaEntryIndex < MaRowEnd; MaEntryIndex++)
                {
                    if (MaColInd[MaEntryIndex] == MaColIndex)
                    {
                        MaValues[MaEntryIndex] += invAij[tIdx + i * lda];
                        break;
                    }
                }
            }
        }

        // to avoid race conditions when accessing rows of Ma, all threads need to be in sync.
        __syncthreads();
    }
}


// This kernel perturbs the diagonal of Ma (i.e. adds a specified value to each diag entry)
template <typename IndexType, typename ValueType>
__global__
void  perturb_Ma_diag_kernel( const IndexType *MaRowOffsets, const IndexType *MaColInd,
                              ValueType *MaValues, const IndexType AnumRows,
                              const ValueType perturb_mag, const bool *Ma_nzDiagRows )
{
    for (int tIdx = threadIdx.x + blockDim.x * blockIdx.x; tIdx < AnumRows; tIdx += blockDim.x * gridDim.x)
    {
        // each thread perturbs one diagonal entry of Ma
        const int MaRowBegin = MaRowOffsets[tIdx];
        const int MaRowEnd   = MaRowOffsets[tIdx + 1];
        bool diagIsZero = 1;

        for (int j = MaRowBegin; j < MaRowEnd; j++)
        {
            // find the diagonal entry in the current row
            if (tIdx == MaColInd[j])
            {
                diagIsZero = 0;

                if (!Ma_nzDiagRows[tIdx]) { MaValues[j] = perturb_mag; break;}

                MaValues[j] += perturb_mag;
                break;
            }
        }

        if (diagIsZero)
        {
            printf("\nWARNING: found zero diagonal entry of Ma in row %d. Ma_nzDiagRows[%d] = %d \n",
                   tIdx, tIdx, Ma_nzDiagRows[tIdx]);
            break;
        }
    }
}

// This kernel initializes each column of P to v_x.
template <typename IndexType, typename ValueType, typename ValueTypeB>
__global__
void init_Pvalues_kernel( const IndexType *PcolOffsets, const IndexType *ProwInd,
                          const IndexType PnumCols, const ValueTypeB *v_x,
                          ValueType *Pvalues )
{
    for (int tIdx = threadIdx.x + blockDim.x * blockIdx.x; tIdx < PnumCols; tIdx += blockDim.x * gridDim.x)
    {
        // each thread works on one column of P in CSC format (P^T in CSR format)
        const unsigned int PcolBegin = PcolOffsets[tIdx];
        const unsigned int PcolEnd   = PcolOffsets[tIdx + 1];

        for (int j = PcolBegin; j < PcolEnd; j++)
        {
            // go thru each entry in the given column of P
            Pvalues[j] = v_x[ProwInd[j]];
        }
    }
}



////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <AMGX_VecPrecision t_vecPrec, AMGX_MatPrecision t_matPrec, AMGX_IndPrecision t_indPrec>
EM_Interpolator<TemplateConfig<AMGX_device, t_vecPrec, t_matPrec, t_indPrec> >::
EM_Interpolator() : EM_InterpolatorBase<TConfig_d>(),
    m_cuds_handle(0),
    m_dense_Aijs(0),
    m_dense_invAijs(0),
    m_ipiv(0),
    m_cuds_info(0),
    m_cuds_wspace(0)
{
    // Allocate a handle for cudense
    cusolverStatus_t status = cusolverDnCreate(&m_cuds_handle);

    if ( status != CUSOLVER_STATUS_SUCCESS )
    {
        FatalError( "Could not create the CUDENSE handle", AMGX_ERR_CUDA_FAILURE );
    }

    // Define the cudense stream.
    status = cusolverDnSetStream(m_cuds_handle, thrust::global_thread_handle::get_stream());

    if ( status != CUSOLVER_STATUS_SUCCESS )
    {
        FatalError( "Could not set the stream for CUDENSE", AMGX_ERR_CUDA_FAILURE );
    }

    allocMem(m_cuds_info, sizeof(int), false);
}


template <AMGX_VecPrecision t_vecPrec, AMGX_MatPrecision t_matPrec, AMGX_IndPrecision t_indPrec>
EM_Interpolator<TemplateConfig<AMGX_device, t_vecPrec, t_matPrec, t_indPrec> >::~EM_Interpolator()
{
    if (m_cuds_handle)
    {
        cusolverDnDestroy(m_cuds_handle);
    }

    if (m_dense_Aijs)
    {
        thrust::global_thread_handle::cudaFreeAsync(m_dense_Aijs);
        m_dense_Aijs = 0;
    }

    if (m_dense_invAijs)
    {
        thrust::global_thread_handle::cudaFreeAsync(m_dense_invAijs);
        m_dense_invAijs = 0;
    }

    if (m_ipiv)
    {
        thrust::global_thread_handle::cudaFreeAsync(m_ipiv);
        m_ipiv = 0;
    }

    if (m_cuds_wspace)
    {
        thrust::global_thread_handle::cudaFreeAsync(m_cuds_wspace);
        m_cuds_wspace = 0;
    }

    if (m_cuds_info)
    {
        thrust::global_thread_handle::cudaFreeAsync(m_cuds_info);
        m_cuds_info = 0;
    }
}



// Private functions for the implementation on device

// Find the sparse dimensions of P (i.e. number of non-zeros in each column of P)
template <AMGX_VecPrecision t_vecPrec, AMGX_MatPrecision t_matPrec, AMGX_IndPrecision t_indPrec>
void EM_Interpolator<TemplateConfig<AMGX_device, t_vecPrec, t_matPrec, t_indPrec> >
::computePsparsity( const Matrix_d &A, const IVector &cf_map, const IVector &coarse_idx,
                    Matrix_d &P, IVector &PnnzPerCol,
                    BVector &Ma_nzDiagRows )
{
    typedef typename Matrix_d::index_type IndexType;
    typedef typename Matrix_d::value_type ValueType;
    // Assemble all the raw pointers needed
    // Matrix A
    const IndexType *ArowOffsets_ptr = A.row_offsets.raw();
    const IndexType *AcolInd_ptr     = A.col_indices.raw();
    const ValueType *Avalues_ptr     = A.values.raw();
    const IndexType AnumRows = (IndexType) A.get_num_rows();
    // choose blocksize. Using 1 thread / row for now
    const int blocksize = 64;
    const int numBlocks = std::min (AMGX_GRID_MAX_SIZE, (int) (AnumRows / blocksize + 1));
    // temporary vectors - diagonal and non-zero offsets
    VVector Adiag(A.get_num_rows(), 0.0);
    ValueType *Adiag_ptr  = Adiag.raw();
    // fullToCoarseMap array gives the COARSE index corresponding to the given FULL index
    // (meaningless if the full index corresponds to a FINE pt)
    IntVector fullToCoarseMap(AnumRows, 0);
    int *fullToCoarseMap_ptr = fullToCoarseMap.raw();
    thrust::transform_exclusive_scan( cf_map.begin(), cf_map.end(), fullToCoarseMap.begin(),
                                      is_coarse(), 0, thrust::plus<int>() );
    cudaCheckError();
    // Extract the diagonal of A
    find_diag_kernel_indexed_dia <<< numBlocks, blocksize>>>(AnumRows, A.diag.raw(), Avalues_ptr, Adiag_ptr);
    cudaCheckError();
    // Coarse/Fine map
    const int *cf_map_ptr = cf_map.raw();
    // Find the sparse dimensions of P (i.e. number of non-zeros in each column of P)
    // Count the number of non-zeros in the interpolation if
    const int numCoarse = coarse_idx.size();
    const int *coarse_idx_ptr = coarse_idx.raw();
    VVector thresh_per_row(AnumRows);
    count_PnnzPerCol_kernel<IndexType, ValueType>
    <<< numBlocks, blocksize>>>(ArowOffsets_ptr, AcolInd_ptr, Avalues_ptr,
                                Adiag_ptr, numCoarse,
                                cf_map_ptr, fullToCoarseMap_ptr, coarse_idx_ptr,
                                PnnzPerCol.raw(), thresh_per_row.raw());
    printMatrixStats(A);
    cudaCheckError();
    // Column offsets of P (CSC format)
    IntVector PnzOffsets(numCoarse + 1);
    PnzOffsets[0] = 0;
    // get the offsets in P with an inclusive scan
    thrust_wrapper::inclusive_scan(PnnzPerCol.begin(), PnnzPerCol.end(), PnzOffsets.begin() + 1);
    cudaCheckError();
    // get total num of non-zeros in P
    const int Pnnz = PnzOffsets[numCoarse];
    // resize P
    P.resize(0, 0, 0, 1);
    P.addProps(CSR);  // IMPORTANT: P actually stores P^T in CSR format
    //            because we build P by columns (and CSC format is not supported)
    P.resize(numCoarse, AnumRows, Pnnz, 1);
    // set P offsets (P column offsets or P^T row offsets)
    thrust::copy(PnzOffsets.begin(), PnzOffsets.end(), P.row_offsets.begin());
    cudaCheckError();
    // treat P as if it were in the CSC format.
    const IndexType *PcolOffsets_ptr = P.row_offsets.raw();
    IndexType *ProwInd_ptr           = P.col_indices.raw();
    bool *Ma_nzDiagRows_ptr = Ma_nzDiagRows.raw();
    // Since CSC format is not supported in AMGX matrix, we instead
    // compute P^T (which is stored in P) in the CSR format.
    // Initialize P by performing greedy aggregation based on the coarse set (cf_map)
    // The values of P remain uninitialized, since at this point we only get the sparsity pattern of P.
    init_ProwInd_greedy_aggregation_kernel<IndexType, ValueType>
    <<< numBlocks, blocksize>>>(ArowOffsets_ptr, AcolInd_ptr, Avalues_ptr,
                                Adiag_ptr, AnumRows,
                                cf_map_ptr, fullToCoarseMap_ptr,
                                PcolOffsets_ptr, ProwInd_ptr,
                                Ma_nzDiagRows_ptr);
    cudaCheckError();
}


// Compute the submatrices Aij's of A and their inverses invAij's
template <AMGX_VecPrecision t_vecPrec, AMGX_MatPrecision t_matPrec, AMGX_IndPrecision t_indPrec>
void EM_Interpolator<TemplateConfig<AMGX_device, t_vecPrec, t_matPrec, t_indPrec> >
::computeAijSubmatrices(const Matrix_d &A, const int numCoarse, const Matrix_d &P,
                        ValueType *dense_Aijs, ValueType *dense_invAijs,
                        const IntVector &AijOffsets, int *ipiv,
                        cusolverDnHandle_t &cuds_handle, int *cuds_info)
{
    typedef typename Matrix_d::index_type IndexType;
    typedef typename Matrix_d::value_type ValueType;
    // choose blocksize. Using 1 thread / row for now
    const int blocksize = 64;
    const int numBlocks = std::min (AMGX_GRID_MAX_SIZE, (int) (A.get_num_rows() / blocksize + 1));
    // Assemble all the raw pointers needed
    // Matrix A
    const IndexType *ArowOffsets_ptr = A.row_offsets.raw();
    const IndexType *AcolInd_ptr     = A.col_indices.raw();
    const ValueType *Avalues_ptr     = A.values.raw();
    // Treat P as if it were in the CSC format.
    const IndexType *PcolOffsets_ptr = P.row_offsets.raw();
    const IndexType *ProwInd_ptr     = P.col_indices.raw();
    const IndexType *AijOffsets_ptr  = AijOffsets.raw();
    // Extract Aij's from A based on the sparsity pattern of each column of P.
    // Column major format is used for storage of Aij's.
    extract_dense_Aijs_col_major_kernel<IndexType, ValueType>
    <<< numBlocks, blocksize>>>(ArowOffsets_ptr, AcolInd_ptr, Avalues_ptr,
                                PcolOffsets_ptr, ProwInd_ptr, numCoarse,
                                AijOffsets_ptr, dense_Aijs);
    cudaCheckError();
    // Initialize invAij's to (dense) identity matrices.
    init_dense_invAijs_col_major_kernel<IndexType, ValueType>
    <<< numBlocks, blocksize>>>(PcolOffsets_ptr, numCoarse,
                                AijOffsets_ptr, dense_invAijs);
    cudaCheckError();
    // Sequentially perform dense LU factorization on all Aij's
    // LU factors stored in place of the Aij's
    // Then, perform inversion of Aij submatrices by solving with RHS == identity matrix
    cusolverStatus_t cudsStat;

    if (!cuds_handle)
    {
        cusolverStatus_t status = cusolverDnCreate(&m_cuds_handle);

        if ( status != CUSOLVER_STATUS_SUCCESS )
        {
            FatalError( "Could not create the CUDENSE handle", AMGX_ERR_CUDA_FAILURE );
        }

        // Define the cudense stream.
        status = cusolverDnSetStream(m_cuds_handle, thrust::global_thread_handle::get_stream());

        if ( status != CUSOLVER_STATUS_SUCCESS )
        {
            FatalError( "Could not set the stream for CUDENSE", AMGX_ERR_CUDA_FAILURE );
        }
    }

    // Pointer to info from cudense
    if (!cuds_info) { cuds_info = (int *) malloc(sizeof(int)); }

    // Offset of cudense pivoting sequence for each Aij.
    int ipivOffset = 0;
    IVector_h sizes(numCoarse);
    int max_wsize = 0;

    for (int j = 0; j < numCoarse; j++)
    {
        int cur_wsize = 0;
        int AijOffset  = AijOffsets[j];
        int AijNumRows = P.row_offsets[j + 1] - P.row_offsets[j];
        int lda = AijNumRows;
        cusolverStatus_t status = cusolverDnXgetrf_bufferSize(cuds_handle, AijNumRows, AijNumRows, dense_Aijs + AijOffset, lda, &cur_wsize);

        if (status != CUSOLVER_STATUS_SUCCESS)
        {
            FatalError("failed to perform LU factorization", AMGX_ERR_INTERNAL);
        }

        max_wsize = std::max(cur_wsize, max_wsize);
    }

    allocMem(m_cuds_wspace, max_wsize, false);

    for (int j = 0; j < numCoarse; j++)
    {
        int AijOffset  = AijOffsets[j];                           // offset of the current Aij
        int AijNumRows = P.row_offsets[j + 1] - P.row_offsets[j]; // dimension of the next Aij within dense_Aijsc
        int lda = AijNumRows;                                     // leading dim for the next Aij (col-major)
        sizes[j] = AijNumRows;
        // [L,U,P] = lu(Aij) (LU factors stored in place of the current Aij)
        cudsStat = cusolverDnXgetrf(cuds_handle, AijNumRows, AijNumRows,
                                    dense_Aijs + AijOffset, lda, m_cuds_wspace,
                                    ipiv + ipivOffset, cuds_info);

        if (cudsStat != CUSOLVER_STATUS_SUCCESS)
        {
            FatalError("failed to perform LU factorization", AMGX_ERR_INTERNAL);
        }

        // invAij = U\(L\P) (i.e. find inverse of Aij).
        cudsStat = cusolverDnXgetrs(cuds_handle, CUBLAS_OP_N, AijNumRows, AijNumRows,
                                    dense_Aijs + AijOffset, lda,
                                    ipiv + ipivOffset, dense_invAijs + AijOffset, lda, cuds_info );

        if (cudsStat != CUSOLVER_STATUS_SUCCESS)
        {
            FatalError("failed to perform triangular solve", AMGX_ERR_INTERNAL);
        }

        ipivOffset += AijNumRows;   // starting offset of the next pivoting sequence
    }
}


// Compute Ma matrix
template <AMGX_VecPrecision t_vecPrec, AMGX_MatPrecision t_matPrec, AMGX_IndPrecision t_indPrec>
void EM_Interpolator<TemplateConfig<AMGX_device, t_vecPrec, t_matPrec, t_indPrec> >
::computeMa(Matrix_d &Ma, const int AnumRows, const int numCoarse, const Matrix_d &P,
            const ValueType *dense_invAijs, const IntVector &AijOffsets,
            const BVector &Ma_nzDiagRows,
            const bool perturb_Ma_diag, const ValueType perturb_mag)
{
    typedef typename Matrix_d::index_type IndexType;
    typedef typename Matrix_d::value_type ValueType;
    // choose blocksize. Using 1 thread / row for now
    const int blocksize = 64;
    const int numBlocks = std::min (AMGX_GRID_MAX_SIZE, (int) (AnumRows / blocksize + 1));
    // Treat P as if it were in CSC format.
    const IndexType *PcolOffsets_ptr = P.row_offsets.raw();
    const IndexType *ProwInd_ptr     = P.col_indices.raw();
    // Number of non-zeros in each row of Ma
    IntVector Ma_nnzPerRow(AnumRows, 0);
    // 1. Compute Ma_nnzPerRow (kernel).
    count_Ma_nnz_per_row_kernel <<< numBlocks, blocksize>>>( PcolOffsets_ptr, ProwInd_ptr, numCoarse,
            Ma_nnzPerRow.raw(), AnumRows, Ma_nzDiagRows.raw() );
    cudaCheckError();
    // 2. Resize Ma and assign MaRowOffsets by performing a scan on Ma_nnzPerRow.
    // Get the offsets in Ma with an inclusive scan (in-place)
    thrust_wrapper::inclusive_scan(Ma_nnzPerRow.begin(), Ma_nnzPerRow.end(), Ma_nnzPerRow.begin());
    cudaCheckError();
    // Get total num of non-zeros in Ma
    const int Ma_nnz = Ma_nnzPerRow[AnumRows - 1];
    // Resize Ma
    Ma.resize(0, 0, 0, 1);
    Ma.addProps(CSR);
    Ma.resize(AnumRows, AnumRows, Ma_nnz, 1);
    Ma.row_offsets[0] = 0;
    // set Ma row offsets
    thrust::copy(Ma_nnzPerRow.begin(), Ma_nnzPerRow.end(), Ma.row_offsets.begin() + 1);
    cudaCheckError();
    const IndexType *MaRowOffsets_ptr = Ma.row_offsets.raw();
    IndexType *MaColInd_ptr           = Ma.col_indices.raw();
    ValueType *MaValues_ptr           = Ma.values.raw();
    // 3. Get the unsorted column indices of Ma.
    get_unsorted_MaColInd_kernel <<< numBlocks, blocksize>>>(PcolOffsets_ptr, ProwInd_ptr, numCoarse,
            MaRowOffsets_ptr, MaColInd_ptr, AnumRows,
            Ma_nzDiagRows.raw());
    cudaCheckError();

    // 4. Sort column indices to get the final MaColInd.
    for (unsigned int MaRow = 0; MaRow < AnumRows; MaRow++)
    {
        int MaRowBegin = Ma.row_offsets[MaRow];
        int MaRowEnd   = Ma.row_offsets[MaRow + 1];

        if (MaRowEnd - MaRowBegin > 1)
        {
            thrust_wrapper::sort( Ma.col_indices.begin() + Ma.row_offsets[MaRow],
                          Ma.col_indices.begin() + Ma.row_offsets[MaRow + 1] );
        }
    }

    // 5. Compute MaValues by summing up the "spattered" inverses of Aij submatrices.
    add_invAijs_to_Ma_col_major_kernel<IndexType, ValueType>
    <<< numBlocks, blocksize>>>(PcolOffsets_ptr, ProwInd_ptr, numCoarse,
                                AijOffsets.raw(), dense_invAijs,
                                MaRowOffsets_ptr, MaColInd_ptr,
                                MaValues_ptr);
    cudaCheckError();

    // Perturb the diagonal of Ma here (prevents Ma from being singular)
    // Ma = 1.0e-8*speye(n,n);
    if (perturb_Ma_diag)
    {
        // Default: perturb_mag = 1.0e-8
        perturb_Ma_diag_kernel<IndexType, ValueType>
        <<< numBlocks, blocksize>>>(MaRowOffsets_ptr, MaColInd_ptr,
                                    MaValues_ptr, AnumRows,
                                    perturb_mag, Ma_nzDiagRows.raw());
        cudaCheckError();
    }

    Ma.set_initialized(1);
}


// Solve for x: Ma*x=e (solved with pcg)
template <AMGX_VecPrecision t_vecPrec, AMGX_MatPrecision t_matPrec, AMGX_IndPrecision t_indPrec>
void EM_Interpolator<TemplateConfig<AMGX_device, t_vecPrec, t_matPrec, t_indPrec> >
::solveMa_e(Matrix_d &Ma, const int AnumRows, Vector_d &v_x)
{
    typedef typename Matrix_d::value_type ValueType;
    // 1. Initialize x (vector of 0's) and e (vector of 1's)
    v_x.resize(AnumRows, 0.0);
    Vector_d v_e(AnumRows, 1.0);
    // ------------------------------ define pcg solver ---------------------------------
    // temporary config and pointer to main config (json format)
    AMG_Config cfg;
    std::string cfg_string = "{ \"config_version\": 2";
    cfg_string += ", \"solver\": { \"scope\": \"pcg_scope\", \"solver\": \"PCG\"";   // Set the solver to PCG
    cfg_string += " ,\"preconditioner\": \"NOSOLVER\"";    // IMPORTANT to have this in order to allocate PCG
    cfg_string += " } }";
    cfg.parseParameterString(cfg_string.c_str());
    // define and allocate solver
    Solver<TConfig_d> *Ma_solver = 0;
    Ma_solver = SolverFactory<TConfig_d>::allocate(cfg, "default", "solver");
    // ------------------------------ end define pcg ------------------------------------
    const int Ma_solveIts = 20;
    const ValueType Ma_solveTol = (ValueType) 1.e-4;
    Ma_solver->setup(Ma, false);
    Ma_solver->set_max_iters(Ma_solveIts);
    Ma_solver->setTolerance(Ma_solveTol);
    // Apply pcg to solve x = Ma \ e
    Ma_solver->solve( v_e, v_x, true );
    Vector_d v_res(AnumRows, 0.);
    Ma_solver->compute_residual( v_e, v_x, v_res );

    if (Ma_solver) { delete Ma_solver; Ma_solver = 0; }
}


// Compute the values of P (using LU factorization of Aij submatrices)
template <AMGX_VecPrecision t_vecPrec, AMGX_MatPrecision t_matPrec, AMGX_IndPrecision t_indPrec>
void EM_Interpolator<TemplateConfig<AMGX_device, t_vecPrec, t_matPrec, t_indPrec> >
::computePvalues( const int AnumRows, const int numCoarse, Matrix_d &P, const Vector_d &v_x,
                  const ValueType *dense_Aijs, const IntVector &AijOffsets, const int *ipiv,
                  cusolverDnHandle_t &cuds_handle, int *cuds_info )
{
    typedef typename Matrix_d::index_type IndexType;
    typedef typename Matrix_d::value_type ValueType;
    typedef typename Vector_d::value_type ValueTypeB;
    // choose blocksize. Using 1 thread / row for now
    const int blocksize = 64;
    const int numBlocks = std::min (AMGX_GRID_MAX_SIZE, (int) (AnumRows / blocksize + 1));
    // Treat P as if it were in CSC format.
    const IndexType *PcolOffsets_ptr = P.row_offsets.raw();
    const IndexType *ProwInd_ptr     = P.col_indices.raw();
    ValueType *Pvalues_ptr           = P.values.raw();
    init_Pvalues_kernel<IndexType, ValueType, ValueTypeB>
    <<< numBlocks, blocksize>>>(PcolOffsets_ptr, ProwInd_ptr, numCoarse,
                                v_x.raw(), Pvalues_ptr);
    cudaCheckError();
    cusolverStatus_t cudsStat;

    if (!cuds_handle)
    {
        cusolverStatus_t status = cusolverDnCreate(&m_cuds_handle);

        if ( status != CUSOLVER_STATUS_SUCCESS )
        {
            FatalError( "Could not create the CUDENSE handle", AMGX_ERR_CUDA_FAILURE );
        }

        // Define the cudense stream.
        status = cusolverDnSetStream(m_cuds_handle, thrust::global_thread_handle::get_stream());

        if ( status != CUSOLVER_STATUS_SUCCESS )
        {
            FatalError( "Could not set the stream for CUDENSE", AMGX_ERR_CUDA_FAILURE );
        }
    }

    // Pointer to info from cudense
    if (!cuds_info) { cuds_info = (int *) malloc(sizeof(int)); }

    // Offset of cudense pivoting sequence for each Aij.
    int ipivOffset = 0;

    for (int j = 0; j < numCoarse; j++)
    {
        int AijOffset  = AijOffsets[j];
        int PcolOffset = P.row_offsets[j];
        // dimension of the next Aij within dense_Aijs
        int AijNumRows = P.row_offsets[j + 1] - P.row_offsets[j];
        int lda = AijNumRows;  // leading dim for the next Aij (col-major)

        if (lda == 0) { continue; }

        // invAij = U\(L\P) (i.e. find inverse of Aij).
        cudsStat = cusolverDnXgetrs(cuds_handle, CUBLAS_OP_N, AijNumRows, 1,
                                    dense_Aijs + AijOffset, lda,
                                    ipiv + ipivOffset, Pvalues_ptr + PcolOffset, lda, cuds_info);

        if (cudsStat != CUSOLVER_STATUS_SUCCESS)
        { FatalError("failed to perform triangular solve", AMGX_ERR_INTERNAL); }

        ipivOffset += AijNumRows;   // starting offset of the next pivoting sequence
    }

    if (cuds_info)  { free(cuds_info);  cuds_info = 0; }
}



// Compute the values of P (using inverses of Aij submatrices)
template <AMGX_VecPrecision t_vecPrec, AMGX_MatPrecision t_matPrec, AMGX_IndPrecision t_indPrec>
void EM_Interpolator<TemplateConfig<AMGX_device, t_vecPrec, t_matPrec, t_indPrec> >
::computePvalues( const int AnumRows, const int numCoarse, Matrix_d &P, const Vector_d &v_x,
                  const ValueType *dense_invAijs, const IntVector &AijOffsets )
{
    typedef typename Matrix_d::index_type IndexType;
    typedef typename Matrix_d::value_type ValueType;
    typedef typename Vector_d::value_type ValueTypeB;
    // choose blocksize. Using 1 thread / row for now
    const int blocksize = 64;
    const int numBlocks = std::min (AMGX_GRID_MAX_SIZE, (int) (AnumRows / blocksize + 1));
    // Treat P as if it were in CSC format.
    const IndexType *PcolOffsets_ptr = P.row_offsets.raw();
    const IndexType *ProwInd_ptr     = P.col_indices.raw();
    ValueType *Pvalues_ptr           = P.values.raw();
    init_Pvalues_kernel<IndexType, ValueType, ValueTypeB>
    <<< numBlocks, blocksize>>>(PcolOffsets_ptr, ProwInd_ptr, numCoarse,
                                v_x.raw(), Pvalues_ptr);
    cudaCheckError();
    ValueType *place_holder_ptr = 0;
    allocMem(place_holder_ptr, sizeof(ValueType), false);
    ValueType one = (ValueType) 1.;
    ValueType zero = (ValueType) 0.;

    for (int j = 0; j < numCoarse; j++)
    {
        int AijOffset  = AijOffsets[j];
        int PcolOffset = P.row_offsets[j];
        // dimension of the next Aij within dense_Aijs
        int AijNumRows = P.row_offsets[j + 1] - P.row_offsets[j];
        int lda = AijNumRows;  // leading dim for the next Aij (col-major)

        if (lda == 0) { continue; }

        Cublas::gemv(false, AijNumRows, AijNumRows, &one,
                     dense_invAijs + AijOffset, lda,
                     Pvalues_ptr + PcolOffset, 1,
                     &zero, place_holder_ptr, 1);
        cudaDeviceSynchronize();
    }

    if (place_holder_ptr) { thrust::global_thread_handle::cudaFreeAsync(place_holder_ptr); place_holder_ptr = 0; }
}



// ----------------------------
//  specialization for device
// ----------------------------
template <AMGX_VecPrecision t_vecPrec, AMGX_MatPrecision t_matPrec, AMGX_IndPrecision t_indPrec>
void EM_Interpolator<TemplateConfig<AMGX_device, t_vecPrec, t_matPrec, t_indPrec> >
::generateInterpolationMatrix_1x1(const Matrix_d &A, const IntVector &cf_map,
                                  Matrix_d &P, void *amg)
{
    typedef typename Matrix_d::index_type IndexType;
    typedef typename Matrix_d::value_type ValueType;
    // Number of rows in Matrix A
    const IndexType AnumRows = (IndexType) A.get_num_rows();
    // Count the number of coarse points
    const int numCoarse = (int) thrust::count(cf_map.begin(), cf_map.end(), (int)COARSE);
    cudaCheckError();
    IVector coarse_idx(numCoarse, -1);
    thrust::counting_iterator<int> zero(0);
    thrust::counting_iterator<int> zero_plus_nc = zero + numCoarse;
    thrust::copy_if(zero, zero_plus_nc, cf_map.begin(), coarse_idx.begin(), is_coarse());
    // ..................................................................................
    // ------------------------ Begin Compute P code ------------------------------------
    // Store the number of non-zeros in each column of the interpolation matrix P
    IntVector PnnzPerCol(numCoarse);
    BVector Ma_nzDiagRows(AnumRows, 0);
    // Find the sparse dimensions of P (i.e. number of non-zeros in each column of P)
    computePsparsity(A, cf_map, coarse_idx, P, PnnzPerCol, Ma_nzDiagRows);
    // Get total num of non-zeros in P
    int Pnnz = P.row_offsets[numCoarse];
    // ..................................................................................
    // Now follows Jacobi smoothing (skip for now since it might be unnecessary)
    // ..................................................................................
    // -------------------------- End Compute P code ------------------------------------
    // ..................................................................................
    // ..................................................................................
    // ----------------------- Begin Energy Minimization Code ---------------------------
    // Given:
    //        A: the problem matrix
    //        P: sparsity pattern for P (n x nc)
    // Main loop (Ma here is the inverse of Ma in the paper)
    // I. Build Ma.
    // 0. Allocate space for storing LU factors for each Aij submatrix of A (use dense format)
    //    There are numCoarse of LU factors.
    //    Since the number of non-zeros in each column of P determines the size of each Aij,
    //    we know the dimensions of all Aij based on the row offsets of P^T.
    // Calculate offset of each Aij matrix in the dense_Aijs array
    IntVector AijOffsets(numCoarse + 1);
    AijOffsets[0] = 0;
    // Get the offsets with an inclusive scan (nnz of each column of P is the dimension of each Aij)
    thrust::transform_inclusive_scan( PnnzPerCol.begin(), PnnzPerCol.end(), AijOffsets.begin() + 1,
                                      square(), thrust::plus<int>() );
    cudaCheckError();
    PnnzPerCol.clear();
    // Cumulative num of entries in all (dense) Aijs (last entry in the offsets)
    const int Aijs_size = AijOffsets[numCoarse];
    // Allocate memory for the dense Aij submatrices and initialize to zero.
    allocMem(m_dense_Aijs, Aijs_size, true);
    // Allocate memory for the dense invAij's and initialize to zero.
    allocMem(m_dense_invAijs, Aijs_size, true);
    // Allocate memory for cudense pivoting sequence.
    allocMem(m_ipiv, Pnnz, false);
    // Compute the submatrices Aij's of A and the inverses invAij's
    computeAijSubmatrices(A, numCoarse, P, m_dense_Aijs, m_dense_invAijs,
                          AijOffsets, m_ipiv, m_cuds_handle, m_cuds_info);
    // Ma matrix build sequence.
    // 1. Compute Ma_nnzPerRow (kernel).
    // 2. Resize Ma and assign MaRowOffsets by performing a scan on Ma_nnzPerRow (thrust).
    // 3. Get the unsorted column indices of Ma (kernel).
    // 4. Sort column indices to get the final MaColInd (thrust).
    // 5. Compute MaValues by summing up the "spattered" inverses of Aij submatrices (kernel).
    // Declare Ma
    Matrix_d Ma;
    // Compute Ma matrix
    computeMa(Ma, AnumRows, numCoarse, P, m_dense_invAijs, AijOffsets, Ma_nzDiagRows);

    if (m_dense_invAijs) { thrust::global_thread_handle::cudaFreeAsync(m_dense_invAijs);  m_dense_invAijs = 0; }

    /* At this point, we are done with the following Matlab code analogue.
    for (int j=1; j<=nc; j++)
    {
      // Find the dofs in the j-th block
      ij = find(p(:,j));              // col vec of subindices
      ni = length(ij);
      It = sparse(n,ni);
      It(ij,:) = speye(ni);
      Aij{j} = inv(A(ij,ij));
      // Form Ma
      // A(ii,ii): submatrix of A corresponding to the dofs in the i-th block
      Ma = Ma + It*(Aij{j}*It_T);
    }
    */
    Ma_nzDiagRows.clear();
    // II. Solve for x: Ma*x=e
    //m_Mae_norm = (ValueType) 0.;
    Vector_d v_x;
    solveMa_e(Ma, AnumRows, v_x);
    cudaDeviceSynchronize();
    Ma.set_initialized(0);
    Ma.resize(0, 0, 0, 1);
    /*  Now, we are done with the following Matlab code analogue.
    e = ones(n,1);
    Mae = zeros(size(e));  // Ma\e;
    [Mae,flag,relres,iter,resvecCG] = pcg(Ma,e,1e-4,100);
    */
    // III. Construct the columns of P in parallel.
    // P is already initialized (and at this point stores P^T in CSR format)
    // We will be performing the following operation to build each column of P
    // P(:,j) = It{j}*(invAij{j}*(It^T*x));
    // This is done with the following steps.
    // First, apply It{j}^T mask to x to get x{j} (project x onto the dimension of Aij{j}).
    // Then solve Aij\x{j}
    // Compute the values of P
    computePvalues( AnumRows, numCoarse, P, v_x, m_dense_Aijs, AijOffsets, m_ipiv, m_cuds_handle, m_cuds_info );

    if (m_dense_Aijs) { thrust::global_thread_handle::cudaFreeAsync(m_dense_Aijs);  m_dense_Aijs = 0; }

    if (m_ipiv)       { thrust::global_thread_handle::cudaFreeAsync(m_ipiv);        m_ipiv = 0; }

    if (m_cuds_info)  { thrust::global_thread_handle::cudaFreeAsync(m_cuds_info);   m_cuds_info = 0; }

    //if (m_dense_invAijs) { thrust::global_thread_handle::cudaFreeAsync(m_dense_invAijs);  m_dense_invAijs = 0; }
    /* Now, we are done building columns of P (stored as rows of P^T in CSR).
     * Here's the equivalent matlab code.
    P = zeros(n,nc);
    for (int j=1; j<=nc; j++)
    {
      // Find the dofs in the j-th block
      ij=find(p(:,j));
      ni = length(ij);
      It = zeros(n,ni);
      It(ij,:) = eye(ni);

      // Form j-th column of P
      P(:,j) = It*(Aij{j}*(It_T*Mae));
    }
    */
    //TODO: truncate rows/cols of P to keep the sparsity low on all levels.
    // ------------------------- End Energy Minimization Code ---------------------------
    // ..................................................................................
}



template< class T_Config>
void EM_InterpolatorBase<T_Config>
::generateInterpolationMatrix(const Matrix<T_Config> &A, const IntVector &cf_map,
                              Matrix<T_Config> &P, void *amg)
{
    P.set_initialized(0);

    if (A.get_block_size() == 1)
    {
        generateInterpolationMatrix_1x1(A, cf_map, P, amg);
    }
    else
    {
        FatalError("Unsupported block size for distance1 interpolator", AMGX_ERR_NOT_SUPPORTED_BLOCKSIZE);
    }

    P.computeDiagonal();
    P.set_initialized(1);
}

#define AMGX_CASE_LINE(CASE) template class EM_InterpolatorBase<TemplateMode<CASE>::Type>;
AMGX_FORALL_BUILDS(AMGX_CASE_LINE)
#undef AMGX_CASE_LINE

#define AMGX_CASE_LINE(CASE) template class EM_Interpolator<TemplateMode<CASE>::Type>;
AMGX_FORALL_BUILDS(AMGX_CASE_LINE)
#undef AMGX_CASE_LINE

} // namespace energymin
} // namespace amgx

