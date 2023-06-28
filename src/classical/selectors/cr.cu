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

#include "solvers/solver.h"
#include <classical/selectors/cr.h>
#include <cutil.h>
#include <util.h>
#include <types.h>

#include <thrust/count.h>
#include <thrust/inner_product.h>
#include <thrust/iterator/constant_iterator.h>
#include <thrust/extrema.h>
#include <thrust/reduce.h>
#include <thrust/functional.h>
#include <thrust/random.h>
#include <thrust/transform.h>
#include <thrust_wrapper.h>
#include <solvers/block_common_solver.h>


namespace amgx
{

namespace classical
{

///////////////////////////////////////////////////////////////////////////////////////////////////

/*************************************************************************
 * Generate uniform random vector
 ************************************************************************/
struct prg
{
    float a, b;

    __host__ __device__
    prg(float _a = 0.f, float _b = 1.f) : a(_a), b(_b) {};

    __host__ __device__
    float operator()(const unsigned int n) const
    {
        thrust::default_random_engine rng;
        thrust::uniform_real_distribution<float> dist(a, b);
        rng.discard(n);
        return dist(rng);
    }
};


// Initialization of test error vector with random entries
template <class Vector>
void initRandom(Vector &vec)
{
    const unsigned int size = vec.size();
    thrust::counting_iterator<unsigned int> index_sequence_begin(0);
    thrust::transform(index_sequence_begin, index_sequence_begin + size,
                      vec.begin(), prg(0.f, 1.f));
}

// Zero out the entries of a vector corresponding to the coarse points
template <typename ValueType>
__global__
void zero_coarse_points_kernel(const int *cf_map, const int numRows, ValueType *vec)
{
    for ( int tIdx = threadIdx.x + blockDim.x * blockIdx.x; tIdx < numRows; tIdx += blockDim.x * gridDim.x )
    {
        // if point is coarse, zero out the corresponding entry of vec
        if (cf_map[tIdx] == COARSE)
        {
            vec[tIdx] = 0.;
        }
    }
}

// Create a vector which extracts fine points from the error vector and is
// zero at the coarse points
template <typename ValueType>
__global__
void get_fine_error_kernel( const int *cf_map, const int numRows,
                            const ValueType *v_err, ValueType *vec )
{
    for ( int tIdx = threadIdx.x + blockDim.x * blockIdx.x; tIdx < numRows; tIdx += blockDim.x * gridDim.x )
    {
        // if point is coarse, zero out the corresponding entry of vec
        if (cf_map[tIdx] == COARSE)
        {
            vec[tIdx] = 0.;
        }
        else
        {
            vec[tIdx] = v_err[tIdx];
        }
    }
}


// Update only the fine points of the error vector (v_err) based on input vec.
template <typename ValueType>
__global__
void update_fine_zero_coarse_error_kernel(const int *cf_map, const int numRows,
        ValueType *v_err, const ValueType *vec)
{
    for ( int tIdx = threadIdx.x + blockDim.x * blockIdx.x; tIdx < numRows; tIdx += blockDim.x * gridDim.x )
    {
        // if point is coarse, zero out the corresponding entry of vec
        if (cf_map[tIdx] == FINE)
        {
            v_err[tIdx] = vec[tIdx];
        }
        else
        {
            v_err[tIdx] = (ValueType) 0;
        }
    }
}


// Update coarse and fine points from the matrix coloring taking the first
// levels colors to form the independent set.
template <typename IndexType>
__global__
void update_cf_map_kernel(IndexType *cf_map, const int numRows, const int levels, const IndexType *A_row_colors)
{
    //coarse = [coarse; find(independent)];
    for ( int tIdx = threadIdx.x + blockDim.x * blockIdx.x; tIdx < numRows; tIdx += blockDim.x * gridDim.x )
    {
        // go over each row
        if (A_row_colors[tIdx] <= levels) // double check that this is what we want
        {
            cf_map[tIdx] = COARSE;
        }
    }
}



#define EXPERIMENTAL_CR

#ifdef EXPERIMENTAL_CR



// Update only the fine points of the error vector (v_err) based on input vec.
template <typename IndexType, typename ValueType>
__global__
void compute_Asc_nnzPerRow_kernel2(const IndexType *ArowOffsets, const IndexType *AcolInd,
                                   const ValueType *Avalues, const ValueType *AdiagValues,
                                   const IndexType AnumRows, const IndexType *cf_map,
                                   IndexType *Asc_nnzPerRow, ValueType *row_thresh, const ValueType thresh)
{
    for (int tIdx = threadIdx.x + blockDim.x * blockIdx.x; tIdx < AnumRows; tIdx += blockDim.x * gridDim.x)
    {
        // each thread works on one "fine" row of A at a time
        const int ArowBegin = ArowOffsets[tIdx];
        const int ArowEnd   = ArowOffsets[tIdx + 1];

        if (cf_map[tIdx] == FINE)
        {
            // work with fine pts only
            // Matlab pseudo-code:
            //  Asc(:,i) = A(:,i)/A(i,i)<-thresh;
            // but this thresh is dynamic
            ValueType fine_avg_thresh = 0.;
            int fine_count = 0;

            for (int j = ArowBegin; j < ArowEnd; j++)
            {
                int AjColIndex = AcolInd[j];

                if ((AjColIndex != tIdx) && (cf_map[AjColIndex] == FINE))
                {
                    fine_avg_thresh += Avalues[j];
                    fine_count ++;
                }
            }

            fine_avg_thresh *= thresh / fine_count;
            row_thresh[tIdx] = fine_avg_thresh;

            for (int j = ArowBegin; j < ArowEnd; j++)
            {
                int AjColIndex = AcolInd[j];

                if (AjColIndex == tIdx)
                {
                    /*Asc_nnzPerRow[tIdx]++;*/
                    continue;
                }

                if (cf_map[AjColIndex] == FINE)
                {
                    if (Avalues[j] / AdiagValues[tIdx] < fine_avg_thresh / AdiagValues[tIdx] || Avalues[j] / AdiagValues[AjColIndex] < fine_avg_thresh / AdiagValues[AjColIndex])
                    {
                        Asc_nnzPerRow[tIdx]++;
                    }
                }
            }
        }
    }
}



// Update only the fine points of the error vector (v_err) based on input vec.
template <typename IndexType, typename ValueType>
__global__
void compute_AscColInd_kernel(const IndexType *ArowOffsets, const IndexType *AcolInd,
                              const ValueType *Avalues, const ValueType *AdiagValues,
                              const IndexType AnumRows, const IndexType *cf_map,
                              const IndexType *AscRowOffsets, IndexType *AscColInd,
                              const ValueType *row_thresh)
{
    for (int tIdx = threadIdx.x + blockDim.x * blockIdx.x; tIdx < AnumRows; tIdx += blockDim.x * gridDim.x)
    {
        // each thread works on one "coarse" row of A at a time
        const int ArowBegin = ArowOffsets[tIdx];
        const int ArowEnd   = ArowOffsets[tIdx + 1];
        const int AscRowEnd = AscRowOffsets[tIdx + 1];
        ValueType rowMax(0);

        if (cf_map[tIdx] == FINE) // tIdx = j (in matlab) i.e. row of A
        {
            // work with fine pts only
            // Matlab pseudo-code:
            // Asc(:,i) = A(:,i)/A(i,i)<-thresh;
            int AscEntryNum = AscRowOffsets[tIdx];  // starting location of the current column of P

            // calculate max(-rowentries)
            for (int j = ArowBegin; j < ArowEnd && AscEntryNum < AscRowEnd; j++)
            {
                rowMax = max( rowMax, -Avalues[j]);
            }

            for (int j = ArowBegin; j < ArowEnd && AscEntryNum < AscRowEnd; j++)
            {
                // go thru each "fine" column of A
                int AjColIndex = AcolInd[j];

                if (AjColIndex == tIdx) { /*AscColInd[AscEntryNum] = AjColIndex; AscEntryNum++;*/ continue; }

                if (cf_map[AjColIndex] == FINE)
                {
                    if (-Avalues[j] >= row_thresh[tIdx]*rowMax)
                    {
                        AscColInd[AscEntryNum] = AjColIndex;
                        AscEntryNum++;
                    }
                }
            }
        }
    }
}

#else


// Update only the fine points of the error vector (v_err) based on input vec.
template <typename IndexType, typename ValueType>
__global__
void compute_Asc_nnzPerRow_kernel(const IndexType *ArowOffsets, const IndexType *AcolInd,
                                  const ValueType *Avalues, const ValueType *AdiagValues,
                                  const IndexType AnumRows, const IndexType *cf_map,
                                  IndexType *Asc_nnzPerRow, const ValueType thresh)
{
    for (int tIdx = threadIdx.x + blockDim.x * blockIdx.x; tIdx < AnumRows; tIdx += blockDim.x * gridDim.x)
    {
        // each thread works on one "fine" row of A at a time
        const int ArowBegin = ArowOffsets[tIdx];
        const int ArowEnd   = ArowOffsets[tIdx + 1];

        if (cf_map[tIdx] == FINE)
        {
            // work with fine pts only
            // Matlab pseudo-code:
            //  Asc(:,i) = A(:,i)/A(i,i)<-thresh;
            for (int j = ArowBegin; j < ArowEnd; j++)
            {
                // go thru each "fine" column of A
                int AjColIndex = AcolInd[j];

                if (AjColIndex == tIdx)
                {
                    /*Asc_nnzPerRow[tIdx]++;*/
                    continue;
                }

                if (cf_map[AjColIndex] == FINE)
                {
                    if (Avalues[j] / AdiagValues[tIdx] < -thresh || Avalues[j] / AdiagValues[AjColIndex] < -thresh)
                    {
                        Asc_nnzPerRow[tIdx]++;
                    }
                }
            }
        }
    }
}



// Update only the fine points of the error vector (v_err) based on input vec.
template <typename IndexType, typename ValueType>
__global__
void compute_AscColInd_kernel(const IndexType *ArowOffsets, const IndexType *AcolInd,
                              const ValueType *Avalues, const ValueType *AdiagValues,
                              const IndexType AnumRows, const IndexType *cf_map,
                              const IndexType *AscRowOffsets, IndexType *AscColInd,
                              const ValueType thresh)
{
    for (int tIdx = threadIdx.x + blockDim.x * blockIdx.x; tIdx < AnumRows; tIdx += blockDim.x * gridDim.x)
    {
        // each thread works on one "coarse" row of A at a time
        const int ArowBegin = ArowOffsets[tIdx];
        const int ArowEnd   = ArowOffsets[tIdx + 1];
        const int AscRowEnd = AscRowOffsets[tIdx + 1];

        if (cf_map[tIdx] == FINE) // tIdx = j (in matlab) i.e. row of A
        {
            // work with fine pts only
            // Matlab pseudo-code:
            // Asc(:,i) = A(:,i)/A(i,i)<-thresh;
            int AscEntryNum = AscRowOffsets[tIdx];  // starting location of the current column of P

            for (int j = ArowBegin; j < ArowEnd && AscEntryNum < AscRowEnd; j++)
            {
                // go thru each "fine" column of A
                int AjColIndex = AcolInd[j];

                if (AjColIndex == tIdx) { /*AscColInd[AscEntryNum] = AjColIndex; AscEntryNum++;*/ continue; }

                if (cf_map[AjColIndex] == FINE)
                {
                    if ((Avalues[j] / AdiagValues[tIdx] < -thresh) || (Avalues[j] / AdiagValues[AjColIndex] < -thresh))
                    {
                        AscColInd[AscEntryNum] = AjColIndex;
                        AscEntryNum++;
                    }
                }
            }
        }
    }
}

#endif
///////////////////////////////////////////////////////////////////////////////////////////////////

template <AMGX_VecPrecision t_vecPrec, AMGX_MatPrecision t_matPrec, AMGX_IndPrecision t_indPrec>
CR_Selector<TemplateConfig<AMGX_device, t_vecPrec, t_matPrec, t_indPrec> >::
CR_Selector() : CR_SelectorBase<TConfig_d>(),
    m_smoother(0) {}


template <AMGX_VecPrecision t_vecPrec, AMGX_MatPrecision t_matPrec, AMGX_IndPrecision t_indPrec>
CR_Selector<TemplateConfig<AMGX_device, t_vecPrec, t_matPrec, t_indPrec> >::
~CR_Selector()
{
    if (m_smoother) { delete m_smoother; }
}


// Private functions for the implementation on device

// Presmooth error
template <AMGX_VecPrecision t_vecPrec, AMGX_MatPrecision t_matPrec, AMGX_IndPrecision t_indPrec>
void CR_Selector<TemplateConfig<AMGX_device, t_vecPrec, t_matPrec, t_indPrec> >
::presmoothFineError( Matrix_d &A, const VVector &AdiagValues, const IVector &cf_map,
                      Vector_d &v_u, Vector_d &v_tmp, Vector_d &v_z,
                      Solver<TConfig_d> *smoother, ValueType &norm0,
                      const ValueType rho_thresh, const int pre )
{
    typedef typename Matrix_d::index_type IndexType;
    typedef typename Vector_d::value_type ValueTypeB;
    // Matrix A
    const IndexType AnumRows = (IndexType) A.get_num_rows();
    // Coarse/Fine map
    const int *cf_map_ptr = cf_map.raw();
    // Vector with the initial 'fine' error
    ValueTypeB *v_u_ptr   = v_u.raw();
    // choose blocksize. Using 1 thread / row for now
    const int blocksize = 64;
    const int numBlocks = std::min (AMGX_GRID_MAX_SIZE, (int) (AnumRows / blocksize + 1));
    // Compute 'energy norm' of v_u (normalize error before smoothing)
    // v_tmp = diag(Aff).*v_u
    thrust::transform(AdiagValues.begin(), AdiagValues.end(),
                      v_u.begin(), v_tmp.begin(),
                      thrust::multiplies<ValueTypeB>());
    cudaCheckError();
    // norm0 = sqrt(v_tmp' * v_u)
    norm0 = sqrt( thrust::inner_product(v_u.begin(), v_u.end(),
                                        v_tmp.begin(), ValueTypeB(0.)) );
    cudaCheckError();
    // normalize: v_u = v_u / (norm0 + 1e-12);
    thrust::transform(v_u.begin(), v_u.end(),
                      thrust::make_constant_iterator(norm0 + 1.0e-12), v_u.begin(),
                      thrust::divides<ValueTypeB>());
    cudaCheckError();

    // ---------------------------- begin error presmoothing --------------------------
    for (int k = 1; k <= 5; k++)
    {
        // Presmooth with Symmetric Gauss-Seidel (or ilu) for pre number of sm(ooth) s(teps)
        // WARNING: zeroing out technique used here is not equivalent to the fine matrix pre-smoothing!
        for (int sms = 1; sms <= pre; sms++)
        {
            smoother->solve_iteration( v_z, v_u, false );
            zero_coarse_points_kernel<ValueTypeB>
            <<< numBlocks, blocksize>>>(cf_map_ptr, AnumRows, v_u_ptr);
            cudaCheckError();
        }

        // -------------------- compute 'energy norm' of v_u ------------------------------
        // v_tmp = diag(Aff).*v_u
        thrust::transform(AdiagValues.begin(), AdiagValues.end(),
                          v_u.begin(), v_tmp.begin(),
                          thrust::multiplies<ValueTypeB>());
        cudaCheckError();
        // norm0 = sqrt(v_tmp' * v_u)
        norm0 = sqrt( thrust::inner_product(v_u.begin(), v_u.end(),
                                            v_tmp.begin(), ValueTypeB(0.)) );
        cudaCheckError();
        // -------------------- done with 'energy norm' of v_u ----------------------------
        // normalize: v_u = v_u / (norm0 + 1e-12);
        thrust::transform(v_u.begin(), v_u.end(),
                          thrust::make_constant_iterator(norm0 + 1.0e-12), v_u.begin(),
                          thrust::divides<ValueTypeB>());
        cudaCheckError();

        // norm0 := rho
        if (norm0 > 5 || norm0 < rho_thresh) { break; }
    }

    // ---------------------------- end error presmoothing ------------------------------
}


// A single iteration of CR while loop.
template <AMGX_VecPrecision t_vecPrec, AMGX_MatPrecision t_matPrec, AMGX_IndPrecision t_indPrec>
void CR_Selector<TemplateConfig<AMGX_device, t_vecPrec, t_matPrec, t_indPrec> >
::CR_iteration( Matrix_d &A, IVector &cf_map, int &numFine,
                Vector_d &v_err, ValueType &norm0,
                const ValueType maxit )
{
    typedef typename Matrix_d::index_type IndexType;
    typedef typename Matrix_d::value_type ValueType;
    typedef typename Vector_d::value_type ValueTypeB;
    // Matrix A
    const IndexType AnumRows = (IndexType) A.get_num_rows();
    // Coarse/Fine map
    IndexType *cf_map_ptr = cf_map.raw();
    // Raw ptr for the row colors from the multi-color smoother
    const IndexType *A_row_colors_ptr;
    // choose blocksize. Using 1 thread / row for now
    const int blocksize = 64;
    const int numBlocks = std::min (AMGX_GRID_MAX_SIZE, (int) (AnumRows / blocksize + 1));
    const int pre   = 10;
    const ValueType alpha = 0.7;
    const ValueType rho_thresh = 1.0e-2;
    const IndexType *ArowOffsets_ptr = A.row_offsets.raw();
    const IndexType *AcolInd_ptr     = A.col_indices.raw();
    const ValueType *Avalues_ptr     = A.values.raw();
    // temporary vectors - diagonal and non-zero offsets
    VVector AdiagValues(AnumRows, 0.0);
    // Extract the diagonal of A
    find_diag_kernel_indexed_dia <<< numBlocks, blocksize>>>(AnumRows, A.diag.raw(), Avalues_ptr, AdiagValues.raw());
    cudaCheckError();
    // ----------------------------------- define smoother ------------------------------------------
    // Set the smoother to Gauss-Seidel (json config format)
    AMG_Config cfg;
    std::string cfg_string = "{ \"config_version\": 2";
    cfg_string += ", \"solver\": { \"scope\": \"presmoothError\", \"solver\": \"MULTICOLOR_GS\"";
    cfg_string += " ,\"matrix_coloring_scheme\": \"PARALLEL_GREEDY\"";
    cfg_string += " } }";
    cfg.parseParameterString(cfg_string.c_str());
    // Allocate smoother
    m_smoother = SolverFactory<TConfig_d>::allocate(cfg, "default", "solver");
    // --------------------------------- end define smoother ----------------------------------------
    Vector_d v_z(AnumRows, 0.0);
    Vector_d v_u(AnumRows);
    Vector_d v_tmp(v_u.size());
    ValueTypeB *v_err_ptr = v_err.raw();
    ValueTypeB *v_u_ptr = v_u.raw();
    // Store the number of non-zeros in each column of the interpolation matrix P
    IVector Asc_nnzPerRow(AnumRows, 0);
#ifdef EXPERIMENTAL_CR
    VVector Asc_thresh_per_row(AnumRows, 0);
    ValueTypeB CR_P_trunc = 0.9; // truncation coefficient for CR P matrix, [0.0 ... 1.0]
    ValueTypeB Asc_thresh = (CR_P_trunc < epsilon(CR_P_trunc)) ? (-1. / epsilon(CR_P_trunc)) : (-1. / CR_P_trunc + 1);
    compute_Asc_nnzPerRow_kernel2<IndexType, ValueType>
    <<< numBlocks, blocksize>>>(ArowOffsets_ptr, AcolInd_ptr,
                                Avalues_ptr, AdiagValues.raw(),
                                AnumRows, cf_map_ptr,
                                Asc_nnzPerRow.raw(), Asc_thresh_per_row.raw(), Asc_thresh);
#else
    compute_Asc_nnzPerRow_kernel<IndexType, ValueType>
    <<< numBlocks, blocksize>>>(ArowOffsets_ptr, AcolInd_ptr,
                                Avalues_ptr, AdiagValues.raw(),
                                AnumRows, cf_map_ptr,
                                Asc_nnzPerRow.raw(), 0.1);
#endif
    cudaCheckError();
    // get the offsets in Asc with an inclusive scan
    thrust_wrapper::inclusive_scan(Asc_nnzPerRow.begin(), Asc_nnzPerRow.end(), Asc_nnzPerRow.begin());
    cudaCheckError();
    // get total num of non-zeros in P
    const int Asc_nnz = Asc_nnzPerRow[AnumRows - 1];
    // Declare Asc (strong connections of A)
    Matrix_d Asc;
    // resize Asc
    Asc.resize(0, 0, 0, 1);
    Asc.addProps(CSR);
    Asc.resize(AnumRows, AnumRows, Asc_nnz, 1);
    Asc.row_offsets[0] = 0;
    // set P offsets (P column offsets or P^T row offsets)
    thrust::copy(Asc_nnzPerRow.begin(), Asc_nnzPerRow.end(), Asc.row_offsets.begin() + 1);
    cudaCheckError();
#ifdef EXPERIMENTAL_CR
    compute_AscColInd_kernel<IndexType, ValueType>
    <<< numBlocks, blocksize>>>(ArowOffsets_ptr, AcolInd_ptr,
                                Avalues_ptr, AdiagValues.raw(),
                                AnumRows, cf_map_ptr,
                                Asc.row_offsets.raw(), Asc.col_indices.raw(), Asc_thresh_per_row.raw());
#else
    compute_AscColInd_kernel<IndexType, ValueType>
    <<< numBlocks, blocksize>>>(ArowOffsets_ptr, AcolInd_ptr,
                                Avalues_ptr, AdiagValues.raw(),
                                AnumRows, cf_map_ptr,
                                Asc.row_offsets.raw(), Asc.col_indices.raw(), 0.25);
#endif
    cudaCheckError();
    Asc.set_initialized(0);
    Asc.colorMatrix(cfg, "presmoothError");
    Asc.set_initialized(1);
    // get the coloring of matrix
    A_row_colors_ptr = Asc.getMatrixColoring().getRowColors().raw();
    m_smoother->setup(A, false);
    int levels = 1;     // number of CR iterations

    while (levels < 5)
    {
        // Here, we normally (i.e. as is done in Brannick's matlab code) we need to smooth using Aff,
        // however, since we are using Gauss-Seidel (symmetric or not) we can instead use A and
        // simply zero out v_err at the coarse points of each iteration.
        // v_u = v_err(fine)
        get_fine_error_kernel<ValueTypeB>
        <<< numBlocks, blocksize>>>(cf_map_ptr, AnumRows, v_err_ptr, v_u_ptr);
        cudaCheckError();
        // Presmooth error at the current FINE points.
        presmoothFineError( A, AdiagValues, cf_map, v_u, v_tmp, v_z,
                            m_smoother, norm0, rho_thresh, pre );

        // check for convergence and break if it's fast enough
        if (norm0 <= alpha && levels > 1)
        {
            break;
        }

        // v_err(fine) = v_u;
        update_fine_zero_coarse_error_kernel<ValueTypeB>
        <<< numBlocks, blocksize>>>(cf_map_ptr, AnumRows, v_err_ptr, v_u_ptr);
        cudaCheckError();
        // TODO: add aggressive coarsening here.
        // if (AggressiveCoarsening) { A2 = Al*Al; }
        // else                      { A2 = Al; }
        // Add nodes with lowest colors (<=levels) to the coarse set.
        update_cf_map_kernel<IndexType>
        <<< numBlocks, blocksize>>>(cf_map_ptr, AnumRows, levels - 1, A_row_colors_ptr);
        cudaCheckError();
        // Get the new FINE points count.
        numFine = (int) thrust::count(cf_map.begin(), cf_map.end(), (int)FINE);
        cudaCheckError();

        if (numFine == 0) { break; }  // no fine points left, just exit

        levels++;
    } // end while (levels < maxit)
} // end CR_iteration()


/*************************************************************************
 * Implementing the CR algorithm
 ************************************************************************/
// ----------------------------
//  specialization for device
// ----------------------------
template <AMGX_VecPrecision t_vecPrec, AMGX_MatPrecision t_matPrec, AMGX_IndPrecision t_indPrec>
void CR_Selector<TemplateConfig<AMGX_device, t_vecPrec, t_matPrec, t_indPrec> >
::markCoarseFinePoints_1x1( Matrix_d &A,
                            IVector &cf_map)
{
    if (A.hasProps(DIAG)) { FatalError("Unsupported separate diag", AMGX_ERR_NOT_IMPLEMENTED); }

    typedef typename Matrix_d::index_type IndexType;
    typedef typename Matrix_d::value_type ValueType;
    // Dimension of A.
    const int AnumRows  = (int) A.get_num_rows();

    if (AnumRows == 0) { return; }

    // Choose blocksize. Using 1 thread / row for now
    const int blocksize = 256;
    const int numBlocks = std::min(AMGX_GRID_MAX_SIZE, (int) ((AnumRows + blocksize - 1) / blocksize));
    const int maxit = 5;  // Max num of CR iterations.
    ValueType norm0;      // Stores the energy norm of the smoothed error in CR iteration.
    int *cf_map_ptr = cf_map.raw();
    // Initially all points are set to FINE - this is done in the level
    // Initial number of fine points (all points are fine).
    int numFine = AnumRows;
    // Randomly initialize error vector v_err: AnumRows x 1 vector.
    Vector_d v_err(AnumRows);
    initRandom(v_err);
    // Perform Compatible Relaxation (CR) iteration.
    CR_iteration(A, cf_map, numFine, v_err, norm0, maxit);
    int numCoarse = AnumRows - numFine;
    printf("CR: numrows = %d, numCoarse = %d\n", AnumRows, numCoarse);
} // end markCoarseFinePoints_1x1 (device specialization)


template <class T_Config>
void CR_SelectorBase<T_Config>::markCoarseFinePoints( Matrix<TConfig> &A,
        FVector &weights,
        const BVector &s_con,
        IVector &cf_map,
        IVector &scratch,
        int cf_map_init)
{
    ViewType oldView = A.currentView();
    A.setView(OWNED);

    if (A.get_block_size() == 1)
    {
        markCoarseFinePoints_1x1(A, cf_map);
    }
    else
    {
        FatalError("Unsupported block size CR selector", AMGX_ERR_NOT_SUPPORTED_BLOCKSIZE);
    }

    A.setView(oldView);
}



/****************************************
 * Explicit instantiations
 ***************************************/
#define AMGX_CASE_LINE(CASE) template class CR_SelectorBase<TemplateMode<CASE>::Type>;
AMGX_FORALL_BUILDS(AMGX_CASE_LINE)
#undef AMGX_CASE_LINE

#define AMGX_CASE_LINE(CASE) template class CR_Selector<TemplateMode<CASE>::Type>;
AMGX_FORALL_BUILDS(AMGX_CASE_LINE)
#undef AMGX_CASE_LINE

} // namespace classical

} // namespace amgx
