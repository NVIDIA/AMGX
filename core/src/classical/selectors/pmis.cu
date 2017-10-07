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

#include <classical/selectors/pmis.h>
#include <classical/interpolators/common.h>
#include <cutil.h>
#include <util.h>
#include <types.h>

#include <thrust/count.h>
#include <thrust/execution_policy.h>

namespace amgx
{

namespace classical
{
/*************************************************************************
 * marks the strongest connected (and indepent) points as coarse
 ************************************************************************/

template< AMGX_VecPrecision t_vecPrec, AMGX_MatPrecision t_matPrec, AMGX_IndPrecision t_indPrec >
struct MarkAdditionalPoints
{
    static const AMGX_VecPrecision vecPrec = (AMGX_VecPrecision)t_indPrec;
    typedef Vector<TemplateConfig<AMGX_host, vecPrec, t_matPrec, t_indPrec> > IndVector;
    static int markAdditionalCoarsePoints(const Matrix<TemplateConfig<AMGX_host, t_vecPrec, t_matPrec, t_indPrec> > &A,
                                          const Vector<TemplateConfig<AMGX_host, AMGX_vecFloat, t_matPrec, t_indPrec> > &weights,
                                          const Vector<TemplateConfig<AMGX_host, AMGX_vecBool, t_matPrec, t_indPrec> > &s_con,
                                          IndVector &cf_map_in,
                                          IndVector &cf_map_out,
                                          IndVector &mark)
    {
        int num_coarse = 0;

        //deep copy cf_map_in into cf_map_out
        for (int i = 0; i < A.get_num_rows(); i++)
        {
            cf_map_out[i] = cf_map_in[i];
        }

        // set all unassigned points to be coarse
        for (int i = 0; i < A.get_num_rows(); i++)
        {
            mark[i] = 0;

            if (cf_map_in[i] == UNASSIGNED)
            {
                cf_map_out[i] = COARSE;
                mark[i] = 1;
            }
        }

        // now run through newly marked coarse points and make them unassigned again if necessary
        for (int i = 0; i < A.get_num_rows(); i++)
        {
            if (mark[i] > 0) // newly marked C point
            {
                for (int j = A.row_offsets[i]; j < A.row_offsets[i + 1]; j++)
                {
                    if (!s_con[j]) { continue; } // skip weak connections

                    int jcol = A.col_indices[j];

                    if (mark[jcol] && weights[jcol] > 1) // other point is also newly coarse
                    {
                        // if my weight is larger, they can't be a coarse point
                        if (weights[i] > weights[jcol])
                        {
                            cf_map_out[jcol] = UNASSIGNED;
                        }
                        // if their weight is large, I can't be a coarse point
                        else if (weights[jcol] > weights[i])
                        {
                            cf_map_out[i] = UNASSIGNED;
                        }
                    }
                }
            }
        }

        // finally count the number of coarse points
        for (int i = 0; i < A.get_num_rows(); i++)
        {
            num_coarse += static_cast<int>(cf_map_out[i] == COARSE);
        }

        return num_coarse;
    }

    /*************************************************************************
     * marks all neighbors of coarse points as fine
     ************************************************************************/
    static int markAdditionalFinePoints(const Matrix<TemplateConfig<AMGX_host, t_vecPrec, t_matPrec, t_indPrec> > &A,
                                        const Vector<TemplateConfig<AMGX_host, AMGX_vecBool, t_matPrec, t_indPrec> > &s_con,
                                        IndVector &cf_map_in,
                                        IndVector &cf_map_out)
    {
        int num_fine = 0;

        //deep copy cf_map_in into cf_map_out
        for (int i = 0; i < A.get_num_rows(); i++)
        {
            cf_map_out[i] = cf_map_in[i];
        }

        //for each row of the matrix
        for (int i = 0; i < A.get_num_rows(); i++)
        {
            //if already fine add it to the count
            num_fine += static_cast<int>(cf_map_in[i] == FINE);

            if (cf_map_in[i] == UNASSIGNED)
            {
                // for each off diagonal, if I have a new strongly connected coarse neighbour
                // set to fine
                for (int j = A.row_offsets[i]; j < A.row_offsets[i + 1]; j++)
                {
                    if (!s_con[j]) { continue; } //skip weakly connected points

                    int jcol = A.col_indices[j];

                    if (cf_map_in[jcol] == COARSE)
                    {
                        cf_map_out[i] = FINE;
                        num_fine++;
                        break;
                    }
                }
            }
        }

        return num_fine;
    }
};

/*************************************************************************
 * mark coarse and fine points
 ************************************************************************/
template <AMGX_VecPrecision t_vecPrec, AMGX_MatPrecision t_matPrec, AMGX_IndPrecision t_indPrec>
void PMIS_Selector<TemplateConfig<AMGX_host, t_vecPrec, t_matPrec, t_indPrec> >
::markCoarseFinePoints_1x1(Matrix_h &A,
                           FVector &weights,
                           const BVector &s_con,
                           IVector &cf_map,
                           IVector &scratch,
                           int cf_map_init)
{
    int num_coarse = 0, num_fine = 0;
    int num_strong_fine = 0;

    //mark all unconnected points as fine
    for (int i = 0; i < A.get_num_rows(); i++)
    {
        int numj = A.row_offsets[i + 1] - A.row_offsets[i];

        if (numj == 0) //if there are no columns
        {
            cf_map[i] = FINE;
        }
        else if (numj == 1 && A.col_indices[A.row_offsets[i]] == i) //if there is only 1 column and it is the diagonal
        {
            cf_map[i] = FINE;
        }
        else if (weights[i] < 1) // no point strongly depends on point
        {
            cf_map[i] = FINE;
        }
        else
        {
            cf_map[i] = UNASSIGNED;
        }

        // check to see if a point has no strong connections to anyone
        // needs more justification / knowledge of everywhere this condition is applied
        bool isolated = true;

        for (int j = A.row_offsets[i]; j < A.row_offsets[i + 1]; j++)
        {
            if (s_con[j])
            {
                isolated = false;
                break;
            }
        }

        if (isolated)
        {
            cf_map[i] = STRONG_FINE;
            num_strong_fine++;
            weights[i] = 0.F;
        }
    }

    // temp storage for marking coarse points
    IntVector mark(A.get_num_rows(), 0);
    int iterations = 0;

    do
    {
        ++iterations;
        num_coarse = MarkAdditionalPoints<t_vecPrec, t_matPrec, t_indPrec>::markAdditionalCoarsePoints(A, weights, s_con, cf_map, scratch, mark);
        num_fine = MarkAdditionalPoints<t_vecPrec, t_matPrec, t_indPrec>::markAdditionalFinePoints(A, s_con, scratch, cf_map);
    }
    while (num_coarse + num_fine + num_strong_fine < A.get_num_rows());

    // now we have \hat{C} and the diagonals, we can start calculating the weights
    // remap cf_map (to be like in distance1)
}

/*************************************************************************
 * device kernels
 ************************************************************************/

/*************************************************************************
 * Perform the initial marking for each row
 ************************************************************************/
template <typename IndexType>
__global__
void initialMarkingKernel(const IndexType *offsets, const IndexType *column_indices,
                          const int numRows, int *cf_map, float *weights, const bool *s_con)
{
    for ( int tIdx = threadIdx.x + blockDim.x * blockIdx.x; tIdx < numRows; tIdx += blockDim.x * gridDim.x )
    {
        const int offset = offsets[tIdx];
        const int numj = offsets[tIdx + 1] - offset;

        // set initial C/F/U
        if (numj == 0) // no points
        {
            cf_map[tIdx] = FINE;
        }
        else if (numj == 1 && column_indices[offset] == tIdx) // only the diagonal
        {
            cf_map[tIdx] = FINE;
        }
        else if (weights[tIdx] < 1) // no strong dependancies
        {
            cf_map[tIdx] = FINE;
        }
        else
        {
            cf_map[tIdx] = UNASSIGNED;
        }

        bool isolated = true;

        for (int j = offsets[tIdx]; j < offsets[tIdx + 1]; j++)
        {
            if (s_con[j])
            {
                isolated = false;
                break;
            }
        }

        if (isolated)
        {
            cf_map[tIdx] = STRONG_FINE;
            weights[tIdx] = 0.F;
        }
    }
}

template <typename IndexType>
__global__
void initialMarkingCFInit3Kernel(const IndexType *offsets, const IndexType *column_indices,
                                 const int numRows, int *cf_map, float *weights, const bool *s_con)
{
    for ( int tIdx = threadIdx.x + blockDim.x * blockIdx.x; tIdx < numRows; tIdx += blockDim.x * gridDim.x )
    {
        const int offset = offsets[tIdx];
        const int numj = offsets[tIdx + 1] - offset;

        // set initial C/F/U
        if (numj == 0) // no points
        {
            cf_map[tIdx] = FINE;
        }
        else if (numj == 1 && column_indices[offset] == tIdx) // only the diagonal
        {
            cf_map[tIdx] = FINE;
        }
        else if (weights[tIdx] < 1) // no strong dependancies
        {
            cf_map[tIdx] = FINE;
        }
        else
        {
            cf_map[tIdx] = UNASSIGNED;
        }

        bool isolated = true;

        for (int j = offsets[tIdx]; j < offsets[tIdx + 1]; j++)
        {
            if (s_con[j])
            {
                isolated = false;
                break;
            }
        }

        if (isolated)
        {
            cf_map[tIdx] = COARSE;
            weights[tIdx] = 0.F;
        }
    }
}

/*************************************************************************
 * Perform the initial marking for each row
 ************************************************************************/
template <typename IndexType>
__global__
void initialMarkingCfInitKernel(const IndexType *offsets, const IndexType *column_indices,
                                const int numRows, int *cf_map, float *weights, const bool *s_con, IndexType *mark)
{
    for ( int tIdx = threadIdx.x + blockDim.x * blockIdx.x; tIdx < numRows; tIdx += blockDim.x * gridDim.x )
    {
        const int offset = offsets[tIdx];
        const int numj = offsets[tIdx + 1] - offset;

        // Check if on boundary, this is unmarking even coarse points
        for (int j = offsets[tIdx]; j < offsets[tIdx + 1]; j++)
        {
            if (column_indices[j] > numRows)
            {
                cf_map[tIdx] = UNASSIGNED;
                mark[tIdx] = 1;
                break;
            }
        }

        // set initial C/F/U
        if (numj == 0) // no points
        {
            cf_map[tIdx] = FINE;
        }
        else if (numj == 1 && column_indices[offset] == tIdx) // only the diagonal
        {
            cf_map[tIdx] = FINE;
        }
        else if (weights[tIdx] < 1) // no strong dependancies
        {
            cf_map[tIdx] = FINE;
        }
        else if (cf_map[tIdx] == STRONG_FINE)
        {
            // Do nothing, leave as strong fine
            weights[tIdx] = 0.F;
        }
        else if (cf_map[tIdx] == FINE)
        {
            // Fine point, mark as unassigned
            cf_map[tIdx] = UNASSIGNED;
            mark[tIdx] = 1;
        }
    }
}


/*************************************************************************
 * Mark every currently unassigned point as coarse
 ************************************************************************/
template <typename IndexType>
__global__
void markUnassignedAsCoarseKernel(const int numRows,
                                  const float *weights,
                                  IndexType *cf_map_in,
                                  IndexType *cf_map_out, IndexType *mark)
{
    for ( int tIdx = threadIdx.x + blockDim.x * blockIdx.x; tIdx < numRows; tIdx += blockDim.x * gridDim.x )
    {
        const IndexType map_in = cf_map_in[tIdx];
        const int unassigned = map_in == UNASSIGNED ? 1 : 0;
        mark[tIdx] = unassigned;

        // Mark unassigned piont as coarse
        if (weights[tIdx] > 1.)
        {
            cf_map_out[tIdx] = unassigned ? COARSE : map_in;
        }
        else
        {
            cf_map_out[tIdx] = map_in;
        }
    }
}

/*************************************************************************
 * Unset all newly set coarse points who shouldn't be
 ************************************************************************/
template <typename IndexType>
__global__
void markAdditionalCoarsePointsKernel(const IndexType *offsets,
                                      const IndexType *column_indices,
                                      const int numRows, const float *weights,
                                      const bool *s_con,
                                      IndexType *cf_map_out, IndexType *mark)
{
    for ( int tIdx = threadIdx.x + blockDim.x * blockIdx.x; tIdx < numRows; tIdx += blockDim.x * gridDim.x )
    {
        // run through newly marked coarse points & set to unassigned again if necessary
        if (mark[tIdx] > 0)
        {
            for ( int j = offsets[tIdx], j_end = offsets[tIdx + 1] ; j < j_end; j++ )
            {
                if (!s_con[j]) { continue; } //these are always false for edges pointing outside of the 1-ring halo

                int jcol = column_indices[j];
                float w_col = weights[jcol];
                float w_row = weights[tIdx];

                if ( mark[jcol] && w_col > 1.0f )
                {
                    // If my weight is bigger, they can't be a coarse point
                    if ( w_row > w_col )
                    {
                        cf_map_out[jcol] = UNASSIGNED;
                    }
                    // If their weight is bigger, I can't be a coarse point
                    else if ( w_col > w_row )
                    {
                        cf_map_out[tIdx] = UNASSIGNED;
                    }
                }
            }
        }
    }
}

/*************************************************************************
 * mark every neighbour of a newly assigned coarse point to be fine
 ************************************************************************/
template <typename IndexType>
__global__
void markAdditionalFinePointsKernel(const IndexType *offsets, const IndexType *column_indices,
                                    const int numRows, const bool *s_con,
                                    int *cf_map_in, int *cf_map_out)
{
    for ( int tIdx = threadIdx.x + blockDim.x * blockIdx.x ; tIdx < numRows ; tIdx += blockDim.x * gridDim.x )
    {
        const int map_in = cf_map_in[tIdx];
        // go over each row
        bool fine = false;

        if ( map_in == UNASSIGNED )
        {
            // for each column
            for ( int j = offsets[tIdx], j_end = offsets[tIdx + 1] ; !fine && j < j_end ; ++j )
            {
                if ( !s_con[j] )
                {
                    continue;    // skip weakly connected points
                }

                int jcol = column_indices[j];
                fine = cf_map_in[jcol] == COARSE;
            }
        }

        cf_map_out[tIdx] = fine ? FINE : map_in;
    }
}

template <AMGX_VecPrecision t_vecPrec, AMGX_MatPrecision t_matPrec, AMGX_IndPrecision t_indPrec>
void PMIS_Selector<TemplateConfig<AMGX_device, t_vecPrec, t_matPrec, t_indPrec> >::markCoarseFinePoints_1x1(Matrix_d &A,
        FVector &weights,
        const BVector &s_con,
        IVector &cf_map,
        IVector &scratch,
        int cf_map_init)
{
    if (A.hasProps(DIAG))
    {
        FatalError("Unsupported separate diag", AMGX_ERR_NOT_IMPLEMENTED);
    }

    typedef typename Matrix_d::index_type IndexType;
    typedef typename Matrix_d::value_type ValueType;
    // Choose blocksize. Using 1 thread / row for now
    const int blockSize = 256;
    const int numBlocks = min (AMGX_GRID_MAX_SIZE, (int) ((A.get_num_rows() + blockSize - 1) / blockSize));
    const int numRows = (int) A.get_num_rows();
    // raw pointers from the cusp arrays
    const IndexType *offsets_ptr = A.row_offsets.raw();
    const IndexType *column_indices_ptr = A.col_indices.raw();
    float *weights_ptr = weights.raw();
    const bool *s_con_ptr = s_con.raw();
    int *cf_map_ptr = cf_map.raw();
    int *scratch_ptr = scratch.raw();
    IVector mark(cf_map.size());
    // do the initial marking of points
    thrust::fill(mark.begin(), mark.end(), 0);
    cudaCheckError();

    if (numRows > 0)
    {
        if (cf_map_init == 0)
        {
            initialMarkingKernel <<< numBlocks, blockSize>>>(offsets_ptr, column_indices_ptr, numRows, cf_map_ptr, weights_ptr, s_con_ptr);
        }
        else if (cf_map_init == 1)
        {
            initialMarkingCfInitKernel <<< numBlocks, blockSize>>>(offsets_ptr, column_indices_ptr, numRows, cf_map_ptr, weights_ptr, s_con_ptr, mark.raw());
        }
        else if (cf_map_init == 3)
        {
            initialMarkingCFInit3Kernel <<< numBlocks, blockSize>>>(offsets_ptr, column_indices_ptr, numRows, cf_map_ptr, weights_ptr, s_con_ptr);
        }
        else
        {
            FatalError("invalid value of cf_map_init in pmis selector, exiting", AMGX_ERR_NOT_IMPLEMENTED);
        }
    }

    cudaCheckError();
    // exchange initial cf_map for the 1st ring halos
    cf_map.dirtybit = 1;

    if (!A.is_matrix_singleGPU())
    {
        A.manager->exchange_halo(cf_map, cf_map.tag);
    }

    // count the # of coarse and fine points we have now
    int numUnassigned = 0;
    // array needed for marking of additional coarse points
    int one_ring_size, offset;

    if (!A.is_matrix_singleGPU())
    {
        A.getOffsetAndSizeForView(FULL, &offset, &one_ring_size);
    }
    else
    {
        one_ring_size = numRows;
    }

    const int numBlocks1Ring = min (AMGX_GRID_MAX_SIZE, (int) ((one_ring_size + blockSize - 1) / blockSize));
    // iterate until all points have been classified
    int numUnassignedMax = numRows;
    //int numUnassignedMaxPrevious;
    int iter = 0;

    do
    {
        //numUnassignedMaxPrevious = numUnassignedMax;
        if (numRows > 0)
        {
            if (iter || !cf_map_init)
            {
                // Laucnhed for 1-ring
                markUnassignedAsCoarseKernel <<< numBlocks1Ring, blockSize>>>(one_ring_size, weights_ptr, cf_map_ptr,
                        scratch_ptr, mark.raw());
                markAdditionalCoarsePointsKernel <<< numBlocks, blockSize>>>(offsets_ptr, column_indices_ptr,
                        numRows, weights_ptr, s_con_ptr,
                        scratch_ptr, mark.raw());
            }
            else
            {
                thrust::copy(cf_map.begin(), cf_map.begin() + one_ring_size, scratch.begin());
                cudaCheckError();
            }

            // If a neighbor marked one of my node as unassigned, need to mark it
            // Taking the min (unassigned == -4, coarse = -1)
            scratch.dirtybit = 1;

            if (!A.is_matrix_singleGPU())
            {
                A.manager->min_from_halo(scratch, scratch.tag);
            }

            cudaCheckError();
            // Now exchange with neighbors scratch for my own nodes
            scratch.dirtybit = 1;

            if (!A.is_matrix_singleGPU())
            {
                // exchange updated CF map for halos (stored in scratch)
                A.manager->exchange_halo(scratch, scratch.tag);
            }

            cudaCheckError();
            markAdditionalFinePointsKernel <<< numBlocks, blockSize>>>(offsets_ptr, column_indices_ptr,
                    numRows, s_con_ptr, scratch_ptr,
                    cf_map_ptr);
            cudaCheckError();
        }  // num rows > 0

        // count # of points still unassigned
        numUnassigned = (int) thrust::count(cf_map.begin(), cf_map.begin() + numRows, (int)UNASSIGNED);
        cudaCheckError();
        numUnassignedMax = numUnassigned;
        cudaCheckError();
        cf_map.dirtybit = 1;

        if (A.is_matrix_distributed())
        {
            // exchange CF map for halos
            A.manager->exchange_halo(cf_map, cf_map.tag);
            // get max across all processes
            A.manager->getComms()->all_reduce_max(numUnassigned, numUnassignedMax);
        }

        iter++;
    }
    while ( numUnassignedMax != 0 );
}

template <class T_Config>
void PMIS_SelectorBase< T_Config>::markCoarseFinePoints(Matrix< T_Config> &A,
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
        markCoarseFinePoints_1x1(A, weights, s_con, cf_map, scratch, cf_map_init);
    }
    else
    {
        FatalError("Unsupported block size PMIS selector", AMGX_ERR_NOT_SUPPORTED_BLOCKSIZE);
    }

    A.setView(oldView);
}

/****************************************
 * Explict instantiations
 ***************************************/
#define AMGX_CASE_LINE(CASE) template class PMIS_SelectorBase<TemplateMode<CASE>::Type>;
AMGX_FORALL_BUILDS(AMGX_CASE_LINE)
#undef AMGX_CASE_LINE

#define AMGX_CASE_LINE(CASE) template class PMIS_Selector<TemplateMode<CASE>::Type>;
AMGX_FORALL_BUILDS(AMGX_CASE_LINE)
#undef AMGX_CASE_LINE

} // namespace classical

} // namespace amgx
