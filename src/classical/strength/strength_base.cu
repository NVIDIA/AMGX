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

#include <fstream>
#include <cutil.h>
#include <types.h>
#include <classical/strength/ahat.h>
#include <classical/strength/all.h>
#include <thrust/detail/integer_traits.h>
#include <float.h>
#include <specific_spmv.h>
#include <sm_utils.inl>

#include <thrust/iterator/transform_iterator.h>
#include <thrust/transform.h>
#include <thrust/logical.h>

/*
* Note:
* This implementation assumes that off-diag entries all have the opposite sign
* comparing to the diag entry. This is true for most practical cases.
* It would even work if the offending off-diag entries are just a few.
* But if there are many off-diag entries violate this assumption,
* the interpolation based on this strength would be inaccurate.
* This is explained in "Intro to Algebraic multigrid" by K. Stuben.
*/


namespace amgx
{

template< typename T_Config >
Strength_BaseBase<T_Config>::Strength_BaseBase(AMG_Config &cfg,
        const std::string &cfg_scope)
{
    alpha = cfg.AMG_Config::getParameter<double>("strength_threshold", cfg_scope);
}

/*************************************************************************
* "random" hash function for both device and host
************************************************************************/
__host__ __device__ __forceinline__
static float ourHash(const int i)
{
    unsigned int a = i;
    a = (a + 0x7ed55d16) + (a << 12);
    a = (a ^ 0xc761c23c) + (a >> 19);
    a = (a + 0x165667b1) + (a << 5);
    a = (a ^ 0xd3a2646c) + (a << 9);
    a = (a + 0xfd7046c5) + (a << 3);
    a = (a ^ 0xb55a4f09) + (a >> 16);
    return (a ^ 0x4a51e590) / (float)UINT_MAX;
}

/*************************************************************************
* Computes the strength matrix and the connection weights
* Described in \S 4.1 of:
* "Reducing complexity in parallel algebraic multigrid preconditioners"
*
************************************************************************/
template <AMGX_VecPrecision t_vecPrec, AMGX_MatPrecision t_matPrec,
          AMGX_IndPrecision t_indPrec>
void
Strength_Base<TemplateConfig<AMGX_host, t_vecPrec, t_matPrec, t_indPrec> >::
computeStrongConnectionsAndWeights_1x1(Matrix_h &A,
                                       BVector &s_con,
                                       FVector &weights,
                                       const double max_row_sum)
{
    bool compute_row_sum = (max_row_sum < 1.0);
    VVector sums_ptr;

    // get the (normalised) row sums
    if (compute_row_sum)
    {
        sums_ptr.resize(A.get_num_rows());
        weightedRowSum(A, sums_ptr);
        cudaCheckError();
    }

    // get min/max off-diag, depending on sign of diagonal
    for (int row = 0; row < A.get_num_rows(); row++)
    {
        ValueType diag(0), minVal(0), maxVal(0);
        ValueType row_sum = compute_row_sum ? sums_ptr[row] : -1;
        int rowEnd = A.row_offsets[row + 1];

        for (int j = A.row_offsets[row]; j < rowEnd; j++)
        {
            int col = A.col_indices[j];
            ValueType val = A.values[j];

            if (col == row)
            {
                diag = val;
            }
            else
            {
                minVal = min(minVal, val);
                maxVal = max(maxVal, val);
            }
        }

        //set the threshold for being strongly connected
        ValueType threshold = (diag < 0) ? maxVal : minVal;
        threshold *= this->alpha;

        //initialize the weight to a small random number
        if (A.is_matrix_singleGPU())
        {
            weights[row] += ourHash(row);
        }
        else
        {
            weights[row] += ourHash((int)A.manager->base_index() + row);
        }

        // sum the column of S - # of points that strongly connect to me
        for (int j = A.row_offsets[row]; j < rowEnd; j++)
        {
            bool is_strongly_connected = false;

            if (compute_row_sum && row_sum > max_row_sum)
            {
                is_strongly_connected = false;
            }
            else
                is_strongly_connected =
                    this->strongly_connected(A.values[j], threshold, diag);

            int col = A.col_indices[j];
            s_con[j] = (col != row) && is_strongly_connected;
            weights[A.col_indices[j]] += s_con[j] ? 1. : 0.;
        }
    }
}


template <AMGX_VecPrecision t_vecPrec, AMGX_MatPrecision t_matPrec,
          AMGX_IndPrecision t_indPrec>
void
Strength_Base<TemplateConfig<AMGX_host, t_vecPrec, t_matPrec, t_indPrec> >::
computeWeights_1x1(Matrix_h &S,
                   FVector &weights)
{
    for (int row = 0; row < S.get_num_rows(); row++)
    {
        //initialize the weight to a small random number
        if (S.is_matrix_singleGPU())
        {
            weights[row] += ourHash(row);
        }
        else
        {
            weights[row] += ourHash(S.manager->base_index() + row);
        }

        int rowEnd = S.row_offsets[row + 1];

        for (int j = S.row_offsets[row]; j < rowEnd; j++)
        {
            int col = S.col_indices[j];

            if (col != row) { weights[col] += 1.; }
        }
    }
}


template <typename ValueType>
__device__
bool stronglyConnectedAHat(ValueType value, ValueType threshold, ValueType diag)
{
    return (diag < ValueType(0)) ? value > threshold : value < threshold;
}

/*************************************************************************
* Computes the strength matrix and the connection weights (device)
************************************************************************/
template< typename IndexType, typename ValueType, int kCtaSize, bool singleGPU >
__global__
void computeStrongConnectionsAndWeightsKernel( const IndexType *A_rows,
        const IndexType *A_cols,
        const ValueType *A_vals,
        int A_num_rows,
        bool *s_con,
        float *weights,
        ValueType alpha,
        ValueType *row_sum,
        const double max_row_sum,
        int64_t base_index)
{
    // One warp works on each row and hence one iteration handles
    // num_warps*numBlock rows. This means atomicAdd() is inevitable.
    const int num_warps = kCtaSize / 32;
    __shared__ volatile ValueType smem[kCtaSize];
    __shared__ volatile ValueType s_diag[num_warps];
    __shared__ volatile ValueType s_threshold[num_warps];
    const int warpId = threadIdx.x / 32;
    const int laneId = threadIdx.x % 32;
    int aRowId = blockIdx.x * blockDim.x + threadIdx.x;

    if (aRowId < A_num_rows)
    {
        ValueType minVal(0), maxVal(0);

        ValueType diag = ValueType(0);

        utils::syncwarp();

        // Row sum
        ValueType rowSum = -1.0;

        if (max_row_sum < 1.0) { rowSum = row_sum[aRowId]; }

        // get diagonal, min/max off-diagonals
        const int aRowBegin = A_rows[aRowId  ];
        const int aRowEnd   = A_rows[aRowId + 1];

        for ( IndexType aRowIt = aRowBegin; aRowIt < aRowEnd ; ++aRowIt)
        {
            IndexType aColId = A_cols[aRowIt];
            ValueType aValue = A_vals[aRowIt];

            if ( aColId == aRowId ) // only one thread evaluates to true.
            {
                diag = aValue;
            }

            if ( aColId != aRowId )
            {
                minVal = min( minVal, aValue );
                maxVal = max( maxVal, aValue );
            }
        }

        if ( singleGPU )
        {
            atomicAdd(&weights[aRowId], ourHash(aRowId));
        }
        else
        {
            atomicAdd(&weights[aRowId], ourHash( (int) base_index + aRowId));
        }

        utils::syncwarp();

        // Big assumption: diag and off-diag always have the opposite sign.
        // If diag entry is negative, then all off-diag entries must be positive.
        // This means max off-diag is to be used to compute the threshold.
        // If diag entry is positve, the min off-diag is used instead.
        ValueType threshold;
        if ( diag < ValueType(0) )
        {
            threshold = maxVal*alpha;
        }
        else
        {
            threshold = minVal*alpha;
        }

        utils::syncwarp();

        // sum of the column of S
        for ( IndexType aRowIt = aRowBegin;  aRowIt < aRowEnd ; ++aRowIt)
        {
            IndexType aColId = A_cols[aRowIt];
            ValueType aValue = A_vals[aRowIt];
            bool is_strongly_connected = false;

            if (max_row_sum < 1.0 && rowSum > max_row_sum)
            {
                is_strongly_connected = false;
            }
            else
            {
                bool is_off_diagonal = aColId != aRowId;
                is_strongly_connected = is_off_diagonal &&
                                        stronglyConnectedAHat( aValue, threshold, diag );
            }

            if ( is_strongly_connected && aColId < A_num_rows)
            {
                atomicAdd( &weights[aColId], 1.0f );
            }

            s_con[aRowIt] = is_strongly_connected;
        }
    }
}


template< typename IndexType, typename ValueType, int kCtaSize, bool singleGPU >
__global__
void computeWeightsKernel( const IndexType *A_rows,
                           const IndexType *A_cols,
                           int A_num_rows,
                           float *weights,
                           int64_t base_index)
{
    const int num_warps = kCtaSize / 32;
    const int num_rows_per_iter = num_warps * gridDim.x;
    const int warpId = threadIdx.x / 32;
    const int laneId = threadIdx.x % 32;

    for ( int aRowId = blockIdx.x * num_warps + warpId ; aRowId < A_num_rows ;
            aRowId += num_rows_per_iter )
    {
        if ( laneId == 0 )
        {
            if ( singleGPU )
            {
                atomicAdd( &weights[aRowId], ourHash(aRowId) );
            }
            else
            {
                atomicAdd( &weights[aRowId], ourHash( (int) base_index + aRowId) );
            }
        }

        const int aRowBegin = A_rows[aRowId  ];
        const int aRowEnd   = A_rows[aRowId + 1];

        for ( IndexType aRowIt = aRowBegin + laneId ; utils::any( aRowIt < aRowEnd ) ;
                aRowIt += 32 )
        {
            IndexType aColId = aRowIt < aRowEnd ? A_cols[aRowIt] : -1;
            bool is_off_diagonal = aRowIt < aRowEnd && aColId != aRowId;

            if (is_off_diagonal)
            {
                atomicAdd( &weights[aColId], 1.0f );
            }
        }
    }
}


/*************************************************************************
* Computes the strength matrix and the connection weights (device)
************************************************************************/
template <AMGX_VecPrecision t_vecPrec, AMGX_MatPrecision t_matPrec,
          AMGX_IndPrecision t_indPrec>
void Strength_Base<TemplateConfig<AMGX_device, t_vecPrec, t_matPrec, t_indPrec> >::
computeStrongConnectionsAndWeights_1x1(Matrix_d &A,
                                       BVector &s_con,
                                       FVector &weights,
                                       const double max_row_sum)
{
    typedef TemplateConfig<AMGX_device, t_vecPrec, t_matPrec, t_indPrec> TConfig_d;
    typedef typename Matrix_d::index_type IndexType;
    typedef typename Matrix_d::value_type ValueType;
    // get the raw pointers for everything I need
    const IndexType *offsets_ptr = A.row_offsets.raw();
    const IndexType *column_indices_ptr = A.col_indices.raw();
    const ValueType *values_ptr = A.values.raw();
    bool *s_con_ptr = s_con.raw();
    float *weights_ptr = weights.raw();
    bool compute_row_sum = (max_row_sum < 1.0);

    if (A.get_num_rows() == 0) { compute_row_sum = false; }

    VVector sums_ptr;

    if (compute_row_sum)
    {
        sums_ptr.resize(A.get_num_rows());
        weightedRowSum(A, sums_ptr);
        cudaCheckError();
    }

    // choose a blocksize. Use 1 warp per row
    const int blockSize = 128;
    const int numBlocks = A.get_num_rows() / blockSize + 1;

    if (A.get_num_rows() > 0)
    {
        if (A.is_matrix_singleGPU())
            computeStrongConnectionsAndWeightsKernel<IndexType, ValueType, blockSize, true>
            <<< numBlocks, blockSize>>>(
                A.row_offsets.raw(),
                A.col_indices.raw(),
                A.values.raw(),
                A.get_num_rows(),
                s_con.raw(),
                weights.raw(),
                this->alpha,
                compute_row_sum ? sums_ptr.raw() : NULL,
                max_row_sum,
                0);
        else
            computeStrongConnectionsAndWeightsKernel<IndexType, ValueType, blockSize, false>
            <<< numBlocks, blockSize>>>(
                A.row_offsets.raw(),
                A.col_indices.raw(),
                A.values.raw(),
                A.get_num_rows(),
                s_con.raw(),
                weights.raw(),
                this->alpha,
                compute_row_sum ? sums_ptr.raw() : NULL,
                max_row_sum,
                A.manager->base_index());
    }

    if (!A.is_matrix_singleGPU() && A.currentView() == OWNED)
    {
        // Need to add neighbors contribution to my weights
        weights.dirtybit = 1;
        A.manager->add_from_halo(weights, weights.tag);
    }

    cudaCheckError();
}


template <AMGX_VecPrecision t_vecPrec, AMGX_MatPrecision t_matPrec,
          AMGX_IndPrecision t_indPrec>
void Strength_Base<TemplateConfig<AMGX_device, t_vecPrec, t_matPrec, t_indPrec> >::
computeWeights_1x1(Matrix_d &S,
                   FVector &weights)
{
    typedef TemplateConfig<AMGX_device, t_vecPrec, t_matPrec, t_indPrec> TConfig_d;
    typedef typename Matrix_d::index_type IndexType;
    typedef typename Matrix_d::value_type ValueType;
    // get the raw pointers for everything I need
    const IndexType *offsets_ptr = S.row_offsets.raw();
    const IndexType *column_indices_ptr = S.col_indices.raw();
    float *weights_ptr = weights.raw();
    // choose a blocksize. Use 1 thread per row
    const int blockSize = 256;
    const int numWarps  = blockSize / 32;
    const int numBlocks = min( 4096, (int) (S.get_num_rows() + numWarps - 1) / numWarps );
    cudaDeviceSynchronize();
    cudaCheckError();

    // call the CUDA kernel
    if (S.is_matrix_singleGPU())
        computeWeightsKernel<IndexType, ValueType, blockSize, true>
        <<< numBlocks, blockSize>>>(
            S.row_offsets.raw(),
            S.col_indices.raw(),
            S.get_num_rows(),
            weights.raw(),
            0);
    else
        computeWeightsKernel<IndexType, ValueType, blockSize, false>
        <<< numBlocks, blockSize>>>(
            S.row_offsets.raw(),
            S.col_indices.raw(),
            S.get_num_rows(),
            weights.raw(),
            S.manager->base_index());

    cudaCheckError();

    if (!S.is_matrix_singleGPU() && S.currentView() == OWNED)
    {
        // Need to add neighbors contribution to my weights
        weights.dirtybit = 1;
        S.manager->add_from_halo(weights, weights.tag);
    }

    cudaCheckError();
}

template<class T_Config>
void Strength_BaseBase<T_Config>::
computeStrongConnectionsAndWeights(Matrix<T_Config> &A,
                                   BVector &s_con,
                                   FVector &weights,
                                   const double max_row_sum
                                  )
{
    if (A.get_block_size() == 1)
    {
        computeStrongConnectionsAndWeights_1x1(A, s_con, weights, max_row_sum);
    }
    else
        FatalError("Unsupported block size for strong connections",
                   AMGX_ERR_NOT_SUPPORTED_BLOCKSIZE);
}

template<class T_Config>
void Strength_BaseBase<T_Config>::computeWeights(Matrix<T_Config> &S,
        FVector &weights
                                                )
{
    if (S.get_block_size() == 1)
    {
        computeWeights_1x1(S, weights);
    }
    else
        FatalError("Unsupported block size for strong connections",
                   AMGX_ERR_NOT_SUPPORTED_BLOCKSIZE);
}


/****************************************
 * Explict instantiations
 ***************************************/
#define AMGX_CASE_LINE(CASE) template class Strength_Base<TemplateMode<CASE>::Type>;
AMGX_FORALL_BUILDS(AMGX_CASE_LINE)
#undef AMGX_CASE_LINE

#define AMGX_CASE_LINE(CASE) template class Strength_BaseBase<TemplateMode<CASE>::Type>;
AMGX_FORALL_BUILDS(AMGX_CASE_LINE)
#undef AMGX_CASE_LINE

} // namespace amgx

