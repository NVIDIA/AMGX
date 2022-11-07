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
//#include <classical/strength/ahat.h>
//#include <classical/strength/all.h>
#include <classical/strength/affinity.h>
#include <thrust/detail/integer_traits.h>
#include <float.h>
#include <specific_spmv.h>

#include <util.h>

#include <thrust/iterator/transform_iterator.h>
#include <thrust/transform.h>
#include <thrust/logical.h>

#include <thrust/iterator/counting_iterator.h>
#include <thrust/random.h>

#include <sm_utils.inl>
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

namespace
{
int level_counter = 0;
bool is_random_initialized = false;
}

template <AMGX_VecPrecision t_vecPrec, AMGX_MatPrecision t_matPrec, AMGX_IndPrecision t_indPrec>
Strength_Affinity<TemplateConfig<AMGX_device, t_vecPrec, t_matPrec, t_indPrec> >::Strength_Affinity(AMG_Config &cfg,
        const std::string &cfg_scope)  : Strength_AffinityBase<TConfig_d>(cfg, cfg_scope)
{
    AMG_Config default_cfg;
    std::string default_cfg_scope = "default";
    affinity_iters = cfg.AMG_Config::getParameter<int>("affinity_iterations", cfg_scope);
    n_TV = cfg.AMG_Config::getParameter<int>("affinity_vectors", cfg_scope);

    if (n_TV > 32)
    {
        FatalError("Strength affinity: Test vectors > 32 is not supported, exiting", AMGX_ERR_NOT_SUPPORTED_TARGET);
    }

    solver = new MulticolorGaussSeidelSolver<TConfig_d>(cfg, cfg_scope);
    this->solver->set_max_iters(affinity_iters);
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

struct prg
{
    float a, b;

    __host__ __device__
    prg(float _a = -1.f, float _b = 1.f) : a(_a), b(_b) {};

    __host__ __device__
    float operator()(const unsigned int n) const
    {
#if 1
        amgx::thrust::default_random_engine rng;
        amgx::thrust::uniform_real_distribution<float> dist(a, b);
        rng.discard(n);
        return dist(rng);
#else
        float ru = ourHash(n);
        return (ru - 0.5f) * 2.0f;
#endif
    }
};


template <class Vector>
void initRandom(Vector &vec, int size)
{
    vec.resize(size);
    amgx::thrust::counting_iterator<unsigned int> index_sequence_begin(0);
    thrust_wrapper::transform(index_sequence_begin,
                      index_sequence_begin + size,
                      vec.begin(),
                      prg(-1.f, 1.f));
}

// Device utils
template <class T>
static __device__ __inline__ T cahedRead (const T *ar)
{
    return utils::Ld<utils::LD_NC>::load(ar);
}

// ! shfl is defined for int/float only in arch 600
template< typename ValueType >
__forceinline__ __device__ void reduce_inwarp_mul(
    const ValueType vValueA,
    const ValueType vValueB,
    const int vecId,
    const int n_per_warp,
    volatile ValueType *smem,
    double *s_xy)
{
    ValueType sum = vValueA * vValueB;
#pragma unroll

    for ( int offset = 16 / n_per_warp ; offset > 0 ; offset /= 2 )
    {
        sum += utils::shfl_xor(sum, offset, 16);
    }

    *s_xy = sum;
}

template< typename ValueType >
__forceinline__ __device__ void reduce_inblock_mul(
    const int blockSize,
    const ValueType vValueA,
    const ValueType vValueB,
    const int vecId,
    const int n_per_block,
    volatile ValueType *smem,
    double *s_xy)
{
    ValueType sum = vValueA * vValueB;
    smem[threadIdx.x] = sum;
    __syncthreads();

    if (blockSize >= 512)
    {
        if (vecId < 256 / n_per_block)
        {
            smem[threadIdx.x] = sum = sum + smem[threadIdx.x + 256 / n_per_block];
        }

        __syncthreads();
    }

    if (blockSize >= 256)
    {
        if (vecId < 128 / n_per_block)
        {
            smem[threadIdx.x] = sum = sum + smem[threadIdx.x + 128 / n_per_block];
        }

        __syncthreads();
    }

    if (blockSize >= 128)
    {
        if (vecId <  64 / n_per_block)
        {
            smem[threadIdx.x] = sum = sum + smem[threadIdx.x +  64 / n_per_block];
        }

        __syncthreads();
    }

#pragma unroll

    for ( int offset = 16  ; offset > 0 ; offset /= 2 )
        if ( vecId < offset )
        {
            smem[threadIdx.x] = sum = sum + smem[threadIdx.x + offset];
        }

    if ( vecId == 0 )
    {
        // If laneId=0, then sum is in smem[threadIdx.x].
        *s_xy = sum;
    }
}

/*************************************************************************
* Computes affinity matrix (device)
************************************************************************/

template< typename IndexType, typename ValueTypeA, typename ValueTypeB, int kCtaSize >
__global__
void computeAffinity_1x1_Kernel(const int *A_rows,
                                const int *A_cols,
                                const ValueTypeB *X,
                                const int nTV,
                                const int A_num_rows,
                                ValueTypeA *affinity
                               )
{
    const ValueTypeB epsilon = 1.e-12;
    const int tid =  blockDim.x * blockIdx.x + threadIdx.x;
    const int vid = tid % nTV;
    const int num_vecs_per_warp = 32 / nTV;
    const int num_rows_per_iter = gridDim.x * blockDim.x / nTV;
    //const int num_vecs_per_block = blockDim.x / nTV;
    ValueTypeB *smem = NULL;
    double s_xx, s_xy, s_yy;

    for ( int aRowId = tid / nTV ; aRowId < A_num_rows ;
            aRowId += num_rows_per_iter )
    {
        const int aRowBegin = A_rows[aRowId];
        const int aRowEnd   = A_rows[aRowId + 1];
        ValueTypeB vValueA = X[aRowId * nTV + vid];
        reduce_inwarp_mul(vValueA, vValueA, vid, num_vecs_per_warp, smem, &s_xx);
#if 0

        if (num_vecs_per_warp > 0)
        {
            reduce_inwarp_mul(vValueA, vValueA, vid, num_vecs_per_warp, smem, &s_xx);
        }
        else
        {
            reduce_inblock_mul(kCtaSize, vValueA, vValueA, vid, num_vecs_per_block, smem, &s_xx);
        }

#endif

        for ( IndexType aRowIt = aRowBegin ; aRowIt <  aRowEnd;
                aRowIt++ )
        {
            IndexType aColId = A_cols[aRowIt];

            if (aColId != aRowId)
            {
                ValueTypeB vValueB = cahedRead(X + aColId * nTV + vid);
                reduce_inwarp_mul(vValueA, vValueB, vid, num_vecs_per_warp, smem, &s_xy);
                reduce_inwarp_mul(vValueB, vValueB, vid, num_vecs_per_warp, smem, &s_yy);

                if (vid == 0)
                {
                    affinity[aRowIt] = (s_xy * s_xy / (s_xx * s_yy + epsilon));
                }
            }
        }
    }
}


/*************************************************************************
* Computes the strength matrix and the connection weights (device)
************************************************************************/

template< typename IndexType, typename ValueType, int kCtaSize, bool singleGPU >
__global__
void computeStrongConnectionsAndWeightsFromAffinityKernel(
    const int *A_rows,
    const int *A_cols,
    const ValueType *A_vals,
    int A_num_rows,
    bool *s_con,
    float *weights,
    ValueType alpha,
    int64_t base_index)
{
    // One warp works on each row and hence one iteration handles
    // num_warps*numBlock rows. This means atomicAdd() is inevitable.
    const int num_warps = kCtaSize / 32;
    const int num_rows_per_iter = num_warps * gridDim.x;
    __shared__ volatile ValueType smem[kCtaSize];
    __shared__ volatile ValueType s_threshold[num_warps];
    const int warpId = threadIdx.x / 32;
    const int laneId = threadIdx.x % 32;

    for ( int aRowId = blockIdx.x * num_warps + warpId ; aRowId < A_num_rows ;
            aRowId += num_rows_per_iter )
    {
        ValueType maxVal(0);
        // get diagonal, min/max off-idagonals
        const int aRowBegin = A_rows[aRowId  ];
        const int aRowEnd   = A_rows[aRowId + 1];

        for ( IndexType aRowIt = aRowBegin + laneId ; utils::any( aRowIt < aRowEnd ) ;
                aRowIt += 32 )
        {
            IndexType aColId = aRowIt < aRowEnd ? A_cols[aRowIt] : -1;
            ValueType aValue = aRowIt < aRowEnd ? A_vals[aRowIt] : ValueType(0);
            bool is_off_diagonal = aRowIt < aRowEnd && aColId != aRowId;

            if ( is_off_diagonal )
            {
                maxVal = max( maxVal, aValue );
            }
        }

        // init weights[] with a random number
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

        smem[threadIdx.x] = maxVal;
#pragma unroll

        for ( int offset = 16 ; offset > 0 ; offset /= 2 )
            if ( laneId < offset )
            {
                smem[threadIdx.x] = maxVal = max( maxVal, smem[threadIdx.x + offset] );
            }

        if ( laneId == 0 )
        {
            // If laneId=0, then maxVal or minVal is in smem[threadIdx.x].
            s_threshold[warpId] = smem[threadIdx.x] * alpha;
        }

        // sum of the column of S
        for ( IndexType aRowIt = aRowBegin + laneId ; utils::any( aRowIt < aRowEnd ) ;
                aRowIt += 32 )
        {
            IndexType aColId = aRowIt < aRowEnd ? A_cols[aRowIt] : -1;
            ValueType aValue = aRowIt < aRowEnd ? A_vals[aRowIt] : ValueType(0);
            bool is_off_diagonal = aRowIt < aRowEnd && aColId != aRowId;
            bool is_strongly_connected = is_off_diagonal && (aValue > s_threshold[warpId]);

            if ( is_strongly_connected && aRowIt < aRowEnd && aColId < A_num_rows)
            {
                atomicAdd( &weights[aColId], 1.0f );
            }

            if ( aRowIt < aRowEnd )
            {
                s_con[aRowIt] = is_strongly_connected;
            }
        }
    }
}

// Save to file
template <class T_Config>
void dump_matrix_vector(const char *fname, const Matrix<T_Config> &A, const Vector<T_Config> &solution)
{
    unsigned int num_nz = A.get_num_nz();
    unsigned int num_rows = A.get_num_rows();
    unsigned int vec_sz = solution.size();
    std::ofstream fout;
    fout.open(fname, std::ofstream::out | std::ofstream::app);
    fout << num_rows << " " << num_nz << "  " << vec_sz << std::endl;
    typename Vector<T_Config>::value_type val;

    for (int i = 0; i < num_rows + 1; i++)
    {
        fout << A.row_offsets[i] << std::endl;
    }

    for (int i = 0; i < num_nz; i++)
    {
        fout << A.col_indices[i] << std::endl;
    }

    fout.precision(14);
    fout.width(16);

    for (int i = 0; i < num_nz; i++)
    {
        val = A.values[i];
        fout << val << std::endl;
    }

    for (int i = 0; i < vec_sz; i++)
    {
        val = solution[i];
        fout << val << std::endl;
    }

    fout << std::endl;
    fout.close();
}

/*************************************************************************
* Computes the strength matrix and the connection weights (device)
************************************************************************/
template <AMGX_VecPrecision t_vecPrec, AMGX_MatPrecision t_matPrec,
          AMGX_IndPrecision t_indPrec>
void Strength_Affinity<TemplateConfig<AMGX_device, t_vecPrec, t_matPrec, t_indPrec> >::
computeStrongConnectionsAndWeights_1x1(Matrix_d &A,
                                       BVector &s_con,
                                       FVector &weights,
                                       const double max_row_sum)
{
    typedef TemplateConfig<AMGX_device, t_vecPrec, t_matPrec, t_indPrec> TConfig_d;
    typedef typename Matrix_d::index_type IndexType;
    typedef typename Matrix_d::value_type ValueTypeA;
    typedef typename Vector<TConfig_d>::value_type ValueTypeB;
    bool level_opt_flag = ((int) max_row_sum != -1);
    m_aff.copy(A);

    if (!is_random_initialized)
    {
        unsigned int nTV_num_rows = this->n_TV * A.get_num_rows();
        m_rhs.resize(nTV_num_rows, 0.0);
        initRandom(m_x, nTV_num_rows);

        if (level_opt_flag) { is_random_initialized = true; }

        level_counter = n_TV;
    }
    else
    {
        this->n_TV = min(level_counter, 32);
        unsigned int nTV_num_rows = this->n_TV * A.get_num_rows();
        m_rhs.resize(nTV_num_rows, 0.0);

        if (m_x.size() < nTV_num_rows)
        {
            initRandom(m_x, nTV_num_rows);
        }
    }

    level_counter *= 2;
    this->solver->setup_no_throw(m_aff, false);
    TimerCPU timer(true);
    timer.start();

    for (int iter = 0; iter < affinity_iters; iter ++)
    {
        this->solver->batch_smooth_1x1_fast(m_aff, this->n_TV, m_rhs, m_x);
    }

    cudaDeviceSynchronize();
    double elapsed = timer.elapsed();
// End of GS check
    // get the raw pointers for everything I need
    const IndexType *offsets_ptr = m_aff.row_offsets.raw();
    const IndexType *column_indices_ptr = m_aff.col_indices.raw();
    bool *s_con_ptr = s_con.raw();
    float *weights_ptr = weights.raw();
    ValueTypeA *m_aff_values_ptr = m_aff.values.raw();
    //m_aff_values.raw();
    ValueTypeB *m_x_ptr = m_x.raw();
    // choose a blocksize. Use 1 warp per row
    const int blockSize = 256;
    const int numWarps  = blockSize / 32;
    const int numBlocks = (int) (A.get_num_rows() + numWarps - 1) / numWarps;

    if (A.get_num_rows() > 0)
    {
        computeAffinity_1x1_Kernel<IndexType, ValueTypeA, ValueTypeB, blockSize>
        <<< numBlocks, blockSize>>>(
            A.row_offsets.raw(),
            A.col_indices.raw(),
            m_x_ptr,
            this->n_TV,
            A.get_num_rows(),
            m_aff_values_ptr
        );

        if (A.is_matrix_singleGPU())
            computeStrongConnectionsAndWeightsFromAffinityKernel<IndexType, ValueTypeA, blockSize, true>
            <<< numBlocks, blockSize>>>(
                m_aff.row_offsets.raw(),
                m_aff.col_indices.raw(),
                m_aff_values_ptr,
                m_aff.get_num_rows(),
                s_con.raw(),
                weights.raw(),
                this->alpha,
                int64_t(0));
        else
            computeStrongConnectionsAndWeightsFromAffinityKernel<IndexType, ValueTypeA, blockSize, false>
            <<< numBlocks, blockSize>>>(
                m_aff.row_offsets.raw(),
                m_aff.col_indices.raw(),
                m_aff_values_ptr,
                m_aff.get_num_rows(),
                s_con.raw(),
                weights.raw(),
                this->alpha,
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


/****************************************
 * Explict instantiations
 ***************************************/

#define AMGX_CASE_LINE(CASE) template class Strength_AffinityBase<TemplateMode<CASE>::Type>;
AMGX_FORALL_BUILDS(AMGX_CASE_LINE)
#undef AMGX_CASE_LINE

#define AMGX_CASE_LINE(CASE) template class Strength_Affinity<TemplateMode<CASE>::Type>;
AMGX_FORALL_BUILDS(AMGX_CASE_LINE)
#undef AMGX_CASE_LINE

} // namespace amgx


