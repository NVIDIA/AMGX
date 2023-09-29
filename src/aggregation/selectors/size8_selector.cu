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

#include <cutil.h>
#include <util.h>
#include <types.h>
#include <basic_types.h>
#include <texture.h>
#include <matrix_analysis.h>
#include <aggregation/selectors/size8_selector.h>

#include <thrust/count.h> //count
#include <thrust/sort.h> //sort
#include <thrust/unique.h> //unique
#include <thrust/remove.h> //remove
#include <thrust/transform_scan.h> //transform_inclusive_scan

#include <cusp/detail/format_utils.h> //offsets_to_indices

#include <iostream>
#include <strided_reduction.h>
#include <aggregation/selectors/selector_kernels.h>
namespace amgx
{


namespace strided_reduction
{
template<int STRIDE, class scalar_t, class OP>
void count_block_results_pinned_memory(scalar_t *out_host, const int n_blocks, scalar_t *out_d, const OP &op = OP(), cudaStream_t stream = 0)
{
    strided_reduction_collect_partials<scalar_t, STRIDE, 32, OP> <<< 1, 32, 0, stream>>>(out_host, out_d, n_blocks);
    cudaCheckError();
}


template<class scalar_t, class OP>
scalar_t count_block_results_pinned_memory(const int a, const int i, const int n_blocks, scalar_t *out_d, const OP &op = OP(), cudaStream_t stream = 0) //STRIDE=1 case
{
    static scalar_t *ret = 0;
    static cudaEvent_t  throttle_event = 0;
    const int buffers = 1;

    if (ret == 0)
    {
        amgx::memory::cudaMallocHost((void **)&ret, buffers * sizeof(scalar_t));
        ret[0] = 0;
        cudaEventCreateWithFlags(&throttle_event, cudaEventDisableTiming);
    }

    int ib = i % buffers;
    count_block_results_pinned_memory<1, scalar_t, OP>(ret + ib, n_blocks, out_d, op, stream);

    if (ib == buffers - 1)
    {
        cudaEventRecord(throttle_event);
        cudaEventSynchronize(throttle_event);
        scalar_t tot = 0;

        for (int j = 0; j < buffers; j++)
        {
            tot += ret[j];
        }

        return tot + buffers - 1;
    }
    else
    {
        return -1;
    }
}


}

void analyze_coloring(device_vector_alloc<int> aggregates_d, device_vector_alloc<int> colors_d);

namespace aggregation
{
namespace size8_selector
{

// include common routines for all selectors
#include <aggregation/selectors/common_selector.h>

// ------------------------
//  Kernels
// ------------------------

// findStrongestNeighbour kernel for block_dia_csr_matrix format
// Reads the weight from edge_weights array
template <typename IndexType>
__global__
void findStrongestNeighbourBlockDiaCsr_StoreWeight_2(const IndexType *row_offsets, const IndexType *column_indices,
        const float *edge_weights, const IndexType num_block_rows, IndexType *aggregated, IndexType *aggregates, int *strongest_neighbour, IndexType *partner_index, float *weight_strongest_neighbour)
{
    float weight;
    int jcol, jmin, jmax;
    int partner0, partner1, partner2;
    int agg_jcol;

    for (int tid = threadIdx.x + blockDim.x * blockIdx.x; tid < num_block_rows; tid += blockDim.x * gridDim.x)
    {
        float max_weight_unaggregated = 0.;
        float max_weight_aggregated = 0.;
        int strongest_unaggregated = -1;
        int strongest_aggregated = -1;

        if (aggregated[tid] == -1) // Unaggregated row
        {
            partner0 = partner_index[tid];
            partner1 = partner_index[num_block_rows + tid];
            partner2 = partner_index[2 * num_block_rows + tid];
            jmin = row_offsets[tid];
            jmax = row_offsets[tid + 1];

            for (int j = jmin; j < jmax; j++)
            {
                jcol = column_indices[j];

                if (jcol == tid || jcol >= num_block_rows) { continue; }

                weight = edge_weights[j];
                agg_jcol = aggregated[jcol];

                if (jcol != partner0 && jcol != partner1 && jcol != partner2)
                {
                    if (agg_jcol == -1 && (weight > max_weight_unaggregated || (weight == max_weight_unaggregated && jcol > strongest_unaggregated))) // unaggregated
                    {
                        max_weight_unaggregated = weight;
                        strongest_unaggregated = jcol;
                    }
                    else if (agg_jcol != -1 && (weight > max_weight_aggregated || (weight == max_weight_aggregated && jcol > strongest_aggregated))) // unaggregated
                    {
                        max_weight_aggregated = weight;
                        strongest_aggregated = jcol;
                    }
                }
            }

            if (strongest_unaggregated == -1 && strongest_aggregated != -1) // all neighbours are aggregated, store the strongest aggregated
            {
                weight_strongest_neighbour[tid] = -max_weight_aggregated;
                strongest_neighbour[tid] = aggregates[strongest_aggregated];
            }
            else if (strongest_unaggregated != -1)
            {
                weight_strongest_neighbour[tid] = max_weight_unaggregated;
                strongest_neighbour[tid] = aggregates[strongest_unaggregated];
            }
        }
    }
}

// findStrongestNeighbour kernel for block_dia_csr_matrix format
// Reads the weight from edge_weights array
template <typename IndexType>
__global__
void agreeOnProposal_2(const IndexType *row_offsets, const IndexType *column_indices,
                       IndexType num_block_rows, IndexType *aggregated, int *strongest_neighbour, float *weight_strongest_neighbour, IndexType *partner_index, int *aggregates, int deterministic)
{
    int partner[3];
    float weight[3];

    for (int tid = threadIdx.x + blockDim.x * blockIdx.x; tid < num_block_rows; tid += blockDim.x * gridDim.x)
    {
        int strongest_partner = -1;
        float my_weight = 0.;

        if (aggregated[tid] == -1)
        {
            my_weight = weight_strongest_neighbour[tid];
#pragma unroll

            for (int m = 0; m < 3; m++)
            {
                partner[m] = partner_index[tid + m * num_block_rows];
            }

#pragma unroll

            for (int m = 0; m < 3; m++)
            {
                weight[m] = weight_strongest_neighbour[partner[m]];
            }

#pragma unroll

            for (int m = 0; m < 3; m++)
            {
                //if (weight[m] > my_weight && weight[m] > 0.) // there is a partner that has an unaggregated neighbour
                if (weight[m] > my_weight)
                {
                    if (weight[m] > 0.) // there is a partner that has an unaggregated neighbour
                    {
                        strongest_partner = m;
                        my_weight = weight[m];
                    }
                }
                else if (weight[m] < my_weight)   // there is a partner without an unaggregated neighbour, whose neighbour is stronger than mine
                {
                    if (my_weight < 0.)
                    {
                        strongest_partner = m;
                        my_weight = weight[m];
                    }
                }
            }

            if (my_weight < 0.) // means all neighbours of vertices in aggregate are aggregated, merge to another aggregate
            {
                if (!deterministic)
                {
                    aggregated[tid] = 1;
                    aggregates[tid] = strongest_partner != -1 ? strongest_neighbour[partner[strongest_partner]] : strongest_neighbour[tid];
                }
            }
            else if (strongest_partner != -1)   // store my partner's pick
            {
                strongest_neighbour[tid] = strongest_neighbour[partner[strongest_partner]];
            }
        }
    }
}

template <typename IndexType>
__global__
void agreeOnProposal_2_deterministic(int *strongest_neighbour_out, const IndexType *row_offsets, const IndexType *column_indices,
                                     IndexType num_block_rows, IndexType *aggregated, int *strongest_neighbour, float *weight_strongest_neighbour, IndexType *partner_index, int *aggregates, int deterministic)
{
    int partner[3];
    float weight[3];

    for (int tid = threadIdx.x + blockDim.x * blockIdx.x; tid < num_block_rows; tid += blockDim.x * gridDim.x)
    {
        int strongest_partner = -1;
        float my_weight = 0.;
        //copy here to avoid redundant copies before launching the kernel
        int new_strongest_neighbour_out = strongest_neighbour[tid];

        if (aggregated[tid] == -1)
        {
            my_weight = weight_strongest_neighbour[tid];
#pragma unroll

            for (int m = 0; m < 3; m++)
            {
                partner[m] = partner_index[tid + m * num_block_rows];
            }

#pragma unroll

            for (int m = 0; m < 3; m++)
            {
                weight[m] = weight_strongest_neighbour[partner[m]];
            }

#pragma unroll

            for (int m = 0; m < 3; m++)
            {
                if (weight[m] > my_weight)
                {
                    if (weight[m] > 0.) // there is a partner that has an unaggregated neighbour
                    {
                        strongest_partner = m;
                        my_weight = weight[m];
                    }
                }
                else if (weight[m] < my_weight)   // there is a partner without an unaggregated neighbour, whose neighbour is stronger than mine
                {
                    if (my_weight < 0.)
                    {
                        strongest_partner = m;
                        my_weight = weight[m];
                    }
                }
            }

            if (my_weight < 0.) // means all neighbours of vertices in aggregate are aggregated, merge to another aggregate
            {
                if (!deterministic)
                {
                    aggregated[tid] = 1;
                    aggregates[tid] = strongest_partner != -1 ? strongest_neighbour[partner[strongest_partner]] : strongest_neighbour[tid];
                }
            }
            else if (strongest_partner != -1)   // store my partner's pick
            {
                new_strongest_neighbour_out = strongest_neighbour[partner[strongest_partner]];
            }
        }

        //copy here to avoid redundant copies before launching the kernel
        strongest_neighbour_out[tid] = new_strongest_neighbour_out;
    }
}


// Kernel that checks if perfect matchs exist
template <typename IndexType>
__global__
void matchAggregatesSize4(IndexType *aggregates, IndexType *aggregated, IndexType *strongest_neighbour, IndexType *partner_index, const IndexType num_rows)
{
    int potential_match, potential_match_neighbour, my_aggregate;

    for (int tid = threadIdx.x + blockDim.x * blockIdx.x; tid < num_rows; tid += blockDim.x * gridDim.x)
    {
        if (aggregated[tid] == -1) // Unaggregated row
        {
            potential_match = strongest_neighbour[tid];

            if (potential_match != -1)
            {
                potential_match_neighbour = strongest_neighbour[potential_match];
                my_aggregate = aggregates[tid];

                if (potential_match_neighbour == my_aggregate) // we have a match
                {
                    aggregated[tid] = 1;
                    aggregates[tid] = ( potential_match > my_aggregate) ? my_aggregate : potential_match;
                    partner_index[tid + num_rows] = potential_match;
                    partner_index[tid + 2 * num_rows] = partner_index[potential_match];
                }
            }
        }
    }
}


template <typename IndexType>
__global__
void assignUnassignedVertices_2(IndexType *partner_index, const IndexType num_rows)
{
    for (int tid = threadIdx.x + blockDim.x * blockIdx.x; tid < num_rows; tid += blockDim.x * gridDim.x)
    {
        if (partner_index[num_rows + tid] == -1) // Unaggregated row
        {
            partner_index[num_rows + tid] = tid;
        }

        if (partner_index[2 * num_rows + tid] == -1) // Unaggregated row
        {
            partner_index[2 * num_rows + tid] = tid;
        }
    }
}


// -----------------
//  Methods
// ----------------

// Constructor
template<class T_Config>
Size8SelectorBase<T_Config>::Size8SelectorBase(AMG_Config &cfg, const std::string &cfg_scope)
{
    deterministic = cfg.AMG_Config::template getParameter<IndexType>("determinism_flag", "default");
    max_iterations = cfg.AMG_Config::template getParameter<IndexType>("max_matching_iterations", cfg_scope);
    numUnassigned_tol = cfg.AMG_Config::template getParameter<double>("max_unassigned_percentage", cfg_scope);
    m_aggregation_edge_weight_component = cfg.AMG_Config::template getParameter<int>("aggregation_edge_weight_component", cfg_scope);
    weight_formula = cfg.AMG_Config::template getParameter<int>("weight_formula", cfg_scope);
}


// setAggregates for block_dia_csr_matrix_h format
template <AMGX_VecPrecision t_vecPrec, AMGX_MatPrecision t_matPrec, AMGX_IndPrecision t_indPrec>
void Size8Selector<TemplateConfig<AMGX_host, t_vecPrec, t_matPrec, t_indPrec> >::setAggregates_common_sqblock(const Matrix_h &A,
        IVector &aggregates, IVector &aggregates_global, int &num_aggregates)
{
    FatalError("Size8 selector: setAggregates not implemented on CPU, exiting", AMGX_ERR_NOT_SUPPORTED_TARGET);
}


template <int NUM_COLS, typename IndexType, typename ValueType>
__global__ //__launch_bounds__(256,2)
void computeEdgeWeightsBlockDiaCsr_V2_1(
    const IndexType *row_offsets,
    //const int* __myrestrict row_offsets,
    const IndexType *row_indices,
    const IndexType *column_indices,
    const IndexType *dia_values,
    const ValueType *nonzero_values,
    //const ValueType* __restrict nonzero_values,
    const IndexType num_nonzero_blocks,
    float *str_edge_weights, float *rand_edge_weights, int num_owned, int bsize, int component)
{
    int tid = threadIdx.x + blockDim.x * blockIdx.x;
    int i, j;
    int bsize_sq = bsize * bsize;
    float kvalue;
    int matrix_weight_entry = component * bsize + component;
    bool valid_tid;

    while (utils::any(valid_tid = tid < num_nonzero_blocks))
    {
        i = -1;
        double d1, d2, w1;

        if (valid_tid)
        {
            if ( rand_edge_weights != NULL )
            {
                rand_edge_weights[tid] = random_weight(i, j, num_owned);
            }

            i  = row_indices[tid];
            j  = column_indices[tid];
            d1 = types::util<ValueType>::abs(__cachingLoad(&nonzero_values[__load_nc(dia_values + i) * bsize_sq + matrix_weight_entry]));
            d2 = types::util<ValueType>::abs(__cachingLoad(&nonzero_values[__load_nc(dia_values + j) * bsize_sq + matrix_weight_entry]));
        }

        const bool valid_j = valid_tid && i != j && j < num_owned;
        int ki = -1; //my transpose index, initialized to not found
        //int diag_j = -1; //j diagonal index

        if (!utils::any(valid_j))
        {
            continue;
        }

        int kmin = 0, kmax = 0;

        if (valid_j)
        {
            kmin = __cachingLoad(&row_offsets[j  ]);
            kmax = __cachingLoad(&row_offsets[j + 1]);
        }

        for ( int k = kmin ; k < kmax ; ++k )
        {
            const int idx = __load_nc(column_indices + k);

            if (idx == i)
            {
                ki = k;    //find the transpose ji
            }
        }

        kvalue = 0.0f;

        if (ki > -1)
        {
            kvalue = types::util<ValueType>::abs(__cachingLoad(&nonzero_values[ki * bsize_sq + matrix_weight_entry]));
        }

        if (valid_tid)
        {
            w1 = types::util<ValueType>::abs(__cachingLoad(&nonzero_values[tid * bsize_sq + matrix_weight_entry]));
            str_edge_weights[tid] = 0.5 * (w1 + kvalue) / ( (float) max(d1, d2) ) * valid_j;
        }

        tid += gridDim.x * blockDim.x;
    }
}

#if AMGX_ASYNCCPU_PROOF_OF_CONCEPT

template<class T, class M, class V>
struct async_size8_task : public task_cuda
{
    T *size8;
    M *A;
    V *aggregates;
    V *aggregates_global;
    int *num_aggregates;

    typedef typename T::IndexType IndexType;

    void run()
    {
        const IndexType nnz_per_row = (*A).get_num_nz() / (*A).get_num_rows();

        if (0)
        {
        }
        else if (nnz_per_row > 2)
        {
            size8->template setAggregates_common_sqblock_avg_specialized<4>(*A, *aggregates, *aggregates_global, *num_aggregates);
        }
        else if (nnz_per_row > 1)
        {
            size8->template setAggregates_common_sqblock_avg_specialized<4>(*A, *aggregates, *aggregates_global, *num_aggregates);
        }
        else
        {
            size8->template setAggregates_common_sqblock_avg_specialized<2>(*A, *aggregates, *aggregates_global, *num_aggregates);
        }
    }
};


template<class T, class M, class V>
async_size8_task<T, M, V> *make_async_size8_task(T *size8, M &A, V &aggregates, V &aggregates_global, int &num_aggregates)
{
    async_size8_task<T, M, V> *ret = new async_size8_task<T, M, V>;
    ret->size8 = size8;
    ret->A = &A;
    ret->aggregates = &aggregates;
    ret->aggregates_global = &aggregates_global;
    ret->num_aggregates = &num_aggregates;
    static task_chain_cuda_streamset *ss = new task_chain_cuda_streamset(1);
    task_chain_cuda *cr = new task_chain_cuda(ss);
    cr->append(ret, asyncmanager::singleton()->main_thread_queue(2));
    return ret;
}

#endif

template <AMGX_VecPrecision t_vecPrec, AMGX_MatPrecision t_matPrec, AMGX_IndPrecision t_indPrec>
void Size8Selector<TemplateConfig<AMGX_device, t_vecPrec, t_matPrec, t_indPrec> >::setAggregates_common_sqblock(const Matrix_d &A,
        typename Matrix_d::IVector &aggregates,
        typename Matrix_d::IVector &aggregates_global,
        int &num_aggregates)
{
#if AMGX_ASYNCCPU_PROOF_OF_CONCEPT
    bool task = false;
    bool push = false;

    if (task)
    {
        task_cuda *t = make_async_size8_task(this, A, aggregates, aggregates_global, num_aggregates);
        enqueue_async_get_receipt(asyncmanager::singleton()->global_parallel_queue, t)->wait();
        return;
    }

    cudaStream_t stream_old;
    static cudaStream_t stream = 0;

    if (push)
    {
        stream_old = amgx::thrust::global_thread_handle::threadStream[getCurrentThreadId()];
        cudaStreamSynchronize(amgx::thrust::global_thread_handle::threadStream[getCurrentThreadId()]);

        if (stream == 0) { cudaStreamCreate(&stream); }

        amgx::thrust::global_thread_handle::threadStream[getCurrentThreadId()] = stream;
    }

#endif
    const IndexType nnz_per_row = A.get_num_nz() / A.get_num_rows();

    if (0)
    {
    }
    else if (nnz_per_row > 2)
    {
        setAggregates_common_sqblock_avg_specialized<4>(A, aggregates, aggregates_global, num_aggregates);
    }
    else if (nnz_per_row > 1)
    {
        setAggregates_common_sqblock_avg_specialized<4>(A, aggregates, aggregates_global, num_aggregates);
    }
    else
    {
        setAggregates_common_sqblock_avg_specialized<2>(A, aggregates, aggregates_global, num_aggregates);
    }

#if AMGX_ASYNCCPU_PROOF_OF_CONCEPT

    if (push)
    {
        //cudaStreamSynchronize(stream);
        //amgx::thrust::global_thread_handle::threadStream[getCurrentThreadId()] = stream_old;
    }

#endif
}

// setAggregates for block_dia_csr_matrix_d format
template <AMGX_VecPrecision t_vecPrec, AMGX_MatPrecision t_matPrec, AMGX_IndPrecision t_indPrec>
template<int AVG_NNZ>
void Size8Selector<TemplateConfig<AMGX_device, t_vecPrec, t_matPrec, t_indPrec> >:: setAggregates_common_sqblock_avg_specialized(const Matrix_d &A,
        typename Matrix_d::IVector &aggregates, typename Matrix_d::IVector &aggregates_global, int &num_aggregates)
{
#if AMGX_ASYNCCPU_PROOF_OF_CONCEPT
    cudaStream_t stream = amgx::thrust::global_thread_handle::threadStream[getCurrentThreadId()];
#else
    cudaStream_t stream = amgx::thrust::global_thread_handle::get_stream();
#endif
    const IndexType num_block_rows = A.get_num_rows();
    const IndexType num_nonzero_blocks = A.get_num_nz();

    if (!A.is_matrix_singleGPU())
    {
        aggregates.resize(A.manager->halo_offset(A.manager->num_neighbors()));
    }
    else
    {
        aggregates.resize(num_block_rows);
    }

    // Initially, put each vertex in its own aggregate
    thrust_wrapper::sequence<AMGX_device>(aggregates.begin(), aggregates.begin() + num_block_rows);
    cudaCheckError();
    IndexType *aggregates_ptr = aggregates.raw();
    // Create row_indices array
    IndexType total_nz = (A.is_matrix_singleGPU()) ? num_nonzero_blocks : A.manager->num_nz_all();
    typename Matrix_d::IVector row_indices(total_nz);
    cusp::detail::offsets_to_indices(A.row_offsets, row_indices);
    const IndexType *A_row_offsets_ptr = A.row_offsets.raw();
    const IndexType *A_row_indices_ptr = row_indices.raw();
    const IndexType *A_column_indices_ptr = A.col_indices.raw();
    const IndexType *A_dia_idx_ptr = A.diag.raw();
    //const ValueType *A_dia_val_ptr = amgx::thrust::raw_pointer_cast(&A.values[A.get_block_size()*A.diagOffset()]);
    const ValueType *A_nonzero_values_ptr = A.values.raw();
    typename Matrix_d::IVector strongest_neighbour(num_block_rows, -1);
    typename Matrix_d::IVector partner_index(3 * num_block_rows, -1);
    typename Matrix_d::IVector strongest_neighbour_tmp(num_block_rows);
    IndexType *strongest_neighbour_ptr = strongest_neighbour.raw();
    IndexType *partner_index_ptr = partner_index.raw();
    const int threads_per_block = 256;
    const int num_blocks = std::min( AMGX_GRID_MAX_SIZE, (num_block_rows - 1) / threads_per_block + 1);
    int numUnassigned = num_block_rows;
    int numUnassigned_previous = numUnassigned;
    Vector<TemplateConfig<AMGX_device, AMGX_vecFloat, t_matPrec, t_indPrec> > edge_weights(num_nonzero_blocks + 8); //8-padded
    float *edge_weights_ptr = edge_weights.raw();
    float *rand_edge_weights_ptr = NULL;
    const int num_blocks_V2 = std::min( AMGX_GRID_MAX_SIZE, (num_nonzero_blocks - 1) / threads_per_block + 1);
    // Compute the edge weights
#if AMGX_ASYNCCPU_PROOF_OF_CONCEPT
    int avoid_thrust_count = 1;//0;
    int newFindStrongest = 1;//0;
    int newWeights = 1;//0;
#else
    int avoid_thrust_count = 0;//0;
    int newFindStrongest = 0;//0;
    int newWeights = 0;//0;
#endif
    int usenumassignedassumption = false;

    if (newWeights == 0)
    {
        cudaFuncSetCacheConfig(computeEdgeWeightsBlockDiaCsr_V2<IndexType, ValueType, float>, cudaFuncCachePreferL1);
        computeEdgeWeightsBlockDiaCsr_V2 <<< num_blocks_V2, threads_per_block, 0, stream>>>(
            A_row_offsets_ptr, A_row_indices_ptr, A_column_indices_ptr, A_dia_idx_ptr, A_nonzero_values_ptr, num_nonzero_blocks, edge_weights_ptr, rand_edge_weights_ptr, num_block_rows, A.get_block_dimy(), this->m_aggregation_edge_weight_component, this->weight_formula);
        cudaCheckError();
    }
    else
    {
        cudaFuncSetCacheConfig(computeEdgeWeightsBlockDiaCsr_V2_1<AVG_NNZ, IndexType, ValueType>, cudaFuncCachePreferL1);
        computeEdgeWeightsBlockDiaCsr_V2_1<AVG_NNZ, IndexType, ValueType> <<< num_blocks_V2, threads_per_block, 0, stream>>>(
            A_row_offsets_ptr, A_row_indices_ptr, A_column_indices_ptr, A_dia_idx_ptr, A_nonzero_values_ptr, num_nonzero_blocks, edge_weights_ptr, rand_edge_weights_ptr, num_block_rows, A.get_block_dimy(), this->m_aggregation_edge_weight_component);
        cudaCheckError();
    }

    // -------------------------------------------------
    // First create aggregates of size 2
    // -------------------------------------------------
    int icount = 0;
    const int num_blocks_1024 = std::min( 13 * 2, (num_block_rows - 1) / 1024 + 1);
    device_vector_alloc<int> sets_per_block_t(num_blocks_1024);
    int *sets_per_block = amgx::thrust::raw_pointer_cast(sets_per_block_t.data());
    cudaCheckError();

    do
    {
        if (newFindStrongest)
        {
            if (numUnassigned == num_block_rows && usenumassignedassumption)
            {
                cudaFuncSetCacheConfig(my_findStrongestNeighbourBlockDiaCsr_NoMerge<AVG_NNZ, ALGORITHM_NOMERGE, 1, 0, int>, cudaFuncCachePreferL1);
                my_findStrongestNeighbourBlockDiaCsr_NoMerge<AVG_NNZ, ALGORITHM_NOMERGE, 1, 0, int> <<< num_blocks, threads_per_block, 0, stream>>>(
                    A_row_offsets_ptr, A_column_indices_ptr, edge_weights_ptr, num_block_rows, num_nonzero_blocks,  0, 0, strongest_neighbour_ptr, partner_index_ptr, 0, this->deterministic, 0, 0);
            }
            else
            {
                cudaFuncSetCacheConfig(my_findStrongestNeighbourBlockDiaCsr_NoMerge<AVG_NNZ, ALGORITHM_NOMERGE, 0, 0, int>, cudaFuncCachePreferL1);
                my_findStrongestNeighbourBlockDiaCsr_NoMerge<AVG_NNZ, ALGORITHM_NOMERGE, 0, 0, int> <<< num_blocks, threads_per_block, 0, stream>>>(
                    A_row_offsets_ptr, A_column_indices_ptr, edge_weights_ptr, num_block_rows, num_nonzero_blocks, 0, 0, strongest_neighbour_ptr, partner_index_ptr, 0, this->deterministic, 0, 0);
            }
        }
        else
        {
            findStrongestNeighbourBlockDiaCsr_NoMerge <<< num_blocks, threads_per_block, 0, stream>>>(A_row_offsets_ptr, A_column_indices_ptr, edge_weights_ptr, num_block_rows, partner_index_ptr, strongest_neighbour_ptr, this->deterministic);
        }

        cudaCheckError();
        // Look for perfect matches
        numUnassigned_previous = numUnassigned;

        if (avoid_thrust_count == 0)
        {
            matchEdges <<< num_blocks, threads_per_block, 0, stream>>>(num_block_rows, partner_index_ptr, aggregates_ptr, strongest_neighbour_ptr);
            cudaCheckError();
            numUnassigned = (int)amgx::thrust::count(partner_index.begin(), partner_index.begin() + num_block_rows, -1);
            cudaCheckError();
        }
        else
        {
            my_MatchEdges <<< num_blocks_1024, 1024, 0, stream>>>(num_block_rows, partner_index_ptr, aggregates_ptr, strongest_neighbour_ptr, sets_per_block);
            cudaCheckError();
            numUnassigned = numUnassigned_previous - amgx::strided_reduction::count_block_results_pinned_memory(0, icount, num_blocks_1024, sets_per_block, amgx::strided_reduction::op_sum(), stream);
        }

        icount++;
    }
    while (!(numUnassigned == 0 || icount > this->max_iterations || 1.0 * numUnassigned / num_block_rows < this->numUnassigned_tol || numUnassigned_previous == numUnassigned));

    assignUnassignedVertices <<< num_blocks, threads_per_block, 0, stream>>>(partner_index_ptr, num_block_rows);
    cudaCheckError();
    // -------------------------------------------------
    // Merge aggregates to create aggregates of size 4
    // -------------------------------------------------
    Vector<TemplateConfig<AMGX_device, AMGX_vecFloat, t_matPrec, t_indPrec> > weight_strongest_neighbour(num_block_rows, -1);
    float *weight_strongest_neighbour_ptr = weight_strongest_neighbour.raw();
    // At this point, partner index contain either your index or your neighbours index, depending on weither you're matched or not
    // aggregates contain the largest vertex index of vertices in aggregate
    typename Matrix_d::IVector aggregated(num_block_rows, -1);
    IndexType *aggregated_ptr = aggregated.raw();
    // now used as flag to check if aggregated or not
    icount = 0;
    numUnassigned = num_block_rows;
    numUnassigned_previous = numUnassigned;

    do
    {
        // Each vertex stores in strongest_neighbour the aggregates number of strongest neighbour and the weight of connection
        if (newFindStrongest)
        {
            if (numUnassigned == num_block_rows && usenumassignedassumption)
            {
                cudaFuncSetCacheConfig(my_findStrongestNeighbourBlockDiaCsr_NoMerge<AVG_NNZ, ALGORITHM_STOREWEIGHTS, 1, 0, int>, cudaFuncCachePreferL1);
                my_findStrongestNeighbourBlockDiaCsr_NoMerge<AVG_NNZ, ALGORITHM_STOREWEIGHTS, 1, 0, int> <<< num_blocks, threads_per_block, 0, stream>>>(A_row_offsets_ptr, A_column_indices_ptr, edge_weights_ptr, num_block_rows, num_nonzero_blocks, aggregated_ptr, aggregates_ptr, strongest_neighbour_ptr, partner_index_ptr, weight_strongest_neighbour_ptr, this->deterministic, 0, 0);
            }
            else
            {
                cudaFuncSetCacheConfig(my_findStrongestNeighbourBlockDiaCsr_NoMerge<AVG_NNZ, ALGORITHM_STOREWEIGHTS, 0, 0, int>, cudaFuncCachePreferL1);
                my_findStrongestNeighbourBlockDiaCsr_NoMerge<AVG_NNZ, ALGORITHM_STOREWEIGHTS, 0, 0, int> <<< num_blocks, threads_per_block, 0, stream>>>(A_row_offsets_ptr, A_column_indices_ptr, edge_weights_ptr, num_block_rows, num_nonzero_blocks,  aggregated_ptr, aggregates_ptr, strongest_neighbour_ptr, partner_index_ptr, weight_strongest_neighbour_ptr, this->deterministic, 0, 0);
            }
        }
        else
        {
            findStrongestNeighbourBlockDiaCsr_StoreWeight <<< num_blocks, threads_per_block, 0, stream>>>(A_row_offsets_ptr, A_column_indices_ptr, edge_weights_ptr, num_block_rows, aggregated_ptr, aggregates_ptr, strongest_neighbour_ptr, partner_index_ptr, weight_strongest_neighbour_ptr, this->deterministic);
        }

        cudaCheckError();
        // Each vertex in same aggregates will agree on aggregates to propose too, and both store the aggregate number they want to match with
        agreeOnProposal <<< num_blocks, threads_per_block, 0, stream>>>(A_row_offsets_ptr, A_column_indices_ptr, num_block_rows, aggregated_ptr, strongest_neighbour_ptr, weight_strongest_neighbour_ptr, partner_index_ptr, aggregates_ptr);
        cudaCheckError();
        numUnassigned_previous = numUnassigned;

        if (avoid_thrust_count == 0)
        {
            matchAggregatesSize4 <IndexType> <<< num_blocks, threads_per_block, 0, stream>>>(aggregates_ptr, aggregated_ptr, strongest_neighbour_ptr, partner_index_ptr, num_block_rows);
            numUnassigned = amgx::thrust::count(aggregated.begin(), aggregated.end(), -1);
        }
        else
        {
            my_matchAggregatesSize4 <<< num_blocks_1024, 1024, 0, stream>>>(aggregates_ptr, aggregated_ptr, strongest_neighbour_ptr, partner_index_ptr, num_block_rows, sets_per_block);
            numUnassigned = numUnassigned_previous - amgx::strided_reduction::count_block_results_pinned_memory(1, icount, num_blocks_1024, sets_per_block, amgx::strided_reduction::op_sum(), stream);
        }

        cudaCheckError();
        icount++;
    }
    while (!(numUnassigned == 0 || icount > this->max_iterations || 1.0 * numUnassigned / num_block_rows < this->numUnassigned_tol || numUnassigned == numUnassigned_previous) );

    assignUnassignedVertices_2 <<< num_blocks, threads_per_block, 0, stream>>>(partner_index_ptr, num_block_rows);
    cudaCheckError();
    // -------------------------------------------------
    // Merge aggregates to create aggregates of size 8
    // -------------------------------------------------
    thrust_wrapper::fill<AMGX_device>(aggregated.begin(), aggregated.end(), -1);
    cudaCheckError();
    thrust_wrapper::fill<AMGX_device>(weight_strongest_neighbour.begin(), weight_strongest_neighbour.end(), -1.);
    cudaCheckError();
    icount = 0;
    numUnassigned = num_block_rows;
    numUnassigned_previous = numUnassigned;

    do
    {
        // Each vertex stores in strongest_neighbour the aggregates number of strongest neighbour and the weight of connection
        if (newFindStrongest)
        {
            if (numUnassigned == num_block_rows && usenumassignedassumption)
            {
                cudaFuncSetCacheConfig(my_findStrongestNeighbourBlockDiaCsr_NoMerge<AVG_NNZ, ALGORITHM_STOREWEIGHTS_2, 1, 0, int>, cudaFuncCachePreferL1);
                my_findStrongestNeighbourBlockDiaCsr_NoMerge<AVG_NNZ, ALGORITHM_STOREWEIGHTS_2, 1, 0, int> <<< num_blocks, threads_per_block, 0, stream>>>(A_row_offsets_ptr, A_column_indices_ptr, edge_weights_ptr, num_block_rows, num_nonzero_blocks,  aggregated_ptr, aggregates_ptr, strongest_neighbour_ptr, partner_index_ptr, weight_strongest_neighbour_ptr, this->deterministic, 0, 0);
            }
            else
            {
                cudaFuncSetCacheConfig(my_findStrongestNeighbourBlockDiaCsr_NoMerge<AVG_NNZ, ALGORITHM_STOREWEIGHTS_2, 0, 0, int>, cudaFuncCachePreferL1);
                my_findStrongestNeighbourBlockDiaCsr_NoMerge<AVG_NNZ, ALGORITHM_STOREWEIGHTS_2, 0, 0, int> <<< num_blocks, threads_per_block, 0, stream>>>(A_row_offsets_ptr, A_column_indices_ptr, edge_weights_ptr, num_block_rows, num_nonzero_blocks, aggregated_ptr, aggregates_ptr, strongest_neighbour_ptr, partner_index_ptr, weight_strongest_neighbour_ptr, this->deterministic, 0, 0);
            }
        }
        else
        {
            findStrongestNeighbourBlockDiaCsr_StoreWeight_2 <<< num_blocks, threads_per_block, 0, stream>>>(A_row_offsets_ptr, A_column_indices_ptr, edge_weights_ptr, num_block_rows, aggregated_ptr, aggregates_ptr, strongest_neighbour_ptr, partner_index_ptr, weight_strongest_neighbour_ptr);
        }

        cudaCheckError();

        // Each vertex in same aggregates will agree on aggregates to propose too, and both store the aggregate number they want to match with
        if (!this->deterministic)
        {
            agreeOnProposal_2 <<< num_blocks, threads_per_block, 0, stream>>>(A_row_offsets_ptr, A_column_indices_ptr, num_block_rows, aggregated_ptr, strongest_neighbour_ptr, weight_strongest_neighbour_ptr, partner_index_ptr, aggregates_ptr, this->deterministic);
        }
        else
        {
            //strongest_neighbour_tmp = strongest_neighbour; // copied that directly in the kernel
            agreeOnProposal_2_deterministic <<< num_blocks, threads_per_block, 0, stream>>>(
                strongest_neighbour_tmp.raw(),
                A_row_offsets_ptr,
                A_column_indices_ptr, num_block_rows,
                aggregated_ptr, strongest_neighbour_ptr,
                weight_strongest_neighbour_ptr, partner_index_ptr, aggregates_ptr, this->deterministic);
            strongest_neighbour_tmp.swap(strongest_neighbour);
            strongest_neighbour_ptr = strongest_neighbour.raw(); //re-saving the correct pointer..
        }

        cudaCheckError();
        numUnassigned_previous = numUnassigned;

        if (avoid_thrust_count == 0)
        {
            matchAggregates <IndexType> <<< num_blocks, threads_per_block, 0, stream>>>(aggregates_ptr, aggregated_ptr, strongest_neighbour_ptr, num_block_rows);
            numUnassigned = amgx::thrust::count(aggregated.begin(), aggregated.end(), -1);
        }
        else
        {
            my_matchAggregates <<< num_blocks_1024, 1024, 0, stream>>>(aggregates_ptr, aggregated_ptr, strongest_neighbour_ptr, num_block_rows, sets_per_block);
            numUnassigned = numUnassigned_previous - amgx::strided_reduction::count_block_results_pinned_memory(2, icount, num_blocks_1024, sets_per_block, amgx::strided_reduction::op_sum(), stream);
        }

        cudaCheckError();
        icount++;
    }
    while (!(numUnassigned == 0 || icount > this->max_iterations || 1.0 * numUnassigned / num_block_rows < this->numUnassigned_tol || numUnassigned_previous == numUnassigned) );

    // Merge remaining vertices with current aggregates
    int local_iter = 0;

    if (!this->deterministic)
    {
        while (numUnassigned != 0)
        {
            mergeWithExistingAggregatesBlockDiaCsr <<< num_blocks, threads_per_block, 0, stream>>>(A_row_offsets_ptr, A_column_indices_ptr, edge_weights_ptr, num_block_rows, aggregates_ptr, aggregated_ptr, this->deterministic, (IndexType *) NULL, local_iter > 1);
            cudaCheckError();
            numUnassigned_previous = numUnassigned;
            numUnassigned = (int)amgx::thrust::count(aggregated.begin(), aggregated.end(), -1);
            cudaCheckError();
            local_iter++;
        }
    }
    else
    {
        typename Matrix_d::IVector aggregates_candidate(num_block_rows, -1);

        while (numUnassigned != 0)
        {
            // allow singletons only from the 2nd local iteration
            mergeWithExistingAggregatesBlockDiaCsr <<< num_blocks, threads_per_block, 0, stream>>>(A_row_offsets_ptr, A_column_indices_ptr, edge_weights_ptr, num_block_rows, aggregates_ptr, aggregated_ptr, this->deterministic, aggregates_candidate.raw(), local_iter > 1);
            cudaCheckError();
            numUnassigned_previous = numUnassigned;

            if (avoid_thrust_count == 0)
            {
                joinExistingAggregates <<< num_blocks, threads_per_block, 0, stream>>>(num_block_rows, aggregates_ptr, aggregated_ptr, aggregates_candidate.raw());
                numUnassigned = (int)amgx::thrust::count(aggregated.begin(), aggregated.end(), -1);
            }
            else
            {
                my_joinExistingAggregates <<< num_blocks_1024, 1024, 0, stream>>>(num_block_rows, aggregates_ptr, aggregated_ptr, aggregates_candidate.raw(), sets_per_block);
                numUnassigned = numUnassigned_previous - amgx::strided_reduction::count_block_results_pinned_memory(3, local_iter, num_blocks_1024, sets_per_block, amgx::strided_reduction::op_sum(), stream);
            }

            cudaCheckError();
            local_iter++;
        }

        aggregates_candidate.resize(0);
    }

    this->renumberAndCountAggregates(aggregates, aggregates_global, num_block_rows, num_aggregates);
    //analyze_coloring(aggregates, A.getMatrixColoring().getRowColors());
}

template <class T_Config>
void Size8SelectorBase<T_Config>::setAggregates(Matrix<T_Config> &A,
        IVector &aggregates, IVector &aggregates_global, int &num_aggregates)
{
    if (A.get_block_dimx() == A.get_block_dimy())
    {
        setAggregates_common_sqblock( A, aggregates, aggregates_global, num_aggregates );
    }
    else
    {
        FatalError("Unsupported block size for Size8Selector", AMGX_ERR_NOT_SUPPORTED_BLOCKSIZE);
    }
}

// -------------------------
//  Explict instantiations
// -------------------------
#define AMGX_CASE_LINE(CASE) template class Size8SelectorBase<TemplateMode<CASE>::Type>;
AMGX_FORALL_BUILDS(AMGX_CASE_LINE)
AMGX_FORCOMPLEX_BUILDS(AMGX_CASE_LINE)
#undef AMGX_CASE_LINE
#define AMGX_CASE_LINE(CASE) template class Size8Selector<TemplateMode<CASE>::Type>;
AMGX_FORALL_BUILDS(AMGX_CASE_LINE)
AMGX_FORCOMPLEX_BUILDS(AMGX_CASE_LINE)
#undef AMGX_CASE_LINE
}
}
}
