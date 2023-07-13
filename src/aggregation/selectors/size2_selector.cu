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

#include <aggregation/selectors/size2_selector.h>
#include <cutil.h>
#include <util.h>
#include <types.h>
#include <basic_types.h>
#include <texture.h>
#include <matrix_analysis.h>

#include <async_event.h>
#include <determinism_checker.h>

#include <thrust/count.h> //count
#include <thrust/sort.h> //sort
#include <thrust/binary_search.h> //lower_bound
#include <thrust/unique.h> //unique
#include <cusp/detail/format_utils.h> //offsets_to_indices

#define EXPERIMENTAL_ITERATIVE_MATCHING

namespace amgx
{
namespace aggregation
{
namespace size2_selector
{

// include common routines for all selectors
#include <aggregation/selectors/common_selector.h>

// ------------------------
//  Kernels
// ------------------------

#ifndef DELETE
// Kernel to compute the weight of the edges with block_dia_csr format
template <typename IndexType, typename ValueType>
__global__
void computeEdgeWeightsBlockDiaCsr(const IndexType *row_offsets, const IndexType *column_indices,
                                   const ValueType *dia_values, const ValueType *nonzero_values, const IndexType num_block_rows, float *edge_weights, int bsize)
{
    int tid = threadIdx.x + blockDim.x * blockIdx.x;
    int jcol;
    int bsize_sq = bsize * bsize;

    while (tid < num_block_rows)
    {
        for (int j = row_offsets[tid]; j < row_offsets[tid + 1]; j++)
        {
            jcol = column_indices[j];

            if (jcol >= num_block_rows) { continue; }

            // Compute edge weight
            for (int k = row_offsets[jcol]; k < row_offsets[jcol + 1]; k++)
            {
                if (column_indices[k] == tid)
                {
                    edge_weights[j] =  (float) 0.5 * (fabs(nonzero_values[j * bsize_sq]) + fabs(nonzero_values[k * bsize_sq]))
                                       / max( fabs(dia_values[tid * bsize_sq]), fabs(dia_values[jcol * bsize_sq]));
                    break;
                }
            }
        }

        tid += gridDim.x * blockDim.x;
    }
}
#endif

// findStrongestNeighbour kernel for csr_matrix format
// computes weight on the fly
template <typename IndexType, typename ValueType>
__global__
void findStrongestNeighbourCsr(const IndexType *row_offsets, const IndexType *column_indices,
                               const ValueType *values, const ValueType *diag, const IndexType num_rows, IndexType *aggregates, int *strongest_neighbour)
{
    int tid = threadIdx.x + blockDim.x * blockIdx.x;
    ValueType weight;
    int jcol;

    while (tid < num_rows)
    {
        ValueType max_weight_unaggregated = 0.;
        ValueType max_weight_aggregated = 0.;
        int strongest_unaggregated = -1;
        int strongest_aggregated = -1;

        if (aggregates[tid] == -1) // Unaggregated row
        {
            for (int j = row_offsets[tid]; j < row_offsets[tid + 1]; j++)
            {
                jcol = column_indices[j];

                if (tid == jcol || jcol >= num_rows) { continue; }

                // Compute edge weight
                //weight = fabs(values[j])/max( fabs(diag[tid]),fabs(diag[jcol]));
                for (int k = row_offsets[jcol]; k < row_offsets[jcol + 1]; k++)
                {
                    if (column_indices[k] == tid)
                    {
                        weight = 0.5 * (fabs(values[j]) + fabs(values[k])) / max( fabs(diag[tid]), fabs(diag[jcol]));
                        break;
                    }
                }

                // Identify strongest aggregated and unaggregated neighbours
                if (aggregates[jcol] == -1 && (weight > max_weight_unaggregated || (weight == max_weight_unaggregated && jcol > strongest_unaggregated))) // unaggregated
                {
                    max_weight_unaggregated = weight;
                    strongest_unaggregated = jcol;
                }
                else if (aggregates[jcol] != -1 && (weight > max_weight_aggregated || (weight == max_weight_aggregated && jcol > strongest_aggregated))) // aggregated
                {
                    max_weight_aggregated = weight;
                    strongest_aggregated = jcol;
                }
            }

            if (strongest_unaggregated == -1 && strongest_aggregated != -1) // All neighbours are aggregated
            {
                aggregates[tid] = aggregates[strongest_aggregated];
            }
            else if (strongest_unaggregated != -1)
            {
                strongest_neighbour[tid] = strongest_unaggregated;
            }
            else
            {
                strongest_neighbour[tid] = tid;
            }
        }

        tid += gridDim.x * blockDim.x;
    }
}

// findStrongestNeighbour kernel for block_dia_csr_matrix format
// computes weight on the fly
template <typename IndexType, typename ValueType>
__global__
void findStrongestNeighbourBlockDiaCsr(const IndexType *row_offsets, const IndexType *column_indices,
                                       const ValueType *dia_values, const ValueType *nonzero_values, const IndexType num_block_rows, IndexType *aggregates, int *strongest_neighbour, int bsize)
{
    int tid = threadIdx.x + blockDim.x * blockIdx.x;
    ValueType weight;
    int jcol;
    int bsize_sq = bsize * bsize;

    while (tid < num_block_rows)
    {
        ValueType max_weight_unaggregated = 0.;
        ValueType max_weight_aggregated = 0.;
        int strongest_unaggregated = -1;
        int strongest_aggregated = -1;

        if (aggregates[tid] == -1) // Unaggregated row
        {
            for (int j = row_offsets[tid]; j < row_offsets[tid + 1]; j++)
            {
                jcol = column_indices[j];

                if (jcol >= num_block_rows) { continue; }

                // Compute edge weight
                for (int k = row_offsets[jcol]; k < row_offsets[jcol + 1]; k++)
                {
                    if (column_indices[k] == tid)
                    {
                        weight = 0.5 * (fabs(nonzero_values[j * bsize_sq]) + fabs(nonzero_values[k * bsize_sq]))
                                 / max( fabs(dia_values[tid * bsize_sq]), fabs(dia_values[jcol * bsize_sq]));
                        break;
                    }
                }

                // Identify strongest aggregated and unaggregated neighbours
                if (aggregates[jcol] == -1 && (weight > max_weight_unaggregated || (weight == max_weight_unaggregated && jcol > strongest_unaggregated))) // unaggregated
                {
                    max_weight_unaggregated = weight;
                    strongest_unaggregated = jcol;
                }
                else if (aggregates[jcol] != -1 && (weight > max_weight_aggregated || (weight == max_weight_aggregated && jcol > strongest_aggregated))) // aggregated
                {
                    max_weight_aggregated = weight;
                    strongest_aggregated = jcol;
                }
            }

            if (strongest_unaggregated == -1 && strongest_aggregated != -1) // All neighbours are aggregated
                // Put in same aggregate as strongest neighbour
            {
                aggregates[tid] = aggregates[strongest_aggregated];
            }
            else if (strongest_unaggregated != -1)
            {
                strongest_neighbour[tid] = strongest_unaggregated;
            }
            else
            {
                strongest_neighbour[tid] = tid;
            }
        }

        tid += gridDim.x * blockDim.x;
    }
}


__device__
float random_weight2(int i, int j)
{
#define RAND_MULTIPLIER         1145637293
    unsigned long i_min = (min(i, j) * RAND_MULTIPLIER);
    unsigned long i_max = (max(i, j) * RAND_MULTIPLIER);
    return ((float)i_min / i_max);
}


// findStrongestNeighbour kernel for block_dia_csr_matrix format
// Reads the weight from edge_weights array
template <typename IndexType>
__global__
void findStrongestNeighbourBlockDiaCsr_V2(const IndexType *row_offsets, const IndexType *column_indices,
        float *edge_weights, IndexType num_block_rows, IndexType *aggregates,
        IndexType *strongest_neighbour_1phase, IndexType *strongest_neighbour,
        const size_t bsize, int phase, bool merge_singletons)
{
    int tid = threadIdx.x + blockDim.x * blockIdx.x;
    float   weight;
    int jcol;

    while (tid < num_block_rows)
    {
        int strongest_unaggregated = -1;
        int strongest_aggregated = -1;
        float   max_weight_unaggregated = 0.;
        float   max_weight_aggregated = 0.;

        if (aggregates[tid] == -1) // Unaggregated row
        {
            for (int j = row_offsets[tid]; j < row_offsets[tid + 1]; j++)
            {
                jcol = column_indices[j];

                if (phase == 1) { weight = edge_weights[j]; }
                else { weight = random_weight2(tid, jcol); }

                if (tid == jcol || jcol >= num_block_rows) { continue; }    // skip diagonal and halo

                if (phase == 2 && strongest_neighbour_1phase[jcol] != tid) { continue; } // if 2nd phase only accept those who gave a hand on the 1st phase

                // Identify strongest aggregated and unaggregated neighbours
                if (aggregates[jcol] == -1 && (weight > max_weight_unaggregated || (weight == max_weight_unaggregated && jcol > strongest_unaggregated))) // unaggregated
                {
                    max_weight_unaggregated = weight;
                    strongest_unaggregated = jcol;
                }
                else if (aggregates[jcol] != -1 && (weight > max_weight_aggregated || (weight == max_weight_aggregated && jcol > strongest_aggregated))) // aggregated
                {
                    max_weight_aggregated = weight;
                    strongest_aggregated = jcol;
                }
            }

            if (strongest_unaggregated == -1 && strongest_aggregated != -1) // All neighbours are aggregated
            {
                if ( merge_singletons )
                    // Put in same aggregate as strongest neighbour
                {
                    aggregates[tid] = aggregates[strongest_aggregated];
                }
                else
                {
                    aggregates[tid] = tid;
                }
            }
            else if (strongest_unaggregated != -1)
            {
                if (phase == 2)
                {
                    float rand_w1 = random_weight2(tid, strongest_neighbour_1phase[tid]);
                    strongest_neighbour[tid] = max_weight_unaggregated > rand_w1 ? strongest_unaggregated : strongest_neighbour_1phase[tid];
                }
                else { strongest_neighbour_1phase[tid] = strongest_unaggregated; }
            }
            else
            {
                if (phase == 2) { strongest_neighbour[tid] = strongest_neighbour_1phase[tid]; }
                else { strongest_neighbour_1phase[tid] = tid; }
            }
        }

        tid += gridDim.x * blockDim.x;
    }
}

// Kernel that checks if perfect matchs exist
template <typename IndexType>
__global__
void matchEdges(const IndexType num_rows, IndexType *aggregates, int *strongest_neighbour)
{
    int tid = threadIdx.x + blockDim.x * blockIdx.x;
    int potential_match, potential_match_neighbour;

    while (tid < num_rows)
    {
        if (aggregates[tid] == -1) // Unaggregated row
        {
            potential_match = strongest_neighbour[tid];
            potential_match_neighbour = strongest_neighbour[potential_match];

            if (potential_match != -1 && potential_match_neighbour == tid) // we have a match
            {
                aggregates[tid] = ( potential_match > tid ) ? tid : potential_match;
            }
        }

        tid += gridDim.x * blockDim.x;
    }
}

template <typename IndexType, int block_size>
__global__
void countAggregates(const IndexType num_rows, IndexType *aggregates, int *num_unaggregated)
{
    int tid = threadIdx.x + blockDim.x * blockIdx.x;
    int c = 0;
    int i = tid;

    while ( i < num_rows )
    {
        c += ( aggregates[i] == -1 );
        i += gridDim.x * blockDim.x;
    }

    __shared__ int smem[block_size];
    smem[threadIdx.x] = c;
    __syncthreads();

    for ( int off = blockDim.x / 2; off >= 1; off = off / 2 )
    {
        if ( threadIdx.x < off )
        {
            smem[threadIdx.x] += smem[threadIdx.x + off];
        }
        __syncthreads();
    }

    if ( threadIdx.x == 0 )
    {
        atomicAdd(num_unaggregated, smem[0]);
    }
}

// Kernel that merges unaggregated vertices its strongest aggregated neighbour
// For csr_matrix_format
template <typename IndexType, typename ValueType>
__global__
void mergeWithExistingAggregatesCsr(const IndexType *row_offsets, const IndexType *column_indices, const ValueType *values,
                                    const ValueType *diag, const int num_rows, IndexType *aggregates, int deterministic, IndexType *aggregates_candidate)
{
    int tid = threadIdx.x + blockDim.x * blockIdx.x;
    int jcol;
    ValueType weight;

    while (tid < num_rows)
    {
        ValueType max_weight_aggregated = 0.;
        int strongest_aggregated = -1;

        if (aggregates[tid] == -1) // Unaggregated row
        {
            for (int j = row_offsets[tid]; j < row_offsets[tid + 1]; j++)
            {
                jcol = column_indices[j];

                if (tid == jcol || jcol >= num_rows) { continue; }

                // Compute edge weight
                weight = fabs(values[j]) / max( fabs(diag[tid]), fabs(diag[jcol]));

                // Identify strongest aggregated neighbour
                if (aggregates[jcol] != -1 && (weight > max_weight_aggregated || (weight == max_weight_aggregated && jcol > strongest_aggregated))) // aggregated
                {
                    max_weight_aggregated = weight;
                    strongest_aggregated = jcol;
                }
            }

            if (strongest_aggregated != -1) // Found a neighbour to aggregate to
            {
                if (deterministic)
                {
                    aggregates_candidate[tid] = aggregates[strongest_aggregated];
                }
                else
                {
                    // Put in same aggregate as strongest neighbour
                    aggregates[tid] = aggregates[strongest_aggregated];
                }
            }
            else // All neighbours are unaggregated, leave alone
            {
                if (deterministic)
                {
                    aggregates_candidate[tid] = tid;
                }
                else
                {
                    aggregates[tid] = tid;
                }
            }
        }

        tid += gridDim.x * blockDim.x;
    }
}


template <typename IndexType>
__global__
void joinExistingAggregates(IndexType num_rows, IndexType *aggregates, IndexType *aggregates_candidate)
{
    int tid = threadIdx.x + blockDim.x * blockIdx.x;

    while (tid < num_rows)
    {
        if (aggregates[tid] == -1 && aggregates_candidate[tid] != -1) // Unaggregated row
        {
            aggregates[tid] = aggregates_candidate[tid];
        }

        tid += gridDim.x * blockDim.x;
    }
}

// Kernel that merges unaggregated vertices its strongest aggregated neighbour
// Edge weights are computed on the fly
// For block_dia_csr_matrix_format
template <typename IndexType, typename ValueType>
__global__
void mergeWithExistingAggregatesBlockDiaCsr(const IndexType *row_offsets, const IndexType *column_indices, const ValueType *dia_values, const ValueType *nonzero_values,
        const int num_block_rows, IndexType *aggregates, int bsize, int deterministic, IndexType *aggregates_candidate)
{
    int tid = threadIdx.x + blockDim.x * blockIdx.x;
    int jcol;
    ValueType weight;
    int bsize_sq = bsize * bsize;

    while (tid < num_block_rows)
    {
        int strongest_aggregated = -1;
        ValueType max_weight_aggregated = 0.;

        if (aggregates[tid] == -1) // Unaggregated row
        {
            for (int j = row_offsets[tid]; j < row_offsets[tid + 1]; j++)
            {
                jcol = column_indices[j];

                if (jcol >= num_block_rows) { continue; }

                // Compute edge weight
                weight = fabs(nonzero_values[j * bsize_sq]) / max( fabs(dia_values[tid * bsize_sq]), fabs(dia_values[jcol * bsize_sq]));

                // Identify strongest aggregated neighbour
                if (aggregates[jcol] != -1 && (weight > max_weight_aggregated || (weight == max_weight_aggregated && jcol > strongest_aggregated))) // aggregated
                {
                    max_weight_aggregated = weight;
                    strongest_aggregated = jcol;
                }
            }

            if (strongest_aggregated != -1) // Found a neighbour to aggregate to
            {
                if (deterministic)
                {
                    aggregates_candidate[tid] = aggregates[strongest_aggregated];
                }
                else
                {
                    // Put in same aggregate as strongest neighbour
                    aggregates[tid] = aggregates[strongest_aggregated];
                }
            }
            else // All neighbours are unaggregated, leave alone
            {
                if (deterministic)
                {
                    aggregates_candidate[tid] = tid;
                }
                else
                {
                    aggregates[tid] = tid;
                }
            }
        }

        tid += gridDim.x * blockDim.x;
    }
}

// Kernel that merges unaggregated vertices its strongest aggregated neighbour
// Weights are read from edge_weights array
// For block_dia_csr_matrix_format
template <typename IndexType>
__global__
void mergeWithExistingAggregatesBlockDiaCsr_V2(const IndexType *row_offsets, const IndexType *column_indices, const float *edge_weights,
        const int num_block_rows, IndexType *aggregates, int bsize, const int deterministic, IndexType *aggregates_candidate)
{
    int tid = threadIdx.x + blockDim.x * blockIdx.x;
    int jcol;
    float weight;

    while (tid < num_block_rows)
    {
        float max_weight_aggregated = 0.;
        int strongest_aggregated = -1;

        if (aggregates[tid] == -1) // Unaggregated row
        {
            for (int j = row_offsets[tid]; j < row_offsets[tid + 1]; j++)
            {
                // Compute edge weight
                weight = edge_weights[j];
                jcol = column_indices[j];

                if (jcol == tid || jcol >= num_block_rows) { continue; }    // skip diagonal

                // Identify strongest aggregated neighbour
                if (aggregates[jcol] != -1 && (weight > max_weight_aggregated || (weight == max_weight_aggregated && jcol > strongest_aggregated))) //
                {
                    max_weight_aggregated = weight;
                    strongest_aggregated = jcol;
                }
            }

            if (strongest_aggregated != -1) // Found a neighbour to aggregate to
            {
                if (deterministic)
                {
                    aggregates_candidate[tid] = aggregates[strongest_aggregated];
                }
                else
                {
                    // Put in same aggregate as strongest neighbour
                    aggregates[tid] = aggregates[strongest_aggregated];
                }
            }
            else // All neighbours are unaggregated, leave alone
            {
                if (deterministic)
                {
                    aggregates_candidate[tid] = tid;
                }
                else
                {
                    aggregates[tid] = tid;
                }
            }
        }

        tid += gridDim.x * blockDim.x;
    }
}

// Kernel to extract diagonal for csr_matrix format
template <typename IndexType, typename ValueType>
__global__
void getDiagonalKernel(const IndexType *offsets, const IndexType *column_indices,
                       const ValueType *values, const IndexType numRows, ValueType *diagonal)
{
    int tIdx = threadIdx.x + blockDim.x * blockIdx.x;

    while (tIdx < numRows)
    {
        const int offset = offsets[tIdx];
        const int numj = offsets[tIdx + 1] - offset;

        for (int j = offset; j < offset + numj; j++)
        {
            int jcol = column_indices[j];

            if (tIdx == jcol)
            {
                diagonal[tIdx] = values[j];
            }
        }

        tIdx += gridDim.x * blockDim.x;
    }
}

// Kernel to extract diagonal for csr_matrix format
template <typename IndexType, typename ValueType>
__global__
void getDiagonalKernelNoDiaProp(const IndexType *dia_idx, const ValueType *values, const IndexType numRows, ValueType *diagonal)
{
    int tIdx = threadIdx.x + blockDim.x * blockIdx.x;

    while (tIdx < numRows)
    {
        diagonal[tIdx] = values[dia_idx[tIdx]];
        tIdx += gridDim.x * blockDim.x;
    }
}

// -----------------
//  Methods
// ----------------

// Constructor
template<class T_Config>
Size2SelectorBase<T_Config>::Size2SelectorBase(AMG_Config &cfg, const std::string &cfg_scope)
{
    deterministic = cfg.AMG_Config::template getParameter<IndexType>("determinism_flag", "default");
    max_iterations = cfg.AMG_Config::template getParameter<IndexType>("max_matching_iterations", cfg_scope);
    numUnassigned_tol = cfg.AMG_Config::template getParameter<double>("max_unassigned_percentage", cfg_scope);
    two_phase = cfg.AMG_Config::template getParameter<int>("handshaking_phases", cfg_scope) == 2;
    m_aggregation_edge_weight_component = cfg.AMG_Config::template getParameter<int>("aggregation_edge_weight_component", cfg_scope);
    merge_singletons = cfg.AMG_Config::template getParameter<int>("merge_singletons", cfg_scope) == 1;
    weight_formula = cfg.AMG_Config::template getParameter<int>("weight_formula", cfg_scope);
}

// setAggregates for csr_matrix_h format
template <AMGX_VecPrecision t_vecPrec, AMGX_MatPrecision t_matPrec, AMGX_IndPrecision t_indPrec>
void Size2Selector<TemplateConfig<AMGX_host, t_vecPrec, t_matPrec, t_indPrec> >::setAggregates_1x1(const Matrix_h &A,
        typename Matrix_h::IVector &aggregates,  typename Matrix_h::IVector &aggregates_global, int &num_aggregates)
{
    FatalError("Size2 selector: setAggregates not implemented on CPU, exiting", AMGX_ERR_NOT_SUPPORTED_TARGET);
}

// setAggregates for block_dia_csr_matrix_h format
template <AMGX_VecPrecision t_vecPrec, AMGX_MatPrecision t_matPrec, AMGX_IndPrecision t_indPrec>
void Size2Selector<TemplateConfig<AMGX_host, t_vecPrec, t_matPrec, t_indPrec> >::setAggregates_common_sqblocks(const Matrix_h &A,
        typename Matrix_h::IVector &aggregates, typename Matrix_h::IVector &aggregates_global, int &num_aggregates)
{
    FatalError("Size2 selector: setAggregates not implemented on CPU, exiting", AMGX_ERR_NOT_SUPPORTED_TARGET);
}

#ifndef DELETE
// setAggregates for csr_matrix_d format
template <AMGX_VecPrecision t_vecPrec, AMGX_MatPrecision t_matPrec, AMGX_IndPrecision t_indPrec>
void Size2Selector<TemplateConfig<AMGX_device, t_vecPrec, t_matPrec, t_indPrec> >::setAggregates_1x1(
    const Matrix_d &A, typename Matrix_d::IVector &aggregates, typename Matrix_d::IVector &aggregates_global, int &num_aggregates)

{
    if (!A.is_matrix_singleGPU())
    {
        aggregates.resize(A.manager->halo_offset(A.manager->num_neighbors()));
    }
    else
    {
        aggregates.resize(A.get_num_rows());
    }

    thrust::fill(aggregates.begin(), aggregates.end(), -1);
    cudaCheckError();
    //typedef typename csr_matrix_d::index_type IndexType;
    //typedef typename csr_matrix_d::value_type ValueType;
    const IndexType *A_row_offsets_ptr = A.row_offsets.raw();
    const IndexType *A_column_indices_ptr = A.col_indices.raw();
    const ValueType *A_values_ptr = A.values.raw();
    const IndexType *A_dia_ptr = A.diag.raw();
    IndexType num_rows = (int)A.get_num_rows();
    typename Matrix_d::IVector strongest_neighbour(num_rows, -1);
    typename Matrix_d::MVector diag(num_rows, 0);
    Vector<TemplateConfig<AMGX_device, AMGX_vecUInt, t_matPrec, t_indPrec> > aggregated(num_rows, 0);
    IndexType *strongest_neighbour_ptr = strongest_neighbour.raw();
    ValueType *diag_ptr = diag.raw();
    IndexType *aggregates_ptr = aggregates.raw();
    const int threads_per_block = 256;
    const int num_blocks = std::min( AMGX_GRID_MAX_SIZE, (num_rows - 1) / threads_per_block + 1 );
    getDiagonalKernelNoDiaProp <<< num_blocks, threads_per_block>>>(A_dia_ptr, A_values_ptr, num_rows, diag_ptr);
    cudaCheckError();
    int numUnassigned = num_rows;
    int numUnassigned_previous = numUnassigned;
    int icount = 0;

    do
    {
        // For each row, find the strongest neighbour who hasn't been assigned
        findStrongestNeighbourCsr <<< num_blocks, threads_per_block>>>(A_row_offsets_ptr, A_column_indices_ptr, A_values_ptr, diag_ptr, num_rows, aggregates_ptr, strongest_neighbour_ptr);
        cudaCheckError();
        // Look for perfect matches. Also, for nodes without unaggregated neighbours, merge with aggregate containing strongest neighbour
        matchEdges <<< num_blocks, threads_per_block>>>(num_rows, aggregates_ptr, strongest_neighbour_ptr);
        cudaCheckError();
        numUnassigned_previous = numUnassigned;
        numUnassigned = (int)thrust::count(aggregates.begin(), aggregates.begin() + num_rows, -1);
        cudaCheckError();
        icount++;
    }
    while (!(numUnassigned == 0 || icount > this->max_iterations || 1.0 * numUnassigned / num_rows < this->numUnassigned_tol || numUnassigned_previous == numUnassigned));

    // Merge remaining vertices with current aggregates
    if (!this->deterministic)
    {
        while (numUnassigned != 0 )
        {
            mergeWithExistingAggregatesCsr <<< num_blocks, threads_per_block>>>(A_row_offsets_ptr, A_column_indices_ptr, A_values_ptr,
                    diag_ptr, num_rows, aggregates_ptr, this->deterministic, (IndexType *) NULL);
            cudaCheckError();
            numUnassigned = (int)thrust::count(aggregates.begin(), aggregates.begin() + num_rows, -1);
            cudaCheckError();
        };
    }
    else
    {
        typename Matrix_d::IVector aggregates_candidate(num_rows, -1);

        while (numUnassigned != 0 )
        {
            mergeWithExistingAggregatesCsr <<< num_blocks, threads_per_block>>>(A_row_offsets_ptr, A_column_indices_ptr, A_values_ptr,
                    diag_ptr, num_rows, aggregates_ptr, this->deterministic, aggregates_candidate.raw());
            // Sync here
            joinExistingAggregates <<< num_blocks, threads_per_block>>>(num_rows, aggregates_ptr, aggregates_candidate.raw());
            cudaCheckError();
            numUnassigned = (int)thrust::count(aggregates.begin(), aggregates.begin() + num_rows, -1);
            cudaCheckError();
        };

        aggregates_candidate.resize(0);
    }

    this->renumberAndCountAggregates(aggregates, aggregates_global, num_rows, num_aggregates);
}
#endif

// setAggregates for block_dia_csr_matrix_d format
template <AMGX_VecPrecision t_vecPrec, AMGX_MatPrecision t_matPrec, AMGX_IndPrecision t_indPrec>
void Size2Selector<TemplateConfig<AMGX_device, t_vecPrec, t_matPrec, t_indPrec> >::setAggregates_common_sqblocks(
    const Matrix_d &A, typename Matrix_d::IVector &aggregates, typename Matrix_d::IVector &aggregates_global, int &num_aggregates)
{
    IndexType num_block_rows = (int) A.get_num_rows();
    IndexType num_nonzero_blocks = (int) A.get_num_nz();
    // both ways are supported
    IndexType total_nz = (A.is_matrix_singleGPU()) ? num_nonzero_blocks : A.manager->num_nz_all();
    typename Matrix_d::IVector row_indices(total_nz);
    cusp::detail::offsets_to_indices(A.row_offsets, row_indices);
    IndexType total_rows = (A.is_matrix_singleGPU()) ? A.get_num_rows() : A.manager->num_rows_all();
    aggregates.resize(total_rows);
    thrust::fill(aggregates.begin(), aggregates.end(), -1);
    cudaCheckError();
    const IndexType *A_row_offsets_ptr = A.row_offsets.raw();
    const IndexType *A_row_indices_ptr = row_indices.raw();
    const IndexType *A_column_indices_ptr = A.col_indices.raw();
    const IndexType *A_dia_idx_ptr = A.diag.raw();
    const ValueType *A_nonzero_values_ptr = A.values.raw();
    typename Matrix_d::IVector strongest_neighbour(num_block_rows, -1);
    typename Matrix_d::IVector strongest_neighbour_1phase(num_block_rows, -1);
    Vector<TemplateConfig<AMGX_device, AMGX_vecUInt, t_matPrec, t_indPrec> > aggregated(num_block_rows, 0);
    IndexType *strongest_neighbour_ptr = strongest_neighbour.raw();
    IndexType *strongest_neighbour_1phase_ptr = strongest_neighbour_1phase.raw();
    IndexType *aggregates_ptr = aggregates.raw();
    const int threads_per_block = 256;
    const int num_blocks = std::min( AMGX_GRID_MAX_SIZE, (num_block_rows - 1) / threads_per_block + 1 );
    const int num_blocks_V2 = std::min( AMGX_GRID_MAX_SIZE, (num_nonzero_blocks - 1) / threads_per_block + 1);
    int numUnassigned = num_block_rows;
    int numUnassigned_previous = numUnassigned;
    Vector<TemplateConfig<AMGX_device, AMGX_vecFloat, t_matPrec, t_indPrec> > edge_weights(num_nonzero_blocks, -1);
    float *edge_weights_ptr = edge_weights.raw();
    float *rand_edge_weights_ptr = NULL;//(this->two_phase ? rand_edge_weights.raw() : NULL);
    // Compute the edge weights
    cudaFuncSetCacheConfig(computeEdgeWeightsBlockDiaCsr_V2<IndexType, ValueType, float>, cudaFuncCachePreferL1);
    computeEdgeWeightsBlockDiaCsr_V2 <<< num_blocks_V2, threads_per_block, 0, thrust::global_thread_handle::get_stream()>>>(A_row_offsets_ptr, A_row_indices_ptr, A_column_indices_ptr, A_dia_idx_ptr, A_nonzero_values_ptr, num_nonzero_blocks, edge_weights_ptr, rand_edge_weights_ptr, num_block_rows, A.get_block_dimy(), this->m_aggregation_edge_weight_component, this->weight_formula);
    cudaCheckError();
    cudaStream_t str = thrust::global_thread_handle::get_stream();
#ifdef EXPERIMENTAL_ITERATIVE_MATCHING
    AsyncEvent *throttle_event = new AsyncEvent;
    throttle_event->create();
    // TODO: pinned host memory
    typename Matrix_h::IVector h_unagg_vec(1);
    typename Matrix_d::IVector d_unagg_vec(1);
    int *unaggregated = h_unagg_vec.raw();
    int *d_unaggregated = d_unagg_vec.raw();
#endif
    int icount, s = 1;
    {
        icount = 0;
        float *weights_ptr = edge_weights_ptr;

        do
        {
            if ( !this->two_phase )
            {
                // 1-phase handshaking
                findStrongestNeighbourBlockDiaCsr_V2 <<< num_blocks, threads_per_block, 0, str>>>(A_row_offsets_ptr, A_column_indices_ptr, weights_ptr, num_block_rows, aggregates_ptr, strongest_neighbour_ptr, strongest_neighbour_ptr, A.get_block_dimy(), 1, this->merge_singletons);
                cudaCheckError();
            }
            else
            {
                // 2-phase handshaking
                findStrongestNeighbourBlockDiaCsr_V2 <<< num_blocks, threads_per_block, 0, str>>>(A_row_offsets_ptr, A_column_indices_ptr, weights_ptr, num_block_rows, aggregates_ptr, strongest_neighbour_1phase_ptr, strongest_neighbour_ptr, A.get_block_dimy(), 1, this->merge_singletons);
                cudaCheckError();
                // 2nd phase: for each block_row, find the strongest neighbour among those who gave hand on 1st phase
                findStrongestNeighbourBlockDiaCsr_V2 <<< num_blocks, threads_per_block, 0, str>>>(A_row_offsets_ptr, A_column_indices_ptr, weights_ptr, num_block_rows, aggregates_ptr, strongest_neighbour_1phase_ptr, strongest_neighbour_ptr, A.get_block_dimy(), 2, this->merge_singletons);
                cudaCheckError();
            }

            // Look for perfect matches. Also, for nodes without unaggregated neighbours, merge with aggregate containing strongest neighbour
            matchEdges <<< num_blocks, threads_per_block, 0, str>>>(num_block_rows, aggregates_ptr, strongest_neighbour_ptr);
            cudaCheckError();
#ifdef EXPERIMENTAL_ITERATIVE_MATCHING
            s = (icount & 1);

            if ( s == 0 )
            {
                // count unaggregated vertices
                cudaMemsetAsync(d_unaggregated, 0, sizeof(int), str);
                countAggregates<IndexType, threads_per_block> <<< num_blocks, threads_per_block, 0, str>>>(num_block_rows, aggregates_ptr, d_unaggregated);
                cudaCheckError();
                cudaMemcpyAsync(unaggregated, d_unaggregated, sizeof(int), cudaMemcpyDeviceToHost, str);
                throttle_event->record(str);
            }
            else
            {
                throttle_event->sync();
                numUnassigned_previous = numUnassigned;
                numUnassigned = *unaggregated;
            }

#else
            cudaStreamSynchronize(str);
            numUnassigned_previous = numUnassigned;
            numUnassigned = (int)thrust::count(aggregates.begin(), aggregates.begin() + num_block_rows, -1);
            cudaCheckError();
#endif
            icount++;
        }
        while ( (s == 0) || !(numUnassigned == 0 || icount > this->max_iterations || 1.0 * numUnassigned / num_block_rows < this->numUnassigned_tol || numUnassigned == numUnassigned_previous));
    }
#ifdef EXPERIMENTAL_ITERATIVE_MATCHING
    delete throttle_event;
#endif

    if ( this->merge_singletons )
    {
        // Merge remaining vertices with current aggregates
        if (!this->deterministic)
        {
            while (numUnassigned != 0)
            {
                mergeWithExistingAggregatesBlockDiaCsr_V2 <<< num_blocks, threads_per_block, 0, str>>>(A_row_offsets_ptr, A_column_indices_ptr, edge_weights_ptr, num_block_rows, aggregates_ptr, A.get_block_dimy(), this->deterministic, (IndexType *) NULL);
                cudaCheckError();
                numUnassigned = (int)thrust::count(aggregates.begin(), aggregates.begin() + num_block_rows, -1);
                cudaCheckError();
            }
        }
        else
        {
            typename Matrix_d::IVector aggregates_candidate(num_block_rows, -1);

            while (numUnassigned != 0)
            {
                mergeWithExistingAggregatesBlockDiaCsr_V2 <<< num_blocks, threads_per_block, 0, str>>>(A_row_offsets_ptr, A_column_indices_ptr, edge_weights_ptr, num_block_rows, aggregates_ptr, A.get_block_dimy(), this->deterministic, aggregates_candidate.raw());
                cudaCheckError();
                joinExistingAggregates <<< num_blocks, threads_per_block, 0, str>>>(num_block_rows, aggregates_ptr, aggregates_candidate.raw());
                cudaCheckError();
                numUnassigned = (int)thrust::count(aggregates.begin(), aggregates.begin() + num_block_rows, -1);
                cudaCheckError();
            }

            aggregates_candidate.resize(0);
        }
    }
    else
    {
        //make singletons
        aggregateSingletons <<< num_blocks, threads_per_block, 0, str>>>( aggregates_ptr, num_block_rows );
        cudaCheckError();
    }

    this->renumberAndCountAggregates(aggregates, aggregates_global, num_block_rows, num_aggregates);
}

template<class T_Config>
void Size2SelectorBase<T_Config>::setAggregates(Matrix<T_Config> &A,
        IVector &aggregates, IVector &aggregates_global, int &num_aggregates)
{
    if (A.get_block_dimx() == A.get_block_dimy())
    {
        setAggregates_common_sqblocks( A, aggregates, aggregates_global, num_aggregates );
    }
    else
    {
        FatalError("Unsupported block size for Size2", AMGX_ERR_NOT_SUPPORTED_BLOCKSIZE);
    }
}


// -------------------------
//  Explict instantiations
// -------------------------
#define AMGX_CASE_LINE(CASE) template class Size2SelectorBase<TemplateMode<CASE>::Type>;
AMGX_FORALL_BUILDS(AMGX_CASE_LINE)
#undef AMGX_CASE_LINE
#define AMGX_CASE_LINE(CASE) template class Size2Selector<TemplateMode<CASE>::Type>;
AMGX_FORALL_BUILDS(AMGX_CASE_LINE)
#undef AMGX_CASE_LINE

}
}
}
