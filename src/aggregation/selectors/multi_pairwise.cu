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

#include <aggregation/selectors/multi_pairwise.h>
#include <cutil.h>
#include <util.h>
#include <types.h>
#include <basic_types.h>
#include <texture.h>
#include <matrix_analysis.h>
#include <transpose.h>

#include <async_event.h>

#include <thrust/count.h> //count
#include <thrust/sort.h> //sort
#include <thrust/binary_search.h> //lower_bound
#include <thrust/unique.h> //unique
#include <thrust/host_vector.h>
#include <cusp/detail/format_utils.h> //offsets_to_indices
#include <determinism_checker.h>
#include <solvers/solver.h>

#include <aggregation/coarseAgenerators/thrust_coarse_A_generator.h>
#include <aggregation/coarseAgenerators/low_deg_coarse_A_generator.h>

#include <omp.h>

#define EXPERIMENTAL_ITERATIVE_MATCHING

namespace amgx
{
namespace aggregation
{
namespace multi_pairwise
{

// include common routines for all selectors
#include <aggregation/selectors/common_selector.h>

// ------------------------
//    Kernels
// ------------------------

__device__
float random_weight2(int i, int j)
{
#define RAND_MULTIPLIER                 1145637293
    unsigned long i_min = (min(i, j) * RAND_MULTIPLIER);
    unsigned long i_max = (max(i, j) * RAND_MULTIPLIER);
    return ((float)i_min / i_max);
}

__device__
unsigned long random_weight3(int i, int j)
{
    unsigned long a;
    a = (i + j) ^ 8;
    a = (a + 0x7ed55d16) + (a << 12);
    a = (a ^ 0xc761c23c) + (a >> 19);
    a = (a + 0x165667b1) + (a << 5);
    a = (a ^ 0xd3a2646c) + (a << 9);
    a = (a + 0xfd7046c5) + (a << 3);
    a = (a ^ 0xb55a4f09) + (a >> 16);
    return a;
}


// findStrongestNeighbour kernel for block_dia_csr_matrix format
// Reads the weight from edge_weights array
template <typename IndexType, typename MatrixValueType>
__global__
void findStrongestNeighbourBlockDiaCsr_V2(const IndexType *row_offsets, const IndexType *column_indices,
        MatrixValueType *edge_weights, IndexType num_block_rows, IndexType *aggregates,
        IndexType *strongest_neighbour_1phase, IndexType *strongest_neighbour,
        const size_t bsize, int phase, int merge_singletons)
{
    int tid = threadIdx.x + blockDim.x * blockIdx.x;
    MatrixValueType weight;
    int jcol;

    while (tid < num_block_rows)
    {
        int strongest_unaggregated = -1;
        int strongest_aggregated = -1;
        MatrixValueType     max_weight_unaggregated = 0.;
        MatrixValueType     max_weight_aggregated = 0.;

        if (aggregates[tid] == -1) // Unaggregated row
        {
            for (int j = row_offsets[tid]; j < row_offsets[tid + 1]; j++)
            {
                jcol = column_indices[j];

                if (phase == 1) { weight = edge_weights[j]; }
                else { weight = random_weight2(tid, jcol); }

                if (tid == jcol || jcol >= num_block_rows) { continue; }    // skip diagonal and halo

                if (phase == 2 && strongest_neighbour_1phase[jcol] != tid) { continue; } // if 2nd phase only accept those who gave a hand on the 1st phase

                // Identify strongest aggregated and unaggregated neighbours (method by multi_pairwise)
                if (aggregates[jcol] == -1 && weight > 0.0 && (weight > max_weight_unaggregated || (weight == max_weight_unaggregated && random_weight3(tid, jcol) > random_weight3(tid, strongest_unaggregated)))) // unaggregated
                {
                    max_weight_unaggregated = weight;
                    strongest_unaggregated = jcol;
                }
                else if (aggregates[jcol] != -1 && weight > 0.0 && (weight > max_weight_aggregated || (weight == max_weight_aggregated && random_weight3(tid, jcol) > random_weight3(tid, strongest_aggregated)))) // aggregated
                {
                    max_weight_aggregated = weight;
                    strongest_aggregated = jcol;
                }
            }

            if (strongest_unaggregated == -1 && strongest_aggregated != -1) // All neighbours are aggregated
            {
                if ( merge_singletons == 1 )
                    // Put in same aggregate as strongest neighbour
                {
                    aggregates[tid] = aggregates[strongest_aggregated];
                }
                else
                    // create singleton
                {
                    aggregates[tid] = tid;
                }
            }
            else if (strongest_unaggregated != -1)
            {
                if (phase == 2)
                {
                    MatrixValueType rand_w1 = random_weight2(tid, strongest_neighbour_1phase[tid]);
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


template <typename IndexType>
__device__
bool atomicJoin( IndexType node, IndexType aggregate, IndexType *aggregates, int *sizes, int allowed )
{
    int mySize = sizes[node];
    int theirSize = sizes[aggregate];
    int theirSizeOld = theirSize;

    do
    {
        int newSize = mySize + theirSize;

        if ( newSize > allowed )
        {
            return false;
        }

        theirSizeOld = theirSize;
        theirSize = atomicCAS( &sizes[aggregate], theirSize, newSize );
    }
    while ( theirSize != theirSizeOld );

    aggregates[node] = aggregate;
    return true;
}

template <typename IndexType, typename MatrixValueType, bool use_degree>
__global__
void findStrongestNeighbourBlockDiaCsr_V3(const IndexType *row_offsets,
        const IndexType *column_indices,
        MatrixValueType *edge_weights,
        IndexType num_block_rows,
        IndexType *aggregates,
        IndexType *strongest_neighbour,
        int *sizes,
        int *degree,
        const size_t bsize,
        int max_aggregate_size,
        int merge_singletons)
{
    int tid = threadIdx.x + blockDim.x * blockIdx.x;
    MatrixValueType weight;
    int jcol;

    while (tid < num_block_rows)
    {
        int strongest_unaggregated = -1;
        int strongest_aggregated = -1;
        int lowest_degree;

        if ( use_degree )
        {
            lowest_degree = degree[tid];    //only interested in finding lower degree than self
        }
        else
        {
            lowest_degree = 0;    //if we decide to not use degree than just propose to the strongest edge
        }

        int lowest_degree_neighbor = tid;
        MatrixValueType     lowest_degree_weight = 1e100; //high start value, so that same degree neighbor won't become lowest degree neighbor
        MatrixValueType     max_weight_unaggregated = 0.;
        MatrixValueType     max_weight_aggregated = 0.;
        int mySize;

        if ( merge_singletons == 2 )
        {
            mySize = sizes[tid];
        }
        else
        {
            mySize = 0;
        }

        if ( merge_singletons != 2 )
        {
            max_aggregate_size = 100000;
        }

        //this aggregate is already full
        if (mySize == max_aggregate_size)
        {
            aggregates[tid] = tid;
        }

        if (aggregates[tid] == -1) // Unaggregated row
        {
            for (int j = row_offsets[tid]; j < row_offsets[tid + 1]; j++)
            {
                jcol = column_indices[j];

                if (tid == jcol || jcol >= num_block_rows) { continue; }    // skip diagonal and halo

                weight = edge_weights[j];

                if (weight <= 0.0) { continue; }

                if ( aggregates[jcol] != -1 ) //aggregated neighbor
                {
                    int theirSize;

                    if ( merge_singletons == 2 )
                    {
                        theirSize = aggregates[sizes[jcol]];
                    }
                    else
                    {
                        theirSize = 0;
                    }

                    //if all neighbors are aggregated, find the strongest edge to neighbor aggregate that is not full yet
                    if (mySize + theirSize <= max_aggregate_size &&
                            (weight > max_weight_aggregated)) // aggregated
                    {
                        max_weight_aggregated = weight;
                        strongest_aggregated = jcol;
                    }
                }
                else //unaggregated neighbor
                {
                    if ( use_degree && merge_singletons == 2 )
                    {
                        int theirSize = sizes[jcol];
                        //get lowest degree neighbor or find out that there is no lower degree neighbor
                        int current_degree = degree[jcol];

                        if ( mySize + theirSize <= max_aggregate_size && (current_degree < lowest_degree || current_degree == lowest_degree && weight > lowest_degree_weight) )
                        {
                            lowest_degree = current_degree;
                            lowest_degree_weight = weight;
                            lowest_degree_neighbor = jcol;
                        }

                        //get highest weight neighbor
                        if ( mySize + theirSize <= max_aggregate_size && (weight > max_weight_unaggregated) )
                        {
                            strongest_unaggregated = jcol;
                            max_weight_unaggregated = weight;
                        }
                    }

                    if ( use_degree && merge_singletons != 2 ) //same as above but ignore sizes
                    {
                        //get lowest degree neighbor or find out that there is no lower degree neighbor
                        int current_degree = degree[jcol];

                        if ( current_degree < lowest_degree || current_degree == lowest_degree && weight > lowest_degree_weight)
                        {
                            lowest_degree = current_degree;
                            lowest_degree_weight = weight;
                            lowest_degree_neighbor = jcol;
                        }

                        //get highest weight neighbor
                        if (weight > max_weight_unaggregated)
                        {
                            strongest_unaggregated = jcol;
                            max_weight_unaggregated = weight;
                        }
                    }

                    if ( !use_degree && merge_singletons == 2 )
                    {
                        //get highest weight neighbor only but pay attention to the aggregate sizes
                        int theirSize = sizes[jcol];                        //get highest weight neighbor

                        if ( mySize + theirSize <= max_aggregate_size && (weight > max_weight_unaggregated) )
                        {
                            strongest_unaggregated = jcol;
                            max_weight_unaggregated = weight;
                        }
                    }

                    if ( !use_degree && merge_singletons != 2 )
                    {
                        //just highest weight
                        if (weight > max_weight_unaggregated)
                        {
                            strongest_unaggregated = jcol;
                            max_weight_unaggregated = weight;
                        }
                    }
                }
            }

            //prefer lowest degree neighbor
            if ( lowest_degree_neighbor != tid )
            {
                strongest_unaggregated = lowest_degree_neighbor;
            }

            if (strongest_unaggregated != -1) //Unaggregated neighbor exists
            {
                strongest_neighbour[tid] = strongest_unaggregated;    //assign strongest aggregated
            }

            if (strongest_unaggregated == -1 && strongest_aggregated != -1) // All neighbours are aggregated but small enough aggregated neighbors exist
            {
                if ( merge_singletons == 0 )
                {
                    aggregates[tid] = tid;
                }

                if ( merge_singletons == 1 )
                {
                    aggregates[tid] = aggregates[strongest_aggregated];
                }

                if ( merge_singletons == 2)
                {
                    atomicJoin( tid, aggregates[strongest_aggregated], aggregates, sizes, max_aggregate_size );    //try to join, can fail. maybe it works next round.
                }
            }

            if (strongest_unaggregated == -1 && strongest_aggregated == -1) //no feasable neighbor at all, become singleton
            {
                strongest_neighbour[tid] = tid;    //become singleton
            }
        }

        tid += gridDim.x * blockDim.x;
    }
}

template <typename IndexType, typename ValueType>
__global__
void computeDegree( const IndexType *ia, const IndexType *ja, const ValueType *weights, IndexType *aggregates, IndexType *sizes, IndexType *degree, IndexType numRows, IndexType max_aggregate_size)
{
    int i = threadIdx.x + blockDim.x * blockIdx.x;

    while ( i < numRows )
    {
        int myDegree = 0;
        IndexType ia_ip1 = ia[i + 1];

        for ( IndexType ii = ia[i]; ii < ia_ip1; ii++ )
        {
            IndexType j = ja[ii];

            if ( j == i )
            {
                continue;
            }

            int mySize, theirSize;

            if ( sizes != NULL )
            {
                mySize = sizes[i];
                theirSize = sizes[j];
            }
            else
            {
                mySize = theirSize = 0;
            }

            if ( weights[ii] > 0.0 && aggregates[j] == -1 && mySize + theirSize <= max_aggregate_size )
            {
                myDegree++;
            }
        }

        degree[i] = myDegree;
        i += gridDim.x * blockDim.x;
    }
}


template <typename IndexType, typename ValueType>
__global__
void mergeSingletonsSmart(const IndexType *ia, const IndexType *ja, const ValueType *weights, IndexType *aggregates, IndexType *sizes, IndexType numRows, int max_aggregate_size)
{
    int tid = threadIdx.x + blockDim.x * blockIdx.x;

    while ( tid < numRows )
    {
        //unaggregated nodes try to join or create their own aggregate
        if ( aggregates[tid] == -1 )
        {
            bool joined = false;

            while ( !joined )
            {
                int neighbor_aggregate = -1;
                ValueType max_weight = 0.0;
                IndexType mySize = sizes[tid];

                for (IndexType ii = ia[tid]; ii < ia[tid + 1]; ii++)
                {
                    IndexType j = ja[ii];

                    if (j == tid || j >= numRows) { continue; }

                    if ( aggregates[j] != -1 && sizes[aggregates[j]] + mySize <= max_aggregate_size && weights[ii] > max_weight )
                    {
                        neighbor_aggregate = aggregates[j];
                        max_weight = weights[ii];
                    }
                }

                //no possible neighbor found
                if ( neighbor_aggregate == -1 )
                {
                    //create own aggregate
                    aggregates[tid] = tid;
                    joined = true;
                }
                else
                {
                    //try to join
                    joined = atomicJoin( tid, neighbor_aggregate, aggregates, sizes, max_aggregate_size );
                }
            }
        }

        tid += gridDim.x * blockDim.x;
    }
}


template <typename IndexType>
__global__
void updateAggregateSizes( IndexType *sizesSource, IndexType *sizes, IndexType *aggregates, IndexType numRows )
{
    int tid = threadIdx.x + blockDim.x * blockIdx.x;

    while ( tid < numRows )
    {
        IndexType agg = aggregates[tid];
        IndexType aggregateSize = sizes[agg];
        IndexType mySize = sizesSource[tid];

        while ( mySize > aggregateSize )
        {
            aggregateSize = atomicCAS( &sizes[agg], aggregateSize, mySize );
        }

        tid += gridDim.x * blockDim.x;
    }
}


// Kernel that checks if perfect matchs exist
template <typename IndexType>
__global__
void matchEdges(const IndexType num_rows, IndexType *aggregates, int *strongest_neighbour, IndexType *sizes)
{
    int tid = threadIdx.x + blockDim.x * blockIdx.x;
    int potential_match, potential_match_neighbour;

    while (tid < num_rows)
    {
        if (aggregates[tid] == -1) // Unaggregated row
        {
            potential_match = strongest_neighbour[tid];
            potential_match_neighbour = strongest_neighbour[potential_match];

            if ( potential_match == tid )
            {
                aggregates[tid] = tid;
            }
            else if (potential_match != -1 && potential_match_neighbour == tid && tid < potential_match) // we have a match
            {
                aggregates[tid] = tid;
                aggregates[potential_match] = tid;

                if ( sizes != NULL)
                {
                    sizes[tid] += sizes[potential_match];
                }
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

    __shared__ volatile int smem[block_size];
    smem[threadIdx.x] = c;
    __syncthreads();

    for ( int off = blockDim.x / 2; off >= 32; off = off / 2 )
    {
        if ( threadIdx.x < off )
        {
            smem[threadIdx.x] += smem[threadIdx.x + off];
        }

        __syncthreads();
    }

    // warp reduce
    if ( threadIdx.x < 32 )
    {
        smem[threadIdx.x] += smem[threadIdx.x + 16];
        smem[threadIdx.x] += smem[threadIdx.x + 8];
        smem[threadIdx.x] += smem[threadIdx.x + 4];
        smem[threadIdx.x] += smem[threadIdx.x + 2];
        smem[threadIdx.x] += smem[threadIdx.x + 1];
    }

    if ( threadIdx.x == 0 )
    {
        atomicAdd(num_unaggregated, smem[0]);
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
// Weights are read from edge_weights array
// For block_dia_csr_matrix_format
template <typename IndexType, typename MatrixValueType>
__global__
void mergeWithExistingAggregatesBlockDiaCsr_V2(const IndexType *row_offsets, const IndexType *column_indices, const MatrixValueType *edge_weights,
        const int num_block_rows, IndexType *aggregates, int bsize, const int deterministic, IndexType *aggregates_candidate)
{
    int tid = threadIdx.x + blockDim.x * blockIdx.x;
    int jcol;
    MatrixValueType weight;

    while (tid < num_block_rows)
    {
        MatrixValueType max_weight_aggregated = 0.;
        int strongest_aggregated = -1;

        if (aggregates[tid] == -1) // Unaggregated row
        {
            for (int j = row_offsets[tid]; j < row_offsets[tid + 1]; j++)
            {
                // Compute edge weight
                weight = edge_weights[j];
                jcol = column_indices[j];

                if (jcol == tid || jcol >= num_block_rows) { continue; }    // skip diagonal

                if ( aggregates[jcol] == num_block_rows ) { continue; } // skip dd rows

                // Identify strongest aggregated neighbour
                if (aggregates[jcol] != -1 && weight > 0 && (weight > max_weight_aggregated || (weight == max_weight_aggregated && random_weight3( tid, jcol ) > random_weight3( tid, strongest_aggregated )))) //
                {
                    max_weight_aggregated = weight;
                    strongest_aggregated = jcol;
                }
            }

            if (strongest_aggregated != -1) // Found a neighbour to aggregate to
            {
                if (deterministic == 1)
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
                if (deterministic == 1)
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

// filter edge weights like this:
// set w_ij = 0 iff
// w_ij < alpha * sqrt( max_k{w_ik} * max_l{w_jl} )
// alpha is some constant, 0.25 or 0.5 should work fine
template<typename IndexType, typename ValueType>
__global__
void filterWeights( const IndexType *row_offsets, const IndexType *row_indices, const IndexType *col_indices, const IndexType *diag, const ValueType *old_weights, ValueType *new_weights, IndexType num_nonzero_blocks, IndexType num_owned, ValueType alpha )
{
    int tid = threadIdx.x + blockDim.x * blockIdx.x;
    int i, j, kmin, kmax;
    ValueType max_ik, max_jl;

    while ( tid < num_nonzero_blocks )
    {
        i = row_indices[tid];
        j = col_indices[tid];

        if ( i != j && j < num_owned )
        {
            //find max_k{w_ik}
            kmin = row_offsets[i];
            kmax = row_offsets[i + 1];
            max_ik = 0.0;

            for (int k = kmin; k < kmax; k++)
            {
                if ( col_indices[k] != i && old_weights[k] > max_ik )
                {
                    max_ik = old_weights[k];
                }
            }

            //find max_l{w_jl}
            kmin = row_offsets[j];
            kmax = row_offsets[j + 1];
            max_jl = 0.0;

            for (int l = kmin; l < kmax; l++)
            {
                if ( col_indices[l] != j && old_weights[l] > max_jl )
                {
                    max_jl = old_weights[l];
                }
            }

            //test squared inequality
            if ( old_weights[tid] * old_weights[tid] < alpha * alpha * max_ik * max_jl )
            {
                new_weights[tid] = 0.0;
            }
            else //rescale to relative importance. this should also increase the chance of a handshake
            {
                new_weights[tid] = old_weights[tid];
            }

            //              new_weights[tid] = old_weights[tid] / sqrt(max_ik*max_jl);
        }

        tid += gridDim.x * blockDim.x;
    }
}


template<typename IndexType, typename ValueType>
__global__
void gatherValuesInterleaved( const ValueType *inValues, ValueType *outValues, IndexType nnz, int sq_blocksize, int index_offset )
{
    int tid = threadIdx.x + blockDim.x * blockIdx.x;

    while ( tid < nnz )
    {
        //at least the write is coalesced
        outValues[tid] = inValues[tid * sq_blocksize + index_offset];
        tid += gridDim.x * blockDim.x;
    }
}


template<typename IndexType, typename ValueTypeV, typename ValueTypeM>
__global__
void addToWeights( ValueTypeM *edge_weights, const ValueTypeV *x, const IndexType *row_indices, IndexType *col_indices, IndexType nnz, double scale )
{
    int tid = threadIdx.x + blockDim.x * blockIdx.x;

    while ( tid < nnz )
    {
        int i = row_indices[tid];
        int j = col_indices[tid];
        edge_weights[tid] -= static_cast<ValueTypeM>( scale * fabs( x[i] - x[j] ) );
        tid += gridDim.x * blockDim.x;
    }
}

template <typename ValueType, typename IndexType>
__global__
void rescaleVector( ValueType *x, IndexType numRows )
{
    int tid = threadIdx.x + blockDim.x * blockIdx.x;

    while ( tid < numRows )
    {
        x[tid] = 2 * x[tid] - 1;
        tid += gridDim.x * blockDim.x;
    }
}

// -----------------
//    Methods
// ----------------

// Constructor
template<class T_Config>
MultiPairwiseSelectorBase<T_Config>::MultiPairwiseSelectorBase(AMG_Config &cfg, const std::string &cfg_scope)
{
    deterministic = cfg.AMG_Config::template getParameter<IndexType>("determinism_flag", "default");
    max_iterations = cfg.AMG_Config::template getParameter<IndexType>("max_matching_iterations", cfg_scope);
    numUnassigned_tol = cfg.AMG_Config::template getParameter<double>("max_unassigned_percentage", cfg_scope);
    two_phase = cfg.AMG_Config::template getParameter<int>("handshaking_phases", cfg_scope) == 2;
    m_aggregation_edge_weight_component = cfg.AMG_Config::template getParameter<int>("aggregation_edge_weight_component", cfg_scope);
    aggregation_passes = cfg.AMG_Config::template getParameter<int>("aggregation_passes", cfg_scope); //default to size 8 aggregates. maybe its more convenient to have that as a config parameter
    filter_weights = cfg.AMG_Config::template getParameter<int>("filter_weights", cfg_scope); //by default: no filtering
    filter_weights_alpha = cfg.AMG_Config::template getParameter<double>( "filter_weights_alpha", cfg_scope ); //default to 0.25
    full_ghost_level = cfg.AMG_Config::template getParameter<int>( "full_ghost_level", cfg_scope ); //defaults to 0
    notay_weights = cfg.AMG_Config::template getParameter<int>( "notay_weights", cfg_scope ); //defaults to 0
    ghost_offdiag_limit = cfg.AMG_Config::template getParameter<int>( "ghost_offdiag_limit", cfg_scope ); //defaults to 0
    merge_singletons = cfg.AMG_Config::template getParameter<int>( "merge_singletons", cfg_scope ); //defaults to 1
    weight_formula = cfg.AMG_Config::template getParameter<int>( "weight_formula", cfg_scope ); //wheight formula defaults to 0
    serial_matching = cfg.AMG_Config::template getParameter<int>( "serial_matching", cfg_scope ) != 0; //will use a serial matching algorithm instead of handshake
    modified_handshake = cfg.AMG_Config::template getParameter<int>("modified_handshake", cfg_scope ) == 1;
    //passes = 1 -> max = 3
    //passes = 2 -> max = 5
    //passes = 3 -> max = 10
    //passes = 4 -> max = 18
    max_aggregate_size = 2;

    for (int i = 1; i < aggregation_passes; i ++)
    {
        max_aggregate_size *= 2;
    }

    max_aggregate_size += aggregation_passes - (aggregation_passes / 2);
    mCfg = cfg;
    mCfg_scope = cfg_scope;
}

// setAggregates for block_dia_csr_matrix_h format
template <AMGX_VecPrecision t_vecPrec, AMGX_MatPrecision t_matPrec, AMGX_IndPrecision t_indPrec>
void MultiPairwiseSelector<TemplateConfig<AMGX_host, t_vecPrec, t_matPrec, t_indPrec> >::setAggregates_common_sqblocks(Matrix_h &A,
        typename Matrix_h::IVector &aggregates, typename Matrix_h::IVector &aggregates_global, int &num_aggregates, MVector &edge_weights, IVector &sizes)
{
    FatalError("MultiPairwise selector: setAggregates not implemented on CPU, exiting", AMGX_ERR_NOT_SUPPORTED_TARGET);
}

// device specialization

//edge_weights is an in/out parameter:
//if its size is zero, the edge_weights will be computed from A and stored into edge_weights
//else the edge_weights will not be computed and assumed to be valid for the given A. the value array of A is not used in this case
template <AMGX_VecPrecision t_vecPrec, AMGX_MatPrecision t_matPrec, AMGX_IndPrecision t_indPrec>
void MultiPairwiseSelector<TemplateConfig<AMGX_device, t_vecPrec, t_matPrec, t_indPrec> >::setAggregates_common_sqblocks(Matrix_d &A,
        typename Matrix_d::IVector &aggregates,
        typename Matrix_d::IVector &aggregates_global,
        int &num_aggregates,
        MVector &edge_weights,
        IVector &sizes)
{
    IndexType num_block_rows = (int) A.get_num_rows();
    IndexType num_nonzero_blocks = (int) A.get_num_nz();
    // both ways are supported
    IndexType total_nz = (A.is_matrix_singleGPU()) ? num_nonzero_blocks : A.manager->num_nz_all();
    typename Matrix_d::IVector &row_indices = A.row_indices;
    row_indices.resize( total_nz);
    cusp::detail::offsets_to_indices(A.row_offsets, row_indices);
    IndexType total_rows = (A.is_matrix_singleGPU()) ? A.get_num_rows() : A.manager->num_rows_all();
    aggregates.resize(total_rows);
    thrust_wrapper::fill<AMGX_device>(aggregates.begin(), aggregates.end(), -1);
    cudaCheckError();

    if ( this->merge_singletons == 2 && sizes.size() == 0 )
    {
        sizes.resize( total_rows, 1 );    //init with all ones
    }

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
    int numUnassigned = num_block_rows;
    int numUnassigned_previous = numUnassigned;
    bool computeWeights = ( edge_weights.size() == 0 );

    if (computeWeights)
    {
        if ( A.hasProps( DIAG ) )
        {
            edge_weights.resize( num_nonzero_blocks + num_block_rows, 0.0 );
        }
        else
        {
            edge_weights.resize( num_nonzero_blocks + 1, -1 );    //+1 is important to some algorithms
        }
    }

    ValueType *edge_weights_ptr = edge_weights.raw();
    ValueType *rand_edge_weights_ptr = NULL;
    cudaStream_t str = amgx::thrust::global_thread_handle::get_stream();

    // Compute the edge weights
    if ( computeWeights )
    {
        const int num_blocks_V2 = std::min( AMGX_GRID_MAX_SIZE, (num_nonzero_blocks - 1) / threads_per_block + 1);
        //compute with std formula
        cudaFuncSetCacheConfig(computeEdgeWeightsBlockDiaCsr_V2<IndexType, ValueType, ValueType>, cudaFuncCachePreferL1);
        computeEdgeWeightsBlockDiaCsr_V2 <<< num_blocks_V2, threads_per_block, 0, str>>>(A_row_offsets_ptr,
                A_row_indices_ptr,
                A_column_indices_ptr,
                A_dia_idx_ptr,
                A_nonzero_values_ptr,
                num_nonzero_blocks,
                edge_weights_ptr,
                rand_edge_weights_ptr,
                num_block_rows,
                A.get_block_dimy(),
                this->m_aggregation_edge_weight_component,
                this->weight_formula);
        cudaCheckError();
    }

    //filter weights if desired
    if ( this->filter_weights == 1 )
    {
        MVector tmp( edge_weights.size() );
        const int num_blocks_filter = std::min( AMGX_GRID_MAX_SIZE, (num_nonzero_blocks - 1) / threads_per_block + 1);
        cudaStreamSynchronize(str);
        cudaCheckError();
        filterWeights <<< num_blocks_filter, threads_per_block, 0, str>>>( A_row_offsets_ptr,
                A_row_indices_ptr,
                A_column_indices_ptr,
                A_dia_idx_ptr,
                edge_weights_ptr,
                tmp.raw(),
                num_nonzero_blocks,
                num_block_rows,
                this->filter_weights_alpha);
        cudaStreamSynchronize(str);
        cudaCheckError();
        tmp.swap( edge_weights );
        edge_weights_ptr = edge_weights.raw();
    }

// compute matching
    if ( !this->serial_matching )
    {
        IVector degree;

        if ( this->modified_handshake )
        {
            degree.resize( num_block_rows );
        }

#ifdef EXPERIMENTAL_ITERATIVE_MATCHING
        // TODO: allocate host pinned memory
        AsyncEvent *throttle_event = new AsyncEvent;
        throttle_event->create();
        typename Matrix_h::IVector h_unagg_vec(1);
        typename Matrix_d::IVector d_unagg_vec(1);
        int *unaggregated = h_unagg_vec.raw();
        int *d_unaggregated = d_unagg_vec.raw();
#endif
        int icount, s = 1;
        {
            icount = 0;
            ValueType *weights_ptr = edge_weights_ptr;

            do
            {
                if ( !this->two_phase )
                {
                    if ( this->modified_handshake )
                        computeDegree <<< num_blocks, threads_per_block, 0, str>>>(A_row_offsets_ptr,
                                A_column_indices_ptr,
                                weights_ptr,
                                aggregates_ptr,
                                sizes.raw(),
                                degree.raw(),
                                num_block_rows,
                                this->max_aggregate_size );

                    // 1-phase handshaking
                    if ( this->modified_handshake )
                        findStrongestNeighbourBlockDiaCsr_V3<IndexType, ValueType, true>
                        <<< num_blocks, threads_per_block, 0, str>>>(A_row_offsets_ptr,
                                A_column_indices_ptr,
                                weights_ptr,
                                num_block_rows,
                                aggregates_ptr,
                                strongest_neighbour_ptr,
                                sizes.raw(),
                                degree.raw(),
                                A.get_block_dimy(),
                                this->max_aggregate_size,
                                this->merge_singletons);
                    else
                        findStrongestNeighbourBlockDiaCsr_V3<IndexType, ValueType, false>
                        <<< num_blocks, threads_per_block, 0, str>>>(A_row_offsets_ptr,
                                A_column_indices_ptr,
                                weights_ptr,
                                num_block_rows,
                                aggregates_ptr,
                                strongest_neighbour_ptr,
                                sizes.raw(),
                                degree.raw(),
                                A.get_block_dimy(),
                                this->max_aggregate_size,
                                this->merge_singletons);

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
                if ( this->merge_singletons == 2 )
                {
                    matchEdges <<< num_blocks, threads_per_block, 0, str>>>(num_block_rows, aggregates_ptr, strongest_neighbour_ptr, sizes.raw());
                }
                else
                {
                    matchEdges <<< num_blocks, threads_per_block, 0, str>>>(num_block_rows, aggregates_ptr, strongest_neighbour_ptr, (int *)NULL);
                }

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
                numUnassigned = (int)amgx::thrust::count(aggregates.begin(), aggregates.begin() + num_block_rows, -1);
                cudaCheckError();
#endif
                icount++;
            }
            while ( (s == 0) || !(numUnassigned == 0 || icount > this->max_iterations || 1.0 * numUnassigned / num_block_rows < this->numUnassigned_tol || numUnassigned == numUnassigned_previous));

        }
        //    printf("%i,\n", icount);
#ifdef EXPERIMENTAL_ITERATIVE_MATCHING
        delete throttle_event;
#endif
    }
    else
    {
        computeMatchingSerialGreedy( A, aggregates, num_aggregates, edge_weights );
    }

    if ( this->merge_singletons == 1 )
    {
        // Merge remaining vertices with current aggregates
        if (this->deterministic != 1)
        {
            while (numUnassigned != 0)
            {
                mergeWithExistingAggregatesBlockDiaCsr_V2 <<< num_blocks, threads_per_block, 0, str>>>(A_row_offsets_ptr, A_column_indices_ptr, edge_weights_ptr, num_block_rows, aggregates_ptr, A.get_block_dimy(), this->deterministic, (IndexType *) NULL);
                cudaCheckError();
                numUnassigned = (int)amgx::thrust::count(aggregates.begin(), aggregates.begin() + num_block_rows, -1);
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
                numUnassigned = (int)amgx::thrust::count(aggregates.begin(), aggregates.begin() + num_block_rows, -1);
                cudaCheckError();
            }

            aggregates_candidate.resize(0);
        }
    }
    else if (this->merge_singletons == 0 )
    {
        //make singletons
        aggregateSingletons <<< num_blocks, threads_per_block, 0, str>>>( aggregates_ptr, num_block_rows );
        cudaCheckError();
    }
    else if ( this->merge_singletons == 2 )
    {
        //merges all remaining singletons into adequate neighbors if possible
        mergeSingletonsSmart <<< num_blocks, threads_per_block, 0, str>>>(A_row_offsets_ptr,
                A_column_indices_ptr,
                edge_weights_ptr,
                aggregates_ptr,
                sizes.raw(),
                num_block_rows,
                this->max_aggregate_size);
        cudaCheckError();
    }

    //This will assign num_aggregates to the pseudo aggregate without counting it. Perfect!
    this->renumberAndCountAggregates(aggregates, aggregates_global, num_block_rows, num_aggregates);

    if ( this->merge_singletons == 2 )
    {
        //udpate the sizes vector, so it matches the renumbered aggregates size
        IVector sizesSource;
        sizesSource.swap( sizes );
        sizes.resize( num_aggregates, 1 );
        updateAggregateSizes <<< num_blocks, threads_per_block, 0, str>>>( sizesSource.raw(), sizes.raw(), aggregates_ptr, num_block_rows );
        cudaCheckError();
    }

}


//instead of a handshake, we use a serial greedy algorithm to compute a better matching
//the algorithm:
// 1. compute degree of every node and sort nodes by degree into double linked list
// 2. while non-isolated nodes left:
//        take node with minimum degree > 0
//        find strongest edge to unaggregated node and assign to new aggregate
//        remove both nodes from linked list
//        decrease degree of each neighbor by one for each of the two nodes
//        update list
template <AMGX_VecPrecision t_vecPrec, AMGX_MatPrecision t_matPrec, AMGX_IndPrecision t_indPrec>
void MultiPairwiseSelector<TemplateConfig<AMGX_device, t_vecPrec, t_matPrec, t_indPrec> >::computeMatchingSerialGreedy( const Matrix_d &A, IVector &aggregates, int &numAggregates, MVector &edge_weights)
{
    IndexType numRows = A.row_offsets.size() - 1;
    IndexType nnz = A.col_indices.size();
    //allocate memory on host
    IndexType *ia = new IndexType[numRows + 1];
    IndexType *ja = new IndexType[nnz];
    ValueType *w = new ValueType[nnz];
    IndexType *agg = new IndexType[numRows];
    IndexType *deg = new IndexType[numRows];
    //copy
    cudaMemcpy( ia, A.row_offsets.raw(), sizeof(IndexType) * (numRows + 1), cudaMemcpyDeviceToHost );
    cudaMemcpy( ja, A.col_indices.raw(), sizeof(IndexType)*nnz, cudaMemcpyDeviceToHost );
    cudaMemcpy( w, edge_weights.raw(), sizeof(ValueType)*nnz, cudaMemcpyDeviceToHost );
    //init agg and compute the degree of each aggregate
    int max_degree = 0;

    for (IndexType i = 0; i < numRows; i++)
    {
        agg[i] = -1;
        int degree = 0;

        for (IndexType ii = ia[i]; ii < ia[i + 1]; ii++) //only care for positive weights
        {
            if ( ja[ii] != i && w[ii] > 0.0 )
            {
                degree++;
            }
        }

        if ( degree > max_degree )
        {
            max_degree = degree;
        }

        deg[i] = degree;
    }

    if ( max_degree >= numRows )
    {
        FatalError( "max degree is greater than numRows.", AMGX_ERR_UNKNOWN );
    }

    //init double linked list
    IndexType *fwd = new IndexType[numRows + max_degree + 1];
    IndexType *bwd = new IndexType[numRows + max_degree + 1];

    for (IndexType i = 0; i < numRows + max_degree + 1; i++)
    {
        fwd[i] = i;
        bwd[i] = i;
    }

    IndexType nodesLeft = numRows;
    numAggregates = 0;

    //insert nodes into list
    for (IndexType i = numRows - 1; i >= 0; i--) //inserting in backward order the nodes will be sorted by index in case of same degree
    {
        //insert forward following root
        fwd[i] = fwd[numRows + deg[i]];
        fwd[numRows + deg[i]] = i;
        //insert backward
        bwd[i] = numRows + deg[i];
        bwd[fwd[i]] = i;

        //isolated nodes cannot be aggregated
        if ( deg[i] == 0 )
        {
            nodesLeft--;
        }
    }

    while ( nodesLeft > 0 )
    {
        IndexType node = numRows;
        int degree;

        for (degree = 1; degree <= max_degree; degree++)
        {
            //list not empty -> select node
            if ( fwd[numRows + degree] < numRows ) //selecting the first node will select the most recently inserted one or the one with lowest index. both is preferable
            {
                node = fwd[numRows + degree];
            }

            if ( node < numRows )
            {
                break;
            }
        }

        //no node with degree > 1 found even though nodesLeft > 0
        if ( node == numRows )
        {
            FatalError("nodeLeft counting or list invalid", AMGX_ERR_UNKNOWN );
        }

        if ( agg[node] != -1 )
        {
            FatalError("node is already aggregated", AMGX_ERR_UNKNOWN );
        }

        //find strongest edge
        ValueType max_weight = 0.0;
        IndexType max_node = numRows; //use this as gatekeeper, so if weight == 0 the node index will not be greater than this

        for (IndexType ii = ia[node]; ii < ia[node + 1]; ii++)
        {
            IndexType j = ja[ii];

            if ( agg[j] != -1 || j == node)
            {
                continue;
            }

            if ( w[ii] > 0.0 )
            {
                degree--;
            }

            //deterministic, doesn't selects 0 weight.
            if ( w[ii] > max_weight || (w[ii] == max_weight && j > max_node) ) //always taking the edge pointing to the max node can give good alignment if numbering is structured
            {
                max_node = j;
                max_weight = w[ii];
            }
        } //Note that there has to be at least one neighbor node because degree of node is at least 1.

        if ( max_node == numRows )
        {
            FatalError( "node  has no neighbor although degree of node is at least 1", AMGX_ERR_UNKNOWN );
        }

        if ( degree != 0 )
        {
            FatalError( "node degree corrupted", AMGX_ERR_UNKNOWN );
        }

        //aggregate
        agg[node] = node;
        agg[max_node] = node;
        numAggregates++;
        nodesLeft -= 2;
        //remove from list
        fwd[bwd[node]] = fwd[node];
        bwd[fwd[node]] = bwd[node];
        fwd[bwd[max_node]] = fwd[max_node];
        bwd[fwd[max_node]] = bwd[max_node];

        //update neighbors and list
        //max_node first
        for (IndexType ii = ia[max_node]; ii < ia[max_node + 1]; ii++)
        {
            IndexType j = ja[ii];

            if ( agg[j] != -1  || w[ii] <= 0.0)
            {
                continue;
            }

            //remove j from list
            fwd[bwd[j]] = fwd[j];
            bwd[fwd[j]] = bwd[j];
            //update degree of j
            deg[j]--;
            //add j back to start of the list
            fwd[j] = fwd[numRows + deg[j]];
            bwd[j] = numRows + deg[j];
            bwd[fwd[j]] = j;
            fwd[bwd[j]] = j;

            if (deg[j] == 0)
            {
                nodesLeft--;
            }
        }

        //node second, this will prefer nodes neighbors over max_nodes neighbors when choosing the next node
        for (IndexType ii = ia[node]; ii < ia[node + 1]; ii++)
        {
            IndexType j = ja[ii];

            if ( agg[j] != -1  || w[ii] <= 0.0)
            {
                continue;
            }

            //remove j from list
            fwd[bwd[j]] = fwd[j];
            bwd[fwd[j]] = bwd[j];
            //update degree of j
            deg[j]--;
            //add j back to start of the list
            fwd[j] = fwd[numRows + deg[j]];
            bwd[j] = numRows + deg[j];
            bwd[fwd[j]] = j;
            fwd[bwd[j]] = j;

            if (deg[j] == 0)
            {
                nodesLeft--;
            }
        }
    }

    //copy result back to device
    cudaMemcpy( aggregates.raw(), agg, sizeof(IndexType)*numRows, cudaMemcpyHostToDevice );

    //assert matching
    for (IndexType node = 0; node < numRows; node++)
    {
        if ( agg[node] == -1 )
        {
            continue;
        }

        for ( IndexType partner = 0; partner < numRows; partner++)
        {
            if ( agg[partner] == agg[node] )
            {
                if ( partner == node )
                {
                    continue;
                }

                bool neighbor = false;

                for (IndexType ii = ia[node]; ii < ia[node + 1]; ii++)
                    if ( ja[ii] == partner )
                    {
                        neighbor = true;
                        break;
                    }

                if ( !neighbor )
                {
                    for (IndexType ii = ia[partner]; ii < ia[partner + 1]; ii++)
                        if ( ja[ii] == node )
                        {
                            neighbor = true;
                            break;
                        }
                }

                if ( !neighbor )
                {
                    FatalError("Internal error in aggregation selector", AMGX_ERR_INTERNAL);
                }
            }
        }
    }

    //you shall not leak memory
    delete[] ia;
    delete[] ja;
    delete[] w;
    delete[] agg;
    delete[] deg;
    delete[] fwd;
    delete[] bwd;
}




//this kernel merges aggregate2 into aggregate1
template<typename IndexType>
__global__
void mergeAggregates(IndexType *aggregate1, const IndexType *aggregate2, IndexType sizeAggregate1, IndexType sizeAggregate2, IndexType sizeAggregate3)
{
    int tid = threadIdx.x + blockDim.x * blockIdx.x;

    while ( tid < sizeAggregate1 )
    {
        if ( aggregate1[tid] == sizeAggregate2 )
        {
            aggregate1[tid] = sizeAggregate3;
        }
        else
        {
            aggregate1[tid] = aggregate2[aggregate1[tid]];
        }

        tid += gridDim.x * blockDim.x;
    }
}



template <AMGX_VecPrecision t_vecPrec, AMGX_MatPrecision t_matPrec, AMGX_IndPrecision t_indPrec>
void MultiPairwiseSelector<TemplateConfig<AMGX_host, t_vecPrec, t_matPrec, t_indPrec> >::computeIncompleteGalerkin( const Matrix_h &A,
        Matrix_h &Ac,
        const typename Matrix_h::IVector &aggregates,
        const typename Matrix_h::IVector &R_row_offsets,
        const typename Matrix_h::IVector &R_column_indices,
        const int num_aggregates )
{
    FatalError("computeIncomlpetegalerkin is not supported on host. Run with ghost_offdiag_limit=0 instead.", AMGX_ERR_NOT_SUPPORTED_TARGET);
}

template <AMGX_VecPrecision t_vecPrec, AMGX_MatPrecision t_matPrec, AMGX_IndPrecision t_indPrec>
void MultiPairwiseSelector<TemplateConfig<AMGX_device, t_vecPrec, t_matPrec, t_indPrec> >::computeIncompleteGalerkin( const Matrix_d &A,
        Matrix_d &Ac,
        const typename Matrix_d::IVector &aggregates,
        const typename Matrix_d::IVector &R_row_offsets,
        const typename Matrix_d::IVector &R_column_indices,
        const int num_aggregates )
{
    FatalError("computeIncomlpetegalerkin is not implemented yet. run with ghost_offdiag_limit=0 instead.", AMGX_ERR_NOT_IMPLEMENTED);
}

template<class TConfig>
void MultiPairwiseSelectorBase<TConfig>::assertRestriction( const IVector &R_row_offsets, const IVector &R_col_indices, const IVector &aggregates )
{
    int *r_ia = new int[R_row_offsets.size()];
    int *r_ja = new int[R_col_indices.size()];
    int *agg = new int[aggregates.size()];
    int *used_col = new int[aggregates.size()];

    for ( int i = 0; i < aggregates.size(); i++ )
    {
        used_col[i] = 0;
    }

    cudaMemcpy( r_ia, R_row_offsets.raw(), sizeof(int)*R_row_offsets.size(), cudaMemcpyDeviceToHost );
    cudaMemcpy( r_ja, R_col_indices.raw(), sizeof(int)*R_col_indices.size(), cudaMemcpyDeviceToHost );
    cudaMemcpy( agg, aggregates.raw(), sizeof(int)*aggregates.size(), cudaMemcpyDeviceToHost );

    for ( int i = 0; i < R_row_offsets.size() - 1; i++ )
    {
        for ( int ii = r_ia[i]; ii < r_ia[i + 1]; ii++ )
        {
            int j = r_ja[ii];
            used_col[j]++;

            if ( used_col[j] > 1 )
            {
                std::cout << "column " << j << " is present at least " << used_col[j] << " times" << std::endl;
            }

            if ( j < 0 || j >= aggregates.size() )
            {
                std::cout << "Error: j out of bounds, j = " << j << " and numRows = " << aggregates.size() << std::endl;
            }
            else if  ( agg[j] != i )
            {
                std::cout << "Error: agg[" << j << "] = " << agg[j] << " != " << i << std::endl;
            }
        }
    }

    std::cout << "assert restriction done" << std::endl;
}

template<class T_Config>
void MultiPairwiseSelectorBase<T_Config>::setAggregates(Matrix<T_Config> &A,
        IVector &aggregates, IVector &aggregates_global, int &num_aggregates)
{
    if (A.get_block_dimx() == A.get_block_dimy())
    {
        //ghost level matrix. this is a probably a weight matrix
        Matrix<TConfig> ghostA;
        ghostA.values.resize(0);
        //prolongation and restriction operator. this is only needed in when LowDegCoarseAGenerator is used
        IVector R_row_offsets;
        IVector R_col_indices;
        //holds the size of each aggregate
        IVector sizes;
        sizes.resize(0);
        //aggregates for ghost level
        IVector aggregates_current;
        IVector aggregates_global_current;
        bool aggregates_initialized = true;

        if (aggregates.size() == 0)
        {
            aggregates_initialized = false;

            if (!A.is_matrix_singleGPU())
            {
                aggregates.resize(A.manager->halo_offset(A.manager->num_neighbors()));
            }
            else
            {
                aggregates.resize(A.get_num_rows());
            }
        }

        //for mergeAggregates kernel
        const int threads_per_block = 256;
        const int num_blocks = std::min( AMGX_GRID_MAX_SIZE, (A.get_num_rows() - 1) / threads_per_block + 1 );
        cudaStream_t stream = amgx::thrust::global_thread_handle::get_stream();
        //initialize and prepare weight matrix
        Matrix<TConfig> w;
        w.set_initialized(0);
        w.addProps(CSR);
        w.delProps(COO);
        w.setColsReorderedByColor(false);
        w.resize( 0, 0, 0, 1, 1, true ); //empty scalar 0x0 matrix
        w.values.resize(0); //matrix resize sets the values array to nnz+1 for no apparent reason
        IndexType targetSize = 1;
        //initialize coarse A generator
        CoarseAGenerator<TConfig> *cag;
        const bool use_restriction = true;
        const bool shrink_ghost_level = false;
        cag = new LowDegCoarseAGenerator<TConfig>(mCfg, mCfg_scope);
        // This will make coarseAGenerator to allocate more memory inside of  galerkin
        ghostA.manager = new DistributedManager<TConfig>();
        w.manager = new DistributedManager<TConfig>();
        Matrix<TConfig> *curA = &A;

        //foreach pass do:
        // 1. build aggregates and weights
        // 2. create weight matrix (in full_ghost_level mode this is the input matrix or the last ghostlevel matrix)
        // 3. if in full ghost level mode, build R
        // 4. compute next level
        for (int current_pass = 1; true; current_pass++)
        {
            const IndexType numRows = curA->get_num_rows();
            const IndexType nnz = curA->get_num_nz();
            targetSize *= 2;

            if ( full_ghost_level )
            {
                w.values.resize(0);    //compute weights from curA
            }
            else
            {
                w.values.swap( ghostA.values );    //use the weights computed with the galerkin operator (this will do nothing in the first pass, both values have size 0)
            }

            // create aggregates from correct input matrix
            setAggregates_common_sqblocks( *curA, aggregates_current, aggregates_global_current, num_aggregates, w.values, sizes );


            if ( current_pass > 1 )
            {
                //merge original aggregate with the newly created ones
                mergeAggregates <<< num_blocks, threads_per_block, 0, stream >>>( aggregates.raw(), aggregates_current.raw(), A.get_num_rows(), numRows, num_aggregates );
                cudaCheckError();
                //mergeAggregates<<< num_blocks, threads_per_block, 0, stream >>>( aggregates_global.raw(), aggregates_global_current.raw(), A.get_num_rows() );
                //cudaCheckError();
            }

            //try to free memory
            if ( full_ghost_level )
            {
                //then we don't need to save the weights, only for original level to do post processing
                w.values.resize(0);
            }
            else
            {
                //save edge weights for original level later
                //in that case we can throw away the values of ghostA as we will use the values to compute the next ghost level
                ghostA.values.resize(0);
            }

            // this is the break condition for the loop
            if ( current_pass >= aggregation_passes || num_aggregates <= 1 || num_aggregates == numRows)
            {
                //this means, aggregates has not been initialized yet
                if ( !aggregates_initialized )
                {
                    aggregates.swap( aggregates_current );
                }

                if ( !aggregates_initialized )
                {
                    aggregates_global.swap( aggregates_global_current );
                }

                cudaStreamSynchronize( stream );
                cudaCheckError();

                break;
            }

            //prepare A to be corrupted
            curA->set_initialized(0);
            //swap in ia, ja from curA
            w.row_offsets.swap( curA->row_offsets );
            w.col_indices.swap( curA->col_indices );

            if ( full_ghost_level )
            {
                if ( shrink_ghost_level && curA->get_block_dimx() > 1)
                {
                    //set w to correct size
                    w.values.resize( nnz );
                    //define grid and offsets
                    const int num_blocks_inter = std::min( (int)AMGX_GRID_MAX_SIZE, (int)(nnz - 1) / threads_per_block + 1 );
                    const int sq_blocksize = A.get_block_dimx() * A.get_block_dimy();
                    const int index_offset = A.get_block_dimy() * m_aggregation_edge_weight_component + m_aggregation_edge_weight_component;
                    //do the interleaved copy
                    gatherValuesInterleaved <<< num_blocks_inter, threads_per_block, 0, stream>>>( A.values.raw(), w.values.raw(), nnz, sq_blocksize, index_offset );
                    cudaStreamSynchronize( stream );
                    cudaCheckError();
                }
                else
                {
                    w.values.swap( curA->values );
                }
            }

            w.diag.swap( curA->diag );

            //resize to inform the matrix of its new size
            if ( full_ghost_level && !shrink_ghost_level )
            {
                w.set_block_dimx( A.get_block_dimx() );
                w.set_block_dimy( A.get_block_dimy() );
            }
            else
            {
                w.set_block_dimx( 1 );
                w.set_block_dimy( 1 );
            }

            w.set_num_rows( numRows );
            w.set_num_cols( numRows );
            w.set_num_nz( nnz );
            w.set_allow_recompute_diag( false );

            if ( curA->hasProps( DIAG ) )
            {
                w.addProps( DIAG );
            }

            //ready to use
            w.set_initialized(1);

            //compute restriction operator
            if ( use_restriction )
            {
                IVector R_row_indices(aggregates_current);
                R_row_offsets.resize(num_aggregates + 2);
                R_col_indices.resize(numRows);
                thrust_wrapper::sequence<TConfig::memSpace>(R_col_indices.begin(), R_col_indices.end());
                cudaCheckError();
                amgx::thrust::sort_by_key(R_row_indices.begin(), R_row_indices.end(), R_col_indices.begin());
                cudaCheckError();
                cusp::detail::indices_to_offsets(R_row_indices, R_row_offsets);
                cudaCheckError();
                //delete last row, which holds the pseudo aggregate
                R_row_offsets.resize( num_aggregates + 1);
                R_col_indices.resize( R_row_offsets[num_aggregates] );
            }

            // 3. compute galerkin ghost level
            if ( ghost_offdiag_limit == 0 )
            {
                //compute full galerkin
                cag->computeAOperator(w,
                                      ghostA,
                                      aggregates_current,
                                      R_row_offsets,
                                      R_col_indices,
                                      num_aggregates);
            }
            else
            {
                //compute incomplete galerkin
                computeIncompleteGalerkin(w,
                                          ghostA,
                                          aggregates_current,
                                          R_row_offsets,
                                          R_col_indices,
                                          num_aggregates);
            }

            //from now on w will be destroyed again.
            w.set_initialized(0);

            //repair the original A matrix. its ia and ja are in w
            if ( current_pass == 1 )
            {
                //swap back
                w.row_offsets.swap( A.row_offsets );
                w.col_indices.swap( A.col_indices );

                //only in that case we have swapped the values
                if ( full_ghost_level && !shrink_ghost_level )
                {
                    w.values.swap( A.values );
                }

                //save the edge weights of the original level
                A.diag.swap( w.diag );
                A.set_initialized(1); //A is repaired now
                //save the first aggregates into the original aggregate vector so we can merge them later
                aggregates.swap( aggregates_current );
                aggregates_global.swap( aggregates_global_current );
                aggregates_initialized = true;
                curA = &ghostA;
            }
        }

        delete cag;
    }
    else
    {
        FatalError("Unsupported block size for MultiPairwise", AMGX_ERR_NOT_SUPPORTED_BLOCKSIZE);
    }
}

// -------------------------
//    Explict instantiations
// -------------------------
#define AMGX_CASE_LINE(CASE) template class MultiPairwiseSelectorBase<TemplateMode<CASE>::Type>;
AMGX_FORALL_BUILDS(AMGX_CASE_LINE)
#undef AMGX_CASE_LINE
#define AMGX_CASE_LINE(CASE) template class MultiPairwiseSelector<TemplateMode<CASE>::Type>;
AMGX_FORALL_BUILDS(AMGX_CASE_LINE)
#undef AMGX_CASE_LINE

}
}
}
