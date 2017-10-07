/* Copyright (c) 2013-2017, NVIDIA CORPORATION. All rights reserved.
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

#include <memory_intrinsics.h>
#include <strided_reduction.h>
#include <cstdio>

using namespace amgx;

namespace amgx
{
namespace aggregation
{
namespace size8_selector
{


template <int NUM_COLS, typename IndexType>
__global__  __launch_bounds__(256, 4)
void my_findStrongestNeighbourBlockDiaCsr_NoMergeClean(
    const IndexType *row_offsets, const IndexType *column_indices,
    const float *edge_weights, const IndexType num_block_rows, const IndexType num_nonzero,
    IndexType *aggregated, IndexType *aggregates, int *strongest_neighbour,
    IndexType *partner_index, float *weight_strongest_neighbour, int deterministic,
    const IndexType *unassigned_rows,
    const int num_unassigned_row)
{
    int tid = threadIdx.x + blockDim.x * blockIdx.x;
    const int lane_id = utils::lane_id();
    bool valid_tid = false;

    for (; utils::any(valid_tid = tid < num_block_rows); tid += gridDim.x * blockDim.x)
    {
        int jmin = -NUM_COLS * 2, jmax = -NUM_COLS * 4;
        float weight;
        int jcol;
        float max_weight_unaggregated = 0;
        int strongest_unaggregated = -1;
        bool is_unassigned = false;

        if (valid_tid)
        {
            is_unassigned = (__load_streaming(partner_index + tid) == -1);
        }

        if (is_unassigned) // Unaggregated row
        {
            jmin = __load_all(row_offsets + tid);
            jmax = __load_all(row_offsets + tid + 1);
        }

        if (utils::any(is_unassigned))
        {
            int jj = jmin - amgx::strided_reduction::warp_loader<int, NUM_COLS>::align_shift(jmin);

            for (; utils::any(jj < jmax, utils::activemask()); jj += NUM_COLS)
            {
                int   I[NUM_COLS];
                float W[NUM_COLS];
                int   P[NUM_COLS];
                amgx::strided_reduction::warp_loader<int, NUM_COLS>::load(column_indices, jj, num_nonzero, I);

#pragma unroll
                for (int i = 0; i < NUM_COLS; i++)
                {
                    int j = jj + i;
                    jcol = I[i];

                    if (j >= jmin && j < jmax)
                    {
                        P[i] = __load_nc(partner_index + jcol); //make this load ASAP
                    }
                }

                amgx::strided_reduction::warp_loader<float, NUM_COLS>::load(edge_weights, jj, num_nonzero, W);

#pragma unroll
                for (int i = 0; i < NUM_COLS; i++)
                {
                    weight = W[i];
                    jcol = I[i];
                    int j = jj + i;

                    if (j >= jmin && j < jmax)
                    {
                        if (weight > max_weight_unaggregated || (weight == max_weight_unaggregated && jcol > strongest_unaggregated)) // unaggregated
                        {
                            if (tid != jcol && P[i] == -1)
                            {
                                max_weight_unaggregated = weight;
                                strongest_unaggregated = jcol;
                            }
                        }
                    }
                }
            }

            if (strongest_unaggregated == -1) // All neighbours are aggregated
            {
                // Put in its own aggregate
                if (!deterministic && is_unassigned)
                {
                    partner_index[tid] = tid;
                }
            }
            else
            {
                strongest_neighbour[tid] = strongest_unaggregated;
            }
        }
    }
}



#define ALGORITHM_NOMERGE 0
#define ALGORITHM_STOREWEIGHTS 1
#define ALGORITHM_STOREWEIGHTS_2 2


template <int NUM_COLS, int ALGORITHM, int ASSUME_ALL_UNASSIGNED, int LOAD_ONLY_UNASSIGNED, typename IndexType>
__global__  __launch_bounds__(256, 4)
void my_findStrongestNeighbourBlockDiaCsr_NoMerge(
    const IndexType *row_offsets, const IndexType *column_indices,
    const float *edge_weights, const IndexType num_block_rows, const IndexType num_nonzero,
    IndexType *aggregated, IndexType *aggregates, int *strongest_neighbour,
    IndexType *partner_index, float *weight_strongest_neighbour, int deterministic,
    const IndexType *n_unassigned_per_block, const IndexType *unassigned_per_block
    //const int num_unassigned_row
    //const IndexType *unassigned_rows,
)
{
    int tid = threadIdx.x + blockDim.x * blockIdx.x;
    int bid = blockIdx.x;
    const int lane_id = utils::lane_id();
    bool valid_tid = false;

    for (; utils::any( valid_tid = tid < num_block_rows); tid += gridDim.x * blockDim.x)
    {
        int jmin = -NUM_COLS * 2, jmax = -NUM_COLS * 4;
        float weight;
        int jcol;
        float max_weight_unaggregated = 0;
        int strongest_unaggregated = -1;
        float max_weight_aggregated = 0.;
        int strongest_aggregated = -1;
        int partner = -1;
        int partner0, partner1, partner2;
        int agg_jcol;
        bool is_unassigned = false;
        int  rowi = -1;

        if (LOAD_ONLY_UNASSIGNED)
        {
            if (valid_tid)
            {
                rowi  = unassigned_per_block[tid];//unassigned_per_block+bid*256+threadIdx.x);
                is_unassigned = (__load_nc(partner_index + rowi) == -1);
            }

            if (is_unassigned)
            {
                jmin = __load_nc(row_offsets + rowi);
                jmax = __load_nc(row_offsets + rowi + 1);
            }
        }
        else
        {
            rowi = tid;

            if (ALGORITHM == ALGORITHM_NOMERGE)
            {
                if (valid_tid) { is_unassigned = (__load_streaming(partner_index + tid) == -1); }
            }
            else //ALGORITHM_STOREWEIGHTS or ALGORITHM_STOREWEIGHTS_2
            {
                if (valid_tid) { is_unassigned = (__load_streaming(aggregated + tid) == -1); }
            }

            if (is_unassigned) // mind the else above
            {
                jmin = __load_global(row_offsets + rowi);
                jmax = __load_lastuse(row_offsets + rowi + 1);
            }
        }

        if (utils::any(is_unassigned))
        {
            if (is_unassigned) // Unaggregated row
            {
                if (ALGORITHM == ALGORITHM_STOREWEIGHTS)
                {
                    partner  = partner_index[rowi];
                }
                else if (ALGORITHM == ALGORITHM_STOREWEIGHTS_2)
                {
                    partner0 = partner_index[rowi];
                    partner1 = partner_index[num_block_rows + rowi];
                    partner2 = partner_index[2 * num_block_rows + rowi];
                }
            }

            int jj = jmin - amgx::strided_reduction::warp_loader<int, NUM_COLS>::align_shift(jmin);

            for (; utils::any(jj < jmax && jmax >= 0); jj += NUM_COLS)
            {
                int   I[NUM_COLS];
                float W[NUM_COLS];
                int   P[NUM_COLS];
                int jj_ok = (jj >= 0 && jj < jmax && jmax >= 0) ? jj : 0;

                amgx::strided_reduction::warp_loader<int, NUM_COLS>::load(column_indices, jj_ok, num_nonzero, I);
#pragma unroll

                for (int i = 0; i < NUM_COLS; i++)
                {
                    int j = jj + i;
                    jcol = I[i];

                    if (j >= jmin && j < jmax)
                    {
                        if (ALGORITHM == ALGORITHM_NOMERGE)
                        {
                            P[i] = __load_nc(partner_index + jcol);    //make this load ASAP
                        }
                        else
                        {
                            P[i] = __load_nc(aggregated + jcol);
                        }
                    }
                }

                amgx::strided_reduction::warp_loader<float, NUM_COLS>::load(edge_weights, jj_ok, num_nonzero, W);
#pragma unroll

                for (int i = 0; i < NUM_COLS; i++)
                {
                    weight = W[i];
                    jcol = I[i];
                    int j = jj + i;

                    if (j >= jmin && j < jmax)
                    {
                        if (ALGORITHM == ALGORITHM_NOMERGE)
                        {
                            if (weight > max_weight_unaggregated || (weight == max_weight_unaggregated && jcol > strongest_unaggregated)) // unaggregated
                            {
                                if (rowi != jcol && P[i] == -1)
                                {
                                    max_weight_unaggregated = weight;
                                    strongest_unaggregated = jcol;
                                }
                            }
                        }
                        else
                        {
                            bool partner_condition;

                            if (ALGORITHM == ALGORITHM_STOREWEIGHTS)
                            {
                                partner_condition = jcol != partner;
                            }
                            else if (ALGORITHM == ALGORITHM_STOREWEIGHTS_2)
                            {
                                partner_condition = jcol != partner0 && jcol != partner1 && jcol != partner2;
                            }

                            agg_jcol = P[i];

                            if (partner_condition && rowi != jcol)
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
                    }
                }
            }

            if (valid_tid && is_unassigned)
            {
                if (ALGORITHM == ALGORITHM_NOMERGE)
                {
                    if (strongest_unaggregated == -1) // All neighbours are aggregated
                    {
                        // Put in its own aggregate
                        if (!deterministic)
                        {
                            partner_index[rowi] = rowi;
                        }
                    }
                    else
                    {
                        strongest_neighbour[rowi] = strongest_unaggregated;
                    }
                }
                else if (ALGORITHM == ALGORITHM_STOREWEIGHTS)
                {
                    if (strongest_unaggregated == -1) // All neighbours are aggregated
                    {
                        if (!deterministic)
                        {
                            if (strongest_aggregated != -1)
                            {
                                aggregates[tid] = aggregates[strongest_aggregated];
                                aggregated[tid] = 1;
                                aggregates[partner] = aggregates[strongest_aggregated];
                                aggregated[partner] = 1;
                            }
                            else  // leave in its own aggregate
                            {
                                aggregated[partner] = 1;
                                aggregated[tid] = 1;
                            }
                        }
                    }
                    else // Found an unaggregated aggregate
                    {
                        weight_strongest_neighbour[tid] = max_weight_unaggregated;
                        strongest_neighbour[tid] = aggregates[strongest_unaggregated];
                    }
                }
                else if (ALGORITHM == ALGORITHM_STOREWEIGHTS_2)
                {
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

        bid += gridDim.x;
    }
}




#define INST_my_findStrongestNeighbourBlockDiaCsr_NoMerge(numcols, algo,assume,c) template __global__ void my_findStrongestNeighbourBlockDiaCsr_NoMerge<numcols,algo,assume,c>(\
const int *row_offsets, const int *column_indices,\
const float *edge_weights, const int num_block_rows,const int num_nonzero,\
int *aggregated, int *aggregates, int *strongest_neighbour,\
int *partner_index, float *weight_strongest_neighbour, int deterministic,\
const int* n_unassigned_per_block, const int * unassigned_per_block);

#define INSTmy_findStrongestNeighbourBlockDiaCsr_NoMergeClean(numcols) template __global__ void my_findStrongestNeighbourBlockDiaCsr_NoMergeClean<numcols,int>(\
const int *row_offsets, const int *column_indices,\
const float *edge_weights, const int num_block_rows,const int num_nonzero,\
int *aggregated, int *aggregates, int *strongest_neighbour,\
int *partner_index, float *weight_strongest_neighbour, int deterministic,\
const int *unassigned_rows,\
const int num_unassigned_row);

INST_my_findStrongestNeighbourBlockDiaCsr_NoMerge(1, ALGORITHM_NOMERGE, 0, 0)
INST_my_findStrongestNeighbourBlockDiaCsr_NoMerge(1, ALGORITHM_STOREWEIGHTS, 0, 0)
INST_my_findStrongestNeighbourBlockDiaCsr_NoMerge(1, ALGORITHM_STOREWEIGHTS_2, 0, 0)
INST_my_findStrongestNeighbourBlockDiaCsr_NoMerge(1, ALGORITHM_NOMERGE, 1, 0)
INST_my_findStrongestNeighbourBlockDiaCsr_NoMerge(1, ALGORITHM_STOREWEIGHTS, 1, 0)
INST_my_findStrongestNeighbourBlockDiaCsr_NoMerge(1, ALGORITHM_STOREWEIGHTS_2, 1, 0)
INST_my_findStrongestNeighbourBlockDiaCsr_NoMerge(1, ALGORITHM_NOMERGE, 0, 1) //load only unassigned

INST_my_findStrongestNeighbourBlockDiaCsr_NoMerge(2, ALGORITHM_NOMERGE, 0, 0)
INST_my_findStrongestNeighbourBlockDiaCsr_NoMerge(2, ALGORITHM_STOREWEIGHTS, 0, 0)
INST_my_findStrongestNeighbourBlockDiaCsr_NoMerge(2, ALGORITHM_STOREWEIGHTS_2, 0, 0)
INST_my_findStrongestNeighbourBlockDiaCsr_NoMerge(2, ALGORITHM_NOMERGE, 1, 0)
INST_my_findStrongestNeighbourBlockDiaCsr_NoMerge(2, ALGORITHM_STOREWEIGHTS, 1, 0)
INST_my_findStrongestNeighbourBlockDiaCsr_NoMerge(2, ALGORITHM_STOREWEIGHTS_2, 1, 0)
INST_my_findStrongestNeighbourBlockDiaCsr_NoMerge(2, ALGORITHM_NOMERGE, 0, 1) //load only unassigned

INST_my_findStrongestNeighbourBlockDiaCsr_NoMerge(4, ALGORITHM_NOMERGE, 0, 0)
INST_my_findStrongestNeighbourBlockDiaCsr_NoMerge(4, ALGORITHM_STOREWEIGHTS, 0, 0)
INST_my_findStrongestNeighbourBlockDiaCsr_NoMerge(4, ALGORITHM_STOREWEIGHTS_2, 0, 0)
INST_my_findStrongestNeighbourBlockDiaCsr_NoMerge(4, ALGORITHM_NOMERGE, 1, 0)
INST_my_findStrongestNeighbourBlockDiaCsr_NoMerge(4, ALGORITHM_STOREWEIGHTS, 1, 0)
INST_my_findStrongestNeighbourBlockDiaCsr_NoMerge(4, ALGORITHM_STOREWEIGHTS_2, 1, 0)
INST_my_findStrongestNeighbourBlockDiaCsr_NoMerge(4, ALGORITHM_NOMERGE, 0, 1) //load only unassigned


INSTmy_findStrongestNeighbourBlockDiaCsr_NoMergeClean(1)
INSTmy_findStrongestNeighbourBlockDiaCsr_NoMergeClean(2)
INSTmy_findStrongestNeighbourBlockDiaCsr_NoMergeClean(4)

#define __load_ __load_streaming

template<int ALREADY_COMPACT>
__global__ void my_blockCompact(
    int *partner_index, const int num_rows,
    int *unassigned_per_block_in,
    int *n_unassigned_per_block, int *unassigned_per_block)
{
    int bid = blockIdx.x; //RMV
    int tid = threadIdx.x + blockDim.x * blockIdx.x;

    for (; bid < num_rows / 256 + 1/*__any(tid < num_rows)*/; tid += gridDim.x * blockDim.x)
    {
        int row = tid;
        bool no_partner = 0; //RMV

        if (tid < num_rows)
        {
            if (ALREADY_COMPACT)
            {
                row = unassigned_per_block_in[tid];
            }

            if (partner_index[row] == -1) // Unaggregated row
            {
                no_partner = 1;
            }
        }

        amgx::strided_reduction::block_binary_compaction<256, 32, 1>(
            n_unassigned_per_block, unassigned_per_block, bid,
            no_partner, row);
        bid += gridDim.x;
    }
}

template __global__ void my_blockCompact<0>(
    int *partner_index, const int num_rows,
    int *unassigned_per_block_in,
    int *n_unassigned_per_block, int *unassigned_per_block);
template __global__ void my_blockCompact<1>(
    int *partner_index, const int num_rows,
    int *unassigned_per_block_in,
    int *n_unassigned_per_block, int *unassigned_per_block);

__global__ void my_MatchEdgesWithCompaction(const int num_rows, int *partner_index, int *aggregates, const int *strongest_neighbour, int *sets_per_block,
        int *unassigned_per_block_in, int *n_unassigned_per_block, int *unassigned_per_block
                                           )
{
    int potential_match, potential_match_neighbour;
    int warp_count = 0;
    int bid = blockIdx.x; //RMV
    int tid = threadIdx.x + blockDim.x * blockIdx.x;

    for (; bid < num_rows / 1024 + 1/*__any(tid < num_rows)*/; tid += gridDim.x * blockDim.x)
    {
        int row = tid;
        bool no_partner = 0;

        if (tid < num_rows)
        {
            if (partner_index[row] == -1) // Unaggregated row
            {
                no_partner = 1;
                potential_match = strongest_neighbour[row];

                if (potential_match != -1)
                {
                    potential_match_neighbour = __load_(strongest_neighbour + potential_match);

                    if ( potential_match_neighbour == row ) // we have a match
                    {
                        no_partner = 0;
                        //partner_notnull = 1;//RMV
                        partner_index[row] = potential_match;
                        aggregates[row] = ( potential_match > row ) ? row : potential_match;
                    }
                }
            }
        }

        amgx::strided_reduction::block_binary_compaction<1024, 32, 1>(
            n_unassigned_per_block, unassigned_per_block, bid,
            no_partner, row);
        warp_count += amgx::strided_reduction::warp_binary_count(no_partner);
        bid += gridDim.x;
    }

    amgx::strided_reduction::block_count<1, 1024, 32, int>(warp_count, sets_per_block);
}

__global__ void my_MatchEdges(const int num_rows, int *partner_index, int *aggregates, const int *strongest_neighbour, int *sets_per_block)
{
    int potential_match, potential_match_neighbour;
    int warp_count = 0;

    for (int tid = threadIdx.x + blockDim.x * blockIdx.x; utils::any(tid < num_rows); tid += gridDim.x * blockDim.x)
    {
        bool has_set_partner_index = 0;

        if (tid < num_rows)
        {
            if (partner_index[tid] == -1) // Unaggregated row
            {
                potential_match = strongest_neighbour[tid];

                if (potential_match != -1)
                {
                    potential_match_neighbour = __load_(strongest_neighbour + potential_match);

                    if ( potential_match_neighbour == tid ) // we have a match
                    {
                        has_set_partner_index = 1;
                        partner_index[tid] = potential_match;
                        aggregates[tid] = ( potential_match > tid) ? tid : potential_match;
                    }
                }
            }
        }

        warp_count += amgx::strided_reduction::warp_binary_count(has_set_partner_index);
    }

    amgx::strided_reduction::block_count<1, 1024, 32, int>(warp_count, sets_per_block);
}


// matchEdges
__global__ void my_joinExistingAggregates(int num_rows, int *aggregates, int *aggregated, int *aggregates_candidate, int *sets_per_block)
{
    int tid = threadIdx.x + blockDim.x * blockIdx.x;
    int warp_count = 0;

    while  (utils::any(tid < num_rows))
    {
        bool has_set_partner_index = 0;

        if (tid < num_rows)
        {
            if (aggregated[tid] == -1 && aggregates_candidate[tid] != -1) // Unaggregated row
            {
                aggregates[tid] = aggregates_candidate[tid];
                aggregated[tid] = 1;
                has_set_partner_index = 1;
            }
        }

        warp_count += amgx::strided_reduction::warp_binary_count(has_set_partner_index);
        tid += gridDim.x * blockDim.x;
    }

    amgx::strided_reduction::block_count<1, 1024, 32, int>(warp_count, sets_per_block);
}


// Kernel that checks if perfect matchs exist
__global__ void my_matchAggregates(int *aggregates, int *aggregated, int *strongest_neighbour, const int num_rows, int *sets_per_block)
{
    int tid = threadIdx.x + blockDim.x * blockIdx.x;
    int potential_match, potential_match_neighbour, my_aggregate;
    int warp_count = 0;

    while (utils::any(tid < num_rows))
    {
        bool has_set_partner_index = 0;

        if (tid < num_rows) if (aggregated[tid] == -1) // Unaggregated row
            {
                potential_match = strongest_neighbour[tid];

                if (potential_match != -1)
                {
                    potential_match_neighbour = __load_(strongest_neighbour + potential_match); //or global
                    my_aggregate = aggregates[tid];

                    if (potential_match_neighbour == my_aggregate) // we have a match
                    {
                        has_set_partner_index = 1;
                        aggregated[tid] = 1;
                        aggregates[tid] = ( potential_match > my_aggregate) ? my_aggregate : potential_match;
                    }
                }
            }

        warp_count += amgx::strided_reduction::warp_binary_count(has_set_partner_index);
        tid += gridDim.x * blockDim.x;
    }

    amgx::strided_reduction::block_count<1, 1024, 32, int>(warp_count, sets_per_block);
}


// Kernel that checks if perfect matchs exist
__global__ void my_matchAggregatesSize4(int *aggregates, int *aggregated, int *strongest_neighbour, int *partner_index, const int num_rows, int *sets_per_block)
{
    int potential_match, potential_match_neighbour, my_aggregate;
    int warp_count = 0;

    for (int tid = threadIdx.x + blockDim.x * blockIdx.x; utils::any(tid < num_rows); tid += blockDim.x * gridDim.x)
    {
        bool has_set_partner_index = 0;

        if (tid < num_rows) if (aggregated[tid] == -1) // Unaggregated row
            {
                potential_match = strongest_neighbour[tid];

                if (potential_match != -1)
                {
                    potential_match_neighbour = __load_(strongest_neighbour + potential_match);
                    my_aggregate = aggregates[tid];

                    if (potential_match_neighbour == my_aggregate) // we have a match
                    {
                        has_set_partner_index = 1;
                        aggregated[tid] = 1;
                        aggregates[tid] = ( potential_match > my_aggregate) ? my_aggregate : potential_match;
                        partner_index[tid + num_rows] = potential_match;
                        partner_index[tid + 2 * num_rows] = partner_index[potential_match];
                    }
                }
            }

        warp_count += amgx::strided_reduction::warp_binary_count(has_set_partner_index);
    }

    amgx::strided_reduction::block_count<1, 1024, 32, int>(warp_count, sets_per_block);
}

}
}
}

