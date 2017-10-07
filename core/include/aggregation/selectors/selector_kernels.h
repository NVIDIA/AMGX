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

#pragma once

namespace amgx
{
namespace aggregation
{
namespace size8_selector
{


#define ALGORITHM_NOMERGE 0
#define ALGORITHM_STOREWEIGHTS 1
#define ALGORITHM_STOREWEIGHTS_2 2

template <int NUM_COLS_, typename IndexType>
__global__  __launch_bounds__(256, 4)
void my_findStrongestNeighbourBlockDiaCsr_NoMergeClean(
    const IndexType *row_offsets, const IndexType *column_indices,
    const float *edge_weights, const IndexType num_block_rows, const IndexType num_nonzero,
    IndexType *aggregated, IndexType *aggregates, int *strongest_neighbour,
    IndexType *partner_index, float *weight_strongest_neighbour, int deterministic,
    const IndexType *unassigned_rows,
    const int num_unassigned_row);

template <int NUM_COLS, int ALGORITHM, int ASSUME_ALL_UNASSIGNED, int LOAD_ONLY_UNASSIGNED, typename IndexType>
__global__  __launch_bounds__(256, 4)
void my_findStrongestNeighbourBlockDiaCsr_NoMerge(
    const IndexType *row_offsets, const IndexType *column_indices,
    const float *edge_weights, const IndexType num_block_rows, const IndexType num_nonzero,
    IndexType *aggregated, IndexType *aggregates, int *strongest_neighbour,
    IndexType *partner_index, float *weight_strongest_neighbour, int deterministic,
    const IndexType *n_unassigned_per_block,
    const IndexType *unassigned_per_block);

__global__ void my_MatchEdges(const int num_rows, int *partner_index, int *aggregates, const int *strongest_neighbour, int *sets_per_block);
__global__ void my_joinExistingAggregates(int num_rows, int *aggregates, int *aggregated, int *aggregates_candidate, int *sets_per_block);
__global__ void my_matchAggregates(int *aggregates, int *aggregated, int *strongest_neighbour, const int num_rows, int *sets_per_block);
__global__ void my_matchAggregatesSize4(int *aggregates, int *aggregated, int *strongest_neighbour, int *partner_index, const int num_rows, int *sets_per_block);
template<int ALREADY_COMPACT> __global__ void my_blockCompact(int *partner_index, const int num_row, int *in_unassigned_per_block, int *n_unassigned_per_block, int *unassigned_per_block);
__global__ void my_MatchEdgesWithCompaction(const int num_rows, int *partner_index, int *aggregates, const int *strongest_neighbour, int *sets_per_block,
        int *unassigned_per_block_in, int *n_unassigned_per_block, int *unassigned_per_block);


}
}
}

