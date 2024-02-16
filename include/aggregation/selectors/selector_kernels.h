// SPDX-FileCopyrightText: 2013 - 2024 NVIDIA CORPORATION. All Rights Reserved.
//
// SPDX-License-Identifier: BSD-3-Clause

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

