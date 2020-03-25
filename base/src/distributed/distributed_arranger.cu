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

#include <distributed/distributed_arranger.h>
#include <thrust/sequence.h>
#include <thrust/iterator/constant_iterator.h>
#include <thrust/functional.h>
#include <basic_types.h>
#include <thrust/host_vector.h>
#include <thrust/scan.h>
#include <thrust/count.h>
#include <thrust/sort.h>
#include <thrust/adjacent_difference.h>
#include <thrust/gather.h>
#include <cutil.h>
#include <util.h>
#include <sm_utils.inl>
#include <csr_multiply.h>
#include <algorithm>

#include <amgx_types/util.h>

namespace amgx
{

/***************************************
 * Source Definitions
 ***************************************/

struct is_less_than_zero
{
    __host__ __device__
    bool operator()(int x) const
    {
        return x < 0;
    }
};


struct is_non_neg
{
    __host__ __device__
    bool operator()(const int &x)
    {
        return x >= 0;
    }
};

template<typename IndexType, int cta_size >
__global__
void flag_boundary_nodes_P(const IndexType *A_rows, const IndexType *A_cols, int *boundary_flags,  int num_owned_coarse_pts, int A_num_rows, int* nnz_interior)
{
    const int nWarps = cta_size / 32;
    // The coordinates of the thread inside the CTA/warp.
    const int warpId = threadIdx.x / 32;
    const int laneId = threadIdx.x % 32;

    int nnz_int = 0;

    // Loop over rows of A.
    for ( int aRowId = blockIdx.x * nWarps + warpId ; aRowId < A_num_rows ; aRowId += gridDim.x * nWarps )
    {
        // Load A row IDs.
        int aColIt  = A_rows[aRowId  ];
        int aColEnd = A_rows[aRowId + 1];
        int nnz = aColEnd - aColIt;
        bool halo_found = 0;

        // Iterate over the columns of A.
        for ( aColIt += laneId ; utils::any( aColIt < aColEnd ) && !halo_found; aColIt += 32 )
        {
            // Columns of A maps to rows of B. Each thread of the warp loads its A-col/B-row ID.
            int aColId = aColIt < aColEnd ? A_cols[aColIt] : -1;
            bool is_halo = (aColIt < aColEnd) && (aColId >= num_owned_coarse_pts);

            // Push coarse and strongly connected nodes in the set.
            if (utils::any(is_halo))
            {
                halo_found = true;
                break;
            }
        }

        if (laneId == 0)
        {
            if(halo_found)
            {
                boundary_flags[aRowId] = 1;
            }
            else
            {
                nnz_int += nnz;
            }
        }
    }

    if(laneId == 0)
    {
        atomicAdd(nnz_interior, nnz_int);
    }
}


__global__
void flag_boundary_nodes(int *b2l_maps, int *boundary_flags, int size )
{
    for (int tidx = threadIdx.x + blockIdx.x * blockDim.x; tidx < size; tidx += blockDim.x * gridDim.x)
    {
        boundary_flags[b2l_maps[tidx]] = 1;
    }
}

__global__
void flag_interior_nodes(int *boundary_flags, int *interior_flags, int size )
{
    for (int tidx = threadIdx.x + blockIdx.x * blockDim.x; tidx < size; tidx += blockDim.x * gridDim.x)
    {
        interior_flags[tidx] = !boundary_flags[tidx];
    }
}

__global__
void fill_rows_lists(int *__restrict__ interior_rows_lists, int *__restrict__ boundary_rows_lists, const int *__restrict__ interior_flags, const int *__restrict__ boundary_flags, int size )
{
    for (int tidx = threadIdx.x + blockIdx.x * blockDim.x; tidx < size; tidx += blockDim.x * gridDim.x)
    {
        int bdy_flag = boundary_flags[tidx];

        if (bdy_flag != boundary_flags[tidx + 1]) { boundary_rows_lists[bdy_flag] = tidx; }

        int int_flag = interior_flags[tidx];

        if (int_flag != interior_flags[tidx + 1]) { interior_rows_lists[int_flag] = tidx; }
    }
}


template<typename IndexType, int cta_size >
__global__
void fill_B_B2L_map(const IndexType *maps_offsets, const IndexType *A_b2l_map, const IndexType *row_offsets, IndexType *C_hat_b2l_map, int size, int rank)
{
    const int nWarps = cta_size / 32;
    // The coordinates of the thread inside the CTA/warp.
    const int warpId = threadIdx.x / 32;
    const int laneId = threadIdx.x % 32;

    // Loop over rows of A.
    for ( int aRowId = blockIdx.x * nWarps + warpId ; aRowId < size ; aRowId += gridDim.x * nWarps )
    {
        // Load A row IDs.
        int aColIt  = maps_offsets[aRowId  ];
        int size = maps_offsets[aRowId + 1] - aColIt;
        int start = row_offsets[A_b2l_map[aRowId]];
        int end = start + size;

        // Iterate over the columns of A.
        for ( start += laneId, aColIt += laneId ; utils::any( start < end ) ; start += 32, aColIt += 32 )
        {
            if (start < end)
            {
                C_hat_b2l_map[aColIt] = start;
            }
        }
    }
}

template <typename INDEX_TYPE>
__global__ void fill_halo_rows_row_ids(int64_t *halo_rows_row_ids, INDEX_TYPE *halo_rows_local_row_ids, int64_t base_index, INDEX_TYPE num_owned_coarse_pts, INDEX_TYPE size)
{
    for (int tidx = threadIdx.x + blockIdx.x * blockDim.x; tidx < size; tidx += blockDim.x * gridDim.x)
    {
        int64_t gid = halo_rows_row_ids[tidx];
        halo_rows_local_row_ids[tidx] = (INDEX_TYPE) (gid - base_index);

        if (gid < base_index || gid > base_index + num_owned_coarse_pts)
        {
            printf("Should not be here in fill_halo_rows_row_ids\n");
        }
    }
}

__global__
void fill_local_to_global_map(int *bdy_list, int64_t *local_to_global, int64_t offset, int size )
{
    for (int tidx = threadIdx.x + blockIdx.x * blockDim.x; tidx < size; tidx += blockDim.x * gridDim.x)
    {
        local_to_global[tidx] = bdy_list[tidx] + offset;
    }
}

template <typename INDEX_TYPE>
__global__ void remove_fine_halo(int64_t *cf_map_global, INDEX_TYPE *flag, INDEX_TYPE size)
{
    for (int tidx = threadIdx.x + blockIdx.x * blockDim.x; tidx < size; tidx += blockDim.x * gridDim.x)
    {
        flag[tidx] = (cf_map_global[tidx] == -1) ? 0 : 1;
    }
}

template <typename INDEX_TYPE>
__global__ void fill_P_local_to_global(int64_t *P_local_to_global, int64_t *cf_map_global, INDEX_TYPE *flag, INDEX_TYPE size)
{
    for (int tidx = threadIdx.x + blockIdx.x * blockDim.x; tidx < size; tidx += blockDim.x * gridDim.x)
    {
        if (cf_map_global[tidx] != -1)
        {
            P_local_to_global[flag[tidx]] = cf_map_global[tidx];
        }
    }
}

template <int coop>
__global__ void calc_num_neighbors(INDEX_TYPE *row_offsets, INDEX_TYPE *col_indices, int64_t *part_offsets, INDEX_TYPE *exists, INDEX_TYPE num_part, INDEX_TYPE my_id, INDEX_TYPE num_rows)
{
    int row = blockIdx.x * blockDim.x / coop + threadIdx.x / coop;
    int coopIdx = threadIdx.x % coop;

    while (row < num_rows)
    {
        for (int i = row_offsets[row] + coopIdx; i < row_offsets[row + 1]; i += coop)
        {
            INDEX_TYPE col = col_indices[i];

            if (col < part_offsets[my_id] || col >= part_offsets[my_id + 1])
            {
                int part = 0;

                while (part < num_part && (col < part_offsets[part] || col >= part_offsets[part + 1])) { part++; }

                if (part < num_part && (col >= part_offsets[part] && col < part_offsets[part + 1]))
                {
                    exists[part] = 1;
                }
            }
        }

        row += gridDim.x * blockDim.x / coop;
    }
}

// this is assuming local indices (used in matrix A)
template <int coop>
__global__ void calc_num_neighbors_v2_local(INDEX_TYPE *row_offsets, INDEX_TYPE *col_indices, int64_t *part_offsets, INDEX_TYPE *exists, int64_t *local_to_global_map, INDEX_TYPE num_part, INDEX_TYPE my_id, INDEX_TYPE num_rows)
{
    int row = blockIdx.x * blockDim.x / coop + threadIdx.x / coop;
    int coopIdx = threadIdx.x % coop;

    while (row < num_rows)
    {
        for (int i = row_offsets[row] + coopIdx; i < row_offsets[row + 1]; i += coop)
        {
            INDEX_TYPE col = col_indices[i];

            // Check if column point to halo node
            if (col >= num_rows)
            {
                int64_t gcol = local_to_global_map[col - num_rows];
                int part = 0;

                while (part < num_part && (gcol < part_offsets[part] || gcol >= part_offsets[part + 1]))
                {
                    part++;
                }

                if (part < num_part && (gcol >= part_offsets[part] && gcol < part_offsets[part + 1]))
                {
                    exists[part] = 1;
                }
            }
        }

        row += gridDim.x * blockDim.x / coop;
    }
}

// this is assuming global indices (used in halo rows)
template <int coop>
__global__ void calc_num_neighbors_v2_global(INDEX_TYPE *row_offsets, int64_t *col_indices, int64_t *part_offsets, INDEX_TYPE *exists, INDEX_TYPE num_part, INDEX_TYPE my_id, INDEX_TYPE num_rows)
{
    int row = blockIdx.x * blockDim.x / coop + threadIdx.x / coop;
    int coopIdx = threadIdx.x % coop;
    int64_t start = part_offsets[my_id];
    int64_t end = part_offsets[my_id + 1];

    while (row < num_rows)
    {
        for (int i = row_offsets[row] + coopIdx; i < row_offsets[row + 1]; i += coop)
        {
            int64_t col = col_indices[i];

            // Check if column point to halo node
            if (col < start || col >= end)
            {
                int part = 0;

                while (part < num_part && (col < part_offsets[part] || col >= part_offsets[part + 1]))
                {
                    part++;
                }

                if (part < num_part && (col >= part_offsets[part] && col < part_offsets[part + 1]))
                {
                    exists[part] = 1;
                }
            }
        }

        row += gridDim.x * blockDim.x / coop;
    }
}

template <int coop>
__global__ void flag_halo_nodes(INDEX_TYPE *row_offsets, INDEX_TYPE *col_indices, int64_t *part_ranges, INDEX_TYPE *flags, INDEX_TYPE *flag_offsets, int64_t base, INDEX_TYPE range, INDEX_TYPE num_part, INDEX_TYPE num_rows)
{
    int row = blockIdx.x * blockDim.x / coop + threadIdx.x / coop;
    int coopIdx = threadIdx.x % coop;

    while (row < num_rows)
    {
        for (int i = row_offsets[row] + coopIdx; i < row_offsets[row + 1]; i += coop)
        {
            INDEX_TYPE col = col_indices[i];

            if (col < base || col >= base + range)
            {
                int part = 0;

                while (part < num_part && (col < part_ranges[2 * part] || col >= part_ranges[2 * part + 1])) { part++; }

                if (part < num_part && (col >= part_ranges[2 * part] && col < part_ranges[2 * part + 1]))
                {
                    flags[flag_offsets[part] + col - part_ranges[2 * part]] = 1;
                }
            }
        }

        row += gridDim.x * blockDim.x / coop;
    }
}

template <int coop, int set_val>
__global__ void flag_halo_nodes_local(INDEX_TYPE *row_offsets, INDEX_TYPE *col_indices, int64_t *part_ranges, INDEX_TYPE *flags, INDEX_TYPE *flag_offsets, int64_t base, INDEX_TYPE range, INDEX_TYPE num_part, INDEX_TYPE num_rows, int64_t *local_to_global)
{
    int row = blockIdx.x * blockDim.x / coop + threadIdx.x / coop;
    int coopIdx = threadIdx.x % coop;

    while (row < num_rows)
    {
        for (int i = row_offsets[row] + coopIdx; i < row_offsets[row + 1]; i += coop)
        {
            INDEX_TYPE col = col_indices[i];

            if (col >= num_rows)
            {
                int64_t global_col = local_to_global[col - num_rows];
                int part = 0;

                while (part < num_part && (global_col < part_ranges[2 * part] || global_col >= part_ranges[2 * part + 1])) { part++; }

                if (part < num_part && (global_col >= part_ranges[2 * part] && global_col < part_ranges[2 * part + 1]))
                {
                    flags[flag_offsets[part] + global_col - part_ranges[2 * part]] = set_val > 0 ? set_val : -col - 1; // if -1, then set -(local index)-1 to restore later
                }
            }
        }

        row += gridDim.x * blockDim.x / coop;
    }
}

__global__ void flag_halo_nodes_from_local_to_global_map(int64_t *part_ranges, INDEX_TYPE *flags, INDEX_TYPE *flag_offsets, int64_t base, INDEX_TYPE range, INDEX_TYPE num_part, INDEX_TYPE size, int64_t *local_to_global)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    while (tid < size)
    {
        int64_t global_col = local_to_global[tid];
        int part = 0;

        while (part < num_part && (global_col < part_ranges[2 * part] || global_col >= part_ranges[2 * part + 1]))
        {
            part++;
        }

        if (part < num_part && (global_col >= part_ranges[2 * part] && global_col < part_ranges[2 * part + 1]))
        {
            int col_id = range + tid;
            flags[flag_offsets[part] + global_col - part_ranges[2 * part]] = -1 - col_id;
        }

        tid += gridDim.x * blockDim.x;
    }
}

__global__ void flag_halo_nodes_local_v2(int64_t *part_ranges, INDEX_TYPE *flags, INDEX_TYPE *flag_offsets, int64_t base, INDEX_TYPE range, INDEX_TYPE num_part, INDEX_TYPE local_to_global_size, int64_t *local_to_global)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    while (tid < local_to_global_size)
    {
        int64_t global_col = local_to_global[tid];
        int col =  range + tid;
        // Identify the partition that owns that node
        int part = 0;

        while (part < num_part && (global_col < part_ranges[2 * part] || global_col >= part_ranges[2 * part + 1])) { part++; }

        // Flag the corresponding node in flags array
        if (part < num_part && (global_col >= part_ranges[2 * part] && global_col < part_ranges[2 * part + 1]))
        {
            flags[flag_offsets[part] + global_col - part_ranges[2 * part]] = -col - 1;
        }

        tid += gridDim.x * blockDim.x;
    }
}


template <int coop>
__global__ void flag_halo_nodes_global_v2(INDEX_TYPE *row_offsets, int64_t *global_col_indices, int64_t *part_ranges, INDEX_TYPE *flags, INDEX_TYPE *flag_offsets, int64_t base, INDEX_TYPE range, INDEX_TYPE num_part, INDEX_TYPE num_rows)
{
    int row = blockIdx.x * blockDim.x / coop + threadIdx.x / coop;
    int coopIdx = threadIdx.x % coop;

    while (row < num_rows)
    {
        for (int i = row_offsets[row] + coopIdx; i < row_offsets[row + 1]; i += coop)
        {
            int64_t global_col = global_col_indices[i];

            if (global_col < base || global_col >= base + (int64_t) range)
            {
                int part = 0;

                while (part < num_part && (global_col < part_ranges[2 * part] || global_col >= part_ranges[2 * part + 1])) { part++; }

                if (part < num_part && (global_col >= part_ranges[2 * part] && global_col < part_ranges[2 * part + 1]))
                {
                    flags[flag_offsets[part] + (int) (global_col - part_ranges[2 * part])] = 1;
                }
            }
        }

        row += gridDim.x * blockDim.x / coop;
    }
}




template <int coop, int set_val>
__global__ void flag_halo_nodes_global(INDEX_TYPE *row_offsets, INDEX_TYPE num_rows, int64_t *global_col_indices, int64_t *part_ranges, INDEX_TYPE *flags, INDEX_TYPE *flag_offsets, int64_t base, INDEX_TYPE range, INDEX_TYPE num_neighbors, INDEX_TYPE *local_col_indices)
{
    int row = blockIdx.x * blockDim.x / coop + threadIdx.x / coop;
    int coopIdx = threadIdx.x % coop;

    while (row < num_rows)
    {
        for (int i = row_offsets[row] + coopIdx; i < row_offsets[row + 1]; i += coop)
        {
            int64_t col = global_col_indices[i];

            if (col < base || col >= base + range)
            {
                // exterior node
                int part = 0;

                while (part < num_neighbors && (col < part_ranges[2 * part] || col >= part_ranges[2 * part + 1])) { part++; }

                if (part < num_neighbors && (col >= part_ranges[2 * part] && col < part_ranges[2 * part + 1]))
                {
                    // check if the flag is already set (i.e. 1-ring node) then skip it
                    int flag = flags[flag_offsets[part] + col - part_ranges[2 * part]];

                    if (flag == 0)
                    {
                        flags[flag_offsets[part] + col - part_ranges[2 * part]] = set_val;
                    }
                    else if (flag < 0)    // it is possible that flag is already set to 1 in this kernel, so if it's negative then it's the 1st ring
                    {
                        // update local index, note that local to global mapping is unchanged for 1st ring
                        local_col_indices[i] = -flags[flag_offsets[part] + col - part_ranges[2 * part]] - 1;
                    }
                }
            }
            else
            {
                // interior node
                local_col_indices[i] = col - base;
            }
        }

        row += gridDim.x * blockDim.x / coop;
    }
}

template <int coop>
__global__ void renumber_col_indices_global(INDEX_TYPE *row_offsets, int *col_indices, int64_t *global_col_indices, int64_t *part_ranges, INDEX_TYPE *flags, INDEX_TYPE *flag_offsets, int64_t base, INDEX_TYPE range, INDEX_TYPE num_neighbors, INDEX_TYPE num_rows, int rank)
{
    int row = blockIdx.x * blockDim.x / coop + threadIdx.x / coop;
    int coopIdx = threadIdx.x % coop;

    while (row < num_rows)
    {
        for (int i = row_offsets[row] + coopIdx; i < row_offsets[row + 1]; i += coop)
        {
            int64_t col = global_col_indices[i];

            if (col < base || col >= base + range)
            {
                // Need to get his local index from the flag array
                int part = 0;

                while ( part < num_neighbors && (col < part_ranges[2 * part] || col >= part_ranges[2 * part + 1])) { part++; }

                if (part < num_neighbors && (col >= part_ranges[2 * part] && col < part_ranges[2 * part + 1]))
                {
                    int pos_in_flag = flag_offsets[part] + col - part_ranges[2 * part];
                    //int local_col = flags[pos_in_flag];
                    col_indices[i] = range + flags[pos_in_flag];
                    //bool flag = (local_col != flags[pos_in_flag+1]);

                    if (flags[pos_in_flag] == flags[pos_in_flag + 1])
                    {
                        printf("should not be here in renumber_col_indices_global\n");
                    }

                    //if (flag)
                }
            }
            else
            {
                // interior node
                col_indices[i] = col - base;
            }
        }

        row += gridDim.x * blockDim.x / coop;
    }
}



template <int coop>
__global__ void calc_new_halo_mapping(INDEX_TYPE *row_offsets, INDEX_TYPE *col_indices, int64_t *part_ranges, INDEX_TYPE num_part, INDEX_TYPE num_rows, int64_t *local_to_global, int64_t *new_local_to_global, INDEX_TYPE *halo_indices, int halo_base, int rank)
{
    int row = blockIdx.x * blockDim.x / coop + threadIdx.x / coop;
    int coopIdx = threadIdx.x % coop;

    while (row < num_rows)
    {
        for (int i = row_offsets[row] + coopIdx; i < row_offsets[row + 1]; i += coop)
        {
            INDEX_TYPE col = col_indices[i];

            if (col >= num_rows)
            {
                int64_t glob_col = local_to_global[col - num_rows];
                int part = 0;

                while (part < num_part && (glob_col < part_ranges[2 * part] || glob_col >= part_ranges[2 * part + 1])) { part++; }

                if (part < num_part && (glob_col >= part_ranges[2 * part] && glob_col < part_ranges[2 * part + 1]))
                {
                    int pos = glob_col - part_ranges[2 * part];
                    int actual_col = halo_base + halo_indices[pos];
                    col_indices[i] = -1 - actual_col;
                    new_local_to_global[actual_col - num_rows] = glob_col;
                    // when we renumber we also store mapping to use later in residual restriction since matrix R has other order of halo indices
                    // store both direct and inverse renumbering for P & R matrices
                    //halo_P_renumbering[col - num_rows] = actual_col;
                    //halo_R_renumbering[actual_col - num_rows] = col;
                }
            }
        }

        row += gridDim.x * blockDim.x / coop;
    }
}


template <int coop>
__global__ void calc_new_halo_mapping_v3(INDEX_TYPE *row_offsets, INDEX_TYPE *col_indices, int64_t *part_ranges, INDEX_TYPE num_part, INDEX_TYPE num_rows, int64_t *local_to_global, int64_t *new_local_to_global, INDEX_TYPE *halo_indices, int halo_base, int rank)
{
    int row = blockIdx.x * blockDim.x / coop + threadIdx.x / coop;
    int coopIdx = threadIdx.x % coop;

    while (row < num_rows)
    {
        for (int i = row_offsets[row] + coopIdx; i < row_offsets[row + 1]; i += coop)
        {
            INDEX_TYPE col = col_indices[i];

            if (col >= num_rows)
            {
                int64_t glob_col = local_to_global[col - num_rows];
                int part = 0;

                while (part < num_part && (glob_col < part_ranges[2 * part] || glob_col >= part_ranges[2 * part + 1])) { part++; }

                if (part < num_part && (glob_col >= part_ranges[2 * part] && glob_col < part_ranges[2 * part + 1]))
                {
                    int pos = glob_col - part_ranges[2 * part];
                    int actual_col = halo_base + halo_indices[pos];
                    col_indices[i] = -1 - actual_col;
                    new_local_to_global[actual_col - num_rows] = glob_col;
                    // when we renumber we also store mapping to use later in residual restriction since matrix R has other order of halo indices
                    // store both direct and inverse renumbering for P & R matrices
                    //halo_P_renumbering[col - num_rows] = actual_col;
                    //halo_R_renumbering[actual_col - num_rows] = col;
                }
            }
        }

        row += gridDim.x * blockDim.x / coop;
    }
}

template <int coop>
__global__ void flip_sign_col(INDEX_TYPE *row_offsets, INDEX_TYPE *col_indices, int num_rows)
{
    int row = blockIdx.x * blockDim.x / coop + threadIdx.x / coop;
    int coopIdx = threadIdx.x % coop;

    while (row < num_rows)
    {
        for (int i = row_offsets[row] + coopIdx; i < row_offsets[row + 1]; i += coop)
        {
            INDEX_TYPE col = col_indices[i];

            if (col < 0)
            {
                col_indices[i] = -col_indices[i] - 1;
            }
        }

        row += gridDim.x * blockDim.x / coop;
    }
}

template <int coop>
__global__ void calc_new_halo_mapping_ring2(INDEX_TYPE *row_offsets, INDEX_TYPE num_rows, int64_t *global_col_indices, int64_t *part_ranges, int64_t base, INDEX_TYPE range, INDEX_TYPE num_neighbors, INDEX_TYPE *halo_offsets, INDEX_TYPE *neighbor_offsets, INDEX_TYPE *neighbor_nodes, int64_t *local_to_global, INDEX_TYPE *local_col_indices)
{
    int row = blockIdx.x * blockDim.x / coop + threadIdx.x / coop;
    int coopIdx = threadIdx.x % coop;

    while (row < num_rows)
    {
        for (int i = row_offsets[row] + coopIdx; i < row_offsets[row + 1]; i += coop)
        {
            int64_t col = global_col_indices[i];

            if ((col < base || col >= base + range) && local_col_indices[i] == -1)
            {
                // update only 2nd ring halo indices here, 1st ring is updated already
                int part = 0;

                while (part < num_neighbors && (col < part_ranges[2 * part] || col >= part_ranges[2 * part + 1])) { part++; }

                if (part < num_neighbors && (col >= part_ranges[2 * part] && col < part_ranges[2 * part + 1]))
                {
                    int pos = col - part_ranges[2 * part];
                    local_col_indices[i] = halo_offsets[part] + neighbor_nodes[neighbor_offsets[part] + pos];
                    local_to_global[local_col_indices[i] - range] = col;
                }
            }
        }

        row += gridDim.x * blockDim.x / coop;
    }
}

template <int coop>
__global__ void renumber_col_indices(INDEX_TYPE *row_offsets, INDEX_TYPE *col_indices, int64_t *part_ranges, INDEX_TYPE *halo_renum, INDEX_TYPE *halo_offsets, int64_t base, INDEX_TYPE range, INDEX_TYPE num_part, INDEX_TYPE num_rows, int64_t *local_to_global_map)
{
    int row = blockIdx.x * blockDim.x / coop + threadIdx.x / coop;
    int coopIdx = threadIdx.x % coop;

    while (row < num_rows)
    {
        for (int i = row_offsets[row] + coopIdx; i < row_offsets[row + 1]; i += coop)
        {
            INDEX_TYPE col = col_indices[i];

            if (col < base || col >= base + range)
            {
                int part = 0;

                while (part < num_part && (col < part_ranges[2 * part] || col >= part_ranges[2 * part + 1])) { part++; }

                if (part < num_part && (col >= part_ranges[2 * part] && col < part_ranges[2 * part + 1]))
                {
                    int local_index = halo_renum[halo_offsets[part] + col - part_ranges[2 * part]];
                    local_to_global_map[local_index - range] = col_indices[i];
                    col_indices[i] = local_index;
                }
            }
            else
            {
                col_indices[i] = col - base;
            }
        }

        row += gridDim.x * blockDim.x / coop;
    }
}

__global__ void populate_B2L(INDEX_TYPE *indexing, INDEX_TYPE *output, INDEX_TYPE last, INDEX_TYPE size)
{
    int row = blockIdx.x * blockDim.x + threadIdx.x;

    while (row < size)
    {
        if (row == size - 1)
        {
            if (last) { output[indexing[row]] = row; }
        }
        else if (indexing[row] != indexing[row + 1])
        {
            output[indexing[row]] = row;
        }

        row += gridDim.x * blockDim.x;
    }
}

__global__ void populate_bdy_list_and_l2h_map(INDEX_TYPE *l2h_map, INDEX_TYPE *halo_nodes, INDEX_TYPE *halo_indices, INDEX_TYPE *bdy_list, INDEX_TYPE size)
{
    int row = blockIdx.x * blockDim.x + threadIdx.x;

    while (row < size)
    {
        int scanned_id = halo_indices[row];

        if (scanned_id != halo_indices[row + 1])
        {
            bdy_list[scanned_id] = row;
            l2h_map[scanned_id] = -halo_nodes[row] - 1;
        }

        row += gridDim.x * blockDim.x;
    }
}


__global__ void populate_B2L_v2(INDEX_TYPE *indexing, INDEX_TYPE *output, INDEX_TYPE offset, INDEX_TYPE size)
{
    int row = blockIdx.x * blockDim.x + threadIdx.x;

    while (row < size)
    {
        if (indexing[row] != indexing[row + 1])
        {
            output[indexing[row] - offset] = row;
        }

        row += gridDim.x * blockDim.x;
    }
}



template <int coop, class T>
__global__ void export_matrix_elements(INDEX_TYPE *row_offsets, T *values, INDEX_TYPE bsize, INDEX_TYPE *maps, INDEX_TYPE *pointers, T *output, INDEX_TYPE *col_indices, INDEX_TYPE *output2, INDEX_TYPE size)
{
    int idx = blockIdx.x * blockDim.x / coop + threadIdx.x / coop;
    int coopIdx = threadIdx.x % coop;

    while (idx < size)
    {
        int row = maps[idx];
        INDEX_TYPE src_base = row_offsets[row];
        INDEX_TYPE dst_base = pointers[idx];

        for (int m = coopIdx; m < row_offsets[row + 1]*bsize - src_base * bsize; m += coop)
        {
            output[dst_base * bsize + m] = values[src_base * bsize + m];
        }

        for (int m = coopIdx; m < row_offsets[row + 1] - src_base; m += coop)
        {
            output2[dst_base + m] = col_indices[src_base + m];
        }

        idx += gridDim.x * blockDim.x / coop;
    }
}

template <int coop, class T>
__global__ void export_matrix_elements_global(INDEX_TYPE *row_offsets, T *values, INDEX_TYPE bsize, INDEX_TYPE *maps, INDEX_TYPE *pointers, T *output, INDEX_TYPE *col_indices, int64_t *output2, INDEX_TYPE size, int64_t *local_to_global, INDEX_TYPE *q, INDEX_TYPE num_owned_pts, int64_t base_index)
{
    int idx = blockIdx.x * blockDim.x / coop + threadIdx.x / coop;
    int coopIdx = threadIdx.x % coop;

    while (idx < size)
    {
        int row = maps[idx];

        if (q != NULL)
        {
            row = q[row];
        }

        INDEX_TYPE src_base = row_offsets[row];
        INDEX_TYPE dst_base = pointers[idx];

        for (int m = coopIdx; m < row_offsets[row + 1]*bsize - src_base * bsize; m += coop)
        {
            output[dst_base * bsize + m] = values[src_base * bsize + m];
        }

        for (int m = coopIdx; m < row_offsets[row + 1] - src_base; m += coop)
        {
            int col = col_indices[src_base + m];

            if (col < num_owned_pts)
            {
                output2[dst_base + m] = (int64_t) col_indices[src_base + m] + base_index;
            }
            else
            {
                output2[dst_base + m] = local_to_global[col_indices[src_base + m] - num_owned_pts];
            }
        }

        idx += gridDim.x * blockDim.x / coop;
    }
}

template <class T>
__global__ void export_matrix_diagonal(T *values, INDEX_TYPE bsize, INDEX_TYPE *maps, T *output, INDEX_TYPE size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    while (idx < size)
    {
        int row = maps[idx];
        INDEX_TYPE src_base = row;
        INDEX_TYPE dst_base = idx;

        for (int m = 0; m < bsize; m++)
        {
            output[dst_base * bsize + m] = values[src_base * bsize + m];
        }

        idx += gridDim.x * blockDim.x;
    }
}

__global__ void write_matrix_rowsize(INDEX_TYPE *maps, INDEX_TYPE *row_offsets, INDEX_TYPE size, INDEX_TYPE *output)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    while (idx < size)
    {
        int row = maps[idx];
        output[idx] = row_offsets[row + 1] - row_offsets[row];
        idx += gridDim.x * blockDim.x;
    }
}

__global__ void find_next_ring(INDEX_TYPE *maps, INDEX_TYPE *row_offsets, INDEX_TYPE *col_indices, INDEX_TYPE size, INDEX_TYPE base, INDEX_TYPE range, INDEX_TYPE *halo_rings)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    while (idx < size)
    {
        int row = maps[idx];

        for (int col = row_offsets[row]; col < row_offsets[row + 1]; col++)
        {
            int colIdx = col_indices[col];

            if (colIdx >= base && colIdx < base + range)
            {
                halo_rings[colIdx - base] = 1;
            }
        }

        idx += gridDim.x * blockDim.x;
    }
}

__global__ void remove_previous_rings(INDEX_TYPE *map, INDEX_TYPE *halo_rings, INDEX_TYPE size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    while (idx < size)
    {
        int row = map[idx];
        halo_rings[row] = 0;
        idx += gridDim.x * blockDim.x;
    }
}

__global__ void renumber_halo_lists(INDEX_TYPE *halo_list, INDEX_TYPE *halo_renum, int64_t base, INDEX_TYPE size)
{
    int row = blockIdx.x * blockDim.x + threadIdx.x;

    while (row < size)
    {
        halo_list[row] = halo_renum[halo_list[row] - base];
        row += gridDim.x * blockDim.x;
    }
}

__global__ void write_ring_rows(INDEX_TYPE *indices, INDEX_TYPE *maps, INDEX_TYPE size)
{
    int row = blockIdx.x * blockDim.x + threadIdx.x;

    while (row < size)
    {
        int idx = indices[row];

        if (idx == 1 && row == 0)
        {
            maps[idx - 1] = row;
        }
        else if (row > 0 && idx != indices[row - 1])
        {
            maps[idx - 1] = row;
        }

        row += gridDim.x * blockDim.x;
    }
}

template <class T_Config>
void DistributedArrangerBase<T_Config>::set_part_offsets(const INDEX_TYPE num_part, const VecInt_t *part_offsets_h)
{
    part_offsets.resize(num_part + 1);
    thrust::copy(&part_offsets_h[0], &part_offsets_h[num_part + 1], part_offsets.begin());
    cudaCheckError();
    this->num_part =  num_part;
}

// TODO Make distributed reader use 64bit ints for offsets
template <class T_Config>
void DistributedArrangerBase<T_Config>::set_part_offsets(const INDEX_TYPE num_part, const int64_t *part_offsets_h)
{
    part_offsets.resize(num_part + 1);
    thrust::copy(&part_offsets_h[0], &part_offsets_h[num_part + 1], part_offsets.begin());
    cudaCheckError();
    this->num_part =  num_part;
}

template <AMGX_VecPrecision t_vecPrec, AMGX_MatPrecision t_matPrec, AMGX_IndPrecision t_indPrec>
void DistributedArranger<TemplateConfig<AMGX_device, t_vecPrec, t_matPrec, t_indPrec> >::create_B2L(Matrix_d &A,
        INDEX_TYPE my_id, INDEX_TYPE rings,
        int64_t &base_index, INDEX_TYPE &index_range,
        Vector<ivec_value_type_h> &neighbors,
        I64Vector_d &halo_ranges,
        std::vector<IVector > &B2L_maps,
        std::vector<IVector > &L2H_maps,
        std::vector<std::vector<VecInt_t> > &B2L_rings,
        DistributedComms<TConfig_d> **comms,
        std::vector<Matrix_d > **halo_rows,
        std::vector<Manager_d > **halo_btl)
{
    I64Vector_h halo_ranges_h;
    // initialize base_index, index_range, neighbors, halo_ranges_h
    this->create_neighbors(A, my_id, base_index, index_range, neighbors, halo_ranges_h);
    this->create_B2L_from_neighbors(A, my_id, rings, base_index, index_range, neighbors,
                                    halo_ranges_h, halo_ranges, B2L_maps, L2H_maps, B2L_rings, comms, halo_rows, halo_btl);
}

template <AMGX_VecPrecision t_vecPrec, AMGX_MatPrecision t_matPrec, AMGX_IndPrecision t_indPrec>
void DistributedArranger<TemplateConfig<AMGX_device, t_vecPrec, t_matPrec, t_indPrec> >::create_B2L_from_maps(Matrix_d &A,
        INDEX_TYPE my_id, INDEX_TYPE rings,
        int64_t base_index, INDEX_TYPE index_range,
        Vector<ivec_value_type_h> &neighbors,
        I64Vector_d &halo_ranges,
        std::vector<IVector > &B2L_maps,
        std::vector<std::vector<VecInt_t> > &B2L_rings,
        DistributedComms<TConfig_d> **comms,
        std::vector<Matrix_d > **halo_rows,
        std::vector<Manager_d > **halo_btl)
{
    int num_neighbors = neighbors.size();
    this->halo_coloring = (*comms)->halo_coloring;
    std::vector<IVector *> halo_lists;
    halo_lists.resize(num_neighbors);
    // initialize halo_btl
    this->create_halo_btl(A, rings, num_neighbors, neighbors, base_index, index_range, my_id, B2L_maps, B2L_rings, halo_btl, halo_lists);
    // initialize halo_rows
    this->create_halo_rows(A, rings, num_neighbors, B2L_maps, B2L_rings, halo_rows, halo_lists);

    //this->create_halo_matrices(A, 1, base_index, index_range, my_id, neighbors, B2L_maps, B2L_rings, &(*halo_rows),  &(*halo_btl));
    if (this->halo_coloring != LAST)
    {
        Matrix_d tmpA(0, 0, 0, A.getProps());
        tmpA.set_block_dimx(A.get_block_dimx());
        tmpA.set_block_dimy(A.get_block_dimy());
        tmpA.setResources(A.getResources());
        DistributedManager<TConfig_d> tmp_manager(my_id, base_index, index_range, tmpA, neighbors, halo_ranges, B2L_maps, B2L_rings, comms, NULL, NULL);
        tmpA.setManager(tmp_manager);
        tmpA.manager->getComms()->exchange_matrix_halo(**halo_rows, **halo_btl, tmpA);
    }
}

template <AMGX_VecPrecision t_vecPrec, AMGX_MatPrecision t_matPrec, AMGX_IndPrecision t_indPrec>
void DistributedArranger<TemplateConfig<AMGX_device, t_vecPrec, t_matPrec, t_indPrec> >::create_B2L_from_maps(Matrix_d &A,
        INDEX_TYPE my_id, INDEX_TYPE rings,
        Vector<ivec_value_type_h> &neighbors,
        std::vector<IVector > &B2L_maps,
        std::vector<IVector > &L2H_maps,
        std::vector<std::vector<VecInt_t> > &B2L_rings,
        DistributedComms<TConfig_d> **comms,
        std::vector<Matrix_d > **halo_rows,
        std::vector<Manager_d > **halo_btl)
{
    this->halo_coloring = (*comms)->halo_coloring;
    int num_neighbors = neighbors.size();
    int64_t base_index = 0;
    int index_range = A.get_num_rows();
    std::vector<IVector *> halo_lists;
    halo_lists.resize(num_neighbors);

    for (int i = 0; i < num_neighbors; i++)
    {
        halo_lists[i] = &L2H_maps[i];
    }

    /*
     EXAMPLE
     Say we want 2 rings of halo. The output of these function for the example partition 1 that at this point looks like:
    A.num_rows = 4; A.num_nz = 20
    A.row_offsets = [0 4 11 15 20]
    A.col_indices = [4 0 1 2
                   4 5 0 1 2 3 7
                   0 1 2 3
                   1 2 3 6 7]
    num_neighbors=2; neighbors = [0 2]
    B2L_rings[[0 2][0 2]] B2L_maps [[0 1][1 3]] (Converted from the following input: btl_sizes = [0 2 4] btl_maps = [0 1| 1 3])
    L2H_maps (and halo_lists) [[4 5][6 7]]
    halo_rows = NULL, halo_btl=NULL
    (global indices mapping to local indices: 4-0 5-1 6-2 7-3 2-4 3-5 9-6 10-7)
     */
    // initialize ring>1 B2L_maps and B2L_rings
    this->create_rings(A, rings, num_neighbors, B2L_maps, B2L_rings);
    /*
     EXAMPLE
     having mapped out two rings of halo (already had one), B2L_maps and B2L_rings look like:
     B2L_rings[[0 2 4][0 2 4]] B2L_maps[[0 1| 2 3][1 3| 0 2]]
     */
    // initialize halo_btl, creating DistributedManager objects for each halo row matrix to be sent to each neighbor, storing B2L and L2H maps
    this->create_halo_btl(A, rings, num_neighbors, neighbors, base_index, index_range, my_id, B2L_maps, B2L_rings, halo_btl, halo_lists);
    /*
     EXAMPLE
     We created to two DistributedManager objects for the two neighbors. if halo_coloring != LAST, then the relevant fields are:
     halo_btl[0] (for neighbor ID 0)
     global_id = 1; base_index=0; index_range=6 B2L_rings[0] = [0 2 4] B2L_maps[0] = [0 1| 2 3] L2H_maps = [4 5]
     halo_btl[1] (for neighbor ID 2)
     global_id = 1; base_index=0; index_range=8 B2L_rings[0] = [0 2 4] B2L_maps[0] = [1 3| 0 2] L2H_maps = [6 7]
     */
    // Extract halo rows, ring by ring
    this->create_halo_rows(A, rings, num_neighbors, B2L_maps, B2L_rings, halo_rows, halo_lists);

    /*
     EXAMPLE
     Extracted two rings of rows for the two neighbors, the two halo row matrices look like:
     halo_rows[0] (for neighbor ID 0)
     halo_rows[0].row_offsets = [0 4 11 15 20]
     halo_rows[0].col_indices = [4 0 1 2
                    4 5 0 1 2 3 7
                    0 1 2 3
                    1 2 3 6 7]

     halo_rows[1] (for neighbor ID 2)
     halo_rows[1].row_offsets = [0 7 12 16 20]
     halo_rows[1].col_indices = [4 5 0 1 2 3 7
                    1 2 3 6 7
                    4 0 1 2
                    0 1 2 3]
      Note, that the ordering of rows corresponds to the list of indices in B2L_maps[i] (and halo_btl[i].B2L_maps[0])
     */
    //this->create_halo_matrices(A, 1, base_index, index_range, my_id, neighbors, B2L_maps, B2L_rings, &(*halo_rows),  &(*halo_btl));
    if (this->halo_coloring != LAST)   //For multiple rings, we still need 0-num_rings-1 rows, just not the last one.
    {
        //Set up dummy matrix and dummy manager, and exchange the halo rows
        Matrix_d tmpA(0, 0, 0, A.getProps());
        tmpA.set_block_dimx(A.get_block_dimx());
        tmpA.set_block_dimy(A.get_block_dimy());
        tmpA.setResources(A.getResources());
        I64Vector_d tmp_halo_ranges;
        DistributedManager<TConfig_d> tmp_manager(my_id, base_index, index_range, tmpA, neighbors, tmp_halo_ranges, B2L_maps, B2L_rings, comms, NULL, NULL);
        tmpA.setManager(tmp_manager);
        tmpA.manager->getComms()->exchange_matrix_halo(**halo_rows, **halo_btl, tmpA);
    }

    /*
     EXAMPLE
     The partitions exchange their halo_rows and halo_btl fields, partition 1 ends up with:
     halo_btl[0] (received from neighbor ID 0)
     global_id = 0; base_index=0; index_range=6; B2L_rings[0] = [0 2 4]; B2L_maps[0] = [2 3| 0 1] L2H_maps = [4 5]
     halo_rows[0].row_offsets = [0 5 13 17 21]
     halo_rows[0].col_indices = [1 2 3 4 5
                      0 1 2 3 4 5 6 7
                      0 1 3 6
                      0 1 2 3]

     halo_btl[1] (received from neighbor ID 2)
     global_id = 2; base_index=0; index_range=8; B2L_rings[0] = [0 2 4]; B2L_maps[0] = [1 2| 0 3] L2H_maps = [6 7]
     halo_rows[1].row_offsets = [0 4 11 16 20]
     halo_rows[1].col_indices = [7 1 2 3
                      5 6 7 0 1 2 3
                      4 5 0 2 3
                      0 1 2 3]
     */
}

template <AMGX_VecPrecision t_vecPrec, AMGX_MatPrecision t_matPrec, AMGX_IndPrecision t_indPrec>
void DistributedArranger<TemplateConfig<AMGX_device, t_vecPrec, t_matPrec, t_indPrec> >::renumber_to_local(Matrix_d &A,
        std::vector<IVector> &boundary_lists,
        std::vector<IVector *> &halo_lists,
        INDEX_TYPE my_id,
        int64_t base_index, INDEX_TYPE index_range,
        Vector<ivec_value_type_h> &neighbors,
        I64Vector_h &halo_ranges_h,
        I64Vector_d &halo_ranges,
        IVector &halo_nodes)
{
    //IVector &halo_nodes = *halo_nodes_p;
    int size = A.get_num_rows();
    int num_blocks = min(4096, (size + 127) / 128);
    int num_neighbors = halo_ranges_h.size() / 2;
    int total_rows_of_neighbors = 0;
    std::vector<int> halo_offsets(num_neighbors + 1, 0);

    for (int i = 0; i < num_neighbors; i ++)
    {
        total_rows_of_neighbors += halo_ranges_h[2 * i + 1] - halo_ranges_h[2 * i];
        halo_offsets[i + 1] = halo_offsets[i] + halo_ranges_h[2 * i + 1] - halo_ranges_h[2 * i];
    }

    //IVector halo_nodes(total_rows_of_neighbors);
    IVector halo_offsets_d(num_neighbors + 1);
    thrust::copy(halo_offsets.begin(), halo_offsets.end(), halo_offsets_d.begin());
    int last_halo = halo_nodes[halo_nodes.size() - 1];
    //create renumbering of halo indices (contiguously neighbor - by - neighbor, indexing starting with the number of owned rows
    thrust::exclusive_scan(halo_nodes.begin(), halo_nodes.end(), halo_nodes.begin(), size);
    cudaCheckError();
    A.manager->local_to_global_map.resize(halo_nodes[halo_nodes.size() - 1] - size + last_halo);
    renumber_col_indices<16> <<< num_blocks, 128>>>(A.row_offsets.raw(), A.col_indices.raw(), halo_ranges.raw(), halo_nodes.raw(), halo_offsets_d.raw(), base_index, index_range, num_neighbors, size, A.manager->local_to_global_map.raw());
    cudaCheckError();

    for (int i =  0; i < num_neighbors; i ++)
    {
        int size = halo_lists[i]->size();
        int num_blocks = min(4096, (size + 127) / 128);
        renumber_halo_lists <<< num_blocks, 128>>>(halo_lists[i]->raw(), halo_nodes.raw() + halo_offsets[i], halo_ranges_h[2 * i], size);
    }

    cudaCheckError();
}

template <AMGX_VecPrecision t_vecPrec, AMGX_MatPrecision t_matPrec, AMGX_IndPrecision t_indPrec>
void DistributedArranger<TemplateConfig<AMGX_device, t_vecPrec, t_matPrec, t_indPrec> >::renumber_to_local(Matrix_d &A,
        std::vector<IVector> &boundary_lists, std::vector<IVector *> &halo_lists,
        INDEX_TYPE my_id, I64Vector_h &halo_ranges_h, IVector &halo_nodes)
{
    //IVector &halo_nodes = *halo_nodes_p;
    int size = A.get_num_rows();
    int num_blocks = min(4096, (size + 127) / 128);
    int num_neighbors = halo_ranges_h.size() / 2;
    int total_rows_of_neighbors = 0;
    std::vector<int> halo_offsets(num_neighbors + 1, 0);

    for (int i = 0; i < num_neighbors; i ++)
    {
        total_rows_of_neighbors += halo_ranges_h[2 * i + 1] - halo_ranges_h[2 * i];
        halo_offsets[i + 1] = halo_offsets[i] + halo_ranges_h[2 * i + 1] - halo_ranges_h[2 * i];
    }

    //IVector halo_nodes(total_rows_of_neighbors+1);
    IVector halo_offsets_d(num_neighbors + 1);
    thrust::copy(halo_offsets.begin(), halo_offsets.end(), halo_offsets_d.begin());
    int last_halo = (halo_nodes.size() > 0) ? halo_nodes[halo_nodes.size() - 1] : 0;
    //create renumbering of halo indices (contiguously neighbor - by - neighbor, indexing starting with the number of owned rows
    thrust::exclusive_scan(halo_nodes.begin(), halo_nodes.end(), halo_nodes.begin(), size);
    cudaCheckError();

    if (halo_nodes.size() > 0)
    {
        A.manager->local_to_global_map.resize(halo_nodes[halo_nodes.size() - 1] - size + last_halo);
    }

    renumber_col_indices<16> <<< num_blocks, 128>>>(A.row_offsets.raw(), A.col_indices.raw(), A.manager->halo_ranges.raw(), halo_nodes.raw(), halo_offsets_d.raw(), A.manager->base_index(), A.manager->index_range(), num_neighbors, size, A.manager->local_to_global_map.raw());
    cudaCheckError();

    for (int i =  0; i < num_neighbors; i ++)
    {
        int size = halo_lists[i]->size();
        int num_blocks = min(4096, (size + 127) / 128);
        renumber_halo_lists <<< num_blocks, 128>>>(halo_lists[i]->raw(), halo_nodes.raw() + halo_offsets[i], halo_ranges_h[2 * i], size);
    }

    cudaCheckError();
}

template <AMGX_VecPrecision t_vecPrec, AMGX_MatPrecision t_matPrec, AMGX_IndPrecision t_indPrec>
void DistributedArranger<TemplateConfig<AMGX_device, t_vecPrec, t_matPrec, t_indPrec> >::prepare_local(Matrix_d &A,
        INDEX_TYPE my_id, INDEX_TYPE rings,
        int64_t base_index, INDEX_TYPE index_range,
        Vector<ivec_value_type_h> &neighbors,
        I64Vector_h &halo_ranges_h,
        I64Vector_d &halo_ranges,
        std::vector<IVector > &L2H_maps,
        std::vector<IVector> &boundary_lists,
        std::vector<IVector *> &halo_lists,
        DistributedComms<TConfig_d> **comms)
{
    int num_neighbors = neighbors.size();
    (*comms)->set_neighbors(num_neighbors);
    IVector halo_nodes;
    // copy halo_ranges_h to device and populate boundary_lists
    this->create_boundary_lists(A, base_index, index_range, halo_ranges_h, halo_ranges, boundary_lists, halo_nodes);
    L2H_maps.resize(num_neighbors);
    halo_lists.resize(num_neighbors);

    for (int i = 0; i < num_neighbors; i++)
    {
        L2H_maps[i].resize(boundary_lists[i].size());
        thrust::transform(boundary_lists[i].begin(), boundary_lists[i].end(), thrust::constant_iterator<INDEX_TYPE > (halo_ranges_h[2 * i]), L2H_maps[i].begin(), thrust::plus<INDEX_TYPE > ());
        halo_lists[i] = &L2H_maps[i];
    }

    cudaCheckError();
    // setting up tmp manager in order to use Comm API
    Matrix_d tmpA(0, 0, 0, A.getProps());
    tmpA.set_block_dimx(A.get_block_dimx());
    tmpA.set_block_dimy(A.get_block_dimy());
    tmpA.setResources(A.getResources());
    std::vector<IVector > B2L_maps;
    std::vector<std::vector<VecInt_t> > B2L_rings;
    DistributedManager<TConfig_d> tmp_manager(my_id, base_index, index_range, tmpA, neighbors, halo_ranges, B2L_maps, B2L_rings, comms, NULL, NULL);
    tmpA.setManager(tmp_manager);
    tmpA.manager->getComms()->exchange_vectors(boundary_lists, tmpA, 0);
    this->renumber_to_local(A, boundary_lists, halo_lists, my_id, base_index, index_range, neighbors, halo_ranges_h, halo_ranges, halo_nodes);
};

template <AMGX_VecPrecision t_vecPrec, AMGX_MatPrecision t_matPrec, AMGX_IndPrecision t_indPrec>
void DistributedArranger<TemplateConfig<AMGX_device, t_vecPrec, t_matPrec, t_indPrec> >::create_B2L_from_neighbors(Matrix_d &A,
        INDEX_TYPE my_id, INDEX_TYPE rings,
        int64_t base_index, INDEX_TYPE index_range,
        Vector<ivec_value_type_h> &neighbors,
        I64Vector_h &halo_ranges_h,
        I64Vector_d &halo_ranges,
        std::vector<IVector > &B2L_maps,
        std::vector<IVector > &L2H_maps,
        std::vector<std::vector<VecInt_t> > &B2L_rings,
        DistributedComms<TConfig_d> **comms,
        std::vector<Matrix_d > **halo_rows,
        std::vector<Manager_d > **halo_btl)
{
    this->halo_coloring = (*comms)->halo_coloring;
    std::vector<IVector> boundary_lists;
    std::vector<IVector *> halo_lists;
    this->prepare_local(A, my_id, rings, base_index, index_range, neighbors, halo_ranges_h, halo_ranges, L2H_maps, boundary_lists, halo_lists, comms);
    int num_neighbors = neighbors.size();
    // initialize BTL_maps and BTl_rings
    this->create_maps(A, rings, num_neighbors, B2L_maps, B2L_rings, boundary_lists);
    // initialize halo_btl
    this->create_halo_btl(A, rings, num_neighbors, neighbors, base_index, index_range, my_id, B2L_maps, B2L_rings, halo_btl, halo_lists);
    // initialize halo_rows
    this->create_halo_rows(A, rings, num_neighbors, B2L_maps, B2L_rings, halo_rows, halo_lists);

    if (this->halo_coloring != LAST)
    {
        // setting up tmp manager in order to use Comm API
        Matrix_d tmpA(0, 0, 0, A.getProps());
        tmpA.set_block_dimx(A.get_block_dimx());
        tmpA.set_block_dimy(A.get_block_dimy());
        tmpA.setResources(A.getResources());
        DistributedManager<TConfig_d> tmp_manager(my_id, base_index, index_range, tmpA, neighbors, halo_ranges, B2L_maps, B2L_rings, comms, NULL, NULL);
        tmpA.setManager(tmp_manager);
        tmpA.manager->getComms()->exchange_matrix_halo(**halo_rows, **halo_btl, tmpA);
    }
}

template <AMGX_VecPrecision t_vecPrec, AMGX_MatPrecision t_matPrec, AMGX_IndPrecision t_indPrec>
void DistributedArranger<TemplateConfig<AMGX_device, t_vecPrec, t_matPrec, t_indPrec> >::create_neighbors(Matrix_d &A, INDEX_TYPE my_id,
        int64_t &base_index, INDEX_TYPE &index_range,
        Vector<ivec_value_type_h> &neighbors,
        I64Vector_h &halo_ranges_h)
{
    //if (this->num_part>32) FatalError("Matrix slice Level 1 module supports < 32 partitions", AMGX_ERR_NOT_IMPLEMENTED);
    if (this->num_part == 0 || this->part_offsets.size() == 0)
    {
        FatalError("Partitioning scheme is not set", AMGX_ERR_BAD_PARAMETERS);
    }

    IVector part_sizes(this->num_part, 0);
    IVector inv_neighbors(this->num_part, 0);
    int size = A.get_num_rows();
    int num_blocks = min(4096, (size + 127) / 128);
    base_index = this->part_offsets[my_id];
    index_range = this->part_offsets[my_id + 1] - this->part_offsets[my_id];
    //
    // Step 1: flag neighbors that I have an edge to
    //
    //go through all the column indices and compare it to index ranges of nodes
    calc_num_neighbors<16> <<< num_blocks, 128>>>(A.row_offsets.raw(), A.col_indices.raw(), this->part_offsets.raw(), part_sizes.raw(), this->num_part, my_id, size);
    cudaCheckError();
    std::vector<int> neighbor_flags(this->num_part);
    thrust::copy(part_sizes.begin(), part_sizes.end(), neighbor_flags.begin());
    //total number of neighbors
    int num_neighbors = thrust::reduce(neighbor_flags.begin(), neighbor_flags.end());
    cudaCheckError();
    neighbors.resize(num_neighbors);
    halo_ranges_h.resize(num_neighbors * 2);
    //initialize manager
    int active_part = 0;

    for (int i = 0; i < this->num_part; i++)
    {
        if (neighbor_flags[i] > 0)
        {
            neighbors[active_part] = i;
            halo_ranges_h[2 * active_part] = this->part_offsets[i];
            halo_ranges_h[2 * active_part + 1] = this->part_offsets[i + 1];
            active_part++;
        }
    }
}


// need for CUDA 4.2
template <AMGX_VecPrecision t_vecPrec, AMGX_MatPrecision t_matPrec, AMGX_IndPrecision t_indPrec>
void DistributedArranger<TemplateConfig<AMGX_device, t_vecPrec, t_matPrec, t_indPrec> >::create_neighbors(Matrix_d &A, INDEX_TYPE my_id,
        I64Vector_h &halo_ranges_h)
{
    if (this->num_part == 0 || this->part_offsets.size() == 0)
    {
        FatalError("Partitioning scheme is not set", AMGX_ERR_BAD_PARAMETERS);
    }

    IVector part_sizes(this->num_part, 0);
    IVector inv_neighbors(this->num_part, 0);
    int size = A.get_num_rows();
    int num_blocks = min(4096, (size + 127) / 128);
    A.manager->set_base_index(this->part_offsets[my_id]);
    A.manager->set_index_range(this->part_offsets[my_id + 1] - this->part_offsets[my_id]);
    //
    // Step 1: flag neighbors that I have an edge to
    //
    //go through all the column indices and compare it to index ranges of nodes
    calc_num_neighbors<16> <<< num_blocks, 128>>>(A.row_offsets.raw(), A.col_indices.raw(), this->part_offsets.raw(), part_sizes.raw(), this->num_part, my_id, size);
    cudaCheckError();
    std::vector<int> neighbor_flags(this->num_part);
    thrust::copy(part_sizes.begin(), part_sizes.end(), neighbor_flags.begin());
    //total number of neighbors
    int num_neighbors = thrust::reduce(neighbor_flags.begin(), neighbor_flags.end());
    cudaCheckError();
    A.manager->neighbors.resize(num_neighbors);
    halo_ranges_h.resize(num_neighbors * 2);
    //initialize manager
    int active_part = 0;

    for (int i = 0; i < this->num_part; i++)
    {
        if (neighbor_flags[i] > 0)
        {
            A.manager->neighbors[active_part] = i;
            halo_ranges_h[2 * active_part] = this->part_offsets[i];
            halo_ranges_h[2 * active_part + 1] = this->part_offsets[i + 1];
            active_part++;
        }
    }
}

template <AMGX_VecPrecision t_vecPrec, AMGX_MatPrecision t_matPrec, AMGX_IndPrecision t_indPrec>
void DistributedArranger<TemplateConfig<AMGX_device, t_vecPrec, t_matPrec, t_indPrec> >::update_neighbors_list(
    Matrix_d &A, IVector_h &neighbors, I64Vector_h &halo_ranges_h, I64Vector &halo_ranges, I64Vector_h &part_offsets_h, I64Vector &part_offsets, std::vector<IVector> &halo_rows_row_offsets, std::vector<I64Vector> &halo_rows_col_indices)
{
    int num_partitions = A.manager->get_num_partitions();
    int my_id = A.manager->global_id();
    int num_neighbors = neighbors.size();     // current number of neighbors
    int total_halo_rows = 0;
    int total_halo_nnz = 0;
    IVector neighbor_flags(num_partitions, 0);

    for (int i = 0; i < halo_rows_row_offsets.size(); i++)
    {
        int num_halo_rows = halo_rows_row_offsets[i].size() - 1;

        if (num_halo_rows > 0)
        {
            total_halo_rows += num_halo_rows;
            total_halo_nnz += halo_rows_row_offsets[i][num_halo_rows];
            int num_blocks = min(4096, (num_halo_rows + 127) / 128);
            calc_num_neighbors_v2_global<16> <<< num_blocks, 128>>>(halo_rows_row_offsets[i].raw(), halo_rows_col_indices[i].raw(),
                    part_offsets.raw(), neighbor_flags.raw(), num_partitions, my_id, num_halo_rows);
        }
    }

    cudaCheckError();
    IVector_h neighbor_flags_h = neighbor_flags;

    // unset 1-ring neighbors & myself
    for (int i = 0; i < num_neighbors; i++)
    {
        neighbor_flags_h[neighbors[i]] = 0;
    }

    neighbor_flags_h[my_id] = 0;
    // this will update neighbor list and halo ranges, note that we don't change 1-ring neighbors order
    append_neighbors(A, neighbors, halo_ranges_h, halo_ranges, neighbor_flags_h, part_offsets_h);
}

template <AMGX_VecPrecision t_vecPrec, AMGX_MatPrecision t_matPrec, AMGX_IndPrecision t_indPrec>
void DistributedArranger<TemplateConfig<AMGX_device, t_vecPrec, t_matPrec, t_indPrec> >::append_neighbors(Matrix_d &A, IVector_h &neighbors, I64Vector_h &halo_ranges_h, I64Vector &halo_ranges, IVector_h &neighbor_flags, I64Vector_h &part_offsets_h)
{
    // This function creates arrays neighbors, halo_ranges_h and halo_ranges
    // base on neighbor_flags
    // Here do an MPI_allgather to figure out which partitions need data from me
    // This is required for non-symmetric matrices
    int num_part = A.manager->get_num_partitions();
    int my_id = A.manager->global_id();
    // pack 0/1 array into array of integers (size/32)
    int packed_size = (num_part + 32 - 1) / 32;
    IVector_h packed_nf(packed_size, 0);

    for (int i = 0; i < num_part; i++)
    {
        int packed_pos = i / 32;
        int bit_pos = i % 32;
        packed_nf[packed_pos] += (neighbor_flags[i] << bit_pos);
    }

    // exchange packed neighbor flags
    IVector_h gathered_packed_nf;
    A.manager->getComms()->all_gather_v(packed_nf, gathered_packed_nf, num_part);
    // assign neighbors that have edges to me
    int my_id_pos = my_id / 32;
    int my_id_bit = my_id % 32;

    for (int i = 0; i < num_part; i++)
        if (gathered_packed_nf[i * packed_size + my_id_pos] & (1 << my_id_bit)) // check my bit
        {
            neighbor_flags[i] = 1;
        }

    // compute total number of new neighbors
    int new_neighbors = thrust::reduce(neighbor_flags.begin(), neighbor_flags.end());
    cudaCheckError();
    // save old size
    int old_neighbors = neighbors.size();
    neighbors.resize(old_neighbors + new_neighbors);
    halo_ranges_h.resize(old_neighbors * 2 + new_neighbors * 2);
    // initialize manager->neighbors and manager->halo_ranges_h for the new nodes
    int active_part = old_neighbors;
    this->num_part = num_part;

    for (int i = 0; i < num_part; i++)
    {
        if (neighbor_flags[i] > 0)
        {
            neighbors[active_part] = i;
            halo_ranges_h[2 * active_part] = part_offsets_h[i];
            halo_ranges_h[2 * active_part + 1] = part_offsets_h[i + 1];
            active_part++;
        }
    }

    halo_ranges.resize(old_neighbors * 2 + new_neighbors * 2);
    thrust::copy(halo_ranges_h.begin() + old_neighbors * 2, halo_ranges_h.end(), halo_ranges.begin() + old_neighbors * 2);
    cudaCheckError();
}


template <AMGX_VecPrecision t_vecPrec, AMGX_MatPrecision t_matPrec, AMGX_IndPrecision t_indPrec>
void DistributedArranger<TemplateConfig<AMGX_device, t_vecPrec, t_matPrec, t_indPrec> >::create_neighbors_v2(Matrix_d &A)
{
    // This function counts the number of neighbors, identifies the id of each neighbor,
    // And creates array halo_ranges, which is the range of rows for each neighbor
    // Outputs: A.manager->neighbors
    //          A.manager->halo_ranges_h
    //          A.manager->halo_ranges
    int num_part = A.manager->get_num_partitions();
    int my_id = A.manager->global_id();
    A.manager->set_base_index(A.manager->part_offsets_h[my_id]);
    A.manager->set_index_range(A.manager->part_offsets_h[my_id + 1] - A.manager->part_offsets_h[my_id]);
    IVector inv_neighbors(num_part, 0);
    IVector part_sizes(num_part, 0);
    int size = A.get_num_rows();

    if (size == 0)
    {
        A.manager->neighbors.resize(0);
    }

    // TODO: Are these the optimal block_sizes?
    int num_blocks = min(4096, (size + 127) / 128);

    // -----------------------------------------------
    // Step 1: flag neighbors that I have an edge to
    // -----------------------------------------------
    //go through all the column indices and check if they are halo vertices
    if (num_blocks != 0)
    {
        calc_num_neighbors_v2_local<16> <<< num_blocks, 128>>>(A.row_offsets.raw(), A.col_indices.raw(), A.manager->part_offsets.raw(), part_sizes.raw(), A.manager->local_to_global_map.raw(), num_part, my_id, size);
        cudaCheckError();
    }

    // D2H copy
    IVector_h neighbor_flags(num_part);
    thrust::copy(part_sizes.begin(), part_sizes.end(), neighbor_flags.begin());
    cudaCheckError();
    append_neighbors(A, A.manager->neighbors, A.manager->halo_ranges_h, A.manager->halo_ranges, neighbor_flags, A.manager->part_offsets_h);
}

template <AMGX_VecPrecision t_vecPrec, AMGX_MatPrecision t_matPrec, AMGX_IndPrecision t_indPrec>
void DistributedArranger<TemplateConfig<AMGX_device, t_vecPrec, t_matPrec, t_indPrec> >::create_neighbors_v2_global(Matrix_d &A, I64Vector_d &A_col_indices_global)
{
    // This function counts the number of neighbors, identifies the id of each neighbor,
    // And creates array halo_ranges, which is the range of rows for each neighbor
    // Outputs: A.manager->neighbors
    //          A.manager->halo_ranges_h
    //          A.manager->halo_ranges
    int num_part = A.manager->get_num_partitions();
    int my_id = A.manager->global_id();
    IVector part_sizes(num_part, 0);
    int size = A.get_num_rows();

    if (size == 0)
    {
        A.manager->neighbors.resize(0);
    }

    // TODO: Are these the optimal block_sizes?
    int num_blocks = min(4096, (size + 127) / 128);

    // -----------------------------------------------
    // Step 1: flag neighbors that I have an edge to
    // -----------------------------------------------
    //go through all the column indices and check if they are halo vertices
    if (num_blocks != 0)
    {
        calc_num_neighbors_v2_global<16> <<< num_blocks, 128>>>(A.row_offsets.raw(), A_col_indices_global.raw(), A.manager->part_offsets.raw(), part_sizes.raw(), num_part, my_id, size);
        cudaCheckError();
    }

    // D2H copy
    IVector_h neighbor_flags(num_part);
    thrust::copy(part_sizes.begin(), part_sizes.end(), neighbor_flags.begin());
    cudaCheckError();
    append_neighbors(A, A.manager->neighbors, A.manager->halo_ranges_h, A.manager->halo_ranges, neighbor_flags, A.manager->part_offsets_h);
}

template <AMGX_VecPrecision t_vecPrec, AMGX_MatPrecision t_matPrec, AMGX_IndPrecision t_indPrec>
void DistributedArranger<TemplateConfig<AMGX_device, t_vecPrec, t_matPrec, t_indPrec> >::create_B2L_one_ring(Matrix_d &A)
{
    std::vector<IVector> boundary_lists;
    // create my boundary lists, renumber halos, create L2H maps
    create_boundary_lists_v2(A, boundary_lists);
    // compute halo offsets
    int neighbors = A.manager->num_neighbors();
    A.manager->halo_offsets.resize(neighbors + 1, 0);

    for (int i = 0; i < neighbors; i++)
    {
        A.manager->halo_offsets[i] = boundary_lists[i].size();
    }

    thrust::exclusive_scan(A.manager->halo_offsets.begin(), A.manager->halo_offsets.end(), A.manager->halo_offsets.begin(), A.get_num_rows());
    cudaCheckError();
    // receive neighbors boundary lists for b2l maps
    A.manager->getComms()->exchange_vectors(boundary_lists, A, 0);
    // copy boundary lists to B2L maps
    // release internal boundary lists array
    A.manager->B2L_maps.resize(neighbors);

    for (int i = 0; i < neighbors; i++)
    {
        A.manager->B2L_maps[i] = boundary_lists[i];
    }

    A.manager->B2L_rings.resize(neighbors);

    for (int i = 0; i < neighbors; i++)
    {
        A.manager->B2L_rings[i].resize(2);
        A.manager->B2L_rings[i][0] = 0;
        A.manager->B2L_rings[i][1] = A.manager->B2L_maps[i].size();
    }

    this->initialize_B2L_maps_offsets(A, 1);
    A.manager->A = &A;
}

template <AMGX_VecPrecision t_vecPrec, AMGX_MatPrecision t_matPrec, AMGX_IndPrecision t_indPrec>
void DistributedArranger<TemplateConfig<AMGX_device, t_vecPrec, t_matPrec, t_indPrec> >::createRowsLists(Matrix_d &A, bool is_matrix_P)
{
    int A_num_rows, offset;
    A.getOffsetAndSizeForView(OWNED, &offset, &A_num_rows);
    IVector boundary_flags(A_num_rows + 1, 0);
    IVector interior_flags(A_num_rows + 1, 0);

    int num_nz_interior;
    if (!is_matrix_P)
    {
        for (int i = 0; i < A.manager->neighbors.size(); i++)
        {
            int size = A.manager->B2L_rings[i][1];
            int num_blocks = min(4096, (size + 127) / 128);

            if ( size > 0)
            {
                flag_boundary_nodes <<< num_blocks, 128>>>(A.manager->B2L_maps[i].raw(), boundary_flags.raw(), size);
            }
        }
    }
    else
    {
        const int cta_size = 256;
        const int nWarps = cta_size / 32;
        int grid_size = min( 4096, (int) (A_num_rows + nWarps - 1) / nWarps);

        if (A_num_rows > 0)
        {
            IVector nnz_interior(1, 0);
            flag_boundary_nodes_P<index_type, cta_size> <<< grid_size, cta_size>>>(A.row_offsets.raw(), A.col_indices.raw(), boundary_flags.raw(), A.manager->halo_offsets[0], A_num_rows, nnz_interior.raw());
            cudaMemcpy(&num_nz_interior, nnz_interior.raw(), sizeof(int), cudaMemcpyDefault);
        }
    }

    cudaCheckError();
    int size = A_num_rows;
    int num_blocks = min(4096, (size + 127) / 128);

    if (size > 0)
    {
        flag_interior_nodes <<< num_blocks, 128>>>(boundary_flags.raw(), interior_flags.raw(), size);
        cudaCheckError();
    }

    if (boundary_flags.size() > 1)
    {
        thrust::exclusive_scan(boundary_flags.begin(), boundary_flags.end(), boundary_flags.begin());
        cudaCheckError();
    }

    thrust::exclusive_scan(interior_flags.begin(), interior_flags.end(), interior_flags.begin());
    cudaCheckError();
    int num_interior_rows = interior_flags[size];
    int num_boundary_rows = boundary_flags[size];
    A.manager->interior_rows_list.resize(num_interior_rows);
    A.manager->boundary_rows_list.resize(num_boundary_rows);

    // Create the list of interior and boundary nodes, using the interior_flag and boundary_flag
    if (size > 0)
    {
        fill_rows_lists <<< num_blocks, 128>>>(A.manager->interior_rows_list.raw(), A.manager->boundary_rows_list.raw(), interior_flags.raw(), boundary_flags.raw(), size);
        cudaCheckError();
    }

    // Manually set the view sizes for P
    if(is_matrix_P)
    {
        int num_rows_owned = num_interior_rows + num_boundary_rows;
        int num_nz_owned = A.row_offsets[num_rows_owned];
        A.manager->setViewSizes(num_interior_rows, num_nz_interior, num_rows_owned, num_nz_owned, num_rows_owned, num_nz_owned, A.get_num_rows(), A.get_num_nz());
    }
}

template <AMGX_VecPrecision t_vecPrec, AMGX_MatPrecision t_matPrec, AMGX_IndPrecision t_indPrec>
void DistributedArranger<TemplateConfig<AMGX_device, t_vecPrec, t_matPrec, t_indPrec> >::create_boundary_lists_v3(Matrix_d &A)
{
    // -----------------------------------------------
    // Flag all nodes in neighbors that I will need (that is I have an edge pointing to it)
    // -----------------------------------------------
    int A_num_rows = A.get_num_rows();

    if (A_num_rows == 0)
    {
        return;
    }

    int num_neighbors = A.manager->neighbors.size();
    // compute halo offsets
    int total_rows_of_neighbors = 0;
    std::vector<int> halo_offsets(num_neighbors + 1, 0);
    int max_neighbor_size = 0;

    for (int i = 0; i < num_neighbors; i ++)
    {
        int num_rows_neighbor = A.manager->halo_ranges_h[2 * i + 1] - A.manager->halo_ranges_h[2 * i];
        total_rows_of_neighbors += num_rows_neighbor;
        halo_offsets[i + 1] = halo_offsets[i] + num_rows_neighbor;
        max_neighbor_size = max_neighbor_size > num_rows_neighbor ? max_neighbor_size : num_rows_neighbor;;
    }

    // copy offsets to device
    IVector halo_offsets_d(num_neighbors + 1);
    thrust::copy(halo_offsets.begin(), halo_offsets.end(), halo_offsets_d.begin());
    // flag neighbors nodes that I need
    IVector halo_nodes(total_rows_of_neighbors + 1);
    thrust::fill(halo_nodes.begin(), halo_nodes.end(), 0);
    cudaCheckError();
    int size = A.manager->local_to_global_map.size();
    int num_blocks = min(4096, (size + 127) / 128);

    if (size > 0)
    {
        flag_halo_nodes_from_local_to_global_map <<< num_blocks, 128>>>(A.manager->halo_ranges.raw(), halo_nodes.raw(), halo_offsets_d.raw(), A.manager->base_index(), A.manager->index_range(), num_neighbors, size, A.manager->local_to_global_map.raw());
        cudaCheckError();
    }

    IVector halo_indices(max_neighbor_size + 1);
    A.manager->L2H_maps.resize(num_neighbors);
    A.manager->B2L_maps.resize(num_neighbors);
    is_less_than_zero pred;

    for (int i = 0; i < num_neighbors; i++)
    {
        // compute halo indices map [<neighbor inner node>] = 0-based halo number
        int size = halo_offsets[i + 1] - halo_offsets[i];
        thrust::replace_copy_if(halo_nodes.begin() + halo_offsets[i], halo_nodes.begin() + halo_offsets[i + 1], halo_indices.begin(), pred, 1);
        thrust::exclusive_scan(halo_indices.begin(), halo_indices.begin() + size + 1, halo_indices.begin());
        cudaCheckError();
        int num_halo = halo_indices[size];
        A.manager->L2H_maps[i].resize(num_halo);
        A.manager->B2L_maps[i].resize(num_halo);
        // create my boundary lists = list of neighbor inner nodes corresponding to halo numbers 0..M
        // For each node in B2L_map, need to find corresponding local id and store in L2H_map
        int num_blocks = min(4096, (size + 127) / 128);
        populate_bdy_list_and_l2h_map <<< num_blocks, 128>>>(A.manager->L2H_maps[i].raw(),  halo_nodes.raw() + halo_offsets[i], halo_indices.raw(), A.manager->B2L_maps[i].raw(), size);
        cudaCheckError();
    }
}


template <AMGX_VecPrecision t_vecPrec, AMGX_MatPrecision t_matPrec, AMGX_IndPrecision t_indPrec>
void DistributedArranger<TemplateConfig<AMGX_device, t_vecPrec, t_matPrec, t_indPrec> >::initialize_B2L_maps_offsets(Matrix_d &A, int num_rings)
{
    // Compute the total number of bdy rows in each ring
    std::vector<IVector_h> B2L_maps_offsets_h(num_rings);
    A.manager->B2L_rings_sizes.resize(num_rings);
    int num_neighbors = A.manager->neighbors.size();

    for (int k = 0; k < num_rings; k++)
    {
        B2L_maps_offsets_h[k].resize(num_neighbors + 1);;
        B2L_maps_offsets_h[k][0] = 0;

        for (int j = 0; j < num_neighbors; j++)
        {
            B2L_maps_offsets_h[k][j + 1] = B2L_maps_offsets_h[k][j] + A.manager->B2L_rings[j][k + 1];
        }

        A.manager->B2L_rings_sizes[k] = B2L_maps_offsets_h[k][num_neighbors];
    }

    // Copy maps_offsets to device
    A.manager->B2L_maps_offsets.resize(num_rings);

    for (int i = 0; i < num_rings; i++)
    {
        A.manager->B2L_maps_offsets[i] = B2L_maps_offsets_h[i];
    }

    // Store the B2L_maps ptrs on the device
    std::vector<int *> B2L_maps_ptrs_h(num_neighbors);

    for (int j = 0; j < num_neighbors; j++)
    {
        B2L_maps_ptrs_h[j] = A.manager->B2L_maps[j].raw();
    }

    A.manager->B2L_maps_ptrs = B2L_maps_ptrs_h;
}


template <AMGX_VecPrecision t_vecPrec, AMGX_MatPrecision t_matPrec, AMGX_IndPrecision t_indPrec>
void DistributedArranger<TemplateConfig<AMGX_device, t_vecPrec, t_matPrec, t_indPrec> >::create_one_ring_halo_rows(Matrix_d &A)
{
    // what it does:
    //   appends halo rows to matrix A
    //   creates new B2L_rings, B2L_maps and L2H_maps
    // input:
    //   matrix A and 1-ring B2L_maps
    int num_partitions = A.manager->get_num_partitions();
    int my_id = A.manager->global_id();
    int num_neighbors = A.manager->B2L_maps.size();
    int num_ring1_neighbors = num_neighbors;
    int num_ring1_indices = A.manager->local_to_global_map.size();
    std::vector<IVector> halo_row_offsets(num_neighbors);
    std::vector<I64Vector> halo_global_indices(num_neighbors);
    std::vector<IVector> halo_local_indices(num_neighbors);
    std::vector<MVector> halo_values(num_neighbors);
    // step 1: setup halo rows with global indices
    create_halo_rows_global_indices(A, halo_row_offsets, halo_global_indices, halo_values);
    // step 2: exchange halo rows
    std::vector<I64Vector> dummy_halo_ids(0);
    A.manager->getComms()->exchange_matrix_halo(halo_row_offsets, halo_global_indices, halo_values, dummy_halo_ids, A.manager->neighbors, A.manager->global_id());
    // step 3: append the list  f neighbors with the new ring-2 neighbors
    update_neighbors_list(A, A.manager->neighbors, A.manager->halo_ranges_h, A.manager->halo_ranges, A.manager->part_offsets_h, A.manager->part_offsets, halo_row_offsets, halo_global_indices);
    num_neighbors = A.manager->neighbors.size();
    // step 4: mark neighbor nodes, create local halo indices for the 2nd ring
    // this function also creates boundary lists for the 2nd ring if boundary lists array size > 0
    std::vector<IVector> boundary_lists(num_neighbors);
    compute_local_halo_indices(A.row_offsets, A.col_indices, halo_row_offsets, halo_global_indices, halo_local_indices, A.manager->local_to_global_map, boundary_lists,  A.manager->neighbors, A.manager->halo_ranges_h, A.manager->halo_ranges, A.manager->halo_offsets, A.manager->base_index(), A.manager->index_range(), A.get_num_rows(), 1);
    // update renumbering arrays (set identity for 2nd ring)
    //A.manager->halo_P_renumbering.resize(A.manager->local_to_global_map.size());
    //A.manager->halo_R_renumbering.resize(A.manager->local_to_global_map.size());
    //thrust::sequence(A.manager->halo_P_renumbering.begin() + num_ring1_indices, A.manager->halo_P_renumbering.end(), num_ring1_indices + A.get_num_rows());
    //thrust::sequence(A.manager->halo_R_renumbering.begin() + num_ring1_indices, A.manager->halo_R_renumbering.end(), num_ring1_indices + A.get_num_rows());
    // step 5: update L2H maps = identity
    A.manager->getComms()->set_neighbors(A.manager->neighbors.size());
    // step 6: receive neighbors boundary lists for 2-ring b2l maps
    A.manager->getComms()->exchange_vectors(boundary_lists, A, 0);
    // step 7: update B2L rings & B2L maps
    int rings = 2;
    A.manager->B2L_rings.resize(num_neighbors);
    A.manager->B2L_maps.resize(num_neighbors);
    // modify 1-ring where necessary
    int ring = 0;

    for (int i = 0; i < num_neighbors; i++)
    {
        // set size to 0 for 2-ring neighbors only
        if (i >= num_ring1_neighbors)
        {
            A.manager->B2L_maps[i].resize(0);
        }

        A.manager->B2L_rings[i].resize(rings + 1);
        A.manager->B2L_rings[i][0] = 0;
        A.manager->B2L_rings[i][ring + 1] = A.manager->B2L_maps[i].size();
    }

    // fill up 2-ring maps
    ring = 1;

    for (int i = 0; i < num_neighbors; i++)
    {
        // append 2nd ring
        int ring1_size = A.manager->B2L_maps[i].size();
        A.manager->B2L_maps[i].resize(ring1_size + boundary_lists[i].size());
        thrust::copy(boundary_lists[i].begin(), boundary_lists[i].end(), A.manager->B2L_maps[i].begin() + ring1_size);
        A.manager->B2L_rings[i][ring + 1] = A.manager->B2L_maps[i].size();
    }

    cudaCheckError();
    this->initialize_B2L_maps_offsets(A, 2);
    // step 8: append halo_rows to matrix A
    // compute new # of rows & nnz
    this->append_halo_rows(A, halo_row_offsets, halo_local_indices, halo_values);
    // initialize the manager
    A.manager->set_initialized(A.row_offsets);
    // Compute the diagonal
    // TODO: Should only compute diagonal of 1-ring halo rows
    ViewType oldView = A.currentView();
    A.setView(FULL);
    A.set_allow_recompute_diag(true);
    A.computeDiagonal();
    A.setView(oldView);
    // the following steps are necessary only for latency hiding/renumbering, i.e. to use reorder_matrix()
#if 0
    // step 6: for 2 rings: setup halo_btl
    create_halo_btl_multiple_rings(A, 2);
#endif
}

// need for CUDA 4.2
template <AMGX_VecPrecision t_vecPrec, AMGX_MatPrecision t_matPrec, AMGX_IndPrecision t_indPrec>
void DistributedArranger<TemplateConfig<AMGX_device, t_vecPrec, t_matPrec, t_indPrec> >::create_boundary_lists(Matrix_d &A,
        int64_t base_index, INDEX_TYPE index_range,
        I64Vector_h &halo_ranges_h, I64Vector_d &halo_ranges,
        std::vector<IVector> &boundary_lists, IVector &halo_nodes)
{
    //
    // Step 2: flag all nodes in neighbors that I will need (that is I have an edge pointing to it)
    //
    int size = A.get_num_rows();
    int num_blocks = min(4096, (size + 127) / 128);
    int num_neighbors = halo_ranges_h.size() / 2;
    halo_ranges.resize(num_neighbors * 2);
    thrust::copy(halo_ranges_h.begin(), halo_ranges_h.end(), halo_ranges.begin());
    cudaCheckError();
    boundary_lists.resize(num_neighbors);
    int total_rows_of_neighbors = 0;
    std::vector<int> halo_offsets(num_neighbors + 1, 0);
    int max_neighbor_size = 0;

    for (int i = 0; i < num_neighbors; i ++)
    {
        total_rows_of_neighbors += halo_ranges_h[2 * i + 1] - halo_ranges_h[2 * i];
        halo_offsets[i + 1] = halo_offsets[i] + halo_ranges_h[2 * i + 1] - halo_ranges_h[2 * i];
        max_neighbor_size = max_neighbor_size > (halo_ranges_h[2 * i + 1] - halo_ranges_h[2 * i]) ? max_neighbor_size : (halo_ranges_h[2 * i + 1] - halo_ranges_h[2 * i]);;
    }

    IVector halo_offsets_d(num_neighbors + 1);
    thrust::copy(halo_offsets.begin(), halo_offsets.end(), halo_offsets_d.begin());
    halo_nodes.resize(total_rows_of_neighbors);
    thrust::fill(halo_nodes.begin(), halo_nodes.end(), 0);
    cudaCheckError();
    flag_halo_nodes<16> <<< num_blocks, 128>>>(A.row_offsets.raw(), A.col_indices.raw(), halo_ranges.raw(), halo_nodes.raw(), halo_offsets_d.raw(), base_index, index_range, num_neighbors, size);
    cudaCheckError();
    //
    // Step 3: Write the 1-ring halo indices for each neighbor
    //
    IVector halo_indices(max_neighbor_size);

    for (int i = 0; i < num_neighbors; i++)
    {
        int last = halo_nodes[halo_offsets[i + 1] - 1];
        thrust::exclusive_scan(halo_nodes.begin() + halo_offsets[i], halo_nodes.begin() + halo_offsets[i + 1], halo_indices.begin());
        int num_halo = halo_indices[halo_offsets[i + 1] - halo_offsets[i] - 1] + last;
        //A.manager.B2L_maps[i].resize(num_halo);
        boundary_lists[i].resize(num_halo);
        int size = halo_offsets[i + 1] - halo_offsets[i];
        //int num_blocks=min(4096,(size+127)/128);
        populate_B2L <<< num_blocks, 128>>>(halo_indices.raw(), boundary_lists[i].raw(),/*A.manager.B2L_maps[i].raw(),*/ last, size);
    }

    cudaCheckError();
}

template <AMGX_VecPrecision t_vecPrec, AMGX_MatPrecision t_matPrec, AMGX_IndPrecision t_indPrec>
void DistributedArranger<TemplateConfig<AMGX_device, t_vecPrec, t_matPrec, t_indPrec> >::create_boundary_lists(Matrix_d &A,
        I64Vector_h &halo_ranges_h,
        std::vector<IVector> &boundary_lists, IVector &halo_nodes)
{
    //
    // Step 2: flag all nodes in neighbors that I will need (that is I have an edge pointing to it)
    //
    int size = A.get_num_rows();
    int num_blocks = min(4096, (size + 127) / 128);
    int num_neighbors = halo_ranges_h.size() / 2;
    A.manager->halo_ranges.resize(num_neighbors * 2);
    thrust::copy(halo_ranges_h.begin(), halo_ranges_h.end(), A.manager->halo_ranges.begin());
    cudaCheckError();
    boundary_lists.resize(num_neighbors);
    int total_rows_of_neighbors = 0;
    std::vector<int> halo_offsets(num_neighbors + 1, 0);
    int max_neighbor_size = 0;

    for (int i = 0; i < num_neighbors; i ++)
    {
        total_rows_of_neighbors += halo_ranges_h[2 * i + 1] - halo_ranges_h[2 * i];
        halo_offsets[i + 1] = halo_offsets[i] + halo_ranges_h[2 * i + 1] - halo_ranges_h[2 * i];
        max_neighbor_size = max_neighbor_size > (halo_ranges_h[2 * i + 1] - halo_ranges_h[2 * i]) ? max_neighbor_size : (halo_ranges_h[2 * i + 1] - halo_ranges_h[2 * i]);;
    }

    IVector halo_offsets_d(num_neighbors + 1);
    thrust::copy(halo_offsets.begin(), halo_offsets.end(), halo_offsets_d.begin());
    halo_nodes.resize(total_rows_of_neighbors);;
    thrust::fill(halo_nodes.begin(), halo_nodes.end(), 0);
    cudaCheckError();
    flag_halo_nodes<16> <<< num_blocks, 128>>>(A.row_offsets.raw(), A.col_indices.raw(), A.manager->halo_ranges.raw(), halo_nodes.raw(), halo_offsets_d.raw(), A.manager->base_index(), A.manager->index_range(), num_neighbors, size);
    cudaCheckError();
    //
    // Step 3: Write the 1-ring halo indices for each neighbor
    //
    IVector halo_indices(max_neighbor_size);

    for (int i = 0; i < num_neighbors; i++)
    {
        int last = halo_nodes[halo_offsets[i + 1] - 1];
        thrust::exclusive_scan(halo_nodes.begin() + halo_offsets[i], halo_nodes.begin() + halo_offsets[i + 1], halo_indices.begin());
        int num_halo = halo_indices[halo_offsets[i + 1] - halo_offsets[i] - 1] + last;
        //A.manager.B2L_maps[i].resize(num_halo);
        boundary_lists[i].resize(num_halo);
        int size = halo_offsets[i + 1] - halo_offsets[i];
        //int num_blocks=min(4096,(size+127)/128);
        populate_B2L <<< num_blocks, 128>>>(halo_indices.raw(), boundary_lists[i].raw(),/*A.manager.B2L_maps[i].raw(),*/ last, size);
    }

    cudaCheckError();
}

template <AMGX_VecPrecision t_vecPrec, AMGX_MatPrecision t_matPrec, AMGX_IndPrecision t_indPrec>
void DistributedArranger<TemplateConfig<AMGX_device, t_vecPrec, t_matPrec, t_indPrec> >::create_boundary_lists_v2(Matrix_d &A, std::vector<IVector> &boundary_lists)
{
    // -----------------------------------------------
    // Flag all nodes in neighbors that I will need (that is I have an edge pointing to it)
    // -----------------------------------------------
    int size = A.get_num_rows();

    if (size == 0)
    {
        return;
    }

    //TODO: Are these the optimal block_sizes?
    int num_blocks = min(4096, (size + 127) / 128);
    int num_neighbors = A.manager->neighbors.size();
    boundary_lists.resize(num_neighbors);
    // compute halo offsets
    int total_rows_of_neighbors = 0;
    std::vector<int> halo_offsets(num_neighbors + 1, 0);
    int max_neighbor_size = 0;

    for (int i = 0; i < num_neighbors; i ++)
    {
        int num_rows_neighbor = A.manager->halo_ranges_h[2 * i + 1] - A.manager->halo_ranges_h[2 * i];
        total_rows_of_neighbors += num_rows_neighbor;
        halo_offsets[i + 1] = halo_offsets[i] + num_rows_neighbor;
        max_neighbor_size = max_neighbor_size > num_rows_neighbor ? max_neighbor_size : num_rows_neighbor;;
    }

    // copy offsets to device
    IVector halo_offsets_d(num_neighbors + 1);
    thrust::copy(halo_offsets.begin(), halo_offsets.end(), halo_offsets_d.begin());
    // flag neighbors nodes that I need
    IVector halo_nodes(total_rows_of_neighbors);
    thrust::fill(halo_nodes.begin(), halo_nodes.end(), 0);
    cudaCheckError();
    flag_halo_nodes_local<16, 1> <<< num_blocks, 128>>>(A.row_offsets.raw(), A.col_indices.raw(), A.manager->halo_ranges.raw(), halo_nodes.raw(), halo_offsets_d.raw(), A.manager->base_index(), A.manager->index_range(), num_neighbors, size, A.manager->local_to_global_map.raw());
    cudaCheckError();
    // store renumbering so later can use on P
    //A.manager->halo_P_renumbering.resize(A.manager->local_to_global_map.size());
    //A.manager->halo_R_renumbering.resize(A.manager->local_to_global_map.size());
    // we need to update local toglobal maps as well because we will be renumbering halo nodes
    I64Vector new_local_to_global_map(A.manager->local_to_global_map.size(), -1);
    // compute new local halo indices for each neighbor node
    // renumbering: compute new column indices sorted by global index
    IVector halo_indices(max_neighbor_size);
    std::vector<int> num_halo(num_neighbors);
    std::vector<int> last(num_neighbors);
    std::vector<IVector *> halo_lists;
    halo_lists.resize(num_neighbors);
    A.manager->L2H_maps.resize(num_neighbors);
    int halo_base = size;

    for (int i = 0; i < num_neighbors; i++)
    {
        // compute halo indices map [<neighbor inner node>] = 0-based halo number
        last[i] = halo_nodes[halo_offsets[i + 1] - 1];
        thrust::exclusive_scan(halo_nodes.begin() + halo_offsets[i], halo_nodes.begin() + halo_offsets[i + 1], halo_indices.begin());
        num_halo[i] = halo_indices[halo_offsets[i + 1] - halo_offsets[i] - 1] + last[i];
        // renumber my columns for this particular neighbor
        calc_new_halo_mapping<16> <<< num_blocks, 128>>>(A.row_offsets.raw(), A.col_indices.raw(), A.manager->halo_ranges.raw() + (int64_t) (2 * i), 1, size, A.manager->local_to_global_map.raw(), new_local_to_global_map.raw(), halo_indices.raw(), halo_base, A.manager->global_id());
        //calc_new_halo_mapping_v2<16><<<num_blocks, 128>>>( A.manager->halo_ranges.raw() + (int64_t) (2*i), 1, size, A.manager->local_to_global_map.raw(), new_local_to_global_map.raw(), halo_indices.raw(), halo_base, A.manager->local_to_global_map.size(),A.manager->global_id());
        // create my boundary lists = list of neighbor inner nodes corresponding to halo numbers 0..M
        boundary_lists[i].resize(num_halo[i]);
        int size = halo_offsets[i + 1] - halo_offsets[i];
        populate_B2L <<< num_blocks, 128>>>(halo_indices.raw(), boundary_lists[i].raw(), last[i], size);
        // identity L2H
        A.manager->L2H_maps[i].resize(num_halo[i]);
        thrust::sequence(A.manager->L2H_maps[i].begin(), A.manager->L2H_maps[i].end(), halo_base);
        halo_lists[i] = &A.manager->L2H_maps[i];
        // continous numbering for halo indices
        halo_base += num_halo[i];
    }

    cudaCheckError();
    // Now change the negative col to positive
    flip_sign_col<16> <<< num_blocks, 128>>>(A.row_offsets.raw(), A.col_indices.raw(), size);
    cudaCheckError();
    // copy new local to global mapping
    thrust::copy(new_local_to_global_map.begin(), new_local_to_global_map.end(), A.manager->local_to_global_map.begin());
    cudaCheckError();
}

template <AMGX_VecPrecision t_vecPrec, AMGX_MatPrecision t_matPrec, AMGX_IndPrecision t_indPrec>
void DistributedArranger<TemplateConfig<AMGX_device, t_vecPrec, t_matPrec, t_indPrec> >::compute_local_halo_indices( IVector &A_row_offsets, IVector &A_col_indices, std::vector<IVector> &halo_row_offsets, std::vector<I64Vector> &halo_global_indices,
        std::vector<IVector> &halo_local_indices, I64Vector &local_to_global, std::vector<IVector> &boundary_lists, IVector_h &neighbors, I64Vector_h &halo_ranges_h, I64Vector &halo_ranges, IVector_h &halo_offsets, int64_t base_index, int64_t index_range, int A_num_rows, int current_num_rings)
{
    // This function checks the halo_col_indices received from the neighbors, and identifies
    // new halo_indices and  updates halo_offsets, local_to_global_map accordingly
    // input: halo row offsets,
    //        halo global column indices
    //        current local to global map
    //        neighbors (new discovered neighbors should already be included)
    //        halo_ranges for the neighbors
    //
    // output: halo offsets,
    //         halo local column indices,
    //         updated local to global map (in place)
    int size = A_num_rows;
    //TODO: Are these the optimal block_sizes?
    int num_blocks = min(4096, (size + 127) / 128);
    int num_neighbors = neighbors.size();
    // compute neighbor offsets & total number of neighbor rows
    int total_rows_of_neighbors = 0;
    std::vector<int> neighbor_offsets_h(num_neighbors + 1, 0);
    int max_neighbor_size = 0;

    for (int i = 0; i < num_neighbors; i ++)
    {
        total_rows_of_neighbors += halo_ranges_h[2 * i + 1] - halo_ranges_h[2 * i];
        neighbor_offsets_h[i + 1] = neighbor_offsets_h[i] + halo_ranges_h[2 * i + 1] - halo_ranges_h[2 * i];
        max_neighbor_size = max_neighbor_size > (halo_ranges_h[2 * i + 1] - halo_ranges_h[2 * i]) ? max_neighbor_size : (halo_ranges_h[2 * i + 1] - halo_ranges_h[2 * i]);
    }

    // copy offsets to device
    IVector neighbor_offsets(num_neighbors + 1);
    thrust::copy(neighbor_offsets_h.begin(), neighbor_offsets_h.end(), neighbor_offsets.begin());
    // store flags for all neighbor nodes
    IVector neighbor_nodes(total_rows_of_neighbors);
    // flag neighbor nodes that are in the existing rings as -(local_index)-1
    thrust::fill(neighbor_nodes.begin(), neighbor_nodes.end(), 0);
    cudaCheckError();
    int local_to_global_size = local_to_global.size();
    int num_blocks2 = min(4096, (local_to_global_size + 127) / 128);

    if (local_to_global_size != 0)
    {
        flag_halo_nodes_local_v2 <<< num_blocks2, 128>>>(halo_ranges.raw(), neighbor_nodes.raw(), neighbor_offsets.raw(), base_index, index_range, num_neighbors, local_to_global_size, local_to_global.raw());
        cudaCheckError();
    }

    // 1) flag NEW neighbors nodes that I need as 1, they will be in the 2nd ring
    // 2) fill out local indices for 1st ring & internal indices
    int num_halos = halo_row_offsets.size();

    for (int i = 0; i < num_halos; i++)
    {
        int size = halo_global_indices[i].size();

        if (size > 0)
        {
            halo_local_indices[i].resize(size);
            thrust::fill(halo_local_indices[i].begin(), halo_local_indices[i].end(), -1); // fill with -1
            // TODO: launch only on halo rows
            flag_halo_nodes_global<16, 1> <<< num_blocks, 128>>>(halo_row_offsets[i].raw(), halo_row_offsets[i].size() - 1, halo_global_indices[i].raw(), halo_ranges.raw(), neighbor_nodes.raw(), neighbor_offsets.raw(), base_index, index_range, num_neighbors, halo_local_indices[i].raw());
        }
    }

    cudaCheckError();
    // replace all negative values with 0 in neighbor flags
    is_less_than_zero pred;
    thrust::replace_if(neighbor_nodes.begin(), neighbor_nodes.end(), pred, 0);
    cudaCheckError();
    // fill halo offsets for the current number of  ring for new neighbors (it will be of size 0)
    int current_num_neighbors = (halo_offsets.size() - 1) / current_num_rings;;
    int num_rings = current_num_rings + 1;
    IVector_h new_halo_offsets(num_rings * num_neighbors + 1);

    for (int j = 0; j < current_num_rings; j++)
    {
        for (int i = 0 ; i <= current_num_neighbors; i++)
        {
            new_halo_offsets[num_neighbors * j + i] = halo_offsets[current_num_neighbors * j + i];
        }

        for (int i = current_num_neighbors; i < num_neighbors; i++)
        {
            new_halo_offsets[num_neighbors * j + i + 1] = new_halo_offsets[num_neighbors * j + i];
        }
    }

    halo_offsets = new_halo_offsets;
    int ring = current_num_rings;
    int current_num_halo_indices = local_to_global.size();
    //int halo_base = index_range + current_num_halo_indices;
    cudaCheckError();

    // compute neighbors nodes indices (in-place) for each neighbor
    for (int i = 0; i < num_neighbors; i++)
    {
        int last_node, num_halo;

        if (neighbor_offsets_h[i + 1] != neighbor_offsets_h[i])
        {
            last_node = neighbor_nodes[neighbor_offsets_h[i + 1] - 1];
            thrust::exclusive_scan(neighbor_nodes.begin() + neighbor_offsets_h[i], neighbor_nodes.begin() + neighbor_offsets_h[i + 1], neighbor_nodes.begin() + neighbor_offsets_h[i]);
            num_halo = neighbor_nodes[neighbor_offsets_h[i + 1] - 1] + last_node;
        }
        else
        {
            num_halo = 0;
            last_node = 0;
        }

        // update halo offsets (L2H)
        halo_offsets[ring * num_neighbors + i + 1] = halo_offsets[ring * num_neighbors + i] + num_halo;

        // if size = 0 then we don't need to compute it
        if (boundary_lists.size() > 0)
        {
            // create my boundary lists = list of neighbor inner nodes corresponding to halo numbers 0..M
            // basically this will be our 2-ring B2L_maps
            boundary_lists[i].resize(num_halo);
            int size = neighbor_offsets_h[i + 1] - neighbor_offsets_h[i];
            num_blocks = min(4096, (size + 127) / 128);

            if (size > 0)
            {
                populate_B2L <<< num_blocks, 128>>>(neighbor_nodes.raw() + neighbor_offsets_h[i], boundary_lists[i].raw(), last_node, size);
                cudaCheckError();
            }
        }
    }

    cudaCheckError();
    // compute local indices and new local to global mapping
    int new_num_halo_indices = halo_offsets[num_rings * num_neighbors] - halo_offsets[current_num_rings * num_neighbors];
    local_to_global.resize(current_num_halo_indices + new_num_halo_indices);
    IVector halo_offsets_d(halo_offsets.size());
    thrust::copy(halo_offsets.begin(), halo_offsets.end(), halo_offsets_d.begin());
    cudaCheckError();

    // do this for all ring-1 neighbors
    for (int i = 0; i < num_halos; i++)
    {
        int num_neighbor_rows = halo_row_offsets[i].size() - 1;
        num_blocks = min(4096, (num_neighbor_rows + 127) / 128);

        if (num_blocks > 0)
        {
            calc_new_halo_mapping_ring2<16> <<< num_blocks, 128>>>(
                halo_row_offsets[i].raw(), num_neighbor_rows, halo_global_indices[i].raw(),                     // halo rows to process
                halo_ranges.raw(), base_index, index_range, num_neighbors,                          // ranges and # of neighbors
                halo_offsets_d.raw() + current_num_rings * num_neighbors, neighbor_offsets.raw(), neighbor_nodes.raw(),     // halo offsets, neighbor offsets and indices
                local_to_global.raw(), halo_local_indices[i].raw());                                    // output
        }
    }

    cudaCheckError();
}


template <AMGX_VecPrecision t_vecPrec, AMGX_MatPrecision t_matPrec, AMGX_IndPrecision t_indPrec>
void DistributedArranger<TemplateConfig<AMGX_device, t_vecPrec, t_matPrec, t_indPrec> >::create_rings(Matrix_d &A, INDEX_TYPE rings,
        INDEX_TYPE neighbors,
        std::vector<IVector > &B2L_maps,
        std::vector<std::vector<VecInt_t> > &B2L_rings)
{
    int size = A.get_num_rows();
    int num_blocks = min(4096, (size + 127) / 128);
    B2L_rings.resize(neighbors);

    for (int i = 0; i < neighbors; i ++)
    {
        B2L_rings[i].resize(rings + 1);
        B2L_rings[i][0] = 0;
    }

    /*
     EXAMPLE
     For example matrix. partition 1, the matrix look like:
     A.row_offsets = [0 4 11 15 20]
     A.col_indices = [4 0 1 2
                    4 5 0 1 2 3 7
                    0 1 2 3
                    1 2 3 6 7]
     num_neighbors=2; neighbors = [0 2]
     B2L_rings[[0 2][0 2]] B2L_maps [[0 1][1 3]] (Converted from the following input: btl_sizes = [0 2 4] btl_maps = [0 1| 1 3])
     L2H_maps (and halo_lists) [[4 5][6 7]]
     and we want to map further halo rings
     */
    //Scratch space computation
    for (int i = 0; i < neighbors; i++)
    {
        B2L_rings[i][1] = B2L_maps[i].size();
    }

    //flag array
    IVector halo_rings(A.get_num_rows());

    for (int i = 0; i < neighbors; i++)
    {
        //
        // For each ring: flag nodes that are neighbors of the previous ring, then remove the nodes of the previous rings from the marking
        // count the number of marked nodes, and append their row indices to the B2L_maps
        //
        for (int ring = 1; ring < rings; ring++)
        {
            size = B2L_rings[i][ring] - B2L_rings[i][ring - 1];
            thrust::fill(halo_rings.begin(), halo_rings.begin() + size + 1, 0);
            num_blocks = min(4096, (size + 127) / 128);
            //Set 1 in flag array for any column index appreaing in the rows that belong to the previous ring
            find_next_ring <<< num_blocks, 128>>>(B2L_maps[i].raw() + B2L_rings[i][ring - 1], A.row_offsets.raw(), A.col_indices.raw(), size, 0/*base_index*/, /*index_range*/size, halo_rings.raw());
            /*
            EXAMPLE
            for neighbor ID 0, halo_rings array initially [0 0 0 0], find_next_ring look at rows in the first ring [0 1] and sets halo_rings array positions to 1 for each column index in those rows:
            halo_rings: [1 1 1 1]
             */
            //Remove all flags that belong to rows in any previous rings
            remove_previous_rings <<< num_blocks, 128>>>(B2L_maps[i].raw(), halo_rings.raw(), B2L_rings[i][ring]);
            /*
             EXAMPLE
             for neighbor ID 0, halo_rings array [1 1 1 1], remove_previous_rings sets halo_rings positions to 0 for each index in previous B2L rings ([0 1] in this case):
             halo_rings: [0 0 1 1]
             */
            //Count, index and append them to B2L_maps and update B2L_rings
            thrust::inclusive_scan(halo_rings.begin(), halo_rings.begin() + size + 1, halo_rings.begin());
            B2L_rings[i][ring + 1] = B2L_rings[i][ring] + halo_rings[size];
            B2L_maps[i].resize(B2L_rings[i][ring + 1]);

            /*
             EXAMPLE
             after inclusive scan: [0 0 1 2], B2L_rings[0][2] = 2+2, and B2L_maps[0] gets resized to 4
             */
            if (B2L_rings[i][ring] == B2L_rings[i][ring + 1])
            {
                for (int fill = ring; fill <= rings; fill++) { B2L_rings[i][fill] = B2L_rings[i][ring]; }

                break;
            }

            num_blocks = min(4096, (A.get_num_rows() + 127) / 128);
            write_ring_rows <<< num_blocks, 128>>>(halo_rings.raw(), B2L_maps[i].raw() + B2L_rings[i][ring], A.get_num_rows());
            /*
             EXAMPLE
             appends positions i to B2L_maps, where halo_rings[i-1] != halo_rings[i]
             B2L_maps[0] = [0 1| 2 3] with B2L_rings[0] = [0 2 4]
             */
        }

        cudaCheckError();
    }
}

// CUDA 4.2
template <AMGX_VecPrecision t_vecPrec, AMGX_MatPrecision t_matPrec, AMGX_IndPrecision t_indPrec>
void DistributedArranger<TemplateConfig<AMGX_device, t_vecPrec, t_matPrec, t_indPrec> >::create_rings(Matrix_d &A, INDEX_TYPE rings)
{
    int neighbors = A.manager->num_neighbors();
    int size = A.get_num_rows();
    int num_blocks = min(4096, (size + 127) / 128);
    A.manager->B2L_rings.resize(neighbors * rings + 1);

    for (int i = 0; i < neighbors; i ++)
    {
        A.manager->B2L_rings[i].resize(rings + 1);
        A.manager->B2L_rings[i][0] = 0;
    }

    //Scratch space computation
    for (int i = 0; i < neighbors; i++)
    {
        A.manager->B2L_rings[i][1] = A.manager->B2L_maps[i].size();
    }

    IVector halo_rings(A.get_num_rows());

    for (int i = 0; i < neighbors; i++)
    {
        //
        // For each ring: flag nodes that are neighbors of the previous ring, then remove the nodes of the previous ring from the marking
        // count the number of marked nodes, and append their row indices to the B2L_maps
        //
        for (int ring = 1; ring < rings; ring++)
        {
            size = A.manager->B2L_rings[i][ring] - A.manager->B2L_rings[i][ring - 1];
            thrust::fill(halo_rings.begin(), halo_rings.begin() + size + 1, 0);
            num_blocks = min(4096, (size + 127) / 128);
            find_next_ring <<< num_blocks, 128>>>(A.manager->B2L_maps[i].raw() + A.manager->B2L_rings[i][ring - 1], A.row_offsets.raw(), A.col_indices.raw(), size, 0/*base_index*/, /*index_range*/size, halo_rings.raw());
            remove_previous_rings <<< num_blocks, 128>>>(A.manager->B2L_maps[i].raw(), halo_rings.raw(), A.manager->B2L_rings[i][ring]);
            thrust::inclusive_scan(halo_rings.begin(), halo_rings.begin() + size + 1, halo_rings.begin());
            A.manager->B2L_rings[i][ring + 1] = A.manager->B2L_rings[i][ring] + halo_rings[size];
            A.manager->B2L_maps[i].resize(A.manager->B2L_rings[i][ring + 1]);

            if (A.manager->B2L_rings[i][ring] == A.manager->B2L_rings[i][ring + 1])
            {
                for (int fill = ring; fill <= rings; fill++) { A.manager->B2L_rings[i][fill] = A.manager->B2L_rings[i][ring]; }

                break;
            }

            num_blocks = min(4096, (A.get_num_rows() + 127) / 128);
            write_ring_rows <<< num_blocks, 128>>>(halo_rings.raw(), A.manager->B2L_maps[i].raw() + A.manager->B2L_rings[i][ring], A.get_num_rows());
        }

        cudaCheckError();
    }
}

template <AMGX_VecPrecision t_vecPrec, AMGX_MatPrecision t_matPrec, AMGX_IndPrecision t_indPrec>
void DistributedArranger<TemplateConfig<AMGX_device, t_vecPrec, t_matPrec, t_indPrec> >::create_maps(Matrix_d &A, INDEX_TYPE rings,
        INDEX_TYPE neighbors,
        std::vector<IVector > &B2L_maps,
        std::vector<std::vector<VecInt_t> > &B2L_rings, std::vector<IVector> &boundary_lists)
{
    int size = A.get_num_rows();
    int num_blocks = min(4096, (size + 127) / 128);
    B2L_maps.resize(neighbors);

    //
    // Step 4: export n-ring halo matrices
    //
    for (int i = 0; i < neighbors; i++)
    {
        B2L_maps[i] = boundary_lists[i];
    }

    create_rings(A, rings, neighbors, B2L_maps, B2L_rings);
    //B2L_rings.resize(neighbors*rings+1);
}

// need for CUDA 4.2
template <AMGX_VecPrecision t_vecPrec, AMGX_MatPrecision t_matPrec, AMGX_IndPrecision t_indPrec>
void DistributedArranger<TemplateConfig<AMGX_device, t_vecPrec, t_matPrec, t_indPrec> >::create_maps(Matrix_d &A, INDEX_TYPE rings, std::vector<IVector> &boundary_lists)
{
    int size = A.get_num_rows();
    int num_blocks = min(4096, (size + 127) / 128);
    int neighbors = A.manager->num_neighbors();
    A.manager->B2L_maps.resize(neighbors);

    //
    // Step 4: export n-ring halo matrices
    //
    for (int i = 0; i < neighbors; i++)
    {
        A.manager->B2L_maps[i] = boundary_lists[i];
    }

    create_rings(A, rings);
    // A.manager->B2L_rings.resize(neighbors*rings+1);
}

template <AMGX_VecPrecision t_vecPrec, AMGX_MatPrecision t_matPrec, AMGX_IndPrecision t_indPrec>
void DistributedArranger<TemplateConfig<AMGX_device, t_vecPrec, t_matPrec, t_indPrec> >::create_halo_rows(Matrix_d &A, INDEX_TYPE rings, INDEX_TYPE neighbors, std::vector<IVector > &B2L_maps,
        std::vector<std::vector<VecInt_t> > &B2L_rings, std::vector<Matrix_d > **halo_rows_p, std::vector<IVector *> &halo_lists)
{
    typedef typename MatPrecisionMap<t_matPrec>::Type ValueTypeA;
    *halo_rows_p = new std::vector<Matrix<TConfig> >(neighbors);
    std::vector<Matrix<TConfig> > &halo_rows = **halo_rows_p;

    if (this->halo_coloring == LAST)   //Create dummy matrices
    {
        for (int i = 0; i < neighbors; i++)
        {
            halo_rows[i].addProps(CSR);

            if (A.hasProps(DIAG)) { halo_rows[i].addProps(DIAG); }

            halo_rows[i].resize(halo_lists[i]->size(), halo_lists[i]->size(), (A.hasProps(DIAG) ? 0 : halo_lists[i]->size()), A.get_block_dimy(), A.get_block_dimx(), 1);

            if (A.hasProps(DIAG))
            {
                thrust::fill(halo_rows[i].row_offsets.begin(), halo_rows[i].row_offsets.end(), 0);
            }
            else
            {
                thrust::sequence(halo_rows[i].row_offsets.begin(), halo_rows[i].row_offsets.end());
                thrust::copy(halo_lists[i]->begin(), halo_lists[i]->end(), halo_rows[i].col_indices.begin());
            }

            thrust::fill(halo_rows[i].values.begin(), halo_rows[i].values.end(), types::util< ValueTypeA >::get_one());
            cudaCheckError();
        }

        return;
    }

    //Scratch space computation
    int max_size = 0;

    for (int i = 0; i < neighbors; i++)
    {
        max_size = max_size > B2L_maps[i].size() ? max_size : B2L_maps[i].size();
        //B2L_rings[i][1]=B2L_maps[i].size(); //I don't know what this was doing here, definitely stupid for rings > 1
    }

    IVector matrix_halo_sizes((max_size + 1)*rings);

    for (int i = 0; i < neighbors; i++)
    {
        //
        // Write the length of the rows in the order of B2L_maps, then calculate row_offsets
        //
        int size = B2L_rings[i][rings];
        matrix_halo_sizes.resize(size + 1);
        int num_blocks = min(4096, (size + 127) / 128);
        write_matrix_rowsize <<< num_blocks, 128>>>(B2L_maps[i].raw(), A.row_offsets.raw(), size, matrix_halo_sizes.raw());
        thrust::exclusive_scan(matrix_halo_sizes.begin(), matrix_halo_sizes.begin() + size + 1, matrix_halo_sizes.begin());
        int nnz_count =  matrix_halo_sizes[size];
        //
        // Resize export halo matrix, and copy over the rows
        //
        halo_rows[i].addProps(CSR);

        if (A.hasProps(DIAG)) { halo_rows[i].addProps(DIAG); }

        halo_rows[i].resize(B2L_maps[i].size(), B2L_maps[i].size(), nnz_count, A.get_block_dimy(), A.get_block_dimx(), 1);
        export_matrix_elements<32> <<< num_blocks, 128>>>(A.row_offsets.raw(), A.values.raw(), A.get_block_size(), B2L_maps[i].raw(), matrix_halo_sizes.raw(), halo_rows[i].values.raw(), A.col_indices.raw(), halo_rows[i].col_indices.raw(), size);
        thrust::copy(matrix_halo_sizes.begin(), matrix_halo_sizes.begin() + size + 1, halo_rows[i].row_offsets.begin());

        if (A.hasProps(DIAG))
        {
            export_matrix_diagonal <<< num_blocks, 128>>>(A.values.raw() + A.row_offsets[A.get_num_rows()]*A.get_block_size(), A.get_block_size(), B2L_maps[i].raw(), halo_rows[i].values.raw() + halo_rows[i].row_offsets[halo_rows[i].get_num_rows()]*A.get_block_size(), size);
        }
    }

    cudaCheckError();
}

template <AMGX_VecPrecision t_vecPrec, AMGX_MatPrecision t_matPrec, AMGX_IndPrecision t_indPrec>
void DistributedArranger<TemplateConfig<AMGX_device, t_vecPrec, t_matPrec, t_indPrec> >::initialize_manager(Matrix_d &A, Matrix_d &P, int num_owned_coarse_pts)
{
    // This function initializes the manager of matrix P based on value of num_owned_coarse_pts
    // The following member of the manager are initialized:
    // global_id, num_partitions, comms (copied from manager of matrix A)
    // part_offsets_h, part_offsets, base_index, index_range, num_rows_global, halo_offsets[0] (those are initialized based on value of num_owned_coarse_pts of all partitions in communicator)
    if (P.manager == NULL)
    {
        P.manager = new DistributedManager<TConfig_d>();
    }

    P.setView(OWNED);
    // Copy some information from manager of matrix A
    P.manager->A = &P;
    P.manager->setComms(A.manager->getComms());
    index_type num_parts = A.manager->get_num_partitions();
    index_type my_rank = A.manager->global_id();
    P.manager->set_global_id(my_rank);
    P.manager->set_num_partitions(num_parts);
    // Do a global reduce to obtain part_offsets for RAP
    IVector_h gathered_count;
    A.manager->getComms()->all_gather(num_owned_coarse_pts, gathered_count, num_parts);
    // Compute P.manager->part_offsets
    P.manager->part_offsets_h.resize(num_parts + 1);
    P.manager->part_offsets_h[0] = (int64_t) 0;

    for (int i = 1; i < num_parts + 1; i++)
    {
        P.manager->part_offsets_h[i] = P.manager->part_offsets_h[i - 1] + (int64_t) gathered_count[i - 1];
    }

    P.manager->part_offsets = P.manager->part_offsets_h;
    // Compute P.manager->halo_ranges
    P.manager->set_base_index(P.manager->part_offsets_h[my_rank]);
    P.manager->set_index_range(num_owned_coarse_pts);
    P.manager->num_rows_global = P.manager->part_offsets_h[num_parts];
    // Set halo_offsets[0]
    P.manager->halo_offsets.resize(1);
    P.manager->halo_offsets[0] = num_owned_coarse_pts;
}

template <AMGX_VecPrecision t_vecPrec, AMGX_MatPrecision t_matPrec, AMGX_IndPrecision t_indPrec>
void DistributedArranger<TemplateConfig<AMGX_device, t_vecPrec, t_matPrec, t_indPrec> >::initialize_manager_from_global_col_indices(Matrix_d &P, I64Vector_d &P_col_indices_global)
{
    // This function finds the number of neighbors of P
    // fills halo_offsets, local_to_global_map, B2L_maps based on global col indices of matrix P
    int num_owned_coarse_pts = P.manager->halo_offsets[0];
    // By checking the global col indices, identify neighbors
    P.manager->neighbors.resize(0);
    this->create_neighbors_v2_global(P, P_col_indices_global);
    int num_neighbors = P.manager->neighbors.size();
    P.manager->getComms()->set_neighbors(num_neighbors);
    P.manager->halo_offsets.resize(num_neighbors + 1, 0);
    int total_rows_of_neighbors = 0;
    std::vector<int> flag_offsets_h(num_neighbors + 1, 0);

    for (int i = 0; i < num_neighbors; i ++)
    {
        int num_rows = (int) (P.manager->halo_ranges_h[2 * i + 1] - P.manager->halo_ranges_h[2 * i]);
        total_rows_of_neighbors += num_rows;
        flag_offsets_h[i + 1] = flag_offsets_h[i] + num_rows;
    }

    // Copy to device
    IVector flag_offsets_d(num_neighbors + 1);
    thrust::copy(flag_offsets_h.begin(), flag_offsets_h.end(), flag_offsets_d.begin());
    // Allocate halo_nodes array that will be used to flag halo nodes
    // Flag_offsets array can be used to determine where each neighbor begins
    IVector halo_nodes(total_rows_of_neighbors + 1);
    thrust::fill(halo_nodes.begin(), halo_nodes.end(), 0);
    cudaCheckError();
    int num_blocks = min(4096, (P.get_num_rows() + 127) / 128);

    if (P.get_num_rows() > 0)
    {
        flag_halo_nodes_global_v2<16> <<< num_blocks, 128>>>(P.row_offsets.raw(), P_col_indices_global.raw(), P.manager->halo_ranges.raw(), halo_nodes.raw(), flag_offsets_d.raw(), P.manager->base_index(), P.manager->index_range(), num_neighbors, P.get_num_rows());
        cudaCheckError();
    }

    // Step 3: Write the 1-ring global nodes for each neighbor
    thrust::exclusive_scan(halo_nodes.begin(), halo_nodes.end(), halo_nodes.begin());
    cudaCheckError();
    std::vector<IVector> bdy_lists;
    bdy_lists.resize(num_neighbors);
    int count = 0;

    for (int i = 0; i < num_neighbors; i++)
    {
        int num_halo = halo_nodes[flag_offsets_h[i + 1]] - count;
        P.manager->halo_offsets[i + 1] = P.manager->halo_offsets[i] + num_halo;
        bdy_lists[i].resize(num_halo);

        if (num_halo > 0)
        {
            int size = flag_offsets_h[i + 1] - flag_offsets_h[i];
            int num_blocks = min(4096, (size + 127) / 128);
            populate_B2L_v2 <<< num_blocks, 128>>>(halo_nodes.raw() + flag_offsets_h[i], bdy_lists[i].raw(), count, size);
        }

        count += num_halo;
    }

    cudaCheckError();

    // Fill the col_indices
    if (P.get_num_rows() > 0)
    {
        renumber_col_indices_global<16> <<< num_blocks, 128>>>(P.row_offsets.raw(), P.col_indices.raw(), P_col_indices_global.raw(), P.manager->halo_ranges.raw(), halo_nodes.raw(), flag_offsets_d.raw(), P.manager->base_index(), P.manager->index_range(), num_neighbors, P.get_num_rows(), P.manager->global_id());
        cudaCheckError();
    }

    // Create P.local_to_global_map
    P.manager->local_to_global_map.resize(count);
    count = 0;

    for (int i = 0; i < num_neighbors; i++)
    {
        int num_halo = bdy_lists[i].size();

        if (num_halo > 0)
        {
            int num_blocks = min(4096, (num_halo + 127) / 128);
            fill_local_to_global_map <<< num_blocks, 128>>>(bdy_lists[i].raw(), P.manager->local_to_global_map.raw() + count, P.manager->halo_ranges_h[2 * i], num_halo);
        }

        count += num_halo;
    }

    cudaCheckError();
    P.manager->getComms()->exchange_vectors(bdy_lists, P, 0);
    // copy boundary lists to B2L maps
    // release internal boundary lists array
    P.manager->B2L_maps.resize(num_neighbors);

    for (int i = 0; i < num_neighbors; i++)
    {
        P.manager->B2L_maps[i] = bdy_lists[i];
    }

    P.manager->B2L_rings.resize(num_neighbors);

    for (int i = 0; i < num_neighbors; i++)
    {
        P.manager->B2L_rings[i].resize(2);
        P.manager->B2L_rings[i][0] = 0;
        P.manager->B2L_rings[i][1] = P.manager->B2L_maps[i].size();
    }

    this->initialize_B2L_maps_offsets(P, 1);
    P.manager->_num_interior_nodes = P.get_num_rows();
    P.manager->_num_boundary_nodes = 0;
    P.manager->set_initialized(P.row_offsets);
    int size_one_ring, offset;
    P.manager->getOffsetAndSizeForView(FULL, &offset, &size_one_ring);
    P.setView(OWNED);
}


template <AMGX_VecPrecision t_vecPrec, AMGX_MatPrecision t_matPrec, AMGX_IndPrecision t_indPrec>
void DistributedArranger<TemplateConfig<AMGX_device, t_vecPrec, t_matPrec, t_indPrec> >::createTempManager(Matrix_d &B, Matrix_d &A, IVector &row_offsets)
{
    // Initialize P.manager
    if (B.manager == NULL)
    {
        B.manager = new DistributedManager<TConfig_d>();
    }

    index_type num_parts = A.manager->get_num_partitions();
    index_type my_rank = A.manager->global_id();
    B.manager->A = &B;
    B.manager->setComms(A.manager->getComms());
    B.manager->set_global_id(my_rank);
    B.manager->set_num_partitions(num_parts);
    B.manager->neighbors = A.manager->neighbors;
    int num_neighbors = A.manager->neighbors.size();
    B.manager->getComms()->set_neighbors(num_neighbors);
    // --------------------------------------
    // Create B.manager->B2L_maps
    // --------------------------------------
    B.manager->B2L_maps.resize(num_neighbors);
    int max_B2L_maps_size = 0;

    for (int i = 0; i < num_neighbors; i++)
    {
        int A_b2l_map_size = A.manager->B2L_rings[i][1];

        if (A_b2l_map_size > max_B2L_maps_size)
        {
            max_B2L_maps_size = A_b2l_map_size;
        }
    }

    IVector row_sizes(row_offsets.size() - 1);
    thrust::adjacent_difference(row_offsets.begin() + 1, row_offsets.end(), row_sizes.begin());
    cudaCheckError();
    IVector B2L_maps_offsets(max_B2L_maps_size + 1);

    for (int i = 0; i < num_neighbors; i++)
    {
        int A_b2l_map_size = A.manager->B2L_rings[i][1];

        if (A_b2l_map_size > 0)
        {
            // For each node in the B2L_map for that neighbor, write the row size in B2L_maps_offsets
            thrust::gather(A.manager->B2L_maps[i].begin(), A.manager->B2L_maps[i].begin() + A_b2l_map_size, row_sizes.begin(), B2L_maps_offsets.begin());
            // Do exclusive scan, to go from row length to row offsets
            thrust::exclusive_scan(B2L_maps_offsets.begin(), B2L_maps_offsets.begin() + A_b2l_map_size + 1, B2L_maps_offsets.begin());
            B.manager->B2L_maps[i].resize(B2L_maps_offsets[A_b2l_map_size]);
            const int cta_size = 128;
            const int nWarps = cta_size / 32;
            int grid_size = min( 4096, (int) (A_b2l_map_size + nWarps - 1) / nWarps);
            // Populate the B2L_map array
            fill_B_B2L_map<int, cta_size > <<< grid_size, cta_size >>>(
                B2L_maps_offsets.raw(),
                A.manager->B2L_maps[i].raw(),
                row_offsets.raw(),
                B.manager->B2L_maps[i].raw(),
                A_b2l_map_size,
                A.manager->global_id());
        }
        else
        {
            B.manager->B2L_maps[i].resize(0);
        }

        cudaCheckError();
    }

    B.manager->B2L_rings.resize(num_neighbors);

    for (int i = 0; i < num_neighbors; i++)
    {
        B.manager->B2L_rings[i].resize(2);
        B.manager->B2L_rings[i][0] = 0;
        B.manager->B2L_rings[i][1] = B.manager->B2L_maps[i].size();
    }

    // Initialize the B2L_maps_offsets (required by gather_v2 and scatter_v2) from the B2L_rings values
    this->initialize_B2L_maps_offsets(B, 1);
    B.manager->halo_offsets.resize(num_neighbors + 1);

    // Fill halo_offsets
    for (int i = 0; i < num_neighbors + 1; i++)
    {
        B.manager->halo_offsets[i] = row_offsets[A.manager->halo_offsets[i]];
    }
}


template <AMGX_VecPrecision t_vecPrec, AMGX_MatPrecision t_matPrec, AMGX_IndPrecision t_indPrec>
void DistributedArranger<TemplateConfig<AMGX_device, t_vecPrec, t_matPrec, t_indPrec> >::exchange_halo_rows_P(Matrix_d &A, Matrix_d &P, I64Vector &RAP_local_to_global_map, IVector_h &P_neighbors, I64Vector_h &P_halo_ranges_h, I64Vector &P_halo_ranges, IVector_h &P_halo_offsets, I64Vector_h &RAP_part_offsets_h, I64Vector &RAP_part_offsets, index_type num_owned_coarse_pts, int64_t coarse_base_index)
{
    if (P.hasProps(DIAG) || P.get_block_size() != 1)
    {
        FatalError("P with external diagonal or block_size != 1 not supported", AMGX_ERR_NOT_IMPLEMENTED);
    }

    int num_neighbors = A.manager->num_neighbors() ;
    std::vector<IVector> halo_rows_P_row_offsets;
    std::vector<I64Vector> halo_rows_P_col_indices;
    std::vector<IVector> halo_rows_P_local_col_indices;
    std::vector<MVector> halo_rows_P_values;
    // In this function, store in halo_rows_P_row_offsets, halo_rows_P_col_indices and halo_rows_P_values, the rows of P that need to be sent to each neighbors
    // halo_rows_P_col_indices stores global indices
    this->pack_halo_rows_P(A, P, halo_rows_P_row_offsets, halo_rows_P_col_indices, halo_rows_P_values, RAP_local_to_global_map, num_owned_coarse_pts, coarse_base_index);
    // Do the exchange with the neighbors
    // On return, halo_rows_P_rows_offsets, halo_rows_P_col_indices and halo_rows_P_values stores the rows of P received from each neighbor (rows needed to perform A*P)
    std::vector<I64Vector> dummy_halo_ids(0);
    A.manager->getComms()->exchange_matrix_halo(halo_rows_P_row_offsets, halo_rows_P_col_indices, halo_rows_P_values, dummy_halo_ids, A.manager->neighbors, A.manager->global_id());
    halo_rows_P_local_col_indices.resize(halo_rows_P_col_indices.size());

    for (int i = 0; i < halo_rows_P_col_indices.size(); i++)
    {
        halo_rows_P_local_col_indices[i].resize(halo_rows_P_col_indices[i].size());
    }

    // Update the list of neighbors "P_neighbors" and the corresponding ranges, offsets
    update_neighbors_list(A, P_neighbors, P_halo_ranges_h, P_halo_ranges, RAP_part_offsets_h, RAP_part_offsets, halo_rows_P_row_offsets, halo_rows_P_col_indices);
    std::vector<IVector> dummy_boundary_list(0);
    int current_num_rings = 1;
    halo_rows_P_local_col_indices.resize(halo_rows_P_col_indices.size());
    // Convert the global indices in the rows just received to local indices
    // This function updates halo offsets, halo_rows_P_local_col_indices, local_to_global_map
    // This should not modify the existing P manager
    this->compute_local_halo_indices( P.row_offsets, P.col_indices, halo_rows_P_row_offsets, halo_rows_P_col_indices, halo_rows_P_local_col_indices, RAP_local_to_global_map,  dummy_boundary_list, P_neighbors, P_halo_ranges_h, P_halo_ranges, P_halo_offsets, coarse_base_index, num_owned_coarse_pts, P.get_num_rows(), current_num_rings);
    // Append the new rows to the matrix P
    P.set_initialized(0);
    this->append_halo_rows(P, halo_rows_P_row_offsets, halo_rows_P_local_col_indices, halo_rows_P_values);
    P.set_initialized(1);
}


template <AMGX_VecPrecision t_vecPrec, AMGX_MatPrecision t_matPrec, AMGX_IndPrecision t_indPrec>
void DistributedArranger<TemplateConfig<AMGX_device, t_vecPrec, t_matPrec, t_indPrec> >::append_halo_rows(Matrix_d &A, std::vector<IVector> &halo_row_offsets, std::vector<IVector> &halo_local_indices, std::vector<MVector> &halo_values )
{
    int new_num_rows = A.get_num_rows();
    int new_num_nnz = A.row_offsets[A.get_num_rows()];
    int cur_row = A.get_num_rows();
    int cur_offset = new_num_nnz;
    int num_neighbors = halo_row_offsets.size();

    for (int i = 0; i < num_neighbors; i++)
    {
        int size = halo_row_offsets[i].size();

        if (size != 0)
        {
            new_num_rows += halo_row_offsets[i].size() - 1;
            new_num_nnz += halo_local_indices[i].size();
        }
    }

    A.resize(new_num_rows, new_num_rows, new_num_nnz, 1, 1, 1);

    for (int i = 0; i < num_neighbors; i++)
    {
        int num_halo_rows = halo_row_offsets[i].size() - 1;

        if (num_halo_rows > 0)
        {
            // update halo row offsets in-place
            thrust::transform(halo_row_offsets[i].begin(), halo_row_offsets[i].end(), thrust::constant_iterator<INDEX_TYPE>(cur_offset), halo_row_offsets[i].begin(), thrust::plus<INDEX_TYPE>());
            // insert halo rows
            thrust::copy(halo_row_offsets[i].begin(), halo_row_offsets[i].end() - 1, A.row_offsets.begin() + cur_row);
            thrust::copy(halo_local_indices[i].begin(), halo_local_indices[i].end(), A.col_indices.begin() + cur_offset);
            thrust::copy(halo_values[i].begin(), halo_values[i].end(), A.values.begin() + cur_offset);
            // update counters
            cur_offset = halo_row_offsets[i][num_halo_rows];
            cur_row += num_halo_rows;
        }
    }

    cudaCheckError();
    A.row_offsets[A.get_num_rows()] = cur_offset;
    int num_cols = -1;
    num_cols = thrust::reduce(A.col_indices.begin(), A.col_indices.end(), num_cols, thrust::maximum<int>()) + 1;
    cudaCheckError();
    A.set_num_cols(num_cols);
}


template <AMGX_VecPrecision t_vecPrec, AMGX_MatPrecision t_matPrec, AMGX_IndPrecision t_indPrec>
void DistributedArranger<TemplateConfig<AMGX_device, t_vecPrec, t_matPrec, t_indPrec> >::exchange_RAP_ext(Matrix_d &RAP, Matrix_d &RAP_full, Matrix_d &A, Matrix_d &P, IVector_h &P_halo_offsets, I64Vector &RAP_local_to_global_map, IVector_h &P_neighbors, I64Vector_h &P_halo_ranges_h, I64Vector &P_halo_ranges, I64Vector_h &RAP_part_offsets_h, I64Vector &RAP_part_offsets, index_type num_owned_coarse_pts, int64_t coarse_base_index, void *wk)
{
    if (RAP_full.hasProps(DIAG) || RAP_full.get_block_size() != 1)
    {
        FatalError("RAP_full with external diagonal or block_size != 1 not supported", AMGX_ERR_NOT_IMPLEMENTED);
    }

    // --------------------------------------------------------------------------------------
    // Pack the rows of RAP_full that need to be sent to neighbors, using the manager of P
    // --------------------------------------------------------------------------------------
    std::vector<IVector> halo_rows_RAP_row_offsets;
    std::vector<I64Vector> halo_rows_RAP_row_ids;
    std::vector<I64Vector> halo_rows_RAP_col_indices;
    std::vector<MVector> halo_rows_RAP_values;
    this->pack_halo_rows_RAP(RAP_full, P, halo_rows_RAP_row_offsets, halo_rows_RAP_col_indices, halo_rows_RAP_values, halo_rows_RAP_row_ids, RAP_local_to_global_map);
    // Send the rows to the neighbors, receive RAP rows from neighbors
    // Col indices received from neighbors are global indices
    // We also receive from neighbors the global row ids
    P.manager->getComms()->exchange_matrix_halo(halo_rows_RAP_row_offsets, halo_rows_RAP_col_indices, halo_rows_RAP_values, halo_rows_RAP_row_ids, P.manager->neighbors, P.manager->global_id());
    std::vector<IVector> halo_rows_RAP_local_col_indices;
    halo_rows_RAP_local_col_indices.resize(halo_rows_RAP_col_indices.size());

    for (int i = 0; i < halo_rows_RAP_col_indices.size(); i++)
    {
        halo_rows_RAP_local_col_indices[i].resize(halo_rows_RAP_col_indices[i].size());
    }

    // This function finds the new number of neighbors, renumber halo_cols with local_ids, modifies halo_offsets
    update_neighbors_list(A, P_neighbors, P_halo_ranges_h, P_halo_ranges, RAP_part_offsets_h, RAP_part_offsets, halo_rows_RAP_row_offsets, halo_rows_RAP_col_indices);
    // Now renumber find local indices for global col indices, update local_to_global
    std::vector<IVector> dummy_boundary_list(0);
    int current_num_rings = 2;
    this->compute_local_halo_indices(RAP_full.row_offsets, RAP_full.col_indices, halo_rows_RAP_row_offsets, halo_rows_RAP_col_indices, halo_rows_RAP_local_col_indices, RAP_local_to_global_map, dummy_boundary_list, P_neighbors, P_halo_ranges_h, P_halo_ranges, P_halo_offsets, coarse_base_index, num_owned_coarse_pts, num_owned_coarse_pts, current_num_rings );
    // Now find local row ids from global ids of rows received
    std::vector<IVector> halo_rows_RAP_local_row_ids;
    halo_rows_RAP_local_row_ids.resize(halo_rows_RAP_row_ids.size());

    for (int i = 0; i < halo_rows_RAP_row_ids.size(); i++)
    {
        int size = halo_rows_RAP_row_ids[i].size();
        halo_rows_RAP_local_row_ids[i].resize(size);

        if (size != 0)
        {
            int cta_size = 128;
            int grid_size = std::min( 4096, (size + cta_size - 1) / cta_size);
            fill_halo_rows_row_ids <<< grid_size, cta_size >>>( halo_rows_RAP_row_ids[i].raw(), halo_rows_RAP_local_row_ids[i].raw(), coarse_base_index, num_owned_coarse_pts, size);
        }
    }

    cudaCheckError();
    // Add the rows received from neighbors to owned rows in RAP_full and create RAP matrix
    RAP.set_initialized( 0 );
    RAP.set_num_rows(num_owned_coarse_pts);
    RAP.set_num_cols(num_owned_coarse_pts + RAP.manager->local_to_global_map.size());
    RAP.row_offsets.resize(num_owned_coarse_pts + 1);
    RAP_full.setView(OWNED);
    CSR_Multiply<TConfig_d>::csr_RAP_sparse_add( RAP, RAP_full, halo_rows_RAP_row_offsets, halo_rows_RAP_local_col_indices, halo_rows_RAP_values, halo_rows_RAP_local_row_ids, wk );
    RAP.set_initialized( 1 );
}


template <AMGX_VecPrecision t_vecPrec, AMGX_MatPrecision t_matPrec, AMGX_IndPrecision t_indPrec>
void DistributedArranger<TemplateConfig<AMGX_device, t_vecPrec, t_matPrec, t_indPrec> >::pack_halo_rows_P(Matrix_d &A, Matrix_d &P, std::vector<IVector> &halo_rows_P_row_offsets, std::vector<I64Vector> &halo_rows_P_col_indices, std::vector<MVector> &halo_rows_P_values, I64Vector &RAP_local_to_global_map, index_type num_owned_coarse_pts, int64_t coarse_base_index)
{
    int num_neighbors = A.manager->num_neighbors();
    halo_rows_P_row_offsets.resize(num_neighbors);
    halo_rows_P_col_indices.resize(num_neighbors);
    halo_rows_P_values.resize(num_neighbors);
    int num_rings_to_send = 1;
    //Scratch space computation
    int max_size = 0;

    for (int i = 0; i < num_neighbors; i++)
    {
        max_size = max_size > A.manager->B2L_rings[i][num_rings_to_send] ? max_size : A.manager->B2L_rings[i][num_rings_to_send];
    }

    IVector matrix_halo_sizes(max_size + 1);

    // Here only using the 1-ring of matrix A
    for (int i = 0; i < num_neighbors; i++)
    {
        // Write the length of the rows in the order of B2L_maps, then calculate row_offsets
        int size = A.manager->B2L_rings[i][num_rings_to_send];

        if (size != 0)
        {
            //matrix_halo_sizes.resize(size+1);
            int num_blocks = min(4096, (size + 127) / 128);
            write_matrix_rowsize <<< num_blocks, 128>>>(A.manager->B2L_maps[i].raw(), P.row_offsets.raw(), size, matrix_halo_sizes.raw());
            thrust::exclusive_scan(matrix_halo_sizes.begin(), matrix_halo_sizes.begin() + size + 1, matrix_halo_sizes.begin());
            int nnz_count =  matrix_halo_sizes[size];
            // Resize export halo matrix, and copy over the rows
            halo_rows_P_row_offsets[i].resize(size + 1);
            halo_rows_P_col_indices[i].resize(nnz_count);
            halo_rows_P_values[i].resize(nnz_count);
            /* WARNING: Since A is reordered (into interior and boundary nodes), while R and P are not reordered,
                        you must unreorder A when performing R*A*P product in ordre to obtain the correct result. */
            export_matrix_elements_global<32> <<< num_blocks, 128>>>(P.row_offsets.raw(), P.values.raw(), P.get_block_size(), A.manager->B2L_maps[i].raw(), matrix_halo_sizes.raw(), halo_rows_P_values[i].raw(), P.col_indices.raw(), halo_rows_P_col_indices[i].raw(), size, RAP_local_to_global_map.raw(), NULL /*A.manager->inverse_renumbering.raw()*/, num_owned_coarse_pts, coarse_base_index);
            thrust::copy(matrix_halo_sizes.begin(), matrix_halo_sizes.begin() + size + 1, halo_rows_P_row_offsets[i].begin());
        }
        else
        {
            halo_rows_P_row_offsets[i].resize(0);
            halo_rows_P_col_indices[i].resize(0);
            halo_rows_P_values[i].resize(0);
        }
    }

    cudaCheckError();
}

template <AMGX_VecPrecision t_vecPrec, AMGX_MatPrecision t_matPrec, AMGX_IndPrecision t_indPrec>
void DistributedArranger<TemplateConfig<AMGX_device, t_vecPrec, t_matPrec, t_indPrec> >::pack_halo_rows_RAP(Matrix_d &RAP, Matrix_d &P, std::vector<IVector> &halo_rows_RAP_row_offsets, std::vector<I64Vector> &halo_rows_RAP_col_indices, std::vector<MVector> &halo_rows_RAP_values, std::vector<I64Vector> &halo_rows_RAP_row_ids, I64Vector &RAP_local_to_global_map)
{
    int num_neighbors = P.manager->neighbors.size();
    int num_owned_coarse_pts = P.manager->halo_offsets[0];
    int64_t P_col_base_index = P.manager->base_index();
    halo_rows_RAP_row_offsets.resize(num_neighbors);
    halo_rows_RAP_row_ids.resize(num_neighbors);
    halo_rows_RAP_col_indices.resize(num_neighbors);
    halo_rows_RAP_values.resize(num_neighbors);
    // Create tmp_B2L_maps that contains the list of rows to send
    std::vector<IVector> tmp_B2L_maps(num_neighbors);

    // Fill B2L_maps with list of halo indices for all rings to send
    for (int i = 0; i < num_neighbors; i++)
    {
        tmp_B2L_maps[i].resize(0);
        halo_rows_RAP_row_ids[i].resize(0);
        int begin = P.manager->halo_offsets[i];
        int end =  P.manager->halo_offsets[i + 1];

        if (begin > RAP.row_offsets.size() - 1)
        {
            break;
        }

        int local_to_global_begin = begin - P.manager->halo_offsets[0];
        int size = end - begin;

        if (end > RAP.row_offsets.size() - 1)
        {
            size = RAP.row_offsets.size() - 1 - begin;
        }

        if (size > 0)
        {
            tmp_B2L_maps[i].resize(size);
            halo_rows_RAP_row_ids[i].resize(size);
            thrust::sequence(tmp_B2L_maps[i].begin(), tmp_B2L_maps[i].begin() + size, begin);
            thrust::copy(P.manager->local_to_global_map.begin() + (int64_t) local_to_global_begin, P.manager->local_to_global_map.begin() + (int64_t) local_to_global_begin + (int64_t) size, halo_rows_RAP_row_ids[i].begin());
        }
    }

    cudaCheckError();
    //Scratch space computation
    int max_size = 0;

    for (int i = 0; i < num_neighbors; i++)
    {
        int size = tmp_B2L_maps[i].size();
        max_size = max_size > size ? max_size : size;
    }

    IVector matrix_halo_sizes(max_size + 1);

    for (int i = 0; i < num_neighbors; i++)
    {
        // Write the length of the rows in the order of B2L_maps, then calculate row_offsets
        int size = tmp_B2L_maps[i].size();

        if (size > 0)
        {
            //matrix_halo_sizes.resize(size+1);
            int num_blocks = min(4096, (size + 127) / 128);
            write_matrix_rowsize <<< num_blocks, 128>>>(tmp_B2L_maps[i].raw(), RAP.row_offsets.raw(), size, matrix_halo_sizes.raw());
            thrust::exclusive_scan(matrix_halo_sizes.begin(), matrix_halo_sizes.begin() + size + 1, matrix_halo_sizes.begin());
            int nnz_count =  matrix_halo_sizes[size];
            // Resize export halo matrix, and copy over the rows
            halo_rows_RAP_row_offsets[i].resize(tmp_B2L_maps[i].size() + 1);
            halo_rows_RAP_col_indices[i].resize(nnz_count);
            halo_rows_RAP_values[i].resize(nnz_count);
            export_matrix_elements_global<32> <<< num_blocks, 128>>>(RAP.row_offsets.raw(), RAP.values.raw(), RAP.get_block_size(), tmp_B2L_maps[i].raw(), matrix_halo_sizes.raw(), halo_rows_RAP_values[i].raw(), RAP.col_indices.raw(), halo_rows_RAP_col_indices[i].raw(), size, RAP_local_to_global_map.raw(), NULL, num_owned_coarse_pts, P_col_base_index);
            thrust::copy(matrix_halo_sizes.begin(), matrix_halo_sizes.begin() + size + 1, halo_rows_RAP_row_offsets[i].begin());
        }
        else
        {
            halo_rows_RAP_row_offsets[i].resize(0);
            halo_rows_RAP_col_indices[i].resize(0);
            halo_rows_RAP_values[i].resize(0);
        }
    }

    cudaCheckError();
}


template <AMGX_VecPrecision t_vecPrec, AMGX_MatPrecision t_matPrec, AMGX_IndPrecision t_indPrec>
void DistributedArranger<TemplateConfig<AMGX_device, t_vecPrec, t_matPrec, t_indPrec> >::sparse_matrix_add(Matrix_d &RAP, std::vector<IVector> &halo_rows_RAP_row_offsets, std::vector<I64Vector> &halo_rows_RAP_col_indices, std::vector<MVector> &halo_rows_RAP_values, std::vector<I64Vector> halo_rows_RAP_row_ids, I64Vector_d &P_local_to_global_map, index_type P_col_num_owned_pts, int64_t P_col_base_index)
{
    // Look at csr_galerkin_product implementation, use hash
}


// CUDA 4.2
template <AMGX_VecPrecision t_vecPrec, AMGX_MatPrecision t_matPrec, AMGX_IndPrecision t_indPrec>
void DistributedArranger<TemplateConfig<AMGX_device, t_vecPrec, t_matPrec, t_indPrec> >::create_halo_rows(Matrix_d &A,
        std::vector<IVector *> &halo_lists
                                                                                                         )
{
    int neighbors = A.manager->num_neighbors();

    if (neighbors == 0)
    {
        return;
    }

    int rings = (A.manager->B2L_rings.size() - 1) / neighbors;
    A.manager->halo_rows = new std::vector<Matrix<TConfig> >(neighbors);
    std::vector<Matrix<TConfig> > &halo_rows = *A.manager->halo_rows;

    if (this->halo_coloring == LAST)   //Create dummy matrices, only containing diagonal elements
    {
        for (int i = 0; i < neighbors; i++)
        {
            halo_rows[i].addProps(CSR);

            if (A.hasProps(DIAG)) { halo_rows[i].addProps(DIAG); }

            halo_rows[i].resize(halo_lists[i]->size(), halo_lists[i]->size(), (A.hasProps(DIAG) ? 0 : halo_lists[i]->size()), A.get_block_dimy(), A.get_block_dimx(), 1);

            if (A.hasProps(DIAG))
            {
                thrust::fill(halo_rows[i].row_offsets.begin(), halo_rows[i].row_offsets.end(), 0);
            }
            else
            {
                thrust::sequence(halo_rows[i].row_offsets.begin(), halo_rows[i].row_offsets.end());
                thrust::copy(halo_lists[i]->begin(), halo_lists[i]->end(), halo_rows[i].col_indices.begin());
            }

            cudaCheckError();
        }

        return;
    }

    //Scratch space computation
    int max_size = 0;

    for (int i = 0; i < neighbors; i++)
    {
        max_size = max_size > A.manager->B2L_maps[i].size() ? max_size : A.manager->B2L_maps[i].size();
        //B2L_rings[i][1]=B2L_maps[i].size(); //I don't know what this was doing here, definitely stupid for rings > 1
    }

    IVector matrix_halo_sizes((max_size + 1)*rings);

    for (int i = 0; i < neighbors; i++)
    {
        //
        // Write the length of the rows in the order of B2L_maps, then calculate row_offsets
        //
        int size = A.manager->B2L_rings[i][rings];
        matrix_halo_sizes.resize(size + 1);
        int num_blocks = min(4096, (size + 127) / 128);
        write_matrix_rowsize <<< num_blocks, 128>>>(A.manager->B2L_maps[i].raw(), A.row_offsets.raw(), size, matrix_halo_sizes.raw());
        thrust::exclusive_scan(matrix_halo_sizes.begin(), matrix_halo_sizes.begin() + size + 1, matrix_halo_sizes.begin());
        cudaCheckError();
        int nnz_count =  matrix_halo_sizes[size];
        //
        // Resize export halo matrix, and copy over the rows
        //
        halo_rows[i].addProps(CSR);

        if (A.hasProps(DIAG)) { halo_rows[i].addProps(DIAG); }

        halo_rows[i].resize(A.manager->B2L_maps[i].size(), A.manager->B2L_maps[i].size(), nnz_count, A.get_block_dimy(), A.get_block_dimx(), 1);
        export_matrix_elements<32> <<< num_blocks, 128>>>(A.row_offsets.raw(), A.values.raw(), A.get_block_size(), A.manager->B2L_maps[i].raw(), matrix_halo_sizes.raw(), halo_rows[i].values.raw(), A.col_indices.raw(), halo_rows[i].col_indices.raw(), size);
        thrust::copy(matrix_halo_sizes.begin(), matrix_halo_sizes.begin() + size + 1, halo_rows[i].row_offsets.begin());
        cudaCheckError();

        if (A.hasProps(DIAG))
        {
            export_matrix_diagonal <<< num_blocks, 128>>>(A.values.raw() + A.row_offsets[A.get_num_rows()]*A.get_block_size(), A.get_block_size(), A.manager->B2L_maps[i].raw(), halo_rows[i].values.raw() + halo_rows[i].row_offsets[halo_rows[i].get_num_rows()]*A.get_block_size(), size);
        }
    }

    cudaCheckError();
}

template <AMGX_VecPrecision t_vecPrec, AMGX_MatPrecision t_matPrec, AMGX_IndPrecision t_indPrec>
void DistributedArranger<TemplateConfig<AMGX_device, t_vecPrec, t_matPrec, t_indPrec> >::create_halo_rows_global_indices(Matrix_d &A,
        std::vector<IVector> &halo_rows_row_offsets,
        std::vector<I64Vector> &halo_rows_col_indices,
        std::vector<MVector> &halo_rows_values)
{
    // in this function we assume that we only have 1-ring so far
    int neighbors = A.manager->num_neighbors();

    for (int i = 0; i < neighbors; i++)
    {
        // compute row offsets and nnz count
        int size = A.manager->B2L_maps[i].size();
        halo_rows_row_offsets[i].resize(size + 1);

        if (size > 0)
        {
            int num_blocks = min(4096, (size + 127) / 128);
            write_matrix_rowsize <<< num_blocks, 128>>>(A.manager->B2L_maps[i].raw(), A.row_offsets.raw(), size, halo_rows_row_offsets[i].raw());
            thrust::exclusive_scan(halo_rows_row_offsets[i].begin(), halo_rows_row_offsets[i].begin() + size + 1, halo_rows_row_offsets[i].begin());
            // compute global indices
            int nnz_count = halo_rows_row_offsets[i][size];
            halo_rows_col_indices[i].resize(nnz_count);
            halo_rows_values[i].resize(nnz_count);
            export_matrix_elements_global<32> <<< num_blocks, 128>>>(A.row_offsets.raw(), A.values.raw(), A.get_block_size(), A.manager->B2L_maps[i].raw(), halo_rows_row_offsets[i].raw(), halo_rows_values[i].raw(), A.col_indices.raw(), halo_rows_col_indices[i].raw(), size, A.manager->local_to_global_map.raw(), NULL, A.get_num_rows(), A.manager->base_index());
        }
    }

    cudaCheckError();
}

template <AMGX_VecPrecision t_vecPrec, AMGX_MatPrecision t_matPrec, AMGX_IndPrecision t_indPrec>
void DistributedArranger<TemplateConfig<AMGX_device, t_vecPrec, t_matPrec, t_indPrec> >::create_halo_btl(Matrix_d &A, INDEX_TYPE rings, INDEX_TYPE neighbors,
        Vector<ivec_value_type_h> &neighbors_list,
        int64_t base_index, INDEX_TYPE index_range,
        INDEX_TYPE global_id, std::vector<IVector > &B2L_maps,
        std::vector<std::vector<VecInt_t> > &B2L_rings,
        std::vector<Manager_d > **halo_btl_p,

        std::vector<IVector *> &halo_lists
                                                                                                        )
{
    *halo_btl_p = new std::vector<DistributedManager<TConfig> >(neighbors);
    std::vector<DistributedManager<TConfig> > &halo_btl = **halo_btl_p;
    /*
    //calling constructors on the allocated memory
    for(int i=0; i<neighbors; i++)
        new (&halo_btl[i]) DistributedManager<TConfig>();
    */

    if (this->halo_coloring != LAST)
    {
        //If we are sending matrices over
        for (int i = 0; i < neighbors; i++)
        {
            //
            // Transfer indexing data too (global indices of exported rows, and sizes of the halo rings)
            //
            /*
            EXAMPLE
            For example partition 1, with 2 rings, and the following B2L and L2H (called halo_lists here) info
            num_neighbors=2; neighbors = [0 2]
            B2L_rings[[0 2 4][0 2 4]] B2L_maps[[0 1| 2 3][1 3| 0 2]]
            L2H_maps (halo_lists) [[4 5][6 7]]
            */
            int size = B2L_rings[i][rings];
            halo_btl[i].set_global_id(global_id);
            halo_btl[i].set_base_index(0);
            halo_btl[i].resize(1, rings);
            halo_btl[i].B2L_maps[0].resize(size);
            halo_btl[i].L2H_maps.resize(1);
            halo_btl[i].L2H_maps[0].resize(halo_lists[i]->size());
            //send over B2L and L2H maps corresponding to current neighbor, so that it will be able to make sense of indices (local to this partition)
            thrust::copy(B2L_maps[i].begin(), B2L_maps[i].end(), halo_btl[i].B2L_maps[0].begin());
            thrust::copy(halo_lists[i]->begin(), halo_lists[i]->end(), halo_btl[i].L2H_maps[0].begin());
            int max_index = thrust::reduce(halo_lists[i]->begin(), halo_lists[i]->end(), (int)0, thrust::maximum<int>()) + 1;
            max_index = max_index > A.get_num_rows() ? max_index : A.get_num_rows();
            halo_btl[i].set_index_range(max_index);
            thrust::copy(B2L_rings[i].begin(), B2L_rings[i].end(), halo_btl[i].B2L_rings[0].begin());
            /*
             EXAMPLE
             These fields are going to be:
             halo_btl[0] (for neighbor ID 0)
             global_id = 1; base_index=0; index_range=6 B2L_rings[0] = [0 2 4] B2L_maps[0] = [0 1| 2 3] L2H_maps = [4 5]
             halo_btl[1] (for neighbor ID 2)
            global_id = 1; base_index=0; index_range=8 B2L_rings[0] = [0 2 4] B2L_maps[0] = [1 3| 0 2] L2H_maps = [6 7]
             */
        }
    }
    else
    {
        if (rings > 1) { FatalError("halo_coloring=LAST dummy halo rows creation not implemented", AMGX_ERR_NOT_IMPLEMENTED); }

        //Create dummy halo matrices
        for (int i = 0; i < neighbors; i++)
        {
            int size = halo_lists[i]->size();
            halo_btl[i].set_global_id(neighbors_list[i]);
            halo_btl[i].set_base_index(0);
            halo_btl[i].resize(1, rings);
            halo_btl[i].B2L_maps[0].resize(size);
            halo_btl[i].L2H_maps.resize(1);
            halo_btl[i].L2H_maps[0].resize(B2L_rings[i][rings]);
            //Even though in this case matrices are not being sent over, we create the same data structures, so they can be handled the same way
            thrust::copy(B2L_maps[i].begin(), B2L_maps[i].end(), halo_btl[i].L2H_maps[0].begin());
            thrust::copy(halo_lists[i]->begin(), halo_lists[i]->end(), halo_btl[i].B2L_maps[0].begin());
            int max_index = thrust::reduce(halo_lists[i]->begin(), halo_lists[i]->end(), (int)0, thrust::maximum<int>()) + 1;
            max_index = max_index > A.get_num_rows() ? max_index : A.get_num_rows();
            halo_btl[i].set_index_range(max_index);
            halo_btl[i].B2L_rings[0][0] = 0;
            halo_btl[i].B2L_rings[0][1] = size;
        }
    }

    cudaCheckError();
}

// CUDA 4.2
template <AMGX_VecPrecision t_vecPrec, AMGX_MatPrecision t_matPrec, AMGX_IndPrecision t_indPrec>
void DistributedArranger<TemplateConfig<AMGX_device, t_vecPrec, t_matPrec, t_indPrec> >::create_halo_btl(Matrix_d &A,
        std::vector<IVector *> &halo_lists)
{
    int neighbors = A.manager->num_neighbors();

    if (neighbors == 0)
    {
        return;
    }

    int rings = (A.manager->B2L_rings.size() - 1) / neighbors;
    A.manager->halo_btl = new std::vector<DistributedManager<TConfig> >(neighbors);
    std::vector<DistributedManager<TConfig> > &halo_btl = *A.manager->halo_btl;

    if (this->halo_coloring != LAST)
    {
        for (int i = 0; i < neighbors; i++)
        {
            //
            // Transfer indexing data too (global indices of exported rows, and sizes of the halo rings)
            //
            int size = A.manager->B2L_rings[i][rings];
            halo_btl[i].set_global_id(A.manager->global_id());
            halo_btl[i].set_base_index(0);
            //Adding one more ring to store halo numbering
            halo_btl[i].resize(1, rings);
            halo_btl[i].B2L_maps[0].resize(size);
            halo_btl[i].L2H_maps.resize(1);
            halo_btl[i].L2H_maps[0].resize(halo_lists[i]->size());
            thrust::copy(A.manager->B2L_maps[i].begin(), A.manager->B2L_maps[i].end(), halo_btl[i].B2L_maps[0].begin());
            thrust::copy(halo_lists[i]->begin(), halo_lists[i]->end(), halo_btl[i].L2H_maps[0].begin());
            int max_index = thrust::reduce(halo_lists[i]->begin(), halo_lists[i]->end(), (int)0, thrust::maximum<int>()) + 1;
            max_index = max_index > A.get_num_rows() ? max_index : A.get_num_rows();
            halo_btl[i].set_index_range(max_index);
            thrust::copy(A.manager->B2L_rings[i].begin(), A.manager->B2L_rings[i].end(), halo_btl[i].B2L_rings[0].begin());
        }
    }
    else
    {
        if (rings > 1) { FatalError("halo_coloring=LAST dummy halo rows creation not implemented", AMGX_ERR_NOT_IMPLEMENTED); }

        //Create dummy halo matrices
        for (int i = 0; i < neighbors; i++)
        {
            int size = halo_lists[i]->size();
            halo_btl[i].set_global_id(A.manager->neighbors[i]);
            halo_btl[i].set_base_index(0);
            //Adding one more ring to store halo numbering
            halo_btl[i].resize(1, rings);
            halo_btl[i].B2L_maps[0].resize(size);
            halo_btl[i].L2H_maps.resize(1);
            halo_btl[i].L2H_maps[0].resize(A.manager->B2L_rings[i][rings]);
            thrust::copy(A.manager->B2L_maps[i].begin(), A.manager->B2L_maps[i].end(), halo_btl[i].L2H_maps[0].begin());
            thrust::copy(halo_lists[i]->begin(), halo_lists[i]->end(), halo_btl[i].B2L_maps[0].begin());
            int max_index = thrust::reduce(halo_lists[i]->begin(), halo_lists[i]->end(), (int)0, thrust::maximum<int>()) + 1;
            max_index = max_index > A.get_num_rows() ? max_index : A.get_num_rows();
            halo_btl[i].set_index_range(max_index);
            halo_btl[i].B2L_rings[0][0] = 0;
            halo_btl[i].B2L_rings[0][1] = size;
        }
    }

    cudaCheckError();
}

template <AMGX_VecPrecision t_vecPrec, AMGX_MatPrecision t_matPrec, AMGX_IndPrecision t_indPrec>
void DistributedArranger<TemplateConfig<AMGX_device, t_vecPrec, t_matPrec, t_indPrec> >::create_halo_btl_multiple_rings(Matrix_d &A, int rings)
{
    // creates halo btl managers
    // required input from A.manager: num_neighbors, B2L_rings, B2L_maps, L2H_maps
    int neighbors = A.manager->num_neighbors();
    A.manager->halo_btl = new std::vector<DistributedManager<TConfig> >(neighbors);
    std::vector<DistributedManager<TConfig> > &halo_btl = *A.manager->halo_btl;

    for (int i = 0; i < neighbors; i++)
    {
        // Transfer indexing data too (global indices of exported rows, and sizes of the halo rings)
        int size = A.manager->B2L_rings[i][rings];
        halo_btl[i].set_global_id(A.manager->global_id());
        halo_btl[i].set_base_index(0);
        // Adding one more ring to store halo numbering
        halo_btl[i].resize(1, rings);
        halo_btl[i].B2L_maps[0].resize(size);
        halo_btl[i].L2H_maps.resize(1);
        halo_btl[i].L2H_maps[0].resize(A.manager->L2H_maps[i].size());
        thrust::copy(A.manager->B2L_maps[i].begin(), A.manager->B2L_maps[i].end(), halo_btl[i].B2L_maps[0].begin());
        thrust::copy(A.manager->L2H_maps[i].begin(), A.manager->L2H_maps[i].end(), halo_btl[i].L2H_maps[0].begin());
        thrust::copy(A.manager->B2L_rings[i].begin(), A.manager->B2L_rings[i].end(), halo_btl[i].B2L_rings[0].begin());
        // compute index range
        int max_index = thrust::reduce(A.manager->L2H_maps[i].begin(), A.manager->L2H_maps[i].end(), (int)0, thrust::maximum<int>()) + 1;
        max_index = max_index > A.get_num_rows() ? max_index : A.get_num_rows();
        halo_btl[i].set_index_range(max_index);
    }

    cudaCheckError();
}

template <AMGX_VecPrecision t_vecPrec, AMGX_MatPrecision t_matPrec, AMGX_IndPrecision t_indPrec>
void DistributedArranger<TemplateConfig<AMGX_device, t_vecPrec, t_matPrec, t_indPrec> >::create_halo_matrices(Matrix_d &A, INDEX_TYPE rings,
        int64_t base_index, INDEX_TYPE index_range,
        INDEX_TYPE global_id, Vector<ivec_value_type_h> &neighbors_,
        std::vector<IVector > &B2L_maps,
        std::vector<std::vector<VecInt_t> > &B2L_rings,
        std::vector<Matrix_d > **halo_rows_p,
        std::vector<Manager_d > **halo_btl_p, std::vector<IVector> &boundary_lists, std::vector<IVector *> &halo_lists)
{
    int neighbors = neighbors_.size();
    this->create_maps(A, rings, neighbors, B2L_maps, B2L_rings, boundary_lists);
    this->create_halo_btl(A, rings, neighbors, neighbors_, base_index, index_range, global_id, B2L_maps, B2L_rings, halo_btl_p, halo_lists);
    this->create_halo_rows(A, rings, neighbors, B2L_maps, B2L_rings, halo_rows_p, halo_lists);
}

template <AMGX_VecPrecision t_vecPrec, AMGX_MatPrecision t_matPrec, AMGX_IndPrecision t_indPrec>
void DistributedArranger<TemplateConfig<AMGX_device, t_vecPrec, t_matPrec, t_indPrec> >::create_B2L(Matrix_d &A, INDEX_TYPE my_id, INDEX_TYPE rings)
{
    this->halo_coloring = A.manager->getComms()->halo_coloring;
    I64Vector_h halo_ranges_h;
    this->create_neighbors(A, my_id, halo_ranges_h);
    A.manager->set_global_id(my_id);
    int num_neighbors = A.manager->neighbors.size();
    A.manager->resize(num_neighbors, rings);
    A.manager->getComms()->set_neighbors(num_neighbors);
    std::vector<IVector> boundary_lists;
    IVector halo_nodes;
    this->create_boundary_lists(A, halo_ranges_h, boundary_lists, halo_nodes);
    A.manager->L2H_maps.resize(num_neighbors);
    std::vector<IVector *> halo_lists;
    halo_lists.resize(num_neighbors);

    for (int i = 0; i < num_neighbors; i++)
    {
        A.manager->L2H_maps[i].resize(boundary_lists[i].size());
        thrust::transform(boundary_lists[i].begin(), boundary_lists[i].end(), thrust::constant_iterator<INDEX_TYPE > (halo_ranges_h[2 * i]), A.manager->L2H_maps[i].begin(), thrust::plus<INDEX_TYPE > ());
        halo_lists[i] = &A.manager->L2H_maps[i];
    }

    cudaCheckError();
    A.manager->getComms()->exchange_vectors(boundary_lists, A, 0);
    this->renumber_to_local(A, boundary_lists, halo_lists, my_id, halo_ranges_h, halo_nodes);
    this->create_halo_matrices(A, 1, boundary_lists, halo_lists);

    if (this->halo_coloring != LAST)
    {
        A.manager->getComms()->exchange_matrix_halo(*A.manager->halo_rows, *A.manager->halo_btl, A);
    }

    A.manager->A = &A;
}

template <AMGX_VecPrecision t_vecPrec, AMGX_MatPrecision t_matPrec, AMGX_IndPrecision t_indPrec>
void DistributedArranger<TemplateConfig<AMGX_device, t_vecPrec, t_matPrec, t_indPrec> >::create_halo_matrices(Matrix_d &A, INDEX_TYPE rings, std::vector<IVector> &boundary_lists, std::vector<IVector *> &halo_lists)
{
    this->create_maps(A, rings, boundary_lists);
    this->create_halo_btl(A, halo_lists);
    this->create_halo_rows(A, halo_lists);
}

template <AMGX_VecPrecision t_vecPrec, AMGX_MatPrecision t_matPrec, AMGX_IndPrecision t_indPrec>
void DistributedArranger<TemplateConfig<AMGX_host, t_vecPrec, t_matPrec, t_indPrec> >::create_B2L(Matrix_h &A, INDEX_TYPE my_id, INDEX_TYPE rings)
{
}

/****************************************
 * Explict instantiations
 ***************************************/
#define AMGX_CASE_LINE(CASE) template class DistributedArrangerBase<TemplateMode<CASE>::Type>;
AMGX_FORALL_BUILDS(AMGX_CASE_LINE)
AMGX_FORCOMPLEX_BUILDS(AMGX_CASE_LINE)
#undef AMGX_CASE_LINE

#define AMGX_CASE_LINE(CASE) template class DistributedArranger<TemplateMode<CASE>::Type>;
AMGX_FORALL_BUILDS(AMGX_CASE_LINE)
AMGX_FORCOMPLEX_BUILDS(AMGX_CASE_LINE)
#undef AMGX_CASE_LINE

} // namespace amgx




