/* Copyright (c) 2011-2019, NVIDIA CORPORATION. All rights reserved.
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

#include <distributed/distributed_manager.h>
#include <distributed/comms_mpi_gpudirect.h>
#include <distributed/comms_mpi_hostbuffer_stream.h>
#include <distributed/comms_visitors.h>
#include <thrust/scan.h>
#include <thrust/sort.h>
#include <thrust/remove.h>
#include <thrust/unique.h>
#include <thrust/binary_search.h>
#include <thrust/iterator/constant_iterator.h>
#include <thrust_wrapper.h>
#include <basic_types.h>
#include <error.h>
#include <util.h>
#include <types.h>
#include <iostream>
#include <sstream>
#include <fstream>
#include <assert.h>
#include "cuda_runtime.h"
#include "reorder_partition.h"

#include "amgx_types/util.h"

#include <algorithm>
#include <iostream> //debug only:

struct is_my_part : public thrust::unary_function<int, bool>
{
    const int _my_part;
    is_my_part(int my_part) : _my_part(my_part) { }

    __host__ __device__
    bool operator()(const int part)
    {
        return (part == _my_part);
    }
};

using namespace std;
namespace amgx
{

static int insertDiagonals = 1;

template <typename index_type>
static __device__ __forceinline__
index_type internal_index(index_type i, index_type j, index_type k, index_type nx, index_type ny, index_type nz)
{
    return k * (nx * ny) + j * nx + i;
}

template <typename index_type>
static __device__ __forceinline__
int64_t get_global_offset(index_type p, index_type q, index_type r, index_type P, index_type Q, index_type R, index_type num_rows)
{
    int rank_id = r * (P * Q) + q * P + p;
    return ((int64_t) rank_id) * ((int64_t) num_rows);
}

template <typename index_type>
__global__
void poisson7pt_count_row_len(index_type *row_len, index_type nx, index_type ny, index_type nz, index_type p, index_type q, index_type r, index_type P, index_type Q, index_type R, index_type num_rows)
{
    for (int tidx = threadIdx.x + blockIdx.x * blockDim.x; tidx < num_rows ; tidx += blockDim.x * gridDim.x)
    {
        /* compute p,q,r from P,Q,R and myid */
        int i = tidx % nx; // Position in x direction
        int j = (( tidx - i) / nx) % ny; // Position in y
        int k = ( tidx - i - nx * j) / ( nx * ny ); // Position in z
        int substract = ((i == 0) && (p == 0));
        substract += ((i == nx - 1) && (p == P - 1));
        substract += ((j == 0) && (q == 0));
        substract += ((j == ny - 1) && (q == Q - 1));
        substract += ((k == 0) && (r == 0));
        substract += ((k == nz - 1) && (r == R - 1));
        // Store 7 in position (num_rows+1), such that row_len[num_rows+1] = 0
        //substract = (tidx == num_rows+1) ? 7 : substract;
        row_len[tidx] = 7 - substract;
    }
}

template <typename index_type, typename mat_value_type>
__global__
void poisson7pt_set_col_values(const index_type *__restrict__ row_offsets, index_type *__restrict__ col_indices, mat_value_type *__restrict__  values, index_type *__restrict__ diag, int64_t *__restrict__ local_to_global, index_type nx, index_type ny, index_type nz, index_type p, index_type q, index_type r, index_type P, index_type Q, index_type R, index_type num_rows)
{
    for (int row = threadIdx.x + blockIdx.x * blockDim.x; row < num_rows ; row += blockDim.x * gridDim.x)
    {
        /* compute p,q,r from P,Q,R and myid */
        int i = row % nx; // Position in x direction
        int j = (( row - i) / nx) % ny; // Position in y
        int k = ( row - i - nx * j) / ( nx * ny ); // Position in z
        int halo_offset = num_rows;
        int pos = row_offsets[row];
        // Diagonal element
        diag[row] = pos;
        col_indices[pos] = row;
        values[pos++] = types::util<mat_value_type>::get_one() * 6.;

        // ----------------------------
        // Neighbor at position i-1
        // ----------------------------
        if (i)
        {
            // Has a i-1 neighbor, which is an internal node at position (i-1,j,k)
            col_indices[pos] = internal_index(i - 1, j, k, nx, ny, nz);
            values[pos++] = types::util<mat_value_type>::invert(types::util<mat_value_type>::get_one());
        }
        else if (p)
        {
            // Has a i-1 neighbor, which is a halo node
            int halo_index = halo_offset + k * ny + j;
            col_indices[pos] = halo_index;
            values[pos++] = types::util<mat_value_type>::invert(types::util<mat_value_type>::get_one());
            int64_t global_offset = get_global_offset(p - 1, q, r, P, Q, R, num_rows);
            local_to_global[halo_index - num_rows] = global_offset + internal_index(nx - 1, j, k, nx, ny, nz);
        }

        if (p)
        {
            halo_offset += ny * nz;
        }

        // ----------------------------
        // Neighbor at position i+1
        // ----------------------------
        if (i < nx - 1)
        {
            // Has i+1 neighbor, which is an internal node at position (i+1,j,k)
            col_indices[pos] = internal_index(i + 1, j, k, nx, ny, nz);
            values[pos++] = types::util<mat_value_type>::invert(types::util<mat_value_type>::get_one());
        }
        else
        {
            if (p < P - 1)
            {
                // Has i+1 neighbor, which is a halo node
                int halo_index = halo_offset + k * ny + j;
                col_indices[pos] = halo_index;
                values[pos++] = types::util<mat_value_type>::invert(types::util<mat_value_type>::get_one());
                int64_t global_offset = get_global_offset(p + 1, q, r, P, Q, R, num_rows);
                local_to_global[halo_index - num_rows] = global_offset + internal_index(0, j, k, nx, ny, nz);
            }
        }

        if (p < P - 1)
        {
            halo_offset += ny * nz;
        }

        // ----------------------------
        // Neighbor at position j-1
        // ----------------------------
        if (j)
        {
            // Has a j-1 neighbor, which is an internal node at position (i,j-1,k)
            col_indices[pos] = internal_index(i, j - 1, k, nx, ny, nz);
            values[pos++] = types::util<mat_value_type>::invert(types::util<mat_value_type>::get_one());
        }
        else if (q)
        {
            // Has a j-1 neighbor, which is a halo node
            int halo_index = halo_offset + k * nx + i;
            col_indices[pos] = halo_index;
            values[pos++] = types::util<mat_value_type>::invert(types::util<mat_value_type>::get_one());
            int64_t global_offset = get_global_offset(p, q - 1, r, P, Q, R, num_rows);
            local_to_global[halo_index - num_rows] = global_offset + internal_index(i, ny - 1, k, nx, ny, nz);
        }

        if (q)
        {
            halo_offset += nx * nz;
        }

        // ----------------------------
        // Neighbor at position j+1
        // ----------------------------
        if (j < ny - 1)
        {
            // Has a j+1 neighbor, which is an internal node at position (i,j+1,k)
            col_indices[pos] = internal_index(i, j + 1, k, nx, ny, nz);
            values[pos++] = types::util<mat_value_type>::invert(types::util<mat_value_type>::get_one());
        }
        else
        {
            if (q < Q - 1)
            {
                // Has a j+1 neighbor, which is a halo node
                int halo_index = halo_offset + k * nx + i;
                col_indices[pos] = halo_index;
                values[pos++] = types::util<mat_value_type>::invert(types::util<mat_value_type>::get_one());
                int64_t global_offset = get_global_offset(p, q + 1, r, P, Q, R, num_rows);
                local_to_global[halo_index - num_rows] = global_offset + internal_index(i, 0, k, nx, ny, nz);
            }
        }

        if (q < Q - 1)
        {
            halo_offset += nx * nz;
        }

        // ----------------------------
        // Neighbor at position k-1
        // ----------------------------
        if (k)
        {
            // Has a k-1 neighbor, which is an internal node at position (i,j,k-1)
            col_indices[pos] = internal_index(i, j, k - 1, nx, ny, nz);
            values[pos++] = types::util<mat_value_type>::invert(types::util<mat_value_type>::get_one());
        }
        else if (r)
        {
            // Has a k-1 neighbor, which is a halo node
            int halo_index = halo_offset + j * nx + i;
            col_indices[pos] = halo_index;
            values[pos++] = types::util<mat_value_type>::invert(types::util<mat_value_type>::get_one());
            int64_t global_offset = get_global_offset(p, q, r - 1, P, Q, R, num_rows);
            local_to_global[halo_index - num_rows] = global_offset + internal_index(i, j, nz - 1, nx, ny, nz);
        }

        if (r)
        {
            halo_offset += nx * ny;
        }

        // ----------------------------
        // Neighbor at position k+1
        // ----------------------------
        if (k < nz - 1)
        {
            // Has a k+1 neighbor, which is an internal node at position (i,j,k+1)
            col_indices[pos] = internal_index(i, j, k + 1, nx, ny, nz);
            values[pos++] = types::util<mat_value_type>::invert(types::util<mat_value_type>::get_one());
        }
        else
        {
            if (r < R - 1)
            {
                // Has a k+1 neighbor, which is a halo node
                int halo_index = halo_offset + j * nx + i;
                col_indices[pos] = halo_index;
                values[pos++] = types::util<mat_value_type>::invert(types::util<mat_value_type>::get_one());
                int64_t global_offset = get_global_offset(p, q, r + 1, P, Q, R, num_rows);
                local_to_global[halo_index - num_rows] = global_offset + internal_index(i, j, 0, nx, ny, nz);
            }
        }

        if (r < R - 1)
        {
            halo_offset += nx * ny;
        }
    }
}

__global__ void flag_halo_ids_kernel(INDEX_TYPE *flags, INDEX_TYPE *ids, INDEX_TYPE offset, INDEX_TYPE size, INDEX_TYPE upper)
{
    int idx = blockDim.x * blockIdx.x + threadIdx.x;

    while (idx < size)
    {
        flags[ids[idx] - offset] = 1;
        idx += blockDim.x * gridDim.x;
    }
}

__global__ void read_halo_ids_kernel(INDEX_TYPE *flags, INDEX_TYPE *ids, INDEX_TYPE offset, INDEX_TYPE size)
{
    int idx = blockDim.x * blockIdx.x + threadIdx.x;

    while (idx < size)
    {
        ids[idx] = flags[ids[idx] - offset];
        idx += blockDim.x * gridDim.x;
    }
}

template<class T>
__global__ void reorder_vector_values(T *dst, const T *src, const INDEX_TYPE *map, INDEX_TYPE blocksize, INDEX_TYPE num_rows)
{
    int row = blockIdx.x * (blockDim.x / blocksize) + threadIdx.x / blocksize; //vectorised by block size
    int vec_id = threadIdx.x % blocksize;

    if (threadIdx.x >= (blockDim.x / blocksize)*blocksize ) { return; }

    while (row < num_rows)
    {
        dst[map[row]*blocksize + vec_id] = src[row * blocksize + vec_id];
        row += gridDim.x * (blockDim.x / blocksize);
    }
}

template<class T>
__global__ void inverse_reorder_vector_values(T *dst, T *src, INDEX_TYPE *map, INDEX_TYPE blocksize, INDEX_TYPE num_rows)
{
    int row = blockIdx.x * (blockDim.x / blocksize) + threadIdx.x / blocksize;
    int vec_id = threadIdx.x % blocksize;

    if (threadIdx.x >= (blockDim.x / blocksize)*blocksize ) { return; }

    while (row < num_rows)
    {
        dst[row * blocksize + vec_id] = src[map[row] * blocksize + vec_id];
        row += gridDim.x * (blockDim.x / blocksize);
    }
}

__global__ void remove_boundary_kernel(INDEX_TYPE *flags, INDEX_TYPE *maps, INDEX_TYPE size)
{
    int element = blockIdx.x * blockDim.x + threadIdx.x;

    while (element < size)
    {
        flags[maps[element]] = 0; //this won't be a problem, because we are overwriting the same thing
        element += blockDim.x * gridDim.x;
    }
}

__global__ void get_unassigned_kernel(INDEX_TYPE *unassigned_flags, INDEX_TYPE *map, INDEX_TYPE *output, INDEX_TYPE part_size, INDEX_TYPE uf_size )
{
    int idx = blockDim.x * blockIdx.x + threadIdx.x;

    while (idx < part_size)
    {
        if (map[idx] < uf_size)
        {
            if (unassigned_flags[map[idx]] == 0)
            {
                unassigned_flags[map[idx]] = 1;
                output[idx] = 1;
            }
        }

        idx += blockDim.x * gridDim.x;
    }
}

__global__ void set_unassigned_kernel(INDEX_TYPE *part_assigned_flags, INDEX_TYPE *part_num, INDEX_TYPE *map, INDEX_TYPE *renum, INDEX_TYPE part_size, INDEX_TYPE max_element, INDEX_TYPE renum_size /*, INDEX_TYPE rank*/)
{
    int idx = blockDim.x * blockIdx.x + threadIdx.x;

    while (idx < part_size)
    {
        if (map[idx] < renum_size)
        {
            if (part_assigned_flags[idx] == 1)
            {
                renum[map[idx]] = max_element + part_num[idx];
            }

            //also update the B2L map
            map[idx] = renum[map[idx]];
        }

        idx += blockDim.x * gridDim.x;
    }
}

__global__ void renumber_b2l_maps(INDEX_TYPE *map, INDEX_TYPE *renum, INDEX_TYPE part_size, INDEX_TYPE renum_size)
{
    int idx = blockDim.x * blockIdx.x + threadIdx.x;

    while (idx < part_size)
    {
        if (map[idx] < renum_size)
        {
            //update the B2L map
            map[idx] = renum[map[idx]];
            idx += blockDim.x * gridDim.x;
        }
    }
}

__global__ void calc_inverse_renumbering(INDEX_TYPE *renum, INDEX_TYPE *irenum, INDEX_TYPE max_element)
{
    int idx = blockDim.x * blockIdx.x + threadIdx.x;

    while (idx < max_element)
    {
        if (renum[idx] < 0 || renum[idx] >= max_element) { printf("Renumbering error: %d %d\n", renum[idx], max_element); }

        irenum[renum[idx]] = idx;
        idx += blockDim.x * gridDim.x;
    }
}

__global__ void create_halo_mapping(INDEX_TYPE *mapping, INDEX_TYPE *node_list, int64_t base_index, INDEX_TYPE map_offset, INDEX_TYPE size)
{
    int row = blockIdx.x * blockDim.x + threadIdx.x;

    while (row < size)
    {
        int idx = node_list[row] - base_index;
        mapping[idx] = map_offset + row;
        row += blockDim.x * gridDim.x;
    }
}

__global__ void apply_h2l2b_mapping(INDEX_TYPE *mapping, INDEX_TYPE *node_list, int64_t base_index, INDEX_TYPE *b2l_map, INDEX_TYPE size)
{
    int row = blockIdx.x * blockDim.x + threadIdx.x;

    while (row < size)
    {
        int idx = node_list[row] - base_index;
        mapping[idx] = b2l_map[row];
        row += blockDim.x * gridDim.x;
    }
}

template <int coop>
__global__ void map_col_indices_and_count_rowlen(INDEX_TYPE *row_offsets, INDEX_TYPE *col_indices, INDEX_TYPE *row_length,
        INDEX_TYPE *mapping, INDEX_TYPE num_rows, INDEX_TYPE insert_diagonal)
{
    extern __shared__ volatile int reduction[];
    int row = blockIdx.x * blockDim.x / coop + threadIdx.x / coop;
    int coopIdx = threadIdx.x % coop;

    while (row < num_rows)
    {
        int valid = 0;

        for (int idx = row_offsets[row] + coopIdx; idx < row_offsets[row + 1]; idx += coop) //this may look horrible, but I expect low branch divergence, because col indices in a row usually belong to the same partition (or at most one more)
        {
            int colIdx = col_indices[idx];
            int new_col_idx = mapping[colIdx];

            if (new_col_idx >= 0)
            {
                valid++;
                col_indices[idx] = new_col_idx;
            }
            else
            {
                col_indices[idx] = -1;
            }
        }

        reduction[threadIdx.x] = valid;

        for (int s = 2; s > 0; s >>= 1)
        {
            if (coopIdx < s)
            {
                reduction[threadIdx.x] += reduction[threadIdx.x + s];
            }

            __syncthreads();
        }

        if (coopIdx == 0)
        {
            row_length[row] = reduction[threadIdx.x] + insert_diagonal;
        }

        row += gridDim.x * blockDim.x / coop;
    }
}


__global__ void renumber_P_col_indices(INDEX_TYPE *__restrict__ col_indices, const INDEX_TYPE *__restrict__ renum, INDEX_TYPE num_owned_coarse_pts, INDEX_TYPE num_owned_fine_pts)
{
    int idx = blockDim.x * blockIdx.x + threadIdx.x;

    while (idx < num_owned_fine_pts )
    {
        INDEX_TYPE col_id = col_indices[idx];

        if (col_id < num_owned_coarse_pts)
        {
            col_indices[idx] = renum[col_id];
        }

        idx += blockDim.x * gridDim.x;
    }
}

template <int coop, class T>
__global__ void reorder_R_matrix(const INDEX_TYPE *old_rows, const INDEX_TYPE *old_cols, const T *old_vals, const INDEX_TYPE *rows, INDEX_TYPE *cols, T *vals, const INDEX_TYPE *renumbering, INDEX_TYPE bsize, INDEX_TYPE num_rows, INDEX_TYPE num_owned_rows)
{
    int row = blockIdx.x * blockDim.x / coop  + threadIdx.x / coop;
    int coopIdx = threadIdx.x % coop;

    while (row < num_rows)
    {
        INDEX_TYPE src_base = old_rows[row];
        INDEX_TYPE dst_base = row < num_owned_rows ? rows[renumbering[row]] : src_base;

        for (int i = coopIdx; i < old_rows[row + 1]*bsize - src_base * bsize; i += coop)
        {
            vals[dst_base * bsize + i] = old_vals[src_base * bsize + i];
        }

        for (int i = coopIdx; i < old_rows[row + 1] - src_base; i += coop)
        {
            cols[dst_base + i] = old_cols[src_base + i];
        }

        row += blockDim.x * gridDim.x / coop;
    }
}

template <int coop, class T>
__global__ void reorder_whole_matrix(INDEX_TYPE *old_rows, INDEX_TYPE *old_cols, T *old_vals, INDEX_TYPE *rows, INDEX_TYPE *cols, T *vals, INDEX_TYPE *renumbering, INDEX_TYPE bsize, INDEX_TYPE num_rows, INDEX_TYPE insert_diagonal)
{
    int row = blockIdx.x * blockDim.x / coop  + threadIdx.x / coop;
    int coopIdx = threadIdx.x % coop;

    while (row < num_rows)
    {
        INDEX_TYPE src_base = old_rows[row];
        INDEX_TYPE dst_base = rows[renumbering[row]];

        if (insert_diagonal)
        {
            if (coopIdx == 0) { cols[dst_base] = renumbering[row]; }

            for (int i = coopIdx; i < bsize; i += coop)
            {
                vals[dst_base * bsize + i] = old_vals[(old_rows[num_rows] + row) * bsize + i];
            }

            dst_base++;
        }

        for (int i = coopIdx; i < old_rows[row + 1]*bsize - src_base * bsize; i += coop)
        {
            vals[dst_base * bsize + i] = old_vals[src_base * bsize + i];
        }

        for (int i = coopIdx; i < old_rows[row + 1] - src_base; i += coop)
        {
            cols[dst_base + i] = old_cols[src_base + i];
        }

        row += blockDim.x * gridDim.x / coop;
    }
}

template <int coop, class T>
__global__ void replace_values_matrix(const T *src_vals_h, const T *src_diag_h, const INDEX_TYPE *old_rows, const INDEX_TYPE *rows, T *vals, const INDEX_TYPE *renumbering, INDEX_TYPE bsize, INDEX_TYPE num_rows)
{
    int row = blockIdx.x * blockDim.x / coop  + threadIdx.x / coop;
    int coopIdx = threadIdx.x % coop;

    while (row < num_rows)
    {
        INDEX_TYPE src_base = old_rows[row];
        INDEX_TYPE dst_base = rows[renumbering[row]];

        for (int i = coopIdx; i < bsize; i += coop)
        {
            vals[dst_base * bsize + i] = src_diag_h[row * bsize + i];
        }

        dst_base++;

        for (int i = coopIdx; i < old_rows[row + 1]*bsize - src_base * bsize; i += coop)
        {
            vals[dst_base * bsize + i] = src_vals_h[src_base * bsize + i];
        }

        row += blockDim.x * gridDim.x / coop;
    }
}

template <int coop, class T>
__global__ void replace_values_matrix(const T *src_vals_h, const INDEX_TYPE *old_rows, const INDEX_TYPE *rows, T *vals, const INDEX_TYPE *renumbering, INDEX_TYPE bsize, INDEX_TYPE num_rows)
{
    int row = blockIdx.x * blockDim.x / coop  + threadIdx.x / coop;
    int coopIdx = threadIdx.x % coop;

    while (row < num_rows)
    {
        INDEX_TYPE src_base = old_rows[row];
        INDEX_TYPE dst_base = rows[renumbering[row]];

        for (int i = coopIdx; i < old_rows[row + 1]*bsize - src_base * bsize; i += coop)
        {
            vals[dst_base * bsize + i] = src_vals_h[src_base * bsize + i];
        }

        row += blockDim.x * gridDim.x / coop;
    }
}

//TODO: optimize by vectorizing
template <class T>
__global__ void reorder_whole_halo_matrix(INDEX_TYPE *old_rows, INDEX_TYPE *old_cols, T *old_vals, INDEX_TYPE *rows, INDEX_TYPE *cols, T *vals,
        INDEX_TYPE *renumbering, INDEX_TYPE bsize, INDEX_TYPE num_rows, INDEX_TYPE insert_diagonal,
        INDEX_TYPE global_offset, INDEX_TYPE local_offset, INDEX_TYPE halo_rows)
{
    int row = blockIdx.x * blockDim.x + threadIdx.x;

    while (row < num_rows)
    {
        INDEX_TYPE src_base = old_rows[row];
        INDEX_TYPE dst = rows[row];

        if (insert_diagonal)
        {
            cols[dst] = global_offset + row;

            for (int j = 0; j < bsize; j++) { vals[dst * bsize + j] = old_vals[(old_rows[halo_rows - local_offset] + local_offset + row) * bsize + j]; }

            dst++;
        }

        for (int i = 0; i < old_rows[row + 1] - src_base; i++)
        {
            INDEX_TYPE colIdx = old_cols[src_base + i];

            if (colIdx >= 0)
            {
                cols[dst] = colIdx;

                for (int j = 0; j < bsize; j++) { vals[dst * bsize + j] = old_vals[(src_base + i) * bsize + j]; }

                dst++;
            }
        }

        row += blockDim.x * gridDim.x;
    }
}

__global__ void calc_rowlen_reorder(INDEX_TYPE *row_offsets, INDEX_TYPE *row_len, INDEX_TYPE *map, INDEX_TYPE size, INDEX_TYPE insert_diag)
{
    int idx = blockDim.x * blockIdx.x + threadIdx.x;

    while (idx < size)
    {
        row_len[map[idx]] = row_offsets[idx + 1] - row_offsets[idx] + insert_diag;
        idx += blockDim.x * gridDim.x;
    }
}

template <class TConfig >
inline void DistributedManagerBase<TConfig>::set_initialized(IVector &row_offsets)
{
    // For P and R sizes the sizes are fixed at creation
    if(m_fixed_view_size)
    {
        return;
    }

    if (neighbors.size() > 0)
    {
        //distributed: cache num_rows/num_nz for different views
        _num_rows_interior = _num_interior_nodes;
        _num_nz_interior = row_offsets[_num_rows_interior];
        _num_rows_owned = _num_interior_nodes + _num_boundary_nodes;
        _num_nz_owned = row_offsets[_num_rows_owned];
        _num_rows_full = halo_offsets[neighbors.size()];

        if (_num_rows_full >= row_offsets.size())
        {
            _num_nz_full = row_offsets[row_offsets.size() - 1];
        }
        else
        {
            _num_nz_full = row_offsets[_num_rows_full];
        }

        _num_rows_all = halo_offsets[halo_offsets.size() - 1];
        _num_nz_all = _num_nz_full;
    }
    else
    {
        _num_rows_interior = _num_interior_nodes;
        _num_nz_interior = row_offsets[_num_rows_interior];
        _num_rows_owned = _num_interior_nodes;
        _num_nz_owned = row_offsets[_num_rows_owned];
        _num_rows_full = _num_rows_owned;
        _num_nz_full = _num_nz_owned;
        _num_rows_all = _num_rows_owned;
        _num_nz_all = _num_nz_owned;
    }
}

template <class TConfig>
inline DistributedManagerBase<TConfig>::DistributedManagerBase(Matrix<TConfig> &a) :
    m_fine_level_comms(NULL), A(&a), m_pinned_buffer_size(0), m_pinned_buffer(NULL), _num_interior_nodes(0), _num_boundary_nodes(0), _comms(NULL), has_B2L(false),
    neighbors(_neighbors), B2L_maps(_B2L_maps), L2H_maps(_L2H_maps),  B2L_rings(_B2L_rings),
    halo_rows_ref_count(0), halo_btl_ref_count(0), halo_ranges(_halo_ranges), halo_ranges_h(_halo_ranges_h), part_offsets(_part_offsets), part_offsets_h(_part_offsets_h), halo_rows(NULL), halo_btl(NULL), m_fixed_view_size(false)
{
    cudaEventCreate(&comm_event);
    cudaStreamCreateWithFlags(&m_int_stream, cudaStreamNonBlocking);
    cudaStreamCreateWithFlags(&m_bdy_stream, cudaStreamNonBlocking);
    this->createComms(A->getResources());
    int my_id = this->getComms()->get_global_id();
    int num_parts = this->getComms()->get_num_partitions();
    this->set_global_id(my_id);
    this->set_num_partitions(num_parts);
}

template <AMGX_VecPrecision t_vecPrec, AMGX_MatPrecision t_matPrec, AMGX_IndPrecision t_indPrec>
void DistributedManager<TemplateConfig<AMGX_device, t_vecPrec, t_matPrec, t_indPrec> >::generatePoisson7pt(int nx, int ny, int nz, int P, int Q, int R)
{
    int my_id = this->getComms()->get_global_id();
    int p, q, r;

    if (nx < P  || ny < Q || nz < R)
    {
        FatalError("(nx < P) or (ny < Q) or (nz < R) not supported\n", AMGX_ERR_NOT_IMPLEMENTED);
    }

    /* compute p,q,r from P,Q,R and myid */
    p = my_id % P; // Position in x direction
    q = (( my_id - p) / P) % Q; // Position in y
    r = ( my_id - p - P * q) / ( P * Q ); // Position in z
    // Create A.row_indices, A.col_indices, A.values, A.diag
    int num_rows = nx * ny * nz;
    int num_nonzeros = num_rows * 7; // Ignoring any boundary, 7 nnz per row
    int num_substract = 0;

    if (p == 0) { num_substract += ny * nz; }

    if (p == P - 1) { num_substract += ny * nz; }

    if (q == 0) { num_substract += nx * nz; }

    if (q == Q - 1) { num_substract += nx * nz; }

    if (r == 0) { num_substract += nx * ny; }

    if (r == R - 1) { num_substract += nx * ny; }

    num_nonzeros -= num_substract;
    int num_halo_nodes = 2 * (ny * nz + nx * nz + nx * ny) - num_substract;
    this->local_to_global_map.resize(num_halo_nodes);
    this->A->set_initialized(0);
    this->A->resize(0, 0, 0, 1, 1, 1);
    this->A->addProps(CSR);
    this->A->resize(num_rows, num_rows + num_halo_nodes, num_nonzeros, 1, 1, 1);
    const int cta_size = 128;
    const int grid_size = std::min( 4096, (num_rows + cta_size - 1) / cta_size );
    poisson7pt_count_row_len <<< grid_size, cta_size>>>(this->A->row_offsets.raw(), nx, ny, nz, p, q, r, P, Q, R, num_rows);
    thrust_wrapper::exclusive_scan(this->A->row_offsets.begin(), this->A->row_offsets.end(), this->A->row_offsets.begin());
    cudaCheckError();
    // Now set nonzeros columns and values
    // TODO: vectorize this
    const int grid_size2 = std::min( 4096, (num_rows + cta_size - 1) / cta_size );
    poisson7pt_set_col_values <<< grid_size2, cta_size>>>
    (this->A->row_offsets.raw(),
     this->A->col_indices.raw(),
     this->A->values.raw(),
     this->A->diag.raw(),
     this->local_to_global_map.raw(),
     nx, ny, nz,
     p, q, r,
     P, Q, R,
     num_rows);
    cudaCheckError();
    // fill parts_offsets_h
    // All ranks have same number of nodes
    int num_ranks = P * Q * R;
    this->part_offsets_h.resize(num_ranks + 1);
    this->part_offsets_h[0] = (int64_t) 0;

    for (int i = 1; i < num_ranks + 1; i++)
    {
        this->part_offsets_h[i] = this->part_offsets_h[i - 1] + (int64_t) num_rows;
    }

    // Device to host copy
    this->part_offsets = this->part_offsets_h;
    this->num_rows_global = P * Q * R * nx * ny * nz;
//  this->A->set_initialized(1);
}

template <AMGX_VecPrecision t_vecPrec, AMGX_MatPrecision t_matPrec, AMGX_IndPrecision t_indPrec>
template <typename t_colIndex>
void DistributedManager<TemplateConfig<AMGX_device, t_vecPrec, t_matPrec, t_indPrec> >::loadDistributed_SetOffsets(
    int num_ranks, int num_rows_global, const t_colIndex* partition_offsets)
{
    // fill part offsets internal data structures
    this->part_offsets_h.resize(num_ranks + 1);

    for (int i = 0; i <= num_ranks; i++)
    {
        this->part_offsets_h[i] = partition_offsets[i];
    }
    // copy to device
    this->part_offsets = this->part_offsets_h;
    // set num of global rows
    this->num_rows_global = num_rows_global;
    cudaCheckError();
}

template <AMGX_VecPrecision t_vecPrec, AMGX_MatPrecision t_matPrec, AMGX_IndPrecision t_indPrec>
template <typename t_colIndex>
map<t_colIndex, int> DistributedManager<TemplateConfig<AMGX_device, t_vecPrec, t_matPrec, t_indPrec> >::loadDistributed_LocalToGlobal(int num_rows, I64Vector_h &off_diag_cols)
{
    // sort global column indices
    thrust::sort(off_diag_cols.begin(), off_diag_cols.end());
    // find unique columns and set local <-> global mappings
    // 1) Removed unneeded vector 2) Create map on host first, upload later (less thrust calls)
    I64Vector_h local_to_global_h;
    map<t_colIndex, int> global_to_local;        // temporary

    if (off_diag_cols.size() > 0)
    {
        global_to_local[off_diag_cols[0]] = num_rows;
        local_to_global_h.push_back(off_diag_cols[0]);
    }

    for (int i = 1; i < off_diag_cols.size(); i++)
    {
        if (off_diag_cols[i] != off_diag_cols[i - 1])
        {
            global_to_local[off_diag_cols[i]] = num_rows + local_to_global_h.size();
            local_to_global_h.push_back(off_diag_cols[i]);
        }
    }
    // Upload finished map in one piece
    this->local_to_global_map.resize(local_to_global_h.size());
    thrust::copy(local_to_global_h.begin(), local_to_global_h.end(), this->local_to_global_map.begin());
    return global_to_local;
}

template <AMGX_VecPrecision t_vecPrec, AMGX_MatPrecision t_matPrec, AMGX_IndPrecision t_indPrec>
void DistributedManager<TemplateConfig<AMGX_device, t_vecPrec, t_matPrec, t_indPrec> >::loadDistributed_InitLocalMatrix(
    IVector_h local_col_indices,
    int num_rows,
    int num_nonzeros,
    const int block_dimx,
    const int block_dimy,
    const int *row_offsets,
    const mat_value_type *values,
    const void *diag)
{
    // init local matrix
    this->A->set_initialized(0);
    this->A->resize(0, 0, 0, 1, 1, 1);
    this->A->addProps(CSR);

    if (diag)
    {
        this->A->addProps(DIAG);
    }

    this->A->resize(num_rows, num_rows + this->local_to_global_map.size(), num_nonzeros, block_dimx, block_dimy, 1);
    cudaCheckError();
    // set local matrix
    thrust::copy(row_offsets, row_offsets + num_rows + 1, this->A->row_offsets.begin());
    this->A->col_indices = local_col_indices;

    thrust::copy(values, values + num_nonzeros * block_dimx * block_dimy, this->A->values.begin());
    cudaCheckError();

    // setup diagonal
    if (diag)
    {
        cudaMemcpy(this->A->values.raw() + this->A->diagOffset()*this->A->get_block_size(), diag, sizeof(mat_value_type) * num_rows * block_dimx * block_dimy, cudaMemcpyDefault);
    }
    else
    {
        this->A->computeDiagonal();
    }
    cudaCheckError();
}

template <AMGX_VecPrecision t_vecPrec, AMGX_MatPrecision t_matPrec, AMGX_IndPrecision t_indPrec>
template <typename t_colIndex>
void DistributedManager<TemplateConfig<AMGX_device, t_vecPrec, t_matPrec, t_indPrec> >::loadDistributedMatrixPartitionVec(
    int num_rows,
    int num_nonzeros,
    const int block_dimx,
    const int block_dimy,
    const int *row_offsets,
    const t_colIndex *col_indices,
    const mat_value_type *values,
    int num_ranks,
    int num_rows_global,
    const void *diag,
    const int *partition)
{
    // fetch my rank
    int my_id = this->getComms()->get_global_id();
    // setup partition vector
    IVector_h partitionVec(num_rows_global);

    if (partition == NULL)
    {
        IVector_h rowCounts(num_ranks);
        this->getComms()->all_gather(num_rows, rowCounts, 1);

        int p = 0;
        for (int i = 0; i < num_ranks; ++i)
        {
            for (int j = 0; j < rowCounts[i]; ++j)
            {
                partitionVec[p++] = i;
            }
        }
    }
    else
    {
        // use existing partition info
        for (int i = 0; i < num_rows_global; i++)
        {
            partitionVec[i] = partition[i];
        }
    }

    // compute partition offsets (based on number of elements per partition). Will be modified when calculating partition map.
    t_colIndex *partition_offsets = (t_colIndex *)calloc(num_ranks + 1, sizeof(t_colIndex));

    for (int i = 0; i < num_rows_global; i++)
    {
        int pvi = partitionVec[i];
        partition_offsets[pvi + 1]++;
    }

    thrust::inclusive_scan(partition_offsets, partition_offsets + num_ranks + 1, partition_offsets);

    loadDistributed_SetOffsets(num_ranks, num_rows_global, partition_offsets);

    // compute partition map (which tells you how the global elements are mapped into the partitions)
    t_colIndex *partition_map = (t_colIndex *)calloc(num_rows_global, sizeof(t_colIndex));

    for (int i = 0; i < num_rows_global; i++)
    {
        int     pvi = partitionVec[i];
        t_colIndex poi = partition_offsets[pvi];
        partition_map[poi] = i;
        partition_offsets[pvi]++;
    }
    free(partition_offsets);

    // compute the inverse partition map
    t_colIndex *ipartition_map = (t_colIndex *)calloc(num_rows_global, sizeof(t_colIndex));

    for (int i = 0; i < num_rows_global; i++)
    {
        ipartition_map[partition_map[i]] = i;
    }
    free(partition_map);

    int h_cidx_allocated = 0;
    const t_colIndex *h_col_indices_global = (const t_colIndex *)this->getHostPointerForData(col_indices, num_nonzeros * sizeof(t_colIndex), &h_cidx_allocated);
    // gather all off-diag columns
    I64Vector_h off_diag_cols;

    for (int i = 0; i < num_nonzeros; i++)
    {
        if (partitionVec[h_col_indices_global[i]] != my_id)
        {
            off_diag_cols.push_back(ipartition_map[h_col_indices_global[i]]);
        }
    }

    auto global_to_local = loadDistributed_LocalToGlobal<t_colIndex>(num_rows, off_diag_cols);

    // set 1, then scan to compute local row indices
    IVector_h my_indices(num_rows_global);

    for (int i = 0; i < num_nonzeros; i++)
    {
        if (partitionVec[h_col_indices_global[i]] == my_id)     // find my local columns and set to 1
        {
            my_indices[ipartition_map[h_col_indices_global[i]]] = 1;
        }
    }

    thrust::exclusive_scan(my_indices.begin(), my_indices.end(), my_indices.begin());
    // remap colums to local
    IVector_h local_col_indices(num_nonzeros);

    for (int i = 0; i < num_nonzeros; i++)
    {
        if (partitionVec[h_col_indices_global[i]] != my_id)
        {
            // off-diag
            local_col_indices[i] = global_to_local[ipartition_map[h_col_indices_global[i]]];
        }
        else
        {
            // diag
            local_col_indices[i] = my_indices[ipartition_map[h_col_indices_global[i]]];
        }
    }
    free(ipartition_map);

    loadDistributed_InitLocalMatrix(local_col_indices, num_rows, num_nonzeros, block_dimx, block_dimy, row_offsets, values, diag);

    cudaCheckError();

    // don't free possibly allocated pinned buffer, since it could be used later. if it would not - it would be deallocated automatically
    /*if (h_cidx_allocated)
    {
      free((void*)h_col_indices_global);
    }*/
}

template <AMGX_VecPrecision t_vecPrec, AMGX_MatPrecision t_matPrec, AMGX_IndPrecision t_indPrec>
template <typename t_colIndex>
void DistributedManager<TemplateConfig<AMGX_device, t_vecPrec, t_matPrec, t_indPrec> >::loadDistributedMatrixPartitionOffsets(
    int num_rows,
    int num_nonzeros,
    const int block_dimx,
    const int block_dimy,
    const int *row_offsets,
    const t_colIndex *col_indices,
    const mat_value_type *values,
    int num_ranks,
    int num_rows_global,
    const void *diag,
    const t_colIndex *partition_offsets)
{
    // fetch my rank
    int my_id = this->getComms()->get_global_id();
    // sanity check, cheap to perform, and helps prevent harder-to-debug errors later on
    if (!std::is_sorted(partition_offsets, partition_offsets + num_ranks + 1)) {
        FatalError("Partition offsets are not sorted.", AMGX_ERR_BAD_PARAMETERS);
    }
    loadDistributed_SetOffsets(num_ranks, num_rows_global, partition_offsets);

    // Create predicate to determine if a column is in the local diagonal block
    t_colIndex my_first_col = this->part_offsets_h[my_id];
    t_colIndex one_past_my_last_col = this->part_offsets_h[my_id + 1];
    auto in_local_diagonal_block = [my_first_col, one_past_my_last_col](const t_colIndex col_index) {
        return col_index >= my_first_col && col_index < one_past_my_last_col;
    };

    int h_cidx_allocated = 0;
    const t_colIndex *h_col_indices_global = (const t_colIndex *)this->getHostPointerForData(col_indices, num_nonzeros * sizeof(t_colIndex), &h_cidx_allocated);
    // gather all off-diag columns
    I64Vector_h off_diag_cols;
    for (int i = 0; i < num_nonzeros; i++)
    {
        if (!in_local_diagonal_block(h_col_indices_global[i]))
        {
            off_diag_cols.push_back(h_col_indices_global[i]);
        }
    }
    auto global_to_local = loadDistributed_LocalToGlobal<t_colIndex>(num_rows, off_diag_cols);
    // set 1, then scan to compute local row indices
    // "coordinate-shift" columns so they lie in much smaller range of my diagonal indices
    int diagonal_size = this->part_offsets_h[my_id  + 1] - this->part_offsets_h[my_id];
    IVector_h my_indices(diagonal_size);
    for (int i = 0; i < num_nonzeros; i++)
    {
        t_colIndex col_index = h_col_indices_global[i];
        if (in_local_diagonal_block(h_col_indices_global[i]))     // find my local columns and set to 1
        {
            // olumns that are on *my* diag partition cannot have an index from 0..num_rows_global
            // instead, part_offsets_h[my_id] <= col_index < part_offsets[my_id+1]
            col_index -= this->part_offsets_h[my_id];
            my_indices[col_index] = 1;
        }
    }
    thrust::exclusive_scan(my_indices.begin(), my_indices.end(), my_indices.begin());

    // remap colums to local
    IVector_h local_col_indices(num_nonzeros);
    for (int i = 0; i < num_nonzeros; i++)
    {
        t_colIndex col_index = h_col_indices_global[i];
        if (!in_local_diagonal_block(col_index))
        {
            // off-diag
            local_col_indices[i] = global_to_local[col_index];
        }
        else
        {
            // diag
            col_index -= this->part_offsets_h[my_id];
            local_col_indices[i] = my_indices[col_index];
        }
    }
    loadDistributed_InitLocalMatrix(local_col_indices, num_rows, num_nonzeros, block_dimx, block_dimy, row_offsets, values, diag);
}

template <AMGX_VecPrecision t_vecPrec, AMGX_MatPrecision t_matPrec, AMGX_IndPrecision t_indPrec>
template <typename t_colIndex>
void DistributedManager<TemplateConfig<AMGX_device, t_vecPrec, t_matPrec, t_indPrec> >::loadDistributedMatrix(
    int num_rows,
    int num_nonzeros,
    const int block_dimx,
    const int block_dimy,
    const int *row_offsets,
    const t_colIndex *col_indices,
    const mat_value_type *values,
    int num_ranks,
    int num_rows_global,
    const void *diag,
    const MatrixDistribution &dist)
{
    using PI = MatrixDistribution::PartitionInformation;
    switch (dist.getPartitionInformationStyle()) {
        case PI::PartitionVec:
            loadDistributedMatrixPartitionVec(num_rows, num_nonzeros, block_dimx, block_dimy, 
                row_offsets, col_indices, values, num_ranks, num_rows_global, diag, (const int*) dist.getPartitionData());
            break;
        case PI::PartitionOffsets:
            loadDistributedMatrixPartitionOffsets(num_rows, num_nonzeros, block_dimx, block_dimy, 
                row_offsets, col_indices, values, num_ranks, num_rows_global, diag, (const t_colIndex*) dist.getPartitionData());
            break;
        default:
            FatalError("Unsupported partitioning data format used with loadDistributedMatrix", AMGX_ERR_NOT_IMPLEMENTED);
    }
}

template <AMGX_VecPrecision t_vecPrec, AMGX_MatPrecision t_matPrec, AMGX_IndPrecision t_indPrec>
void DistributedManager<TemplateConfig<AMGX_host, t_vecPrec, t_matPrec, t_indPrec> >::renumberMatrixOneRing(int update_neighbours)
{
    FatalError("Distributed classical AMG not implemented on host", AMGX_ERR_NOT_IMPLEMENTED);
}

template <AMGX_VecPrecision t_vecPrec, AMGX_MatPrecision t_matPrec, AMGX_IndPrecision t_indPrec>
void DistributedManager<TemplateConfig<AMGX_device, t_vecPrec, t_matPrec, t_indPrec> >::renumberMatrixOneRing(int update_neighbours)
{
    // Step 1: Using halo_ranges, flag neighbors and at the same time, flag halo_nodes (flag_halo_nodes_local)
    int my_id = this->global_id();
    int num_parts = this->get_num_partitions();
    this->set_base_index(this->part_offsets_h[my_id]);
    this->set_index_range(this->part_offsets_h[my_id + 1] - this->part_offsets_h[my_id]);
    DistributedArranger<TConfig_d> *prep = new DistributedArranger<TConfig_d>;

    // Create/update list of neighbors
    if (update_neighbours)
    {
        typedef typename TConfig::template setVecPrec<AMGX_vecInt64>::Type i64vec_value_type;
        typedef Vector<i64vec_value_type> I64Vector;
        typedef typename Matrix<TConfig>::MVector MVector;
        std::vector<IVector> halo_row_offsets(this->neighbors.size());
        std::vector<I64Vector> halo_global_indices(this->neighbors.size());
        std::vector<MVector> halo_values(this->neighbors.size());
        prep->create_halo_rows_global_indices(*(this->A), halo_row_offsets, halo_global_indices, halo_values);
        prep->update_neighbors_list(*(this->A), this->neighbors, this->halo_ranges_h, this->halo_ranges, this->part_offsets_h, this->part_offsets, halo_row_offsets, halo_global_indices);
    }
    else
    {
        prep->create_neighbors_v2(*(this->A));
    }

    this->getComms()->set_neighbors(this->neighbors.size());
    // Create B2L_maps and L2H_maps
    prep->create_boundary_lists_v3(*(this->A));
    // halo_offsets
    int neighbors = this->A->manager->num_neighbors();
    int A_num_rows, offset;
    this->A->getOffsetAndSizeForView(OWNED, &offset, &A_num_rows);
    this->halo_offsets.resize(neighbors + 1, 0);
    this->halo_offsets[0] = A_num_rows;

    for (int i = 0; i < neighbors; i++)
    {
        this->halo_offsets[i + 1] = this->halo_offsets[i] + this->B2L_maps[i].size();
    }

    this->getComms()->exchange_vectors(this->A->manager->B2L_maps, *(this->A), 0);
    // Initialize B2L_rings
    int num_neighbors = this->neighbors.size();
    this->B2L_rings.resize(num_neighbors);

    for (int i = 0; i < num_neighbors; i++)
    {
        this->B2L_rings[i].resize(2);
        this->B2L_rings[i][0] = 0;
        this->B2L_rings[i][1] = this->B2L_maps[i].size();
    }

    prep->initialize_B2L_maps_offsets(*(this->A), 1);
    delete prep;
    //Use the exchanged halo row matrices and the boundary/halo index lists to renumber the matrix
    // Step 5: renumber all owned rows and columns
    this->reorder_matrix_owned();
    // Step 6: renumber local_to_global_map
    int num_owned_rows = this->A->manager->halo_offsets[0];
    int size_one_ring;
    this->A->getOffsetAndSizeForView(FULL, &offset, &size_one_ring);
    I64Vector_d global_col_indices(size_one_ring);
    thrust::sequence(global_col_indices.begin(), global_col_indices.begin() + num_owned_rows, this->base_index() );
    cudaCheckError();
    global_col_indices.dirtybit = 1;
    this->exchange_halo(global_col_indices, global_col_indices.tag);
    thrust_wrapper::copy(global_col_indices.begin() + num_owned_rows, global_col_indices.begin() + size_one_ring, this->local_to_global_map.begin(), this->get_int_stream(), true);
    cudaCheckError();
}


template <AMGX_VecPrecision t_vecPrec, AMGX_MatPrecision t_matPrec, AMGX_IndPrecision t_indPrec>
void DistributedManager<TemplateConfig<AMGX_host, t_vecPrec, t_matPrec, t_indPrec> >::renumber_P_R(Matrix_h &P, Matrix_h &R, Matrix_h &A_fine)
{
    FatalError("Distributed classical AMG not implemented on host", AMGX_ERR_NOT_IMPLEMENTED);
}

template <AMGX_VecPrecision t_vecPrec, AMGX_MatPrecision t_matPrec, AMGX_IndPrecision t_indPrec>
void DistributedManager<TemplateConfig<AMGX_device, t_vecPrec, t_matPrec, t_indPrec> >::renumber_P_R(Matrix_d &P, Matrix_d &R, Matrix_d &A_fine)
{
    int cta_size = 256;
    int num_owned_fine_pts = A_fine.manager->halo_offsets[0];
    int num_owned_coarse_pts, offset;
    // matrix Ac
    this->A->getOffsetAndSizeForView(OWNED, &offset, &num_owned_coarse_pts);
    // Renumber the owned col indices of P (not the halo columns ,since P.manager was created assunming some other numbering)
    int nnz_owned_fine_pts = P.row_offsets[num_owned_fine_pts];
    int num_blocks_fine = min(4096, (nnz_owned_fine_pts + cta_size - 1) / cta_size);

    if (num_blocks_fine > 0)
    {
        renumber_P_col_indices <<< num_blocks_fine, cta_size>>>(P.col_indices.raw(), this->renumbering.raw(), num_owned_coarse_pts, nnz_owned_fine_pts);
        cudaCheckError();
    }

    // Renumber the B2L_maps of P
    for (int i = 0; i < P.manager->neighbors.size(); i++)
    {
        thrust::copy(thrust::make_permutation_iterator(this->renumbering.begin(), P.manager->B2L_maps[i].begin()),
                     thrust::make_permutation_iterator(this->renumbering.begin(), P.manager->B2L_maps[i].end()),
                     P.manager->B2L_maps[i].begin());
    }

    cudaCheckError();
    // Don't renumber the L2H_maps or the halo
    // Renumber the local_to_global_map of matrix P (since neighbors renumbered their owned rows)
    // Swap owned rows of R
    IVector new_row_offsets(R.row_offsets.size());
    int insert = 0;
    // Only renumber the owned rows
    int num_blocks_owned = min(4096, (num_owned_coarse_pts + cta_size - 1) / cta_size);

    if (num_blocks_owned > 0)
    {
        calc_rowlen_reorder <<< num_blocks_owned, cta_size >>>(R.row_offsets.raw(), new_row_offsets.raw(), this->renumbering.raw(), num_owned_coarse_pts, insert);
        cudaCheckError();
    }

    thrust_wrapper::exclusive_scan(new_row_offsets.begin(), new_row_offsets.begin() + num_owned_coarse_pts + 1, new_row_offsets.begin());
    cudaCheckError();
    // Copy the row_offsets for halo rows
    thrust::copy(R.row_offsets.begin() + num_owned_coarse_pts, R.row_offsets.end(), new_row_offsets.begin() + num_owned_coarse_pts);
    cudaCheckError();
    // Reorder the rows of R (no need to reorder the column indices)
    int new_nnz = new_row_offsets[new_row_offsets.size() - 1];
    int halo_offset = new_row_offsets[num_owned_coarse_pts];
    typedef typename MatPrecisionMap<t_matPrec>::Type ValueTypeA;
    VVector new_values(new_nnz * R.get_block_size(), types::util< ValueTypeA >::get_zero());
    IVector new_col_indices(new_nnz, 0);
    int num_blocks_total = min(4096, (R.get_num_rows() + cta_size - 1) / cta_size);

    if (num_blocks_total > 0)
    {
        reorder_R_matrix <32> <<< num_blocks_total, 512>>>(R.row_offsets.raw(), R.col_indices.raw(), R.values.raw(), new_row_offsets.raw(), new_col_indices.raw(), new_values.raw(), this->renumbering.raw(), R.get_block_size(), R.get_num_rows(), num_owned_coarse_pts);
        cudaCheckError();
    }

    R.col_indices.swap(new_col_indices);
    R.row_offsets.swap(new_row_offsets);
    R.values.swap(new_values);

    // Renumber the local_to_global_map (since neighbors have changed their owned numbering)
    if (P.manager->neighbors.size() != 0)
    {
        int size_one_ring = P.manager->halo_offsets[P.manager->neighbors.size()];
        I64Vector_d global_col_indices(size_one_ring);
        thrust::sequence(global_col_indices.begin(), global_col_indices.begin() + num_owned_coarse_pts, this->base_index());
        cudaCheckError();
        global_col_indices.dirtybit = 1;
        P.manager->exchange_halo(global_col_indices, global_col_indices.tag);
        thrust_wrapper::copy(global_col_indices.begin() + num_owned_coarse_pts, global_col_indices.begin() + size_one_ring, P.manager->local_to_global_map.begin(), this->get_int_stream(), true);
        cudaCheckError();
    }

    DistributedArranger<TConfig_d> *prep = new DistributedArranger<TConfig_d>;
    prep->initialize_B2L_maps_offsets(P, 1);
    delete prep;
}

template <AMGX_VecPrecision t_vecPrec, AMGX_MatPrecision t_matPrec, AMGX_IndPrecision t_indPrec>
void DistributedManager<TemplateConfig<AMGX_device, t_vecPrec, t_matPrec, t_indPrec> >::createOneRingHaloRows()
{
    // Input:
    // A matrix with 1-ring B2L_maps, 1-ring halo_offsets
    // Outputs:
    // A matrix with: 1-ring rows,
    //                2-ring B2L_maps,
    //                2-ring halo_offsets
    //                2-ring neighbors
    // Implement here:
    // Look at function create_B2L_from_maps, which calls create_rings, create_halo_btl, create_halo_rows and comms->exchange_matrix_halo
    DistributedArranger<TConfig_d> *prep = new DistributedArranger<TConfig_d>;
    prep->create_one_ring_halo_rows(*(this->A));
    // I believe this can be removed since we don't use masked SpMV anymore
    prep->createRowsLists(*(this->A), false);
    delete prep;
    // this is not necessary anymore becasue we don't use latency hiding
    // however in future we might want to get back to this in case we want to use latency hiding
    //this->reorder_matrix();
}

template <class TConfig>
inline DistributedManagerBase<TConfig>::DistributedManagerBase(
    Matrix<TConfig> &a,
    INDEX_TYPE allocated_halo_depth,
    INDEX_TYPE num_import_rings,
    int num_neighbors,
    const VecInt_t *neighbors_) : m_fine_level_comms(NULL), A(&a), m_pinned_buffer_size(0), m_pinned_buffer(NULL), _num_interior_nodes(0), _num_boundary_nodes(0), _comms(NULL), has_B2L(false), neighbors(_neighbors), halo_rows_ref_count(0), halo_rows(NULL), halo_btl_ref_count(0), halo_btl(NULL), halo_ranges(_halo_ranges), halo_ranges_h(_halo_ranges_h), part_offsets(_part_offsets), part_offsets_h(_part_offsets_h),
    B2L_maps(_B2L_maps),  L2H_maps(_L2H_maps), B2L_rings(_B2L_rings), m_fixed_view_size(false)
{
    cudaStreamCreateWithFlags(&m_int_stream, cudaStreamNonBlocking);
    cudaStreamCreateWithFlags(&m_bdy_stream, cudaStreamNonBlocking);

    if (num_import_rings != 1)
    {
        FatalError("num_rings > 1 not supported in fine_level consolidation", AMGX_ERR_NOT_IMPLEMENTED);
    }

    if (allocated_halo_depth != 1)
    {
        FatalError("allocated_halo_depth > 1 not supported in fine_level consolidation", AMGX_ERR_NOT_IMPLEMENTED);
    }

    this->set_num_halo_rings(num_import_rings);
    neighbors.resize(num_neighbors);
    cudaMemcpy(neighbors.raw(), neighbors_, num_neighbors * sizeof(VecInt_t), cudaMemcpyDefault);
    cudaCheckError();
}

template <class TConfig>
inline void DistributedManagerBase<TConfig>::cacheMaps(const VecInt_t *b2l_maps, const VecInt_t *b2l_ptrs, const VecInt_t *l2h_maps, const VecInt_t *l2h_ptrs)
{
    int num_neighbors = this->neighbors.size();
    this->cached_B2L_maps.resize(num_neighbors);
    this->cached_L2H_maps.resize(num_neighbors);

    for (int i = 0; i < num_neighbors; i++)
    {
        int size = b2l_ptrs[i + 1] - b2l_ptrs[i];
        this->cached_B2L_maps[i].resize(size);
        int count = 0;

        for (int j = b2l_ptrs[i]; j < b2l_ptrs[i + 1]; j++)
        {
            this->cached_B2L_maps[i][count] = b2l_maps[j];
            count++;
        }

        size = l2h_ptrs[i + 1] - l2h_ptrs[i];
        this->cached_L2H_maps[i].resize(size);
        count = 0;

        for (int j = l2h_ptrs[i]; j < l2h_ptrs[i + 1]; j++)
        {
            this->cached_L2H_maps[i][count] = l2h_maps[j];
            count++;
        }
    }
}

template <class TConfig>
inline void DistributedManagerBase<TConfig>::cacheMapsOneRing()
{
    int num_neighbors = this->neighbors.size();
    this->cached_B2L_maps.resize(num_neighbors);
    this->cached_L2H_maps.resize(num_neighbors);

    for (int i = 0; i < num_neighbors; i++)
    {
        this->cached_B2L_maps[i] = this->B2L_maps[i];
        this->cached_L2H_maps[i] = this->L2H_maps[i];
    }
}

template <class TConfig>
inline void DistributedManagerBase<TConfig>::cacheMapsOneRing(const VecInt_t **b2l_maps, const VecInt_t *b2l_sizes, const VecInt_t **l2h_maps, const VecInt_t *l2h_sizes)
{
    int num_neighbors = this->neighbors.size();
    this->cached_B2L_maps.resize(num_neighbors);
    this->cached_L2H_maps.resize(num_neighbors);
    // buffering in the case of GPU data. This shouldn't much affect performance
    std::vector<VecInt_t *> b2l_buffer, l2h_buffer;
    std::vector<VecInt_t> b2l_sizes_buffer, l2h_sizes_buffer;
    b2l_buffer.resize(num_neighbors);
    l2h_buffer.resize(num_neighbors);
    b2l_sizes_buffer.resize(num_neighbors);
    l2h_sizes_buffer.resize(num_neighbors);
    cudaMemcpy(&(b2l_sizes_buffer[0]), b2l_sizes, sizeof(VecInt_t) * num_neighbors, cudaMemcpyDefault);
    cudaMemcpy(&(l2h_sizes_buffer[0]), l2h_sizes, sizeof(VecInt_t) * num_neighbors, cudaMemcpyDefault);
    cudaMemcpy(&(b2l_buffer[0]), b2l_maps, sizeof(VecInt_t *) * num_neighbors, cudaMemcpyDefault);
    cudaMemcpy(&(l2h_buffer[0]), l2h_maps, sizeof(VecInt_t *) * num_neighbors, cudaMemcpyDefault);

    // caching all of the maps
    for (int i = 0; i < num_neighbors; i++)
    {
        int size = b2l_sizes_buffer[i];
        this->cached_B2L_maps[i].resize(size);
        cudaMemcpy(&(this->cached_B2L_maps[i][0]), b2l_buffer[i], sizeof(VecInt_t) * size, cudaMemcpyDefault);
        cudaCheckError();
        size = l2h_sizes_buffer[i];
        this->cached_L2H_maps[i].resize(size);
        cudaMemcpy(&(this->cached_L2H_maps[i][0]), l2h_buffer[i], sizeof(VecInt_t) * size, cudaMemcpyDefault);
        cudaCheckError();
    }
}

template <class TConfig>
void DistributedManagerBase<TConfig>::uploadMatrix(int n, int nnz, int block_dimx, int block_dimy, const int *row_ptrs, const int *col_indices, const void *data, const void *diag, Matrix<TConfig> &in_A)
{
    this->A->manager->initializeUploadReorderAll(n, nnz, block_dimx, block_dimy, row_ptrs, col_indices, data, diag, *(this->A));
}

template <class TConfig>
void DistributedManagerBase<TConfig>::checkPinnedBuffer(size_t size)
{
    if ((m_pinned_buffer_size < size) && (m_pinned_buffer != NULL))
    {
        cudaFreeHost(m_pinned_buffer);
        m_pinned_buffer = NULL;
        m_pinned_buffer_size = 0;
    }

    if (m_pinned_buffer == NULL)
    {
        m_pinned_buffer_size = (size_t)(size * 1.1);
        cudaMallocHost(&m_pinned_buffer, m_pinned_buffer_size);
    }
}


template <class TConfig>
DistributedManagerBase<TConfig>::~DistributedManagerBase()
{
    if (m_pinned_buffer != NULL)
    {
        cudaFreeHost(m_pinned_buffer);
    }

    destroyComms();
    // from childrens:
    cudaStreamDestroy(this->m_int_stream);
    cudaStreamDestroy(this->m_bdy_stream);

    if (!this->halo_rows_ref_count && this->halo_rows != NULL)
    {
        delete this->halo_rows;
        this->halo_rows = NULL;
    }

    if (!this->halo_btl_ref_count && this->halo_btl != NULL)
    {
        delete this->halo_btl;
        this->halo_btl = NULL;
    }
}

// if pointer is host pointer - returns data. If it is device pointer - copies it to the m_pinned_buffer and returns pointer to m_pinned_buffer
template <class TConfig>
void *DistributedManagerBase<TConfig>::getHostPointerForData(void *ptr, size_t size, int *allocated)
{
    cudaError_t rc;
    cudaPointerAttributes att;
    void *ptr_h;
    cudaCheckError();
    /* WARNING: We may accept the following types of allocation for ptr:
       1. malloc                    [host memory]
       2. cudaMalloc                [device memory]
       3. malloc + cudaHostRegister [AMGX_pin_memory/AMGX_unpin_memory host memory]
       4. cudaHostAlloc             [pinned host memory form the beginning]

       The correct way to conver these cases is the following:
       cudaPointerAttributes att;
       cudaError_t st = cudaPointerGetAttributes(&att, ptr);
       if (st == cudaSuccess) {
         //you are in case 2, 3 or 4.
       }
       else{
         //you are in case 1.
       }
       The following pattern of checks should be implemented
       cudaPointerAttributes att;
       cudaError_t st = cudaPointerGetAttributes(&att, ptr);
       if (st == cudaSuccess) {
         //you are in case 2 or 4.
       }
       else{
         st = cudaHostGetDevicePointer(ptr_on_device, ptr, 0);
         if (st == cudaSuccess){
           //you are in case 3.
         }
         else{
           //you are in case 1.
         }
       }
       The above pattern will be used whenever we need to process input data.

       Obs.: parameter size is in bytes and
             parameter allocated indicates whether memory was allocated
             and needs to be release later on. */
    /*
    // original implementation
    cudaPointerGetAttributes(&att, ptr);
    if (att.hostPointer == NULL)
    {
      checkPinnedBuffer(size);
      cudaMemcpy(m_pinned_buffer, ptr, size, cudaMemcpyDefault);
      return m_pinned_buffer;
    }
    else {
      return ptr;
    }
    */
    *allocated = 0;
    // get pointer to values on the device
    rc = cudaPointerGetAttributes(&att, ptr);

    if (rc == cudaSuccess)
    {
        //you are in case 2 or 4 from the above comment.
        if (att.hostPointer == NULL)
        {
            //you are in case 2
            checkPinnedBuffer(size);
            rc = cudaMemcpy(m_pinned_buffer, ptr, size, cudaMemcpyDefault);

            if (rc != cudaSuccess)
            {
                FatalError("Could not copy into the temporary (host) storage. Try pinning the memory to avoid the cudaMemcpy.", AMGX_ERR_BAD_PARAMETERS);
            }

            ptr_h = m_pinned_buffer;
            *allocated = 1;
        }
        else
        {
            //you are in case 4
            ptr_h = ptr;
        }
    }
    else
    {
        //you are in case 1 or 3 from the above comment
        ptr_h = ptr;
    }

    cudaGetLastError(); //to reset last error

    /* check for null pointers */
    if (ptr_h == NULL)
    {
        FatalError("Result of (host) allocation of required temporary storage is NULL. Try pinning the memory to reduce storage requirements.", AMGX_ERR_BAD_PARAMETERS);
    }

    return ptr_h;
}

// if pointer is host pointer - returns data. If it is device pointer - copies it to the m_pinned_buffer and returns pointer to m_pinned_buffer
template <class TConfig>
const void *DistributedManagerBase<TConfig>::getHostPointerForData(const void *ptr, size_t size, int *allocated)
{
    cudaError_t rc;
    cudaPointerAttributes att;
    void *ptr_h;
    cudaCheckError();
    /* WARNING: We may accept the following types of allocation for ptr:
       1. malloc                    [host memory]
       2. cudaMalloc                [device memory]
       3. malloc + cudaHostRegister [AMGX_pin_memory/AMGX_unpin_memory host memory]
       4. cudaHostAlloc             [pinned host memory form the beginning]

       The correct way to conver these cases is the following:
       cudaPointerAttributes att;
       cudaError_t st = cudaPointerGetAttributes(&att, ptr);
       if (st == cudaSuccess) {
         //you are in case 2, 3 or 4.
       }
       else{
         //you are in case 1.
       }
       The following pattern of checks should be implemented
       cudaPointerAttributes att;
       cudaError_t st = cudaPointerGetAttributes(&att, ptr);
       if (st == cudaSuccess) {
         //you are in case 2 or 4.
       }
       else{
         st = cudaHostGetDevicePointer(ptr_on_device, ptr, 0);
         if (st == cudaSuccess){
           //you are in case 3.
         }
         else{
           //you are in case 1.
         }
       }
       The above pattern will be used whenever we need to process input data.

       Obs.: parameter size is in bytes and
             parameter allocated indicates whether memory was allocated
             and needs to be release later on. */
    *allocated = 0;
    // get pointer to values on the device
    rc = cudaPointerGetAttributes(&att, ptr);

    if (rc == cudaSuccess)
    {
        //you are in case 2 or 4 from the above comment.
        if (att.hostPointer == NULL)
        {
            //you are in case 2
            checkPinnedBuffer(size);
            rc = cudaMemcpy(m_pinned_buffer, ptr, size, cudaMemcpyDefault);

            if (rc != cudaSuccess)
            {
                FatalError("Could not copy into the temporary (host) storage. Try pinning the memory to avoid the cudaMemcpy.", AMGX_ERR_BAD_PARAMETERS);
            }

            ptr_h = m_pinned_buffer;
            *allocated = 1;
            cudaGetLastError(); //to reset last error
            return ptr_h;
        }
        else
        {
            //you are in case 4
            cudaGetLastError(); //to reset last error
            return ptr;
        }
    }
    else
    {
        cudaGetLastError(); //to reset last error
        //you are in case 1 or 3 from the above comment
        return ptr;
    }
}


template <class TConfig>
void *DistributedManagerBase<TConfig>::getDevicePointerForData(void *ptr, size_t size, int *allocated)
{
    cudaError_t rc;
    cudaPointerAttributes att;
    void *ptr_d;
    cudaCheckError();
    /* WARNING: We may accept the following types of allocation for ptr:
       1. malloc                    [host memory]
       2. cudaMalloc                [device memory]
       3. malloc + cudaHostRegister [AMGX_pin_memory/AMGX_unpin_memory host memory]
       4. cudaHostAlloc             [pinned host memory form the beginning]

       The correct way to conver these cases is the following:
       cudaPointerAttributes att;
       cudaError_t st = cudaPointerGetAttributes(&att, ptr);
       if (st == cudaSuccess) {
         //you are in case 2, 3 or 4.
       }
       else{
         //you are in case 1.
       }
       The following pattern of checks should be implemented
       cudaPointerAttributes att;
       cudaError_t st = cudaPointerGetAttributes(&att, ptr);
       if (st == cudaSuccess) {
         //you are in case 2 or 4.
       }
       else{
         st = cudaHostGetDevicePointer(ptr_on_device, ptr, 0);
         if (st == cudaSuccess){
           //you are in case 3.
         }
         else{
           //you are in case 1.
         }
       }
       The above pattern will be used whenever we need to process input data.

       Obs.: parameter size is in bytes and
             parameter allocated indicates whether memory was allocated
             and needs to be release later on. */
    *allocated = 0;
    // get pointer to values on the device
    rc = cudaPointerGetAttributes(&att, ptr);

    if (rc == cudaSuccess)
    {
        //you are in case 2 or 4 from the above comment.
        ptr_d = (void *)att.devicePointer;
    }
    else
    {
        //you are in case 1 or 3 from the above comment
        rc = cudaHostGetDevicePointer(&ptr_d, ptr, 0);

        if (rc != cudaSuccess)
        {
            //you are in case 1
            rc = cudaMalloc(&ptr_d, size);

            if (rc != cudaSuccess)
            {
                FatalError("Could not allocate required temporary storage. Try pinning the memory to reduce storage requirements.", AMGX_ERR_BAD_PARAMETERS);
            }

            rc = cudaMemcpy(ptr_d, ptr, size, cudaMemcpyDefault);

            if (rc != cudaSuccess)
            {
                FatalError("Could not copy into the temporary storage. Try pinning the memory to avoid the cudaMemcpy.", AMGX_ERR_BAD_PARAMETERS);
            }

            *allocated = 1;
        }
    }

    /* check for null pointers */
    if (ptr_d == NULL)
    {
        FatalError("Result of allocation of required temporary storage is NULL. Try pinning the memory to reduce storage requirements.", AMGX_ERR_BAD_PARAMETERS);
    }

    cudaGetLastError(); //to reset last error
    return ptr_d;
}

template <class TConfig>
const void *DistributedManagerBase<TConfig>::getDevicePointerForData(const void *ptr, size_t size, int *allocated)
{
    cudaError_t rc;
    cudaPointerAttributes att;
    void *ptr_d;
    cudaCheckError();
    /* WARNING: We may accept the following types of allocation for ptr:
       1. malloc                    [host memory]
       2. cudaMalloc                [device memory]
       3. malloc + cudaHostRegister [AMGX_pin_memory/AMGX_unpin_memory host memory]
       4. cudaHostAlloc             [pinned host memory form the beginning]

       The correct way to conver these cases is the following:
       cudaPointerAttributes att;
       cudaError_t st = cudaPointerGetAttributes(&att, ptr);
       if (st == cudaSuccess) {
         //you are in case 2, 3 or 4.
       }
       else{
         //you are in case 1.
       }
       The following pattern of checks should be implemented
       cudaPointerAttributes att;
       cudaError_t st = cudaPointerGetAttributes(&att, ptr);
       if (st == cudaSuccess) {
         //you are in case 2 or 4.
       }
       else{
         st = cudaHostGetDevicePointer(ptr_on_device, ptr, 0);
         if (st == cudaSuccess){
           //you are in case 3.
         }
         else{
           //you are in case 1.
         }
       }
       The above pattern will be used whenever we need to process input data.

       Obs.: parameter size is in bytes and
             parameter allocated indicates whether memory was allocated
             and needs to be release later on. */
    *allocated = 0;
    // get pointer to values on the device
    rc = cudaPointerGetAttributes(&att, ptr);

    if (rc == cudaSuccess)
    {
        //you are in case 2 or 4 from the above comment.
        cudaGetLastError(); //to reset last error
        return (const void *)att.devicePointer;
    }
    else
    {
        //you are in case 1 or 3 from the above comment
        rc = cudaHostGetDevicePointer(&ptr_d, (void *)ptr, 0);

        if (rc != cudaSuccess)
        {
            //you are in case 1
            rc = cudaMalloc(&ptr_d, size);

            if (rc != cudaSuccess)
            {
                FatalError("Could not allocate required temporary storage. Try pinning the memory to reduce storage requirements.", AMGX_ERR_BAD_PARAMETERS);
            }

            rc = cudaMemcpy(ptr_d, ptr, size, cudaMemcpyDefault);

            if (rc != cudaSuccess)
            {
                FatalError("Could not copy into the temporary storage. Try pinning the memory to avoid the cudaMemcpy.", AMGX_ERR_BAD_PARAMETERS);
            }

            *allocated = 1;
            cudaGetLastError(); //to reset last error
            return (const void *)ptr_d;
        }
    }

    /* check for null pointers */
    if (ptr_d == NULL)
    {
        FatalError("Result of allocation of required temporary storage is NULL. Try pinning the memory to reduce storage requirements.", AMGX_ERR_BAD_PARAMETERS);
    }

    // shouldn't get there
    cudaGetLastError(); //to reset last error
    return NULL;
}

template <class TConfig>
void initializeMatrixCopyAll(int n, int nnz, int block_dimx, int block_dimy, const int *row_ptrs, const int *col_indices, const void *data, const void *diag, Matrix<TConfig> *A)
{
    typedef typename TConfig::MatPrec  mat_value_type;
    A->resize( n, n, nnz, block_dimx, block_dimy );
    //Upload the entire matrix
    cudaMemcpy( A->row_offsets.raw(), row_ptrs, (n + 1) * sizeof(int), cudaMemcpyDefault );
    cudaCheckError();
    cudaMemcpy( A->col_indices.raw(), col_indices, (nnz) * sizeof(int), cudaMemcpyDefault );
    cudaCheckError();
    cudaMemcpy( A->values.raw(), (mat_value_type *)data, (nnz * block_dimx * block_dimy) * sizeof(mat_value_type), cudaMemcpyDefault );
    cudaCheckError();

    if (diag)
    {
        cudaMemcpy( A->values.raw() + A->diagOffset()*A->get_block_size(), (mat_value_type *)diag, (n * block_dimx * block_dimy) * sizeof(mat_value_type), cudaMemcpyDefault );
    }
    else
    {
        A->computeDiagonal();
    }

    cudaCheckError();
}

template <class TConfig>
void DistributedManagerBase<TConfig>::updateMapsReorder()
{
    int my_id = this->getComms()->get_global_id();
    DistributedComms<TConfig> *comms_tmp = this->getComms();
    DistributedComms<TConfig> **comms_ = &comms_tmp;
    // Copy B2L_maps in their final place
    int num_neighbors = this->neighbors.size();
    B2L_maps.resize(num_neighbors);
    L2H_maps.resize(num_neighbors);

    for (int i = 0; i < num_neighbors; i++)
    {
        B2L_maps[i] = this->cached_B2L_maps[i];
        L2H_maps[i] = this->cached_L2H_maps[i];
    }

    //Create a DistributedArranger object to map further halo rings and to construct halo row matrices and exchange them (if halo_coloring != LAST)
    DistributedArranger<TConfig> *prep = new DistributedArranger<TConfig>;
    prep->create_B2L_from_maps( (*(this->A)), my_id, this->num_halo_rings(), neighbors,
                                B2L_maps, L2H_maps, B2L_rings, comms_, &halo_rows, &halo_btl);
    DistributedManagerBaseInit(my_id, 0, this->A->get_num_rows(), *(this->A), comms_, NULL, NULL);
    //Use the exchanged halo row matrices and the boundary/halo index lists to renumber the matrix
    this->reorder_matrix();
    prep->initialize_B2L_maps_offsets(*(this->A), this->num_halo_rings());
    delete prep;
}

template <class TConfig>
void DistributedManagerBase<TConfig>::initializeUploadReorderAll(int n, int nnz, int block_dimx, int block_dimy, const int *row_ptrs, const int *col_indices, const void *data, const void *diag, Matrix<TConfig> &in_A)
{
    this->A = &in_A;
    initializeMatrixCopyAll<TConfig>(n, nnz, block_dimx, block_dimy, row_ptrs, col_indices, data, diag, this->A);
    this->updateMapsReorder();
}

template <class TConfig>
void DistributedManagerBase<TConfig>::destroyComms()
{
    if ( (this->_comms) != NULL )
    {
        if (this->_comms->decr_ref_count())
        {
            delete (this->_comms);
            this->_comms = NULL;
        }
    }

    if ( (this->m_fine_level_comms) != NULL)
    {
        if (this->m_fine_level_comms->decr_ref_count())
        {
            delete (this->m_fine_level_comms);
            this->m_fine_level_comms = NULL;
        }
    }
}

template <class TConfig>
void DistributedManagerBase<TConfig>::initComms(Resources *rsrc)
{
    this->createComms(rsrc);
    int my_id = this->getComms()->get_global_id();
    int num_parts = this->getComms()->get_num_partitions();
    this->set_global_id(my_id);
    this->set_num_partitions(num_parts);
}


template <class TConfig>
void DistributedManagerBase<TConfig>::createComms(Resources *rsrc)
{
    // create communicator
#ifdef AMGX_WITH_MPI
    destroyComms();
    if (rsrc == NULL)
        FatalError("Resources should not be NULL", AMGX_ERR_INTERNAL);

    MPI_Comm *mpi_comm = rsrc->getMpiComm();
    AMG_Config *cfg = rsrc->getResourcesConfig();
    std::string comm_value, comm_scope;
    cfg->getParameter<std::string>("communicator", comm_value, "default", comm_scope);
    int rank = -1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    if (comm_value == "MPI_DIRECT")
    {
        _comms = new CommsMPIDirect<TConfig>(*cfg, comm_scope, mpi_comm);
        std::string comm_log("Using CUDA-Aware MPI (GPU Direct) communicator...\n");
        amgx_distributed_output(comm_log.c_str(), comm_log.length());
    }
    else if (comm_value == "MPI")
    {
        _comms =  new CommsMPIHostBufferStream<TConfig>(*cfg, comm_scope, mpi_comm);
        std::string comm_log("Using Normal MPI (Hostbuffer) communicator...\n");
        amgx_distributed_output(comm_log.c_str(), comm_log.length());
    }
    else 
    { 
        FatalError("Bad communicator value", AMGX_ERR_BAD_PARAMETERS); 
    }

#endif
}

template <class TConfig>
void DistributedManagerBase<TConfig>::malloc_export_maps(VecInt_t ***b2l_maps_e, VecInt_t **b2l_maps_sizes_e, VecInt_t ***l2h_maps_e, VecInt_t **l2h_maps_sizes_e)
{
    *b2l_maps_e = (VecInt_t **) malloc(sizeof(VecInt_t *)*this->num_neighbors());
    *l2h_maps_e = (VecInt_t **) malloc(sizeof(VecInt_t *)*this->num_neighbors());
    *b2l_maps_sizes_e = (VecInt_t *) malloc(sizeof(VecInt_t) * (this->num_neighbors()));
    *l2h_maps_sizes_e = (VecInt_t *) malloc(sizeof(VecInt_t) * (this->num_neighbors()));

    for (int i = 0; i < this->num_neighbors(); i++)
    {
        (*b2l_maps_sizes_e)[i] = B2L_maps[i].size();
        (*l2h_maps_sizes_e)[i] = L2H_maps[i].size();
        (*b2l_maps_e)[i] =  (VecInt_t *) malloc(sizeof(VecInt_t) * ( (*b2l_maps_sizes_e)[i]) );

        if (L2H_maps[i].size() != 0)
        {
            (*l2h_maps_e)[i] =  (VecInt_t *) malloc(sizeof(VecInt_t) * ( (*l2h_maps_sizes_e)[i]) );
            thrust::copy(L2H_maps[i].begin(), L2H_maps[i].end(), (*l2h_maps_e)[i]);
        }

        thrust::copy(B2L_maps[i].begin(), B2L_maps[i].end(), (*b2l_maps_e)[i]);
    }

    cudaCheckError();
}

template <AMGX_VecPrecision t_vecPrec, AMGX_MatPrecision t_matPrec, AMGX_IndPrecision t_indPrec>
void DistributedManager<TemplateConfig<AMGX_device, t_vecPrec, t_matPrec, t_indPrec> >::createRenumbering(IVector &renumbering)
{
    int num_neighbors = this->neighbors.size();
    // still renumber if the number of neighbors = 0, to support non-symmetric matrices
    // if (num_neighbors == 0) return;
    /*
      EXAMPLE
      Example matrix, partition 1 arrives with state:
      A.row_offsets = [0 4 11 15 20]
      A.col_indices = [4 0 1 2
                       4 5 0 1 2 3 7
                       0 1 2 3
                       1 2 3 6 7]
      num_neighbors=2; neighbors = [0 2]
      B2L_rings[[0 2 4][0 2 4]] B2L_maps[[0 1| 2 3][1 3| 0 2]]
      L2H_maps (and halo_lists) [[4 5][6 7]]
    */
    int size = 0;

    if (this->L2H_maps.size())
    {
        size = thrust_wrapper::reduce(this->A->col_indices.begin(), this->A->col_indices.end(), int(0), thrust::maximum<int>()) + 1; //Sufficient to do reduction on lth maps
        cudaCheckError();
    }
    else
    {
        size = this->A->get_num_rows();
    }

    int rings = (this->B2L_rings.size() > 0) ? this->B2L_rings[0].size() - 1 : 0;
    //initial size to size+1 so we have the total size after a scan
    renumbering.resize(size + 1);
    int global_size = size;
    //
    // Step 1 - in the main matrix, separate interior and boundary nodes (1/0 in flagArray), renumber interior ones with an exclusive scan
    //
    IVector flagArray(size + 1);
    thrust::fill(flagArray.begin(), flagArray.begin() + size + 1, 1);
    cudaCheckError();

    //sets 1 for interior nodes, 0 for boundary node
    for (int i = 0; i < num_neighbors; i++ )
    {
        int size = this->B2L_rings[i][1];
        int num_blocks = min(4096, (size + 127) / 128);

        if (size > 0)
        {
            remove_boundary_kernel <<< num_blocks, 128>>>(flagArray.raw(), this->B2L_maps[i].raw(), size);
        }

        //If there are any L2H maps
        if (this->L2H_maps.size() && this->L2H_maps[i].size())
        {
            int size = this->L2H_maps[i].size();
            int num_blocks = min(4096, (size + 127) / 128);
            remove_boundary_kernel <<< num_blocks, 128>>>(flagArray.raw(), this->L2H_maps[i].raw(), size);
        }

        cudaCheckError();
    }

    //gets the renumbering of interior nodes
    thrust_wrapper::exclusive_scan(flagArray.begin(), flagArray.begin() + size + 1, renumbering.begin());
    cudaCheckError();
    /*
     EXAMPLE
     After removing 1-ring boundary nodes and halo nodes from flagArray: [0 0 1 0 0 0 0 0]
     After exclusive scan, which gives renumbering for interior nodes (only node #2)
     renumbering: [0 0 0 1 1 1 1 1]
     */
    //
    // Step 2 - Renumber nodes that are in the boundary, stepping through each B2L map, and renumbering ones that have not been renumbered yet
    //
    //what is the biggest B2L size
    INDEX_TYPE max_size = 0;

    for (int i = 0; i < num_neighbors; i++)
    {
        max_size = max_size > this->B2L_rings[i][1] ? max_size : this->B2L_rings[i][1];

        if (this->L2H_maps.size())
        {
            max_size = max_size > this->L2H_maps[i].size() ? max_size : this->L2H_maps[i].size();
        }
    }

    //allocate work vectors (should be pretty small) that are used to renumber boundary nodes
    IVector boundary_renum_flags(max_size);
    IVector boundary_renum(max_size);
    //the number of renumbered nodes so far
    int max_element = renumbering[size];
    this->_num_interior_nodes = max_element;
    this->_num_boundary_nodes = this->A->get_num_rows() - max_element;
    renumbering.resize(size);

    /*
     EXAMPLE
     size = 8
     max_size = 2, max_element = 1, num_interior_nodes=1, num_boundary_nodes = 4-1 = 3
     */

    for (int i = 0; i < num_neighbors; i++)
    {
        //find nodes that are part of the current boundary and they haven't been renumbered yet
        thrust::fill(boundary_renum_flags.begin(), boundary_renum_flags.begin() + max_size, 0);
        int size = this->B2L_rings[i][1];
        int num_blocks = min(4096, (size + 191) / 192);

        if (size > 0)
            get_unassigned_kernel <<< num_blocks, 192>>>(flagArray.raw(),
                    this->B2L_maps[i].raw(),
                    boundary_renum_flags.raw(), size, global_size /*,rank*/);

        //calculate the local renumbering (within this boundary region) of these nodes
        thrust_wrapper::exclusive_scan(boundary_renum_flags.begin(), boundary_renum_flags.begin() + max_size, boundary_renum.begin());

        //apply renumbering to the big numbering table
        if (size > 0)
            set_unassigned_kernel <<< num_blocks, 192>>>(boundary_renum_flags.raw(),
                    boundary_renum.raw(),
                    this->B2L_maps[i].raw(),
                    renumbering.raw(),
                    size, max_element, global_size /*,rank*/);

        //update the number of renumbered nodes
        max_element += boundary_renum[max_size - 1] + boundary_renum_flags[max_size - 1];
        /*
         EXAMPLE
         for neighbor 0 (ID 0)
         boundary_renum_flags = [0 0], size = 2, flagArray [0 0 1 0 0 0 0 0]
         get_unassigned_kernel's output: boundary_renum_flags = [1 1] flagArray [1 1 1 0 0 0 0 0]
         after exclusive scan: boundary_renum [0 1]
         set_unassigned_kernel updates these arrays and renumbers B2L map:
         renumbering = [1 2 0 1 1 1 1 1] B2L_maps[0] = [1 2| 2 3] (note that after element 3 in renumbering and after element 2 we have invalid/not yet updated values)
         max_element = 3

         for neighbor 1 (ID 2)
         get_unassigned_kernels's output: boundary_renum_flags [0 1] flagArray [1 1 1 1 0 0 0 0]
         after exclusive scan boundary_renum [0 0]
         set_unassigned_kernel renumbering [1 2 0 3 1 1 1 1] B2L_maps[1] = [2 3| 0 2]
         max_element = 4
         */
    }

    cudaCheckError();

    //Get renumbering for halo indices
    if (this->L2H_maps.size())
    {
        //TODO: simplify this, we don't need to check whether it has already been renumbered, there is no overlap between halos
        for (int i = 0; i < num_neighbors; i++)
        {
            //find nodes that are part of the current boundary and they haven't been renumbered yet
            thrust::fill(boundary_renum_flags.begin(), boundary_renum_flags.begin() + max_size, 0);
            int size = this->L2H_maps[i].size();
            int num_blocks = min(4096, (size + 191) / 192);

            if (size > 0)
                get_unassigned_kernel <<< num_blocks, 192>>>(flagArray.raw(),
                        this->L2H_maps[i].raw(),
                        boundary_renum_flags.raw(), size, global_size /*,rank*/);

            //calculate the local renumbering (within this boundary region) of these nodes
            thrust_wrapper::exclusive_scan(boundary_renum_flags.begin(), boundary_renum_flags.begin() + max_size, boundary_renum.begin());

            //apply renumbering to the big numbering table
            if (size > 0)
                set_unassigned_kernel <<< num_blocks, 192>>>(boundary_renum_flags.raw(),
                        boundary_renum.raw(),
                        this->L2H_maps[i].raw(),
                        renumbering.raw(),
                        size, max_element, global_size /*,rank*/);

            //update the number of renumbered nodes
            max_element += boundary_renum[max_size - 1] + boundary_renum_flags[max_size - 1];
            /*
             EXAMPLE
             for neighbor 0 (ID 0)
             boundary_renum_flags = [0 0], size = 2, flagArray [1 1 1 1 0 0 0 0]
             get_unassigned_kernel's output: boundary_renum_flags = [1 1] flagArray [1 1 1 1 1 1 0 0]
             after exclusive scan: boundary_renum [0 1]
             set_unassigned_kernel updates these arrays and renumbers B2L map:
             renumbering = [1 2 0 3 4 5 1 1] L2H_maps[0] = [4 5]
             max_element = 6

             for neighbor 1 (ID 2)
             get_unassigned_kernels's output: boundary_renum_flags [1 1] flagArray [1 1 1 1 1 1 1 1]
             after exclusive scan boundary_renum [0 1]
             set_unassigned_kernel renumbering = [1 2 0 3 4 5 6 7] L2H_maps[1] = [6 7]
             max_element = 8
             */
        }

        cudaCheckError();
    }

    //apply renumbering to further halo rings too
    if (rings > 1)
    {
        for (int i = 0; i < num_neighbors; i++)
        {
            int size = this->B2L_rings[i][this->B2L_rings[i].size() - 1] - this->B2L_rings[i][1];
            int num_blocks = min(4096, (size + 127) / 128);
            renumber_b2l_maps <<< num_blocks, 128>>>(this->B2L_maps[i].raw() + this->B2L_rings[i][1], renumbering.raw(), size, global_size /*, rank*/);
        }

        cudaCheckError();
    }

    /*
     EXAMPLE
     renumbers further boundary rings as listed in B2L_maps, since they have not been replaced yet with their renumbered values
     B2L_maps [[1 2| 0 3][2 3| 1 0]]
     */
}

template <AMGX_VecPrecision t_vecPrec, AMGX_MatPrecision t_matPrec, AMGX_IndPrecision t_indPrec>
void DistributedManager<TemplateConfig<AMGX_device, t_vecPrec, t_matPrec, t_indPrec> >::reorder_matrix_owned()
{
    int num_neighbors = this->neighbors.size();
    int size = this->A->get_num_rows();
    int num_blocks = min(4096, (size + 511) / 512);
    int rings = (this->B2L_rings.size() > 0) ? this->B2L_rings[0].size() - 1 : 0;
    this->set_num_halo_rings(rings);
    int diag = this->A->hasProps(DIAG);

    if (diag)
    {
        FatalError("External diag not supported in classical path", AMGX_ERR_NOT_IMPLEMENTED);
    }

//
// Step 1 & 2 - create renumbering
//
    this->createRenumbering(this->renumbering);
    //now we have the full renumbering table in renum, calculate the inverse
    this->inverse_renumbering.resize(this->renumbering.size());

    if (this->renumbering.size() > 1)
    {
        calc_inverse_renumbering <<< min(4096, ((int)this->renumbering.size() + 511) / 512), 512 >>> (this->renumbering.raw(), this->inverse_renumbering.raw(), this->renumbering.size());
        cudaCheckError();
    }

//
// Step 4 - calculate number/offset of nodes in the halos from the neighbors, ring by ring
//
    this->halo_offsets.resize(num_neighbors + 1);
    this->halo_offsets[0] = size;

    for (int i = 0; i < num_neighbors; i++)
    {
        this->halo_offsets[i + 1] = this->halo_offsets[i] + this->L2H_maps[i].size();
    }

    this->set_num_halo_rows(this->halo_offsets[this->halo_offsets.size() - 1] - size);
    int nh = this->num_halo_rows();
    int total_rows = size + nh;
    cudaCheckError();
//
// Step 6 - renumber halo matrices and calculate row length (to eventually append to the big matrix)
//
    int insert = 0;
    //recalculate row_offsets
    IVector new_row_offsets(size + 1);

    if (num_blocks > 0)
    {
        calc_rowlen_reorder <<< num_blocks, 512>>>(this->A->row_offsets.raw(), new_row_offsets.raw(), this->renumbering.raw(), size, insert);
        cudaCheckError();
    }

    thrust::copy(thrust::make_permutation_iterator(this->renumbering.begin(), this->A->col_indices.begin()),
                 thrust::make_permutation_iterator(this->renumbering.begin(), this->A->col_indices.end()),
                 this->A->col_indices.begin());
    cudaCheckError();
    //row_offsets array created by exclusive scan of row sizes
    thrust_wrapper::exclusive_scan(new_row_offsets.begin(), new_row_offsets.begin() + size + 1, new_row_offsets.begin());
    cudaCheckError();
//
// Step 7 - consolidate column indices and values
//
    int new_nnz = new_row_offsets[new_row_offsets.size() - 1];
    typedef typename MatPrecisionMap<t_matPrec>::Type ValueTypeA;
    VVector new_values((new_nnz + 1 )* this->A->get_block_size(), types::util<ValueTypeA>::get_zero());
    IVector new_col_indices(new_nnz, 0);

    //reorder based on row permutation
    if (num_blocks > 0)
    {
        reorder_whole_matrix <32> <<< num_blocks, 512>>>(this->A->row_offsets.raw(), this->A->col_indices.raw(), this->A->values.raw(), new_row_offsets.raw(), new_col_indices.raw(), new_values.raw(), this->renumbering.raw(), this->A->get_block_size(), size, insert);
        cudaCheckError();
    }

    //create and append halo rows size
    //create an identity matrix in CSR format
    int nnz = this->A->get_num_nz();
    IVector identity_csr_rows(nh + 1);
    IVector identity_csr_cols(nh);
    VVector identity_csr_vals(nh, types::util<ValueTypeA>::get_one()); //needs to be changed to MVector, but this definition is messed up in the header file (should fix later)
    thrust::sequence(identity_csr_rows.begin(), identity_csr_rows.end());
    thrust::sequence(identity_csr_cols.begin(), identity_csr_cols.end());
    /*for example, 2x2 identity_csr matrix is created:
      identity_csr_rows = {   0,   1,   2  }
      identity_csr_cols = {   0,   1 }
      identity_csr_vals = { 1.0, 1.0 } */
    //shift identity tmatrix by size = this->A->get_num_rows();
    thrust::transform(identity_csr_rows.begin(), identity_csr_rows.end(), thrust::constant_iterator<INDEX_TYPE>(nnz), identity_csr_rows.begin(), thrust::plus<INDEX_TYPE>());
    thrust::transform(identity_csr_cols.begin(), identity_csr_cols.end(), thrust::constant_iterator<INDEX_TYPE>(size), identity_csr_cols.begin(), thrust::plus<INDEX_TYPE>());
    /*for example, 2x2 identity_csr matrix is created:
      identity_csr_rows = {   0,   1,   2  }
      identity_csr_cols = {size, size+1 }
      identity_csr_vals = { 1.0, 1.0 } */
    /* WARNING: you must be very careful with the view you are setting (cuurently the view coming here by default is ALL = FULL). If
                - classical path is selected then the createOneRingHaloRows -> create_one_ring_halo_rows -> append_halo_rows
                routine will be called. It will overwrite the halo rows setup here (and will use view OWNED, which will ignore the
                halo rows setup here, to determine how the new halo rows should be placed).
                - aggregation path is selected then the extra rows setup here will be used in the R*A*P product, where (in order to match
                dimensions of R and P) it is assumed that (the local partition) matrix A is square, therefore it must be padded by identity
                rows at the bottom to compensate for the "extra" columns that are outside of the main square part. The old routines for the
                aggregation path do this padding at the end of the reorder_matrix routine below.  */
    //ViewType v = this->A->currentView();
    //this->A->setView(ALL);
    //Approach 1: use existing routine to append the identity matrix to the existing one
    //            (seems like too much overhead, also need identity matrix per neighbor)
    //DistributedArranger<TConfig_d> *prep = new DistributedArranger<TConfig_d>;
    //prep->append_halo_rows(this->A, identity_csr_rows, identity_csr_cols, identity_csr_vals);
    //delete prep;
    //Approach 2: custom for this routine
    new_row_offsets.resize(total_rows + 1);
    new_col_indices.resize(nnz + nh);
    new_values.resize(nnz + nh + 1); //extra 1 element stores zero at the end (to follow the original design)
    //new_values[nnz]=-1;        //marker to track the last element
    thrust::copy(identity_csr_rows.begin(), identity_csr_rows.end(), new_row_offsets.begin() + size );
    thrust::copy(identity_csr_cols.begin(), identity_csr_cols.end(), new_col_indices.begin() + nnz);
    thrust::copy(new_values.begin() + nnz,    new_values.begin() + nnz + 1, new_values.begin() + nnz + nh);
    thrust::copy(identity_csr_vals.begin(), identity_csr_vals.end(),  new_values.begin() + nnz);
    /* WARNING: see above. */
    this->A->set_num_cols(total_rows);
    this->A->set_num_rows(total_rows);
    this->A->col_indices.swap(new_col_indices);
    new_row_offsets.resize(total_rows + 1);
    this->A->row_offsets.swap(new_row_offsets);
    new_row_offsets.swap(this->old_row_offsets);
    this->A->values.swap(new_values);
    this->A->m_seq_offsets.resize(total_rows + 1);
    thrust::sequence(this->A->m_seq_offsets.begin(), this->A->m_seq_offsets.end());
    cudaCheckError();
    //TODO: only do this if AMG_Config matrix_halo_exchange!=2
    this->A->delProps(COO);
    if (!insert)
    {
        this->A->computeDiagonal();
    }

    this->set_initialized(this->A->row_offsets);
    this->A->setView(OWNED);
}



template <AMGX_VecPrecision t_vecPrec, AMGX_MatPrecision t_matPrec, AMGX_IndPrecision t_indPrec>
void DistributedManager<TemplateConfig<AMGX_device, t_vecPrec, t_matPrec, t_indPrec> >::reorder_matrix()
{
    int num_neighbors = this->neighbors.size();

    if (num_neighbors == 0) { return; }

    int size = this->A->get_num_rows();
    int num_blocks = min(4096, (size + 511) / 512);
    int rings = this->B2L_rings[0].size() - 1;
    this->set_num_halo_rings(rings);
    int diag = this->A->hasProps(DIAG);
    std::vector<Matrix<TConfig_d> > &halo_rows = *this->halo_rows;
    std::vector<DistributedManager<TConfig_d> > &halo_btl = *this->halo_btl;
    /*
     EXAMPLE
     The example matrix, on partition 1 arrives at this point with the following state:
     num_rings=2
     A.num_rows = 4; A.num_nz = 20
     A.row_offsets = [0 4 11 15 20]
     A.col_indices = [4 0 1 2
                    4 5 0 1 2 3 7
                    0 1 2 3
                    1 2 3 6 7]
     num_neighbors=2; neighbors = [0 2]
     B2L_rings[[0 2 4][0 2 4]] B2L_maps[[0 1| 2 3][1 3| 0 2]]
     L2H_maps (and halo_lists) [[4 5][6 7]]

     With the exchange halo rows:
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
//
// Step 1 & 2 - create renumbering
//
    this->createRenumbering(this->renumbering);
    cudaCheckError();
    /*
     EXAMPLE
     this->renumbering = [1 2 0 3 4 5 6 7]
     B2L_maps = [[1 2| 0 3][2 3| 1 0]]
     L2H_maps = [[4 5][6 7]]
     */
//
// Step 3 - given a full renumbering of owned nodes, calculate inverse renumbering
//
    //now we have the full renumbering table in renum, calculate the inverse
    this->inverse_renumbering.resize(this->renumbering.size());
    calc_inverse_renumbering <<< min(4096, ((int)this->renumbering.size() + 511) / 512), 512 >>> (this->renumbering.raw(), this->inverse_renumbering.raw(), this->renumbering.size());
    cudaCheckError();
    /*
     EXAMPLE
     this->inverse_renumbering = [2 0 1 3 4 5 6 7]
     */
//
// Step 4 - calculate number/offset of nodes in the halos from the neighbors, ring by ring
//
    this->halo_offsets.resize(rings * num_neighbors + 1, 0);

    for (int ring = 0; ring < rings; ring++)
    {
        for (int i = 0; i < num_neighbors; i++)
        {
            this->halo_offsets[ring * num_neighbors + i] = halo_btl[i].B2L_rings[0][ring + 1] - halo_btl[i].B2L_rings[0][ring];
        }
    }

    thrust::exclusive_scan(this->halo_offsets.begin(), this->halo_offsets.end(), this->halo_offsets.begin(), size);
    cudaCheckError();
    this->set_num_halo_rows(this->halo_offsets[this->halo_offsets.size() - 1] - size);
    int total_rows = size + this->num_halo_rows();

    if (total_rows < this->renumbering.size()) { FatalError("total rows < renumbering.size(), send/recv maps should cover all matrix halo columns", AMGX_ERR_NOT_IMPLEMENTED); }

    if (total_rows > this->renumbering.size())
    {
        this->A->getResources()->warning("# owned nodes + # halo nodes > matrix columns: send/recv maps have some unreferences halo indices, they are not directly connected to our partition and therefore we won't compute them, please use 2-ring comms maps if you want to specify 2nd ring neighbors");
    }

    cudaCheckError();
    /*
     EXAMPLE
     halo_offsets [2 2 2 2]
     after exclusive scan: 4 + [0 2 4 6 8] = [4 6 8 10 12]
     num_halo_rows = 8, total_rows = 12
     */
//
// Step 5 - create big mapping table of all halo indices we received (this may use a little too much memory)
//
    //count number of fine rows of neighbors
    thrust::host_vector<INDEX_TYPE> neighbor_rows(num_neighbors + 1);
    int max_num_rows = 0;

    for (int i = 0; i < num_neighbors; i++)
    {
        neighbor_rows[i] = halo_btl[i].index_range();
        max_num_rows = max_num_rows > halo_rows[i].get_num_rows() ? max_num_rows : halo_rows[i].get_num_rows();
    }

    thrust::exclusive_scan(neighbor_rows.begin(), neighbor_rows.end(), neighbor_rows.begin());
    cudaCheckError();
    int total_rows_of_neighbors = neighbor_rows[num_neighbors];
    /*
     EXAMPLE
     neigbor_rows = [0 6 14]
     total_rows_of_neighbors = 14
     */
    IVector halo_mapping(total_rows_of_neighbors);
    thrust::fill(halo_mapping.begin(), halo_mapping.end(), -1);
    cudaCheckError();

    //ring by ring, neighbor by neighbor assign sequentially increasing numbers for halo nodes
    for (int ring = 0; ring < rings; ring++)
    {
        for (int i = 0; i < num_neighbors; i++)
        {
            int size = halo_btl[i].B2L_rings[0][ring + 1] - halo_btl[i].B2L_rings[0][ring];
            int num_blocks = min(4096, (size + 127) / 128);
            //This renumbering has to result in the same renumbering that comes out of L2H renumbering
            create_halo_mapping <<< num_blocks, 128>>>(halo_mapping.raw() + neighbor_rows[i],
                    halo_btl[i].B2L_maps[0].raw() + halo_btl[i].B2L_rings[0][ring],
                    halo_btl[i].base_index(), this->halo_offsets[ring * num_neighbors + i], size);
            cudaCheckError();
            /*
             EXAMPLE
             ring 0 neighbor 0 - halo_btl[0].B2L_maps[0] = [2 3| 0 1] halo_btl[0].L2H_maps = [4 5]
            halo_mapping = [-1 -1 4 5 -1 -1 |-1 -1 -1 -1 -1 -1 -1 -1]
            ring 0 neighbor 1 - halo_btl[1].B2L_maps[0] = [1 2| 0 3] halo_btl[1].L2H_maps = [6 7]
            halo_mapping = [-1 -1 4 5 -1 -1 |-1 6 7 -1 -1 -1 -1 -1]
            ring 1 neighbor 0 - halo_btl[0].B2L_maps[0] = [2 3| 0 1] halo_btl[0].L2H_maps = [4 5]
            halo_mapping = [8 9 4 5 -1 -1 |-1 6 7 -1 -1 -1 -1 -1]
            ring 1 neighbor 1 - halo_btl[1].B2L_maps[0] = [1 2| 0 3] halo_btl[1].L2H_maps = [6 7]
            halo_mapping = [8 9 4 5 -1 -1 |10 6 7 11 -1 -1 -1 -1]

             */
        }
    }

    cudaCheckError();

    for (int i = 0; i < num_neighbors; i++)
    {
        int size = halo_btl[i].L2H_maps[0].size();
        int num_blocks = min(4096, (size + 127) / 128);
        //Map the column indices of the halo rows that point back to boundary nodes
        apply_h2l2b_mapping <<< num_blocks, 128>>>(halo_mapping.raw() + neighbor_rows[i],
                halo_btl[i].L2H_maps[0].raw(),
                halo_btl[i].base_index(), this->B2L_maps[i].raw(), size);
        cudaCheckError();
        /*
         EXAMPLE
         neighbor 0 - mapping back to our own (boundary) indices
         halo_mapping = [8 9 4 5 1 2 |10 6 7 11 -1 -1 -1 -1]
         neighbor 1 - mapping back to our own (boundary) indices
        halo_mapping = [8 9 4 5 1 2 |10 6 7 11 -1 -1 2 3]
         */
    }

    cudaCheckError();
    /*
     EXAMPLE
     neighbor_rows = [0 6 14]
     halo_mapping = [8 9 4 5 1 2 |10 6 7 11 -1 -1 2 3]
     The first part (0-6) of halo_mapping gives a local index for all the indices that we want to know about in halo_btl[0]
     The second part (7-14) gives local indices for halo_btl[1], that is both halo ring there, and the column indices representing vertices in this partition's boundary.
     Note that it does not give indices (-1) for vertices 5 and 6 in neighbor 1 (ID 2), which are column indices connecting it to neighbor 0, hence the two halo regions are not connected
     */
//
// Step 6 - renumber halo matrices and calculate row length (to eventually append to the big matrix)
//
    int insert = 0;

    if (this->A->hasProps(DIAG) && insertDiagonals) { insert = 1; }

    diag = diag && !insertDiagonals;
    //recalculate row_offsets
    IVector new_row_offsets(size + this->num_halo_rows() + 1);
    calc_rowlen_reorder <<< num_blocks, 512>>>(this->A->row_offsets.raw(), new_row_offsets.raw(), this->renumbering.raw(), size, insert);
    cudaCheckError();
    IVector neighbor_rows_d(num_neighbors + 1);
    thrust::copy(neighbor_rows.begin(), neighbor_rows.end(), neighbor_rows_d.begin());
    cudaCheckError();
    /*
     EXAMPLE
     get row length according to renumbering
     new_row_offsets = [4 4 7 5 0 0 0 0 0 0 0 0 0]
     */
    //map column indices of my own matrix
    /*map_col_indices<4><<<num_blocks, 512>>>(this->A->row_offsets.raw(),
                                         this->A->col_indices.raw(),
                                         this->renumbering.raw(),
                                         this->halo_ranges.raw(),
                                         halo_mapping.raw(),
                                         neighbor_rows_d.raw(),
                                         this->base_index(), num_neighbors, size);*/
    thrust::copy(thrust::make_permutation_iterator(this->renumbering.begin(), this->A->col_indices.begin()),
                 thrust::make_permutation_iterator(this->renumbering.begin(), this->A->col_indices.end()),
                 this->A->col_indices.begin());
    cudaCheckError();
    /*
     EXAMPLE
     use this->renumbering = [1 2 0 3 4 5 6 7]
     to map old column indices to new column indices (i.e. according to interior - boundary - halo separation), but do not reshuffle them into their place yet
     A.col_indices = [4 0 1 2
                  4 5 0 1 2 3 7
                  0 1 2 3
                  1 2 3 6 7]
    becomes
    A.col_indices = [4 1 2 0
                  4 5 1 2 0 3 7
                  1 2 0 3
                  2 0 3 6 7]

     */
    cudaCheckError();
    IVector temp_row_len(max_num_rows);

    for (int i = 0; i < num_neighbors; i++)
    {
        //map column indices of halo matrices and count of nonzeros we will keep
        int size = halo_rows[i].get_num_rows();
        int num_blocks = min(4096, (size + 127) / 128);
        map_col_indices_and_count_rowlen<4> <<< num_blocks, 128, 128 * sizeof(INDEX_TYPE)>>>(
            halo_rows[i].row_offsets.raw(),
            halo_rows[i].col_indices.raw(),
            temp_row_len.raw(),
            halo_mapping.raw() + neighbor_rows[i],
            size, insert);
        cudaCheckError();

        //number of nonzeros per row copied into big row sizes array
        for (int ring = 0; ring < rings; ring++)
        {
            thrust::copy(temp_row_len.begin() + halo_btl[i].B2L_rings[0][ring], temp_row_len.begin() + halo_btl[i].B2L_rings[0][ring + 1], new_row_offsets.begin() + this->halo_offsets[ring * num_neighbors + i]);
        }

        cudaCheckError();
        /*
         EXAMPLE
         halo_mapping = [8 9 4 5 1 2 |10 6 7 11 -1 -1 2 3]
         look at halo row matrices, and halo_mapping, count column indices that do not map to -1 and map them to their new, local index
         halo_rows[0].col_indices = [1 2 3 4 5
                    0 1 2 3 4 5 6 7
                    0 1 3 6
                    0 1 2 3]
        becomes
        halo_rows[0].col_indices = [9 4 5 1 2
                    8 9 4 5 1 2 -1 -1
                    8 9 5 -1
                    8 9 4 5]
        with temp_row_len = [5 6 3 4] copied into new_row_offsets: [4 4 7 5| 5 6| 0 0| 3 4| 0 0 0]

        halo_rows[1].col_indices = [7 1 2 3
                    5 6 7 0 1 2 3
                    4 5 0 2 3
                    0 1 2 3]
        becomes
        halo_rows[1].col_indices = [3 6 7 11
                    -1 2 3 10 6 7 11
                    -1 -1 10 7 11
                    10 6 7 11]
        with temp_row_len = [4 6 3 4] copied into new_row_offsets: [4 4 7 5| 5 6| 4 6| 3 4| 3 4 0]
         */
    }

    cudaCheckError();
    //row_offsets array created by exclusive scan of row sizes
    thrust_wrapper::exclusive_scan(new_row_offsets.begin(), new_row_offsets.begin() + size + this->num_halo_rows() + 1, new_row_offsets.begin());
    cudaCheckError();
    /*
     EXAMPLE
     Exclusive scan to get new_row_offsets array:
     new_row_offsets = [0 4 8 15 20| 25 31| 35 41| 44 48| 51 55]
     */
//
// Step 7 - consolidate column indices and values
//
    int new_nnz = new_row_offsets[new_row_offsets.size() - 1];
    typedef typename MatPrecisionMap<t_matPrec>::Type ValueTypeA;
    VVector new_values((new_nnz + 1 + diag * (total_rows - 1))* this->A->get_block_size(), types::util<ValueTypeA>::get_zero());
    IVector new_col_indices(new_nnz, 0);
    //reorder based on row permutation
    reorder_whole_matrix <32> <<< num_blocks, 512>>>(this->A->row_offsets.raw(), this->A->col_indices.raw(), this->A->values.raw(), new_row_offsets.raw(), new_col_indices.raw(), new_values.raw(), this->renumbering.raw(), this->A->get_block_size(), size, insert);
    cudaCheckError();

    if (diag)
    {
        //reorder based on row permutation
        reorder_vector_values <<< num_blocks, 512>>>(new_values.raw() + new_row_offsets[total_rows]*this->A->get_block_size(),
                this->A->values.raw() + this->A->row_offsets[size]*this->A->get_block_size(),
                this->renumbering.raw(),
                this->A->get_block_size(), size);
        cudaCheckError();
    }

    int cumulative_num_rows = size;

    for (int i = 0; i < num_neighbors; i++)
    {
        for (int ring = 0; ring < rings; ring++)
        {
            int num_rows = halo_btl[i].B2L_rings[0][ring + 1] - halo_btl[i].B2L_rings[0][ring];
            int num_blocks = min(4096, (num_rows + 127) / 128);
            //copy in nonzeros that we are keeping
            //TODO: access pattern - should be implemented with warp-wide scans to decide which nonzeros we are keeping and where the rest is going
            reorder_whole_halo_matrix <<< num_blocks, 128>>>(halo_rows[i].row_offsets.raw() + halo_btl[i].B2L_rings[0][ring],
                    halo_rows[i].col_indices.raw(), halo_rows[i].values.raw(),
                    new_row_offsets.raw() + this->halo_offsets[ring * num_neighbors + i],
                    new_col_indices.raw(), new_values.raw(), NULL, this->A->get_block_size(), num_rows,
                    insert, this->halo_offsets[ring * num_neighbors + i], halo_btl[i].B2L_rings[0][ring], halo_btl[i].B2L_rings[0][rings]);

            if (diag)
            {
                thrust::copy(halo_rows[i].values.begin() + (halo_rows[i].row_offsets[halo_rows[i].get_num_rows()] + halo_btl[i].B2L_rings[0][ring])*this->A->get_block_size(),
                             halo_rows[i].values.begin() + (halo_rows[i].row_offsets[halo_rows[i].get_num_rows()] + halo_btl[i].B2L_rings[0][ring + 1])*this->A->get_block_size(),
                             new_values.begin() + (new_row_offsets[total_rows] + cumulative_num_rows)*this->A->get_block_size());
                cumulative_num_rows += num_rows;
            }
        }
    }

    cudaCheckError();
    /*
     EXAMPLE
     copy everything in place, dropping -1 column indices in the halo and reordering the owned rows
     new_row_offsets = [0 4 8 15 20| 25 31| 35 41| 44 48| 51 55]
     new_col_indices = [1 2 0 3
                  4 1 2 0
                  4 5 1 2 0 3 7
                  2 0 3 6 7 -end of owned
                  9 4 5 1 2
                  8 9 4 5 1 2 - end of neighbor 0 ring 0
                  3 6 7 11
                  2 3 10 6 7 11 - end of neighbor 1 ring 0
                  8 9 5
                  8 9 4 5 - end of neighbor 0 ring 1
                  10 7 11
                  10 6 7 11] - end of neighbor 1 ring 1

     */
    this->A->set_num_cols(total_rows);
    this->A->set_num_rows(size);
    this->A->col_indices.swap(new_col_indices);
    new_row_offsets.resize(total_rows + 1);
    this->A->row_offsets.swap(new_row_offsets);
    new_row_offsets.swap(this->old_row_offsets);
    this->A->values.swap(new_values);
    this->A->m_seq_offsets.resize(total_rows + 1);
    thrust::sequence(this->A->m_seq_offsets.begin(), this->A->m_seq_offsets.end());

    if (insert)
    {
        this->A->delProps(DIAG);
        this->A->diag.resize(total_rows);
        thrust::copy(this->A->row_offsets.begin(), this->A->row_offsets.end() - 1, this->A->diag.begin());
    }

    cudaCheckError();
    delete this->halo_rows;
    delete this->halo_btl;
    //set halo_rows and halo_btl to NULL to avoid a potential double free situation in the future
    this->halo_rows = NULL;
    this->halo_btl = NULL;
    this->A->delProps(COO);
    this->A->set_initialized(1);

    //TODO: only do this if AMG_Config matrix_halo_exchange!=2
    if (!insert) { this->A->computeDiagonal(); }

    this->A->setView(OWNED);
}

//function object (functor) for thrust calls (it is a unary operator to add a constant)
template<typename T>
class add_constant_op
{
        const T c;
    public:
        add_constant_op(T _c) : c(_c) {}
        __host__ __device__ T operator()(const T &x) const
        {
            return x + c;
        }
};

template <AMGX_VecPrecision t_vecPrec, AMGX_MatPrecision t_matPrec, AMGX_IndPrecision t_indPrec>
void DistributedManager<TemplateConfig<AMGX_device, t_vecPrec, t_matPrec, t_indPrec> >::obtain_shift_l2g_reordering(index_type n, I64Vector_d &l2g, IVector_d &p, IVector_d &q)
{
    /* WARNING: Exchange halo of the inverse_reordering, which is implicitly based on the local_to_global_map (l2g).
                Notice that it is implicit in the exchange_halo routine, since you are getting exactly the vector
                halo elements, which are exactly the elements you need. They however must be shifted by the partition
                starting points (starting global row indices, which are containe din array part_offsets).
                This allows us to avoid constructing the global vector for inverse permutation,
                as is done in reference MATLAB code. */
    //Recall that part_offsets provide the starting point (global row index) of every partition, in other words,
    //they contain the prefix sum of number of rows assigned to each partition. Also, notice that part_offsets and
    //part_offsets_h have the same values on device and host, respectively. See below few lines for details:
    index_type tag = 1 * 133 + 3 * 7 + 0; //some random number for the tag
    index_type l = p.size();
    q.resize(l);
    thrust::copy     (p.begin(), p.end(),     q.begin());
    thrust::transform(q.begin(), q.end(),     q.begin(), add_constant_op<index_type>(this->part_offsets[this->global_id()]));
    this->exchange_halo(q, tag);
    thrust::sequence (q.begin(), q.begin() + n);
    thrust::transform(q.begin(), q.begin() + n, q.begin(), add_constant_op<index_type>(this->part_offsets[this->global_id()]));
    cudaCheckError();
}

template <AMGX_VecPrecision t_vecPrec, AMGX_MatPrecision t_matPrec, AMGX_IndPrecision t_indPrec>
void DistributedManager<TemplateConfig<AMGX_device, t_vecPrec, t_matPrec, t_indPrec> >::unpack_partition(index_type *Bp, index_type *Bc, mat_value_type *Bv)
{
    index_type l, n, nnz, offset;
    index_type     *ir;
    index_type     *Ap;
    index_type     *Ac;
    mat_value_type *Av;
    IVector q;
    //some initializations
    this->A->getOffsetAndSizeForView(OWNED, &offset, &n);
    this->A->getNnzForView(OWNED, &nnz);
    l  = this->inverse_renumbering.size();
    ir = this->inverse_renumbering.raw();
    Ap = this->A->row_offsets.raw();
    Ac = this->A->col_indices.raw();
    Av = this->A->values.raw();
    //(i) reorder the matrix back (into mixed interior-boundary nodes)
    //applies to rows and columns (out-of-place)
    reorder_partition<index_type, mat_value_type, true, true>
    (n, nnz, Ap, Ac, Av, Bp, Bc, Bv, l, ir);
    cudaCheckError();
    //obtain reordering q that combines the shift of the diagonal block with the off-diagonal block indices conversion from local to global
    this->obtain_shift_l2g_reordering(n, this->local_to_global_map, this->inverse_renumbering, q);
    cudaCheckError();
    //(ii) reorder the matrix back (shift the diagonal block and convert off-diagonal block column indices from local to global)
    //applies columns only (in-place)
    reorder_partition<index_type, mat_value_type, false, true>
    (n, nnz, Bp, Bc, Bv, Bp, Bc, Bv, q.size(), q.raw());
    cudaCheckError();
}

template <class TConfig>
void DistributedManagerBase<TConfig>::createNeighToDestPartMap(IVector_h &neigh_to_part, IVector_h &neighbors, IVector_h &destination_part, int num_neighbors)
{
    neigh_to_part.resize(num_neighbors);

    for (int i = 0; i < num_neighbors; i++)
    {
        neigh_to_part[i] = destination_part[neighbors[i]];
    }
}

template <class TConfig>
void DistributedManagerBase<TConfig>::read_halo_ids(int size, IVector_d &scratch, IVector_d &halo_aggregates, VecInt_t min_index_coarse_halo)
{
    int block_size = 128;
    const int num_blocks = min( AMGX_GRID_MAX_SIZE, (size - 1) / block_size + 1);
    read_halo_ids_kernel <<< num_blocks, block_size>>>(scratch.raw(), halo_aggregates.raw(), min_index_coarse_halo, size);
    cudaCheckError();
}

template <class TConfig>
void DistributedManagerBase<TConfig>::read_halo_ids(int size, IVector_h &scratch, IVector_h &halo_aggregates, VecInt_t min_index_coarse_halo)
{
    FatalError("read_halo_ids not implemented on host yet", AMGX_ERR_NOT_IMPLEMENTED);
}


template <class TConfig>
void DistributedManagerBase<TConfig>::flag_halo_ids(int size, IVector_d &scratch, IVector_d &halo_aggregates, VecInt_t min_index_coarse_halo,  int max_index, int min_index)
{
    int block_size = 128;
    const int num_blocks = min( AMGX_GRID_MAX_SIZE, (size - 1) / block_size + 1);
    flag_halo_ids_kernel <<< num_blocks, block_size>>>(scratch.raw(), halo_aggregates.raw(), min_index_coarse_halo, size, max_index - min_index + 1);
    cudaCheckError();
}

template <class TConfig>
void DistributedManagerBase<TConfig>::flag_halo_ids(int size, IVector_h &scratch, IVector_h &halo_aggregates, VecInt_t min_index_coarse_halo,  int max_index, int min_index)
{
    FatalError("flag_halo_ids not implemented on host yet", AMGX_ERR_NOT_IMPLEMENTED);
}

template <AMGX_VecPrecision t_vecPrec, AMGX_MatPrecision t_matPrec, AMGX_IndPrecision t_indPrec>
void DistributedManager<TemplateConfig<AMGX_device, t_vecPrec, t_matPrec, t_indPrec> >::replaceMatrixCoefficients(int n, int nnz, const mat_value_type *data_pinned, const mat_value_type *diag_pinned)
{
    //matrix parameters
    //int num_nnz  = this->A->get_num_nz();
    int num_rows   = this->halo_offsets[0];
    int total_rows = num_rows + this->num_halo_rows();
    int block_size = this->A->get_block_size();
    mat_value_type *data_hd = NULL;
    mat_value_type *diag_hd = NULL;
    int data_alloc = 0;
    int diag_alloc = 0;
    //cuda parameters
    int num_blocks = min(4096, (num_rows + 127) / 128);

    /* WARNING: the number of non-zero elements (nnz) in the array data_pinned and A->values (num_nnz) might be different at this point.
       1. If the matrix has CSR property and therefore diagonal is included in the matrix this values will be the same.
       2. If the matrix has DIAG property and therefore diagonal is originally stored separately, and later appended to the array
          of values, and subsequently inserted into the matrix than num_nnz = nnz + n. We have to account for this fact when replacing the
          coefficients (and use nnz not num_nnz).
       obs.: see calls to computeDiagonal (matrix.cu), AMGX_matrix_upload and AMGX_replace_coefficients (amgx_c.cu), and
          uploadMatrix and replaceMatrixCoefficients[No|With]Cons (distributed_manager.cu) for details. */

    /* check early exit */
    if (this->neighbors.size() == 0 || this->renumbering.size() == 0)
    {
        return;
    }

    cudaCheckError();
    /* allocate if data and diag if they are not pinned */
    data_hd = (mat_value_type *) this->getDevicePointerForData((void *)data_pinned, nnz * block_size * sizeof(mat_value_type), &data_alloc);

    if (diag_pinned != NULL)
    {
        diag_hd = (mat_value_type *) this->getDevicePointerForData((void *)diag_pinned, num_rows * block_size * sizeof(mat_value_type), &diag_alloc);
    }

    /* replace the values (reordering them if needed) */
    if (insertDiagonals && diag_pinned != NULL)
    {
        replace_values_matrix <32> <<< num_blocks, 512>>>(data_hd, diag_hd, this->old_row_offsets.raw(), this->A->row_offsets.raw(), this->A->values.raw(), this->renumbering.raw(), block_size, num_rows);
    }
    else
    {
        replace_values_matrix <32> <<< num_blocks, 512>>>(data_hd, this->old_row_offsets.raw(), this->A->row_offsets.raw(), this->A->values.raw(), this->renumbering.raw(), block_size, num_rows);

        if (diag_pinned != NULL)
        {
            reorder_vector_values <<< num_blocks, 512>>>(this->A->values.raw() + this->A->row_offsets[total_rows]*block_size, diag_hd, this->renumbering.raw(), block_size, num_rows);
        }
    }

    cudaCheckError();

    /* free memory (if needed) */
    if (data_alloc) { cudaFree(data_hd); }
    if (diag_alloc) { cudaFree(diag_hd); }
}

template <AMGX_VecPrecision t_vecPrec, AMGX_MatPrecision t_matPrec, AMGX_IndPrecision t_indPrec>
void DistributedManager<TemplateConfig<AMGX_device, t_vecPrec, t_matPrec, t_indPrec> >::transformAndUploadVector(VVector_v &v, const void *data, int n, int block_dim)
{
    v.resize(n * block_dim);
    cudaCheckError();
    // Upload on host
    cudaMemcpy(v.raw(), (value_type *)data, n * block_dim * sizeof(value_type), cudaMemcpyDefault);
    cudaCheckError();
    // Permute based on renumbering vector
    transformVector(v);
    int tag = 0;
    // Exchange halos
    this->exchange_halo(v, tag);
}

template <AMGX_VecPrecision t_vecPrec, AMGX_MatPrecision t_matPrec, AMGX_IndPrecision t_indPrec>
void DistributedManager<TemplateConfig<AMGX_device, t_vecPrec, t_matPrec, t_indPrec> >::transformVector(VVector_v &v)
{
    if (this->neighbors.size() == 0) { return; }
    else if (this->renumbering.size() == 0)
    {
        v.resize(this->halo_offsets[this->neighbors.size()]*v.get_block_size());
        return;
    }

    if (v.get_block_size() != this->A->get_block_dimx()) { printf("Blocksize mismatch!\n"); }

    if (v.size() < this->halo_offsets[this->neighbors.size()]*v.get_block_size())
    {
        VVector_v temp(this->halo_offsets[this->neighbors.size()]*v.get_block_size(), types::util<value_type>::get_zero());
        temp.set_block_dimx(v.get_block_dimx());
        temp.set_block_dimy(v.get_block_dimy());

        if (v.size() < this->halo_offsets[0]*this->A->get_block_dimx())
        {
            FatalError("Unknown size of input vector - smaller than the number of rows owned by this partition", AMGX_ERR_NOT_IMPLEMENTED);
        }

        //reorder based on row permutation
        int size = this->halo_offsets[0];
        int num_blocks = min(4096, (size + 511) / 512);
        reorder_vector_values <<< num_blocks, 512>>>(temp.raw(), v.raw(), this->renumbering.raw(), v.get_block_size(), size);
        v.swap(temp);
    }
    else
    {
        VVector_v temp(this->halo_offsets[0]*v.get_block_size());
        int size = this->halo_offsets[0];
        int num_blocks = min(4096, (size + 511) / 512);
        reorder_vector_values <<< num_blocks, 512>>>(temp.raw(), v.raw(), this->renumbering.raw(), v.get_block_size(), size);
        thrust::copy(temp.begin(), temp.end(), v.begin());
    }

    cudaCheckError();
    v.set_transformed();
}

template <AMGX_VecPrecision t_vecPrec, AMGX_MatPrecision t_matPrec, AMGX_IndPrecision t_indPrec>
void DistributedManager<TemplateConfig<AMGX_device, t_vecPrec, t_matPrec, t_indPrec> >::revertVector(VVector_v &v)
{
    if (this->neighbors.size() == 0 || this->renumbering.size() == 0) { return; }

    if (v.get_block_size() != this->A->get_block_dimx()) { printf("Blocksize mismatch!\n"); }

    VVector_v temp(this->halo_offsets[0]*this->A->get_block_dimx());

    if (v.size() < this->halo_offsets[0]*v.get_block_size())
    {
        FatalError("Unknown size of input vector - smaller than the number of rows owned by this partition", AMGX_ERR_NOT_IMPLEMENTED);
    }

    //reorder based on row permutation
    int size = this->halo_offsets[0];
    int num_blocks = min(4096, (size + 511) / 512);
    inverse_reorder_vector_values <<< num_blocks, 512>>>(temp.raw(), v.raw(), this->renumbering.raw(), v.get_block_size(), size);
    cudaCheckError();
    v.resize(this->halo_offsets[0]*this->A->get_block_dimx());
    thrust::copy(temp.begin(), temp.end(), v.begin());
    cudaCheckError();
}

template <AMGX_VecPrecision t_vecPrec, AMGX_MatPrecision t_matPrec, AMGX_IndPrecision t_indPrec>
void DistributedManager<TemplateConfig<AMGX_device, t_vecPrec, t_matPrec, t_indPrec> >::revertAndDownloadVector(VVector_v &v_in, const void *data, int n, int block_dimy)
{
    if ( n == 0 )
    {
        FatalError("Cannot download if size = 0", AMGX_ERR_NOT_IMPLEMENTED);
    }

    if (data == NULL )
    {
        FatalError("Cannot download to a NULL pointer", AMGX_ERR_NOT_IMPLEMENTED);
    }

    if (v_in.size() == 0 )
    {
        FatalError("Cannot download an empty vector", AMGX_ERR_NOT_IMPLEMENTED);
    }

    VVector_v v_out;
    revertVector(v_in, v_out);
    cudaMemcpy((value_type *)data, v_out.raw(), n * block_dimy * sizeof(value_type), cudaMemcpyDefault);
    cudaCheckError();
}

template <AMGX_VecPrecision t_vecPrec, AMGX_MatPrecision t_matPrec, AMGX_IndPrecision t_indPrec>
void DistributedManager<TemplateConfig<AMGX_device, t_vecPrec, t_matPrec, t_indPrec> >::revertVector(VVector_v &v_in, VVector_v &v_out)
{
    if (this->neighbors.size() == 0 || this->renumbering.size() == 0) { return;}

    if (v_in.get_block_size() != this->A->get_block_dimx()) { printf("Blocksize mismatch!\n"); }

    if (v_in.size() < this->halo_offsets[0]*v_in.get_block_size())
    {
        FatalError("Unknown size of input vector - smaller than the number of rows owned by this partition", AMGX_ERR_NOT_IMPLEMENTED);
    }

    int size = this->halo_offsets[0];

    if (v_out.size() != size * this->A->get_block_dimx())
    {
        v_out.resize(size * this->A->get_block_dimx());
    }

    //reorder based on row permutation
    int num_blocks = min(4096, (size + 511) / 512);
    inverse_reorder_vector_values <<< num_blocks, 512>>>(v_out.raw(), v_in.raw(), this->renumbering.raw(), v_in.get_block_size(), size);
    cudaCheckError();
}

template <AMGX_VecPrecision t_vecPrec, AMGX_MatPrecision t_matPrec, AMGX_IndPrecision t_indPrec>
void DistributedManager<TemplateConfig<AMGX_host, t_vecPrec, t_matPrec, t_indPrec> >::transformAndUploadVector(VVector_v &v, const void *data, int n, int block_dim)
{
    if (this->neighbors.size() > 0)
    {
        FatalError("Distributed solve only supported on devices", AMGX_ERR_NOT_IMPLEMENTED);
    }
}


template <AMGX_VecPrecision t_vecPrec, AMGX_MatPrecision t_matPrec, AMGX_IndPrecision t_indPrec>
void DistributedManager<TemplateConfig<AMGX_host, t_vecPrec, t_matPrec, t_indPrec> >::transformVector(VVector_v &v)
{
    if (this->neighbors.size() > 0)
    {
        FatalError("Distributed solve only supported on devices", AMGX_ERR_NOT_IMPLEMENTED);
    }
}

template <AMGX_VecPrecision t_vecPrec, AMGX_MatPrecision t_matPrec, AMGX_IndPrecision t_indPrec>
void DistributedManager<TemplateConfig<AMGX_host, t_vecPrec, t_matPrec, t_indPrec> >::revertAndDownloadVector(VVector_v &v, const void *data, int n, int block_dim)
{
    FatalError("Distributed solve only supported on devices", AMGX_ERR_NOT_IMPLEMENTED);
}

template <AMGX_VecPrecision t_vecPrec, AMGX_MatPrecision t_matPrec, AMGX_IndPrecision t_indPrec>
void DistributedManager<TemplateConfig<AMGX_host, t_vecPrec, t_matPrec, t_indPrec> >::revertVector(VVector_v &v_in, VVector_v &v_out)
{
    if (this->neighbors.size() > 0)
    {
        FatalError("Distributed solve only supported on devices", AMGX_ERR_NOT_IMPLEMENTED);
    }
}

template <AMGX_VecPrecision t_vecPrec, AMGX_MatPrecision t_matPrec, AMGX_IndPrecision t_indPrec>
void DistributedManager<TemplateConfig<AMGX_host, t_vecPrec, t_matPrec, t_indPrec> >::revertVector(VVector_v &v)
{
    if (this->neighbors.size() > 0)
    {
        FatalError("Distributed solve only supported on devices", AMGX_ERR_NOT_IMPLEMENTED);
    }
}

template <AMGX_VecPrecision t_vecPrec, AMGX_MatPrecision t_matPrec, AMGX_IndPrecision t_indPrec>
void DistributedManager<TemplateConfig<AMGX_host, t_vecPrec, t_matPrec, t_indPrec> >::reorder_matrix()
{
    if (this->neighbors.size() > 0)
    {
        FatalError("Distributed solve only supported on devices", AMGX_ERR_NOT_IMPLEMENTED);
    }
}

template <AMGX_VecPrecision t_vecPrec, AMGX_MatPrecision t_matPrec, AMGX_IndPrecision t_indPrec>
void DistributedManager<TemplateConfig<AMGX_host, t_vecPrec, t_matPrec, t_indPrec> >::reorder_matrix_owned()
{
    if (this->neighbors.size() > 0)
    {
        FatalError("Distributed solve only supported on devices", AMGX_ERR_NOT_IMPLEMENTED);
    }
}

template <AMGX_VecPrecision t_vecPrec, AMGX_MatPrecision t_matPrec, AMGX_IndPrecision t_indPrec>
void DistributedManager<TemplateConfig<AMGX_host, t_vecPrec, t_matPrec, t_indPrec> >::obtain_shift_l2g_reordering(index_type n, I64Vector_h &l2g, IVector_h &p, IVector_h &q)
{
    if (this->neighbors.size() > 0)
    {
        FatalError("Distributed solve only supported on devices", AMGX_ERR_NOT_IMPLEMENTED);
    }
}

template <AMGX_VecPrecision t_vecPrec, AMGX_MatPrecision t_matPrec, AMGX_IndPrecision t_indPrec>
void DistributedManager<TemplateConfig<AMGX_host, t_vecPrec, t_matPrec, t_indPrec> >::unpack_partition(index_type *Bp, index_type *Bc, mat_value_type *Bv)
{
    if (this->neighbors.size() > 0)
    {
        FatalError("Distributed solve only supported on devices", AMGX_ERR_NOT_IMPLEMENTED);
    }
}

template <AMGX_VecPrecision t_vecPrec, AMGX_MatPrecision t_matPrec, AMGX_IndPrecision t_indPrec>
void DistributedManager<TemplateConfig<AMGX_host, t_vecPrec, t_matPrec, t_indPrec> >::generatePoisson7pt(int nx, int ny, int nz, int P, int Q, int R)
{
    FatalError("GeneratePoisson7pt only implemented on devices", AMGX_ERR_NOT_IMPLEMENTED);
}

template <AMGX_VecPrecision t_vecPrec, AMGX_MatPrecision t_matPrec, AMGX_IndPrecision t_indPrec>
template <typename t_colIndex>
void DistributedManager<TemplateConfig<AMGX_host, t_vecPrec, t_matPrec, t_indPrec> >::loadDistributedMatrix(
    int num_rows, int num_nonzeros, const int block_dimx, const int block_dimy, const int *row_offsets, 
    const t_colIndex *col_indices, const mat_value_type *values, int num_ranks, int num_rows_global, const void *diag, const MatrixDistribution &dist)
{
    FatalError("loadDistributedMatrix only implemented on devices", AMGX_ERR_NOT_IMPLEMENTED);
}

template <AMGX_VecPrecision t_vecPrec, AMGX_MatPrecision t_matPrec, AMGX_IndPrecision t_indPrec>
void DistributedManager<TemplateConfig<AMGX_host, t_vecPrec, t_matPrec, t_indPrec> >::createOneRingHaloRows()
{
    if (this->neighbors.size() > 0)
    {
        FatalError("Distributed solve only supported on devices", AMGX_ERR_NOT_IMPLEMENTED);
    }
}

template <AMGX_VecPrecision t_vecPrec, AMGX_MatPrecision t_matPrec, AMGX_IndPrecision t_indPrec>
void DistributedManager<TemplateConfig<AMGX_host, t_vecPrec, t_matPrec, t_indPrec> >::replaceMatrixCoefficients(int n, int nnz, const mat_value_type *data, const mat_value_type *diag)
{
    FatalError("Distributed solve only supported on devices", AMGX_ERR_NOT_IMPLEMENTED);
}

template <AMGX_VecPrecision t_vecPrec, AMGX_MatPrecision t_matPrec, AMGX_IndPrecision t_indPrec>
void DistributedManager<TemplateConfig<AMGX_host, t_vecPrec, t_matPrec, t_indPrec> >::createRenumbering(IVector &renumbering)
{
    if (this->neighbors.size() > 0)
    {
        FatalError("Distributed solve only supported on devices", AMGX_ERR_NOT_IMPLEMENTED);
    }
}

template < class TConfig >
void DistributedManagerBase<TConfig>::get_unassigned(IVector_h &flagArray, IVector_h &B2L_map, IVector_h &partition_flags, int size, int fa_size/*, int rank*/)
{
    for (int i = 0; i < size; i++)
    {
        if (B2L_map[i] < fa_size)
        {
            if (flagArray[B2L_map[i]] == 0)
            {
                flagArray[B2L_map[i]] = 1;
                partition_flags[i] = 1;
            }
        }
    }
}

template < class TConfig >
void DistributedManagerBase<TConfig>::set_unassigned(IVector_h &partition_flags, IVector_h &partition_renum, IVector_h &B2L_map, IVector_h &renumbering, int size, int max_element, int renum_size/*, int rank*/)
{
    for (int i = 0; i < size; i++)
    {
        if (B2L_map[i] < renum_size)
        {
            if (partition_flags[i] == 1)
            {
                renumbering[B2L_map[i]] = max_element + partition_renum[i];
            }

            B2L_map[i] = renumbering[B2L_map[i]];
        }
    }
}

/* print manager for target rank to a file or stdout */
template<class TConfig>
void DistributedManagerBase<TConfig>::print(char *f, char *s, int trank)
{
    DistributedManagerBase<TConfig> *m = this;
    int rank = 0;
    int level = 0;
    char filename[1024];
    FILE *fid = NULL;
    int i, j, k, t1, t2;
#ifdef AMGX_WITH_MPI
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
#endif

    //check target rank
    if (rank == trank)
    {
        //check whether to output to stdout or a file
        if (f == NULL)
        {
            fid = stdout;
        }
        else
        {
            level = m->A->amg_level_index;
#ifdef _WIN32
            _snprintf_s(filename, 1024, 1024, "%s_r%d_l%d.mtx", f, rank, level);
#else
            snprintf(filename, 1024, "%s_r%d_l%d.mtx", f, rank, level);
#endif
            fid = fopen(filename, "w");
        }

        cudaDeviceSynchronize();
        cudaCheckError();
        fprintf(fid, "%s\n", s);
        //--- communication info ---
        //compare neighbors
        t1 = m->neighbors.size();
        fprintf(fid, "neighbors %d\n", t1);

        for (i = 0; i < t1; i++)
        {
            k = m->neighbors[i];
            fprintf(fid, "%d\n", k);
        }

        //compare B2L_rings
        t1 = B2L_rings.size();
        fprintf(fid, "B2L_rings %d\n", t1);

        for (i = 0; i < t1; i++)
        {
            t2 = m->B2L_rings[i].size();
            fprintf(fid, "B2L_rings-%d [%d]\n", i, t2);

            for (j = 0; j < t2; j++)
            {
                k = m->B2L_rings[i][j];
                fprintf(fid, "%d\n", k);
            }
        }

        //compare B2L_maps
        t1 = B2L_maps.size();
        fprintf(fid, "B2L_maps %d\n", t1);

        for (i = 0; i < t1; i++)
        {
            t2 = m->B2L_maps[i].size();
            fprintf(fid, "B2L_maps-%d [%d]\n", i, t2);

            for (j = 0; j < t2; j++)
            {
                k = m->B2L_maps[i][j];
                fprintf(fid, "%d\n", k);
            }
        }

        //compare L2H_maps
        t1 = L2H_maps.size();
        fprintf(fid, "L2H_maps %d\n", t1);

        for (i = 0; i < t1; i++)
        {
            t2 = m->L2H_maps[i].size();
            fprintf(fid, "L2H_maps-%d [%d]\n", i, t2);

            for (j = 0; j < t2; j++)
            {
                k = m->L2H_maps[i][j];
                fprintf(fid, "%d\n", k);
            }
        }

        //--- matrix info ---
        fprintf(fid, "num_rows_global=%ld\n", num_rows_global);
        fprintf(fid, "_num_rows_interior=%d\n", m->_num_rows_interior);
        fprintf(fid, "_num_rows_owned=%d\n", m->_num_rows_owned);
        fprintf(fid, "_num_rows_full=%d\n", m->_num_rows_full);
        fprintf(fid, "_num_rows_all=%d\n", m->_num_rows_all);
        fprintf(fid, "_num_nz_interior=%d\n", m->_num_nz_interior);
        fprintf(fid, "_num_nz_owned=%d\n", m->_num_nz_owned);
        fprintf(fid, "_num_nz_full=%d\n", m->_num_nz_full);
        fprintf(fid, "_num_nz_all=%d\n", m->_num_nz_all);
        //compare # halo rows and halo offsets
        fprintf(fid, "# halo rings %d and rows %d\n", m->num_halo_rings(), m->num_halo_rows());
        t1 = m->halo_offsets.size();
        fprintf(fid, "halo_offsets %d\n", t1);

        for (i = 0; i < t1; i++)
        {
            k = m->halo_offsets[i];
            fprintf(fid, "%d\n", k);
        }

        //compare halo ranges
        t1 = m->halo_ranges.size();
        fprintf(fid, "halo_ranges %d\n", t1);

        for (i = 0; i < t1; i++)
        {
            k = m->halo_ranges[i];
            fprintf(fid, "%d\n", k);
        }

        //compare halo ranges (host)
        t1 = m->halo_ranges_h.size();
        fprintf(fid, "halo_ranges_h %d\n", t1);

        for (i = 0; i < t1; i++)
        {
            k = m->halo_ranges_h[i];
            fprintf(fid, "%d\n", k);
        }

        //compare part offsets
        t1 = m->part_offsets.size();
        fprintf(fid, "part_offsets %d\n", t1);

        for (i = 0; i < t1; i++)
        {
            k = m->part_offsets[i];
            fprintf(fid, "%d\n", k);
        }

        //compare part offsets (host)
        t1 = m->part_offsets_h.size();
        fprintf(fid, "part_offsets_h %d\n", t1);

        for (i = 0; i < t1; i++)
        {
            k = m->part_offsets_h[i];
            fprintf(fid, "%d\n", k);
        }

        //compare interior row list
        t1 = m->interior_rows_list.size();
        fprintf(fid, "interior_rows_list %d\n", t1);

        for (i = 0; i < t1; i++)
        {
            k = m->interior_rows_list[i];
            fprintf(fid, "%d\n", k);
        }

        //compare boundary row list
        t1 = m->boundary_rows_list.size();
        fprintf(fid, "boundary_rows_list %d\n", t1);

        for (i = 0; i < t1; i++)
        {
            k = m->boundary_rows_list[i];
            fprintf(fid, "%d\n", k);
        }

        //compare halo1 row list
        t1 = m->halo1_rows_list.size();
        fprintf(fid, "halo1_rows_list %d\n", t1);

        for (i = 0; i < t1; i++)
        {
            k = m->halo1_rows_list[i];
            fprintf(fid, "%d\n", k);
        }

        fprintf(fid, "pointers halo_rows=%p and halo_btl=%p\n", m->halo_rows, m->halo_btl);
        //--- packing info ---
        //compare local to global map
        t1 = m->local_to_global_map.size();
        fprintf(fid, "local_to_global_map %d\n", t1);

        for (i = 0; i < t1; i++)
        {
            k = m->local_to_global_map[i];
            fprintf(fid, "%d\n", k);
        }

        //compare renumbering
        t1 = m->renumbering.size();
        fprintf(fid, "renumbering %d\n", t1);

        for (i = 0; i < t1; i++)
        {
            k = m->renumbering[i];
            fprintf(fid, "%d\n", k);
        }

        //compare inverse renumbering
        t1 = m->inverse_renumbering.size();
        fprintf(fid, "inverse_renumbering %d\n", t1);

        for (i = 0; i < t1; i++)
        {
            k = m->inverse_renumbering[i];
            fprintf(fid, "%d\n", k);
        }

        //--- GPU related and miscellaneous info
        //streams
        fprintf(fid, "streams i=%p, b=%p\n", m->get_int_stream(), m->get_bdy_stream());
        //miscellaneous info
        int64_t bi = m->base_index();       //inlined function
        int np = m->get_num_partitions();   //inlined function
        fprintf(fid, "gid=%d,bi=%ld,np=%d,ir=%d,in=%d,bn=%d\n", m->global_id(), bi, np, m->index_range(), m->num_interior_nodes(), m->num_boundary_nodes());
        cudaDeviceSynchronize();
        cudaGetLastError();

        if (fid != stdout)
        {
            fclose(fid);
        }
    }
}

/* print manager for target rank to a file or stdout (for all ranks) */
template<class TConfig>
void DistributedManagerBase<TConfig>::printToFile(char *f, char *s)
{
    DistributedManagerBase<TConfig> *m = this;
    int rank = 0;
#ifdef AMGX_WITH_MPI
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
#endif
    //notice that print will be called with different (target) rank on different ranks/processes
    m->print(f, s, rank);
}

/* compare two managers */
template<class TConfig>
int DistributedManagerBase<TConfig>::compare(DistributedManagerBase<TConfig> *m2)
{
    DistributedManagerBase<TConfig> *m1 = this;
    int i, j, t1, t2;
    //compare neighbors
    t1 = m1->neighbors.size();
    t2 = m2->neighbors.size();

    if (t1 != t2)
    {
        return 1;
    }

    for (i = 0; i < t1; i++)
    {
        if (m1->neighbors[i] != m2->neighbors[i])
        {
            return 2;
        }
    }

    //compare B2L_rings
    for (i = 0; i < (m1->neighbors.size()); i++)
    {
        t1 = m1->B2L_rings[i].size();
        t2 = m2->B2L_rings[i].size();

        if (t1 != t2)
        {
            return 3;
        }

        for (j = 0; j < t1; j++)
        {
            if (m1->B2L_rings[i][j] != m2->B2L_rings[i][j])
            {
                return 4;
            }
        }
    }

    //compare B2L_maps
    t1 = m1->B2L_maps.size();
    t2 = m2->B2L_maps.size();

    if (t1 != t2)
    {
        return 5;
    }

    for (i = 0; i < t1; i++)
    {
        if (m1->B2L_maps[i] != m2->B2L_maps[i])
        {
            return 6;
        }
    }

    //compare L2H_maps
    t1 = m1->L2H_maps.size();
    t2 = m2->L2H_maps.size();

    if (t1 != t2)
    {
        return 7;
    }

    for (i = 0; i < t1; i++)
    {
        if (m1->L2H_maps[i] != m2->L2H_maps[i])
        {
            return 8;
        }
    }

    return 0;
}

template <AMGX_VecPrecision t_vecPrec, AMGX_MatPrecision t_matPrec, AMGX_IndPrecision t_indPrec>
DistributedManager< TemplateConfig<AMGX_device, t_vecPrec, t_matPrec, t_indPrec> >::~DistributedManager< TemplateConfig<AMGX_device, t_vecPrec, t_matPrec, t_indPrec> >()
{
}

template <AMGX_VecPrecision t_vecPrec, AMGX_MatPrecision t_matPrec, AMGX_IndPrecision t_indPrec>
DistributedManager< TemplateConfig<AMGX_host, t_vecPrec, t_matPrec, t_indPrec> >::~DistributedManager< TemplateConfig<AMGX_host, t_vecPrec, t_matPrec, t_indPrec> >()
{
}

/****************************************
 * Explict instantiations
 ***************************************/
#define AMGX_CASE_LINE(CASE) template class DistributedManager<TemplateMode<CASE>::Type >;
AMGX_FORALL_BUILDS(AMGX_CASE_LINE)
AMGX_FORCOMPLEX_BUILDS(AMGX_CASE_LINE)
#undef AMGX_CASE_LINE

#define AMGX_CASE_LINE(CASE) template void DistributedManager<TemplateMode<CASE>::Type>::loadDistributedMatrix( \
    int, int, const int, const int, const int*, const int *col_indices, const mat_value_type*, int, int, const void*, const MatrixDistribution &dist); \
    template void DistributedManager<TemplateMode<CASE>::Type>::loadDistributedMatrix( \
    int, int, const int, const int, const int*, const int64_t *col_indices, const mat_value_type*, int, int, const void*, const MatrixDistribution &dist);
AMGX_FORALL_BUILDS(AMGX_CASE_LINE)
AMGX_FORCOMPLEX_BUILDS(AMGX_CASE_LINE)
#undef AMGX_CASE_LINE

#define AMGX_CASE_LINE(CASE) template class DistributedManagerBase<TemplateMode<CASE>::Type >;
AMGX_FORALL_BUILDS(AMGX_CASE_LINE)
AMGX_FORCOMPLEX_BUILDS(AMGX_CASE_LINE)
#undef AMGX_CASE_LINE

} // namespace amgx

