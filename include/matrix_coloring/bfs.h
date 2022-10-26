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

#include <thrust/device_vector.h>
#include <thrust/sort.h>
#include <thrust/remove.h>
#include <thrust/unique.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust_wrapper.h>
#include <strided_reduction.h>
#include <vector_thrust_allocator.h>

namespace amgx
{
namespace bfs
{
template<int CTA_SIZE, int WARP_SIZE, class T> __device__ __forceinline__ T warp_scan(const T input)
{
    const int warpId = utils::warp_id();
    const int laneId = utils::lane_id();
    T value = input;
#pragma unroll

    for (int i = 1; i < WARP_SIZE; i *= 2)
    {
        T n = utils::shfl_up(value, i, WARP_SIZE);

        if (laneId >= i)
        {
            value += n;
        }
    }

    return value;
}

template<int CTA_SIZE, int WARP_SIZE, class T> __device__ __forceinline__ T block_scan(const T input, T &total)
{
    const int warpId = utils::warp_id();
    const int laneId = utils::lane_id();
    volatile __shared__ T s_warp_results[CTA_SIZE / WARP_SIZE];
    T thread_result = warp_scan<CTA_SIZE, WARP_SIZE>(input);

    if (laneId == WARP_SIZE - 1)
    {
        s_warp_results[warpId] = thread_result;
    }

    __syncthreads();

    if (warpId == 0)
    {
        T val = 0;

        if (laneId < CTA_SIZE / WARP_SIZE)
        {
            val = s_warp_results[laneId];
        }

        s_warp_results[laneId] = warp_scan<CTA_SIZE, WARP_SIZE>(val);
    }

    __syncthreads();

    if (warpId != 0)
    {
        thread_result += s_warp_results[warpId - 1];
    }

    total = s_warp_results[CTA_SIZE / WARP_SIZE - 1];
    return thread_result;
}

template<int CTA_SIZE, int WARP_SIZE, class T> __device__ __forceinline__ T block_scan(const T input)
{
    T total_unused = 0;
    return block_scan<CTA_SIZE, WARP_SIZE, T>(input, total_unused);
}


//util
template<class T> T *to_ptr(device_vector_alloc<T> &v) { return amgx::thrust::raw_pointer_cast( &v.front() ); };

//expand the list of neighbors
template<int CTA_SIZE, int WARP_SIZE>
__global__ void bfs_expand(
    const int num_rows, const int *row_offsets, const int *col_indices,
    int *distances, int *parents,
    const int iteration,
    int *task_queue_in, const int task_queue_n, int *task_queue_out, int *task_queue_out_tail)
{
    //const int NUM_WARPS_PER_CTA  = CTA_SIZE / WARP_SIZE;
    const int NUM_THREADS_PER_GRID = gridDim.x * CTA_SIZE;
    const int warpId = utils::warp_id();
    const int laneId = utils::lane_id();
    int task_id = blockIdx.x * CTA_SIZE + threadIdx.x;

    for ( ; task_id - threadIdx.x < task_queue_n ; task_id += NUM_THREADS_PER_GRID )
    {
        int row_id    = -1;
        int row_begin = -1;
        int row_end   = -1;

        if (task_id < task_queue_n)
        {
            row_id = task_queue_in[task_id]; //coalesced
            distances[row_id] = iteration;
            row_begin = row_offsets[row_id  ]; //scattered
            row_end   = row_offsets[row_id + 1]; //scattered
        }

        int n_neighbors = row_end - row_begin;
        int n_neighbors_scan = block_scan<CTA_SIZE, WARP_SIZE>(n_neighbors);            //dynamic allocation
        int n_neighbors_exclusive_scan = n_neighbors_scan - n_neighbors;
        __shared__ int block_tail;

        if (threadIdx.x == CTA_SIZE - 1)
        {
            block_tail = atomicAdd(task_queue_out_tail, n_neighbors_scan);
        }

        __syncthreads();
        int block_tail_ = block_tail;

        for (int i = 0; i < n_neighbors; ++i)
        {
            int col = __ldg(col_indices + row_begin + i);
            task_queue_out[block_tail_ + n_neighbors_exclusive_scan + i] = col;
        }

        __syncthreads();
    }
}




struct filter_visited_closure
{
    int *distances_ptr;
    filter_visited_closure(int *di_p) : distances_ptr(di_p) {}

    __device__ inline int operator()(const int &index)
    {
        if (__ldg(distances_ptr + index) >= 0) { return 1; }
        return 0;
    }
};

struct is_head
{
    int *ptr;
    is_head(int *ptr_): ptr(ptr_) {}
    __device__ inline int operator()(const int &index)
    {
        return ptr[index] != ptr[index + 1];
    }
};

//bfs
void bfs(const int start, const int num_rows, const int num_nonzero, const int *row_offsets, const int *col_indices, device_vector_alloc<int> &distances)
{
    distances.resize(num_rows);
    device_vector_alloc<int> task_queue_in(num_nonzero);//num_edges
    device_vector_alloc<int> task_queue_out(num_nonzero);//num_edges
    device_vector_alloc<int> task_queue_out_tail(1);//num_edges
    amgx::thrust::fill(distances.begin(), distances.end(), -1);
    cudaCheckError();
    int *distances_ptr = amgx::thrust::raw_pointer_cast( &distances.front() );
    int *task_queue_in_ptr = amgx::thrust::raw_pointer_cast( &task_queue_in.front() );;
    int *task_queue_out_ptr = amgx::thrust::raw_pointer_cast( &task_queue_out.front() );;
    int *task_queue_out_tail_ptr = amgx::thrust::raw_pointer_cast( &task_queue_out_tail.front() );;
    int task_queue_n = 1;
    task_queue_n = 1;
    task_queue_in[0] = start;
    distances[start] = 0;
    int remaining = num_rows;

    for (int iteration = 0; ; ++iteration)
    {
        int check = amgx::thrust::count(distances.begin(), distances.end(), -1);
        cudaCheckError();
        //update remaining

        if (!(remaining > 0 && task_queue_n > 0)) { break; }

        remaining -= task_queue_n;
        //expand neighbors for each vertex in queue
        task_queue_out_tail[0] = 0;
        bfs_expand<512, 32> <<< 26, 512>>>(num_rows, row_offsets, col_indices,
                                           distances_ptr, 0,
                                           iteration,
                                           task_queue_in_ptr, task_queue_n, task_queue_out_ptr, task_queue_out_tail_ptr);
        task_queue_n = task_queue_out_tail[0];
        cudaCheckError();
        //contract duplicates using amgx::thrust::
        thrust_wrapper::sort(task_queue_out.begin(), task_queue_out.begin() + task_queue_n);
        cudaCheckError();
        task_queue_n = amgx::thrust::unique(task_queue_out.begin(), task_queue_out.begin() + task_queue_n) - task_queue_out.begin();
        cudaCheckError();
        //remove visited: amgx::thrust:: remove_if with counting_iterator
        task_queue_n = amgx::thrust::remove_if(task_queue_out.begin(), task_queue_out.begin() + task_queue_n, filter_visited_closure(distances_ptr)) - task_queue_out.begin();
        cudaCheckError();
        amgx::thrust::copy(task_queue_out.begin(), task_queue_out.begin() + task_queue_n, task_queue_in.begin());
    }
}

/////////////////////// block bfs

template<int CTA_SIZE, int WARP_SIZE>
__global__ void block_seeded_bfs_expand(
    const int num_rows, const int *row_offsets, const int *col_indices,
    int *distances, int *parents, int *visited_by,
    const int iteration,

    const int max_tasks_per_block,
    int *task_queue_in,
    int *task_queue_out,
    int *block_task_queue_count_ptr
)
{
    const int warpId = utils::warp_id();
    const int laneId = utils::lane_id();
    int *block_task_queue = task_queue_in + blockIdx.x * max_tasks_per_block;
    int  block_task_queue_count = block_task_queue_count_ptr[blockIdx.x];
    int *block_task_queue_out = task_queue_out + blockIdx.x * max_tasks_per_block;
    int block_task_queue_count_out = 0;
    int task_id = threadIdx.x;

    for ( ; task_id - threadIdx.x < block_task_queue_count ; task_id += blockDim.x )
    {
        int row_id    = -1;
        int row_begin = -1;
        int row_end   = -1;

        if (task_id < block_task_queue_count)
        {
            row_id = block_task_queue[task_id]; //coalesced
#define AMGX_SEEDEDBFS_DBG 0

            if (row_id >= 0)
            {
                {
#if !AMGX_SEEDEDBFS_DBG
                    distances[row_id] = iteration;
#else
                    int old = atomicCAS(distances + row_id, -1, iteration);

                    if (old != -1)
                    {
                        printf("conflict %d %d\n", row_id, old);
                    }

#endif
                    row_begin = row_offsets[row_id  ]; //scattered
                    row_end   = row_offsets[row_id + 1]; //scattered
                }
            }

#if AMGX_SEEDEDBFS_DBG
            else
            {
                printf("conflict %d\n", row_id);
            }

#endif
        }

        //allocate space with all neigbors discovered
        int n_neighbors = row_end - row_begin;
        int total_added_for_this_task = 0;
        //compute where to write in tail
#define PRE 0
#if PRE
        int n_neighbors_inclusive_scan = block_scan<CTA_SIZE, WARP_SIZE>(n_neighbors, total_added_for_this_task);           //dynamic allocation
        int n_neighbors_exclusive_scan = n_neighbors_inclusive_scan - n_neighbors;

        for (int i = row_begin; i < row_end; ++i)
        {
            int col = __load_nc(col_indices + i);
            int old_visited_by = atomicCAS(visited_by + col, -1, blockIdx.x); //marks atomically the node

            if (old_visited_by == -1) //I am the winner!
            {
            }
            else
            {
                if (old_visited_by != blockIdx.x)
                {
                    //collision[col] |= 1<<blockIdx.x;
                }

                col = -1; //mark invalid
            }

            block_task_queue_out[n_neighbors_exclusive_scan + (i - row_begin)] = col;
        }

#else
        n_neighbors = 0;
        const unsigned int block_bits = 5;
        const unsigned int row_tag = (row_id + 1) << block_bits; //+1: want it != 0

        for (int i = row_begin; i < row_end; ++i)
        {
            int col = __load_nc(col_indices + i);
            int old_visited_by = atomicCAS(visited_by + col, -1, blockIdx.x | row_tag); //marks atomically the node

            if (old_visited_by == -1) //I am the winner!
            {
                n_neighbors++;
            }
            else
            {
                if (old_visited_by != blockIdx.x)
                {
                    //collision[col] |= 1<<blockIdx.x;
                }

                col = -1; //mark invalid
            }
        }

        int n_neighbors_inclusive_scan = block_scan<CTA_SIZE, WARP_SIZE>(n_neighbors, total_added_for_this_task);           //dynamic allocation
        int n_neighbors_exclusive_scan = n_neighbors_inclusive_scan - n_neighbors;
        int lcount = 0;

        for (int i = row_begin; i < row_end; ++i)
        {
            int col = __load_nc(col_indices + i);
            int vb = atomicCAS(visited_by + col, blockIdx.x | row_tag, blockIdx.x); //visited_by[col] & block_mask;

            if (vb == (blockIdx.x | row_tag))
            {
                block_task_queue_out[n_neighbors_exclusive_scan + lcount] = col;
                ++lcount;
            }
        }

#endif
        block_task_queue_out += total_added_for_this_task;
        block_task_queue_count_out += n_neighbors_inclusive_scan;
        __syncthreads();
    }

    if (threadIdx.x == CTA_SIZE - 1)
    {
        block_task_queue_count_ptr[blockIdx.x] = block_task_queue_count_out;
    }
}

//bfs
void block_seeded_bfs(const int num_rows, const int num_nonzero, const int *row_offsets, const int *col_indices, device_vector_alloc<int> &distances)
{
    distances.resize(num_rows);
    device_vector_alloc<int> visited_by(num_rows);
    device_vector_alloc<int> task_queue_in(num_nonzero);//num_edges
    device_vector_alloc<int> task_queue_out(num_nonzero);//num_edges
    device_vector_alloc<int> task_queue_out_tail(1);//num_edges
    device_vector_alloc<int> task_queue_visited_by(num_nonzero);//num_edges
    amgx::thrust::fill(distances.begin(), distances.end(), -1);
    amgx::thrust::fill(visited_by.begin(), visited_by.end(), -1);
    cudaCheckError();
    int num_blocks = 1;
    device_vector_alloc<int> task_queue_block_counts(num_blocks);
    amgx::thrust::host_vector<int> task_queue_block_counts_h(num_blocks);
    //device_vector_alloc<int> task_queue_block_tails(num_blocks);
    device_vector_alloc<int> visited_by_heads(num_blocks + 1);
    int task_queue_n = num_blocks;
    int task_chunk_count = num_nonzero / num_blocks;

    for (int i = 0; i < num_blocks; ++i)
    {
        int start = (i + 0.5f) * num_rows / num_blocks;
        task_queue_in[i * task_chunk_count] = start;
        visited_by[start] = i;
        distances[start] = -1;
        task_queue_block_counts[i] = 1;
        //task_queue_block_tails[i] = i+1;
    }

    int remaining = num_rows;

    for (int iteration = 0; ; ++iteration)
    {
        int check = amgx::thrust::count(distances.begin(), distances.end(), -1);
        cudaCheckError();
        //update remaining
#if AMGX_SEEDEDBFS_DBG
        printf("SEED_BFS Iteration=%d Remaining=%d Check=%d Processing=%d\n", iteration, remaining, check, task_queue_n);
#endif

        if (task_queue_n <= 0) { break; }

        remaining -= task_queue_n;
        //expand neighbors for each vertex in queue
        block_seeded_bfs_expand<256, 32> <<< num_blocks, 256>>>(
            num_rows,
            row_offsets,
            col_indices,
            to_ptr(distances),
            0,
            to_ptr(visited_by),
            iteration,
            task_chunk_count,
            to_ptr(task_queue_in),
            to_ptr(task_queue_out),
            to_ptr(task_queue_block_counts)
        );
        cudaCheckError();
        task_queue_block_counts_h = task_queue_block_counts;
        task_queue_n = 0;

        for (int b = 0; b < num_blocks; b++)
        {
            int n_tasks_block = task_queue_block_counts_h[b];
            task_queue_n += n_tasks_block;
        }

        task_queue_in.swap(task_queue_out);
    }
}

}
}
