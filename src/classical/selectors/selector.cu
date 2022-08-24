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

#include <types.h>
#include <classical/selectors/selector.h>
#include <thrust/count.h>
#include <hash_workspace.h>
#include <thrust_wrapper.h>

#include <algorithm>
#include <assert.h>

#include <device_properties.h>

namespace amgx
{
using namespace std;
namespace classical
{
struct is_non_neg
{
    __host__ __device__
    bool operator()(const int &x)
    {
        return x >= 0;
    }
};

__global__
void coarseMarkKernel(int *cf_map, int *mark, const int numEntries)
{
    for (int tIdx = threadIdx.x + blockIdx.x * blockDim.x; tIdx < numEntries; tIdx += gridDim.x * blockDim.x)
    {
        if (cf_map[tIdx] == COARSE)
        {
            mark[tIdx] = 1;
        }
    }
}

__global__
void modifyCoarseMapKernel(int *cf_map, int *mark, const int numEntries)
{
    for (int tIdx = threadIdx.x + blockIdx.x * blockDim.x; tIdx < numEntries; tIdx += gridDim.x * blockDim.x)
    {
        // entry mark[i] is 0 if fine, desired #+1 if coarse
        if (cf_map[tIdx] == COARSE)
        {
            cf_map[tIdx] = mark[tIdx] - 1;
        }
    }
}

/*
 * Count the # of non-zeros per row
 */
__global__
void nonZerosPerCoarseRowKernel(const int num_rows, const int *cf_map, const int *C_hat_start,
                                const int *C_hat_end, int *nonZerosPerCoarseRow)
{
    for (int tIdx = threadIdx.x + blockIdx.x * blockDim.x; tIdx < num_rows; tIdx += gridDim.x * blockDim.x)
    {
        // If coarse point
        if (cf_map[tIdx] >= 0)
        {
            nonZerosPerCoarseRow[cf_map[tIdx]] = C_hat_end[tIdx] - C_hat_start[tIdx];
        }
    }
}


template <typename IndexType>
__global__
void resolve_boundary(const IndexType *offsets, const IndexType *column_indices, bool *s_con,
                      const int *cf_map_in, const int base_offset, const int numRows)
{
    for ( int tIdx = threadIdx.x + blockDim.x * blockIdx.x; tIdx < numRows; tIdx += blockDim.x * gridDim.x )
    {
        const int offset = offsets[tIdx];
        const int numj = offsets[tIdx + 1] - offset;

        if (cf_map_in[base_offset + tIdx] == COARSE)
        {
            // for each column
            for (int j = offset; j < offset + numj; j++)
            {
                if (!s_con[j]) { continue; } // skip weakly connected points

                int jcol = column_indices[j];

                if (cf_map_in[jcol] == COARSE) // coarse neighbour
                {
                    s_con[j] = false; //make it weak
                }
            }
        }
    }
}

namespace selector
{
#include <sm_utils.inl>
#include <hash_containers_detail.inl> // Included inside the namespace to solve name colisions.

__device__ __forceinline__ int get_work( int *queue, int warp_id )
{
    int offset = -1;

    if ( utils::lane_id() == 0 )
    {
        offset = atomicAdd( queue, 1 );
    }

    return utils::shfl( offset, 0 );
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template< int CTA_SIZE, int WARP_SIZE >
__global__ __launch_bounds__( CTA_SIZE )
void
estimate_c_hat_size_kernel( const int A_num_rows,
                            const int *A_rows,
                            const int *A_cols,
                            const int *cf_map,
                            const bool *s_con,
                            int *C_hat_offsets )
{
    const int NUM_WARPS = CTA_SIZE / WARP_SIZE;
    // A shared location where threads propose a row of B to load.
    __shared__ volatile int s_b_row_ids[CTA_SIZE];
    // The coordinates of the thread inside the CTA/warp.
    const int warp_id = utils::warp_id( );
    const int lane_id = utils::lane_id( );
    // First threads load the row IDs of A needed by the CTA...
    int a_row_id = blockIdx.x * NUM_WARPS + warp_id;

    // Loop over rows of A.
    for ( ; a_row_id < A_num_rows ; a_row_id += gridDim.x * NUM_WARPS )
    {
        // Skip fine rows.
        int coarse_fine_id = cf_map[a_row_id];

        if ( coarse_fine_id < 0)
        {
            continue;
        }

        // Load A row IDs.
        int a_col_begin = A_rows[a_row_id  ];
        int a_col_end   = A_rows[a_row_id + 1];
        // The number of elements.
        int count(0);

        // Iterate over the columns of A to build C_hat.
        for ( int a_col_it = a_col_begin + lane_id ; utils::any( a_col_it < a_col_end ) ; a_col_it += WARP_SIZE )
        {
            // Columns of A maps to rows of B. Each thread of the warp loads its A-col/B-row ID.
            int a_col_id = -1;

            if ( a_col_it < a_col_end )
            {
                a_col_id = A_cols[a_col_it];
            }

            // Is it an off-diagonal element.
            bool is_off_diagonal = a_col_it < a_col_end && a_col_id != a_row_id;
            // Is it strongly connected ?
            bool is_strongly_connected = is_off_diagonal && s_con[a_col_it];
            // Is it fine.
            bool is_fine = is_off_diagonal && cf_map[a_col_id] == FINE;
            // Is it isolated.
            bool is_strong_fine = is_off_diagonal && cf_map[a_col_id] == STRONG_FINE;
            // Is it a coarse and strongly-connected column.
            bool is_coarse_strongly_connected = is_strongly_connected && !is_fine && !is_strong_fine;
            // Push coarse and strongly connected nodes in the set.
            count += __popc( utils::ballot( is_coarse_strongly_connected ) );
            // Is it a fine and strongly-connected column.
            bool is_fine_strongly_connected = is_strongly_connected && is_fine;
            // We collect fine and strongly-collected columns.
            int vote = utils::ballot( is_fine_strongly_connected );
            int dest = __popc( vote & utils::lane_mask_lt() );

            if ( is_fine_strongly_connected )
            {
                s_b_row_ids[warp_id * WARP_SIZE + dest] = a_col_id;
            }

            __syncthreads();

            // For each warp, we have up to 32 rows of B to proceed.
            for ( int k = 0, num_rows = __popc(vote) ; k < num_rows ; ++k )
            {
                // Threads in the warp proceeds columns of B in the range [b_col_it, b_col_end).
                int b_row_id = s_b_row_ids[warp_id * WARP_SIZE + k];
                // Load the range of B.
                int b_col_it  = A_rows[b_row_id + 0];
                int b_col_end = A_rows[b_row_id + 1];

                // _iterate over the range of columns of B.
                for ( b_col_it += lane_id ; utils::any(b_col_it < b_col_end) ; b_col_it += WARP_SIZE )
                {
                    // The ID of the column.
                    int b_col_id = -1;

                    if ( b_col_it < b_col_end )
                    {
                        b_col_id = A_cols[b_col_it];
                    }

                    // Is it an off-diagonal element.
                    is_off_diagonal = b_col_it < b_col_end && b_col_id != b_row_id;
                    // Is it a coarse and strongly-connected column.
                    is_coarse_strongly_connected = is_off_diagonal && s_con[b_col_it] && (cf_map[b_col_id] != FINE && cf_map[b_col_id] != STRONG_FINE);
                    // Push coarse and strongly connected nodes in the set.
                    count += __popc( utils::ballot( is_coarse_strongly_connected ) );
                }
            }
        }

        // Store the number of columns in each row.
        if ( lane_id == 0 )
        {
            C_hat_offsets[a_row_id] = count;
        }
    }
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template< int NUM_THREADS_PER_ROW, int CTA_SIZE, int WARP_SIZE >
__global__ __launch_bounds__( CTA_SIZE )
void
estimate_c_hat_size_kernel( const int A_num_rows,
                            const int *A_rows,
                            const int *A_cols,
                            const int *cf_map,
                            const bool *s_con,
                            int *C_hat_offsets )
{
    const int NUM_WARPS = CTA_SIZE / WARP_SIZE;
    const int NUM_LOADED_ROWS = WARP_SIZE / NUM_THREADS_PER_ROW;
    // A shared location where threads propose a row of B to load.
    __shared__ int s_b_row_ids[CTA_SIZE];
    // The coordinates of the thread inside the CTA/warp.
    const int warp_id = utils::warp_id( );
    const int lane_id = utils::lane_id( );
    // Constants.
    const int lane_id_div_num_threads = lane_id / NUM_THREADS_PER_ROW;
    const int lane_id_mod_num_threads = lane_id % NUM_THREADS_PER_ROW;
    // First threads load the row IDs of A needed by the CTA...
    int a_row_id = blockIdx.x * NUM_WARPS + warp_id;

    // Loop over rows of A.
    for ( ; a_row_id < A_num_rows ; a_row_id += gridDim.x * NUM_WARPS )
    {
        // Skip fine rows.
        int coarse_fine_id = cf_map[a_row_id];

        if ( coarse_fine_id < 0)
        {
            continue;
        }

        // Load A row IDs.
        int a_col_begin = A_rows[a_row_id  ];
        int a_col_end   = A_rows[a_row_id + 1];
        // The number of elements.
        int count(0);

        // Iterate over the columns of A to build C_hat.
        for ( int a_col_it = a_col_begin + lane_id ; utils::any( a_col_it < a_col_end ) ; a_col_it += WARP_SIZE )
        {
            // Columns of A maps to rows of B. Each thread of the warp loads its A-col/B-row ID.
            int a_col_id = -1;

            if ( a_col_it < a_col_end )
            {
                a_col_id = A_cols[a_col_it];
            }

            // Is it an off-diagonal element.
            bool is_off_diagonal = a_col_it < a_col_end && a_col_id != a_row_id;
            // Is it strongly connected ?
            bool is_strongly_connected = is_off_diagonal && s_con[a_col_it];
            // Is it fine.
            bool is_fine = is_off_diagonal && cf_map[a_col_id] == FINE;
            // Is it isolated.
            bool is_strong_fine = is_off_diagonal && cf_map[a_col_id] == STRONG_FINE;
            // Is it a coarse and strongly-connected column.
            bool is_coarse_strongly_connected = is_strongly_connected && !is_fine && !is_strong_fine;
            // Push coarse and strongly connected nodes in the set.
            count += __popc( utils::ballot( is_coarse_strongly_connected ) );
            // Is it a fine and strongly-connected column.
            bool is_fine_strongly_connected = is_strongly_connected && is_fine;
            // We collect fine and strongly-collected columns.
            int vote = utils::ballot( is_fine_strongly_connected );
            int dest = __popc( vote & utils::lane_mask_lt() );

            if ( is_fine_strongly_connected )
            {
                s_b_row_ids[warp_id * WARP_SIZE + dest] = a_col_id;
            }

            // For each warp, we have up to 32 rows of B to proceed.
            for ( int k = 0, num_rows = __popc(vote) ; k < num_rows ; k += NUM_LOADED_ROWS )
            {
                int local_k = k + lane_id_div_num_threads;
                // Is it an active thread.
                bool is_active_k = local_k < num_rows;
                // Threads in the warp proceeds columns of B in the range [b_col_it, b_col_end).
                int b_row_id = -1;

                if ( is_active_k )
                {
                    b_row_id = s_b_row_ids[warp_id * WARP_SIZE + local_k];
                }

                // Load the range of B.
                int b_col_it = 0, b_col_end = 0;

                if ( is_active_k )
                {
                    b_col_it  = A_rows[b_row_id + 0];
                    b_col_end = A_rows[b_row_id + 1];
                }

                // _iterate over the range of columns of B.
                for ( b_col_it += lane_id_mod_num_threads ; utils::any(b_col_it < b_col_end) ; b_col_it += NUM_THREADS_PER_ROW )
                {
                    // The ID of the column.
                    int b_col_id = -1;

                    if ( b_col_it < b_col_end )
                    {
                        b_col_id = A_cols[b_col_it];
                    }

                    // Is it an off-diagonal element.
                    is_off_diagonal = b_col_it < b_col_end && b_col_id != b_row_id;
                    // Is it a coarse and strongly-connected column.
                    is_coarse_strongly_connected = is_off_diagonal && s_con[b_col_it] && (cf_map[b_col_id] != FINE && cf_map[b_col_id] != STRONG_FINE);
                    // Push coarse and strongly connected nodes in the set.
                    count += __popc( utils::ballot( is_coarse_strongly_connected ) );
                }
            }
        }

        // Store the number of columns in each row.
        if ( lane_id == 0 )
        {
            C_hat_offsets[a_row_id] = count;
        }
    }
}


template< int NUM_THREADS_PER_ROW, int CTA_SIZE, int WARP_SIZE >
__global__ __launch_bounds__( CTA_SIZE )
void
estimate_c_hat_size_kernel_opt( const int nunfine,
                            const int *A_rows,
                            const int *A_cols,
                            const int *cf_map,
                            const bool *s_con,
                            int *C_hat_offsets,
                            int* unfine_set )
{
    const int NUM_WARPS = CTA_SIZE / WARP_SIZE;
    const int NUM_LOADED_ROWS = WARP_SIZE / NUM_THREADS_PER_ROW;
    // A shared location where threads propose a row of B to load.
    __shared__ int s_b_row_ids[CTA_SIZE];
    // The coordinates of the thread inside the CTA/warp.
    const int warp_id = utils::warp_id( );
    const int lane_id = utils::lane_id( );
    // Constants.
    const int lane_id_div_num_threads = lane_id / NUM_THREADS_PER_ROW;
    const int lane_id_mod_num_threads = lane_id % NUM_THREADS_PER_ROW;
    // First threads load the row IDs of A needed by the CTA...
    int gid = blockIdx.x * NUM_WARPS + warp_id;

    __shared__ int count_s[NUM_WARPS];

    // Loop over rows of A.
    for (gid < nunfine)
    {
        int a_row_id = unfine_set[a];

        // Iterate over the columns of A to build C_hat.
        for ( int a_col_it = A_rows[a_row_id  ] + lane_id ; a_col_it < A_rows[a_row_id + 1]; a_col_it += WARP_SIZE )
        {
            // Columns of A maps to rows of B. Each thread of the warp loads its A-col/B-row ID.
            int a_col_id = A_cols[a_col_it];
            // Is it an off-diagonal element.
            bool is_off_diagonal = a_col_id != a_row_id;
            // Is it strongly connected ?
            bool is_strongly_connected = is_off_diagonal && s_con[a_col_it];
            // Is it fine.
            bool is_fine = is_off_diagonal && cf_map[a_col_id] == FINE;
            // Is it isolated.
            bool is_strong_fine = is_off_diagonal && cf_map[a_col_id] == STRONG_FINE;
            // Is it a coarse and strongly-connected column.
            bool is_coarse_strongly_connected = is_strongly_connected && !is_fine && !is_strong_fine;

            if(is_coarse_strongly_connected) 
            {
                atomicAdd(count_s[warp_id], 1);
            }

            // Is it a fine and strongly-connected column.
            bool is_fine_strongly_connected = is_strongly_connected && is_fine;

            if (!is_fine_strongly_connected ) continue;

            int b_col_beg = A_rows[a_col_id];

            // _iterate over the range of columns of B.
            for (int b_col_it = b_col_beg; b_col_it < A_row[a_col_id+1]; ++b_col_it)
            {
                // The ID of the column.
                int b_col_id = A_cols[b_col_it];

                // Is it an off-diagonal element.
                is_off_diagonal = b_col_it < b_col_end && b_col_id != b_row_id;
                // Is it a coarse and strongly-connected column.
                is_coarse_strongly_connected = is_off_diagonal && s_con[b_col_it] && (cf_map[b_col_id] != FINE && cf_map[b_col_id] != STRONG_FINE);
                // Push coarse and strongly connected nodes in the set.
                if(is_coarse_strongly_connected)
                {
                    atomicAdd(count_s[warp_id], 1);
                }
            }
        }
    }

    __syncthreads();

    if(gid < nunfine)
    {
        // Store the number of columns in each row.
        if ( lane_id == 0 )
        {
            C_hat_offsets[a_row_id] = count_s[warp_id];
        }
    }
}

template< int CTA_SIZE, int SMEM_SIZE, int WARP_SIZE >
__global__ __launch_bounds__( CTA_SIZE )
void
compute_c_hat_kernel( int A_num_rows,
                      const int *__restrict A_rows,
                      const int *__restrict A_cols,
                      const int *__restrict cf_map,
                      const bool *__restrict s_con,
                      const int *__restrict C_hat_start,
                      int *__restrict C_hat_end,
                      int *__restrict C_hat,
                      int gmem_size,
                      int *g_keys,
                      int *wk_work_queue,
                      int *wk_status )
{
    const int NUM_WARPS = CTA_SIZE / WARP_SIZE;
    // Shared memory to vote.
    __shared__ volatile int s_b_row_ids[CTA_SIZE];
    // The hash keys stored in shared memory.
    __shared__ int s_keys[NUM_WARPS * SMEM_SIZE];
    // The coordinates of the thread inside the CTA/warp.
    const int warp_id = utils::warp_id( );
    const int lane_id = utils::lane_id( );
    // First threads load the row IDs of A needed by the CTA...
    int a_row_id = blockIdx.x * NUM_WARPS + warp_id;
    // Create local storage for the set.
    Hash_set<int, SMEM_SIZE, 4, WARP_SIZE> set( &s_keys[warp_id * SMEM_SIZE], &g_keys[a_row_id * gmem_size], gmem_size );

    // Loop over rows of A.
    for ( ; a_row_id < A_num_rows ; a_row_id = get_work( wk_work_queue, warp_id ) )
    {
        // Skip fine rows.
        int coarse_fine_id = cf_map[a_row_id];

        if ( coarse_fine_id < 0)
        {
            continue;
        }

        // Clear the set.
        set.clear();
        // Load the range of the row.
        __syncthreads();
        int a_col_tmp = -1;

        if ( lane_id < 2 )
        {
            a_col_tmp = A_rows[a_row_id + lane_id];
        }

        int a_col_begin = utils::shfl( a_col_tmp, 0 );
        int a_col_end   = utils::shfl( a_col_tmp, 1 );
        __syncthreads();

        // _iterate over the columns of A to build C_hat.
        for ( int a_col_it = a_col_begin + lane_id ; utils::any(a_col_it < a_col_end) ; a_col_it += WARP_SIZE )
        {
            // Is it an active thread.
            const bool is_active = a_col_it < a_col_end;
            // Columns of A maps to rows of B. Each thread of the warp loads its A-col/B-row ID.
            int a_col_id = -1;

            if ( is_active )
            {
                a_col_id = A_cols[a_col_it];
            }

            // Is it an off-diagonal element.
            bool is_off_diagonal = is_active && a_col_id != a_row_id;
            // Is it strongly connected ?
            bool is_strongly_connected = is_off_diagonal && s_con[a_col_it];
            // Is it fine.
            bool is_fine = is_off_diagonal && cf_map[a_col_id] == FINE;
            // Is it isolated.
            bool is_strong_fine = is_off_diagonal && cf_map[a_col_id] == STRONG_FINE;
            // Is it a coarse and strongly-connected column.
            bool is_coarse_strongly_connected = is_strongly_connected && !is_fine && !is_strong_fine;
            // Push coarse and strongly connected nodes in the set.
            int item = -1;

            if ( is_coarse_strongly_connected )
            {
                item = a_col_id;
            }

            set.insert( item, wk_status );
            //if (a_row_id == 7) printf("inserting %d\n",item);
            // Is it a fine and strongly-connected column.
            bool is_fine_strongly_connected = is_strongly_connected && is_fine;
            // We collect fine and strongly-collected columns.
            int vote = utils::ballot( is_fine_strongly_connected );
            int dest = __popc( vote & utils::lane_mask_lt() );

            if ( is_fine_strongly_connected )
            {
                s_b_row_ids[warp_id * WARP_SIZE + dest] = a_col_id;
            }

            int num_rows = __popc( vote );

            // For each warp, we have up to 32 rows of B to proceed.
            for ( int k = 0 ; k < num_rows ; ++k )
            {
                // Threads in the warp proceeds columns of B in the range [bColIt, bColEnd).
                int uniform_b_row_id = s_b_row_ids[warp_id * WARP_SIZE + k];
                // Load the range of the row of B.
                int b_col_it  = A_rows[uniform_b_row_id + 0];
                int b_col_end = A_rows[uniform_b_row_id + 1];

                // _iterate over the range of columns of B.
                for ( b_col_it += lane_id ; utils::any(b_col_it < b_col_end) ; b_col_it += WARP_SIZE )
                {
                    // The ID of the column.
                    int b_col_id = -1;

                    if ( b_col_it < b_col_end )
                    {
                        b_col_id = A_cols[b_col_it];
                    }

                    // Is it an off-diagonal element.
                    is_off_diagonal = b_col_it < b_col_end && b_col_id != uniform_b_row_id;
                    // Is it a strongly connected node.
                    is_strongly_connected = is_off_diagonal && s_con[b_col_it];
                    // Is it a coarse and strongly-connected column.
                    is_coarse_strongly_connected = is_strongly_connected && (cf_map[b_col_id] != FINE && cf_map[b_col_id] != STRONG_FINE);
                    // Push coarse and strongly connected nodes in the set.
                    int b_item = -1;

                    if ( is_coarse_strongly_connected )
                    {
                        b_item = b_col_id;
                    }

                    set.insert( b_item, wk_status );
                    //if (a_row_id == 7) printf("inserting %d\n",b_item);
                }
            }
        }

        int c_col_it = C_hat_start[a_row_id];
        int count = set.store(&C_hat[c_col_it]);

        if ( lane_id == 0 )
        {
            C_hat_end[a_row_id] = c_col_it + count;
        }
    }
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template< int NUM_THREADS_PER_ROW, int CTA_SIZE, int SMEM_SIZE, int WARP_SIZE >
__global__ __launch_bounds__( CTA_SIZE )
void
compute_c_hat_kernel( int A_num_rows,
                      const int *__restrict A_rows,
                      const int *__restrict A_cols,
                      const int *__restrict cf_map,
                      const bool *__restrict s_con,
                      const int *__restrict C_hat_start,
                      int *__restrict C_hat_end,
                      int *__restrict C_hat,
                      int gmem_size,
                      int *g_keys,
                      int *wk_work_queue,
                      int *wk_status )
{
    const int NUM_WARPS = CTA_SIZE / WARP_SIZE;
    const int NUM_LOADED_ROWS = WARP_SIZE / NUM_THREADS_PER_ROW;
    // Shared memory to vote.
    __shared__ volatile int s_b_row_ids[CTA_SIZE];
    // The hash keys stored in shared memory.
    __shared__ int s_keys[NUM_WARPS * SMEM_SIZE];
    // The coordinates of the thread inside the CTA/warp.
    const int warp_id = utils::warp_id( );
    const int lane_id = utils::lane_id( );
    // Constants.
    const int lane_id_div_num_threads = lane_id / NUM_THREADS_PER_ROW;
    const int lane_id_mod_num_threads = lane_id % NUM_THREADS_PER_ROW;
    // First threads load the row IDs of A needed by the CTA...
    int a_row_id = blockIdx.x * NUM_WARPS + warp_id;
    // Create local storage for the set.
    Hash_set<int, SMEM_SIZE, 4, WARP_SIZE> set( &s_keys[warp_id * SMEM_SIZE], &g_keys[a_row_id * gmem_size], gmem_size );
    // Loop over rows of A.
    for ( ; a_row_id < A_num_rows ; a_row_id = get_work( wk_work_queue, warp_id ) )
    {
        // Skip fine rows.
        int coarse_fine_id = cf_map[a_row_id];

        if ( coarse_fine_id < 0)
        {
            continue;
        }

        // Clear the set.
        set.clear();
        // Load the range of the row.
        int a_col_tmp = -1;

        if ( lane_id < 2 )
        {
            a_col_tmp = A_rows[a_row_id + lane_id];
        }

        int a_col_begin = utils::shfl( a_col_tmp, 0 );
        int a_col_end   = utils::shfl( a_col_tmp, 1 );

        // _iterate over the columns of A to build C_hat.
        for ( int a_col_it = a_col_begin + lane_id ; utils::any(a_col_it < a_col_end) ; a_col_it += WARP_SIZE )
        {
            // Is it an active thread.
            const bool is_active = a_col_it < a_col_end;
            // Columns of A maps to rows of B. Each thread of the warp loads its A-col/B-row ID.
            int a_col_id = -1;

            if ( is_active )
            {
                a_col_id = A_cols[a_col_it];
            }

            // Is it an off-diagonal element.
            bool is_off_diagonal = is_active && a_col_id != a_row_id;
            // Is it strongly connected ?
            bool is_strongly_connected = is_off_diagonal && s_con[a_col_it];
            // Is it fine.
            bool is_fine = is_off_diagonal && cf_map[a_col_id] == FINE;
            // Is it isolated.
            bool is_strong_fine = is_off_diagonal && cf_map[a_col_id] == STRONG_FINE;
            // Is it a coarse and strongly-connected column.
            bool is_coarse_strongly_connected = is_strongly_connected && !is_fine && !is_strong_fine;
            // Push coarse and strongly connected nodes in the set.
            int item = -1;

            if ( is_coarse_strongly_connected )
            {
                item = a_col_id;
            }

            set.insert( item, wk_status );
            //if (a_row_id == 7) printf("inserting item %d, is_off_diagonal = %d, \n",item,is_off_diagonal);
            // Is it a fine and strongly-connected column.
            bool is_fine_strongly_connected = is_strongly_connected && is_fine;
            // We collect fine and strongly-collected columns.
            int vote = utils::ballot( is_fine_strongly_connected );
            int dest = __popc( vote & utils::lane_mask_lt() );

            if ( is_fine_strongly_connected )
            {
                s_b_row_ids[warp_id * WARP_SIZE + dest] = a_col_id;
            }

            int num_rows = __popc( vote );

            // For each warp, we have up to 32 rows of B to proceed.
            for ( int k = 0 ; k < num_rows ; k += NUM_LOADED_ROWS )
            {
                int local_k = k + lane_id_div_num_threads;
                // Is it an active thread.
                bool is_active_k = local_k < num_rows;
                // Threads in the warp proceeds columns of B in the range [bColIt, bColEnd).
                int uniform_b_row_id = -1;

                if ( is_active_k )
                {
                    uniform_b_row_id = s_b_row_ids[warp_id * WARP_SIZE + local_k];
                }

                // Load the range of the row of B.
                int b_col_it = 0, b_col_end = 0;

                if ( is_active_k )
                {
                    b_col_it  = A_rows[uniform_b_row_id + 0];
                    b_col_end = A_rows[uniform_b_row_id + 1];
                }

                // _iterate over the range of columns of B.
                for ( b_col_it += lane_id_mod_num_threads ; utils::any(b_col_it < b_col_end) ; b_col_it += NUM_THREADS_PER_ROW )
                {
                    // The ID of the column.
                    int b_col_id = -1;

                    if ( b_col_it < b_col_end )
                    {
                        b_col_id = A_cols[b_col_it];
                    }

                    // Is it an off-diagonal element.
                    is_off_diagonal = b_col_it < b_col_end && b_col_id != uniform_b_row_id && b_col_id != a_row_id;
                    // Is it a strongly connected node.
                    is_strongly_connected = is_off_diagonal && s_con[b_col_it];
                    // Is it a coarse and strongly-connected column.
                    is_coarse_strongly_connected = is_strongly_connected && (cf_map[b_col_id] != FINE && cf_map[b_col_id] != STRONG_FINE);
                    // Push coarse and strongly connected nodes in the set.
                    int b_item = -1;

                    if ( is_coarse_strongly_connected )
                    {
                        b_item = b_col_id;
                    }

                    set.insert( b_item, wk_status );
                    //if (a_row_id == 7) printf("inserting b_item %d, is_off_diagonal = %d, \n",b_item,is_off_diagonal);
                }
            }
        }

        int c_col_it = C_hat_start[a_row_id];
        int count = set.store(&C_hat[c_col_it]) ;

        if ( lane_id == 0 )
        {
            C_hat_end[a_row_id] = c_col_it + count;
        }
    }
}

template <int GROUP_SIZE, int CTA_SIZE, int HASH_SIZE>
__global__ __launch_bounds__( CTA_SIZE )
void
compute_c_hat_kernel_opt( int unfine_set_size,
                      const int *__restrict A_rows,
                      const int *__restrict A_cols,
                      const bool *__restrict s_con,
                      const int *__restrict C_hat_start,
                      int *__restrict C_hat,
                      int *__restrict C_hat_end,
                      int *__restrict cf_map,
                      int *__restrict unfine_set)
{
    constexpr int SLOT_VACANT = -1;

    // Group indices
    constexpr int ngroups = CTA_SIZE / GROUP_SIZE;
    const int group_id = threadIdx.x / GROUP_SIZE;
    const int lane_id = threadIdx.x % GROUP_SIZE;

    const int gid = blockIdx.x * ngroups + group_id;

    // Dynamic sized shared memory
    extern __shared__ int s[];

    // Hash container storage
    int* key_s = (int*)s;
    int* key_group_s = &key_s[group_id*HASH_SIZE];

    // Initialise the keys and values.
#pragma unroll
    for(int i = threadIdx.x; i < ngroups*HASH_SIZE; i += CTA_SIZE)
    {
        key_s[i] = SLOT_VACANT; // Inserted keys will be in range [0,N]
    }

    int* count_s = (int*)&key_s[ngroups*HASH_SIZE];
    if(lane_id == 0)
    {
        count_s[group_id] = 0;
    }

    __syncthreads();

    int a_row_id = -1;

    if(gid < unfine_set_size)
    {
        // One row of A per group
        a_row_id = unfine_set[gid];

        // Iterate over the columns of A to build C_hat.
        for ( int a_col_it = A_rows[a_row_id] + lane_id ; a_col_it < A_rows[a_row_id + 1]; a_col_it += GROUP_SIZE )
        {
            int a_col_id = A_cols[a_col_it];

            // Is it an off-diagonal element.
            bool is_off_diagonal = a_col_id != a_row_id;

            // Check if we have a strong connection to the point
            bool is_strongly_connected = is_off_diagonal && s_con[a_col_it];

            // Is it fine.
            bool is_fine = is_off_diagonal && cf_map[a_col_id] == FINE;

            // Is it isolated.
            bool is_strong_fine = is_off_diagonal && cf_map[a_col_id] == STRONG_FINE;

            // Is it a coarse and strongly-connected column.
            bool is_coarse_strongly_connected = is_strongly_connected && !is_fine && !is_strong_fine;

            if(is_coarse_strongly_connected) 
            {
                int hash = a_col_id % HASH_SIZE;

                while(true)
                {
                    // XXX You can just avoid this and directly atomicCAS, which is faster?
                    int curr_key = key_group_s[hash];
                    if(curr_key == a_col_id) break;

                    if(curr_key == SLOT_VACANT)
                    {
                        int old_key = atomicCAS(&key_group_s[hash], SLOT_VACANT, a_col_id);
                        if(old_key == SLOT_VACANT || old_key == a_col_id)
                        {
                            break;
                        }
                    }

                    // We did not secure a slot, so linear probe to next slot
                    hash = (hash + 1) % HASH_SIZE;
                }
            }

            bool is_fine_strongly_connected = is_strongly_connected && is_fine;

            if(!is_fine_strongly_connected) continue;

            // Threads in the warp proceeds columns of B in the range [bColIt, bColEnd).
            int b_col_beg = A_rows[a_col_id];

            // Iterate over the range of columns of B.
            for (int b_col_it = b_col_beg; b_col_it < A_rows[a_col_id+1]; ++b_col_it)
            {
                int b_col_id = A_cols[b_col_it];

                // Is it an off-diagonal element.
                is_off_diagonal = b_col_id != a_col_id && b_col_id != a_row_id;
                // Is it a strongly connected node.
                is_strongly_connected = is_off_diagonal && s_con[b_col_it];
                // Is it a coarse and strongly-connected column.
                is_coarse_strongly_connected = is_strongly_connected && (cf_map[b_col_id] != FINE && cf_map[b_col_id] != STRONG_FINE);

                if(!is_coarse_strongly_connected) continue;

                int hash = b_col_id % HASH_SIZE;

                while(true)
                {
                    // XXX You can just avoid this and directly atomicCAS, which is faster?
                    int curr_key = key_group_s[hash];
                    if(curr_key == b_col_id) break;
                    
                    if(curr_key == SLOT_VACANT)
                    {
                        int old_key = atomicCAS(&key_group_s[hash], SLOT_VACANT, b_col_id);
                        if(old_key == SLOT_VACANT || old_key == b_col_id)
                        {
                            break;
                        }
                    }

                    // We did not secure a slot, so linear probe to next slot
                    hash = (hash + 1) % HASH_SIZE;
                }
            }
        }
    }

    __syncthreads();

    // XXX Could change to poll by ballot style compress
    if(gid < unfine_set_size)
    {
        // Store the results.
#pragma unroll
        for(int i = lane_id; i < HASH_SIZE; i += GROUP_SIZE)
        {
            int key = key_group_s[i];
            if(key != SLOT_VACANT)
            {
                int ind = atomicAdd(&count_s[group_id], 1);
                C_hat[C_hat_start[a_row_id] + ind] = key;
            }
        }
    }

    __syncthreads();

    if(gid < unfine_set_size)
    {
        // Store the results.
        if(lane_id == 0)
        {
            C_hat_end[a_row_id] = C_hat_start[a_row_id] + count_s[group_id];
        }
    }
}
} // namespace selector

template <typename IndexType>
void __global__  createCfMapGlobal(const IndexType *cf_map, int64_t *cf_map_global, const int64_t my_part_offset, const IndexType num_owned_fine_pts)
{
    for (int tidx = threadIdx.x + blockIdx.x * blockDim.x; tidx < num_owned_fine_pts ; tidx += blockDim.x * gridDim.x)
    {
        // Only renumber the interior points
        if (cf_map[tidx] >= 0 && tidx < num_owned_fine_pts)
        {
            // Simply shift
            cf_map_global[tidx] = (int64_t) cf_map[tidx] /* my_local_id */ + my_part_offset;
        }
        else
        {
            cf_map_global[tidx] = -1;
        }
    }
}

#include <sm_utils.inl>

enum { WARP_SIZE = 32, GRID_SIZE = 1024, SMEM_SIZE = 128 };

template< typename Value_type, int CTA_SIZE, int WARP_SIZE >
__global__ __launch_bounds__( CTA_SIZE )
void
fillS2ColIndices( const int A_num_rows,
                  const int *__restrict cf_map,
                  const int *__restrict C_hat,
                  const int *__restrict C_hat_start,
                  const int *__restrict C_hat_end,
                  const int *__restrict S2_rows,
                  int *S2_cols,
                  int64_t *S2_global_cols,
                  int64_t *cf_map_global)
{
    const int NUM_WARPS = CTA_SIZE / 32;
    const int NUM_WARPS_PER_GRID = gridDim.x * NUM_WARPS;
    // The coordinates of the thread inside the CTA/warp.
    const int warp_id = utils::warp_id();
    const int lane_id = utils::lane_id();
    // First threads load the row IDs of A needed by the CTA...
    int a_row_id = blockIdx.x * NUM_WARPS + warp_id;

    // Loop over rows of A.
    for ( ; a_row_id < A_num_rows; a_row_id += NUM_WARPS_PER_GRID )
    {
        int coarse_fine_id = cf_map[a_row_id];

        // Skip fine rows.
        if ( coarse_fine_id < 0 )
        {
            continue;
        }

        // Rebuild C_hat.
        int c_hat_it  = C_hat_start[a_row_id];
        int c_hat_end = C_hat_end  [a_row_id];
        int s2_col_it  = S2_rows[coarse_fine_id];

        // Iterate over the columns of A.
        for ( c_hat_it += lane_id ; utils::any( c_hat_it < c_hat_end ) ; c_hat_it += WARP_SIZE, s2_col_it += WARP_SIZE)
        {
            // Columns of A maps to rows of B. Each thread of the warp loads its A-col/B-row ID.
            int s2_col_id(-1);

            if ( c_hat_it < c_hat_end )
            {
                s2_col_id = C_hat[c_hat_it];
            }

            if (s2_col_id != -1)
            {
                S2_cols[s2_col_it + lane_id] = cf_map[s2_col_id];

                if (S2_global_cols != NULL)
                {
                    S2_global_cols[s2_col_it + lane_id] = cf_map_global[s2_col_id];
                }
            }
        }
    }
}





template< AMGX_VecPrecision V, AMGX_MatPrecision M, AMGX_IndPrecision I >
void Selector<TemplateConfig<AMGX_device, V, M, I> >::renumberAndCountCoarsePoints( IVector &cf_map,
        int &num_coarse_points,
        int num_rows)
{
    IVector mark(cf_map.size(), 0);
    int blockSize = 128;
    int num_blocks = min( 4096, (int) (cf_map.size() + blockSize - 1) / blockSize );

    if (cf_map.size() > 0)
    {
        coarseMarkKernel <<< num_blocks, blockSize>>>(cf_map.raw(), mark.raw(), cf_map.size());
        cudaCheckError();
    }

    // get the sequence of values
    thrust_wrapper::inclusive_scan(mark.begin(), mark.end(), mark.begin());
    cudaCheckError();

    // assign to cf_map
    if (cf_map.size() > 0)
    {
        modifyCoarseMapKernel <<< num_blocks, blockSize>>>(cf_map.raw(), mark.raw(), cf_map.size());
        cudaCheckError();
    }


    num_coarse_points = (int) thrust_wrapper::count_if(cf_map.begin(), cf_map.begin() + num_rows, is_non_neg());
    cudaCheckError();
}


template< AMGX_VecPrecision V, AMGX_MatPrecision M, AMGX_IndPrecision I >
void Selector<TemplateConfig<AMGX_host, V, M, I> >::renumberAndCountCoarsePoints( IVector &cf_map,
        int &num_coarse_points,
        int num_rows)
{
    num_coarse_points = 0;

    for (int i = 0; i < num_rows; i++)
    {
        // if coarse
        if (cf_map[i] == COARSE)
        {
            cf_map[i] = num_coarse_points++;
        }
    }
}

__global__
void correctCfMapKernel(int *cf_map, const int *cf_map_scanned, const int *cf_map_s2, int size)
{
    for (int tIdx = threadIdx.x + blockIdx.x * blockDim.x; tIdx < size; tIdx += gridDim.x * blockDim.x)
    {
        if (cf_map[tIdx] == COARSE) // Coarse point
        {
            int coarse_id = cf_map_scanned[tIdx];
            int coarse_id_s2 = cf_map_s2[coarse_id];
            // if cf_map_s2 is STRONG_FINE, mark as COARSE
            cf_map[tIdx] = (coarse_id_s2 == STRONG_FINE) ? COARSE : coarse_id_s2;
        }
    }
}

template <AMGX_VecPrecision t_vecPrec, AMGX_MatPrecision t_matPrec, AMGX_IndPrecision t_indPrec>
void Selector<TemplateConfig<AMGX_host, t_vecPrec, t_matPrec, t_indPrec> >
::correctCfMap(IVector &cf_map, IVector &cf_map_scanned, IVector &cf_map_S2)
{
    FatalError("CorrectCFMap not implemented on host", AMGX_ERR_NOT_IMPLEMENTED);
}


template <AMGX_VecPrecision t_vecPrec, AMGX_MatPrecision t_matPrec, AMGX_IndPrecision t_indPrec>
void Selector<TemplateConfig<AMGX_device, t_vecPrec, t_matPrec, t_indPrec> >::correctCfMap(IVector &cf_map, IVector &cf_map_scanned, IVector &cf_map_S2)
{
    // One thread per row of A
    typedef typename Matrix_d::index_type IndexType;
    typedef typename Matrix_d::value_type ValueType;
    // get raw pointers to data
    const int cta_size = 128;
    const int grid_size = std::min( 4096, ((int) cf_map.size() + cta_size - 1) / cta_size);
    correctCfMapKernel <<< grid_size, cta_size>>>(cf_map.raw(), cf_map_scanned.raw(), cf_map_S2.raw(), cf_map.size());
    cudaCheckError();
}

template <AMGX_VecPrecision t_vecPrec, AMGX_MatPrecision t_matPrec, AMGX_IndPrecision t_indPrec>
void Selector<TemplateConfig<AMGX_host, t_vecPrec, t_matPrec, t_indPrec> >
::createS2(Matrix_h &A,
           Matrix_h &S2,
           const BVector &s_con,
           IVector &cf_map)
{
    FatalError("Aggressive PMIS selector not implemented on host", AMGX_ERR_NOT_IMPLEMENTED);
}

template <AMGX_VecPrecision V, AMGX_MatPrecision M, AMGX_IndPrecision I>
template <int hash_size, int group_size>
void Selector<TemplateConfig<AMGX_device, V, M, I>>::compute_c_hat_opt_dispatch(
        const Matrix_d &A,
        const bool *s_con,
        const int *C_hat_start,
        int *C_hat,
        int *C_hat_end,
        int *cf_map,
        IntVector &unfine_set)
{
    typedef typename IndPrecisionMap<I>::Type Index_type;

    constexpr int cta_size = 128;
    constexpr int ngroups = cta_size / group_size;

    const int grid_size = unfine_set.size() / ngroups + 1;

    cudaDeviceProp deviceProps = getDeviceProperties();
    size_t max_shmem_size = deviceProps.sharedMemPerMultiprocessor;

    constexpr int shmem_size =
        (sizeof(int)+sizeof(Index_type))*ngroups*hash_size + group_size; 

    if(shmem_size > max_shmem_size) 
    { 
        FatalError("In compute_values_opt the requested hash size is larger than max.\n", 
                AMGX_ERR_NOT_IMPLEMENTED); 
    } 

    cudaFuncSetAttribute(amgx::classical::selector::compute_c_hat_kernel_opt 
            <group_size, cta_size, hash_size>, 
            cudaFuncAttributeMaxDynamicSharedMemorySize, shmem_size); 

    amgx::classical::selector::compute_c_hat_kernel_opt
        <group_size, cta_size, hash_size> 
        <<<grid_size, cta_size, shmem_size>>>(
            unfine_set.size(),
            A.row_offsets.raw(),
            A.col_indices.raw(),
            s_con,
            C_hat_start,
            C_hat,
            C_hat_end,
            cf_map,
            unfine_set.raw());
}

// Determines which nodes are not fine, hence excluded
__global__ void set_unfine(int n, int* cf_map, int* unfine, int* unfine_offs)
{
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    if(i >= n) return;

    unfine[i] = (cf_map[i] >= 0) ? 1 : 0;

    if(i == 0) 
    {
        unfine_offs[0] = 0;
    }
}

// Compresses indirection
__global__ void unfine_set_fill(int n, int* unfine, int* unfine_offs, int* unfine_set)
{
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    if(i >= n) return;

    if(unfine[i]) 
    { 
        unfine_set[unfine_offs[i]] = i; 
    } 
}


template <AMGX_VecPrecision t_vecPrec, AMGX_MatPrecision t_matPrec, AMGX_IndPrecision t_indPrec>
void Selector<TemplateConfig<AMGX_device, t_vecPrec, t_matPrec, t_indPrec> >::createS2(Matrix_d &A,
        Matrix_d &S2,
        const BVector &s_con,
        IVector &cf_map)
{
    const int blockSize = 256;
    // One thread per row of A
    typedef typename Matrix_d::index_type IndexType;
    typedef typename Matrix_d::value_type ValueType;
    // get raw pointers to data
    const IndexType *Aoffsets = A.row_offsets.raw();
    const IndexType *Acolumn_indices = A.col_indices.raw();
    const ValueType *Avalues = A.values.raw();
    const IndexType Anum_rows = (int) A.get_num_rows();
    int *cf_map_ptr = cf_map.raw();
    Hash_Workspace<TConfig_d, int> exp_wk(true, GRID_SIZE);
    IntVector C_hat_start( A.get_num_rows() + 1, 0 ), C_hat_end( A.get_num_rows() + 1, 0 );

    bool use_opt_kernels = true;
    int max_c_hat_size = 0;
    int nunfine = 0;
    IntVector unfine_set;

    if(use_opt_kernels)
    {
        IntVector unfine(A.get_num_rows());
        IntVector unfine_offs(A.get_num_rows()+1);
        int nthreads = 128;
        int nblocks = A.get_num_rows() / nthreads + 1;
        set_unfine<<<nblocks, nthreads>>>(A.get_num_rows(), cf_map.raw(), unfine.raw(), unfine_offs.raw());
        thrust::inclusive_scan(thrust::device, unfine.begin(), unfine.end(), unfine_offs.begin()+1);

        nunfine = unfine_offs[unfine_offs.size()-1];
        if(nunfine > 0)
        {
            unfine_set.resize(nunfine);
            unfine_set_fill<<<nblocks, nthreads>>>(A.get_num_rows(), unfine.raw(), unfine_offs.raw(), unfine_set.raw());
        }

        const int CTA_SIZE  = 256;
        const int NUM_WARPS = CTA_SIZE / WARP_SIZE;
        int grid_size = nunfine/NUM_WARPS + 1;

        amgx::classical::selector::estimate_c_hat_size_kernel_opt< 8, CTA_SIZE, WARP_SIZE> <<< grid_size, CTA_SIZE>>>(
                nunfine,
                A.row_offsets.raw(),
                A.col_indices.raw(),
                cf_map.raw(),
                s_con.raw(),
                C_hat_start.raw(),
                unfine_set.raw());
        cudaCheckError();

        max_c_hat_size = thrust::reduce(C_hat_start.begin(), C_hat_start.end(), -1, thrust::maximum<int>());
    }
    else
    {
        const int CTA_SIZE  = 256;
        const int NUM_WARPS = CTA_SIZE / WARP_SIZE;
        int avg_nz_per_row = A.get_num_nz() / A.get_num_rows();
        int grid_size = A.get_num_rows()/NUM_WARPS + 1;

        if ( avg_nz_per_row < 16 )
        {
            amgx::classical::selector::estimate_c_hat_size_kernel< 8, CTA_SIZE, WARP_SIZE> <<< grid_size, CTA_SIZE>>>(
                A.get_num_rows(),
                A.row_offsets.raw(),
                A.col_indices.raw(),
                cf_map.raw(),
                s_con.raw(),
                C_hat_start.raw());
        }
        else
        {
            amgx::classical::selector::estimate_c_hat_size_kernel<CTA_SIZE, WARP_SIZE> <<< grid_size, CTA_SIZE>>>(
                A.get_num_rows(),
                A.row_offsets.raw(),
                A.col_indices.raw(),
                cf_map.raw(),
                s_con.raw(),
                C_hat_start.raw());
        }

        cudaCheckError();
    }

    thrust_wrapper::exclusive_scan(C_hat_start.begin(), C_hat_start.end(), C_hat_start.begin());
    cudaCheckError();

    int nVals = C_hat_start[C_hat_start.size() - 1];
    IntVector C_hat( nVals );

    if(use_opt_kernels)
    {
        int hash_size = pow(2, ceil(log2(max_c_hat_size)));

        if(nunfine > 0)
        {
            switch(hash_size)
            {
                case 1:
                case 2:
                case 4:
                case 8:
                case 16:
                case 32:
                    compute_c_hat_opt_dispatch<32, 8>(A, s_con.raw(), C_hat_start.raw(), C_hat.raw(), C_hat_end.raw(), cf_map.raw(), unfine_set);
                    break;
                case 64:
                    compute_c_hat_opt_dispatch<64, 8>(A, s_con.raw(), C_hat_start.raw(), C_hat.raw(), C_hat_end.raw(), cf_map.raw(), unfine_set);
                    break;
                case 128:
                    compute_c_hat_opt_dispatch<128, 8>(A, s_con.raw(), C_hat_start.raw(), C_hat.raw(), C_hat_end.raw(), cf_map.raw(), unfine_set);
                    break;
                case 256:
                    compute_c_hat_opt_dispatch<256, 16>(A, s_con.raw(), C_hat_start.raw(), C_hat.raw(), C_hat_end.raw(), cf_map.raw(), unfine_set);
                    break;
                case 512:
                    compute_c_hat_opt_dispatch<512, 32>(A, s_con.raw(), C_hat_start.raw(), C_hat.raw(), C_hat_end.raw(), cf_map.raw(), unfine_set);
                    break;
                default:
                    FatalError("Size of C_hat in a single row too large, solve with original solvers.", AMGX_ERR_INTERNAL);
            }
        }
    }
    else
    {
        const int CTA_SIZE  = 256;
        const int NUM_WARPS = CTA_SIZE / WARP_SIZE;
        int avg_nz_per_row = A.get_num_nz() / A.get_num_rows();
        int attempt = 0;

        for ( bool done = false ; !done && attempt < 10 ; ++attempt )
        {
            // Double the amount of GMEM (if needed).
            if ( attempt > 0 )
            {
                exp_wk.expand();
            }

            // Reset the status. 
            // TODO: Optimize with async copies.
            int status = 0;
            cudaMemcpy( exp_wk.get_status(), &status, sizeof(int), cudaMemcpyHostToDevice );
            // Compute the set C_hat.
            int work_offset = GRID_SIZE * NUM_WARPS;
            cudaMemcpy( exp_wk.get_work_queue(), &work_offset, sizeof(int), cudaMemcpyHostToDevice );

            // Run the computation.
            if ( avg_nz_per_row < 16 )
            {
                amgx::classical::selector::compute_c_hat_kernel< 8, CTA_SIZE, SMEM_SIZE, WARP_SIZE> <<< GRID_SIZE, CTA_SIZE>>>(
                    A.get_num_rows(),
                    A.row_offsets.raw(),
                    A.col_indices.raw(),
                    cf_map.raw(),
                    s_con.raw(),
                    C_hat_start.raw(),
                    C_hat_end.raw(),
                    C_hat.raw(),
                    exp_wk.get_gmem_size(),
                    exp_wk.get_keys(),
                    exp_wk.get_work_queue(),
                    exp_wk.get_status() );
            }
            else
            {
                amgx::classical::selector::compute_c_hat_kernel<CTA_SIZE, SMEM_SIZE, WARP_SIZE> <<< GRID_SIZE, CTA_SIZE>>>(
                    A.get_num_rows(),
                    A.row_offsets.raw(),
                    A.col_indices.raw(),
                    cf_map.raw(),
                    s_con.raw(),
                    C_hat_start.raw(),
                    C_hat_end.raw(),
                    C_hat.raw(),
                    exp_wk.get_gmem_size(),
                    exp_wk.get_keys(),
                    exp_wk.get_work_queue(),
                    exp_wk.get_status() );
            }

            cudaCheckError();
            // Read the result from count_non_zeroes.
            cudaMemcpy( &status, exp_wk.get_status(), sizeof(int), cudaMemcpyDeviceToHost );
            done = status == 0;
        }
    }

    // get pointers to data
    int *C_hat_ptr = C_hat.raw();
    int *C_hat_start_ptr = C_hat_start.raw();
    int *C_hat_end_ptr = C_hat_end.raw();
    int S2_num_rows;;
    DistributedArranger<TConfig_d> *prep = NULL;

    if (A.is_matrix_distributed())
    {
        prep = new DistributedArranger<TConfig_d>;
        int num_owned_fine_pts = A.get_num_rows();
        int num_owned_coarse_pts = thrust_wrapper::count_if(cf_map.begin(), cf_map.begin() + num_owned_fine_pts, is_non_neg());
        int num_halo_coarse_pts = thrust_wrapper::count_if(cf_map.begin() + num_owned_fine_pts, cf_map.end(), is_non_neg());
        cudaCheckError();
        S2_num_rows = num_owned_coarse_pts;
        prep->initialize_manager(A, S2, num_owned_coarse_pts);
    }
    else
    {
        S2_num_rows = (int) thrust_wrapper::count_if(cf_map.begin(), cf_map.end(), is_non_neg());
        cudaCheckError();
    }

    // count the number of non-zeros in the interpolation matrix
    int numBlocks = (int) (A.get_num_rows() + blockSize - 1) / blockSize;
    IntVector nonZeroOffsets(S2_num_rows + 1);
    IntVector nonZerosPerRow(S2_num_rows);
    // Updating the number of nonZeros for your own coarse rows
    int *nonZerosPerRow_ptr = nonZerosPerRow.raw();
    nonZerosPerCoarseRowKernel <<< numBlocks, blockSize>>>(Anum_rows, cf_map.raw(), C_hat_start_ptr, C_hat_end_ptr,
            nonZerosPerRow_ptr);
    cudaCheckError();
    // get total with a reduction
    int nonZeros = thrust_wrapper::reduce(nonZerosPerRow.begin(), nonZerosPerRow.end());
    cudaCheckError();
    // get the offsets with an exclusive scan
    thrust_wrapper::exclusive_scan(nonZerosPerRow.begin(), nonZerosPerRow.end(), nonZeroOffsets.begin());
    cudaCheckError();
    nonZeroOffsets[S2_num_rows] = nonZeros;
    // resize S2
    S2.resize(0, 0, 0, 1);
    S2.addProps(CSR);
    S2.set_num_rows(S2_num_rows);
    S2.set_num_cols(S2_num_rows);
    S2.set_num_nz(nonZeros);
    S2.row_offsets.resize(S2_num_rows + 1);
    S2.col_indices.resize(nonZeros);
    S2.diag.resize(S2_num_rows);
    S2.set_block_dimx(A.get_block_dimx());
    S2.set_block_dimy(A.get_block_dimy());
    cudaCheckError();
    // copy nonzero offsets to the P matrix
    thrust::copy(nonZeroOffsets.begin(), nonZeroOffsets.end(), S2.row_offsets.begin());
    cudaCheckError();
    {
        const int CTA_SIZE  = 256;
        // Run the computation.
        typedef typename MatPrecisionMap<t_matPrec>::Type Value_type;

        const int NUM_WARPS = CTA_SIZE / WARP_SIZE;
        if (!A.is_matrix_distributed())
        {
            int grid_size = (A.get_num_rows() / NUM_WARPS) + 1;

            fillS2ColIndices<Value_type, CTA_SIZE, WARP_SIZE> <<< grid_size, CTA_SIZE>>>(
                A.get_num_rows(),
                cf_map.raw(),
                C_hat.raw(),
                C_hat_start.raw(),
                C_hat_end.raw(),
                S2.row_offsets.raw(),
                S2.col_indices.raw(),
                NULL,
                NULL);
        }
        else
        {
            I64Vector_d cf_map_global(cf_map.size());
            int num_owned_fine_pts = A.get_num_rows();
            int my_rank = A.manager->global_id();
            const int cta_size = 128;
            int grid_size = (num_owned_fine_pts + cta_size - 1) / cta_size;
            createCfMapGlobal <<< grid_size, cta_size>>>(cf_map.raw(), cf_map_global.raw(), S2.manager->part_offsets_h[my_rank], num_owned_fine_pts);
            cudaCheckError();

            // Exchange the cf_map_global so that we know the coarse global id of halo nodes
            cf_map_global.dirtybit = 1;
            A.manager->exchange_halo_2ring(cf_map_global, cf_map_global.tag);
            I64Vector_d S2_col_indices_global(S2.col_indices.size());

            grid_size = (A.get_num_rows() / NUM_WARPS) + 1;
            fillS2ColIndices<Value_type, CTA_SIZE, WARP_SIZE> <<< grid_size, CTA_SIZE>>>(
                A.get_num_rows(),
                cf_map.raw(),
                C_hat.raw(),
                C_hat_start.raw(),
                C_hat_end.raw(),
                S2.row_offsets.raw(),
                S2.col_indices.raw(),
                S2_col_indices_global.raw(),
                cf_map_global.raw());
            cudaCheckError();
            prep->initialize_manager_from_global_col_indices(S2, S2_col_indices_global);
        }

        if (prep != NULL) { delete prep; }

        cudaCheckError();
    }
}



template <AMGX_VecPrecision t_vecPrec, AMGX_MatPrecision t_matPrec, AMGX_IndPrecision t_indPrec>
void Selector<TemplateConfig<AMGX_host, t_vecPrec, t_matPrec, t_indPrec> >::demoteStrongEdges(const Matrix_h &A,
        const FVector &weights,
        BVector &s_con,
        const IVector &cf_map,
        const IndexType base_offset)
{
    for ( int tIdx = 0; tIdx < A.get_num_rows() - base_offset; tIdx++)
    {
        const int offset = A.row_offsets[tIdx];
        const int numj = A.row_offsets[tIdx + 1] - offset;

        if (cf_map[base_offset + tIdx] == COARSE)
        {
            // for each column
            for (int j = offset; j < offset + numj; j++)
            {
                if (!s_con[j]) { continue; } // skip weakly connected points

                int jcol = A.col_indices[j];

                if (cf_map[jcol] == COARSE) // coarse neighbour
                {
                    s_con[j] = false; //make it weak
                }
            }
        }
    }
}

template <AMGX_VecPrecision t_vecPrec, AMGX_MatPrecision t_matPrec, AMGX_IndPrecision t_indPrec>
void Selector<TemplateConfig<AMGX_device, t_vecPrec, t_matPrec, t_indPrec> >::demoteStrongEdges(const Matrix_d &A,
        const FVector &weights,
        BVector &s_con,
        const IVector &cf_map,
        const IndexType offset)
{
    const int blockSize = 256;
    const int numBlocks = min (AMGX_GRID_MAX_SIZE, (int) (((A.get_num_rows() - offset) + blockSize - 1) / blockSize));
    const int numRows = (int)(A.get_num_rows() - offset);
    resolve_boundary <<< blockSize, numBlocks>>>(A.row_offsets.raw() + offset, A.col_indices.raw(), s_con.raw(), cf_map.raw(), offset, A.get_num_rows() - offset);
    cudaCheckError();
}



template<class TConfig>
std::map<std::string, SelectorFactory<TConfig>*> &
SelectorFactory<TConfig>::getFactories( )
{
    static std::map<std::string, SelectorFactory<TConfig>*> s_factories;
    return s_factories;
}

template<class TConfig>
void SelectorFactory<TConfig>::registerFactory(string name, SelectorFactory<TConfig> *f)
{
    std::map<std::string, SelectorFactory<TConfig>*> &factories = getFactories( );
    typename map<string, SelectorFactory<TConfig> *>::const_iterator it = factories.find(name);

    if (it != factories.end())
    {
        string error = "SelectorFactory '" + name + "' has already been registered\n";
        FatalError(error.c_str(), AMGX_ERR_CORE);
    }

    factories[name] = f;
}

template<class TConfig>
void SelectorFactory<TConfig>::unregisterFactory(std::string name)
{
    std::map<std::string, SelectorFactory<TConfig>*> &factories = getFactories( );
    typename map<string, SelectorFactory<TConfig> *>::iterator it = factories.find(name);

    if (it == factories.end())
    {
        string error = "SelectorFactory '" + name + "' has not been registered\n";
        FatalError(error.c_str(), AMGX_ERR_CORE);
    }

    SelectorFactory<TConfig> *factory = it->second;
    assert( factory != NULL );
    delete factory;
    factories.erase(it);
}

template<class TConfig>
void SelectorFactory<TConfig>::unregisterFactories( )
{
    std::map<std::string, SelectorFactory<TConfig>*> &factories = getFactories( );
    typename map<std::string, SelectorFactory<TConfig> *>::iterator it = factories.begin( );

    for ( ; it != factories.end( ) ; )
    {
        SelectorFactory<TConfig> *factory = it->second;
        assert( factory != NULL );
        it++;
        delete factory;
    }

    factories.clear( );
}

template<class TConfig>
Selector<TConfig> *SelectorFactory<TConfig>::allocate(AMG_Config &cfg, const std::string &cfg_scope)
{
    std::map<std::string, SelectorFactory<TConfig>*> &factories = getFactories( );
    string selector = cfg.getParameter<string>("selector", cfg_scope);
    typename map<string, SelectorFactory<TConfig> *>::const_iterator it = factories.find(selector);

    if (it == factories.end())
    {
        string error = "SelectorFactory '" + selector + "' has not been registered\n";
        FatalError(error.c_str(), AMGX_ERR_CORE);
    }

    return it->second->create();
};

/****************************************
 * Explict instantiations
 ***************************************/
#define AMGX_CASE_LINE(CASE) template class Selector<TemplateMode<CASE>::Type>;
AMGX_FORALL_BUILDS(AMGX_CASE_LINE)
AMGX_FORCOMPLEX_BUILDS(AMGX_CASE_LINE)
#undef AMGX_CASE_LINE

#define AMGX_CASE_LINE(CASE) template class SelectorFactory<TemplateMode<CASE>::Type>;
AMGX_FORALL_BUILDS(AMGX_CASE_LINE)
AMGX_FORCOMPLEX_BUILDS(AMGX_CASE_LINE)
#undef AMGX_CASE_LINE
} // namespace classical

} // namespace amgx
