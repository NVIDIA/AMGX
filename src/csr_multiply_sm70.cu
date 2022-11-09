/* Copyright (c) 2013-2020, NVIDIA CORPORATION. All rights reserved.
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

#include <cassert>
#include <iostream>
#include <thrust/scan.h>
#include <util.h>
#include <csr_multiply.h>
#include <csr_multiply_sm70.h>
#include <device_properties.h>

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

namespace csr_multiply_sm70
{

#include <amgx_types/util.h>

#include <sm_utils.inl>
#include <hash_containers_sm70.inl> // Included inside the namespace to solve name colisions.

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

__global__
void flag_halo_rows(int *row_ids, int size, int *flagArray, int neighbor, int global_id)
{
    for (int tidx = threadIdx.x + blockIdx.x * blockDim.x; tidx < size; tidx += blockDim.x * gridDim.x)
    {
        int row_id = row_ids[tidx];
        flagArray[row_id] = tidx;
    }
}

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

template< int CTA_SIZE, int SMEM_SIZE, int WARP_SIZE, bool COUNT_ONLY >
__global__ __launch_bounds__( CTA_SIZE )
void
count_non_zeroes_kernel( const int A_num_rows,
                         const int *A_rows,
                         const int *A_cols,
                         const int *B_rows,
                         const int *B_cols,
                         int *C_rows,
                         int *C_cols,
                         int *Aq1,
                         int *Bq1,
                         int *Aq2,
                         int *Bq2,
                         const int gmem_size,
                         int *g_keys,
                         int *wk_work_queue,
                         int *wk_status )
{
    const int NUM_WARPS = CTA_SIZE / WARP_SIZE;
    // The hash keys stored in shared memory.
    __shared__ int s_keys[NUM_WARPS * SMEM_SIZE];
    // The coordinates of the thread inside the CTA/warp.
    const int warp_id = utils::warp_id();
    const int lane_id = utils::lane_id();
    // First threads load the row IDs of A needed by the CTA...
    int a_row_id = blockIdx.x * NUM_WARPS + warp_id;
    // Create local storage for the set.
    Hash_set<int, SMEM_SIZE, 4, WARP_SIZE> set( &s_keys[warp_id * SMEM_SIZE], &g_keys[a_row_id * gmem_size], gmem_size );

    // Loop over rows of A.
    for ( ; a_row_id < A_num_rows ; a_row_id = get_work( wk_work_queue, warp_id ) )
    {
        int c_row_id = a_row_id;

        if (Aq1 != NULL)
        {
            a_row_id = Aq1[a_row_id];
        }

        // Make sure we have to proceed.
        if ( COUNT_ONLY )
        {
            volatile int *status = reinterpret_cast<volatile int *>( wk_status );

            if ( set.has_failed() || *status != 0 )
            {
                return;
            }
        }

        // Clear the set.
        set.clear();
        // Load the range of the row.
        int a_col_tmp = -1;

        if ( lane_id < 2 )
        {
            a_col_tmp = utils::Ld<utils::LD_NC>::load( &A_rows[a_row_id + lane_id] );
        }

        int a_col_it  = utils::shfl( a_col_tmp, 0 );
        int a_col_end = utils::shfl( a_col_tmp, 1 );

        // Iterate over the columns of A.
        for ( a_col_it += lane_id ; utils::any(a_col_it < a_col_end) ; a_col_it += WARP_SIZE )
        {
            // Is it an active thread.
            const bool is_active = a_col_it < a_col_end;
            // Columns of A maps to rows of B. Each thread of the warp loads its A-col/B-row ID.
            int b_row_id = -1;

            if ( is_active )
            {
                b_row_id = utils::Ld<utils::LD_NC>::load( &A_cols[a_col_it] );

                //b_row_id is actually column of A
                if (Aq2 != NULL)
                {
                    b_row_id = Aq2[b_row_id];
	    		
                }

                if (Bq1 != NULL)
                {
                    b_row_id = Bq1[b_row_id];
                }
            }

            // The number of valid rows.
            const int num_rows = __popc( utils::ballot(is_active) );

            // Uniform loop: threads collaborate to load other elements.
            for ( int k = 0 ; k < num_rows ; ++k )
            {
                // Threads in the warp proceeds columns of B in the range [bColIt, bColEnd).
                const int uniform_b_row_id = utils::shfl( b_row_id, k );
                // Load the range of the row of B.
                int b_col_tmp = -1;

                if ( lane_id < 2 )
                {
                    b_col_tmp = utils::Ld<utils::LD_NC>::load( &B_rows[uniform_b_row_id + lane_id] );
                }

                int b_col_it  = utils::shfl( b_col_tmp, 0 );
                int b_col_end = utils::shfl( b_col_tmp, 1 );

                // Iterate over the range of columns of B.
                for ( b_col_it += lane_id ; utils::any(b_col_it < b_col_end) ; b_col_it += WARP_SIZE )
                {
                    int b_col_id = -1;

                    if ( b_col_it < b_col_end )
                    {
                        b_col_id = utils::Ld<utils::LD_NC>::load( &B_cols[b_col_it] );

                        // b_col_id is actually column of B
                        if (Bq2 != NULL)
                        {
                            b_col_id = Bq2[b_col_id];
                        }
                    }

                    set.insert( b_col_id, COUNT_ONLY ? wk_status : NULL );
                }
            }
        }

        // Store the results.
        if ( COUNT_ONLY )
        {
            int count = set.compute_size();

            if ( lane_id == 0 )
            {
                C_rows[c_row_id] = count;
            }
        }
        else
        {
            int c_col_tmp = -1;

            if ( lane_id < 2 )
            {
                c_col_tmp = utils::Ld<utils::LD_NC>::load( &C_rows[c_row_id + lane_id] );
            }

            int c_col_it  = utils::shfl( c_col_tmp, 0 );
            int c_col_end = utils::shfl( c_col_tmp, 1 );
            // Store the results.
            int count = c_col_end - c_col_it;

            if ( count == 0 )
            {
                continue;
            }

            set.store( count, &C_cols[c_col_it] );
        }
    }
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template< int NUM_THREADS_PER_ROW, int CTA_SIZE, int SMEM_SIZE, int WARP_SIZE, bool COUNT_ONLY >
__global__ __launch_bounds__( CTA_SIZE )
void
count_non_zeroes_kernel( const int A_num_rows,
                         const int *__restrict A_rows,
                         const int *__restrict A_cols,
                         const int *__restrict B_rows,
                         const int *__restrict B_cols,
                         int *__restrict C_rows,
                         int *__restrict C_cols,
                         int *Aq1,
                         int *Bq1,
                         int *Aq2,
                         int *Bq2,
                         const int gmem_size,
                         int *g_keys,
                         int *wk_work_queue,
                         int *wk_status )
{
    const int NUM_WARPS = CTA_SIZE / WARP_SIZE;
    const int NUM_LOADED_ROWS = WARP_SIZE / NUM_THREADS_PER_ROW;
    // The hash keys stored in shared memory.
    __shared__ /*volatile*/ int s_keys[NUM_WARPS * SMEM_SIZE];
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

    for ( ; a_row_id < A_num_rows ; a_row_id = get_work( wk_work_queue, warp_id ) )
    {
        int c_row_id = a_row_id;

        if (Aq1 != NULL)
        {
            a_row_id = Aq1[a_row_id];
        }

        // Make sure we have to proceed.
        if ( COUNT_ONLY )
        {
            volatile int *status = reinterpret_cast<volatile int *>( wk_status );

            if ( set.has_failed() || *status != 0 )
            {
                return;
            }
        }

        // Clear the set.
        set.clear();
        // Load the range of the row.
        int a_col_tmp = -1;

        if ( lane_id < 2 )
        {
            a_col_tmp = utils::Ld<utils::LD_NC>::load( &A_rows[a_row_id + lane_id] );
        }

        int a_col_it  = utils::shfl( a_col_tmp, 0 );
        int a_col_end = utils::shfl( a_col_tmp, 1 );

        // Iterate over the columns of A.
        for ( a_col_it += lane_id ; utils::any(a_col_it < a_col_end) ; a_col_it += WARP_SIZE )
        {
            // Is it an active thread.
            const bool is_active = a_col_it < a_col_end;
            // Columns of A maps to rows of B. Each thread of the warp loads its A-col/B-row ID.
            int b_row_id = -1;

            if ( is_active )
            {
                b_row_id = utils::Ld<utils::LD_NC>::load( &A_cols[a_col_it] );

                //b_row_id is actually column of A
                if (Aq2 != NULL)
                {
                    b_row_id = Aq2[b_row_id];
                }

                if (Bq1 != NULL)
                {
                    b_row_id = Bq1[b_row_id];
                }
            }

            const int num_rows = __popc( utils::ballot(is_active) );

            // Uniform loop: threads collaborate to load other elements.
            for ( int k = 0 ; k < num_rows ; k += NUM_LOADED_ROWS )
            {
                int local_k = k + lane_id_div_num_threads;
                // Is it an active thread.
                bool is_active_k = local_k < num_rows;
                // Threads in the warp proceeds columns of B in the range [bColIt, bColEnd).
                const int uniform_b_row_id = utils::shfl( b_row_id, local_k );
                // Load the range of the row of B.
                int b_col_tmp = -1;

                if ( is_active_k && lane_id_mod_num_threads < 2 )
                {
                    b_col_tmp = utils::Ld<utils::LD_NC>::load( &B_rows[uniform_b_row_id + lane_id_mod_num_threads] );
                }

                int b_col_it  = utils::shfl( b_col_tmp, lane_id_div_num_threads * NUM_THREADS_PER_ROW + 0 );
                int b_col_end = utils::shfl( b_col_tmp, lane_id_div_num_threads * NUM_THREADS_PER_ROW + 1 );

                // Iterate over the range of columns of B.
                for ( b_col_it += lane_id_mod_num_threads ; utils::any(b_col_it < b_col_end) ; b_col_it += NUM_THREADS_PER_ROW )
                {
                    int b_col_id = -1;

                    if ( b_col_it < b_col_end )
                    {
                        b_col_id = utils::Ld<utils::LD_NC>::load( &B_cols[b_col_it] );

                        // b_col_id is actually column of B
                        if (Bq2 != NULL)
                        {
                            b_col_id = Bq2[b_col_id];
                        }
                    }

                    set.insert( b_col_id, COUNT_ONLY ? wk_status : NULL );
                }
            }
        }

        // Store the results.
        if ( COUNT_ONLY )
        {
            int count = set.compute_size();

            if ( lane_id == 0 )
            {
                C_rows[c_row_id] = count;
            }
        }
        else
        {
            int c_col_tmp = -1;

            if ( lane_id < 2 )
            {
                c_col_tmp = utils::Ld<utils::LD_NC>::load( &C_rows[c_row_id + lane_id] );
            }

            int c_col_it  = utils::shfl( c_col_tmp, 0 );
            int c_col_end = utils::shfl( c_col_tmp, 1 );
            // Store the results.
            int count = c_col_end - c_col_it;

            if ( count == 0 )
            {
                continue;
            }

            set.store( count, &C_cols[c_col_it] );
        }
    }
}

template <int SMEM_SIZE, int NUM_HASH_FCTS, int WARP_SIZE, bool COUNT_ONLY >
__device__ __forceinline__ void sparse_add_process_row(int row_id, const int *__restrict__ row_offsets, const int *__restrict__ col_indices, int lane_id, Hash_set<int, SMEM_SIZE, NUM_HASH_FCTS, WARP_SIZE> &set, int *wk_status, int global_id, bool print_flag)
{
    // Load the range of the row of RAP_int
    int col_tmp = -1;

    if ( lane_id < 2 )
    {
        col_tmp = utils::Ld<utils::LD_NC>::load( &row_offsets[row_id + lane_id] );
    }

    int col_it  = utils::shfl( col_tmp, 0 );
    int col_end = utils::shfl( col_tmp, 1 );

    // Iterate over the columns of RAP_int
    for ( col_it += lane_id ; utils::any(col_it < col_end) ; col_it += WARP_SIZE )
    {
        int col_id = -1;

        if ( col_it < col_end )
        {
            col_id = utils::Ld<utils::LD_NC>::load( &col_indices[col_it] );
        }

        set.insert( col_id, COUNT_ONLY ? wk_status : NULL );
    }
}


template <typename Value_type, int SMEM_SIZE, int NUM_HASH_FCTS, int WARP_SIZE>
__device__ __forceinline__ void sparse_add_process_row_values(int row_id, const int *__restrict__ row_offsets, const int *__restrict__ col_indices, const Value_type *__restrict vals, int lane_id, Hash_map<int, Value_type, SMEM_SIZE, NUM_HASH_FCTS, WARP_SIZE> &map, int *wk_status)
{
    // Load the range of the row.
    int col_tmp = -1;

    if ( lane_id < 2 )
    {
        col_tmp = utils::Ld<utils::LD_NC>::load( &row_offsets[row_id + lane_id] );
    }

    int col_it  = utils::shfl( col_tmp, 0 );
    int col_end = utils::shfl( col_tmp, 1 );

    // Iterate over the columns of A.
    for ( col_it += lane_id ; utils::any(col_it < col_end) ; col_it += WARP_SIZE )
    {
        const bool is_active = col_it < col_end;
        // Columns of A maps to rows of B. Each thread of the warp loads its A-col/B-row ID.
        int col_id(-1);
        Value_type value = amgx::types::util<Value_type>::get_zero();

        if ( is_active )
        {
            col_id = utils::Ld<utils::LD_NC>::load( &col_indices[col_it] );
            value  = utils::Ld<utils::LD_NC>::load( &vals[col_it] );
        }

        map.insert( col_id, value, wk_status );
    }
}




template<int CTA_SIZE, int SMEM_SIZE, int WARP_SIZE, bool COUNT_ONLY >
__global__ __launch_bounds__( CTA_SIZE )
void
count_non_zeroes_RAP_ext_kernel( const int RAP_int_num_rows,
                                 const int *__restrict RAP_int_rows,
                                 const int *__restrict RAP_int_cols,
                                 int **RAP_ext_row_ptrs,
                                 int **RAP_ext_col_ptrs,
                                 int *__restrict RAP_rows,
                                 int *__restrict RAP_cols,
                                 int **flagArray_ptrs,
                                 const int gmem_size,
                                 int *g_keys,
                                 int *wk_work_queue,
                                 int *wk_status,
                                 int num_neighbors,
                                 int global_id )
{
    const int NUM_WARPS = CTA_SIZE / WARP_SIZE;
    // The hash keys stored in shared memory.
    __shared__ /*volatile*/ int s_keys[NUM_WARPS * SMEM_SIZE];
    // The coordinates of the thread inside the CTA/warp.
    const int warp_id = utils::warp_id( );
    const int lane_id = utils::lane_id( );
    // First threads load the row IDs of A needed by the CTA...
    int rap_int_row_id = blockIdx.x * NUM_WARPS + warp_id;
    // Create local storage for the set.
    Hash_set<int, SMEM_SIZE, 4, WARP_SIZE> set( &s_keys[warp_id * SMEM_SIZE], &g_keys[rap_int_row_id * gmem_size], gmem_size );

    // Loop over rows of RAP_ext
    for ( ; rap_int_row_id < RAP_int_num_rows; rap_int_row_id = get_work( wk_work_queue, warp_id ) )
    {
        // Make sure we have to proceed.
        if ( COUNT_ONLY )
        {
            volatile int *status = reinterpret_cast<volatile int *>( wk_status );

            if ( set.has_failed() || *status != 0 )
            {
                return;
            }
        }

        // Clear the set.
        set.clear();
        // ---------------------------------
        // First process RAP_int
        // ---------------------------------
        bool print_flag = false;

        if (rap_int_row_id == 3 && global_id == 0)
        {
            print_flag = true;
        }

        sparse_add_process_row<SMEM_SIZE, 4, WARP_SIZE, COUNT_ONLY>(rap_int_row_id, RAP_int_rows, RAP_int_cols, lane_id, set, wk_status, global_id, print_flag);

        // ---------------------------------
        // Then process RAP_ext
        // ---------------------------------

        for (int i = 0; i < num_neighbors; i++)
        {
            int flag = flagArray_ptrs[i][rap_int_row_id];

            if (flag != -1)
            {
                int *RAP_ext_rows = RAP_ext_row_ptrs[i];
                int *RAP_ext_cols = RAP_ext_col_ptrs[i];
                int pos_in_row_ext = flag;
                sparse_add_process_row<SMEM_SIZE, 4, WARP_SIZE, COUNT_ONLY>(pos_in_row_ext, RAP_ext_rows, RAP_ext_cols, lane_id, set, wk_status, global_id, print_flag);
            }
        }

        // Store the results.
        if ( COUNT_ONLY )
        {
            int count = set.compute_size();

            if ( lane_id == 0 )
            {
                RAP_rows[rap_int_row_id] = count;
            }
        }
        else
        {
            int rap_col_tmp = -1;

            if ( lane_id < 2 )
            {
                rap_col_tmp = utils::Ld<utils::LD_NC>::load( &RAP_rows[rap_int_row_id + lane_id] );
            }

            int rap_col_it  = utils::shfl( rap_col_tmp, 0 );
            int rap_col_end = utils::shfl( rap_col_tmp, 1 );
            // Store the results.
            int count = rap_col_end - rap_col_it;

            if ( count == 0 )
            {
                continue;
            }

            set.store( count, &RAP_cols[rap_col_it] );
        }
    }
}




///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

struct Without_external_diag
{
    static __device__ __forceinline__ bool is_active  ( int a_col_it, int a_col_end ) { return a_col_it < a_col_end; }
    static __device__ __forceinline__ bool is_boundary( int a_col_it, int a_col_end ) { return false; }
};

// ====================================================================================================================

struct With_external_diag
{
    static __device__ __forceinline__ bool is_active  ( int a_col_it, int a_col_end ) { return a_col_it <= a_col_end; }
    static __device__ __forceinline__ bool is_boundary( int a_col_it, int a_col_end ) { return a_col_it == a_col_end; }
};

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template< int CTA_SIZE, int SMEM_SIZE, int WARP_SIZE, bool COUNT_ONLY, typename Diag_traits >
__global__ __launch_bounds__( CTA_SIZE )
void
count_non_zeroes_ilu1_kernel( const int A_num_rows,
                              const int *__restrict A_rows,
                              const int *__restrict A_cols,
                              const int *__restrict A_coloring,
                              int *__restrict C_rows,
                              int *__restrict C_cols,
                              const int gmem_size,
                              int *g_keys,
                              int *wk_work_queue,
                              int *wk_status )
{
    const int NUM_WARPS = CTA_SIZE / WARP_SIZE;
    // Tables to broadcast values.
    __shared__ volatile int s_b_rows[CTA_SIZE], s_b_colors[CTA_SIZE];
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
        // Make sure we have to proceed.
        if ( COUNT_ONLY )
        {
            volatile int *status = reinterpret_cast<volatile int *>( wk_status );

            if ( set.has_failed() || *status != 0 )
            {
                return;
            }
        }

        // Clear the set.
        set.clear();
        // The color of the row.
        int a_row_color = A_coloring[a_row_id];
        // Load the range of the row.
        int a_col_tmp = -1;

        if ( lane_id < 2 )
        {
            a_col_tmp = utils::Ld<utils::LD_NC>::load( &A_rows[a_row_id + lane_id] );
        }

        int a_col_it  = utils::shfl( a_col_tmp, 0 );
        int a_col_end = utils::shfl( a_col_tmp, 1 );

        // Iterate over the columns of A.
        for ( a_col_it += lane_id ; utils::any(Diag_traits::is_active(a_col_it, a_col_end)) ; a_col_it += WARP_SIZE )
        {
            // Is it an active thread.
            const bool is_active = a_col_it < a_col_end;
            // Columns of A maps to rows of B. Each thread of the warp loads its A-col/B-row ID.
            int b_row_id = -1;

            if ( is_active )
            {
                b_row_id = utils::Ld<utils::LD_NC>::load( &A_cols[a_col_it] );
            }

            if ( Diag_traits::is_boundary(a_col_it, a_col_end) )
            {
                b_row_id = a_row_id;
            }

            // Push the columns in the set.
            set.insert( b_row_id, COUNT_ONLY ? wk_status : NULL );

            // Skip computation if the color of the row is 0.
            if ( a_row_color != 0 )
            {
                // Gather the colors of the columns.
                int b_row_color = -1;

                if ( is_active )
                {
                    b_row_color = A_coloring[b_row_id];
                }

                // The number of valid rows.
                int pred = is_active && b_row_color < a_row_color;
                int vote = utils::ballot( pred );
                int dest = __popc( vote & utils::lane_mask_lt() );

                if ( pred )
                {
                    s_b_rows  [warp_id * WARP_SIZE + dest] = b_row_id;
                    s_b_colors[warp_id * WARP_SIZE + dest] = b_row_color; // TODO: store an int2 rather than 2 ints of SM35 with 64bit banks.
                }

                const int num_rows = __popc( vote );

                // Uniform loop: threads collaborate to load other elements.
                for ( int k = 0 ; k < num_rows ; ++k )
                {
                    // Threads in the warp proceeds columns of B in the range [bColIt, bColEnd).
                    const int uniform_b_row_id = s_b_rows  [warp_id * WARP_SIZE + k];
                    const int uniform_b_color  = s_b_colors[warp_id * WARP_SIZE + k];
                    // Load the range of the row of B.
                    int b_col_tmp = -1;

                    if ( lane_id < 2 )
                    {
                        b_col_tmp = utils::Ld<utils::LD_NC>::load( &A_rows[uniform_b_row_id + lane_id] );
                    }

                    int b_col_it  = utils::shfl( b_col_tmp, 0 );
                    int b_col_end = utils::shfl( b_col_tmp, 1 );

                    // Iterate over the range of columns of B.
                    for ( b_col_it += lane_id ; utils::any(b_col_it < b_col_end) ; b_col_it += WARP_SIZE )
                    {
                        int b_col_id = -1, b_col_color = -1;

                        if ( b_col_it < b_col_end )
                        {
                            b_col_id    = utils::Ld<utils::LD_NC>::load( &A_cols[b_col_it] );
                            b_col_color = utils::Ld<utils::LD_NC>::load( &A_coloring[b_col_id] );
                        }

                        int item = -1;

                        if ( b_col_color >= uniform_b_color && b_col_color != a_row_color )
                        {
                            item = b_col_id;
                        }

                        set.insert( item, COUNT_ONLY ? wk_status : NULL );
                    }
                }
            }
        }

        // Store the results.
        if ( COUNT_ONLY )
        {
            int count = set.compute_size();

            if ( lane_id == 0 )
            {
                C_rows[a_row_id] = count;
            }
        }
        else
        {
            int c_col_tmp = -1;

            if ( lane_id < 2 )
            {
                c_col_tmp = utils::Ld<utils::LD_NC>::load( &C_rows[a_row_id + lane_id] );
            }

            int c_col_it  = utils::shfl( c_col_tmp, 0 );
            int c_col_end = utils::shfl( c_col_tmp, 1 );
            // Store the results.
            int count = c_col_end - c_col_it;

            if ( count == 0 )
            {
                continue;
            }

            set.store( count, &C_cols[c_col_it] );
        }
    }
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template< int NUM_THREADS_PER_ROW, int CTA_SIZE, int SMEM_SIZE, int WARP_SIZE, bool COUNT_ONLY, typename Diag_traits >
__global__ __launch_bounds__( CTA_SIZE )
void
count_non_zeroes_ilu1_kernel( const int A_num_rows,
                              const int *__restrict A_rows,
                              const int *__restrict A_cols,
                              const int *__restrict A_coloring,
                              int *__restrict C_rows,
                              int *__restrict C_cols,
                              const int gmem_size,
                              int *g_keys,
                              int *wk_work_queue,
                              int *wk_status )
{
    const int NUM_WARPS = CTA_SIZE / WARP_SIZE;
    const int NUM_LOADED_ROWS = WARP_SIZE / NUM_THREADS_PER_ROW;
    // Tables to broadcast values.
    __shared__ volatile int s_b_rows[CTA_SIZE], s_b_colors[CTA_SIZE];
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
        // Make sure we have to proceed.
        if ( COUNT_ONLY )
        {
            volatile int *status = reinterpret_cast<volatile int *>( wk_status );

            if ( set.has_failed() || *status != 0 )
            {
                return;
            }
        }

        // Clear the set.
        set.clear();
        // The color of the row.
        int a_row_color = A_coloring[a_row_id];
        // Load the range of the row.
        int a_col_tmp = -1;

        if ( lane_id < 2 )
        {
            a_col_tmp = utils::Ld<utils::LD_NC>::load( &A_rows[a_row_id + lane_id] );
        }

        int a_col_it  = utils::shfl( a_col_tmp, 0 );
        int a_col_end = utils::shfl( a_col_tmp, 1 );

        // Iterate over the columns of A.
        for ( a_col_it += lane_id ; utils::any(Diag_traits::is_active(a_col_it, a_col_end)) ; a_col_it += WARP_SIZE )
        {
            // Is it an active thread.
            const bool is_active = a_col_it < a_col_end;
            // Columns of A maps to rows of B. Each thread of the warp loads its A-col/B-row ID.
            int b_row_id = -1;

            if ( is_active )
            {
                b_row_id = utils::Ld<utils::LD_NC>::load( &A_cols[a_col_it] );
            }

            if ( Diag_traits::is_boundary(a_col_it, a_col_end) )
            {
                b_row_id = a_row_id;
            }

            // Push the columns in the set.
            set.insert( b_row_id, COUNT_ONLY ? wk_status : NULL );

            // Skip computation if the color of the row is 0.
            if ( a_row_color != 0 )
            {
                // Gather the colors of the columns.
                int b_row_color = -1;

                if ( is_active )
                {
                    b_row_color = A_coloring[b_row_id];
                }

                // The number of valid rows.
                int pred = is_active && b_row_color < a_row_color;
                int vote = utils::ballot( pred );
                int dest = __popc( vote & utils::lane_mask_lt() );

                if ( pred )
                {
                    s_b_rows  [warp_id * WARP_SIZE + dest] = b_row_id;
                    s_b_colors[warp_id * WARP_SIZE + dest] = b_row_color;
                }

                const int num_rows = __popc( vote );

                // Uniform loop: threads collaborate to load other elements.
                for ( int k = 0 ; k < num_rows ; k += NUM_LOADED_ROWS )
                {
                    int local_k = k + lane_id_div_num_threads;
                    // Is it an active thread.
                    bool is_active_k = local_k < num_rows;
                    // Threads in the warp proceeds columns of B in the range [bColIt, bColEnd).
                    int uniform_b_row_id = -1, uniform_b_color = -1;

                    if ( is_active_k )
                    {
                        uniform_b_row_id = s_b_rows  [warp_id * WARP_SIZE + local_k];
                        uniform_b_color  = s_b_colors[warp_id * WARP_SIZE + local_k];
                    }

                    // Load the range of the row of B.
                    int b_col_tmp = -1;

                    if ( is_active_k && lane_id_mod_num_threads < 2 )
                    {
                        b_col_tmp = utils::Ld<utils::LD_NC>::load( &A_rows[uniform_b_row_id + lane_id_mod_num_threads] );
                    }

                    int b_col_it  = utils::shfl( b_col_tmp, lane_id_div_num_threads * NUM_THREADS_PER_ROW + 0 );
                    int b_col_end = utils::shfl( b_col_tmp, lane_id_div_num_threads * NUM_THREADS_PER_ROW + 1 );

                    // Iterate over the range of columns of B.
                    for ( b_col_it += lane_id_mod_num_threads ; utils::any(b_col_it < b_col_end) ; b_col_it += NUM_THREADS_PER_ROW )
                    {
                        int b_col_id = -1, b_col_color = -1;

                        if ( b_col_it < b_col_end )
                        {
                            b_col_id    = utils::Ld<utils::LD_NC>::load( &A_cols[b_col_it] );
                            b_col_color = utils::Ld<utils::LD_NC>::load( &A_coloring[b_col_id] );
                        }

                        int item = -1;

                        if ( b_col_color >= uniform_b_color && b_col_color != a_row_color )
                        {
                            item = b_col_id;
                        }

                        set.insert( item, COUNT_ONLY ? wk_status : NULL );
                    }
                }
            }
        }

        // Store the results.
        if ( COUNT_ONLY )
        {
            int count = set.compute_size();

            if ( lane_id == 0 )
            {
                C_rows[a_row_id] = count;
            }
        }
        else
        {
            int c_col_tmp = -1;

            if ( lane_id < 2 )
            {
                c_col_tmp = utils::Ld<utils::LD_NC>::load( &C_rows[a_row_id + lane_id] );
            }

            int c_col_it  = utils::shfl( c_col_tmp, 0 );
            int c_col_end = utils::shfl( c_col_tmp, 1 );
            // Store the results.
            int count = c_col_end - c_col_it;

            if ( count == 0 )
            {
                continue;
            }

            set.store( count, &C_cols[c_col_it] );
        }
    }
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template< typename Value_type, int CTA_SIZE, int SMEM_SIZE, int WARP_SIZE >
__global__ __launch_bounds__( CTA_SIZE, 6 )
void
compute_values_kernel( const int A_num_rows,
                       const int *__restrict A_rows,
                       const int *__restrict A_cols,
                       const Value_type *__restrict A_vals,
                       const int *__restrict B_rows,
                       const int *__restrict B_cols,
                       const Value_type *__restrict B_vals,
                       const int *__restrict C_rows,
                       int *__restrict C_cols,
                       Value_type *__restrict C_vals,
                       int *Aq1,
                       int *Bq1,
                       int *Aq2,
                       int *Bq2,
                       const int gmem_size,
                       int *g_keys,
                       Value_type *g_vals,
                       int *wk_work_queue,
                       int *wk_status )
{
    const int NUM_WARPS = CTA_SIZE / WARP_SIZE;
    // The hash keys stored in shared memory.
    __shared__ /*volatile*/ int s_keys[NUM_WARPS * SMEM_SIZE];
    // The hash values stored in shared memory.
    __shared__ Value_type s_vals[NUM_WARPS * SMEM_SIZE];
    // The coordinates of the thread inside the CTA/warp.
    const int warp_id = utils::warp_id();
    const int lane_id = utils::lane_id();
    // First threads load the row IDs of A needed by the CTA...
    int a_row_id = blockIdx.x * NUM_WARPS + warp_id;
    // Create local storage for the set.
    Hash_map<int, Value_type, SMEM_SIZE, 4, WARP_SIZE> map(&s_keys[warp_id * SMEM_SIZE],
            &g_keys[a_row_id * gmem_size],
            &s_vals[warp_id * SMEM_SIZE],
            &g_vals[a_row_id * gmem_size],
            gmem_size );

    // Loop over rows of A.
    for ( ; a_row_id < A_num_rows ; a_row_id = get_work( wk_work_queue, warp_id ) )
    {
        int c_row_id = a_row_id;

        if (Aq1 != NULL)
        {
            a_row_id = Aq1[a_row_id];
        }

        // Clear the map.
        map.clear();
        // Load the range of the row.
        int a_col_tmp = -1;

        if ( lane_id < 2 )
        {
            a_col_tmp = utils::Ld<utils::LD_NC>::load( &A_rows[a_row_id + lane_id] );
        }

        int a_col_it  = utils::shfl( a_col_tmp, 0 );
        int a_col_end = utils::shfl( a_col_tmp, 1 );

        // Iterate over the columns of A.
        for ( a_col_it += lane_id ; utils::any(a_col_it < a_col_end) ; a_col_it += WARP_SIZE )
        {
            // Is it an active thread.
            const bool is_active = a_col_it < a_col_end;
            // Columns of A maps to rows of B. Each thread of the warp loads its A-col/B-row ID.
            int b_row_id = -1;
            Value_type a_value = amgx::types::util<Value_type>::get_zero();

            if ( is_active )
            {
                b_row_id = utils::Ld<utils::LD_NC>::load( &A_cols[a_col_it] );
                a_value  = utils::Ld<utils::LD_NC>::load( &A_vals[a_col_it] );

                //b_row_id is actually column of A
                if (Aq2 != NULL)
                {
                    b_row_id = Aq2[b_row_id];
                }

                if (Bq1 != NULL)
                {
                    b_row_id = Bq1[b_row_id];
                }
            }

            const int num_rows = __popc( utils::ballot(is_active) );

            // Uniform loop: threads collaborate to load other elements.
            for ( int k = 0 ; k < num_rows ; ++k )
            {
                // Threads in the warp proceeds columns of B in the range [bColIt, bColEnd).
                const int uniform_b_row_id = utils::shfl( b_row_id, k );
                // The value of A.
                const Value_type uniform_a_value = utils::shfl( a_value, k );
                // Load the range of the row of B.
                int b_col_tmp = -1;

                if ( lane_id < 2 )
                {
                    b_col_tmp = utils::Ld<utils::LD_NC>::load( &B_rows[uniform_b_row_id + lane_id] );
                }

                int b_col_it  = utils::shfl( b_col_tmp, 0 );
                int b_col_end = utils::shfl( b_col_tmp, 1 );

                // Iterate over the range of columns of B.
                for ( b_col_it += lane_id ; utils::any(b_col_it < b_col_end) ; b_col_it += WARP_SIZE )
                {
                    int b_col_id = -1;
                    Value_type b_value = amgx::types::util<Value_type>::get_zero();

                    if ( b_col_it < b_col_end )
                    {
                        b_col_id = utils::Ld<utils::LD_NC>::load( &B_cols[b_col_it] );
                        b_value  = utils::Ld<utils::LD_NC>::load( &B_vals[b_col_it] );

                        if (Bq2 != NULL)
                        {
                            b_col_id = Bq2[b_col_id];
                        }
                    }

                    map.insert( b_col_id, uniform_a_value * b_value, wk_status );
                }
            }
        }

        // Store the results.
        int c_col_tmp = -1;

        if ( lane_id < 2 )
        {
            c_col_tmp = utils::Ld<utils::LD_NC>::load( &C_rows[c_row_id + lane_id] );
        }

        int c_col_it  = utils::shfl( c_col_tmp, 0 );
        int c_col_end = utils::shfl( c_col_tmp, 1 );
        // Store the results.
        int count = c_col_end - c_col_it;

        if ( count == 0 )
        {
            continue;
        }

        map.store( count, &C_cols[c_col_it], &C_vals[c_col_it] );
    }

}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template< int NUM_THREADS_PER_ROW, typename Value_type, int CTA_SIZE, int SMEM_SIZE, int WARP_SIZE >
__global__ __launch_bounds__( CTA_SIZE, 6 )
void
compute_values_kernel( const int A_num_rows,
                       const int *__restrict A_rows,
                       const int *__restrict A_cols,
                       const Value_type *__restrict A_vals,
                       const int *__restrict B_rows,
                       const int *__restrict B_cols,
                       const Value_type *__restrict B_vals,
                       const int *__restrict C_rows,
                       int *__restrict C_cols,
                       Value_type *__restrict C_vals,
                       int *Aq1,
                       int *Bq1,
                       int *Aq2,
                       int *Bq2,
                       const int gmem_size,
                       int *g_keys,
                       Value_type *g_vals,
                       int *wk_work_queue,
                       int *wk_status )
{

    const int NUM_WARPS = CTA_SIZE / WARP_SIZE;
    const int NUM_LOADED_ROWS = WARP_SIZE / NUM_THREADS_PER_ROW;
    // The hash keys stored in shared memory.
    __shared__ /*volatile*/ int s_keys[NUM_WARPS * SMEM_SIZE];
    // The hash values stored in shared memory.
    __shared__ Value_type s_vals[NUM_WARPS * SMEM_SIZE];
    // The coordinates of the thread inside the CTA/warp.
    const int warp_id = utils::warp_id( );
    const int lane_id = utils::lane_id( );
    // Constants.
    const int lane_id_div_num_threads = lane_id / NUM_THREADS_PER_ROW;
    const int lane_id_mod_num_threads = lane_id % NUM_THREADS_PER_ROW;
    // First threads load the row IDs of A needed by the CTA...
    int a_row_id = blockIdx.x * NUM_WARPS + warp_id;
    // Create local storage for the set.
    Hash_map<int, Value_type, SMEM_SIZE, 4, WARP_SIZE> map(&s_keys[warp_id * SMEM_SIZE],
            &g_keys[a_row_id * gmem_size],
            &s_vals[warp_id * SMEM_SIZE],
            &g_vals[a_row_id * gmem_size],
            gmem_size );

    // Loop over rows of A.
    for ( ; a_row_id < A_num_rows ; a_row_id = get_work( wk_work_queue, warp_id ) )
    {
        int c_row_id = a_row_id;

        if (Aq1 != NULL)
        {
            a_row_id = Aq1[a_row_id];
        }

        // Clear the map.
        map.clear();
        // Load the range of the row.
        int a_col_tmp = -1;

        if ( lane_id < 2 )
        {
            a_col_tmp = utils::Ld<utils::LD_NC>::load( &A_rows[a_row_id + lane_id] );
        }

        int a_col_it  = utils::shfl( a_col_tmp, 0 );
        int a_col_end = utils::shfl( a_col_tmp, 1 );

        // Iterate over the columns of A.
        for ( a_col_it += lane_id ; utils::any(a_col_it < a_col_end) ; a_col_it += WARP_SIZE )
        {
            // Is it an active thread.
            const bool is_active = a_col_it < a_col_end;
            // Columns of A maps to rows of B. Each thread of the warp loads its A-col/B-row ID.
            int b_row_id(-1);
            Value_type a_value = amgx::types::util<Value_type>::get_zero();

            if ( is_active )
            {
                b_row_id = utils::Ld<utils::LD_NC>::load( &A_cols[a_col_it] );
                a_value  = utils::Ld<utils::LD_NC>::load( &A_vals[a_col_it] );

                //b_row_id is actually column of A
                if (Aq2 != NULL)
                {
                    b_row_id = Aq2[b_row_id];
                }

                if (Bq1 != NULL)
                {
                    b_row_id = Bq1[b_row_id];
                }
            }

            const int num_rows = __popc( utils::ballot(is_active) );

            // Uniform loop: threads collaborate to load other elements.
            for ( int k = 0 ; k < num_rows ; k += NUM_LOADED_ROWS )
            {
                int local_k = k + lane_id_div_num_threads;
                // Is it an active thread.
                bool is_active_k = local_k < num_rows;
                // Threads in the warp proceeds columns of B in the range [bColIt, bColEnd).
                const int uniform_b_row_id = utils::shfl( b_row_id, k + lane_id_div_num_threads );
                // The value of A.
                const Value_type uniform_a_value = utils::shfl( a_value, k + lane_id_div_num_threads );
                // Load the range of the row of B.
                int b_col_tmp = -1;

                if ( is_active_k && lane_id_mod_num_threads < 2 )
                {
                    b_col_tmp = utils::Ld<utils::LD_NC>::load( &B_rows[uniform_b_row_id + lane_id_mod_num_threads] );
                }

                int b_col_it  = utils::shfl( b_col_tmp, lane_id_div_num_threads * NUM_THREADS_PER_ROW + 0 );
                int b_col_end = utils::shfl( b_col_tmp, lane_id_div_num_threads * NUM_THREADS_PER_ROW + 1 );

                // Iterate over the range of columns of B.
                for ( b_col_it += lane_id_mod_num_threads ; utils::any(b_col_it < b_col_end) ; b_col_it += NUM_THREADS_PER_ROW )
                {
                    int b_col_id(-1);
                    Value_type b_value = amgx::types::util<Value_type>::get_zero();

                    if ( b_col_it < b_col_end )
                    {
                        b_col_id = utils::Ld<utils::LD_NC>::load( &B_cols[b_col_it] );
                        b_value  = utils::Ld<utils::LD_NC>::load( &B_vals[b_col_it] );

                        //b_col_id is actually column of B
                        if (Bq2 != NULL)
                        {
                            b_col_id = Bq2[b_col_id];
                        }
                    }

                    map.insert( b_col_id, uniform_a_value * b_value, wk_status );
                }
            }
        }

        // Store the results.
        int c_col_tmp = -1;

        if ( lane_id < 2 )
        {
            c_col_tmp = utils::Ld<utils::LD_NC>::load( &C_rows[c_row_id + lane_id] );
        }

        int c_col_it  = utils::shfl( c_col_tmp, 0 );
        int c_col_end = utils::shfl( c_col_tmp, 1 );
        // Store the results.
        int count = c_col_end - c_col_it;

        if ( count == 0 )
        {
            continue;
        }

        map.store( count, &C_cols[c_col_it], &C_vals[c_col_it] );
    }

}


template< typename Value_type, int CTA_SIZE, int SMEM_SIZE, int WARP_SIZE >
__global__ __launch_bounds__( CTA_SIZE, 6 )
void
compute_values_RAP_ext_kernel( const int RAP_int_num_rows,
                               const int *__restrict RAP_int_rows,
                               const int *__restrict RAP_int_cols,
                               const Value_type *__restrict RAP_int_vals,
                               int **RAP_ext_row_ptrs,
                               int **RAP_ext_col_ptrs,
                               Value_type **RAP_ext_val_ptrs,
                               int *__restrict RAP_rows,
                               int *__restrict RAP_cols,
                               Value_type *__restrict RAP_vals,
                               int **flagArray_ptrs,
                               const int gmem_size,
                               int *g_keys,
                               Value_type *g_vals,
                               int *wk_work_queue,
                               int num_neighbors,
                               int *wk_status )
{
    const int NUM_WARPS = CTA_SIZE / WARP_SIZE;
    // The hash keys stored in shared memory.
    __shared__ /*volatile*/ int s_keys[NUM_WARPS * SMEM_SIZE];
    // The hash values stored in shared memory.
    __shared__ Value_type s_vals[NUM_WARPS * SMEM_SIZE];
    // The coordinates of the thread inside the CTA/warp.
    const int warp_id = utils::warp_id( );
    const int lane_id = utils::lane_id( );
    // First threads load the row IDs of A needed by the CTA...
    int rap_int_row_id = blockIdx.x * NUM_WARPS + warp_id;
    // Create local storage for the set.
    Hash_map<int, Value_type, SMEM_SIZE, 4, WARP_SIZE> map( &s_keys[warp_id * SMEM_SIZE],
            &g_keys[rap_int_row_id * gmem_size],
            &s_vals[warp_id * SMEM_SIZE],
            &g_vals[rap_int_row_id * gmem_size],
            gmem_size );

    // Loop over rows of RAP_ext
    for ( ; rap_int_row_id < RAP_int_num_rows ; rap_int_row_id = get_work( wk_work_queue, warp_id ) )
    {
        // Clear the map.
        map.clear();
        // ---------------------------------
        // First process RAP_int
        // ---------------------------------
        sparse_add_process_row_values<Value_type, SMEM_SIZE, 4, WARP_SIZE>(rap_int_row_id, RAP_int_rows, RAP_int_cols, RAP_int_vals, lane_id, map, wk_status);

        // ---------------------------------
        // Then process RAP_ext
        // ---------------------------------

        for (int i = 0; i < num_neighbors; i++)
        {
            int flag = flagArray_ptrs[i][rap_int_row_id];

            if (flag != -1)
            {
                int *RAP_ext_rows = RAP_ext_row_ptrs[i];
                int *RAP_ext_cols = RAP_ext_col_ptrs[i];
                Value_type *RAP_ext_vals = RAP_ext_val_ptrs[i];
                int pos_in_row_ext = flag;
                sparse_add_process_row_values<Value_type, SMEM_SIZE, 4, WARP_SIZE>(pos_in_row_ext, RAP_ext_rows, RAP_ext_cols, RAP_ext_vals, lane_id, map, wk_status);
            }
        }

        // Store the results.
        int rap_col_tmp = -1;

        if ( lane_id < 2 )
        {
            rap_col_tmp = utils::Ld<utils::LD_NC>::load( &RAP_rows[rap_int_row_id + lane_id] );
        }

        int rap_col_it  = utils::shfl( rap_col_tmp, 0 );
        int rap_col_end = utils::shfl( rap_col_tmp, 1 );
        // Store the results.
        int count = rap_col_end - rap_col_it;

        if ( count == 0 )
        {
            continue;
        }

        map.store( count, &RAP_cols[rap_col_it], &RAP_vals[rap_col_it] );
    }
}




///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace csr_multiply_sm70

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

namespace amgx
{

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

enum { WARP_SIZE = 32, SMEM_SIZE = 128 };

// ====================================================================================================================

template< AMGX_VecPrecision V, AMGX_MatPrecision M, AMGX_IndPrecision I >
CSR_Multiply_Sm70<TemplateConfig<AMGX_device, V, M, I> >::CSR_Multiply_Sm70( bool allocate_values, int grid_size, int max_warp_count, int gmem_size ) :
    Base(allocate_values, grid_size, max_warp_count, gmem_size)
{}

// ====================================================================================================================

template< AMGX_VecPrecision V, AMGX_MatPrecision M, AMGX_IndPrecision I >
void CSR_Multiply_Sm70<TemplateConfig<AMGX_device, V, M, I> >::count_non_zeroes( const Matrix_d &A, const Matrix_d &B, Matrix_d &C, IVector *Aq1, IVector *Bq1, IVector *Aq2, IVector *Bq2 )
{
    const int GRID_SIZE = 1024;
    const int CTA_SIZE  = 256;
    const int NUM_WARPS = CTA_SIZE / WARP_SIZE;
    // Reset work queue.
    int work_offset = GRID_SIZE * NUM_WARPS;
    CUDA_SAFE_CALL( cudaMemcpy( this->m_work_queue, &work_offset, sizeof(int), cudaMemcpyHostToDevice ) );
    

    // Compute non-zero elements.
    switch ( this->m_num_threads_per_row_count )
    {
        case 2:
            csr_multiply_sm70::count_non_zeroes_kernel< 2, CTA_SIZE, SMEM_SIZE, WARP_SIZE, true> <<< GRID_SIZE, CTA_SIZE>>>(
                A.get_num_rows(),
                A.row_offsets.raw(),
                A.col_indices.raw(),
                B.row_offsets.raw(),
                B.col_indices.raw(),
                C.row_offsets.raw(),
                NULL,
                (Aq1 != NULL) ? Aq1->raw() : NULL,
                (Bq1 != NULL) ? Bq1->raw() : NULL,
                (Aq2 != NULL) ? Aq2->raw() : NULL,
                (Bq2 != NULL) ? Bq2->raw() : NULL,
                this->m_gmem_size,
                this->m_keys,
                this->m_work_queue,
                this->m_status );
            break;

        case 4:
            csr_multiply_sm70::count_non_zeroes_kernel< 4, CTA_SIZE, SMEM_SIZE, WARP_SIZE, true> <<< GRID_SIZE, CTA_SIZE>>>(
                A.get_num_rows(),
                A.row_offsets.raw(),
                A.col_indices.raw(),
                B.row_offsets.raw(),
                B.col_indices.raw(),
                C.row_offsets.raw(),
                NULL,
                (Aq1 != NULL) ? Aq1->raw() : NULL,
                (Bq1 != NULL) ? Bq1->raw() : NULL,
                (Aq2 != NULL) ? Aq2->raw() : NULL,
                (Bq2 != NULL) ? Bq2->raw() : NULL,
                this->m_gmem_size,
                this->m_keys,
                this->m_work_queue,
                this->m_status );
            break;

        case 8:
            csr_multiply_sm70::count_non_zeroes_kernel< 8, CTA_SIZE, SMEM_SIZE, WARP_SIZE, true> <<< GRID_SIZE, CTA_SIZE>>>(
                A.get_num_rows(),
                A.row_offsets.raw(),
                A.col_indices.raw(),
                B.row_offsets.raw(),
                B.col_indices.raw(),
                C.row_offsets.raw(),
                NULL,
                (Aq1 != NULL) ? Aq1->raw() : NULL,
                (Bq1 != NULL) ? Bq1->raw() : NULL,
                (Aq2 != NULL) ? Aq2->raw() : NULL,
                (Bq2 != NULL) ? Bq2->raw() : NULL,
                this->m_gmem_size,
                this->m_keys,
                this->m_work_queue,
                this->m_status );
            break;

        case 16:
            csr_multiply_sm70::count_non_zeroes_kernel<16, CTA_SIZE, SMEM_SIZE, WARP_SIZE, true> <<< GRID_SIZE, CTA_SIZE>>>(
                A.get_num_rows(),
                A.row_offsets.raw(),
                A.col_indices.raw(),
                B.row_offsets.raw(),
                B.col_indices.raw(),
                C.row_offsets.raw(),
                NULL,
                (Aq1 != NULL) ? Aq1->raw() : NULL,
                (Bq1 != NULL) ? Bq1->raw() : NULL,
                (Aq2 != NULL) ? Aq2->raw() : NULL,
                (Bq2 != NULL) ? Bq2->raw() : NULL,
                this->m_gmem_size,
                this->m_keys,
                this->m_work_queue,
                this->m_status );
            break;

        default:
            csr_multiply_sm70::count_non_zeroes_kernel<CTA_SIZE, SMEM_SIZE, WARP_SIZE, true> <<< GRID_SIZE, CTA_SIZE>>>(
                A.get_num_rows(),
                A.row_offsets.raw(),
                A.col_indices.raw(),
                B.row_offsets.raw(),
                B.col_indices.raw(),
                C.row_offsets.raw(),
                NULL,
                (Aq1 != NULL) ? Aq1->raw() : NULL,
                (Bq1 != NULL) ? Bq1->raw() : NULL,
                (Aq2 != NULL) ? Aq2->raw() : NULL,
                (Bq2 != NULL) ? Bq2->raw() : NULL,
                this->m_gmem_size,
                this->m_keys,
                this->m_work_queue,
                this->m_status );
    }

    cudaCheckError();
    //CUDA_SAFE_CALL( cudaGetLastError() );

}


template< AMGX_VecPrecision V, AMGX_MatPrecision M, AMGX_IndPrecision I >
void CSR_Multiply_Sm70<TemplateConfig<AMGX_device, V, M, I> >::count_non_zeroes_RAP_sparse_add( Matrix_d &RAP, const Matrix_d &RAP_int, std::vector<IVector> &RAP_ext_row_offsets, std::vector<IVector> &RAP_ext_col_indices, std::vector<MVector> &RAP_ext_values, std::vector<IVector> &RAP_ext_row_ids)

{
    const int GRID_SIZE = 1024;
    const int CTA_SIZE  = 256;
    const int NUM_WARPS = CTA_SIZE / WARP_SIZE;
    // Reset work queue.
    int work_offset = GRID_SIZE * NUM_WARPS;
    CUDA_SAFE_CALL( cudaMemcpy( this->m_work_queue, &work_offset, sizeof(int), cudaMemcpyHostToDevice ) );
    // This is num_owned_coarse_rows
    int RAP_size = RAP.get_num_rows();
    int RAP_int_size = RAP_int.row_offsets.size() - 1;

    if (RAP_int_size < RAP_size)
    {
        FatalError("RAP_int has less rows than RAP, need to modify sparse RAP add to handle that case\n", AMGX_ERR_NOT_IMPLEMENTED);
    }

    // Create a device vector of the raw pointers to the array
    // RAP_ext_row_ids_ptrs
    // RAP_ext_row_offsets_ptrs
    // RAP_ext_col_indices_ptrs
    // RAP_ext_values_ptrs
    int num_neighbors = RAP_ext_row_offsets.size();
    std::vector<IVector> flagArray(num_neighbors);

    for (int i = 0; i < num_neighbors; i++)
    {
        flagArray[i].resize(RAP_size);
        thrust_wrapper::fill(flagArray[i].begin(), flagArray[i].end(), -1);
    }

    cudaCheckError();
    std::vector<int *> flagArray_ptrs_h(num_neighbors);
    std::vector<int *> RAP_ext_row_offsets_ptrs_h(num_neighbors);
    std::vector<int *> RAP_ext_col_indices_ptrs_h(num_neighbors);

    for (int i = 0; i < num_neighbors; i++)
    {
        flagArray_ptrs_h[i] = amgx::thrust::raw_pointer_cast(&flagArray[i][0]);
        RAP_ext_row_offsets_ptrs_h[i] = amgx::thrust::raw_pointer_cast(&RAP_ext_row_offsets[i][0]);
        RAP_ext_col_indices_ptrs_h[i] = amgx::thrust::raw_pointer_cast(&RAP_ext_col_indices[i][0]);
    }

    device_vector_alloc<int *> flagArray_ptrs = flagArray_ptrs_h;
    device_vector_alloc<int *> RAP_ext_row_offsets_ptrs = RAP_ext_row_offsets_ptrs_h;
    device_vector_alloc<int *> RAP_ext_col_indices_ptrs = RAP_ext_col_indices_ptrs_h;

    for (int i = 0; i < num_neighbors; i++)
    {
        int size = RAP_ext_row_ids[i].size();

        if (size != 0)
        {
            int num_blocks = min(4096, (size + 127) / 128);
            //write the position in RAP_ext_row_ids
            csr_multiply_sm70::flag_halo_rows <<< num_blocks, 128>>>(
                RAP_ext_row_ids[i].raw(),
                size,
                flagArray[i].raw(),
                i,
                RAP.manager->global_id());
        }
    }

    csr_multiply_sm70::count_non_zeroes_RAP_ext_kernel<CTA_SIZE, SMEM_SIZE, WARP_SIZE, true> <<< GRID_SIZE, CTA_SIZE>>>(
        RAP_size,
        RAP_int.row_offsets.raw(),
        RAP_int.col_indices.raw(),
        amgx::thrust::raw_pointer_cast(&RAP_ext_row_offsets_ptrs[0]),
        amgx::thrust::raw_pointer_cast(&RAP_ext_col_indices_ptrs[0]),
        RAP.row_offsets.raw(),
        (int *) NULL,
        amgx::thrust::raw_pointer_cast(&flagArray_ptrs[0]),
        this->m_gmem_size,
        this->m_keys,
        this->m_work_queue,
        this->m_status,
        num_neighbors,
        RAP.manager->global_id() );
    cudaCheckError();
    //CUDA_SAFE_CALL( cudaGetLastError() );
}

// ====================================================================================================================

template< int CTA_SIZE, bool COUNT_ONLY, typename Diag_traits, typename Matrix >
static void
count_non_zeroes_ilu1_dispatch( const Matrix &A, Matrix &B, int num_threads_per_row_count, int gmem_size, int *keys, int *work_queue, int *status )
{
    const int GRID_SIZE = 1024;

    switch ( num_threads_per_row_count )
    {
        case 2:
            csr_multiply_sm70::count_non_zeroes_ilu1_kernel< 2, CTA_SIZE, SMEM_SIZE, WARP_SIZE, COUNT_ONLY, Diag_traits> <<< GRID_SIZE, CTA_SIZE>>>(
                A.get_num_rows(),
                A.row_offsets.raw(),
                A.col_indices.raw(),
                amgx::thrust::raw_pointer_cast( &A.getMatrixColoring().getRowColors()[0] ),
                B.row_offsets.raw(),
                B.col_indices.raw(),
                gmem_size,
                keys,
                work_queue,
                status );
            break;

        case 4:
            csr_multiply_sm70::count_non_zeroes_ilu1_kernel< 4, CTA_SIZE, SMEM_SIZE, WARP_SIZE, COUNT_ONLY, Diag_traits> <<< GRID_SIZE, CTA_SIZE>>>(
                A.get_num_rows(),
                A.row_offsets.raw(),
                A.col_indices.raw(),
                amgx::thrust::raw_pointer_cast( &A.getMatrixColoring().getRowColors()[0] ),
                B.row_offsets.raw(),
                B.col_indices.raw(),
                gmem_size,
                keys,
                work_queue,
                status );
            break;

        case 8:
            csr_multiply_sm70::count_non_zeroes_ilu1_kernel< 8, CTA_SIZE, SMEM_SIZE, WARP_SIZE, COUNT_ONLY, Diag_traits> <<< GRID_SIZE, CTA_SIZE>>>(
                A.get_num_rows(),
                A.row_offsets.raw(),
                A.col_indices.raw(),
                amgx::thrust::raw_pointer_cast( &A.getMatrixColoring().getRowColors()[0] ),
                B.row_offsets.raw(),
                B.col_indices.raw(),
                gmem_size,
                keys,
                work_queue,
                status );
            break;

        case 16:
            csr_multiply_sm70::count_non_zeroes_ilu1_kernel<16, CTA_SIZE, SMEM_SIZE, WARP_SIZE, COUNT_ONLY, Diag_traits> <<< GRID_SIZE, CTA_SIZE>>>(
                A.get_num_rows(),
                A.row_offsets.raw(),
                A.col_indices.raw(),
                amgx::thrust::raw_pointer_cast( &A.getMatrixColoring().getRowColors()[0] ),
                B.row_offsets.raw(),
                B.col_indices.raw(),
                gmem_size,
                keys,
                work_queue,
                status );
            break;

        default:
            csr_multiply_sm70::count_non_zeroes_ilu1_kernel<CTA_SIZE, SMEM_SIZE, WARP_SIZE, COUNT_ONLY, Diag_traits> <<< GRID_SIZE, CTA_SIZE>>>(
                A.get_num_rows(),
                A.row_offsets.raw(),
                A.col_indices.raw(),
                amgx::thrust::raw_pointer_cast( &A.getMatrixColoring().getRowColors()[0] ),
                B.row_offsets.raw(),
                B.col_indices.raw(),
                gmem_size,
                keys,
                work_queue,
                status );
    }

    cudaCheckError();
}

// ====================================================================================================================

template< AMGX_VecPrecision V, AMGX_MatPrecision M, AMGX_IndPrecision I >
void CSR_Multiply_Sm70<TemplateConfig<AMGX_device, V, M, I> >::count_non_zeroes_ilu1( const Matrix_d &A, Matrix_d &B )
{
    const int GRID_SIZE = 1024;

    const int CTA_SIZE  = 256;
    const int NUM_WARPS = CTA_SIZE / WARP_SIZE;
    // Reset work queue.
    int work_offset = GRID_SIZE * NUM_WARPS;
    CUDA_SAFE_CALL( cudaMemcpy( this->m_work_queue, &work_offset, sizeof(int), cudaMemcpyHostToDevice ) );

    // Count the number of non zeroes.
    if ( A.hasProps(DIAG) )
        count_non_zeroes_ilu1_dispatch<CTA_SIZE, true, csr_multiply_sm70::With_external_diag, Matrix_d>(
            A,
            B,
            this->m_num_threads_per_row_count,
            this->m_gmem_size,
            this->m_keys,
            this->m_work_queue,
            this->m_status );
    else
        count_non_zeroes_ilu1_dispatch<CTA_SIZE, true, csr_multiply_sm70::Without_external_diag, Matrix_d>(
            A,
            B,
            this->m_num_threads_per_row_count,
            this->m_gmem_size,
            this->m_keys,
            this->m_work_queue,
            this->m_status );

    // Compute non-zero elements.
    CUDA_SAFE_CALL( cudaGetLastError() );
}

// ====================================================================================================================

template< AMGX_VecPrecision V, AMGX_MatPrecision M, AMGX_IndPrecision I >
void CSR_Multiply_Sm70<TemplateConfig<AMGX_device, V, M, I> >::compute_offsets( Matrix_d &C )
{
    amgx::thrust::device_ptr<int> offsets_begin(C.row_offsets.raw());
    amgx::thrust::device_ptr<int> offsets_end  (C.row_offsets.raw() + C.get_num_rows() + 1);
    thrust_wrapper::exclusive_scan( offsets_begin, offsets_end, offsets_begin );
    cudaCheckError();
}

// ====================================================================================================================

template< AMGX_VecPrecision V, AMGX_MatPrecision M, AMGX_IndPrecision I >
void CSR_Multiply_Sm70<TemplateConfig<AMGX_device, V, M, I> >::compute_sparsity( const Matrix_d &A, const Matrix_d &B, Matrix_d &C )
{
    const int GRID_SIZE = 1024;
    const int CTA_SIZE  = 256;
    const int NUM_WARPS = CTA_SIZE / WARP_SIZE;
    // Reset the work queue.
    int work_offset = GRID_SIZE * NUM_WARPS;
    CUDA_SAFE_CALL( cudaMemcpy( this->m_work_queue, &work_offset, sizeof(int), cudaMemcpyHostToDevice ) );

    // Compute the values.
    switch ( this->m_num_threads_per_row_count )
    {
        case 2:
            csr_multiply_sm70::count_non_zeroes_kernel< 2, CTA_SIZE, SMEM_SIZE, WARP_SIZE, false> <<< GRID_SIZE, CTA_SIZE>>>(
                A.get_num_rows(),
                A.row_offsets.raw(),
                A.col_indices.raw(),
                B.row_offsets.raw(),
                B.col_indices.raw(),
                C.row_offsets.raw(),
                C.col_indices.raw(),
                NULL,
                NULL,
                NULL,
                NULL,
                this->m_gmem_size,
                this->m_keys,
                this->m_work_queue,
                NULL );
            break;

        case 4:
            csr_multiply_sm70::count_non_zeroes_kernel< 4, CTA_SIZE, SMEM_SIZE, WARP_SIZE, false> <<< GRID_SIZE, CTA_SIZE>>>(
                A.get_num_rows(),
                A.row_offsets.raw(),
                A.col_indices.raw(),
                B.row_offsets.raw(),
                B.col_indices.raw(),
                C.row_offsets.raw(),
                C.col_indices.raw(),
                NULL,
                NULL,
                NULL,
                NULL,
                this->m_gmem_size,
                this->m_keys,
                this->m_work_queue,
                NULL );
            break;

        case 8:
            csr_multiply_sm70::count_non_zeroes_kernel< 8, CTA_SIZE, SMEM_SIZE, WARP_SIZE, false> <<< GRID_SIZE, CTA_SIZE>>>(
                A.get_num_rows(),
                A.row_offsets.raw(),
                A.col_indices.raw(),
                B.row_offsets.raw(),
                B.col_indices.raw(),
                C.row_offsets.raw(),
                C.col_indices.raw(),
                NULL,
                NULL,
                NULL,
                NULL,
                this->m_gmem_size,
                this->m_keys,
                this->m_work_queue,
                NULL );
            break;

        case 16:
            csr_multiply_sm70::count_non_zeroes_kernel<16, CTA_SIZE, SMEM_SIZE, WARP_SIZE, false> <<< GRID_SIZE, CTA_SIZE>>>(
                A.get_num_rows(),
                A.row_offsets.raw(),
                A.col_indices.raw(),
                B.row_offsets.raw(),
                B.col_indices.raw(),
                C.row_offsets.raw(),
                C.col_indices.raw(),
                NULL,
                NULL,
                NULL,
                NULL,
                this->m_gmem_size,
                this->m_keys,
                this->m_work_queue,
                NULL );
            break;

        default:
            csr_multiply_sm70::count_non_zeroes_kernel<CTA_SIZE, SMEM_SIZE, WARP_SIZE, false> <<< GRID_SIZE, CTA_SIZE>>>(
                A.get_num_rows(),
                A.row_offsets.raw(),
                A.col_indices.raw(),
                B.row_offsets.raw(),
                B.col_indices.raw(),
                C.row_offsets.raw(),
                C.col_indices.raw(),
                NULL,
                NULL,
                NULL,
                NULL,
                this->m_gmem_size,
                this->m_keys,
                this->m_work_queue,
                NULL );
    }

    cudaCheckError();
    //CUDA_SAFE_CALL( cudaGetLastError() );
}

// ====================================================================================================================

template< AMGX_VecPrecision V, AMGX_MatPrecision M, AMGX_IndPrecision I >
void CSR_Multiply_Sm70<TemplateConfig<AMGX_device, V, M, I> >::compute_sparsity_ilu1( const Matrix_d &A, Matrix_d &B )
{
    const int GRID_SIZE = 1024;
    const int CTA_SIZE  = 256;
    const int NUM_WARPS = CTA_SIZE / WARP_SIZE;
    // Reset work queue.
    int work_offset = GRID_SIZE * NUM_WARPS;
    CUDA_SAFE_CALL( cudaMemcpy( this->m_work_queue, &work_offset, sizeof(int), cudaMemcpyHostToDevice ) );

    // Count the number of non zeroes.
    if ( A.hasProps(DIAG) )
        count_non_zeroes_ilu1_dispatch<CTA_SIZE, false, csr_multiply_sm70::With_external_diag, Matrix_d>(
            A,
            B,
            this->m_num_threads_per_row_count,
            this->m_gmem_size,
            this->m_keys,
            this->m_work_queue,
            NULL );
    else
        count_non_zeroes_ilu1_dispatch<CTA_SIZE, false, csr_multiply_sm70::Without_external_diag, Matrix_d>(
            A,
            B,
            this->m_num_threads_per_row_count,
            this->m_gmem_size,
            this->m_keys,
            this->m_work_queue,
            NULL );

    // Make sure it worked properly.
    CUDA_SAFE_CALL( cudaGetLastError() );
}

// ====================================================================================================================

template< AMGX_VecPrecision V, AMGX_MatPrecision M, AMGX_IndPrecision I >
void CSR_Multiply_Sm70<TemplateConfig<AMGX_device, V, M, I> >::compute_values( const Matrix_d &A, const Matrix_d &B, Matrix_d &C, int num_threads, IVector *Aq1, IVector *Bq1, IVector *Aq2, IVector *Bq2  )
{
    const int GRID_SIZE = 1024;
    const int CTA_SIZE  = 128;
    const int NUM_WARPS = CTA_SIZE / WARP_SIZE;
    // Reset the work queue.
    int work_offset = GRID_SIZE * NUM_WARPS;
    CUDA_SAFE_CALL( cudaMemcpy( this->m_work_queue, &work_offset, sizeof(int), cudaMemcpyHostToDevice ) );
    // Compute the values.
    int *status = NULL;

    if ( num_threads != this->m_num_threads_per_row_compute )
    {
        status = this->m_status;
    }

    		

    switch ( num_threads )
    {
        case 2:
            csr_multiply_sm70::compute_values_kernel< 2, Value_type, CTA_SIZE, SMEM_SIZE, WARP_SIZE> <<< GRID_SIZE, CTA_SIZE>>>(
                A.get_num_rows(),
                A.row_offsets.raw(),
                A.col_indices.raw(),
                A.values.raw(),
                B.row_offsets.raw(),
                B.col_indices.raw(),
                B.values.raw(),
                C.row_offsets.raw(),
                C.col_indices.raw(),
                C.values.raw(),
                (Aq1 != NULL) ? Aq1->raw() : NULL,
                (Bq1 != NULL) ? Bq1->raw() : NULL,
                (Aq2 != NULL) ? Aq2->raw() : NULL,
                (Bq2 != NULL) ? Bq2->raw() : NULL,
                this->m_gmem_size,
                this->m_keys,
                this->m_vals,
                this->m_work_queue,
                status );
            break;

        case 4:
            csr_multiply_sm70::compute_values_kernel< 4, Value_type, CTA_SIZE, SMEM_SIZE, WARP_SIZE> <<< GRID_SIZE, CTA_SIZE>>>(
                A.get_num_rows(),
                A.row_offsets.raw(),
                A.col_indices.raw(),
                A.values.raw(),
                B.row_offsets.raw(),
                B.col_indices.raw(),
                B.values.raw(),
                C.row_offsets.raw(),
                C.col_indices.raw(),
                C.values.raw(),
                (Aq1 != NULL) ? Aq1->raw() : NULL,
                (Bq1 != NULL) ? Bq1->raw() : NULL,
                (Aq2 != NULL) ? Aq2->raw() : NULL,
                (Bq2 != NULL) ? Bq2->raw() : NULL,
                this->m_gmem_size,
                this->m_keys,
                this->m_vals,
                this->m_work_queue,
                status );
            break;

        case 8:
            csr_multiply_sm70::compute_values_kernel< 8, Value_type, CTA_SIZE, SMEM_SIZE, WARP_SIZE> <<< GRID_SIZE, CTA_SIZE>>>(
                A.get_num_rows(),
                A.row_offsets.raw(),
                A.col_indices.raw(),
                A.values.raw(),
                B.row_offsets.raw(),
                B.col_indices.raw(),
                B.values.raw(),
                C.row_offsets.raw(),
                C.col_indices.raw(),
                C.values.raw(),
                (Aq1 != NULL) ? Aq1->raw() : NULL,
                (Bq1 != NULL) ? Bq1->raw() : NULL,
                (Aq2 != NULL) ? Aq2->raw() : NULL,
                (Bq2 != NULL) ? Bq2->raw() : NULL,
                this->m_gmem_size,
                this->m_keys,
                this->m_vals,
                this->m_work_queue,
                status );
            break;

        case 16:
            csr_multiply_sm70::compute_values_kernel<16, Value_type, CTA_SIZE, SMEM_SIZE, WARP_SIZE> <<< GRID_SIZE, CTA_SIZE>>>(
                A.get_num_rows(),
                A.row_offsets.raw(),
                A.col_indices.raw(),
                A.values.raw(),
                B.row_offsets.raw(),
                B.col_indices.raw(),
                B.values.raw(),
                C.row_offsets.raw(),
                C.col_indices.raw(),
                C.values.raw(),
                (Aq1 != NULL) ? Aq1->raw() : NULL,
                (Bq1 != NULL) ? Bq1->raw() : NULL,
                (Aq2 != NULL) ? Aq2->raw() : NULL,
                (Bq2 != NULL) ? Bq2->raw() : NULL,
                this->m_gmem_size,
                this->m_keys,
                this->m_vals,
                this->m_work_queue,
                status );
            break;

        default:
            csr_multiply_sm70::compute_values_kernel<Value_type, CTA_SIZE, SMEM_SIZE, WARP_SIZE> <<< GRID_SIZE, CTA_SIZE>>>(
                A.get_num_rows(),
                A.row_offsets.raw(),
                A.col_indices.raw(),
                A.values.raw(),
                B.row_offsets.raw(),
                B.col_indices.raw(),
                B.values.raw(),
                C.row_offsets.raw(),
                C.col_indices.raw(),
                C.values.raw(),
                (Aq1 != NULL) ? Aq1->raw() : NULL,
                (Bq1 != NULL) ? Bq1->raw() : NULL,
                (Aq2 != NULL) ? Aq2->raw() : NULL,
                (Bq2 != NULL) ? Bq2->raw() : NULL,
                this->m_gmem_size,
                this->m_keys,
                this->m_vals,
                this->m_work_queue,
                status );
    }

    cudaCheckError();
    //CUDA_SAFE_CALL( cudaGetLastError() );
}


template< AMGX_VecPrecision V, AMGX_MatPrecision M, AMGX_IndPrecision I >
void CSR_Multiply_Sm70<TemplateConfig<AMGX_device, V, M, I> >::compute_values_RAP_sparse_add( Matrix_d &RAP, const Matrix_d &RAP_int, std::vector<IVector> &RAP_ext_row_offsets, std::vector<IVector> &RAP_ext_col_indices, std::vector<MVector> &RAP_ext_values, std::vector<IVector> &RAP_ext_row_ids, int num_threads)
{
    const int GRID_SIZE = 1024;
    const int CTA_SIZE  = 128;
    const int NUM_WARPS = CTA_SIZE / WARP_SIZE;
    // Reset the work queue.
    int work_offset = GRID_SIZE * NUM_WARPS;
    CUDA_SAFE_CALL( cudaMemcpy( this->m_work_queue, &work_offset, sizeof(int), cudaMemcpyHostToDevice ) );
    // Compute the values.
    int *status = NULL;

    if ( num_threads != this->m_num_threads_per_row_compute )
    {
        status = this->m_status;
    }

    // This is num_owned_coarse_rows
    int RAP_size = RAP.get_num_rows();
    int RAP_int_size = RAP_int.row_offsets.size() - 1;

    if (RAP_int_size < RAP_size)
    {
        FatalError("RAP_int has less rows than RAP, need to modify sparse RAP add to handle that case\n", AMGX_ERR_NOT_IMPLEMENTED);
    }

    //TODO: Optimize: reuse arrays from count nonzeros
    int num_neighbors = RAP_ext_row_offsets.size();
    std::vector<IVector> flagArray(num_neighbors);

    for (int i = 0; i < num_neighbors; i++)
    {
        flagArray[i].resize(RAP_size);
        thrust_wrapper::fill(flagArray[i].begin(), flagArray[i].end(), -1);
    }

    cudaCheckError();
    std::vector<int *> flagArray_ptrs_h(num_neighbors);
    std::vector<int *> RAP_ext_row_offsets_ptrs_h(num_neighbors);
    std::vector<int *> RAP_ext_col_indices_ptrs_h(num_neighbors);
    std::vector<Value_type *> RAP_ext_values_ptrs_h(num_neighbors);

    for (int i = 0; i < num_neighbors; i++)
    {
        flagArray_ptrs_h[i] = amgx::thrust::raw_pointer_cast(&flagArray[i][0]);
        RAP_ext_row_offsets_ptrs_h[i] = amgx::thrust::raw_pointer_cast(&RAP_ext_row_offsets[i][0]);
        RAP_ext_col_indices_ptrs_h[i] = amgx::thrust::raw_pointer_cast(&RAP_ext_col_indices[i][0]);
        RAP_ext_values_ptrs_h[i] = amgx::thrust::raw_pointer_cast(&RAP_ext_values[i][0]);
    }

    device_vector_alloc<int *> flagArray_ptrs = flagArray_ptrs_h;
    device_vector_alloc<int *> RAP_ext_row_offsets_ptrs = RAP_ext_row_offsets_ptrs_h;
    device_vector_alloc<int *> RAP_ext_col_indices_ptrs = RAP_ext_col_indices_ptrs_h;
    device_vector_alloc<Value_type *> RAP_ext_values_ptrs = RAP_ext_values_ptrs_h;

    for (int i = 0; i < num_neighbors; i++)
    {
        int size = RAP_ext_row_ids[i].size();

        if (size != 0)
        {
            int num_blocks = min(4096, (size + 127) / 128);
            //write the position in RAP_ext_row_ids
            csr_multiply_sm70::flag_halo_rows <<< num_blocks, 128>>>(
                RAP_ext_row_ids[i].raw(),
                size,
                flagArray[i].raw(),
                i,
                RAP.manager->global_id());
        }
    }

    cudaCheckError();
    //CUDA_SAFE_CALL( cudaGetLastError() );
    csr_multiply_sm70::compute_values_RAP_ext_kernel< Value_type, CTA_SIZE, SMEM_SIZE, WARP_SIZE> <<< GRID_SIZE, CTA_SIZE>>>(
        RAP_size,
        RAP_int.row_offsets.raw(),
        RAP_int.col_indices.raw(),
        RAP_int.values.raw(),
        amgx::thrust::raw_pointer_cast(&RAP_ext_row_offsets_ptrs[0]),
        amgx::thrust::raw_pointer_cast(&RAP_ext_col_indices_ptrs[0]),
        amgx::thrust::raw_pointer_cast(&RAP_ext_values_ptrs[0]),
        RAP.row_offsets.raw(),
        RAP.col_indices.raw(),
        RAP.values.raw(),
        amgx::thrust::raw_pointer_cast(&flagArray_ptrs[0]),
        this->m_gmem_size,
        this->m_keys,
        this->m_vals,
        this->m_work_queue,
        num_neighbors,
        status );
    cudaCheckError();
    //CUDA_SAFE_CALL( cudaGetLastError() );
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#define AMGX_CASE_LINE(CASE) template class CSR_Multiply_Sm70<TemplateMode<CASE>::Type>;
AMGX_FORALL_BUILDS(AMGX_CASE_LINE)
AMGX_FORCOMPLEX_BUILDS(AMGX_CASE_LINE)
#undef AMGX_CASE_LINE

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace amgx

