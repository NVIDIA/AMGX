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

#include <aggregation/coarseAgenerators/low_deg_coarse_A_generator.h>
#include <thrust/system/detail/generic/reduce_by_key.h>
#include <thrust/scan.h>
#include <thrust/remove.h>
#include <thrust/iterator/transform_iterator.h>
#include <thrust_wrapper.h>
#include <error.h>
#include <cutil.h>
#include <util.h>
#include <types.h>
#include <misc.h>
#include <hash_workspace.h>
#include <matrix_io.h>
#include <device_properties.h>

#include <amgx_types/util.h>

namespace amgx
{
namespace aggregation
{

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#include <sm_utils.inl>

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 700

#include <hash_containers_sm70.inl> // Included inside the namespace to solve name collisions.

static __device__ __forceinline__ int get_work( int *queue, int warp_id, int count = 1 )
{
    int offset = -1;

    if ( utils::lane_id() == 0 )
    {
        offset = atomicAdd( queue, count );
    }

    return utils::shfl( offset, 0 );
}

#else

#include <hash_containers_sm35.inl> // Included inside the namespace to solve name collisions.

static __device__ __forceinline__ int get_work( int *queue, int warp_id, int count = 1 )
{
    int offset = -1;

    if ( utils::lane_id() == 0 )
    {
        offset = atomicAdd( queue, count );
    }

    return utils::shfl( offset, 0 );
}

#endif

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template< int NUM_THREADS_PER_ROW, int CTA_SIZE, int SMEM_SIZE, int WARP_SIZE, bool HAS_DIAG, bool COUNT_ONLY >
__global__ __launch_bounds__( CTA_SIZE )
void
compute_sparsity_kernel( const int  R_num_rows, // same as num_aggregates.
                         const int *R_rows,
                         const int *R_cols,
                         const int *A_rows,
                         const int *A_cols,
                         const int *aggregates,
                         int *Ac_rows,
                         int *Ac_cols,
                         int *Ac_pos,
                         const int gmem_size,
                         int *g_keys,
                         int *wk_work_queue,
                         int *wk_status )
{
    const int NUM_WARPS       = CTA_SIZE  / WARP_SIZE;
    const int NUM_LOADED_ROWS = WARP_SIZE / NUM_THREADS_PER_ROW;
    // The hash keys stored in shared memory.
    __shared__ int s_keys[NUM_WARPS * SMEM_SIZE];
    // The coordinates of the thread inside the CTA/warp.
    const int warp_id = utils::warp_id();
    const int lane_id = utils::lane_id();
    // Constants.
    const int lane_id_div_num_threads = lane_id / NUM_THREADS_PER_ROW;
    const int lane_id_mod_num_threads = lane_id % NUM_THREADS_PER_ROW;
    // First threads load the row IDs of A needed by the CTA...
    int r_row_id = blockIdx.x * NUM_WARPS + warp_id;
    // Create local storage for the set.
    Hash_set<int, SMEM_SIZE, 4, WARP_SIZE> set( &s_keys[warp_id * SMEM_SIZE], &g_keys[r_row_id * gmem_size], gmem_size );
    // Loop over rows of R.
    for ( ; r_row_id < R_num_rows ; r_row_id = get_work( wk_work_queue, warp_id ) )
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
        // Load the range of the row.
        int r_col_it  = R_rows[r_row_id + 0];
        int r_col_end = R_rows[r_row_id + 1];

        // Iterate over the columns of R.
        for ( r_col_it += lane_id ; utils::any(r_col_it < r_col_end) ; r_col_it += WARP_SIZE )
        {
            // Is it an active thread.
            const bool is_active = r_col_it < r_col_end;
            // Columns of R map to rows of A. Each thread of the warp loads its R-col/A-row ID.
            int a_row_id = -1;

            if ( is_active )
            {
                a_row_id = R_cols[r_col_it];
            }

            const int num_rows = __popc( utils::ballot(is_active) );

            // Uniform loop: threads collaborate to load other elements.
            for ( int k = 0 ; k < num_rows ; k += NUM_LOADED_ROWS )
            {
                int local_k = k + lane_id_div_num_threads;
                // Is it an active thread.
                bool is_active_k = local_k < num_rows;
                // Threads in the warp proceeds columns of B in the range [bColIt, bColEnd).
                const int uniform_a_row_id = utils::shfl( a_row_id, local_k );
                // Load the range of the row of B.
                int a_col_it = 0, a_col_end = 0;

                if ( is_active_k )
                {
                    a_col_it  = A_rows[uniform_a_row_id + 0];
                    a_col_end = A_rows[uniform_a_row_id + 1];
                }

                // Iterate over the range of columns of B.
                for ( a_col_it += lane_id_mod_num_threads ; utils::any(a_col_it < a_col_end) ; a_col_it += NUM_THREADS_PER_ROW )
                {
                    int a_col_id = -1, a_agg_id = -1;

                    if ( a_col_it < a_col_end )
                    {
                        a_col_id = A_cols[a_col_it];
                        a_agg_id = aggregates[a_col_id];
                    }

                    if ( HAS_DIAG && a_agg_id == r_row_id )
                    {
                        a_agg_id = -1;
                    }

                    set.insert( a_agg_id, COUNT_ONLY ? wk_status : NULL );
                }
            }
        }

        // Store the results.
        if ( COUNT_ONLY )
        {
            int count = set.compute_size();

            if ( lane_id == 0 )
            {
                Ac_rows[r_row_id] = count;
            }
        }
        else
        {
            int ac_col_it = Ac_rows[r_row_id];
            set.store_with_positions( &Ac_cols[ac_col_it], &Ac_pos[ac_col_it] );
        }
    }
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template< typename Value_type, int NUM_THREADS_PER_ROW, int CTA_SIZE, int SMEM_SIZE, int WARP_SIZE, bool HAS_DIAG >
__global__
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 700
__launch_bounds__( CTA_SIZE, 8 )
#elif defined(__CUDA_ARCH__)
__launch_bounds__( CTA_SIZE, 8 )
#endif
void fill_A_kernel_1x1( const int  R_num_rows,
                        const int *R_rows,
                        const int *R_cols,
                        const int *A_rows,
                        const int *A_cols,
                        const int *A_diag,
                        const Value_type *A_vals,
                        const int *aggregates,
                        const int *Ac_rows,
                        const int *Ac_cols,
                        const int *Ac_pos,
                        const int *Ac_diag,
                        Value_type *Ac_vals,
                        int gmem_size,
                        int *g_keys,
                        Value_type *g_vals,
                        int *wk_work_queue )
{
    const int NUM_WARPS = CTA_SIZE / WARP_SIZE;
    const int NUM_LOADED_ROWS = WARP_SIZE / NUM_THREADS_PER_ROW;
    // The hash keys stored in shared memory.
    __shared__ int s_keys[NUM_WARPS * SMEM_SIZE];
    // The hash values stored in shared memory.
    __shared__ Value_type s_vals[NUM_WARPS * SMEM_SIZE]; 
    // The coordinates of the thread inside the CTA/warp.
    const int warp_id = utils::warp_id();
    const int lane_id = utils::lane_id();
    // Constants.
    const int lane_id_div_num_threads = lane_id / NUM_THREADS_PER_ROW;
    const int lane_id_mod_num_threads = lane_id % NUM_THREADS_PER_ROW;
    // First threads load the row IDs of A needed by the CTA...
    int r_row_id = blockIdx.x * NUM_WARPS + warp_id;
    // Create local storage for the set.
    Hash_map<int, Value_type, SMEM_SIZE, 4, WARP_SIZE> map( &s_keys[warp_id * SMEM_SIZE  ],
            &g_keys[r_row_id * gmem_size ],
            &s_vals[warp_id * SMEM_SIZE],
            &g_vals[r_row_id * gmem_size ], gmem_size );
    // Loop over rows of A.
    for ( ; r_row_id < R_num_rows ; r_row_id = get_work( wk_work_queue, warp_id ) )
    {
        // The indices of the output row.
        int ac_col_it  = Ac_rows[r_row_id + 0];
        int ac_col_end = Ac_rows[r_row_id + 1];
        // Clear the set first. TODO: Make sure it's needed. I don't think it is!!!!
        map.clear();
        // Populate the map.
        map.load( ac_col_end - ac_col_it, &Ac_cols[ac_col_it], &Ac_pos[ac_col_it] );
        int r_col_it  = R_rows[r_row_id + 0];
        int r_col_end = R_rows[r_row_id + 1];
        // The diagonal.
        Value_type r_diag(types::util<Value_type>::get_zero());

        // _iterate over the columns of A to build C_hat.
        for ( r_col_it += lane_id ; utils::any(r_col_it < r_col_end) ; r_col_it += WARP_SIZE )
        {
            // Is it an active thread.
            const bool is_active = r_col_it < r_col_end;
            // Columns of A maps to rows of B. Each thread of the warp loads its A-col/B-row ID.
            int a_row_id = -1;

            if ( is_active )
            {
                a_row_id = R_cols[r_col_it];
            }

            // Update the diagonal (if needed).
            if ( HAS_DIAG && is_active )
            {
                r_diag = r_diag + A_vals[A_diag[a_row_id]];
            }

            const int num_rows = __popc( utils::ballot(is_active) );

            // Uniform loop: threads collaborate to load other elements.
            for ( int k = 0 ; k < num_rows ; k += NUM_LOADED_ROWS )
            {
                int local_k = k + lane_id_div_num_threads;
                // Threads in the warp proceeds columns of B in the range [bColIt, bColEnd).
                const int uniform_a_row_id = utils::shfl( a_row_id, local_k );
                // The range of the row of B.
                int a_col_it = 0, a_col_end = 0;

                if ( local_k < num_rows )
                {
                    a_col_it  = utils::Ld<utils::LD_CG>::load( &A_rows[uniform_a_row_id + 0] );
                    a_col_end = utils::Ld<utils::LD_CG>::load( &A_rows[uniform_a_row_id + 1] );
                }

                // Iterate over the range of columns of B.
                for ( a_col_it += lane_id_mod_num_threads ; utils::any(a_col_it < a_col_end) ; a_col_it += NUM_THREADS_PER_ROW )
                {
                    // Load columns and values.
                    int a_col_id = -1;
                    Value_type a_value(types::util<Value_type>::get_zero());

                    if ( a_col_it < a_col_end )
                    {
                        a_col_id = A_cols[a_col_it];
                        a_value  = A_vals[a_col_it];
                    }

                    // Find the aggregate.
                    int a_agg_id = -1;

                    if ( a_col_it < a_col_end )
                    {
                        a_agg_id = aggregates[a_col_id];
                    }

                    // Update the diag/hash map.
                    if ( HAS_DIAG && a_agg_id == r_row_id )
                    {
                        r_diag = r_diag + a_value;
                        a_agg_id = -1;
                    }

                    map.insert( a_agg_id, a_value, NULL );  // It won't insert. Only update.
                }
            }
        }

        // Update the diagonal.
        if ( HAS_DIAG )
        {
            r_diag = utils::warp_reduce<1, utils::Add>( r_diag );

            if ( lane_id == 0 )
            {
                Ac_vals[Ac_diag[r_row_id]] = r_diag;
            }
        }

        // Store the results.
        int count = ac_col_end - ac_col_it;

        if ( count == 0 )
        {
            continue;
        }

        map.store( count, &Ac_vals[ac_col_it] );
    }
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template< typename Value_type, int CTA_SIZE, int SMEM_SIZE, int WARP_SIZE, bool HAS_DIAG >
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 700
__global__ __launch_bounds__( CTA_SIZE, 8 )
#else 
__global__ __launch_bounds__( CTA_SIZE, 8 )
#endif
void fill_A_kernel_4x4( const int  R_num_rows, // same as num_aggregates.
                        const int *R_rows,
                        const int *R_cols,
                        const int *A_rows,
                        const int *A_cols,
                        const int *A_diag,
                        const Value_type *A_vals,
                        const int *aggregates,
                        const int *Ac_rows,
                        const int *Ac_cols,
                        const int *Ac_pos,
                        const int *Ac_diag,
                        Value_type *Ac_vals,
                        const int gmem_size,
                        int *g_keys,
                        int *g_idx,
                        int *wk_work_queue )
{
    const int NUM_WARPS = CTA_SIZE / WARP_SIZE;
    // The hash keys stored in shared memory.
    __shared__ int s_keys[NUM_WARPS * SMEM_SIZE];
    // The coordinates of the thread inside the CTA/warp.
    const int warp_id = utils::warp_id( );
    const int lane_id = utils::lane_id( );
    // Constants.
    const int lane_id_div_16 = lane_id / 16;
    const int lane_id_mod_16 = lane_id % 16;
    const int warp_offset = 16 * lane_id_div_16;
    // First threads load the row IDs of A needed by the CTA...
    int r_row_id = blockIdx.x * NUM_WARPS + warp_id;
    // My index.
    Hash_index<int, SMEM_SIZE, WARP_SIZE> index( &g_idx[r_row_id * gmem_size] );
    // Create local storage for the set.
    Hash_set<int, SMEM_SIZE, 4, WARP_SIZE> set( &s_keys[warp_id * SMEM_SIZE], &g_keys[r_row_id * gmem_size], gmem_size );
    // Loop over rows of R.
    for ( ; r_row_id < R_num_rows ; r_row_id = get_work( wk_work_queue, warp_id ) )
    {
        // The indices of the row.
        int ac_col_it  = Ac_rows[r_row_id + 0];
        int ac_col_end = Ac_rows[r_row_id + 1];
        // Clear the set first.
        set.clear(true);
        // Populate the index.
        set.load_index( ac_col_end - ac_col_it, &Ac_cols[ac_col_it], &Ac_pos[ac_col_it], index, false );
        // Load the range of the row.
        int r_col_it  = R_rows[r_row_id + 0];
        int r_col_end = R_rows[r_row_id + 1];
        // Diagonal value (each half warp stores a diagonal element).
        Value_type ac_diag(types::util<Value_type>::get_zero());

        // Iterate over the columns of R.
        for ( r_col_it += lane_id_div_16 ; utils::any(r_col_it < r_col_end) ; r_col_it += 2 )
        {
            // Is it an active thread.
            const bool is_active = r_col_it < r_col_end;
            // Columns of R map to rows of A. Each thread of the warp loads its R-col/A-row ID.
            int a_row_id = -1;

            if ( is_active )
            {
                a_row_id = R_cols[r_col_it];
            }

            // Update the diagonal if needed.
            if ( HAS_DIAG && is_active )
            {
                ac_diag = ac_diag + A_vals[16 * A_diag[a_row_id] + lane_id_mod_16];
            }

            // Load the range of the row of A.
            int a_col_begin = 0, a_col_end = 0;

            if ( is_active )
            {
                a_col_begin = A_rows[a_row_id + 0];
                a_col_end   = A_rows[a_row_id + 1];
            }

            // Iterate over the range of columns of B.
            for ( ; utils::any(a_col_begin < a_col_end) ; a_col_begin += 16 )
            {
                int a_col_it = a_col_begin + lane_id_mod_16;
                // Each thread loads a column-ID and an aggregate.
                int a_col_id = -1, ac_col_id = -1;

                if ( a_col_it < a_col_end )
                {
                    a_col_id  = A_cols    [a_col_it];
                    ac_col_id = aggregates[a_col_id];
                }

                // Each thread uses the hashed index to find the position associated with the aggregate.
                int key = ac_col_id;

                if ( HAS_DIAG && ac_col_id == r_row_id )
                {
                    key = -1;
                }

                int ac_idx = ac_col_it + set.find_index( key, index, false );

                // Iterate over the 16 items.
                for ( int k = 0 ; k < 16 ; ++k )
                {
                    int uniform_ac_col = utils::shfl( ac_col_id, warp_offset + k );
                    int uniform_ac_idx = utils::shfl( ac_idx,    warp_offset + k );

                    // Early loop exit.
                    if ( utils::all( uniform_ac_col == -1 ) )
                    {
                        break;
                    }

                    // The index of the item.
                    const int uniform_a_col_it = a_col_begin + k;
                    // Load the value if possible.
                    Value_type a_value(types::util<Value_type>::get_zero());

                    if ( uniform_a_col_it < a_col_end )
                    {
                        a_value = A_vals[16 * uniform_a_col_it + lane_id_mod_16];
                    }

                    // Proceed diagonal if needed.
                    if ( HAS_DIAG && uniform_ac_col == r_row_id )
                    {
                        ac_diag = ac_diag + a_value;
                        uniform_ac_col = -1;
                    }

                    // Get the id of the column computed by the other half warp.
                    int other_ac_col = utils::shfl_xor( uniform_ac_col, 16 );
                    // If both half warps want to write to the same location, we have a conflict!!!
                    int are_fighting = uniform_ac_col == other_ac_col;

                    // Reduce the two values to a single one.
                    if ( uniform_ac_col != -1 && are_fighting )
                    {
                        a_value = a_value + utils::shfl_xor( a_value, 16 );
                    }

                    // If the two half warps fight, only one can be the winner... It's the 1st half!!!
                    int is_winner = !are_fighting || lane_id_div_16 == 0;

                    // Update the value.
                    if ( uniform_ac_col != -1 && is_winner )
                    {
                        Ac_vals[16 * uniform_ac_idx + lane_id_mod_16] = Ac_vals[16 * uniform_ac_idx + lane_id_mod_16] + a_value;
                    }
                }
            }
        }

        if ( HAS_DIAG )
        {
            ac_diag = ac_diag + utils::shfl_xor( ac_diag, 16 );

            if ( lane_id_div_16 == 0 )
            {
                Ac_vals[16 * Ac_diag[r_row_id] + lane_id_mod_16] = ac_diag;
            }
        }
    }
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template< typename Value_type, int N, int CTA_SIZE, int SMEM_SIZE, int WARP_SIZE, bool HAS_DIAG, bool FORCE_DETERMINISM >
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 700
__global__ __launch_bounds__( CTA_SIZE, 8 )
#else
__global__ __launch_bounds__( CTA_SIZE, 8 )
#endif
void fill_A_kernel_NxN( const int  R_num_rows, // same as num_aggregates.
                        const int *R_rows,
                        const int *R_cols,
                        const int *A_rows,
                        const int *A_cols,
                        const int *A_diag,
                        const Value_type *A_vals,
                        const int *aggregates,
                        const int *Ac_rows,
                        const int *Ac_cols,
                        const int *Ac_pos,
                        const int *Ac_diag,
                        Value_type *Ac_vals,
                        const int gmem_size,
                        int *g_keys,
                        int *g_idx,
                        int *wk_work_queue )
{
    const int NUM_WARPS = CTA_SIZE / WARP_SIZE;
    // Squared N.
    const int NxN = N * N;
    // Number of items per warp.
    const int T_WARP = FORCE_DETERMINISM ? 1 : WARP_SIZE / NxN;
    const int NUM_ITEMS_PER_WARP = T_WARP == 0 ? 1 : T_WARP;
    // The hash keys stored in shared memory.
    __shared__ int s_keys[NUM_WARPS * SMEM_SIZE];
    // The coordinates of the thread inside the CTA/warp.
    const int warp_id = utils::warp_id( );
    const int lane_id = utils::lane_id( );
    // Constants.
    const int lane_id_div_NxN = lane_id / NxN;
    const int lane_id_mod_NxN = lane_id % NxN;
    const int warp_offset = NxN * lane_id_div_NxN;
    // First threads load the row IDs of A needed by the CTA...
    int r_row_id = blockIdx.x * NUM_WARPS + warp_id;
    // My index.
    Hash_index<int, SMEM_SIZE, WARP_SIZE> index( &g_idx[r_row_id * gmem_size] );
    // Create local storage for the set.
    Hash_set<int, SMEM_SIZE, 4, WARP_SIZE> set( &s_keys[warp_id * SMEM_SIZE], &g_keys[r_row_id * gmem_size], gmem_size );
    // Loop over rows of R.

    for ( ; r_row_id < R_num_rows ; r_row_id = get_work( wk_work_queue, warp_id ) )
    {
        // The indices of the row.
        int ac_col_it  = Ac_rows[r_row_id + 0];
        int ac_col_end = Ac_rows[r_row_id + 1];
        // Clear the set first.
        set.clear(true);
        // Populate the index.
        set.load_index( ac_col_end - ac_col_it, &Ac_cols[ac_col_it], &Ac_pos[ac_col_it], index, false );
        // Load the range of the row.
        int r_col_it  = R_rows[r_row_id + 0];
        int r_col_end = R_rows[r_row_id + 1];
        // Diagonal value (each half warp stores a diagonal element).
        Value_type ac_diag(types::util<Value_type>::get_zero());

        // Iterate over the columns of R.
        for ( r_col_it += lane_id_div_NxN ; utils::any(r_col_it < r_col_end) ; r_col_it += NUM_ITEMS_PER_WARP )
        {
            // Is it an active thread.
            const bool is_active = r_col_it < r_col_end && lane_id_div_NxN < NUM_ITEMS_PER_WARP;
            // Columns of R map to rows of A. Each thread of the warp loads its R-col/A-row ID.
            int a_row_id = -1;

            if ( is_active )
            {
                a_row_id = R_cols[r_col_it];
            }

            // Update the diagonal if needed.
            if ( HAS_DIAG && is_active )
            {
                ac_diag = ac_diag + A_vals[NxN * A_diag[a_row_id] + lane_id_mod_NxN];
            }

            // Load the range of the row of A.
            int a_col_begin = 0, a_col_end = 0;

            if ( is_active )
            {
                a_col_begin = A_rows[a_row_id + 0];
                a_col_end   = A_rows[a_row_id + 1];
            }

            // Iterate over the range of columns of B.
            for ( ; utils::any(a_col_begin < a_col_end) ; a_col_begin += NxN )
            {
                int a_col_it = a_col_begin + lane_id_mod_NxN;
                // Is it active.
                const bool is_active_k = a_col_it < a_col_end && lane_id_div_NxN < NUM_ITEMS_PER_WARP;
                // Each thread loads a column-ID and an aggregate.
                int a_col_id = -1, ac_col_id = -1;

                if ( is_active_k )
                {
                    a_col_id  = A_cols    [a_col_it];
                    ac_col_id = aggregates[a_col_id];
                }

                // Each thread uses the hashed index to find the position associated with the aggregate.
                int key = ac_col_id;

                if ( HAS_DIAG && ac_col_id == r_row_id )
                {
                    key = -1;
                }

                int ac_idx = ac_col_it + set.find_index( key, index, false );

                // Iterate over the NxN items.
                for ( int k = 0 ; k < NxN ; ++k )
                {
                    int uniform_ac_col = utils::shfl( ac_col_id, warp_offset + k );
                    int uniform_ac_idx = utils::shfl( ac_idx,    warp_offset + k );

                    if ( lane_id_div_NxN >= NUM_ITEMS_PER_WARP )
                    {
                        uniform_ac_col = -1;
                        uniform_ac_idx = -1;
                    }

                    // Early loop exit.
                    if ( utils::all( uniform_ac_col == -1 ) )
                    {
                        break;
                    }

                    // The index of the item.
                    const int uniform_a_col_it = a_col_begin + k;
                    // Load the value if possible.
                    Value_type a_value(types::util<Value_type>::get_zero());

                    if ( uniform_a_col_it < a_col_end && lane_id_div_NxN < NUM_ITEMS_PER_WARP )
                    {
                        a_value = A_vals[NxN * uniform_a_col_it + lane_id_mod_NxN];
                    }

                    // Update the diagonal if it is a diagonal term.
                    if ( HAS_DIAG && uniform_ac_col == r_row_id )
                    {
                        ac_diag = ac_diag + a_value;
                        uniform_ac_col = -1;
                    }

                    // Update the value.
                    if ( uniform_ac_col != -1 )
                    {
                        utils::atomic_add( &Ac_vals[NxN * uniform_ac_idx + lane_id_mod_NxN], a_value );
                    }
                }
            }
        }

        if ( HAS_DIAG )
        {
            if ( !FORCE_DETERMINISM )
            {
                ac_diag = utils::warp_reduce<NxN, utils::Add>( ac_diag );
            }

            if ( lane_id_div_NxN == 0 )
            {
                Ac_vals[NxN * Ac_diag[r_row_id] + lane_id_mod_NxN] = ac_diag;
            }
        }
    }
}

// when blocksize is larger than warp size
template< typename Value_type, int N, int CTA_SIZE, int SMEM_SIZE, int WARP_SIZE, bool HAS_DIAG, bool FORCE_DETERMINISM, int NUM_BLOCK_ITERS_PER_WARP>
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 700
__global__ __launch_bounds__( CTA_SIZE, 8 )
#else
__global__ __launch_bounds__( CTA_SIZE, 8 )
#endif
void fill_A_kernel_NxN_large( const int  R_num_rows, // same as num_aggregates.
                              const int *R_rows,
                              const int *R_cols,
                              const int *A_rows,
                              const int *A_cols,
                              const int *A_diag,
                              const Value_type *A_vals,
                              const int *aggregates,
                              const int *Ac_rows,
                              const int *Ac_cols,
                              const int *Ac_pos,
                              const int *Ac_diag,
                              Value_type *Ac_vals,
                              const int gmem_size,
                              int *g_keys,
                              int *g_idx,
                              int *wk_work_queue )
{
    const int NUM_WARPS = CTA_SIZE / WARP_SIZE;
    // Squared N.
    const int NxN = N * N;
    // Number of items per warp. Let's be chill here and take 1 per warp for large blocks
    const int NUM_ITEMS_PER_WARP = 1;
    // The hash keys stored in shared memory.
    __shared__ int s_keys[NUM_WARPS * SMEM_SIZE];
    // The coordinates of the thread inside the CTA/warp.
    const int warp_id = utils::warp_id( );
    const int lane_id = utils::lane_id( );
    // First threads load the row IDs of A needed by the CTA...
    int r_row_id = blockIdx.x * NUM_WARPS + warp_id;
    // My index.
    Hash_index<int, SMEM_SIZE, WARP_SIZE> index( &g_idx[r_row_id * gmem_size] );
    // Create local storage for the set.
    Hash_set<int, SMEM_SIZE, 4, WARP_SIZE> set( &s_keys[warp_id * SMEM_SIZE], &g_keys[r_row_id * gmem_size], gmem_size );
    // Loop over rows of R.
    for ( ; r_row_id < R_num_rows ; r_row_id = get_work( wk_work_queue, warp_id ) )
    {
        // The indices of the row.
        int ac_col_it  = Ac_rows[r_row_id + 0];
        int ac_col_end = Ac_rows[r_row_id + 1];
        // Clear the set first.
        set.clear(true);
        // Populate the index.
        set.load_index( ac_col_end - ac_col_it, &Ac_cols[ac_col_it], &Ac_pos[ac_col_it], index, false );
        // Load the range of the row.
        int r_col_it  = R_rows[r_row_id + 0];
        int r_col_end = R_rows[r_row_id + 1];
        // Diagonal value (each half warp stores a diagonal element).
        Value_type ac_diag(types::util<Value_type>::get_zero());

        // Iterate over the columns of R.
        for ( ; utils::any(r_col_it < r_col_end) ; r_col_it += NUM_ITEMS_PER_WARP )
        {
            // Columns of R map to rows of A. Each thread of the warp loads its R-col/A-row ID.
            int a_row_id = R_cols[r_col_it];

            // Update the diagonal if needed.
            if ( HAS_DIAG )
            {
                ac_diag = ac_diag + A_vals[NxN * A_diag[a_row_id] + lane_id];
            }

            // Load the range of the row of A.
            int a_col_begin = A_rows[a_row_id + 0];
            int a_col_end   = A_rows[a_row_id + 1];

            // Iterate over the range of columns of B.
            for ( ; utils::any(a_col_begin < a_col_end) ; a_col_begin += NxN )
            {
                int a_col_it = a_col_begin + lane_id;
                // Is it active.
                const bool is_active_k = a_col_it < a_col_end;
                // Each thread loads a column-ID and an aggregate.
                int a_col_id = -1, ac_col_id = -1;

                if ( is_active_k )
                {
                    a_col_id  = A_cols    [a_col_it];
                    ac_col_id = aggregates[a_col_id];
                }

                // Each thread uses the hashed index to find the position associated with the aggregate.
                int key = ac_col_id;

                if ( HAS_DIAG && ac_col_id == r_row_id )
                {
                    key = -1;
                }

                int ac_idx = ac_col_it + set.find_index( key, index, false );

                // Iterate over the NxN items.
                for ( int k = 0 ; k < NxN ; ++k )
                {
                    int uniform_ac_col = utils::shfl( ac_col_id, k );
                    int uniform_ac_idx = utils::shfl( ac_idx,    k );

                    // Early loop exit.
                    if ( utils::all( uniform_ac_col == -1 ) )
                    {
                        break;
                    }

                    // The index of the item.
                    const int uniform_a_col_it = a_col_begin + k;
                    // iterate through the block
#pragma unroll

                    for (int i = 0; i < NUM_BLOCK_ITERS_PER_WARP; i++)
                    {
                        // Load the value if possible.
                        Value_type a_value(types::util<Value_type>::get_zero());

                        if ( uniform_a_col_it < a_col_end && (WARP_SIZE * i + lane_id) < NxN )
                        {
                            a_value = A_vals[NxN * uniform_a_col_it + WARP_SIZE * i + lane_id];
                        }

                        // Update the diagonal if it is a diagonal term.
                        if ( HAS_DIAG && uniform_ac_col == r_row_id )
                        {
                            ac_diag = ac_diag + a_value;
                            uniform_ac_col = -1;
                        }

                        // Update the value.
                        if ( uniform_ac_col != -1 && (WARP_SIZE * i + lane_id) < NxN)
                        {
                            utils::atomic_add( &Ac_vals[NxN * uniform_ac_idx + WARP_SIZE * i + lane_id], a_value );
                        }
                    }
                }
            }
        }

        if ( HAS_DIAG )
        {
            if ( !FORCE_DETERMINISM )
            {
                ac_diag = utils::warp_reduce<NxN, utils::Add>( ac_diag );
            }

            Ac_vals[NxN * Ac_diag[r_row_id] + lane_id] = ac_diag;
        }
    }
}


///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

enum { WARP_SIZE = 32, SMEM_SIZE = 128 };

template< int CTA_SIZE, bool HAS_DIAG, bool COUNT_ONLY, typename Workspace >
static
void compute_sparsity_dispatch( Workspace &hash_wk,
                                const int  R_num_rows,
                                const int *R_rows,
                                const int *R_cols,
                                const int *A_rows,
                                const int *A_cols,
                                const int *aggregates,
                                int *Ac_rows,
                                int *Ac_cols,
                                int *Ac_pos )
{
    cudaDeviceProp props = getDeviceProperties();
    int GRID_SIZE = (props.major >= 7) ? 1024 : 128;

    const int NUM_WARPS = CTA_SIZE / WARP_SIZE;
    int *h_status;
    amgx::memory::cudaMallocHost((void **) &h_status, sizeof(int));
    int *h_work_offset;
    amgx::memory::cudaMallocHost((void **) &h_work_offset, sizeof(int));
    int attempt = 0;
    bool warning_printed = 0;

    for ( bool done = false ; !done && attempt < 10 ; ++attempt )
    {
        // Double the amount of GMEM (if needed).
        if ( attempt > 0 )
        {
            if (!warning_printed)
            {
                amgx_printf("WARNING: Used settings might result in degraded performance for the MG coarsener for this matrix.\n");
                amgx_printf("WARNING: You might want to try different selector or MG algorithm for better performance.\n");
                warning_printed = 1;
            }

            hash_wk.expand();
        }

        // Reset the status.
        int *p_status = h_status;
        *p_status = 0;
        cudaMemcpyAsync( hash_wk.get_status(), p_status, sizeof(int), cudaMemcpyHostToDevice, amgx::thrust::global_thread_handle::get_stream() );
        cudaCheckError();
        // Reset the work queue.
        int *p_work_offset = h_work_offset;
        *p_work_offset = GRID_SIZE * NUM_WARPS;
        cudaMemcpyAsync( hash_wk.get_work_queue(), p_work_offset, sizeof(int), cudaMemcpyHostToDevice, amgx::thrust::global_thread_handle::get_stream() );
        cudaCheckError();
        // Launch the kernel.
        compute_sparsity_kernel<8, CTA_SIZE, SMEM_SIZE, WARP_SIZE, HAS_DIAG, COUNT_ONLY> <<< GRID_SIZE, CTA_SIZE, 0, amgx::thrust::global_thread_handle::get_stream()>>>(
            R_num_rows,
            R_rows,
            R_cols,
            A_rows,
            A_cols,
            aggregates,
            Ac_rows,
            Ac_cols,
            Ac_pos,
            hash_wk.get_gmem_size(),
            hash_wk.get_keys(),
            hash_wk.get_work_queue(),
            hash_wk.get_status() );
        cudaCheckError();
        // Read the result from count_non_zeroes.
        cudaMemcpyAsync( p_status, hash_wk.get_status(), sizeof(int), cudaMemcpyDeviceToHost, amgx::thrust::global_thread_handle::get_stream() );
        cudaStreamSynchronize(amgx::thrust::global_thread_handle::get_stream());
        done = (*p_status == 0);
        cudaCheckError();
    }

    amgx::memory::cudaFreeHost(h_status);
    amgx::memory::cudaFreeHost(h_work_offset);
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template< int CTA_SIZE, bool HAS_DIAG, typename Workspace, typename Value_type >
static
void fill_A_dispatch( Workspace &hash_wk,
                      const int  block_size,
                      const int  R_num_rows, // same as num_aggregates.
                      const int *R_rows,
                      const int *R_cols,
                      const int *A_rows,
                      const int *A_cols,
                      const int *A_diag,
                      const Value_type *A_vals,
                      const int *aggregates,
                      const int *Ac_rows,
                      const int *Ac_cols,
                      const int *Ac_pos,
                      const int *Ac_diag,
                      Value_type *Ac_vals,
                      bool force_determinism )
{
    cudaDeviceProp props = getDeviceProperties();
    int GRID_SIZE = (props.major >= 7) ? 1024 : 128;

    const int NUM_WARPS = CTA_SIZE / WARP_SIZE;
    int work_offset = GRID_SIZE * NUM_WARPS;
    cudaMemcpyAsync( hash_wk.get_work_queue(), &work_offset, sizeof(int), cudaMemcpyHostToDevice, amgx::thrust::global_thread_handle::get_stream() );
    cudaCheckError();

    // Launch the kernel.
    switch ( block_size )
    {
        case 1:
            fill_A_kernel_1x1<Value_type, 8, CTA_SIZE, SMEM_SIZE, 32, HAS_DIAG> <<< GRID_SIZE, CTA_SIZE>>>(
                R_num_rows,
                R_rows,
                R_cols,
                A_rows,
                A_cols,
                A_diag,
                A_vals,
                aggregates,
                Ac_rows,
                Ac_cols,
                Ac_pos,
                Ac_diag,
                Ac_vals,
                hash_wk.get_gmem_size(),
                hash_wk.get_keys(),
                hash_wk.get_vals(),
                hash_wk.get_work_queue() );
            break;

        case 2:
            fill_A_kernel_NxN<Value_type, 2, CTA_SIZE, SMEM_SIZE, 32, HAS_DIAG, false> <<< GRID_SIZE, CTA_SIZE>>>(
                R_num_rows,
                R_rows,
                R_cols,
                A_rows,
                A_cols,
                A_diag,
                A_vals,
                aggregates,
                Ac_rows,
                Ac_cols,
                Ac_pos,
                Ac_diag,
                Ac_vals,
                hash_wk.get_gmem_size(),
                hash_wk.get_keys(),
                reinterpret_cast<int *>( hash_wk.get_vals() ),
                hash_wk.get_work_queue() );
            break;

        case 3:
            fill_A_kernel_NxN<Value_type, 3, CTA_SIZE, SMEM_SIZE, 32, HAS_DIAG, false> <<< GRID_SIZE, CTA_SIZE>>>(
                R_num_rows,
                R_rows,
                R_cols,
                A_rows,
                A_cols,
                A_diag,
                A_vals,
                aggregates,
                Ac_rows,
                Ac_cols,
                Ac_pos,
                Ac_diag,
                Ac_vals,
                hash_wk.get_gmem_size(),
                hash_wk.get_keys(),
                reinterpret_cast<int *>( hash_wk.get_vals() ),
                hash_wk.get_work_queue() );
            break;

        case 4:
            if ( force_determinism )
                fill_A_kernel_NxN<Value_type, 4, CTA_SIZE, SMEM_SIZE, 32, HAS_DIAG, true> <<< GRID_SIZE, CTA_SIZE>>>(
                    R_num_rows,
                    R_rows,
                    R_cols,
                    A_rows,
                    A_cols,
                    A_diag,
                    A_vals,
                    aggregates,
                    Ac_rows,
                    Ac_cols,
                    Ac_pos,
                    Ac_diag,
                    Ac_vals,
                    hash_wk.get_gmem_size(),
                    hash_wk.get_keys(),
                    reinterpret_cast<int *>( hash_wk.get_vals() ),
                    hash_wk.get_work_queue() );
            else
                fill_A_kernel_4x4<Value_type, CTA_SIZE, SMEM_SIZE, 32, HAS_DIAG> <<< GRID_SIZE, CTA_SIZE, 0, amgx::thrust::global_thread_handle::get_stream()>>>(
                    R_num_rows,
                    R_rows,
                    R_cols,
                    A_rows,
                    A_cols,
                    A_diag,
                    A_vals,
                    aggregates,
                    Ac_rows,
                    Ac_cols,
                    Ac_pos,
                    Ac_diag,
                    Ac_vals,
                    hash_wk.get_gmem_size(),
                    hash_wk.get_keys(),
                    reinterpret_cast<int *>( hash_wk.get_vals() ),
                    hash_wk.get_work_queue() );

            break;

        case 5:
            fill_A_kernel_NxN<Value_type, 5, CTA_SIZE, SMEM_SIZE, 32, HAS_DIAG, false> <<< GRID_SIZE, CTA_SIZE>>>(
                R_num_rows,
                R_rows,
                R_cols,
                A_rows,
                A_cols,
                A_diag,
                A_vals,
                aggregates,
                Ac_rows,
                Ac_cols,
                Ac_pos,
                Ac_diag,
                Ac_vals,
                hash_wk.get_gmem_size(),
                hash_wk.get_keys(),
                reinterpret_cast<int *>( hash_wk.get_vals() ),
                hash_wk.get_work_queue() );
            break;

        case 8:
            fill_A_kernel_NxN_large<Value_type, 8, CTA_SIZE, SMEM_SIZE, 32, HAS_DIAG, false, 2> <<< GRID_SIZE, CTA_SIZE>>>(
                R_num_rows,
                R_rows,
                R_cols,
                A_rows,
                A_cols,
                A_diag,
                A_vals,
                aggregates,
                Ac_rows,
                Ac_cols,
                Ac_pos,
                Ac_diag,
                Ac_vals,
                hash_wk.get_gmem_size(),
                hash_wk.get_keys(),
                reinterpret_cast<int *>( hash_wk.get_vals() ),
                hash_wk.get_work_queue() );
            break;

        case 10:
            fill_A_kernel_NxN_large<Value_type, 10, CTA_SIZE, SMEM_SIZE, 32, HAS_DIAG, false, 4> <<< GRID_SIZE, CTA_SIZE>>>(
                R_num_rows,
                R_rows,
                R_cols,
                A_rows,
                A_cols,
                A_diag,
                A_vals,
                aggregates,
                Ac_rows,
                Ac_cols,
                Ac_pos,
                Ac_diag,
                Ac_vals,
                hash_wk.get_gmem_size(),
                hash_wk.get_keys(),
                reinterpret_cast<int *>( hash_wk.get_vals() ),
                hash_wk.get_work_queue() );
            break;

        default:
            FatalError( "LOW_DEG not implemented for this block size", AMGX_ERR_NOT_SUPPORTED_BLOCKSIZE );
    }

    cudaCheckError();
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template< AMGX_VecPrecision V, AMGX_MatPrecision M, AMGX_IndPrecision I >
void
LowDegCoarseAGenerator<TemplateConfig<AMGX_device, V, M, I> >::computeAOperator( const Matrix_d &A,
        Matrix_d &Ac,
        const IVector &aggregates,
        const IVector &R_row_offsets,
        const IVector &R_column_indices,
        const int num_aggregates )
{
    if ( A.get_block_dimx() != A.get_block_dimy() )
    {
        FatalError( "LowDegCoarseAGenerator implemented for squared blocks", AMGX_ERR_NOT_SUPPORTED_BLOCKSIZE );
    }

    // The matrix Ac will be modified.
    Ac.set_initialized(0);
    // Is the diagonal stored separately??
    const int diag_prop = A.hasProps(DIAG);
    // Allocate a workspace for hashing.
    typedef TemplateConfig<AMGX_device, V, M, I> TConfig_d;

    cudaDeviceProp props = getDeviceProperties();
    int grid_size = (props.major >= 7) ? 1024 : 128;

    Hash_Workspace<TConfig_d, int> hash_wk(true, grid_size);

    // Compute row offsets of Ac.
    Ac.addProps(CSR);
    Ac.set_num_rows( num_aggregates );
    Ac.set_num_cols( num_aggregates );
    Ac.row_offsets.resize( num_aggregates + 1 );
    // Compute the number of non-zero elements per row of Ac.
    const int CTA_SIZE  = 128;

    if ( diag_prop )
        compute_sparsity_dispatch<CTA_SIZE, true, true>(
            hash_wk,
            num_aggregates,
            R_row_offsets.raw(),
            R_column_indices.raw(),
            A.row_offsets.raw(),
            A.col_indices.raw(),
            aggregates.raw(),
            Ac.row_offsets.raw(),
            NULL,
            NULL );
    else
        compute_sparsity_dispatch<CTA_SIZE, false, true>(
            hash_wk,
            num_aggregates,
            R_row_offsets.raw(),
            R_column_indices.raw(),
            A.row_offsets.raw(),
            A.col_indices.raw(),
            aggregates.raw(),
            Ac.row_offsets.raw(),
            NULL,
            NULL );

    cudaCheckError();
    // Compute the number of non-zeroes.
    thrust_wrapper::exclusive_scan<AMGX_device>( Ac.row_offsets.begin(), Ac.row_offsets.end(), Ac.row_offsets.begin() );
    cudaCheckError();
    int nonzero_blocks = Ac.row_offsets[num_aggregates];

    if ( diag_prop )
    {
        Ac.addProps(DIAG);
    }

    if ( A.is_matrix_singleGPU() )
    {
        Ac.resize( num_aggregates, num_aggregates, nonzero_blocks, A.get_block_dimy(), A.get_block_dimx(), !diag_prop );
    }
    else
    {
        //have 3% more nz for storage
        Ac.resize_spare( num_aggregates, num_aggregates, nonzero_blocks, A.get_block_dimy(), A.get_block_dimx(), 1.0 );

        if ( diag_prop )
        {
            Ac.computeDiagonal();
        }
    }

    // Vector to store the positions in the hash table.
    device_vector_alloc<int> Ac_pos(nonzero_blocks);

    // Compute the sparsity pattern of the rows of Ac.
    if ( diag_prop )
        compute_sparsity_dispatch<CTA_SIZE, true, false>(
            hash_wk,
            num_aggregates,
            R_row_offsets.raw(),
            R_column_indices.raw(),
            A.row_offsets.raw(),
            A.col_indices.raw(),
            aggregates.raw(),
            Ac.row_offsets.raw(),
            Ac.col_indices.raw(),
            amgx::thrust::raw_pointer_cast( &Ac_pos.front() ));
    else
        compute_sparsity_dispatch<CTA_SIZE, false, false>(
            hash_wk,
            num_aggregates,
            R_row_offsets.raw(),
            R_column_indices.raw(),
            A.row_offsets.raw(),
            A.col_indices.raw(),
            aggregates.raw(),
            Ac.row_offsets.raw(),
            Ac.col_indices.raw(),
            amgx::thrust::raw_pointer_cast( &Ac_pos.front() ));

    cudaCheckError();

    // Reset values if needed.
    if ( A.get_block_dimy() != 1 )
    {
        thrust_wrapper::fill<AMGX_device>( Ac.values.begin(), Ac.values.end(), types::util<ValueType>::get_zero() );
        cudaCheckError();
    }

    // Compute values.
    if ( diag_prop )
    {
        fill_A_dispatch<CTA_SIZE, true>(
            hash_wk,
            A.get_block_dimy(),
            num_aggregates,
            R_row_offsets.raw(),
            R_column_indices.raw(),
            A.row_offsets.raw(),
            A.col_indices.raw(),
            A.diag.raw(),
            A.values.raw(),
            aggregates.raw(),
            Ac.row_offsets.raw(),
            Ac.col_indices.raw(),
            amgx::thrust::raw_pointer_cast( &Ac_pos.front() ),
            Ac.diag.raw(),
            Ac.values.raw(),
            this->m_force_determinism );
    }
    else
    {
        fill_A_dispatch<CTA_SIZE, false>(
            hash_wk,
            A.get_block_dimy(),
            num_aggregates,
            R_row_offsets.raw(),
            R_column_indices.raw(),
            A.row_offsets.raw(),
            A.col_indices.raw(),
            A.diag.raw(),
            A.values.raw(),
            aggregates.raw(),
            Ac.row_offsets.raw(),
            Ac.col_indices.raw(),
            amgx::thrust::raw_pointer_cast( &Ac_pos.front() ),
            Ac.diag.raw(),
            Ac.values.raw(),
            this->m_force_determinism );
    }

    cudaCheckError();

    // Update the diagonal if needed.
    if ( Ac.is_matrix_singleGPU() )
    {
        Ac.computeDiagonal();
    }

    cudaCheckError();
    // Finalize the modification.
    Ac.set_initialized(1);
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template< AMGX_VecPrecision V, AMGX_MatPrecision M, AMGX_IndPrecision I >
void
LowDegCoarseAGenerator<TemplateConfig<AMGX_host, V, M, I> >::computeAOperator( const Matrix_h &h_A,
        Matrix_h &h_Ac,
        const IVector &h_aggregates,
        const IVector &h_R_row_offsets,
        const IVector &h_R_column_indices,
        const int num_aggregates )
{
    h_Ac.set_initialized(0);
    IVector rows;
    IVector inds;
    typename Matrix_h::MVector vals;
    typename Matrix_h::MVector diag;
    int num_nnz = 0;
    int diag_prop = h_A.hasProps(DIAG);

    for ( int row = 0; row < num_aggregates; row++ )
    {
        for ( int col = 0; col < num_aggregates; col++ )
        {
            int fill = 0;
            typename Matrix_h::MVector cur(h_A.get_block_size(), types::util<typename Matrix_h::value_type>::get_zero());

            for ( int rc = h_R_row_offsets[row]; rc < h_R_row_offsets[row + 1]; rc++ )
            {
                int j = h_R_column_indices[rc];

                for ( int ac = h_A.row_offsets[j]; ac < h_A.row_offsets[j + 1] + diag_prop; ac++ )
                {
                    int k = (ac == h_A.row_offsets[j + 1]) ? j : h_A.col_indices[ac];

                    for ( int q = h_R_row_offsets[col]; q < h_R_row_offsets[col + 1]; q++ )
                        if ( k == h_R_column_indices[q] )
                        {
                            fill = 1;
                            int val_idx = (ac == h_A.row_offsets[j + 1]) ? h_A.get_num_nz() + j : ac;

                            for ( int v = 0; v < h_A.get_block_size(); v++)
                            {
                                cur[v] = cur[v] + h_A.values[val_idx * h_A.get_block_size() + v];
                            }
                        }
                }
            }

            if ( fill )
            {
                if ( row != col || !diag_prop )
                {
                    inds.push_back(col);
                    rows.push_back(row);
                    num_nnz++;

                    for ( int v = 0; v < h_A.get_block_size(); v++ )
                    {
                        vals.push_back(cur[v]);
                    }
                }
                else
                {
                    for ( int v = 0; v < h_A.get_block_size(); v++ )
                    {
                        diag.push_back(cur[v]);
                    }
                }
            }
        }
    }

    rows.push_back(-1);

    // add diagonal to the end
    if ( diag_prop )
    {
        for ( int v = 0; v < num_aggregates * h_A.get_block_size(); v++ )
        {
            vals.push_back(diag[v]);
        }
    }
    else
    {
        // Add a zero at the end
        for (int v = 0; v < h_A.get_block_size(); v++)
        {
            vals.push_back(types::util<typename Matrix_h::value_type>::get_zero());
        }
    }

    h_Ac.resize(num_aggregates, num_aggregates, num_nnz, h_A.get_block_dimx(), h_A.get_block_dimy(), 1);
    h_Ac.row_indices = rows;
    h_Ac.col_indices = inds;
    h_Ac.values = vals;
    h_Ac.addProps( CSR | ( diag_prop ? DIAG : 0 ) );
    h_Ac.computeDiagonal();
    h_Ac.set_initialized(1);
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#define AMGX_CASE_LINE(CASE) template class LowDegCoarseAGenerator<TemplateMode<CASE>::Type>;
AMGX_FORALL_BUILDS(AMGX_CASE_LINE)
AMGX_FORCOMPLEX_BUILDS(AMGX_CASE_LINE)
#undef AMGX_CASE_LINE

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace aggregation
} // namespace amgx
