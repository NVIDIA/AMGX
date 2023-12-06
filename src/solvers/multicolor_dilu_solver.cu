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
#include <string.h>
#include <cutil.h>
#include <miscmath.h>
#include <amgx_cusparse.h>
#include <thrust/copy.h>
#include <solvers/multicolor_dilu_solver.h>
#include <solvers/block_common_solver.h>
#include <gaussian_elimination.h>
#include <basic_types.h>
#include <util.h>
#include <texture.h>
#include <matrix_io.h>
#include <thrust/logical.h>
#include <sm_utils.inl>

#include <amgx_types/util.h>

#include <algorithm>

#define AMGX_ILU_COLORING

namespace amgx
{
namespace multicolor_dilu_solver
{

enum { CTA_SIZE = 128, WARP_SIZE = 32 };


///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template< typename Matrix_type, typename Vector_type, int N, int CTA_SIZE, int WARP_SIZE, int NUM_WARP_ITERS_PER_BLOCK >
__global__
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 700
__launch_bounds__( CTA_SIZE, 12 )
#elif defined(__CUDA_ARCH__)
__launch_bounds__( CTA_SIZE, 12 )
#endif
void DILU_setup_NxN_kernel_large( const int *__restrict A_rows,
                                  const int *__restrict A_cols,
                                  const int *__restrict A_diag,
                                  const Matrix_type *__restrict A_vals,
                                  Matrix_type *__restrict Einv,
                                  const int *sorted_rows_by_color,
                                  const int *row_colors,
                                  const int  num_rows_per_color,
                                  const int  current_color )
{
    const int NUM_WARPS_PER_CTA = CTA_SIZE / WARP_SIZE;
    // Squared N.
    const int NxN = N * N;
    // Number of items computer per CTA.
    const int NUM_ITEMS_PER_CTA = NUM_WARPS_PER_CTA;
    // Number of items per grid.
    const int NUM_ITEMS_PER_GRID = gridDim.x * NUM_ITEMS_PER_CTA;
    // The coordinates of the thread inside the CTA/warp.
    const int warp_id = utils::warp_id();
    const int lane_id = utils::lane_id();
    // Shared memory to broadcast column IDs.
    __shared__ volatile int s_a_col_ids[CTA_SIZE];
    __shared__ volatile int s_a_col_its[CTA_SIZE];
    // Each thread keeps its own pointer.
    volatile int *my_s_a_col_ids = &s_a_col_ids[threadIdx.x - lane_id];
    volatile int *my_s_a_col_its = &s_a_col_its[threadIdx.x - lane_id];
    // Shared memory to store the matrices.
    __shared__ volatile Vector_type s_A_mtx[CTA_SIZE * NUM_WARP_ITERS_PER_BLOCK];
    __shared__ volatile Vector_type s_B_mtx[CTA_SIZE * NUM_WARP_ITERS_PER_BLOCK];
    // Each thread keeps its own pointer to shared memory to avoid some extra computations.
    volatile Vector_type *my_s_A_mtx = &s_A_mtx[warp_id * NUM_WARP_ITERS_PER_BLOCK * WARP_SIZE];
    volatile Vector_type *my_s_B_mtx = &s_B_mtx[warp_id * NUM_WARP_ITERS_PER_BLOCK * WARP_SIZE];
    // Shared memory to store the index of the element Aji.
    __shared__ volatile int s_A_ji[NUM_WARPS_PER_CTA];
    // Each thread keeps its own pointer.
    volatile int *my_s_A_ji = &s_A_ji[warp_id];
    // Precomputing some stuff
    int idx[NUM_WARP_ITERS_PER_BLOCK];
    int idy[NUM_WARP_ITERS_PER_BLOCK];
#pragma unroll

    for (int wb = 0; wb < NUM_WARP_ITERS_PER_BLOCK; wb++)
    {
        const int id = (WARP_SIZE * wb + lane_id) % NxN;
        idx[wb] = id / N;
        idy[wb] = id % N;
    }

    // Determine which NxN block the threads work with.
    int a_row_it = blockIdx.x * NUM_ITEMS_PER_CTA + warp_id;

    // Iterate over the rows of the matrix. One warp per row.
    for ( ; utils::any( a_row_it < num_rows_per_color ) ; a_row_it += NUM_ITEMS_PER_GRID )
    {
        int a_row_id = -1;

        if ( a_row_it < num_rows_per_color )
        {
            a_row_id = sorted_rows_by_color[a_row_it];
        }

        // Load the diagonal.
        Vector_type e_out[NUM_WARP_ITERS_PER_BLOCK];
#pragma unroll

        for (int wb = 0; wb < NUM_WARP_ITERS_PER_BLOCK; wb++)
        {
            e_out[wb] = (Vector_type)0.0;
        }

#pragma unroll

        for (int wb = 0; wb < NUM_WARP_ITERS_PER_BLOCK; wb++)
            if ( a_row_id != -1 && (wb * WARP_SIZE + lane_id) < NxN)
            {
                e_out[wb] = A_vals[NxN * A_diag[a_row_id] + wb * WARP_SIZE + lane_id];
            }

        // Skip the 1st iteration of the outer-loop (that loop runs on the host).
        if ( current_color != 0 )
        {
            // Ranges of the rows.
            int a_col_begin(0), a_col_end(0);

            if ( a_row_id != -1 )
            {
                a_col_begin = A_rows[a_row_id  ];
                a_col_end   = A_rows[a_row_id + 1];
            }

            // Iterate over the elements in the columns.
            for ( ; a_col_begin < a_col_end ; a_col_begin += NxN )
            {
                // Each thread loads a single element. If !is_active, a_col_end == 0.
                int a_col_it = a_col_begin + lane_id;
                // The identifier of the column if the iterator is valid.
                int a_col_tmp = -1, a_col_id = -1;

                if ( a_col_it < a_col_end )
                {
                    a_col_tmp = A_cols[a_col_it];
                }

                if ( a_col_tmp != -1 && row_colors[a_col_tmp] < current_color )
                {
                    a_col_id = a_col_tmp;
                }

                // When the diagonal is stored inside the matrix, we have to reject it. We
                // could be using a template parameter but it's not needed since that
                // rejection is really cheap (a couple of extra cycles -- CMP+MOV).
                if ( a_col_id == a_row_id )
                {
                    a_col_id = -1;
                }

                // We partition valid and invalid column ids. Valid ones come first.
                int vote = utils::ballot( a_col_id != -1 );
                int ones = __popc( vote );
                int dest = __popc( vote & utils::lane_mask_lt() );

                if ( a_col_id == -1 )
                {
                    dest = ones + lane_id - dest;
                }

                my_s_a_col_ids[dest] = a_col_id;
                my_s_a_col_its[dest] = a_col_it;
                // Temporary storage with zeros for OOB
                Vector_type my_A[NUM_WARP_ITERS_PER_BLOCK], my_B[NUM_WARP_ITERS_PER_BLOCK];
#pragma unroll

                for (int wb = 0; wb < NUM_WARP_ITERS_PER_BLOCK; wb++)
                {
                    my_A[wb] = (Vector_type)0.0;
                    my_B[wb] = (Vector_type)0.0;
                }

                // Threads collaborate to load the rows.
                for ( int k = 0 ; k < WARP_SIZE ; ++k )
                {
                    // Exchange column indices.
                    const int uniform_a_col_id = my_s_a_col_ids[k];

                    // Early exit.
                    if ( uniform_a_col_id == -1 )
                    {
                        break;
                    }

                    // Load the iterator.
                    const int uniform_a_col_it = my_s_a_col_its[k];
                    // Load the two matrices.
#pragma unroll

                    for (int wb = 0; wb < NUM_WARP_ITERS_PER_BLOCK; wb++)
                        if ((wb * WARP_SIZE + lane_id) < NxN)
                        {
                            my_A[wb] = A_vals[NxN * uniform_a_col_it + wb * WARP_SIZE + lane_id];
                            my_B[wb] = Einv  [NxN * uniform_a_col_id + wb * WARP_SIZE + lane_id];
                        }

#pragma unroll

                    for (int wb = 0; wb < NUM_WARP_ITERS_PER_BLOCK; wb++)
                    {
                        my_s_A_mtx[lane_id + wb * WARP_SIZE] = my_A[wb];
                        my_s_B_mtx[lane_id + wb * WARP_SIZE] = my_B[wb];
                    }

                    // Compute the product of matrices.
#pragma unroll

                    for (int wb = 0; wb < NUM_WARP_ITERS_PER_BLOCK; wb++)
                    {
                        my_A[wb] = (Vector_type)0.0;
#pragma unroll

                        for ( int m = 0 ; m < N ; ++m )
                        {
                            my_A[wb] += my_s_A_mtx[N * idx[wb] + m] * my_s_B_mtx[N * m + idy[wb]];
                        }
                    }

#pragma unroll

                    for (int wb = 0; wb < NUM_WARP_ITERS_PER_BLOCK; wb++)
                        if ((wb * WARP_SIZE + lane_id) < NxN)
                        {
                            my_s_A_mtx[lane_id + wb * WARP_SIZE] = my_A[wb];
                        }

                    // We looking for columns in the two rows we're interested in.
                    int b_col_it  = A_rows[uniform_a_col_id  ];
                    int b_col_end = A_rows[uniform_a_col_id + 1];

                    // Init the marker to -1.
                    if ( lane_id == 0 )
                    {
                        *my_s_A_ji = -1;
                    }

                    // Run the loop.
                    b_col_it += lane_id;
                    int shared_found = utils::ballot( lane_id == 0 && uniform_a_col_id == -1 );

                    do
                    {
                        bool found = b_col_it < b_col_end && A_cols[b_col_it] == a_row_id;

                        if ( found )
                        {
                            *my_s_A_ji = b_col_it;
                        }

                        shared_found = shared_found | utils::ballot(found);
                        b_col_it += NxN;
                    }
                    while ( __popc( shared_found ) == 0 && utils::any( b_col_it < b_col_end ) );

                    // Load the blocks.
                    const int w_aji = *my_s_A_ji;
                    Vector_type my_C[NUM_WARP_ITERS_PER_BLOCK];
#pragma unroll

                    for (int wb = 0; wb < NUM_WARP_ITERS_PER_BLOCK; wb++)
                    {
                        my_C[wb] = (Vector_type)0.0;

                        if ( w_aji != -1 && (wb * WARP_SIZE + lane_id) < NxN)
                        {
                            my_C[wb] = A_vals[NxN * w_aji + wb * WARP_SIZE + lane_id];
                        }

                        my_s_B_mtx[wb * WARP_SIZE + lane_id] = my_C[wb];
                    }

                    // Update e_out.
#pragma unroll

                    for (int wb = 0; wb < NUM_WARP_ITERS_PER_BLOCK; wb++)
                    {
#pragma unroll

                        for ( int m = 0 ; m < N ; ++m )
                        {
                            e_out[wb] -= my_s_A_mtx[N * idx[wb] + m] * my_s_B_mtx[N * m + idy[wb]];
                        }
                    }
                }
            } // a_col_begin < a_col_end
        } // current_color != 0

        // Store e_out in A
#pragma unroll

        for (int wb = 0; wb < NUM_WARP_ITERS_PER_BLOCK; wb++)
        {
            my_s_B_mtx[wb * WARP_SIZE + lane_id] = my_s_A_mtx[wb * WARP_SIZE + lane_id] = e_out[wb];
        }

        // Invert the matrices.
#pragma unroll

        for ( int row = 0 ; row < N ; ++row )
        {
            Vector_type diag(0), diag_tmp = my_s_A_mtx[N * row + row];

            if ( isNotCloseToZero(diag_tmp) )
            {
                diag = Vector_type(1) / diag_tmp;
            }
            else
            {
                diag = Vector_type(1) / epsilon(diag_tmp);
            }

            if ( lane_id < N && lane_id != row)
            {
                my_s_A_mtx[N * row + lane_id] = my_s_B_mtx[N * row + lane_id] = my_s_B_mtx[N * row + lane_id] * diag;
            }

#pragma unroll

            for (int wb = 0; wb < NUM_WARP_ITERS_PER_BLOCK; wb++)
                if ( idx[wb] != row && idy[wb] != row)
                {
                    my_s_A_mtx[wb * WARP_SIZE + lane_id] = my_s_B_mtx[wb * WARP_SIZE + lane_id] - my_s_B_mtx[N * idx[wb] + row] * my_s_B_mtx[N * row + idy[wb]];
                }

            if ( lane_id < N )
            {
                Vector_type tmp = diag;

                if ( lane_id != row )
                {
                    tmp = -my_s_A_mtx[N * lane_id + row] * diag;
                }

                my_s_A_mtx[N * lane_id + row] = tmp;
            }

#pragma unroll

            for (int wb = 0; wb < NUM_WARP_ITERS_PER_BLOCK; wb++)
            {
                my_s_B_mtx[wb * WARP_SIZE + lane_id] = my_s_A_mtx[wb * WARP_SIZE + lane_id];
            }
        }

        // Store the results to Einv.
        if ( a_row_id != -1 )
#pragma unroll
            for (int wb = 0; wb < NUM_WARP_ITERS_PER_BLOCK; wb++)
                if (wb * WARP_SIZE + lane_id < NxN)
                {
                    Einv[NxN * a_row_id + wb * WARP_SIZE + lane_id] = my_s_A_mtx[wb * WARP_SIZE + lane_id];
                }
    }
}


template< typename Matrix_type, typename Vector_type, int N, int CTA_SIZE, int WARP_SIZE >
__global__
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 700
__launch_bounds__( CTA_SIZE, 12 )
#elif defined(__CUDA_ARCH__)
__launch_bounds__( CTA_SIZE, 12 )
#endif
void DILU_setup_NxN_kernel( const int *__restrict A_rows,
                            const int *__restrict A_cols,
                            const int *__restrict A_diag,
                            const Matrix_type *__restrict A_vals,
                            Matrix_type *__restrict Einv,
                            const int *sorted_rows_by_color,
                            const int *row_colors,
                            const int  num_rows_per_color,
                            const int  current_color )
{
    const int NUM_WARPS_PER_CTA = CTA_SIZE / WARP_SIZE;
    // Squared N.
    const int NxN = N * N;
    // Number of items per warp.
    const int NUM_ITEMS_PER_WARP = WARP_SIZE / NxN;
    // Upper-bound on the number of items per warp.
    const int NUM_ITEMS_PER_WARP_CEIL = (WARP_SIZE + NxN - 1) / NxN;
    // Number of items computer per CTA.
    const int NUM_ITEMS_PER_CTA = NUM_ITEMS_PER_WARP * NUM_WARPS_PER_CTA;
    // Number of items per grid.
    const int NUM_ITEMS_PER_GRID = gridDim.x * NUM_ITEMS_PER_CTA;
    // The coordinates of the thread inside the CTA/warp.
    const int warp_id = utils::warp_id();
    const int lane_id = utils::lane_id();
    // Constants.
    const int lane_id_div_NxN = lane_id / NxN;
    const int lane_id_mod_NxN = lane_id % NxN;
    // Useful index to compute matrix products.
    const int lane_id_mod_NxN_div_N = lane_id_mod_NxN / N;
    const int lane_id_mod_NxN_mod_N = lane_id_mod_NxN % N;
    // We need NxN to compute a NxN block. Encode a mask for the first block.
    int mask_tmp = utils::ballot( lane_id_div_NxN == 0 );
    // Mask for ballots. We shift the mask with NxN active bits by the needed number of bits.
    const int mask_NxN = mask_tmp << (lane_id_div_NxN * __popc(mask_tmp));
    // Shared memory to broadcast column IDs.
    __shared__ volatile int s_a_col_ids[CTA_SIZE];
    __shared__ volatile int s_a_col_its[CTA_SIZE];
    // Each thread keeps its own pointer.
    volatile int *my_s_a_col_ids = &s_a_col_ids[threadIdx.x - lane_id_mod_NxN];
    volatile int *my_s_a_col_its = &s_a_col_its[threadIdx.x - lane_id_mod_NxN];
    // Shared memory to store the matrices.
    __shared__ volatile Vector_type s_A_mtx[CTA_SIZE];
    __shared__ volatile Vector_type s_B_mtx[CTA_SIZE];
    // Each thread keeps its own pointer to shared memory to avoid some extra computations.
    volatile Vector_type *my_s_A_mtx = &s_A_mtx[threadIdx.x - lane_id_mod_NxN];
    volatile Vector_type *my_s_B_mtx = &s_B_mtx[threadIdx.x - lane_id_mod_NxN];
    // Shared memory to store the index of the element Aji.
    __shared__ volatile int s_A_ji[NUM_WARPS_PER_CTA * NUM_ITEMS_PER_WARP_CEIL];
    // Each thread keeps its own pointer.
    volatile int *my_s_A_ji = &s_A_ji[warp_id * NUM_ITEMS_PER_WARP_CEIL + lane_id_div_NxN];
    // Determine which NxN block the threads work with.
    int a_row_it = blockIdx.x * NUM_ITEMS_PER_CTA + warp_id * NUM_ITEMS_PER_WARP + lane_id_div_NxN;

    // Iterate over the rows of the matrix. One warp per row.
    for ( ; utils::any( a_row_it < num_rows_per_color ) ; a_row_it += NUM_ITEMS_PER_GRID )
    {
        // Is the thread active? For example, for 5x5 only the first 25 threads are active per warp.
        // At compile time, the compiler will see is_active == true for 2x2 (since NxN & (NxN-1) evals
        // to false ; that's the common trick to determine if a number is a power of 2).
        int is_active = true;

        if ( NxN & (NxN - 1) )
        {
            is_active = lane_id_div_NxN < NUM_ITEMS_PER_WARP;
        }

        int a_row_id = -1;

        if ( is_active && a_row_it < num_rows_per_color )
        {
            a_row_id = sorted_rows_by_color[a_row_it];
        }

        // Load the diagonal.
        Vector_type e_out(0);

        if ( a_row_id != -1 )
        {
            e_out = A_vals[NxN * A_diag[a_row_id] + lane_id_mod_NxN];
        }

        // Skip the 1st iteration of the outer-loop (that loop runs on the host).
        if ( current_color != 0 )
        {
            // Ranges of the rows.
            int a_col_begin(0), a_col_end(0);

            if ( a_row_id != -1 )
            {
                a_col_begin = A_rows[a_row_id  ];
                a_col_end   = A_rows[a_row_id + 1];
            }

            // Iterate over the elements in the columns.
            for ( ; a_col_begin < a_col_end ; a_col_begin += NxN )
            {
                unsigned int active_mask = utils::activemask();
                // Each thread loads a single element. If !is_active, a_col_end == 0.
                int a_col_it = a_col_begin + lane_id_mod_NxN;
                // The identifier of the column if the iterator is valid.
                int a_col_tmp = -1, a_col_id = -1;

                if ( a_col_it < a_col_end )
                {
                    a_col_tmp = A_cols[a_col_it];
                }

                if ( a_col_tmp != -1 && row_colors[a_col_tmp] < current_color )
                {
                    a_col_id = a_col_tmp;
                }

                // When the diagonal is stored inside the matrix, we have to reject it. We
                // could be using a template parameter but it's not needed since that
                // rejection is really cheap (a couple of extra cycles -- CMP+MOV).
                if ( a_col_id == a_row_id )
                {
                    a_col_id = -1;
                }

                // We partition valid and invalid column ids. Valid ones come first.
                int vote = utils::ballot( a_col_id != -1, active_mask ) & mask_NxN;
                int ones = __popc( vote );
                int dest = __popc( vote & utils::lane_mask_lt() );

                if ( a_col_id == -1 )
                {
                    dest = ones + lane_id_mod_NxN - dest;
                }

                my_s_a_col_ids[dest] = a_col_id;
                my_s_a_col_its[dest] = a_col_it;

                // Threads collaborate to load the rows.
                for ( int k = 0 ; k < NxN ; ++k )
                {
                    // Exchange column indices.
                    const int uniform_a_col_id = my_s_a_col_ids[k];

                    // Early exit.
                    if ( utils::all( uniform_a_col_id == -1, active_mask ) )
                    {
                        break;
                    }

                    // Load the iterator.
                    const int uniform_a_col_it = my_s_a_col_its[k];
                    // Load the two matrices.
                    Vector_type my_A(0), my_B(0);

                    if ( uniform_a_col_id != -1 )
                    {
                        my_A = A_vals[NxN * uniform_a_col_it + lane_id_mod_NxN];
                        my_B = Einv  [NxN * uniform_a_col_id + lane_id_mod_NxN];
                    }

                    my_s_A_mtx[lane_id_mod_NxN] = my_A;
                    my_s_B_mtx[lane_id_mod_NxN] = my_B;
                    utils::syncwarp(active_mask);
                    // Compute the product of matrices.
                    Vector_type tmp(0);
#pragma unroll

                    for ( int m = 0 ; m < N ; ++m )
                    {
                        tmp += my_s_A_mtx[N * lane_id_mod_NxN_div_N + m] * my_s_B_mtx[N * m + lane_id_mod_NxN_mod_N];
                    }

                    my_s_A_mtx[lane_id_mod_NxN] = tmp;
                    // We looking for columns in the two rows we're interested in.
                    int b_col_it(0), b_col_end(0);

                    if ( is_active && uniform_a_col_id != -1 )
                    {
                        b_col_it  = A_rows[uniform_a_col_id  ];
                        b_col_end = A_rows[uniform_a_col_id + 1];
                    }

                    // Init the marker to -1.
                    if ( lane_id_mod_NxN == 0 )
                    {
                        *my_s_A_ji = -1;
                    }

                    // Run the loop.
                    b_col_it += lane_id_mod_NxN;
                    int shared_found = utils::ballot( lane_id_mod_NxN == 0 && uniform_a_col_id == -1, active_mask );

                    do
                    {
                        bool found = b_col_it < b_col_end && A_cols[b_col_it] == a_row_id;

                        if ( found )
                        {
                            *my_s_A_ji = b_col_it;
                        }

                        shared_found = shared_found | utils::ballot(found, active_mask);
                        b_col_it += NxN;
                    }
                    while ( __popc( shared_found ) < NUM_ITEMS_PER_WARP && utils::any( b_col_it < b_col_end, active_mask ) );

                    // Load the blocks.
                    const int w_aji = *my_s_A_ji;
                    Vector_type my_C(0);

                    if ( w_aji != -1 )
                    {
                        my_C = A_vals[NxN * w_aji + lane_id_mod_NxN];
                    }

                    my_s_B_mtx[lane_id_mod_NxN] = my_C;
                    // Update e_out.
#pragma unroll

                    for ( int m = 0 ; m < N ; ++m )
                    {
                        e_out -= my_s_A_mtx[N * lane_id_mod_NxN_div_N + m] * my_s_B_mtx[N * m + lane_id_mod_NxN_mod_N];
                    }
                }
            } // a_col_begin < a_col_end
        } // current_color != 0

        // Store e_out in A
        my_s_A_mtx[lane_id_mod_NxN] = e_out;
        // Invert the matrices.
#pragma unroll

        for ( int row = 0 ; row < N ; ++row )
        {
            Vector_type diag(0), diag_tmp = my_s_A_mtx[N * row + row];

            if ( isNotCloseToZero(diag_tmp) )
            {
                diag = Vector_type(1) / diag_tmp;
            }
            else
            {
                diag = Vector_type(1) / epsilon(diag_tmp);
            }

            if ( is_active && lane_id_mod_NxN_div_N == 0 && lane_id_mod_NxN_mod_N != row )
            {
                my_s_A_mtx[N * row + lane_id_mod_NxN_mod_N] *= diag;
            }

            if ( is_active && lane_id_mod_NxN_div_N != row && lane_id_mod_NxN_mod_N != row )
            {
                my_s_A_mtx[lane_id_mod_NxN] -= my_s_A_mtx[N * lane_id_mod_NxN_div_N + row] * my_s_A_mtx[N * row + lane_id_mod_NxN_mod_N];
            }

            if ( is_active && lane_id_mod_NxN_div_N == 0 )
            {
                Vector_type tmp = diag;

                if ( lane_id_mod_NxN_mod_N != row )
                {
                    tmp = -my_s_A_mtx[N * lane_id_mod_NxN_mod_N + row] * diag;
                }

                my_s_A_mtx[N * lane_id_mod_NxN_mod_N + row] = tmp;
            }
            utils::syncwarp();
        }

        // Store the results to Einv.
        if ( a_row_id != -1 )
        {
            Einv[NxN * a_row_id + lane_id_mod_NxN] = my_s_A_mtx[lane_id_mod_NxN];
        }
    }
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template< typename Matrix_type, typename Vector_type, int NUM_THREADS_PER_ROW, int CTA_SIZE, int WARP_SIZE >
__global__
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 700
__launch_bounds__( CTA_SIZE, 16 )
#elif defined(__CUDA_ARCH__)
__launch_bounds__( CTA_SIZE, 16 )
#endif
void DILU_setup_1x1_kernel( const int *__restrict A_rows,
                            const int *__restrict A_cols,
                            const int *__restrict A_diag,
                            const Matrix_type *__restrict A_vals,
                            Matrix_type *__restrict Einv,
                            const int *sorted_rows_by_color,
                            const int *row_colors,
                            const int  num_rows_per_color,
                            const int  current_color )
{
    const int NUM_WARPS_PER_CTA = CTA_SIZE / WARP_SIZE;
    // Number of items per grid.
    const int NUM_WARPS_PER_GRID = gridDim.x * NUM_WARPS_PER_CTA;
    // The coordinates of the thread inside the CTA/warp.
    const int warp_id = utils::warp_id();
    const int lane_id = utils::lane_id();
    // Constants.
    const int lane_id_div_NTPR = lane_id / NUM_THREADS_PER_ROW;
    const int lane_id_mod_NTPR = lane_id % NUM_THREADS_PER_ROW;
    // Shared memory to broadcast column IDs.
    __shared__ int s_a_col_ids[CTA_SIZE];
    // Each thread keeps its own pointer.
    int *my_s_a_col_ids = &s_a_col_ids[warp_id * WARP_SIZE];
    // Shared memory to store the matrices.
    __shared__ int s_A_ji[CTA_SIZE];
    // Each thread keeps its own pointer to shared memory to avoid some extra computations.
    int *my_s_A_ji = &s_A_ji[warp_id * WARP_SIZE];
    // Determine which NxN block the threads work with.
    int a_row_it = blockIdx.x * NUM_WARPS_PER_CTA + warp_id;

    // Iterate over the rows of the matrix. One warp per row.
    for ( ; a_row_it < num_rows_per_color ; a_row_it += NUM_WARPS_PER_GRID )
    {
        int a_row_id = sorted_rows_by_color[a_row_it];
        // Load the diagonal.
        Vector_type e_out(0);

        // Skip the 1st iteration of the outer-loop (that loop runs on the host).
        if ( current_color != 0 )
        {
            // Ranges of the row.
            int a_col_begin = A_rows[a_row_id  ];
            int a_col_end   = A_rows[a_row_id + 1];

            // Iterate over the elements in the columns.
            for ( ; a_col_begin < a_col_end ; a_col_begin += WARP_SIZE )
            {
                // Each thread loads a single element.
                int a_col_it = a_col_begin + lane_id;
                // The identifier of the column if the iterator is valid.
                int a_col_tmp = -1, a_col_id = -1;

                if ( a_col_it < a_col_end )
                {
                    a_col_tmp = A_cols[a_col_it];
                }

                if ( a_col_tmp != -1 && row_colors[a_col_tmp] < current_color )
                {
                    a_col_id = a_col_tmp;
                }

                // When the diagonal is stored inside the matrix, we have to reject it. We
                // could be using a template parameter but it's not needed since that
                // rejection is really cheap (a couple of extra cycles -- CMP+MOV).
                if ( a_col_id == a_row_id )
                {
                    a_col_id = -1;
                }

                // We partition valid and invalid column ids. Valid ones come first.
                int vote = utils::ballot( a_col_id != -1 );
                int ones = __popc( vote );
                int dest = __popc( vote & utils::lane_mask_lt() );

                if ( a_col_id == -1 )
                {
                    dest = ones + lane_id - dest;
                }

                my_s_a_col_ids[dest] = a_col_id;
                // Reset A_jis.
                my_s_A_ji[lane_id] = -1;

                __syncwarp();

                // Threads collaborate to load the rows.
                for ( int k = 0 ; k < ones ; k += WARP_SIZE / NUM_THREADS_PER_ROW )
                {
                    const int local_k = k + lane_id_div_NTPR;
                    // Exchange column indices.
                    int uniform_a_col_id = -1;

                    if ( local_k < ones )
                    {
                        uniform_a_col_id = my_s_a_col_ids[local_k];
                    }

                    // We look for columns in the rows we're interested in.
                    int b_col_it(0), b_col_end(0);

                    if ( uniform_a_col_id != -1 )
                    {
                        b_col_it  = A_rows[uniform_a_col_id  ];
                        b_col_end = A_rows[uniform_a_col_id + 1];
                    }

                    // Run the loop.
                    b_col_it += lane_id_mod_NTPR;
                    int shared_found = utils::ballot( lane_id_mod_NTPR == 0 && uniform_a_col_id == -1 );

                    do
                    {
                        bool found = b_col_it < b_col_end && A_cols[b_col_it] == a_row_id;

                        if ( found )
                        {
                            my_s_A_ji[local_k] = b_col_it;
                        }

                        shared_found = shared_found | utils::ballot(found);
                        b_col_it += NUM_THREADS_PER_ROW;
                    }
                    while ( __popc( shared_found ) < WARP_SIZE / NUM_THREADS_PER_ROW && utils::any( b_col_it < b_col_end ) );
                }

                __syncwarp();

                // Where to get my A_ji from (if any).
                int a_ji_it = my_s_A_ji[dest];
                // Grab A_jis.
                Matrix_type a_ji(0);

                if ( a_ji_it != -1 )
                {
                    a_ji = A_vals[a_ji_it];
                }

                // Update e_out.
                if ( a_col_id != -1 )
                {
                    e_out += a_ji * Einv[a_col_id] * A_vals[a_col_it];
                }
            } // a_col_begin < a_col_end
        } // current_color != 0

        // Reduce the e_outs in one value.
#pragma unroll
        for ( int mask = WARP_SIZE / 2 ; mask > 0 ; mask >>= 1 )
        {
            e_out += utils::shfl_xor( e_out, mask );
        }

        // Store the result.
        if ( lane_id == 0 )
        {
            Matrix_type res = A_vals[A_diag[a_row_id]] - e_out;

            if ( res != Matrix_type(0) )
            {
                res = Matrix_type(1) / res;
            }

            Einv[a_row_id] = static_cast<Vector_type>(res);
        }
    }
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template< int N, bool ROW_MAJOR, int WARP_SIZE, typename Value_type >
static __device__ __forceinline__
Value_type reduce_distributed_vectors( Value_type x, int is_leader )
{
    if ( N & (N - 1) )
    {
#pragma unroll

        for ( int i = 1 ; i < N ; ++i )
        {
            Value_type other_x = utils::shfl_down( x, ROW_MAJOR ? i : N * i );

            if ( is_leader )
            {
                x += other_x;
            }
        }
    }
    else
    {
#pragma unroll

        for ( int i = 1 ; i < N ; i <<= 1 )
        {
            x += utils::shfl_xor( x, ROW_MAJOR ? i : N * i );
        }
    }

    return x;
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template< typename Matrix_type, typename Vector_type, int N, int CTA_SIZE, int WARP_SIZE, bool ROW_MAJOR, bool HAS_EXTERNAL_DIAG >
__global__
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 700
__launch_bounds__( CTA_SIZE, 12 )
#elif defined(__CUDA_ARCH__)
__launch_bounds__( CTA_SIZE, 12 )
#endif
void DILU_forward_NxN_kernel( const int *__restrict A_rows,
                              const int *__restrict A_cols,
                              const Matrix_type *__restrict A_vals,
                              const int *__restrict A_diag,
                              const Vector_type *x,
                              const Vector_type *b,
                              Vector_type *__restrict delta,
                              const int *__restrict sorted_rows_by_color,
                              const int num_rows_per_color,
                              const int current_color,
                              const int *__restrict row_colors,
                              const Matrix_type *__restrict Einv,
                              const ColoringType boundary_coloring,
                              const int boundary_index )
{
    const int NUM_WARPS_PER_CTA = CTA_SIZE / WARP_SIZE;
    // Squared N.
    const int NxN = N * N;
    // Number of items per warp.
    const int NUM_ITEMS_PER_WARP = WARP_SIZE / NxN;
    // Number of items computer per CTA.
    const int NUM_ITEMS_PER_CTA = NUM_ITEMS_PER_WARP * NUM_WARPS_PER_CTA;
    // Number of items per grid.
    const int NUM_ITEMS_PER_GRID = gridDim.x * NUM_ITEMS_PER_CTA;
    // The coordinates of the thread inside the CTA/warp.
    const int warp_id = utils::warp_id();
    const int lane_id = utils::lane_id();
    // Constants.
    const int lane_id_div_NxN = lane_id / NxN;
    const int lane_id_mod_NxN = lane_id % NxN;
    // Useful index to compute matrix products.
    const int lane_id_mod_NxN_div_N = lane_id_mod_NxN / N;
    const int lane_id_mod_NxN_mod_N = lane_id_mod_NxN % N;
    // We to get my data from when I use SHFL.
    const int shfl_offset = lane_id - lane_id_mod_NxN;
    // Shared memory needed to exchange X and delta.
    __shared__ volatile Vector_type s_mem[CTA_SIZE];
    // Each thread keeps its own pointer to shared memory to avoid some extra computations.
    volatile Vector_type *my_s_mem = &s_mem[threadIdx.x - lane_id_mod_NxN];
    // Is the thread active? For example, for 5x5 only the first 25 threads are active per warp.
    // At compile time, the compiler will see is_active == true for 2x2 (since NxN & (NxN-1) evals
    // to false ; that's the common trick to determine if a number is a power of 2).
    int is_active = true;

    if ( NxN & (NxN - 1) )
    {
        is_active = lane_id_div_NxN < NUM_ITEMS_PER_WARP;
    }

    // Determine which NxN block the threads work with.
    int a_row_it = num_rows_per_color;

    if ( is_active )
    {
        a_row_it = blockIdx.x * NUM_ITEMS_PER_CTA + warp_id * NUM_ITEMS_PER_WARP + lane_id_div_NxN;
    }

    // Iterate over the rows of the matrix. One warp per row.
    for ( ; a_row_it < num_rows_per_color ; a_row_it += NUM_ITEMS_PER_GRID )
    {
        int a_row_id = sorted_rows_by_color[a_row_it];
        // Load one block of B.
        Vector_type my_bmAx(0);

        if ( ROW_MAJOR )
        {
            if ( lane_id_mod_NxN_mod_N == 0 )
            {
                my_bmAx = __cachingLoad(&b[N * a_row_id + lane_id_mod_NxN_div_N]);
            }
        }
        else
        {
            if ( lane_id_mod_NxN_div_N == 0 )
            {
                my_bmAx = b[N * a_row_id + lane_id_mod_NxN_mod_N];
            }
        }

        // Don't do anything if X is zero.
        int a_col_begin = A_rows[a_row_id  ];
        int a_col_end   = A_rows[a_row_id + 1];
        // If the diagonal is stored separately, we have a special treatment.
        int a_col_max = a_col_end;

        if ( HAS_EXTERNAL_DIAG )
        {
            ++a_col_max;
        }

        // Each warp load column indices of 32 nonzero blocks
        for ( ; utils::any( a_col_begin < a_col_max ) ; a_col_begin += NxN )
        {
            // Each thread loads a single element. If !is_active, a_col_end == 0.
            int a_col_it = a_col_begin + lane_id_mod_NxN;
            // Get the ID of the column.
            int a_col_id = -1;

            if ( a_col_it < a_col_end )
            {
                a_col_id = A_cols[a_col_it];
            }

            if ( HAS_EXTERNAL_DIAG && a_col_it == a_col_end )
            {
                a_col_id = a_row_id;
            }

            // Determine if the color is valid.
            int a_col_is_valid = false;
#ifdef AMGX_ILU_COLORING
            if ( a_col_id != -1 && current_color != 0 )
            {
                if ( boundary_coloring == FIRST )
                {
                    a_col_is_valid = a_col_id >= boundary_index;
                }
                else
                {
                    a_col_is_valid = a_col_id < boundary_index && row_colors[a_col_id] < current_color;
                }
            }

#else

            if ( a_col_id != -1 && current_color != 0 )
            {
                a_col_is_valid = row_colors[a_col_id] < current_color;
            }

#endif
            // Count the number of active columns.
            // int vote =  utils::ballot(aColId != -1);
            // The number of iterations.
            // int nCols = max( __popc( vote & 0x0000ffff ), __popc( vote & 0xffff0000 ) );

            // Loop over columns. We compute 8 columns per iteration.
            for ( int k = 0 ; k < NxN ; k += N )
            {
                int my_k = k + lane_id_mod_NxN_div_N;
                // Load N blocks of X.
                int uniform_a_col_id = utils::shfl( a_col_id, shfl_offset + my_k );
                int uniform_a_col_is_valid = utils::shfl( a_col_is_valid, shfl_offset + my_k );
                Vector_type my_x(0);

                if ( uniform_a_col_id != -1 )
                {
                    my_x = __cachingLoad(&x[N * uniform_a_col_id + lane_id_mod_NxN_mod_N]);
                }

                if ( uniform_a_col_id != -1 && uniform_a_col_is_valid )
                {
                    my_x += delta[N * uniform_a_col_id + lane_id_mod_NxN_mod_N];
                }

                my_s_mem[lane_id_mod_NxN] = my_x;
                // Load N blocks of A.
#pragma unroll

                for ( int i = 0 ; i < N ; ++i )
                {
                    int uniform_a_col_tmp = a_col_begin + k + i, uniform_a_col_it = -1;

                    if ( uniform_a_col_tmp < a_col_end )
                    {
                        uniform_a_col_it = uniform_a_col_tmp;
                    }

                    if ( HAS_EXTERNAL_DIAG && is_active && uniform_a_col_tmp == a_col_end )
                    {
                        uniform_a_col_it = A_diag[a_row_id];
                    }

                    Matrix_type my_val(0);

                    if ( uniform_a_col_it != -1 )
                    {
                        my_val = A_vals[NxN * uniform_a_col_it + lane_id_mod_NxN];
                    }

                    if ( ROW_MAJOR )
                    {
                        my_bmAx -= my_val * my_s_mem[N * i + lane_id_mod_NxN_mod_N];
                    }
                    else
                    {
                        my_bmAx -= my_val * my_s_mem[N * i + lane_id_mod_NxN_div_N];
                    }
                }
            } // Loop over k
        } // Loop over aColIt

        // Load Einvs.
        Vector_type my_Einv = Einv[NxN * a_row_id + lane_id_mod_NxN];
        // Reduce bmAx terms.
        int is_leader = lane_id_mod_NxN_div_N == 0;

        if ( ROW_MAJOR )
        {
            is_leader = lane_id_mod_NxN_mod_N == 0;
        }

        my_bmAx = reduce_distributed_vectors<N, ROW_MAJOR, WARP_SIZE>( my_bmAx, is_leader );

        // Update the shared terms.
        if ( ROW_MAJOR )
        {
            if ( lane_id_mod_NxN_mod_N == 0 )
            {
                my_s_mem[lane_id_mod_NxN_div_N] = my_bmAx;
            }
        }
        else
        {
            if ( lane_id_mod_NxN_div_N == 0 )
            {
                my_s_mem[lane_id_mod_NxN_mod_N] = my_bmAx;
            }
        }

        // Update the diagonal term.
        if ( ROW_MAJOR )
        {
            my_bmAx = my_Einv * my_s_mem[lane_id_mod_NxN_mod_N];
        }
        else
        {
            my_bmAx = my_Einv * my_s_mem[lane_id_mod_NxN_div_N];
        }

        // Reduce bmAx terms.
        my_bmAx = reduce_distributed_vectors<N, ROW_MAJOR, WARP_SIZE>( my_bmAx, is_leader );

        // Store the results.
        if ( ROW_MAJOR )
        {
            if ( lane_id_mod_NxN_mod_N == 0 )
            {
                delta[N * a_row_id + lane_id_mod_NxN_div_N] = my_bmAx;
            }
        }
        else
        {
            if ( lane_id_mod_NxN_div_N == 0 )
            {
                delta[N * a_row_id + lane_id_mod_NxN_mod_N] = my_bmAx;
            }
        }
    }
}

template< typename Matrix_type, typename Vector_type, int N, int CTA_SIZE, int WARP_SIZE, bool HAS_EXTERNAL_DIAG, int NUM_WARP_ITERS_PER_BLOCK >
__global__
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 700
__launch_bounds__( CTA_SIZE, 12 )
#elif defined(__CUDA_ARCH__)
__launch_bounds__( CTA_SIZE, 12 )
#endif
void DILU_forward_NxN_kernel_large( const int *__restrict A_rows,
                                    const int *__restrict A_cols,
                                    const Matrix_type *__restrict A_vals,
                                    const int *__restrict A_diag,
                                    const Vector_type *x,
                                    const Vector_type *b,
                                    Vector_type *__restrict delta,
                                    const int *__restrict sorted_rows_by_color,
                                    const int num_rows_per_color,
                                    const int current_color,
                                    const int *__restrict row_colors,
                                    const Matrix_type *__restrict Einv,
                                    const ColoringType boundary_coloring,
                                    const int boundary_index )
{
    const int NUM_WARPS_PER_CTA = CTA_SIZE / WARP_SIZE;
    // Squared N.
    const int NxN = N * N;
    // Number of rows computed per CTA.
    const int NUM_ITEMS_PER_CTA = NUM_WARPS_PER_CTA;
    // Number of rows? per grid.
    const int NUM_ITEMS_PER_GRID = CTA_SIZE;
    // The coordinates of the thread inside the CTA/warp.
    const int warp_id = utils::warp_id();
    const int lane_id = utils::lane_id();
    // Constants.
    // Useful index to compute matrix products.
    const int lane_id_div_N = lane_id / N;
    const int lane_id_mod_N = lane_id % N; // id of a lane inside the block
    const int blocks_per_warp = WARP_SIZE / N; // we process this cols per warp per row
    const int row_elems_per_warp  = blocks_per_warp * N;
    // Shared to store bmAx
    __shared__ volatile Vector_type bmAx[CTA_SIZE * NUM_WARP_ITERS_PER_BLOCK];
    volatile Vector_type *my_bmAx_s = &bmAx[warp_id * NUM_WARP_ITERS_PER_BLOCK * WARP_SIZE];
    // Determine which NxN block the threads work with.
    int a_row_it = blockIdx.x * NUM_ITEMS_PER_CTA + warp_id;

    // Iterate over the rows of the matrix. One warp per row.
    for ( ; a_row_it < num_rows_per_color ; a_row_it += NUM_ITEMS_PER_GRID )
    {
        int a_row_id = sorted_rows_by_color[a_row_it];
        // Load one block of B.
        Vector_type my_bmAx(0);

        if ( lane_id < N )
        {
            my_bmAx = __cachingLoad(&b[N * a_row_id + lane_id]);
        }

#pragma unroll

        for (int i = 0; i < NUM_WARP_ITERS_PER_BLOCK; i++)
        {
            my_bmAx_s[WARP_SIZE * i + lane_id] = 0.0;
        }

        // Don't do anything if X is zero.
        int a_col_begin = A_rows[a_row_id  ];
        int a_col_end   = A_rows[a_row_id + 1];
        // If the diagonal is stored separately, we have a special treatment.
        int a_col_max = a_col_end;

        if ( HAS_EXTERNAL_DIAG )
        {
            ++a_col_max;
        }

        // Each warp load column indices of 32 nonzero blocks
        for ( ; utils::any( a_col_begin < a_col_max ) ; a_col_begin += WARP_SIZE ) // NxN
        {
            // Each thread loads a single element. If !is_active, a_col_end == 0.
            int a_col_it = a_col_begin + lane_id;
            // Get the ID of the column.
            int a_col_id = -1;

            if ( a_col_it < a_col_end )
            {
                a_col_id = A_cols[a_col_it];
            }

            if ( HAS_EXTERNAL_DIAG && a_col_it == a_col_end )
            {
                a_col_id = a_row_id;
            }

            // Determine if the color is valid.
            int a_col_is_valid = false;
#ifdef AMGX_ILU_COLORING

            if ( a_col_id != -1 && current_color != 0 )
            {
                if ( boundary_coloring == FIRST )
                {
                    a_col_is_valid = a_col_id >= boundary_index;
                }
                else
                {
                    a_col_is_valid = a_col_id < boundary_index && row_colors[a_col_id] < current_color;
                }
            }

#else

            if ( a_col_id != -1 && current_color != 0 )
            {
                a_col_is_valid = row_colors[a_col_id] < current_color;
            }

#endif

            // Loop over columns. We compute blocks_per_warp columns per iteration.
            for ( int k = 0 ; k < WARP_SIZE ; k += blocks_per_warp )
            {
                // id of the processed block by this thread
                int my_k = k + lane_id_div_N;
                // Load N blocks of X (if valid)
                int uniform_a_col_id = utils::shfl( a_col_id, my_k );
                int uniform_a_col_is_valid = utils::shfl( a_col_is_valid, my_k );
                Vector_type my_x(0);

                if ( uniform_a_col_id != -1 && lane_id < row_elems_per_warp)
                {
                    my_x = __cachingLoad(&x[N * uniform_a_col_id + lane_id_mod_N]);
                }

                if ( uniform_a_col_id != -1 && uniform_a_col_is_valid && lane_id < row_elems_per_warp)
                {
                    my_x += delta[N * uniform_a_col_id + lane_id_mod_N];
                }

                //my_s_mem[lane_id] = my_x;
#pragma unroll

                for ( int i = 0 ; i < blocks_per_warp ; ++i )
                {
                    // k-th batch of blocks, i-th block. each thread process a column/row of a_it = uniform_a_col_tmp
                    int uniform_a_col_tmp = a_col_begin + k + i, uniform_a_col_it = -1;

                    // check if we are going out of bounds/color
                    if ( uniform_a_col_tmp < a_col_end )
                    {
                        uniform_a_col_it = uniform_a_col_tmp;
                    }

                    if ( HAS_EXTERNAL_DIAG && uniform_a_col_tmp == a_col_end )
                    {
                        uniform_a_col_it = A_diag[a_row_id];
                    }

                    // swipe with the whole warp
                    if (uniform_a_col_it != -1)
                    {
                        int block_inside_id = lane_id;
#pragma unroll

                        for (int j = 0; j < NUM_WARP_ITERS_PER_BLOCK; j++)
                        {
                            Matrix_type my_val(0);

                            if ( uniform_a_col_it != -1 && block_inside_id < NxN)
                            {
                                my_val = A_vals[NxN * uniform_a_col_it + block_inside_id];
                            }

                            my_bmAx_s[block_inside_id] -= my_val * utils::shfl(my_x, N * i + block_inside_id % N); // MOD IS SLOW!
                            block_inside_id += WARP_SIZE;
                        }
                    }
                }
            } // Loop over k
        } // Loop over aColIt

        // Load Einvs.
        Vector_type my_Einv[NUM_WARP_ITERS_PER_BLOCK];
#pragma unroll

        for (int j = 0; j < NUM_WARP_ITERS_PER_BLOCK; j++)
        {
            my_Einv[j] = 0.0;
        }

#pragma unroll

        for (int j = 0; j < NUM_WARP_ITERS_PER_BLOCK; j++)
        {
            if ((WARP_SIZE * j + lane_id) < NxN)
            {
                my_Einv[j] = Einv[NxN * a_row_id + WARP_SIZE * j + lane_id];
            }
        }

        // Reduce bmAx terms.
        {
#pragma unroll

            for ( int i = 0 ; i < N ; ++i )
            {
                if ( lane_id < N )
                {
                    my_bmAx += my_bmAx_s[N * lane_id + i];
                }
            }
        }
        // Update the diagonal term.
        int block_inside_id = lane_id;
#pragma unroll

        for (int j = 0; j < NUM_WARP_ITERS_PER_BLOCK; j++)
        {
            my_bmAx_s[block_inside_id] = my_Einv[j] * utils::shfl(my_bmAx, block_inside_id % N);
            block_inside_id += WARP_SIZE;
        }

        // Reduce bmAx terms.
        {
            my_bmAx = 0.0;
#pragma unroll

            for ( int i = 0 ; i < N ; ++i )
            {
                int idx = N * lane_id + i;

                if ( lane_id < N )
                {
                    my_bmAx += my_bmAx_s[idx];
                }
            }
        }

        // Store the results.
        if ( lane_id < N )
        {
            delta[N * a_row_id + lane_id] = my_bmAx;
        }
    }
}


///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template< typename Matrix_type, typename Vector_type, int CTA_SIZE, int WARP_SIZE, bool ROW_MAJOR, bool HAS_EXTERNAL_DIAG >
__global__
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 700
__launch_bounds__( CTA_SIZE, 12 )
#elif defined(__CUDA_ARCH__)
__launch_bounds__( CTA_SIZE, 12 )
#endif
void DILU_forward_4x4_kernel( const int *__restrict A_rows,
                              const int *__restrict A_cols,
                              const Matrix_type *__restrict A_vals,
                              const int *__restrict A_diag,
                              const Vector_type *x,
                              const Vector_type *b,
                              Vector_type *__restrict delta,
                              const int *__restrict sorted_rows_by_color,
                              const int num_rows_per_color,
                              const int current_color,
                              const int *__restrict row_colors,
                              const Matrix_type *__restrict Einv,
                              const ColoringType boundary_coloring,
                              const int boundary_index )
{
    const int NUM_WARPS_PER_CTA = CTA_SIZE / WARP_SIZE;
    // Number of items per warp.
    const int NUM_ITEMS_PER_WARP = WARP_SIZE / 16;
    // Number of items computer per CTA.
    const int NUM_ITEMS_PER_CTA = NUM_ITEMS_PER_WARP * NUM_WARPS_PER_CTA;
    // Number of items per grid.
    const int NUM_ITEMS_PER_GRID = gridDim.x * NUM_ITEMS_PER_CTA;
    // The coordinates of the thread inside the CTA/warp.
    const int warp_id = utils::warp_id();
    const int lane_id = utils::lane_id();
    // Constants.
    const int lane_id_mod_16 = lane_id % 16;
    // Useful index to compute matrix products.
    const int lane_id_mod_16_div_4 = lane_id_mod_16 / 4;
    const int lane_id_mod_16_mod_4 = lane_id_mod_16 % 4;
    // We to get my data from when I use SHFL.
    const int shfl_offset = lane_id - lane_id_mod_16;
    // Shared memory needed to exchange X and delta.
    __shared__ volatile Vector_type s_mem[CTA_SIZE];
    // Each thread keeps its own pointer to shared memory to avoid some extra computations.
    volatile Vector_type *my_s_mem = &s_mem[threadIdx.x - lane_id_mod_16];
    // Determine which 16 block the threads work with.
    int a_row_it = blockIdx.x * NUM_ITEMS_PER_CTA + threadIdx.x / 16;

    // Iterate over the rows of the matrix. One warp per row.
    for ( ; a_row_it < num_rows_per_color ; a_row_it += NUM_ITEMS_PER_GRID )
    {
        int a_row_id = sorted_rows_by_color[a_row_it];
        // Load one block of B.
        Vector_type my_bmAx(0);

        if ( ROW_MAJOR )
        {
            if ( lane_id_mod_16_mod_4 == 0 )
            {
                my_bmAx = __cachingLoad(&b[4 * a_row_id + lane_id_mod_16_div_4]);
            }
        }
        else
        {
            if ( lane_id_mod_16_div_4 == 0 )
            {
                my_bmAx = b[4 * a_row_id + lane_id_mod_16_mod_4];
            }
        }

        // Don't do anything if X is zero.
        int a_col_begin = A_rows[a_row_id  ];
        int a_col_end   = A_rows[a_row_id + 1];
        // If the diagonal is stored separately, we have a special treatment.
        int a_col_max = a_col_end;

        if ( HAS_EXTERNAL_DIAG )
        {
            ++a_col_max;
        }

        // Each warp load column indices of 32 nonzero blocks
        for ( ; utils::any( a_col_begin < a_col_max ) ; a_col_begin += 16 )
        {
            // Each thread loads a single element. If !is_active, a_col_end == 0.
            int a_col_it = a_col_begin + lane_id_mod_16;
            // Get the ID of the column.
            int a_col_id = -1;

            if ( a_col_it < a_col_end )
            {
                a_col_id = A_cols[a_col_it];
            }

            if ( HAS_EXTERNAL_DIAG && a_col_it == a_col_end )
            {
                a_col_id = a_row_id;
            }

            // Determine if the color is valid.
            int a_col_is_valid = false;
#ifdef AMGX_ILU_COLORING

            if ( a_col_id != -1 && current_color != 0 )
            {
                if ( boundary_coloring == FIRST )
                {
                    a_col_is_valid = a_col_id >= boundary_index;
                }
                else
                {
                    a_col_is_valid = a_col_id < boundary_index && row_colors[a_col_id] < current_color;
                }
            }

#else

            if ( a_col_id != -1 && current_color != 0 )
            {
                a_col_is_valid = row_colors[a_col_id] < current_color;
            }

#endif
            // Count the number of active columns.
            // int vote =  utils::ballot(aColId != -1);
            // The number of iterations.
            // int nCols = max( __popc( vote & 0x0000ffff ), __popc( vote & 0xffff0000 ) );

            // Loop over columns. We compute 8 columns per iteration.
            for ( int k = 0 ; k < 16 ; k += 4 )
            {
                int my_k = k + lane_id_mod_16_div_4;
                // Load N blocks of X.
                int uniform_a_col_id = utils::shfl( a_col_id, shfl_offset + my_k );
                int uniform_a_col_is_valid = utils::shfl( a_col_is_valid, shfl_offset + my_k );
                Vector_type my_x(0);

                if ( uniform_a_col_id != -1 )
                {
                    my_x = __cachingLoad(&x[4 * uniform_a_col_id + lane_id_mod_16_mod_4]);
                }

                if ( uniform_a_col_id != -1 && uniform_a_col_is_valid )
                {
                    my_x += delta[4 * uniform_a_col_id + lane_id_mod_16_mod_4];
                }

                my_s_mem[lane_id_mod_16] = my_x;
                // Load N blocks of A.
#pragma unroll

                for ( int i = 0 ; i < 4 ; ++i )
                {
                    int uniform_a_col_tmp = a_col_begin + k + i, uniform_a_col_it = -1;

                    if ( uniform_a_col_tmp < a_col_end )
                    {
                        uniform_a_col_it = uniform_a_col_tmp;
                    }

                    if ( HAS_EXTERNAL_DIAG && uniform_a_col_tmp == a_col_end )
                    {
                        uniform_a_col_it = A_diag[a_row_id];
                    }

                    Matrix_type my_val(0);

                    if ( uniform_a_col_it != -1 )
                    {
                        my_val = A_vals[16 * uniform_a_col_it + lane_id_mod_16];
                    }

                    if ( ROW_MAJOR )
                    {
                        my_bmAx -= my_val * my_s_mem[4 * i + lane_id_mod_16_mod_4];
                    }
                    else
                    {
                        my_bmAx -= my_val * my_s_mem[4 * i + lane_id_mod_16_div_4];
                    }
                }
            } // Loop over k
        } // Loop over aColIt

        // Load Einvs.
        Vector_type my_Einv = Einv[16 * a_row_id + lane_id_mod_16];
        // Reduce bmAx terms.
        int is_leader = lane_id_mod_16_div_4 == 0;

        if ( ROW_MAJOR )
        {
            is_leader = lane_id_mod_16_mod_4 == 0;
        }

        my_bmAx = reduce_distributed_vectors<4, ROW_MAJOR, WARP_SIZE>( my_bmAx, is_leader );

        // Update the shared terms.
        if ( ROW_MAJOR )
        {
            if ( lane_id_mod_16_mod_4 == 0 )
            {
                my_s_mem[lane_id_mod_16_div_4] = my_bmAx;
            }
        }
        else
        {
            if ( lane_id_mod_16_div_4 == 0 )
            {
                my_s_mem[lane_id_mod_16_mod_4] = my_bmAx;
            }
        }

        // Update the diagonal term.
        if ( ROW_MAJOR )
        {
            my_bmAx = my_Einv * my_s_mem[lane_id_mod_16_mod_4];
        }
        else
        {
            my_bmAx = my_Einv * my_s_mem[lane_id_mod_16_div_4];
        }

        // Reduce bmAx terms.
        my_bmAx = reduce_distributed_vectors<4, ROW_MAJOR, WARP_SIZE>( my_bmAx, is_leader );

        // Store the results.
        if ( ROW_MAJOR )
        {
            if ( lane_id_mod_16_mod_4 == 0 )
            {
                delta[4 * a_row_id + lane_id_mod_16_div_4] = my_bmAx;
            }
        }
        else
        {
            if ( lane_id_mod_16_div_4 == 0 )
            {
                delta[4 * a_row_id + lane_id_mod_16_mod_4] = my_bmAx;
            }
        }
    }
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template< typename Matrix_type, typename Vector_type, int CTA_SIZE, bool HAS_EXTERNAL_DIAG >
__global__
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 700
__launch_bounds__( CTA_SIZE, 12 )
#elif defined(__CUDA_ARCH__)
__launch_bounds__( CTA_SIZE, 12 )
#endif
void DILU_forward_4x4_kernel_row_major_vec4( const int *__restrict A_rows,
        const int *__restrict A_cols,
        const Matrix_type *__restrict A_vals,
        const int *__restrict A_diag,
        const Vector_type *x,
        const Vector_type *b,
        Vector_type *__restrict delta,
        const int *__restrict sorted_rows_by_color,
        const int num_rows_per_color,
        const int current_color,
        const int *__restrict row_colors,
        const Matrix_type *Einv,
        const ColoringType boundary_coloring,
        const int boundary_index )
{
    // Number of half warps per CTA.
    const int NUM_HALF_WARPS = CTA_SIZE / 16;
    // Coordinates of the thread.
    const int warp_id = utils::warp_id();
    const int lane_id = utils::lane_id();
    // Coordinates of the thread in the CTA.
    const int thread_id_div_16 = threadIdx.x / 16;
    const int thread_id_mod_16 = threadIdx.x % 16;
    // Useful constants.
    const int thread_id_mod_16_div_4 = thread_id_mod_16 / 4;
    const int thread_id_mod_16_mod_4 = thread_id_mod_16 % 4;
    const int shfl_offset = 16 * (lane_id / 16);
    // Shared memory needed to exchange X and delta.
    __shared__ volatile Vector_type s_mem[CTA_SIZE];
    // Each thread keeps its own pointer to shared memory to avoid some extra computations.
    volatile Vector_type *my_s_mem = &s_mem[16 * thread_id_div_16];
    // The iterator over rows.
    int a_row_it = blockIdx.x * NUM_HALF_WARPS + thread_id_div_16;

    // Iterate over the rows of the matrix. One warp per row.
    for ( ; a_row_it < num_rows_per_color ; a_row_it += gridDim.x * NUM_HALF_WARPS )
    {
        unsigned int active_mask = utils::activemask();
        int a_row_id = sorted_rows_by_color[a_row_it];
        // Load one block of B.
        Vector_type my_bmAx(0);

        if ( thread_id_mod_16_div_4 == 0 )
        {
            my_bmAx = __cachingLoad(&b[4 * a_row_id + thread_id_mod_16_mod_4]);
        }

        // The range of the row.
        int a_col_begin = A_rows[a_row_id  ];
        int a_col_end   = A_rows[a_row_id + 1];
        // If it has an external diagonal, we need one more item to put the diag.
        int a_col_max = a_col_end;

        if ( HAS_EXTERNAL_DIAG )
        {
            ++a_col_max;
        }

        // Each warp load column indices of 32 nonzero blocks
        for ( ; a_col_begin < a_col_max ; a_col_begin += 16 )
        {
            unsigned int active_mask_inner = utils::activemask();
            int a_col_it = a_col_begin + thread_id_mod_16;
            // Get the ID of the column.
            int a_col_id = -1;

            if ( a_col_it < a_col_end )
            {
                a_col_id = __cachingLoad(&A_cols[a_col_it]);
            }

            if ( HAS_EXTERNAL_DIAG && a_col_it == a_col_end )
            {
                a_col_id = a_row_id;
            }

            // Determine if the color is valid.
            int a_col_is_valid = false;
#ifdef AMGX_ILU_COLORING

            if ( a_col_id != -1 && current_color != 0 )
            {
                if ( boundary_coloring == FIRST )
                {
                    a_col_is_valid = a_col_id >= boundary_index;
                }
                else
                {
                    a_col_is_valid = a_col_id < boundary_index && __cachingLoad(&row_colors[a_col_id]) < current_color;
                }
            }

#else

            if ( a_col_id != -1 && current_color != 0 )
            {
                a_col_is_valid = row_colors[a_col_id] < current_color;
            }

#endif

            // Loop over columns. We compute 8 columns per iteration.
            for ( int k = 0 ; k < 16 ; k += 4 )
            {
                int my_k = k + thread_id_mod_16_div_4;
                // Load 8 blocks of X.
                int uniform_a_col_id = utils::shfl( a_col_id, shfl_offset + my_k, warpSize, active_mask_inner );
                int uniform_a_col_is_valid = utils::shfl( a_col_is_valid, shfl_offset + my_k, warpSize, active_mask_inner );
                Vector_type my_x(0);

                if ( uniform_a_col_id != -1 )
                {
                    my_x = __cachingLoad(&x[4 * uniform_a_col_id + thread_id_mod_16_mod_4]);
                }

                if ( uniform_a_col_id != -1 && uniform_a_col_is_valid )
                {
                    my_x += delta[4 * uniform_a_col_id + thread_id_mod_16_mod_4];
                }

                my_s_mem[thread_id_mod_16] = my_x;
                int uniform_a_col_tmp = a_col_begin + my_k, uniform_a_col_it = -1;

                if ( uniform_a_col_tmp < a_col_end )
                {
                    uniform_a_col_it = uniform_a_col_tmp;
                }

                if ( HAS_EXTERNAL_DIAG && uniform_a_col_tmp == a_col_end )
                {
                    uniform_a_col_it = A_diag[a_row_id];
                }

                Matrix_type my_vals[4] = { Matrix_type(0) };

                if ( uniform_a_col_it != -1 )
                {
                    utils::load_vec4( my_vals, &A_vals[16 * uniform_a_col_it + 4 * thread_id_mod_16_mod_4] );
                }

                my_bmAx -= my_vals[0] * my_s_mem[4 * thread_id_mod_16_div_4 + 0];
                my_bmAx -= my_vals[1] * my_s_mem[4 * thread_id_mod_16_div_4 + 1];
                my_bmAx -= my_vals[2] * my_s_mem[4 * thread_id_mod_16_div_4 + 2];
                my_bmAx -= my_vals[3] * my_s_mem[4 * thread_id_mod_16_div_4 + 3];
            }
        }

        // Load Einvs.
        Matrix_type my_Einv = Einv[16 * a_row_id + thread_id_mod_16];
        // Reduce bmAx terms.
        my_bmAx += utils::shfl_xor( my_bmAx, 4, warpSize, active_mask );
        my_bmAx += utils::shfl_xor( my_bmAx, 8, warpSize, active_mask );

        // Update the shared terms.
        if ( thread_id_mod_16_div_4 == 0 )
        {
            my_s_mem[thread_id_mod_16_mod_4] = my_bmAx;
        }

        // Update the diagonal term.
        my_bmAx = my_Einv * my_s_mem[thread_id_mod_16_mod_4];
        // Reduce bmAx terms.
        my_bmAx += utils::shfl_xor( my_bmAx, 1, warpSize, active_mask );
        my_bmAx += utils::shfl_xor( my_bmAx, 2, warpSize, active_mask );

        // Store the results.
        if ( thread_id_mod_16_mod_4 == 0 )
        {
            delta[4 * a_row_id + thread_id_mod_16_div_4] = my_bmAx;
        }
    }
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template< typename Matrix_type, typename Vector_type, int NUM_THREADS_PER_ROW, int CTA_SIZE, int WARP_SIZE, bool HAS_EXTERNAL_DIAG >
__global__
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 700
__launch_bounds__( CTA_SIZE, 12 )
#elif defined(__CUDA_ARCH__)
__launch_bounds__( CTA_SIZE, 12 )
#endif
void DILU_forward_1x1_kernel( const int *__restrict A_rows,
                              const int *__restrict A_cols,
                              const Matrix_type *__restrict A_vals,
                              const int *__restrict A_diag,
                              const Vector_type *x,
                              const Vector_type *b,
                              Vector_type *__restrict delta,
                              const int *__restrict sorted_rows_by_color,
                              const int num_rows_per_color,
                              const int current_color,
                              const int *__restrict row_colors,
                              const Matrix_type *__restrict Einv,
                              const ColoringType boundary_coloring,
                              const int boundary_index )
{
    // Number of items per CTA.
    const int NUM_ROWS_PER_CTA = CTA_SIZE / NUM_THREADS_PER_ROW;
    // Number of items per grid.
    const int NUM_ROWS_PER_GRID = gridDim.x * NUM_ROWS_PER_CTA;
    // The coordinates of the thread inside the CTA/warp.
    const int warp_id = utils::warp_id();
    const int lane_id = utils::lane_id();
    // Constants.
    const int lane_id_mod_NTPR = lane_id % NUM_THREADS_PER_ROW;
    // Determine which NxN block the threads work with.
    int a_row_it = blockIdx.x * NUM_ROWS_PER_CTA + (threadIdx.x / NUM_THREADS_PER_ROW);

    // Iterate over the rows of the matrix. One warp per row.
    for ( ; a_row_it < num_rows_per_color ; a_row_it += NUM_ROWS_PER_GRID )
    {
        int a_row_id = sorted_rows_by_color[a_row_it];
        // Load one block of B.
        Vector_type my_bmAx = amgx::types::util<Vector_type>::get_zero();

        if ( lane_id_mod_NTPR == 0 )
        {
            my_bmAx = __cachingLoad(&b[a_row_id]);
        }

        // If it has an external diag.
        if ( HAS_EXTERNAL_DIAG && lane_id_mod_NTPR == 0 )
        {
            my_bmAx -= A_vals[A_diag[a_row_id]] * x[a_row_id];
        }

        // Don't do anything if X is zero.
        int a_col_it  = A_rows[a_row_id  ];
        int a_col_end = A_rows[a_row_id + 1];

        // If the diagonal is stored separately, we have a special treatment.
        //if( HAS_EXTERNAL_DIAG )
        //  ++a_col_end;

        // Each warp load column indices of 32 nonzero blocks
        for ( a_col_it += lane_id_mod_NTPR ; utils::any( a_col_it < a_col_end ) ; a_col_it += NUM_THREADS_PER_ROW )
        {
            // Get the ID of the column.
            int a_col_id = -1;

            if ( a_col_it < a_col_end )
            {
                a_col_id = A_cols[a_col_it];
            }

            // Ignore the diagonal element since its color is smaller, and been accounted for above
            if (HAS_EXTERNAL_DIAG && a_col_id == a_row_id)
            {
                a_col_id = -1;
            }

            // Load x.
            Vector_type my_x(0);

            if ( a_col_id != -1 )
            {
                my_x = __cachingLoad(&x[a_col_id]);
            }

            // Is it really a valid column (due to coloring).
            int valid = false;
#ifdef AMGX_ILU_COLORING

            if ( a_col_id != -1 && current_color != 0 )
            {
                if ( boundary_coloring == FIRST )
                {
                    valid = a_col_id >= boundary_index;
                }
                else
                {
                    valid = a_col_id < boundary_index && row_colors[a_col_id] < current_color;
                }
            }

#else

            if ( a_col_id != -1 && current_color != 0 )
            {
                valid = row_colors[a_col_tmp] < current_color;
            }

#endif

            // Load my x value.
            if ( valid )
            {
                my_x += delta[a_col_id];
            }

            // Load my item from A.
            Matrix_type my_val(0);

            if ( a_col_it < a_col_end )
            {
                my_val = A_vals[a_col_it];
            }

            // Update bmAx.
            my_bmAx -= my_val * my_x;
        }

        // Reduce bmAx terms.
#pragma unroll

        for ( int mask = NUM_THREADS_PER_ROW / 2 ; mask > 0 ; mask >>= 1 )
        {
            my_bmAx += utils::shfl_xor( my_bmAx, mask );
        }

        // Store the results.
        if ( lane_id_mod_NTPR == 0 )
        {
            delta[a_row_id] = Einv[a_row_id] * my_bmAx;
        }
    }
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template< typename Matrix_type, typename Vector_type, typename WeightType, int N, int CTA_SIZE, int WARP_SIZE, bool ROW_MAJOR >
__global__
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 700
__launch_bounds__( CTA_SIZE, 12 )
#elif defined(__CUDA_ARCH__)
__launch_bounds__( CTA_SIZE, 12 )
#endif
void DILU_backward_NxN_kernel( const int *__restrict A_rows,
                               const int *__restrict A_cols,
                               const Matrix_type *__restrict A_vals,
                               Vector_type *__restrict x,
                               const WeightType weight,
                               const int *__restrict sorted_rows_by_color,
                               const int *__restrict row_colors,
                               const Matrix_type *__restrict Einv,
                               const Vector_type *delta,
                               Vector_type *__restrict Delta,
                               const int num_rows_per_color,
                               const int current_color,
                               const ColoringType boundary_coloring,
                               const int boundary_index )
{
    const int NUM_WARPS_PER_CTA = CTA_SIZE / WARP_SIZE;
    // Squared N.
    const int NxN = N * N;
    // Number of items per warp.
    const int NUM_ITEMS_PER_WARP = WARP_SIZE / NxN;
    // Number of items computer per CTA.
    const int NUM_ITEMS_PER_CTA = NUM_ITEMS_PER_WARP * NUM_WARPS_PER_CTA;
    // Number of items per grid.
    const int NUM_ITEMS_PER_GRID = gridDim.x * NUM_ITEMS_PER_CTA;
    // The coordinates of the thread inside the CTA/warp.
    const int warp_id = utils::warp_id();
    const int lane_id = utils::lane_id();
    // Constants.
    const int lane_id_div_NxN = lane_id / NxN;
    const int lane_id_mod_NxN = lane_id % NxN;
    // Useful index to compute matrix products.
    const int lane_id_mod_NxN_div_N = lane_id_mod_NxN / N;
    const int lane_id_mod_NxN_mod_N = lane_id_mod_NxN % N;
    // We to get my data from when I use SHFL.
    const int shfl_offset = lane_id - lane_id_mod_NxN;
    // Shared memory needed to exchange X and delta.
    __shared__ volatile Vector_type s_mem[CTA_SIZE];
    // Each thread keeps its own pointer to shared memory to avoid some extra computations.
    volatile Vector_type *my_s_mem = &s_mem[threadIdx.x - lane_id_mod_NxN];
    // Is the thread active? For example, for 5x5 only the first 25 threads are active per warp.
    // At compile time, the compiler will see is_active == true for 2x2 (since NxN & (NxN-1) evals
    // to false ; that's the common trick to determine if a number is a power of 2).
    int is_active = true;

    if ( NxN & (NxN - 1) )
    {
        is_active = lane_id_div_NxN < NUM_ITEMS_PER_WARP;
    }

    // Determine which NxN block the threads work with.
    int a_row_it = num_rows_per_color;

    if ( is_active )
    {
        a_row_it = blockIdx.x * NUM_ITEMS_PER_CTA + warp_id * NUM_ITEMS_PER_WARP + lane_id_div_NxN;
    }

    // Iterate over the rows of the matrix. One warp per row.
    for ( ; a_row_it < num_rows_per_color ; a_row_it += NUM_ITEMS_PER_GRID )
    {
        int a_row_id = sorted_rows_by_color[a_row_it];
        // Load one block of B.
        Vector_type my_delta(0);
        // Don't do anything if X is zero.
        int a_col_begin = A_rows[a_row_id  ];
        int a_col_end   = A_rows[a_row_id + 1];

        // Each warp load column indices of 32 nonzero blocks
        for ( ; utils::any( a_col_begin < a_col_end ) ; a_col_begin += NxN )
        {
            // Each thread loads a single element. If !is_active, a_col_end == 0.
            int a_col_it = a_col_begin + lane_id_mod_NxN;
            // Get the ID of the column.
            int a_col_tmp = -1, a_col_id = -1;

            if ( a_col_it < a_col_end )
            {
                a_col_tmp = A_cols[a_col_it];
            }

            // Make sure the column is interesting.
#ifdef AMGX_ILU_COLORING
            int valid = false;

            if ( a_col_tmp != -1 && current_color != 0 )
            {
                if ( boundary_coloring == LAST )
                {
                    valid = a_col_tmp >= boundary_index;
                }
                else
                {
                    valid = a_col_tmp < boundary_index && row_colors[a_col_tmp] > current_color;
                }
            }

#else
            int valid = false;

            if ( a_col_tmp != -1 && row_colors[a_col_tmp] > current_color )
            {
                valid = true;
            }

#endif

            // Set the column id.
            if ( valid )
            {
                a_col_id = a_col_tmp;
            }

            // Count the number of active columns.
            // int vote =  utils::ballot(aColId != -1);
            // The number of iterations.
            // int nCols = max( __popc( vote & 0x0000ffff ), __popc( vote & 0xffff0000 ) );

            // Loop over columns. We compute 8 columns per iteration.
            for ( int k = 0 ; k < NxN ; k += N )
            {
                int my_k = k + lane_id_mod_NxN_div_N;
                // Load N blocks of X.
                int uniform_a_col_id = utils::shfl( a_col_id, shfl_offset + my_k );
                Vector_type my_x(0);

                if ( uniform_a_col_id != -1 )
                {
                    my_x = Delta[N * uniform_a_col_id + lane_id_mod_NxN_mod_N];
                }

                my_s_mem[lane_id_mod_NxN] = my_x;
                // Load N blocks of A.
#pragma unroll

                for ( int i = 0 ; i < N ; ++i )
                {
                    //if( uniform_a_col_id == -1 )
                    //  break;
                    int uniform_a_col_tmp = a_col_begin + k + i, uniform_a_col_it = -1;

                    if ( uniform_a_col_tmp < a_col_end )
                    {
                        uniform_a_col_it = uniform_a_col_tmp;
                    }

                    Matrix_type my_val(0);

                    if ( uniform_a_col_it != -1 )
                    {
                        my_val = A_vals[NxN * uniform_a_col_it + lane_id_mod_NxN];
                    }

                    if ( ROW_MAJOR )
                    {
                        my_delta += my_val * my_s_mem[N * i + lane_id_mod_NxN_mod_N];
                    }
                    else
                    {
                        my_delta += my_val * my_s_mem[N * i + lane_id_mod_NxN_div_N];
                    }
                }
            } // Loop over k
        } // Loop over aColIt

        // Load Einvs.
        Matrix_type my_Einv = Einv[NxN * a_row_id + lane_id_mod_NxN];
        // Reduce bmAx terms.
        int is_leader = lane_id_mod_NxN_div_N == 0;

        if ( ROW_MAJOR )
        {
            is_leader = lane_id_mod_NxN_mod_N == 0;
        }

        my_delta = reduce_distributed_vectors<N, ROW_MAJOR, WARP_SIZE>( my_delta, is_leader );

        // Update the shared terms.
        if ( ROW_MAJOR )
        {
            if ( lane_id_mod_NxN_mod_N == 0 )
            {
                my_s_mem[lane_id_mod_NxN_div_N] = my_delta;
            }
        }
        else
        {
            if ( lane_id_mod_NxN_div_N == 0 )
            {
                my_s_mem[lane_id_mod_NxN_mod_N] = my_delta;
            }
        }

        // Update the diagonal term.
        if ( ROW_MAJOR )
        {
            my_delta = my_Einv * my_s_mem[lane_id_mod_NxN_mod_N];
        }
        else
        {
            my_delta = my_Einv * my_s_mem[lane_id_mod_NxN_div_N];
        }

        // Reduce bmAx terms.
        my_delta = reduce_distributed_vectors<N, ROW_MAJOR, WARP_SIZE>( my_delta, is_leader );

        // Store the results.
        if ( ROW_MAJOR )
        {
            const int offset = N * a_row_id + lane_id_mod_NxN_div_N;
            Vector_type my_b(0), my_x(0);

            if ( lane_id_mod_NxN_mod_N == 0 )
            {
                my_b = __cachingLoad(&delta[offset]);
                my_x = x    [offset];
            }

            my_delta = my_b - my_delta;

            if ( lane_id_mod_NxN_mod_N == 0 )
            {
                x    [offset] = my_x + weight * my_delta;
                Delta[offset] = my_delta;
            }
        }
        else
        {
            const int offset = N * a_row_id + lane_id_mod_NxN_mod_N;
            Vector_type my_b(0), my_x(0);

            if ( lane_id_mod_NxN_div_N == 0 )
            {
                my_b = __cachingLoad(&delta[offset]);
                my_x = x    [offset];
            }

            my_delta = my_b - my_delta;

            if ( lane_id_mod_NxN_div_N == 0 )
            {
                x    [offset] = my_x + weight * my_delta;
                Delta[offset] = my_delta;
            }
        }
    }
}


///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template< typename Matrix_type, typename Vector_type, typename WeightType, int N, int CTA_SIZE, int WARP_SIZE, bool ROW_MAJOR, int NUM_WARP_ITERS_PER_BLOCK >
__global__
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 700
__launch_bounds__( CTA_SIZE, 12 )
#elif defined(__CUDA_ARCH__)
__launch_bounds__( CTA_SIZE, 12 )
#endif
void DILU_backward_NxN_kernel_large( const int *__restrict A_rows,
                                     const int *__restrict A_cols,
                                     const Matrix_type *__restrict A_vals,
                                     Vector_type *__restrict x,
                                     const WeightType weight,
                                     const int *__restrict sorted_rows_by_color,
                                     const int *__restrict row_colors,
                                     const Matrix_type *__restrict Einv,
                                     const Vector_type *delta,
                                     Vector_type *__restrict Delta,
                                     const int num_rows_per_color,
                                     const int current_color,
                                     const ColoringType boundary_coloring,
                                     const int boundary_index )
{
    const int NUM_WARPS_PER_CTA = CTA_SIZE / WARP_SIZE;
    // Squared N.
    const int NxN = N * N;
    // Number of items computer per CTA.
    const int NUM_ITEMS_PER_CTA = NUM_WARPS_PER_CTA;
    // Number of items per grid.
    const int NUM_ITEMS_PER_GRID = gridDim.x * NUM_WARPS_PER_CTA;
    // The coordinates of the thread inside the CTA/warp.
    const int warp_id = utils::warp_id();
    const int lane_id = utils::lane_id();
    // Constants.
    const int lane_id_div_N = lane_id / N;
    const int lane_id_mod_N = lane_id % N; // id of a lane inside the block
    const int blocks_per_warp = WARP_SIZE / N; // we process this cols per warp per row
    const int row_elems_per_warp  = blocks_per_warp * N;
    // Shared to store t_delta
    __shared__ volatile Vector_type delta_s[CTA_SIZE * NUM_WARP_ITERS_PER_BLOCK];
    volatile Vector_type *my_delta_s = &delta_s[warp_id * NUM_WARP_ITERS_PER_BLOCK * WARP_SIZE];
    // Determine which NxN block the threads work with.
    int a_row_it = blockIdx.x * NUM_ITEMS_PER_CTA + warp_id;

    // Iterate over the rows of the matrix. One warp per row.
    for ( ; a_row_it < num_rows_per_color ; a_row_it += NUM_ITEMS_PER_GRID )
    {
        int a_row_id = sorted_rows_by_color[a_row_it];
        // Accumulator
        Vector_type my_delta(0);
        //Vector_type mAx[NUM_WARP_ITERS_PER_BLOCK];
#pragma unroll

        for (int i = 0; i < NUM_WARP_ITERS_PER_BLOCK; i++)
        {
            my_delta_s[WARP_SIZE * i + lane_id] = 0.0;
        }

        // Don't do anything if X is zero.
        int a_col_begin = A_rows[a_row_id  ];
        int a_col_end   = A_rows[a_row_id + 1];

        // Each warp load column indices of 32 nonzero blocks
        for ( ; utils::any( a_col_begin < a_col_end ) ; a_col_begin += WARP_SIZE )
        {
            // Each thread loads a single element. If !is_active, a_col_end == 0.
            int a_col_it = a_col_begin + lane_id;
            // Get the ID of the column.
            int a_col_tmp = -1, a_col_id = -1;

            if ( a_col_it < a_col_end )
            {
                a_col_tmp = A_cols[a_col_it];
            }

            // Make sure the column is interesting.
#ifdef AMGX_ILU_COLORING
            int valid = false;

            if ( a_col_tmp != -1 && current_color != 0 )
            {
                if ( boundary_coloring == LAST )
                {
                    valid = a_col_tmp >= boundary_index;
                }
                else
                {
                    valid = a_col_tmp < boundary_index && row_colors[a_col_tmp] > current_color;
                }
            }

#else
            int valid = false;

            if ( a_col_tmp != -1 && row_colors[a_col_tmp] > current_color )
            {
                valid = true;
            }

#endif

            // Set the column id.
            if ( valid )
            {
                a_col_id = a_col_tmp;
            }

            // Loop over columns. We compute blocks_per_warp columns per iteration.
            for ( int k = 0 ; k < WARP_SIZE ; k += blocks_per_warp )
            {
                // id of the processed block by this thread
                int my_k = k + lane_id_div_N;
                // Load N blocks of X (if valid)
                int uniform_a_col_id = utils::shfl( a_col_id, my_k );
                Vector_type my_x(0);

                if ( uniform_a_col_id != -1 && lane_id < row_elems_per_warp)
                {
                    my_x = Delta[N * uniform_a_col_id + lane_id_mod_N];
                }

                // Load blocks of A.
                // for each block in a batch
#pragma unroll

                for ( int i = 0 ; i < blocks_per_warp ; ++i )
                {
                    // k-th batch of blocks, i-th block. each thread process a column/row of a_it = uniform_a_col_tmp
                    int uniform_a_col_tmp = a_col_begin + k + i, uniform_a_col_it = -1;

                    // check if we are going out of bounds/color
                    if ( uniform_a_col_tmp < a_col_end )
                    {
                        uniform_a_col_it = uniform_a_col_tmp;
                    }

                    // swipe with the whole warp
                    if (uniform_a_col_it != -1)
                    {
                        int block_inside_id = lane_id;
#pragma unroll

                        for (int j = 0; j < NUM_WARP_ITERS_PER_BLOCK; j++)
                        {
                            Matrix_type my_val(0);

                            if ( uniform_a_col_it != -1 && block_inside_id < NxN)
                            {
                                my_val = A_vals[NxN * uniform_a_col_it + block_inside_id];
                            }

                            my_delta_s[block_inside_id] -= my_val * utils::shfl(my_x, N * i + block_inside_id % N); //my_s_mem[N*i + block_inside_id % N]; // MOD IS SLOW!
                            block_inside_id += WARP_SIZE;
                        }
                    }
                }
            } // Loop over k
        } // Loop over aColIt

        // Load Einvs.
        Vector_type my_Einv[NUM_WARP_ITERS_PER_BLOCK];
#pragma unroll

        for (int j = 0; j < NUM_WARP_ITERS_PER_BLOCK; j++)
        {
            my_Einv[j] = 0.0;
        }

#pragma unroll

        for (int j = 0; j < NUM_WARP_ITERS_PER_BLOCK; j++)
        {
            if ((WARP_SIZE * j + lane_id) < NxN)
            {
                my_Einv[j] = Einv[NxN * a_row_id + WARP_SIZE * j + lane_id];
            }
        }

        // Reduce bmAx terms.
        {
#pragma unroll

            for ( int i = 0 ; i < N ; ++i )
            {
                if ( lane_id < N )
                {
                    my_delta += my_delta_s[N * lane_id + i];
                }
            }
        }

        // Update the diagonal term.
        if ( ROW_MAJOR )
        {
            int block_inside_id = lane_id;
#pragma unroll

            for (int j = 0; j < NUM_WARP_ITERS_PER_BLOCK; j++)
            {
                my_delta_s[block_inside_id] = my_Einv[j] * utils::shfl(my_delta, block_inside_id % N);
                block_inside_id += WARP_SIZE;
            }
        }

        // Reduce bmAx terms.
        {
            my_delta = 0.0;
#pragma unroll

            for ( int i = 0 ; i < N ; ++i )
            {
                if ( lane_id < N )
                {
                    my_delta += my_delta_s[N * lane_id + i];
                }
            }
        }

        // Store the results.
        if ( ROW_MAJOR )
        {
            const int offset = N * a_row_id + lane_id;
            Vector_type my_b(0), my_x(0);

            if ( lane_id < N )
            {
                my_b = __cachingLoad(&delta[offset]);
                my_x = x    [offset];
            }

            my_delta = my_b - my_delta;

            if ( lane_id < N )
            {
                x    [offset] = my_x + weight * my_delta;
                Delta[offset] = my_delta;
            }
        }
    }
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template< typename IndexType, typename ValueTypeA, typename ValueTypeB, typename WeightType, int CTA_SIZE, bool ROW_MAJOR >
__global__
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 700
__launch_bounds__( CTA_SIZE, 16 )
#elif defined(__CUDA_ARCH__)
__launch_bounds__( CTA_SIZE, 16 )
#endif
void DILU_backward_4x4_kernel( const IndexType *row_offsets,
                               const IndexType *column_indices,
                               const ValueTypeA *nonzero_values,
                               ValueTypeB *x,
                               const WeightType weight,
                               const int *sorted_rows_by_color,
                               const int *__restrict row_colors,
                               const ValueTypeA *Einv,
                               const ValueTypeB *delta,
                               ValueTypeB *Delta,
                               const int num_rows_per_color,
                               const int current_color,
                               const ColoringType boundary_coloring,
                               const IndexType boundary_index)
{
    const int nHalfWarps = CTA_SIZE / 16; // Number of half warps per CTA.
    const int laneId = utils::lane_id();
    const int halfWarpId = threadIdx.x / 16;
    const int halfLaneId = threadIdx.x % 16;
    const int halfLaneId_div_4 = halfLaneId / 4;
    const int halfLaneId_mod_4 = halfLaneId % 4;
    const int upperHalf = 16 * (laneId / 16);
    // Shared memory needed to exchange X and delta.
    __shared__ volatile ValueTypeB s_mem[CTA_SIZE];
    // Each thread keeps its own pointer to shared memory to avoid some extra computations.
    volatile ValueTypeB *my_s_mem = &s_mem[16 * halfWarpId];

    // Iterate over the rows of the matrix. One warp per two rows.
    for ( int aRowIt = blockIdx.x * nHalfWarps + halfWarpId ; aRowIt < num_rows_per_color ; aRowIt += gridDim.x * nHalfWarps )
    {
        int aRowId = sorted_rows_by_color[aRowIt];
        // Load one block of B.
        ValueTypeB my_delta(0);
        // The range of the rows.
        int aColBegin = row_offsets[aRowId  ];
        int aColEnd   = row_offsets[aRowId + 1];

        // Each warp load column indices of 16 nonzero blocks
        for ( ; aColBegin < aColEnd ; aColBegin += 16 )
        {
            int aColIt = aColBegin + halfLaneId;
            // Get the ID of the column.
            int aColTmp = -1, aColId = -1;

            if ( aColIt < aColEnd )
            {
                aColTmp = column_indices[aColIt];
            }

#ifdef AMGX_ILU_COLORING
            bool valid = (((aColTmp < boundary_index || boundary_coloring == SYNC_COLORS) && (row_colors[aColTmp] > current_color)) || (aColTmp >= boundary_index && boundary_coloring == LAST));

            if ( aColTmp != -1 && valid )
            {
                aColId = aColTmp;
            }

#else

            if ( aColTmp != -1 && row_colors[aColTmp] > current_color )
            {
                aColId = aColTmp;
            }

#endif
            for ( int k = 0 ; k < 16 ; k += 4 )
            {
                int my_k = k + halfLaneId_div_4;
                // Exchange column indices.
                int waColId = utils::shfl( aColId, upperHalf + my_k );
                // Load 8 blocks of X if needed.
                ValueTypeB my_x(0);

                if ( waColId != -1 )
                {
                    my_x = Delta[4 * waColId + halfLaneId_mod_4];
                }

                my_s_mem[halfLaneId] = my_x;
                // Load 8 blocks of A.
#pragma unroll

                for ( int i = 0 ; i < 4 ; ++i )
                {
                    const int k_i = k + i;
                    int w_aColTmp = aColBegin + k_i, w_aColIt = -1;
                    if ( utils::shfl( aColId, upperHalf + k_i ) != -1 && w_aColTmp < aColEnd )
                        w_aColIt = w_aColTmp;

                    ValueTypeA my_val(0);

                    if ( w_aColIt != -1 )
                    {
                        my_val = nonzero_values[16 * w_aColIt + halfLaneId];
                    }

                    if ( ROW_MAJOR )
                    {
                        my_delta += my_val * my_s_mem[4 * i + halfLaneId_mod_4];
                    }
                    else
                    {
                        my_delta += my_val * my_s_mem[4 * i + halfLaneId_div_4];
                    }
                }
            } // Loop over k
        } // Loop over aColIt

        // Load EINV values.
        ValueTypeA my_Einv = Einv[16 * aRowId + halfLaneId];
        // Reduce delta terms.
        if ( ROW_MAJOR )
        {
            my_delta += utils::shfl_xor( my_delta, 1 );
            my_delta += utils::shfl_xor( my_delta, 2 );
        }
        else
        {
            my_delta += utils::shfl_xor( my_delta, 4 );
            my_delta += utils::shfl_xor( my_delta, 8 );
        }

        // Update the shared terms.
        if ( ROW_MAJOR )
        {
            if ( halfLaneId_mod_4 == 0 )
            {
                my_s_mem[halfLaneId_div_4] = my_delta;
            }
        }
        else
        {
            if ( halfLaneId_div_4 == 0 )
            {
                my_s_mem[halfLaneId_mod_4] = my_delta;
            }
        }

        // Update the diagonal term.
        if ( ROW_MAJOR )
        {
            my_delta = my_Einv * my_s_mem[halfLaneId_mod_4];
        }
        else
        {
            my_delta = my_Einv * my_s_mem[halfLaneId_div_4];
        }

        // Regroup results.
        if ( ROW_MAJOR )
        {
            my_delta += utils::shfl_xor( my_delta, 1 );
            my_delta += utils::shfl_xor( my_delta, 2 );
        }
        else
        {
            my_delta += utils::shfl_xor( my_delta, 4 );
            my_delta += utils::shfl_xor( my_delta, 8 );
        }

        // Store the results.
        if ( ROW_MAJOR )
        {
            int offset = 4 * aRowId + halfLaneId_div_4;
            ValueTypeB my_b(0), my_x(0);

            if ( halfLaneId_mod_4 == 0 )
            {
                my_b = __cachingLoad(&delta[offset]);
                my_x = x[offset];
            }

            my_delta = my_b - my_delta;

            if ( halfLaneId_mod_4 == 0 )
            {
                x[offset] = my_x + weight * my_delta;
                Delta[offset] = my_delta;
            }
        }
        else
        {
            int offset = 4 * aRowId + halfLaneId_mod_4;
            ValueTypeB my_b(0), my_x(0);

            if ( halfLaneId_div_4 == 0 )
            {
                my_b = __cachingLoad(&delta[offset]);
                my_x = x[offset];
            }

            my_delta = my_b - my_delta;

            if ( halfLaneId_div_4 == 0 )
            {
                x[offset] = my_x + weight * my_delta;
                Delta[offset] = my_delta;
            }
        }
    }
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template< typename Matrix_type, typename Vector_type, typename WeightType, int CTA_SIZE >
__global__
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 700
__launch_bounds__( CTA_SIZE, 16 )
#elif defined(__CUDA_ARCH__)
__launch_bounds__( CTA_SIZE, 16 )
#endif
void DILU_backward_4x4_kernel_row_major_vec4( const int *__restrict A_rows,
        const int *__restrict A_cols,
        const Matrix_type *__restrict A_vals,
        Vector_type *__restrict x,
        const WeightType weight,
        const int *__restrict sorted_rows_by_color,
        const int *__restrict row_colors,
        const Matrix_type *__restrict Einv,
        const Vector_type *delta,
        Vector_type *__restrict Delta,
        const int num_rows_per_color,
        const int current_color,
        const ColoringType boundary_coloring,
        const int boundary_index )
{
    // Number of half warps per CTA.
    const int NUM_HALF_WARPS = CTA_SIZE / 16;
    // Coordinates of the thread.
    const int warp_id = utils::warp_id();
    const int lane_id = utils::lane_id();
    // Coordinates of the thread in the CTA.
    const int thread_id_div_16 = threadIdx.x / 16;
    const int thread_id_mod_16 = threadIdx.x % 16;
    // Useful constants.
    const int thread_id_mod_16_div_4 = thread_id_mod_16 / 4;
    const int thread_id_mod_16_mod_4 = thread_id_mod_16 % 4;
    const int shfl_offset = 16 * (lane_id / 16);
    // Shared memory needed to exchange X and delta.
    __shared__ volatile Vector_type s_mem[CTA_SIZE];
    // Each thread keeps its own pointer to shared memory to avoid some extra computations.
    volatile Vector_type *my_s_mem = &s_mem[16 * thread_id_div_16];
    // The iterator over rows.
    int a_row_it = blockIdx.x * NUM_HALF_WARPS + thread_id_div_16;

    // Iterate over the rows of the matrix. One warp per row.
    for ( ; a_row_it < num_rows_per_color ; a_row_it += gridDim.x * NUM_HALF_WARPS )
    {
        unsigned int active_mask = utils::activemask();
        int a_row_id = sorted_rows_by_color[a_row_it];
        // Load one block of B.
        Vector_type my_delta(0);
        // Don't do anything if X is zero.
        int a_col_begin = A_rows[a_row_id  ];
        int a_col_end   = A_rows[a_row_id + 1];

        // Each warp load column indices of 32 nonzero blocks
        for ( ; a_col_begin < a_col_end ; a_col_begin += 16 )
        {
            unsigned int active_mask_inner = utils::activemask();
            int a_col_it = a_col_begin + thread_id_mod_16;
            // Get the ID of the column.
            int a_col_id = -1;

            if ( a_col_it < a_col_end )
            {
                a_col_id = __cachingLoad(&A_cols[a_col_it]);
            }

#ifdef AMGX_ILU_COLORING
            int valid = false;

            if ( a_col_id != -1 && current_color != 0 )
            {
                if ( boundary_coloring == LAST )
                {
                    valid = a_col_id >= boundary_index;
                }
                else
                {
                    valid = a_col_id < boundary_index && __cachingLoad(&row_colors[a_col_id]) > current_color;
                }
            }

#else
            int valid = false;

            if ( a_col_id != -1 && row_colors[a_col_id] > current_color )
            {
                valid = true;
            }

#endif

            // Set the column id.
            if ( !valid )
            {
                a_col_id = -1;
            }

            // Loop over columns. We compute 8 columns per iteration.
#pragma unroll 2

            for ( int k = 0 ; k < 16 ; k += 4 )
            {
                int my_k = k + thread_id_mod_16_div_4;
                // Load 8 blocks of X.
                int uniform_a_col_id = utils::shfl( a_col_id, shfl_offset + my_k, warpSize, active_mask_inner );
                Vector_type my_Delta(0);

                if ( uniform_a_col_id != -1 )
                {
                    my_Delta = Delta[4 * uniform_a_col_id + thread_id_mod_16_mod_4];
                }

                my_s_mem[thread_id_mod_16] = my_Delta;
                int uniform_a_col_it = a_col_begin + my_k;

                if ( uniform_a_col_id == -1 || uniform_a_col_it >= a_col_end )
                {
                    uniform_a_col_it = -1;
                }

                Matrix_type my_vals[4] = { Matrix_type(0) };

                if ( uniform_a_col_it != -1 )
                {
                    utils::load_vec4( my_vals, &A_vals[16 * uniform_a_col_it + 4 * thread_id_mod_16_mod_4] );
                }

                my_delta += my_vals[0] * my_s_mem[4 * thread_id_mod_16_div_4 + 0];
                my_delta += my_vals[1] * my_s_mem[4 * thread_id_mod_16_div_4 + 1];
                my_delta += my_vals[2] * my_s_mem[4 * thread_id_mod_16_div_4 + 2];
                my_delta += my_vals[3] * my_s_mem[4 * thread_id_mod_16_div_4 + 3];
            } // Loop over k
        } // Loop over aColIt

        // Load EINV values.
        Matrix_type my_Einv = Einv[16 * a_row_id + thread_id_mod_16];
        // Reduce delta terms.
        my_delta += utils::shfl_xor( my_delta, 4, warpSize, active_mask );
        my_delta += utils::shfl_xor( my_delta, 8, warpSize, active_mask );

        // Update the shared terms.
        if ( thread_id_mod_16_div_4 == 0 )
        {
            my_s_mem[thread_id_mod_16_mod_4] = my_delta;
        }

        // Update the diagonal term.
        my_delta = my_Einv * my_s_mem[thread_id_mod_16_mod_4];
        // Regroup results.
        my_delta += utils::shfl_xor( my_delta, 1, warpSize, active_mask );
        my_delta += utils::shfl_xor( my_delta, 2, warpSize, active_mask );
        // Store the results.
        int offset = 4 * a_row_id + thread_id_mod_16_div_4;
        Vector_type my_b(0), my_x(0);

        if ( thread_id_mod_16_mod_4 == 0 )
        {
            my_b = __cachingLoad(&delta[offset]);
            my_x = x    [offset];
        }

        my_delta = my_b - my_delta;

        if ( thread_id_mod_16_mod_4 == 0 )
        {
            x    [offset] = my_x + weight * my_delta;
            Delta[offset] = my_delta;
        }
    }
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template< typename Matrix_type, typename Vector_type, typename WeightType, int NUM_THREADS_PER_ROW, int CTA_SIZE, int WARP_SIZE >
__global__
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 700
__launch_bounds__( CTA_SIZE, 12 )
#elif defined(__CUDA_ARCH__)
__launch_bounds__( CTA_SIZE, 12 )
#endif
void DILU_backward_1x1_kernel( const int *__restrict A_rows,
                               const int *__restrict A_cols,
                               const Matrix_type *__restrict A_vals,
                               Vector_type *__restrict x,
                               const WeightType weight,
                               const int *__restrict sorted_rows_by_color,
                               const int *__restrict row_colors,
                               const Matrix_type *__restrict Einv,
                               const Vector_type *delta,
                               Vector_type *__restrict Delta,
                               const int num_rows_per_color,
                               const int current_color,
                               const ColoringType boundary_coloring,
                               const int boundary_index )
{
    // Number of items per CTA.
    const int NUM_ROWS_PER_CTA = CTA_SIZE / NUM_THREADS_PER_ROW;
    // Number of items per grid.
    const int NUM_ROWS_PER_GRID = gridDim.x * NUM_ROWS_PER_CTA;
    // The coordinates of the thread inside the CTA/warp.
    const int warp_id = utils::warp_id();
    const int lane_id = utils::lane_id();
    // Constants.
    const int lane_id_mod_NTPR = lane_id % NUM_THREADS_PER_ROW;
    // Determine which NxN block the threads work with.
    int a_row_it = blockIdx.x * NUM_ROWS_PER_CTA + (threadIdx.x / NUM_THREADS_PER_ROW);

    // Iterate over the rows of the matrix. One warp per row.
    for ( ; a_row_it < num_rows_per_color ; a_row_it += NUM_ROWS_PER_GRID )
    {
        int a_row_id = sorted_rows_by_color[a_row_it];
        // Load one block of B.
        Vector_type my_delta(0);
        // Don't do anything if X is zero.
        int a_col_it  = A_rows[a_row_id  ];
        int a_col_end = A_rows[a_row_id + 1];

        // Each warp load column indices of 32 nonzero blocks
        for ( a_col_it += lane_id_mod_NTPR ; utils::any( a_col_it < a_col_end ) ; a_col_it += NUM_THREADS_PER_ROW )
        {
            // Get the ID of the column.
            int a_col_id = -1;

            if ( a_col_it < a_col_end )
            {
                a_col_id = A_cols[a_col_it];
            }

            // Is it really a valid column (due to coloring).
            int valid = false;
#ifdef AMGX_ILU_COLORING

            if ( a_col_id != -1 && current_color != 0 )
            {
                //if( boundary_coloring == LAST )
                //  valid = a_col_id >= boundary_index;
                //else
                //  valid = a_col_id < boundary_index && row_colors[a_col_id] > current_color;
                valid = (((a_col_id < boundary_index || boundary_coloring == SYNC_COLORS) && (row_colors[a_col_id] > current_color)) || (a_col_id >= boundary_index && boundary_coloring == LAST));
            }

#else

            //if( a_col_id != -1 && current_color != 0 )
            if ( a_col_id != -1  )
            {
                valid = row_colors[a_col_id] > current_color;
            }

#endif
            // Load my Delta value.
            Vector_type my_Delta(0);

            if ( valid )
            {
                my_Delta = Delta[a_col_id];
            }

            // Load my item from A.
            Matrix_type my_val(0);

            if ( valid )
            {
                my_val = A_vals[a_col_it];
            }

            // Update bmAx.
            my_delta += my_val * my_Delta;
        }

        // Reduce bmAx terms.
#pragma unroll

        for ( int mask = NUM_THREADS_PER_ROW / 2 ; mask > 0 ; mask >>= 1 )
        {
            my_delta += utils::shfl_xor( my_delta, mask );
        }

        // Store the results.
        if ( lane_id_mod_NTPR == 0 )
        {
            Vector_type my_x = __cachingLoad(&delta[a_row_id]) - Einv[a_row_id] * my_delta;
            x    [a_row_id] += weight * my_x;
            Delta[a_row_id]  = my_x;
        }
    }
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template< typename Matrix_type, typename Vector_type, typename WeightType, int N, int CTA_SIZE >
__global__
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 700
__launch_bounds__( CTA_SIZE, 16 )
#elif defined(__CUDA_ARCH__)
__launch_bounds__( CTA_SIZE, 16 )
#endif
void DILU_backward_NxN_kernel_skip( Vector_type *__restrict x,
                                    const WeightType weight,
                                    const int *__restrict sorted_rows_by_color,
                                    const Vector_type *delta,
                                    Vector_type *__restrict Delta,
                                    const int num_rows_per_color )
{
    const int NUM_ITEMS_PER_CTA = CTA_SIZE / N; // Number of updated block items per CTA
    const int ITEM_ID           = threadIdx.x / N;
    const int ITEM_BLOCK_OFFSET = threadIdx.x % N;
    const int is_active    = ITEM_ID < NUM_ITEMS_PER_CTA;
    // The first row.
    int a_row_it = blockIdx.x * NUM_ITEMS_PER_CTA + ITEM_ID;

    // Iterate over the rows of the matrix. One warp per two rows.
    for ( ; a_row_it < num_rows_per_color ; a_row_it += gridDim.x * NUM_ITEMS_PER_CTA )
    {
        if ( is_active )
        {
            int a_row_id = sorted_rows_by_color[a_row_it];
            const int idx = N * a_row_id + ITEM_BLOCK_OFFSET;
            Vector_type my_b = __cachingLoad(&delta[idx]);
            Vector_type my_x = x[idx];
            x[idx] = my_x + weight * my_b;
            Delta[idx] = my_b;
        }
    }
}

// ----------
// Methods
// ----------

template< typename Matrix_type, typename Vector_type, int N >
void DILU_forward_NxN_dispatch( const int *__restrict A_rows,
                                const int *__restrict A_cols,
                                const Matrix_type *__restrict A_vals,
                                const int *__restrict A_diag,
                                const Vector_type *x,
                                const Vector_type *b,
                                Vector_type *__restrict delta,
                                const int *__restrict sorted_rows_by_color,
                                const int num_rows_per_color,
                                const int current_color,
                                const int *__restrict row_colors,
                                const Matrix_type *__restrict Einv,
                                const ColoringType boundary_coloring,
                                const int boundary_index,
                                const int row_major,
                                const int has_external_diag )
{
    const int NUM_WARPS_PER_CTA = CTA_SIZE / WARP_SIZE;
    // Squared N.
    const int NxN = N * N;
    // Number of items per warp.
    const int NUM_ROWS_PER_WARP = std::max(WARP_SIZE / NxN, 1);
    // Number of items computer per CTA.
    const int NUM_ROWS_PER_CTA = NUM_ROWS_PER_WARP * NUM_WARPS_PER_CTA;
    // The number of threads to launch.
    const int grid_size = std::min( 4096, (num_rows_per_color + NUM_ROWS_PER_CTA - 1) / NUM_ROWS_PER_CTA );
    // Branch to the correct kernel call.
    int code = 2 * (row_major ? 1 : 0) + (has_external_diag ? 1 : 0);

    switch ( code )
    {
        case 0: // Column-major, no external diagonal.
            DILU_forward_NxN_kernel<Matrix_type, Vector_type, N, CTA_SIZE, WARP_SIZE, false, false> <<< grid_size, CTA_SIZE>>>(
                A_rows,
                A_cols,
                A_vals,
                A_diag,
                x,
                b,
                delta,
                sorted_rows_by_color,
                num_rows_per_color,
                current_color,
                row_colors,
                Einv,
                boundary_coloring,
                boundary_index );
            break;

        case 1: // Column-major, external diagonal.
            DILU_forward_NxN_kernel<Matrix_type, Vector_type, N, CTA_SIZE, WARP_SIZE, false, true> <<< grid_size, CTA_SIZE>>>(
                A_rows,
                A_cols,
                A_vals,
                A_diag,
                x,
                b,
                delta,
                sorted_rows_by_color,
                num_rows_per_color,
                current_color,
                row_colors,
                Einv,
                boundary_coloring,
                boundary_index );
            break;

        case 2: // Row-major, no external diagonal.
            DILU_forward_NxN_kernel<Matrix_type, Vector_type, N, CTA_SIZE, WARP_SIZE, true, false> <<< grid_size, CTA_SIZE>>>(
                A_rows,
                A_cols,
                A_vals,
                A_diag,
                x,
                b,
                delta,
                sorted_rows_by_color,
                num_rows_per_color,
                current_color,
                row_colors,
                Einv,
                boundary_coloring,
                boundary_index );
            break;

        case 3: // Row-major, external diagonal.
            DILU_forward_NxN_kernel<Matrix_type, Vector_type, N, CTA_SIZE, WARP_SIZE, true, true> <<< grid_size, CTA_SIZE>>>(
                A_rows,
                A_cols,
                A_vals,
                A_diag,
                x,
                b,
                delta,
                sorted_rows_by_color,
                num_rows_per_color,
                current_color,
                row_colors,
                Einv,
                boundary_coloring,
                boundary_index );
            break;

        default:
            FatalError( "Internal error", AMGX_ERR_NOT_IMPLEMENTED );
    }

    cudaCheckError();
}

template< typename Matrix_type, typename Vector_type, int N >
void DILU_forward_NxN_dispatch_large( const int *__restrict A_rows,
                                      const int *__restrict A_cols,
                                      const Matrix_type *__restrict A_vals,
                                      const int *__restrict A_diag,
                                      const Vector_type *x,
                                      const Vector_type *b,
                                      Vector_type *__restrict delta,
                                      const int *__restrict sorted_rows_by_color,
                                      const int num_rows_per_color,
                                      const int current_color,
                                      const int *__restrict row_colors,
                                      const Matrix_type *__restrict Einv,
                                      const ColoringType boundary_coloring,
                                      const int boundary_index,
                                      const int row_major,
                                      const int has_external_diag )
{
    const int NUM_WARPS_PER_CTA = CTA_SIZE / WARP_SIZE;
    // Squared N.
    const int NxN = N * N;
    // Number of items computer per CTA.
    const int NUM_ROWS_PER_CTA = NUM_WARPS_PER_CTA;
    // Each warp is going to sweep through bloock this many times
    const int NUM_WARP_ITERS_PER_BLOCK = ((NxN - 1) / WARP_SIZE) + 1;
    // The number of threads to launch.
    const int grid_size = std::min( 4096, (num_rows_per_color + NUM_ROWS_PER_CTA - 1) / NUM_ROWS_PER_CTA );

    // Branch to the correct kernel call.
    if (!row_major)
    {
        FatalError("COL MAJOR is not supported for this large block_size", AMGX_ERR_NOT_SUPPORTED_BLOCKSIZE);
    }

    switch ( has_external_diag )
    {
        case 0: // Row-major, no external diagonal.
            DILU_forward_NxN_kernel_large<Matrix_type, Vector_type, N, CTA_SIZE, WARP_SIZE, false, NUM_WARP_ITERS_PER_BLOCK> <<< grid_size, CTA_SIZE>>>(
                A_rows,
                A_cols,
                A_vals,
                A_diag,
                x,
                b,
                delta,
                sorted_rows_by_color,
                num_rows_per_color,
                current_color,
                row_colors,
                Einv,
                boundary_coloring,
                boundary_index );
            break;

        case 1: // Row-major, external diagonal.
            DILU_forward_NxN_kernel_large<Matrix_type, Vector_type, N, CTA_SIZE, WARP_SIZE, true, NUM_WARP_ITERS_PER_BLOCK> <<< grid_size, CTA_SIZE>>>(
                A_rows,
                A_cols,
                A_vals,
                A_diag,
                x,
                b,
                delta,
                sorted_rows_by_color,
                num_rows_per_color,
                current_color,
                row_colors,
                Einv,
                boundary_coloring,
                boundary_index );
            break;

        default:
            FatalError( "Internal error", AMGX_ERR_NOT_IMPLEMENTED );
    }

    cudaCheckError();
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template< typename Matrix_type, typename Vector_type >
void DILU_forward_NxN_dispatch( const int *__restrict A_rows,
                                const int *__restrict A_cols,
                                const Matrix_type *__restrict A_vals,
                                const int *__restrict A_diag,
                                const Vector_type *x,
                                const Vector_type *b,
                                Vector_type *__restrict delta,
                                const int *__restrict sorted_rows_by_color,
                                const int num_rows_per_color,
                                const int current_color,
                                const int *__restrict row_colors,
                                const Matrix_type *__restrict Einv,
                                const ColoringType boundary_coloring,
                                const int boundary_index,
                                const int block_size,
                                const int row_major,
                                const int has_external_diag )
{
    switch ( block_size )
    {
        case 1:
            {
                const int NUM_THREADS_PER_ROW = 8;
                // Number of items computer per CTA.
                const int NUM_ROWS_PER_CTA = CTA_SIZE / NUM_THREADS_PER_ROW;
                // The number of threads to launch.
                const int grid_size = std::min( 4096, (num_rows_per_color + NUM_ROWS_PER_CTA - 1) / NUM_ROWS_PER_CTA );

                if ( has_external_diag )
                {
                    DILU_forward_1x1_kernel<Matrix_type, Vector_type, NUM_THREADS_PER_ROW, CTA_SIZE, WARP_SIZE, true> <<< grid_size, CTA_SIZE>>>(
                        A_rows,
                        A_cols,
                        A_vals,
                        A_diag,
                        x,
                        b,
                        delta,
                        sorted_rows_by_color,
                        num_rows_per_color,
                        current_color,
                        row_colors,
                        Einv,
                        boundary_coloring,
                        boundary_index );
                }
                else
                {
                    DILU_forward_1x1_kernel<Matrix_type, Vector_type, NUM_THREADS_PER_ROW, CTA_SIZE, WARP_SIZE, false> <<< grid_size, CTA_SIZE>>>(
                        A_rows,
                        A_cols,
                        A_vals,
                        A_diag,
                        x,
                        b,
                        delta,
                        sorted_rows_by_color,
                        num_rows_per_color,
                        current_color,
                        row_colors,
                        Einv,
                        boundary_coloring,
                        boundary_index );
                }

                cudaCheckError();
            }
            break;

        case 2:
            DILU_forward_NxN_dispatch<Matrix_type, Vector_type, 2>(
                A_rows,
                A_cols,
                A_vals,
                A_diag,
                x,
                b,
                delta,
                sorted_rows_by_color,
                num_rows_per_color,
                current_color,
                row_colors,
                Einv,
                boundary_coloring,
                boundary_index,
                row_major,
                has_external_diag );
            break;

        case 3:
            DILU_forward_NxN_dispatch<Matrix_type, Vector_type, 3>(
                A_rows,
                A_cols,
                A_vals,
                A_diag,
                x,
                b,
                delta,
                sorted_rows_by_color,
                num_rows_per_color,
                current_color,
                row_colors,
                Einv,
                boundary_coloring,
                boundary_index,
                row_major,
                has_external_diag );
            break;

        case 4:
            if ( row_major )
            {
                // Number of items computer per CTA.
                const int NUM_ROWS_PER_CTA = CTA_SIZE / 16;
                // The number of threads to launch.
                const int grid_size = std::min( 4096, (num_rows_per_color + NUM_ROWS_PER_CTA - 1) / NUM_ROWS_PER_CTA );

                if ( has_external_diag )
                    //DILU_forward_4x4_kernel<Matrix_type, Vector_type, CTA_SIZE, WARP_SIZE, true, true><<<grid_size, CTA_SIZE>>>(
                    DILU_forward_4x4_kernel_row_major_vec4<Matrix_type, Vector_type, CTA_SIZE, true> <<< grid_size, CTA_SIZE>>>(
                        A_rows,
                        A_cols,
                        A_vals,
                        A_diag,
                        x,
                        b,
                        delta,
                        sorted_rows_by_color,
                        num_rows_per_color,
                        current_color,
                        row_colors,
                        Einv,
                        boundary_coloring,
                        boundary_index );
                else
                    DILU_forward_4x4_kernel_row_major_vec4<Matrix_type, Vector_type, CTA_SIZE, false> <<< grid_size, CTA_SIZE>>>(
                        A_rows,
                        A_cols,
                        A_vals,
                        A_diag,
                        x,
                        b,
                        delta,
                        sorted_rows_by_color,
                        num_rows_per_color,
                        current_color,
                        row_colors,
                        Einv,
                        boundary_coloring,
                        boundary_index );

                cudaCheckError();
            }
            else
                DILU_forward_NxN_dispatch<Matrix_type, Vector_type, 4>(
                    A_rows,
                    A_cols,
                    A_vals,
                    A_diag,
                    x,
                    b,
                    delta,
                    sorted_rows_by_color,
                    num_rows_per_color,
                    current_color,
                    row_colors,
                    Einv,
                    boundary_coloring,
                    boundary_index,
                    row_major,
                    has_external_diag );

            break;

        case 5:
            DILU_forward_NxN_dispatch<Matrix_type, Vector_type, 5>(
                A_rows,
                A_cols,
                A_vals,
                A_diag,
                x,
                b,
                delta,
                sorted_rows_by_color,
                num_rows_per_color,
                current_color,
                row_colors,
                Einv,
                boundary_coloring,
                boundary_index,
                row_major,
                has_external_diag );
            break;

        case 8:
            DILU_forward_NxN_dispatch_large<Matrix_type, Vector_type, 8>(
                A_rows,
                A_cols,
                A_vals,
                A_diag,
                x,
                b,
                delta,
                sorted_rows_by_color,
                num_rows_per_color,
                current_color,
                row_colors,
                Einv,
                boundary_coloring,
                boundary_index,
                row_major,
                has_external_diag );
            break;

        case 10:
            DILU_forward_NxN_dispatch_large<Matrix_type, Vector_type, 10>(
                A_rows,
                A_cols,
                A_vals,
                A_diag,
                x,
                b,
                delta,
                sorted_rows_by_color,
                num_rows_per_color,
                current_color,
                row_colors,
                Einv,
                boundary_coloring,
                boundary_index,
                row_major,
                has_external_diag );
            break;

        default:
            FatalError( "Internal error", AMGX_ERR_NOT_IMPLEMENTED );
    }

    cudaCheckError();
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template< typename Matrix_type, typename Vector_type, typename WeightType, int N >
void DILU_backward_NxN_dispatch( const int *__restrict A_rows,
                                 const int *__restrict A_cols,
                                 const Matrix_type *__restrict A_vals,
                                 Vector_type *__restrict x,
                                 const WeightType weight,
                                 const int *__restrict sorted_rows_by_color,
                                 const int *__restrict row_colors,
                                 const Matrix_type *__restrict Einv,
                                 const Vector_type *delta,
                                 Vector_type *__restrict Delta,
                                 const int num_rows_per_color,
                                 const int current_color,
                                 const ColoringType boundary_coloring,
                                 const int boundary_index,
                                 const int row_major )
{
    const int NUM_WARPS_PER_CTA = CTA_SIZE / WARP_SIZE;
    // Squared N.
    const int NxN = N * N;
    // Number of items per warp.
    const int NUM_ROWS_PER_WARP = std::max(WARP_SIZE / NxN, 1);
    // Number of items computer per CTA.
    const int NUM_ROWS_PER_CTA = NUM_ROWS_PER_WARP * NUM_WARPS_PER_CTA;
    // The number of threads to launch.
    const int grid_size = std::min( 4096, (num_rows_per_color + NUM_ROWS_PER_CTA - 1) / NUM_ROWS_PER_CTA );

    // Branch to the correct kernel call.
    if ( row_major )
    {
        DILU_backward_NxN_kernel<Matrix_type, Vector_type, WeightType, N, CTA_SIZE, WARP_SIZE, true> <<< grid_size, CTA_SIZE>>>(
            A_rows,
            A_cols,
            A_vals,
            x,
            weight,
            sorted_rows_by_color,
            row_colors,
            Einv,
            delta,
            Delta,
            num_rows_per_color,
            current_color,
            boundary_coloring,
            boundary_index );
    }
    else
    {
        DILU_backward_NxN_kernel<Matrix_type, Vector_type, WeightType, N, CTA_SIZE, WARP_SIZE, false> <<< grid_size, CTA_SIZE>>>(
            A_rows,
            A_cols,
            A_vals,
            x,
            weight,
            sorted_rows_by_color,
            row_colors,
            Einv,
            delta,
            Delta,
            num_rows_per_color,
            current_color,
            boundary_coloring,
            boundary_index );
    }

    cudaCheckError();
}



template< typename Matrix_type, typename Vector_type, typename WeightType, int N, int NUM_WARP_ITERS_PER_BLOCK >
void DILU_backward_NxN_dispatch_large( const int *__restrict A_rows,
                                       const int *__restrict A_cols,
                                       const Matrix_type *__restrict A_vals,
                                       Vector_type *__restrict x,
                                       const WeightType weight,
                                       const int *__restrict sorted_rows_by_color,
                                       const int *__restrict row_colors,
                                       const Matrix_type *__restrict Einv,
                                       const Vector_type *delta,
                                       Vector_type *__restrict Delta,
                                       const int num_rows_per_color,
                                       const int current_color,
                                       const ColoringType boundary_coloring,
                                       const int boundary_index,
                                       const int row_major )
{
    const int NUM_WARPS_PER_CTA = CTA_SIZE / WARP_SIZE;
    // Number of items computer per CTA.
    const int NUM_ROWS_PER_CTA = NUM_WARPS_PER_CTA;
    // The number of threads to launch.
    const int grid_size = std::min( 4096, (num_rows_per_color + NUM_ROWS_PER_CTA - 1) / NUM_ROWS_PER_CTA );

    // Branch to the correct kernel call.
    if ( row_major )
    {
        DILU_backward_NxN_kernel_large<Matrix_type, Vector_type, WeightType, N, CTA_SIZE, WARP_SIZE, true, NUM_WARP_ITERS_PER_BLOCK> <<< grid_size, CTA_SIZE>>>(
            A_rows,
            A_cols,
            A_vals,
            x,
            weight,
            sorted_rows_by_color,
            row_colors,
            Einv,
            delta,
            Delta,
            num_rows_per_color,
            current_color,
            boundary_coloring,
            boundary_index );
    }
    else
    {
        FatalError("col major is not supported for this blocksize in multicolor DILU solver", AMGX_ERR_NOT_IMPLEMENTED);
    }

    cudaCheckError();
}


///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template< typename Matrix_type, typename Vector_type, typename WeightType>
void DILU_backward_NxN_dispatch( const int *__restrict A_rows,
                                 const int *__restrict A_cols,
                                 const Matrix_type *__restrict A_vals,
                                 Vector_type *__restrict x,
                                 const WeightType weight,
                                 const int *__restrict sorted_rows_by_color,
                                 const int *__restrict row_colors,
                                 const Matrix_type *__restrict Einv,
                                 const Vector_type *delta,
                                 Vector_type *__restrict Delta,
                                 const int num_rows_per_color,
                                 const int current_color,
                                 const ColoringType boundary_coloring,
                                 const int boundary_index,
                                 const int block_size,
                                 const int row_major )
{
    switch ( block_size )
    {
        case 1:
            {
                const int NUM_THREADS_PER_ROW = 8;
                // Number of items computer per CTA.
                const int NUM_ROWS_PER_CTA = CTA_SIZE / NUM_THREADS_PER_ROW;
                // The number of threads to launch.
                const int grid_size = std::min( 4096, (num_rows_per_color + NUM_ROWS_PER_CTA - 1) / NUM_ROWS_PER_CTA );
                DILU_backward_1x1_kernel<Matrix_type, Vector_type, WeightType, NUM_THREADS_PER_ROW, CTA_SIZE, WARP_SIZE> <<< grid_size, CTA_SIZE>>>(
                    A_rows,
                    A_cols,
                    A_vals,
                    x,
                    weight,
                    sorted_rows_by_color,
                    row_colors,
                    Einv,
                    delta,
                    Delta,
                    num_rows_per_color,
                    current_color,
                    boundary_coloring,
                    boundary_index );
                cudaCheckError();
            }
            break;

        case 2:
            DILU_backward_NxN_dispatch<Matrix_type, Vector_type, WeightType, 2>(
                A_rows,
                A_cols,
                A_vals,
                x,
                weight,
                sorted_rows_by_color,
                row_colors,
                Einv,
                delta,
                Delta,
                num_rows_per_color,
                current_color,
                boundary_coloring,
                boundary_index,
                row_major );
            break;

        case 3:
            DILU_backward_NxN_dispatch<Matrix_type, Vector_type, WeightType, 3>(
                A_rows,
                A_cols,
                A_vals,
                x,
                weight,
                sorted_rows_by_color,
                row_colors,
                Einv,
                delta,
                Delta,
                num_rows_per_color,
                current_color,
                boundary_coloring,
                boundary_index,
                row_major );
            break;

        case 4:

            //if( false )
            if ( row_major )
            {
                // Number of items computer per CTA.
                const int NUM_ROWS_PER_CTA = CTA_SIZE / 16;
                // The number of threads to launch.
                const int grid_size = std::min( 4096, (num_rows_per_color + NUM_ROWS_PER_CTA - 1) / NUM_ROWS_PER_CTA );
                //DILU_backward_NxN_kernel<Matrix_type, Vector_type, 4, CTA_SIZE, WARP_SIZE, true><<<grid_size, CTA_SIZE>>>(
                DILU_backward_4x4_kernel_row_major_vec4<Matrix_type, Vector_type, WeightType, CTA_SIZE> <<< grid_size, CTA_SIZE>>>(
                    A_rows,
                    A_cols,
                    A_vals,
                    x,
                    weight,
                    sorted_rows_by_color,
                    row_colors,
                    Einv,
                    delta,
                    Delta,
                    num_rows_per_color,
                    current_color,
                    boundary_coloring,
                    boundary_index );
                cudaCheckError();
            }
            else
                DILU_backward_NxN_dispatch<Matrix_type, Vector_type, WeightType, 4>(
                    A_rows,
                    A_cols,
                    A_vals,
                    x,
                    weight,
                    sorted_rows_by_color,
                    row_colors,
                    Einv,
                    delta,
                    Delta,
                    num_rows_per_color,
                    current_color,
                    boundary_coloring,
                    boundary_index,
                    row_major );

            break;

        case 5:
            DILU_backward_NxN_dispatch<Matrix_type, Vector_type, WeightType, 5>(
                A_rows,
                A_cols,
                A_vals,
                x,
                weight,
                sorted_rows_by_color,
                row_colors,
                Einv,
                delta,
                Delta,
                num_rows_per_color,
                current_color,
                boundary_coloring,
                boundary_index,
                row_major );
            break;

        case 8:
            DILU_backward_NxN_dispatch_large<Matrix_type, Vector_type, WeightType, 8, 2>(
                A_rows,
                A_cols,
                A_vals,
                x,
                weight,
                sorted_rows_by_color,
                row_colors,
                Einv,
                delta,
                Delta,
                num_rows_per_color,
                current_color,
                boundary_coloring,
                boundary_index,
                row_major );
            break;

        case 10:
            DILU_backward_NxN_dispatch_large<Matrix_type, Vector_type, WeightType, 10, 4>(
                A_rows,
                A_cols,
                A_vals,
                x,
                weight,
                sorted_rows_by_color,
                row_colors,
                Einv,
                delta,
                Delta,
                num_rows_per_color,
                current_color,
                boundary_coloring,
                boundary_index,
                row_major );
            break;

        default:
            FatalError( "Internal error", AMGX_ERR_NOT_IMPLEMENTED );
    }

    cudaCheckError();
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template< class T_Config >
MulticolorDILUSolver_Base<T_Config>::MulticolorDILUSolver_Base( AMG_Config &cfg,
        const std::string &cfg_scope,
        ThreadManager *tmng ) :
    Solver<T_Config>( cfg, cfg_scope, tmng )
{
    this->weight = cfg.AMG_Config::template getParameter<double>("relaxation_factor", cfg_scope);
    this->m_reorder_cols_by_color_desired = (cfg.AMG_Config::template getParameter<int>("reorder_cols_by_color", cfg_scope) != 0);
    this->m_insert_diagonal_desired = (cfg.AMG_Config::template getParameter<int>("insert_diag_while_reordering", cfg_scope) != 0);
    this->m_boundary_coloring = cfg.AMG_Config::template getParameter<ColoringType>("boundary_coloring", cfg_scope);
    this->always_obey_coloring = 0;

    if (weight == 0)
    {
        weight = 1.;
        amgx_printf("Warning, setting weight to 1 instead of estimating largest_eigen_value in Multicolor DILU smoother\n");
    }
}

// Destructor
template<class T_Config>
MulticolorDILUSolver_Base<T_Config>::~MulticolorDILUSolver_Base()
{
    Einv.clear();
    Einv.shrink_to_fit();
}

template<class T_Config>
void MulticolorDILUSolver_Base<T_Config>::computeEinv(Matrix<T_Config> &A)
{
    ViewType oldView = A.currentView();
    A.setView(this->m_explicit_A->getViewExterior());

    if ( A.get_block_dimx() != A.get_block_dimy() )
    {
        FatalError("DILU implemented only for squared blocks", AMGX_ERR_NOT_SUPPORTED_BLOCKSIZE);
    }

    if ( A.get_block_dimx() > 32) // actually much more less than 32 doe to register file limitations, but...
    {
        FatalError("DILU implemented only for squared blocks of size <= 32", AMGX_ERR_NOT_SUPPORTED_BLOCKSIZE);
    }

    computeEinv_NxN( A, A.get_block_dimx() );
    A.setView(oldView);
}

template< class T_Config >
void
MulticolorDILUSolver_Base<T_Config>::printSolverParameters() const
{
    std::cout << "relaxation_factor = " << this->weight << std::endl;
}

// Solver setup
template< class T_Config >
void
MulticolorDILUSolver_Base<T_Config>::solver_setup(bool reuse_matrix_structure)
{
    m_explicit_A = dynamic_cast<Matrix<T_Config>*>(this->m_A);

    if (!this->m_explicit_A)
    {
        FatalError("MulticolorDILUSolver only works with explicit matrices", AMGX_ERR_INTERNAL);
    }

    int N = this->m_explicit_A->get_num_cols() * this->m_explicit_A->get_block_dimy();

    if (this->m_explicit_A->getColoringLevel() < 1)
    {
        FatalError("Matrix must be colored to use multicolor dilu solver. Try setting: coloring_level=1 in the configuration file", AMGX_ERR_NOT_IMPLEMENTED);
    }

    m_delta.resize(N);
    m_Delta.resize(N);
    m_delta.set_block_dimy(this->m_explicit_A->get_block_dimy());
    m_Delta.set_block_dimy(this->m_explicit_A->get_block_dimy());
    m_delta.set_block_dimx(1);
    m_Delta.set_block_dimx(1);

    if ( this->m_explicit_A->getBlockFormat() != ROW_MAJOR )
    {
        FatalError("Multicolor DILU solver only supports row major format for the blocks", AMGX_ERR_CONFIGURATION);
    }

    computeEinv( *this->m_explicit_A );
}

//
template< class T_Config >
void
MulticolorDILUSolver_Base<T_Config>::solve_init( VVector &b, VVector &x, bool xIsZero )
{
}

// Solve one iteration
template<class T_Config>
AMGX_STATUS
MulticolorDILUSolver_Base<T_Config>::solve_iteration( VVector &b, VVector &x, bool xIsZero )
{
    if ( this->m_explicit_A->get_block_dimx() != this->m_explicit_A->get_block_dimy() )
    {
        FatalError("DILU implemented only for squared blocks", AMGX_ERR_NOT_SUPPORTED_BLOCKSIZE);
    }

    if ( this->m_explicit_A->get_block_dimx() > 32) // actually much more less than 32 doe to register file limitations, but...
    {
        FatalError("DILU implemented only for squared blocks of size <= 32", AMGX_ERR_NOT_SUPPORTED_BLOCKSIZE);
    }

    if (xIsZero)
    {
        x.dirtybit = 0;
    }

    if (!this->m_explicit_A->is_matrix_singleGPU())
    {
        this->m_explicit_A->manager->exchange_halo_async(x, x.tag);
        this->m_explicit_A->manager->exchange_halo_async(b, b.tag);
    }

    if (this->m_explicit_A->getViewExterior() == this->m_explicit_A->getViewInterior())
    {
        if (!this->m_explicit_A->is_matrix_singleGPU())
        {
            this->m_explicit_A->manager->exchange_halo_wait(x, x.tag);
            this->m_explicit_A->manager->exchange_halo_wait(b, b.tag);
        }
    }

    ViewType oldView = this->m_explicit_A->currentView();
    ViewType flags;
    bool latencyHiding = true;

    if (this->m_explicit_A->is_matrix_singleGPU() || (x.dirtybit == 0 && b.dirtybit == 0))
    {
        latencyHiding = false;
        this->m_explicit_A->setViewExterior();
        flags = (ViewType)(this->m_explicit_A->getViewExterior());
    }
    else
    {
        flags = (ViewType)(this->m_explicit_A->getViewInterior());
        this->m_explicit_A->setViewInterior();
    }

    if (xIsZero)
    {
        thrust_wrapper::fill<T_Config::memSpace>(x.begin(), x.end(), types::util<ValueTypeB>::get_zero());
        cudaCheckError();
    }

    this->smooth_NxN(*this->m_explicit_A, b, x, flags);

    if (latencyHiding)
    {
        if (!this->m_explicit_A->is_matrix_singleGPU())
        {
            this->m_explicit_A->manager->exchange_halo_wait(x, x.tag);
            this->m_explicit_A->manager->exchange_halo_wait(b, b.tag);
        }

        this->m_explicit_A->setViewExterior();
        flags = (ViewType)(~(this->m_explicit_A->getViewInterior()) & this->m_explicit_A->getViewExterior());

        if (flags != 0)
        {
            this->smooth_NxN(*this->m_explicit_A, b, x, flags);
        }
    }

    x.dirtybit = 1;
    this->m_explicit_A->setView(oldView);
    return (this->converged(b, x));
}

template<class T_Config>
void
MulticolorDILUSolver_Base<T_Config>::solve_finalize( VVector &b, VVector &x )
{}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template< AMGX_VecPrecision V, AMGX_MatPrecision M, AMGX_IndPrecision I >
void MulticolorDILUSolver<TemplateConfig<AMGX_host, V, M, I> >::computeEinv_NxN(const Matrix_h &A, const int bsize)
{
    FatalError("Multicolor DILU smoother not implemented for host format, exiting", AMGX_ERR_NOT_SUPPORTED_TARGET);
}

template< AMGX_VecPrecision V, AMGX_MatPrecision M, AMGX_IndPrecision I >
void MulticolorDILUSolver<TemplateConfig<AMGX_host, V, M, I> >::smooth_NxN( const Matrix_h &A, VVector &b, VVector &x, ViewType separation_flag )
{
    FatalError("Haven't implemented Multicolor DILU smoother for host format", AMGX_ERR_NOT_SUPPORTED_TARGET);
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template< AMGX_VecPrecision V, AMGX_MatPrecision M, AMGX_IndPrecision I >
MulticolorDILUSolver<TemplateConfig<AMGX_device, V, M, I> >::MulticolorDILUSolver(
    AMG_Config &cfg,
    const std::string &cfg_scope,
    ThreadManager *tmng ) :
    MulticolorDILUSolver_Base<TemplateConfig<AMGX_device, V, M, I> >( cfg, cfg_scope, tmng )
{
    int device = 0;
    cudaGetDevice( &device );
    cudaDeviceProp properties;
    cudaGetDeviceProperties( &properties, device );
    m_is_kepler = properties.major >= 3;
}

template< AMGX_VecPrecision V, AMGX_MatPrecision M, AMGX_IndPrecision I >
void MulticolorDILUSolver<TemplateConfig<AMGX_device, V, M, I> >::computeEinv_NxN(const Matrix_d &A, const int bsize)
{
    const int bsize_sq = bsize * bsize;
    this->Einv.resize( A.get_num_cols()*bsize_sq, 0.0 );

    // sol::prof_start();
    for ( int i = 0, num_colors = A.getMatrixColoring().getNumColors() ; i < num_colors ; ++i )
    {
        const int color_offset = A.getMatrixColoring().getOffsetsRowsPerColor()[i];
        const int num_rows_per_color = A.getMatrixColoring().getOffsetsRowsPerColor()[i + 1] - color_offset;

        if ( num_rows_per_color == 0 )
        {
            continue;
        }

        const int CTA_SIZE = 128;
        const int NUM_WARPS_PER_CTA = CTA_SIZE / 32;
        int ROWS_PER_WARP = 1;

        if ( bsize_sq > 1 && bsize_sq < 6 )
        {
            ROWS_PER_WARP = 32 / bsize_sq;
        }

        const int ROWS_PER_CTA = ROWS_PER_WARP * NUM_WARPS_PER_CTA;
        const int GRID_SIZE = std::min( 4096, (num_rows_per_color + ROWS_PER_CTA - 1) / ROWS_PER_CTA );
        cudaStream_t stream = amgx::thrust::global_thread_handle::get_stream();

        switch ( bsize )
        {
            case 1:
                DILU_setup_1x1_kernel<ValueTypeA, ValueTypeB, 8, CTA_SIZE, 32> <<< GRID_SIZE, CTA_SIZE, 0, stream>>>(
                    A.row_offsets.raw(),
                    A.col_indices.raw(),
                    A.diag.raw(),
                    A.values.raw(),
                    this->Einv.raw(),
                    A.getMatrixColoring().getSortedRowsByColor().raw() + color_offset,
                    A.getMatrixColoring().getRowColors().raw(),
                    num_rows_per_color,
                    i );
                break;

            case 2:
                DILU_setup_NxN_kernel<ValueTypeA, ValueTypeB, 2, CTA_SIZE, 32> <<< GRID_SIZE, CTA_SIZE, 0, stream>>>(
                    A.row_offsets.raw(),
                    A.col_indices.raw(),
                    A.diag.raw(),
                    A.values.raw(),
                    this->Einv.raw(),
                    A.getMatrixColoring().getSortedRowsByColor().raw() + color_offset,
                    A.getMatrixColoring().getRowColors().raw(),
                    num_rows_per_color,
                    i );
                break;

            case 3:
                DILU_setup_NxN_kernel<ValueTypeA, ValueTypeB, 3, CTA_SIZE, 32> <<< GRID_SIZE, CTA_SIZE, 0, stream>>>(
                    A.row_offsets.raw(),
                    A.col_indices.raw(),
                    A.diag.raw(),
                    A.values.raw(),
                    this->Einv.raw(),
                    A.getMatrixColoring().getSortedRowsByColor().raw() + color_offset,
                    A.getMatrixColoring().getRowColors().raw(),
                    num_rows_per_color,
                    i );
                break;

            case 4:
                DILU_setup_NxN_kernel<ValueTypeA, ValueTypeB, 4, CTA_SIZE, 32> <<< GRID_SIZE, CTA_SIZE, 0, stream>>>(
                    A.row_offsets.raw(),
                    A.col_indices.raw(),
                    A.diag.raw(),
                    A.values.raw(),
                    this->Einv.raw(),
                    A.getMatrixColoring().getSortedRowsByColor().raw() + color_offset,
                    A.getMatrixColoring().getRowColors().raw(),
                    num_rows_per_color,
                    i );
                break;

            case 5:
                DILU_setup_NxN_kernel<ValueTypeA, ValueTypeB, 5, CTA_SIZE, 32> <<< GRID_SIZE, CTA_SIZE, 0, stream>>>(
                    A.row_offsets.raw(),
                    A.col_indices.raw(),
                    A.diag.raw(),
                    A.values.raw(),
                    this->Einv.raw(),
                    A.getMatrixColoring().getSortedRowsByColor().raw() + color_offset,
                    A.getMatrixColoring().getRowColors().raw(),
                    num_rows_per_color,
                    i );
                break;

            case 8:
                DILU_setup_NxN_kernel_large<ValueTypeA, ValueTypeB, 8, CTA_SIZE, 32, 2> <<< GRID_SIZE, CTA_SIZE, 0, stream>>>(
                    A.row_offsets.raw(),
                    A.col_indices.raw(),
                    A.diag.raw(),
                    A.values.raw(),
                    this->Einv.raw(),
                    A.getMatrixColoring().getSortedRowsByColor().raw() + color_offset,
                    A.getMatrixColoring().getRowColors().raw(),
                    num_rows_per_color,
                    i );
                break;

            case 10:
                DILU_setup_NxN_kernel_large<ValueTypeA, ValueTypeB, 10, CTA_SIZE, 32, 4> <<< GRID_SIZE, CTA_SIZE, 0, stream>>>(
                    A.row_offsets.raw(),
                    A.col_indices.raw(),
                    A.diag.raw(),
                    A.values.raw(),
                    this->Einv.raw(),
                    A.getMatrixColoring().getSortedRowsByColor().raw() + color_offset,
                    A.getMatrixColoring().getRowColors().raw(),
                    num_rows_per_color,
                    i );
                break;

            default:
                FatalError( "Multicolor-DILU Setup: block size was not enabled in the code, contact AMGX developers.", AMGX_ERR_NOT_SUPPORTED_BLOCKSIZE );
        }

        cudaCheckError();
    }
}

template< AMGX_VecPrecision V, AMGX_MatPrecision M, AMGX_IndPrecision I >
void MulticolorDILUSolver<TemplateConfig<AMGX_device, V, M, I> >::smooth_NxN( const Matrix_d &A, VVector &b, VVector &x, ViewType separation_flag )
{
    AMGX_CPU_PROFILER( "MulticolorDILUSolver::smooth_NxN " );
    int offset = 0, separation = 0;
    A.getOffsetAndSizeForView(INTERIOR, &offset, &separation);

    // Only have separation=num interior rows if we are only working on the interior
    // and the boundary coloring is FIRST or LAST, otherwise set separation offset to
    // total number of rows
    if ( separation_flag != this->m_explicit_A->getViewInterior() ||
            this->m_explicit_A->getViewExterior() == this->m_explicit_A->getViewInterior() ||
            this->m_boundary_coloring != LAST && this->m_boundary_coloring != FIRST )
    {
        separation = A.row_offsets.size() - 1;
    }
    else
    {
        amgx_printf("separation active\n");
    }

    // --------------------
    // Forward Sweep
    // --------------------
    const int num_colors = this->m_explicit_A->getMatrixColoring().getNumColors();

    for ( int i = 0 ; i < num_colors ; ++i )
    {
        int color_offset(0);

        if ( separation_flag & INTERIOR )
        {
            color_offset = A.getMatrixColoring().getOffsetsRowsPerColor()[i];
        }
        else
        {
            color_offset = A.getMatrixColoring().getSeparationOffsetsRowsPerColor()[i];
        }

        int num_rows_per_color(0);

        if ( separation_flag == this->m_explicit_A->getViewInterior() )
        {
            num_rows_per_color = A.getMatrixColoring().getSeparationOffsetsRowsPerColor()[i];
        }
        else
        {
            num_rows_per_color = A.getMatrixColoring().getOffsetsRowsPerColor()[i + 1];
        }

        num_rows_per_color -= color_offset;

        if ( num_rows_per_color == 0 )
        {
            continue;
        }

        int boundary_index = separation;

        if ( this->m_boundary_coloring == SYNC_COLORS )
        {
            boundary_index = A.get_num_rows();
        }

        DILU_forward_NxN_dispatch(
            A.row_offsets.raw(),
            A.col_indices.raw(),
            A.values.raw(),
            A.diag.raw(),
            x.raw(),
            b.raw(),
            this->m_delta.raw(),
            A.getMatrixColoring().getSortedRowsByColor().raw() + color_offset,
            num_rows_per_color,
            i,
            A.getMatrixColoring().getRowColors().raw(),
            this->Einv.raw(),
            this->m_boundary_coloring,
            boundary_index,
            A.get_block_dimy(),
            A.getBlockFormat() == ROW_MAJOR,
            A.hasProps(DIAG) );
        cudaCheckError();
    }

    // --------------------
    // Backward Sweep
    // --------------------
    for ( int i = num_colors - 1 ; i >= 0 ; --i )
    {
        int color_offset(0);

        if ( separation_flag & INTERIOR )
        {
            color_offset = A.getMatrixColoring().getOffsetsRowsPerColor()[i];
        }
        else
        {
            color_offset = A.getMatrixColoring().getSeparationOffsetsRowsPerColor()[i];
        }

        int num_rows_per_color(0);

        if ( separation_flag == this->m_explicit_A->getViewInterior() )
        {
            num_rows_per_color = A.getMatrixColoring().getSeparationOffsetsRowsPerColor()[i];
        }
        else
        {
            num_rows_per_color = A.getMatrixColoring().getOffsetsRowsPerColor()[i + 1];
        }

        num_rows_per_color -= color_offset;

        if ( num_rows_per_color == 0 )
        {
            continue;
        }

        if ( i == num_colors - 1 )
        {
            const int NUM_ROWS_PER_CTA = CTA_SIZE / A.get_block_dimy();
            const int GRID_SIZE = std::min( 4096, (num_rows_per_color + NUM_ROWS_PER_CTA - 1) / NUM_ROWS_PER_CTA );

            switch ( A.get_block_dimy() )
            {
                case 1:
                    DILU_backward_NxN_kernel_skip<ValueTypeA, ValueTypeB, WeightType, 1, CTA_SIZE> <<< GRID_SIZE, CTA_SIZE>>>(
                        x.raw(),
                        this->weight,
                        A.getMatrixColoring().getSortedRowsByColor().raw() + color_offset,
                        this->m_delta.raw(),
                        this->m_Delta.raw(),
                        num_rows_per_color );
                    break;

                case 2:
                    DILU_backward_NxN_kernel_skip<ValueTypeA, ValueTypeB, WeightType, 2, CTA_SIZE> <<< GRID_SIZE, CTA_SIZE>>>(
                        x.raw(),
                        this->weight,
                        A.getMatrixColoring().getSortedRowsByColor().raw() + color_offset,
                        this->m_delta.raw(),
                        this->m_Delta.raw(),
                        num_rows_per_color );
                    break;

                case 3:
                    DILU_backward_NxN_kernel_skip<ValueTypeA, ValueTypeB, WeightType, 3, CTA_SIZE> <<< GRID_SIZE, CTA_SIZE>>>(
                        x.raw(),
                        this->weight,
                        A.getMatrixColoring().getSortedRowsByColor().raw() + color_offset,
                        this->m_delta.raw(),
                        this->m_Delta.raw(),
                        num_rows_per_color );
                    break;

                case 4:
                    DILU_backward_NxN_kernel_skip<ValueTypeA, ValueTypeB, WeightType, 4, CTA_SIZE> <<< GRID_SIZE, CTA_SIZE>>>(
                        x.raw(),
                        this->weight,
                        A.getMatrixColoring().getSortedRowsByColor().raw() + color_offset,
                        this->m_delta.raw(),
                        this->m_Delta.raw(),
                        num_rows_per_color );
                    break;

                case 5:
                    DILU_backward_NxN_kernel_skip<ValueTypeA, ValueTypeB, WeightType, 5, CTA_SIZE> <<< GRID_SIZE, CTA_SIZE>>>(
                        x.raw(),
                        this->weight,
                        A.getMatrixColoring().getSortedRowsByColor().raw() + color_offset,
                        this->m_delta.raw(),
                        this->m_Delta.raw(),
                        num_rows_per_color );
                    break;

                case 8:
                    DILU_backward_NxN_kernel_skip<ValueTypeA, ValueTypeB, WeightType, 8, CTA_SIZE> <<< GRID_SIZE, CTA_SIZE>>>(
                        x.raw(),
                        this->weight,
                        A.getMatrixColoring().getSortedRowsByColor().raw() + color_offset,
                        this->m_delta.raw(),
                        this->m_Delta.raw(),
                        num_rows_per_color );
                    break;

                case 10:
                    DILU_backward_NxN_kernel_skip<ValueTypeA, ValueTypeB, WeightType, 10, CTA_SIZE> <<< GRID_SIZE, CTA_SIZE>>>(
                        x.raw(),
                        this->weight,
                        A.getMatrixColoring().getSortedRowsByColor().raw() + color_offset,
                        this->m_delta.raw(),
                        this->m_Delta.raw(),
                        num_rows_per_color );
                    break;
            }

            cudaCheckError();
        }
        else
        {
            DILU_backward_NxN_dispatch(
                A.row_offsets.raw(),
                A.col_indices.raw(),
                A.values.raw(),
                x.raw(),
                this->weight,
                A.getMatrixColoring().getSortedRowsByColor().raw() + color_offset,
                A.getMatrixColoring().getRowColors().raw(),
                this->Einv.raw(),
                this->m_delta.raw(),
                this->m_Delta.raw(),
                num_rows_per_color,
                i,
                this->m_boundary_coloring,
                separation,
                A.get_block_dimy(),
                A.getBlockFormat() == ROW_MAJOR );
            cudaCheckError();
        }
    }
}

/****************************************
 * Explict instantiations
 ***************************************/
#define AMGX_CASE_LINE(CASE) template class MulticolorDILUSolver_Base<TemplateMode<CASE>::Type>;
AMGX_FORALL_BUILDS(AMGX_CASE_LINE)
//    AMGX_FORCOMPLEX_BUILDS(AMGX_CASE_LINE)
#undef AMGX_CASE_LINE

#define AMGX_CASE_LINE(CASE) template class MulticolorDILUSolver<TemplateMode<CASE>::Type>;
AMGX_FORALL_BUILDS(AMGX_CASE_LINE)
//    AMGX_FORCOMPLEX_BUILDS(AMGX_CASE_LINE)
#undef AMGX_CASE_LINE

}
} // namespace amgx

