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

#include <basic_types.h>
#include <util.h>
#include <error.h>
#include <types.h>
#include <matrix_coloring/parallel_greedy.h>
#include <cusp/format.h>
#include <cusp/copy.h>
#include <cusp/detail/random.h>
#include <thrust/count.h>
#include <thrust/extrema.h>
#include <sm_utils.inl>

#include <algorithm>

// Pseudo-random number generator

namespace amgx
{

static __host__ __device__ unsigned int hash_function(unsigned int a, unsigned int seed)
{
    a ^= seed;
    a = (a + 0x7ed55d16) + (a << 12);
    a = (a ^ 0xc761c23c) + (a >> 19);
    a = (a + 0x165667b1) + (a << 5);
    a = (a ^ 0xd3a2646c) + (a << 9);
    a = (a + 0xfd7046c5) + (a << 3);
    a = (a ^ 0xb55a4f09) + (a >> 16);
    return a;
}

struct is_zero
{
    __host__ __device__
    bool operator()(int x)
    {
        return x == 0;
    }
};

// ---------------------------
// Kernels
// ---------------------------

template< int CTA_SIZE>
__global__
void color_kernel_warp( const int A_num_rows, const int *A_rows, const int *A_cols, int num_colors, int *new_color, int *A_colors )
{
    const int NUM_WARPS_PER_CTA  = CTA_SIZE / 32;
    const int NUM_WARPS_PER_GRID = gridDim.x * NUM_WARPS_PER_CTA;
    // Thread coordinates.
    const int warp_id = utils::warp_id();
    const int lane_id = utils::lane_id();
    volatile __shared__ unsigned long long colors_used[NUM_WARPS_PER_CTA][32];
    // Row identifier.
    int row_id = blockIdx.x * NUM_WARPS_PER_CTA + warp_id;

    // Iterate over the rows of the matrix.
    for ( ; row_id < A_num_rows ; row_id += NUM_WARPS_PER_GRID )
    {
        colors_used[warp_id][lane_id] = 0ull;
        int row_color = A_colors[row_id];

        if ( row_color != 0 ) // Already colored!!!
        {
            continue;
        }

        int row_hash = hash_function(row_id, 0);
        bool max_row = true;
        // Iterators over my row.
        int row_begin = A_rows[row_id  ];
        int row_end   = A_rows[row_id + 1];

        for ( ; row_begin < row_end ; row_begin += 32)
        {
            // Iterator.
            int row_it = row_begin + lane_id;
            // Get the column index (if the iterator is valid).
            int col_id = -1;

            if ( row_it < row_end )
            {
                col_id = A_cols[row_it];
            }

            // Each thread hashes its column id.
            int col_hash = hash_function( col_id, 0 );
            // Get the color of the column.
            int col_color = -1;

            if ( row_it < row_end && col_id < A_num_rows )
            {
                col_color = A_colors[col_id];
            }

            if (col_color != -1)
            {
                // Set the bit corresponding to that color to 1
                colors_used[warp_id][lane_id] |= (1 << col_color);
            }

            max_row &= (row_hash > col_hash || col_color != 0 || col_color == -1);
        }

        max_row &= utils::all( max_row );

        if (max_row) // If I'm a max
        {
            unsigned int active_mask = utils::activemask();
            // warp reduction to figure out what color to assign
            if (lane_id < 16) { colors_used[warp_id][lane_id] |= colors_used[warp_id][lane_id + 16]; } utils::syncwarp(active_mask);

            if (lane_id < 8) { colors_used[warp_id][lane_id] |= colors_used[warp_id][lane_id + 8]; } utils::syncwarp(active_mask);

            if (lane_id < 4) { colors_used[warp_id][lane_id] |= colors_used[warp_id][lane_id + 4]; } utils::syncwarp(active_mask);

            if (lane_id < 2) { colors_used[warp_id][lane_id] |= colors_used[warp_id][lane_id + 2]; } utils::syncwarp(active_mask);

            if (lane_id < 1) { colors_used[warp_id][lane_id] |= colors_used[warp_id][lane_id + 1]; } utils::syncwarp(active_mask);

            if (lane_id == 0)
            {
                // Find the first unassigned color, starting at 1
#pragma unroll
                for (int i = 1; i < 64; i++)
                {
                    unsigned long long mask = 1 << i;

                    if ( !(colors_used[warp_id][0] & mask) )
                    {
                        A_colors[row_id] = i;

                        if (i >= num_colors) { new_color[0] = 1; }

                        break;
                    }
                }
            }
        }
    }
}

template< int CTA_SIZE>
__global__
void color_kernel_thread( const int A_num_rows, const int *A_rows, const int *A_cols, int *d_num_colors,  int *d_new_color, int *A_colors, int *d_num_uncolored_block, int *d_done)
{
    const int NUM_THREADS_PER_GRID = gridDim.x * CTA_SIZE;
    // Row identifier.
    int row_id = blockIdx.x * CTA_SIZE + threadIdx.x;

    if (d_done[0] == 1)
    {
        return;
    }

    int num_uncolored_thread = 0;

    // Iterate over rows.
    for ( ; row_id < A_num_rows ; row_id += NUM_THREADS_PER_GRID )
    {
        unsigned long long used_colors = 0ull;
        int row_color = A_colors[row_id];

        if ( row_color != 0 ) // Already colored!!!
        {
            continue;
        }
        else
        {
            num_uncolored_thread++;
        }

        int row_hash = hash_function(row_id, 0);
        //int row_hash = row_id;
        bool max_row = true;
        // Iterators over my row.
        int row_begin = A_rows[row_id  ];
        int row_end   = A_rows[row_id + 1];

        for ( int row_it = row_begin; row_it < row_end ; row_it++)
        {
            // Get the column index (if the iterator is valid).
            int col_id = A_cols[row_it];

            if (col_id >= A_num_rows || col_id == row_id) { continue; }

            // Each thread hashes its column id.
            int col_hash = hash_function( col_id, 0 );
            // Get the color of the column.
            int col_color = A_colors[col_id];

            // Update the color.
            if ( col_color > 0)
            {
                used_colors |= 1ull << (64 - col_color);
            }

            // Is it still a max row?
            max_row &= (row_hash > col_hash || col_color != 0);
        }

        if ( max_row )
        {
            int my_color = 64 - utils::bfind( ~used_colors );

            if ( my_color > 0 )
            {
                A_colors[row_id] = my_color;
            }

            if ( my_color >= d_num_colors[0] )
            {
                d_new_color[0] = 1;
            }
        }
    }

    // Do per block reduction
    __shared__ volatile int smem[CTA_SIZE];
    smem[threadIdx.x] = num_uncolored_thread;
    __syncthreads();

    for ( int off = blockDim.x / 2; off >= 32; off = off / 2 )
    {
        if ( threadIdx.x < off )
        {
            smem[threadIdx.x] += smem[threadIdx.x + off];
        }

        __syncthreads();
    }

    // warp reduce
    if ( threadIdx.x < 32 )
    {
        smem[threadIdx.x] += smem[threadIdx.x + 16];
        smem[threadIdx.x] += smem[threadIdx.x + 8];
        smem[threadIdx.x] += smem[threadIdx.x + 4];
        smem[threadIdx.x] += smem[threadIdx.x + 2];
        smem[threadIdx.x] += smem[threadIdx.x + 1];
    }

    if ( threadIdx.x == 0 )
    {
        d_num_uncolored_block[blockIdx.x] = smem[0];
    }
}

template<int MAX_CTA_SIZE, bool TERMINATE_IF_FULL_COLORS_USED>
__global__
void count_uncolored_and_num_colors_kernel(int *d_new_color, int *d_num_colors, int max_uncolored_rows, int *d_num_uncolored_block, int *d_num_uncolored, int *d_done, int *h_done, int *d_start_color, int CTA_SIZE)
{
    // Do block reduction
    __shared__ volatile int smem[MAX_CTA_SIZE];

    if (threadIdx.x >= CTA_SIZE)
    {
        smem[threadIdx.x] = 0;
    }
    else
    {
        smem[threadIdx.x] = d_num_uncolored_block[threadIdx.x];
    }

    __syncthreads();

    for ( int off = blockDim.x / 2; off >= 32; off = off / 2 )
    {
        if ( threadIdx.x < off )
        {
            smem[threadIdx.x] += smem[threadIdx.x + off];
        }

        __syncthreads();
    }

    // warp reduce
    if ( threadIdx.x < 32 )
    {
        smem[threadIdx.x] += smem[threadIdx.x + 16];
        smem[threadIdx.x] += smem[threadIdx.x + 8];
        smem[threadIdx.x] += smem[threadIdx.x + 4];
        smem[threadIdx.x] += smem[threadIdx.x + 2];
        smem[threadIdx.x] += smem[threadIdx.x + 1];
    }

    if ( threadIdx.x == 0 )
    {
        int prev_num_uncolored = d_num_uncolored[0];
        int num_uncolored = smem[0];
        int num_colors = d_num_colors[0];

        if (d_new_color[0])
        {
            num_colors++; // New color has been added
            d_new_color[0] = 0; // Reset to 0
        }

        // Check for termination
        if (TERMINATE_IF_FULL_COLORS_USED)
        {
            // Check for termination
            if (num_colors == 64 || prev_num_uncolored == num_uncolored || num_uncolored <= max_uncolored_rows)
            {
                h_done[0] = 1;
                d_done[0] = 1;
            }
        }
        else
        {
            if (prev_num_uncolored == num_uncolored || num_uncolored <= max_uncolored_rows)
            {
                if (num_uncolored <= max_uncolored_rows)
                {
                    h_done[0] = 1;
                    d_done[0] = 1;
                }
                else
                {
                    d_start_color[0] += 32;
                }
            }
        }

        // Write to gmem
        d_num_uncolored[0] = num_uncolored;
        d_num_colors[0] = num_colors;
    }
}


//////////////////////////////
// ANYRING version: begin
// This unrolled recursion is used to visit neighbors.
//////////////////////////////

template<int COLORING_LEVEL>
struct parallel_greedy_neighborhood
{
    __device__ __inline__ static void visit(int row_id, int row_hash, int visit_id, const int A_num_rows, const int *A_rows, const int *A_cols, int *A_colors,
                                            const int &start_color, bool &max_row, unsigned long long &used_colors
                                           )
    {
        int row_begin = A_rows[visit_id  ];
        int row_end   = A_rows[visit_id + 1];

        for ( int row_it = row_begin; row_it < row_end ; row_it++)
        {
            // Get the column index (if the iterator is valid).
            int col_id = A_cols[row_it];

            if (col_id >= A_num_rows || col_id == visit_id) { continue; }

            // Each thread hashes its column id.
            int col_hash = hash_function( col_id, 0 );
            // Get the color of the column.
            int col_color = A_colors[col_id];

            // Update the color.
            if ( col_color - start_color > 0)
            {
                used_colors |= 1ull << (64 - col_color + start_color);
            }

            // Is it still a max row?
            max_row &= (row_hash > col_hash || (row_hash == col_hash && col_id >= row_id)  || col_color != 0);

            if (!max_row) { break; }

            parallel_greedy_neighborhood < COLORING_LEVEL - 1 >::visit(row_id, row_hash, col_id, A_num_rows, A_rows, A_cols, A_colors, start_color, max_row, used_colors);

            if (!max_row) { break; }
        }
    }
};

template<>
struct parallel_greedy_neighborhood<0>
{
    __device__ __inline__ static void visit(int row_id, int row_hash, int visit_id, const int A_num_rows, const int *A_rows, const int *A_cols, int *A_colors,
                                            const int &start_color, bool &max_row, unsigned long long &used_colors
                                           )
    {
        ;
    }
};


//////////////////////////////
// This kernel is equivalent to color_kernel_thread, but if COLORING_LEVEL>=2 it recurses to neighbors'+ neighbors.
// it also uses start_color_ to continue coloring with parallel greedy instead of falling back to min_max if more than 64 colors are used
//////////////////////////////

template< int CTA_SIZE, int COLORING_LEVEL >
__global__
void color_kernel_thread_anyring( const int A_num_rows, const int *A_rows, const int *A_cols, int *d_num_colors,  int *d_new_color, int *A_colors, int *d_num_uncolored_block, int *d_done,
                                  const int *start_color_)
{
    const int NUM_THREADS_PER_GRID = gridDim.x * CTA_SIZE;
    // Row identifier.
    int row_id = blockIdx.x * CTA_SIZE + threadIdx.x;

    if (d_done[0] == 1)
    {
        return;
    }

    const int start_color = start_color_[0];// (d_num_colors[0] / 64)*64;
    int num_uncolored_thread = 0;

    // Iterate over rows.
    for ( ; row_id < A_num_rows ; row_id += NUM_THREADS_PER_GRID )
    {
        unsigned long long used_colors = 0ull;
        int row_color = A_colors[row_id];

        if ( row_color != 0 ) // Already colored!!!
        {
            continue;
        }
        else
        {
            num_uncolored_thread++;
        }

        int row_hash = hash_function(row_id, 0);
        //int row_hash = row_id;
        bool max_row = true;

        if (COLORING_LEVEL == 1)
        {
            // This path is equivalent to color_kernel_thread, but it uses start_color.
            // Iterators over my row.
            int row_begin = A_rows[row_id  ];
            int row_end   = A_rows[row_id + 1];

            for ( int row_it = row_begin; row_it < row_end ; row_it++)
            {
                // Get the column index (if the iterator is valid).
                int col_id = A_cols[row_it];

                if (col_id >= A_num_rows || col_id == row_id) { continue; }

                // Each thread hashes its column id.
                int col_hash = hash_function( col_id, 0 );
                // Get the color of the column.
                int col_color = A_colors[col_id];

                // Update the color.
                if ( col_color - start_color > 0)
                {
                    used_colors |= 1ull << (64 - col_color + start_color);
                }

                // Is it still a max row?
                max_row &= (row_hash > col_hash || (row_hash == col_hash && col_id >= row_id)  || col_color != 0);

                if (!max_row) { break; }
            }
        }
        else
        {
            //recurse
            parallel_greedy_neighborhood<COLORING_LEVEL>::visit(row_id, row_hash, row_id, A_num_rows, A_rows, A_cols, A_colors, start_color, max_row, used_colors);
        }

        if ( max_row )
        {
            int my_color = 64 - utils::bfind( ~used_colors );

            if ( my_color > 0 && my_color < 64)
            {
                A_colors[row_id] = my_color + start_color;

                if ( my_color + start_color >= d_num_colors[0] )
                {
                    d_new_color[0] = 1;
                }
            }
        }
    }

    // Do per block reduction
    __shared__ volatile int smem[CTA_SIZE];
    smem[threadIdx.x] = num_uncolored_thread;
    __syncthreads();

    for ( int off = blockDim.x / 2; off >= 32; off = off / 2 )
    {
        if ( threadIdx.x < off )
        {
            smem[threadIdx.x] += smem[threadIdx.x + off];
        }

        __syncthreads();
    }

    // warp reduce
    if ( threadIdx.x < 32 )
    {
        smem[threadIdx.x] += smem[threadIdx.x + 16];
        smem[threadIdx.x] += smem[threadIdx.x + 8];
        smem[threadIdx.x] += smem[threadIdx.x + 4];
        smem[threadIdx.x] += smem[threadIdx.x + 2];
        smem[threadIdx.x] += smem[threadIdx.x + 1];
    }

    if ( threadIdx.x == 0 )
    {
        d_num_uncolored_block[blockIdx.x] = smem[0];
    }
}

//////////////////////////////
// ANYRING version: end
//////////////////////////////

template< int CTA_SIZE>
__global__
void color_kernel_thread_fallback( const int A_num_rows, const int *A_rows, const int *A_cols, int num_colors,  int *new_color, int *A_colors )
{
    const int NUM_THREADS_PER_GRID = gridDim.x * CTA_SIZE;
    // Row identifier.
    int row_id = blockIdx.x * CTA_SIZE + threadIdx.x;

    // Iterate over rows.
    for ( ; row_id < A_num_rows ; row_id += NUM_THREADS_PER_GRID )
    {
        int row_color = A_colors[row_id];

        if ( row_color != 0 ) // Already colored!!!
        {
            continue;
        }

        int row_hash = hash_function(row_id, 0);
        bool max_row = true;
        // Iterators over my row.
        int row_begin = A_rows[row_id  ];
        int row_end   = A_rows[row_id + 1];
        int max_color = 0;

        //if (row_id == 2) printf("row_id=%d\n",row_id);
        for ( int row_it = row_begin; row_it < row_end ; row_it++)
        {
            // Get the column index (if the iterator is valid).
            int col_id = A_cols[row_it];

            if (col_id >= A_num_rows) { continue; }

            // Each thread hashes its column id.
            int col_hash = hash_function( col_id, 0 );
            // Get the color of the column.
            int col_color = A_colors[col_id];

            // Update the color.
            if ( col_color > max_color)
            {
                max_color = col_color;
            }

            // Is it still a max row?
            max_row &= (row_hash > col_hash || col_color != 0);
        }

        if ( max_row )
        {
            int my_color = max_color + 1;

            if ( my_color > 0 )
            {
                A_colors[row_id] = my_color;
            }

            if ( my_color >= num_colors )
            {
                new_color[0] = 1;
            }
        }
    }
}




template< int CTA_SIZE, int WARP_SIZE >
__global__
void
dbg_check_coloring_kernel( const int A_num_rows, const int *A_rows, const int *A_cols, const int *A_colors, const int *A_gtlt_count, int *error_found )
{
    const int NUM_WARPS_PER_CTA  = CTA_SIZE / WARP_SIZE;
    const int NUM_WARPS_PER_GRID = gridDim.x * NUM_WARPS_PER_CTA;
    // Thread coordinates.
    const int warp_id = utils::warp_id();
    const int lane_id = utils::lane_id();
    // Row identifier.
    int row_id = blockIdx.x * NUM_WARPS_PER_CTA + warp_id;

    // Iterate over the rows of the matrix.
    for ( ; row_id < A_num_rows ; row_id += NUM_WARPS_PER_GRID )
    {
        int row_color = A_colors[row_id];
        // Iterators over my row.
        int row_begin = A_rows[row_id  ];
        int row_end   = A_rows[row_id + 1];

        for ( ; row_begin < row_end ; row_begin += WARP_SIZE )
        {
            // Iterator.
            int row_it = row_begin + lane_id;
            // Get the column index (if the iterator is valid).
            int col_id = -1;

            if ( row_it < row_end )
            {
                col_id = A_cols[row_it];
            }

            // Get the color of the column.
            int col_color = -1;

            if ( row_it < row_end && col_id < A_num_rows )
            {
                col_color = A_colors[col_id];
            }

            // Is there something wrong ??
            if ( row_id != col_id && row_color == col_color )
            {
                if ( A_gtlt_count != NULL && !error_found[0] )
                {
                    printf( "row_id=%d, row_color=%d, col_id=%d, col_color=%d\n", row_id, row_color, col_id, col_color );
                }

                error_found[0] = 1;
            }
        }
    }
}

// ---------------------------
// Methods
// ---------------------------

template< class T_Config >
Parallel_Greedy_Matrix_Coloring_Base<T_Config>::Parallel_Greedy_Matrix_Coloring_Base( AMG_Config &cfg, const std::string &cfg_scope) : MatrixColoring<T_Config>(cfg, cfg_scope)
{
    if ( this->m_coloring_level > 5 )
    {
        FatalError( "Not implemented for coloring_level > 5", AMGX_ERR_NOT_SUPPORTED_TARGET );
    }

    if ( cfg.AMG_Config::getParameter<IndexType>("determinism_flag", "default"))
    {
        m_uncolored_fraction = 0.0;
    }
    else
    {
        m_uncolored_fraction = cfg.AMG_Config::getParameter<double>("max_uncolored_percentage", cfg_scope);
    }
}

// Block version
template< AMGX_VecPrecision V, AMGX_MatPrecision M, AMGX_IndPrecision I >
void
Parallel_Greedy_Matrix_Coloring<TemplateConfig<AMGX_device, V, M, I> >::colorMatrix( Matrix_d &A )
{
    ViewType oldView = A.currentView();
    this->m_row_colors.resize( A.row_offsets.size() - 1 );

    if  (this->m_halo_coloring == SYNC_COLORS) { A.setView(ALL); }
    else { A.setViewExterior(); }

    const int num_rows = A.get_num_rows();
    const int max_uncolored_rows = static_cast<int>( this->m_uncolored_fraction * num_rows );
    const int CTA_SIZE = 128;
    const int MAX_GRID_SIZE = 1024;

    if (MAX_GRID_SIZE > 1024)
    {
        FatalError("Grid size must be less than 1024\n", AMGX_ERR_CORE);
    }

    const int GRID_SIZE = std::min( MAX_GRID_SIZE, (num_rows + CTA_SIZE - 1) / CTA_SIZE);
    amgx::thrust::fill( this->m_row_colors.begin(), this->m_row_colors.end(), 0 );
    cudaCheckError();
    typedef typename Matrix_d::IVector IVector_d;
    cudaStream_t stream = amgx::thrust::global_thread_handle::get_stream();
    IVector_d d_new_color(1);
    cudaMemsetAsync(d_new_color.raw(), 0, sizeof(int), stream);
    this->m_num_colors = 1;
    IVector_d d_num_colors(1);
    IVector_d d_start_color(1); //for coloring safely without fallback
    cudaMemcpyAsync(d_num_colors.raw(), &this->m_num_colors, sizeof(int), cudaMemcpyHostToDevice, stream);
    int tmp = 0;
    cudaMemcpyAsync(d_start_color.raw(), &tmp, sizeof(int), cudaMemcpyHostToDevice, stream);
    IVector_d d_num_uncolored(1);
    IVector_d d_num_uncolored_block(MAX_GRID_SIZE);
    int *h_done = NULL;
    amgx::thrust::global_thread_handle::cudaMallocHost( (void **) &h_done, sizeof(int));
    IVector_d d_done(1);
    d_done[0] = 0;
    *h_done = 0;
    cudaEvent_t     throttle_event = 0;
    cudaEventCreateWithFlags(&throttle_event, cudaEventDisableTiming);
    int iteration = 0;

    while (h_done[0] == 0)
    {
#define PG_ARGS                 num_rows,\
        A.row_offsets.raw(),\
        A.col_indices.raw(),\
        d_num_colors.raw(),\
        d_new_color.raw(),\
        this->m_row_colors.raw(),\
        d_num_uncolored_block.raw(),\
        d_done.raw()

        if (this->m_coloring_level == 1)
        {
            color_kernel_thread<CTA_SIZE> <<< GRID_SIZE, CTA_SIZE, 0, stream>>>(PG_ARGS);
            count_uncolored_and_num_colors_kernel<MAX_GRID_SIZE, true> <<< 1, MAX_GRID_SIZE, 0, stream>>>(d_new_color.raw(), d_num_colors.raw(), max_uncolored_rows, d_num_uncolored_block.raw(), d_num_uncolored.raw(), d_done.raw(), h_done, d_start_color.raw(), GRID_SIZE);
        }
        else
        {
            if (this->m_coloring_level == 2)
            {
                color_kernel_thread_anyring<CTA_SIZE, 2> <<< GRID_SIZE, CTA_SIZE, 0, stream>>>(PG_ARGS, d_start_color.raw());
            }
            else if (this->m_coloring_level == 3)
            {
                color_kernel_thread_anyring<CTA_SIZE, 3> <<< GRID_SIZE, CTA_SIZE, 0, stream>>>(PG_ARGS, d_start_color.raw());
            }
            else if (this->m_coloring_level == 4)
            {
                color_kernel_thread_anyring<CTA_SIZE, 4> <<< GRID_SIZE, CTA_SIZE, 0, stream>>>(PG_ARGS, d_start_color.raw());
            }
            else if (this->m_coloring_level == 5)
            {
                color_kernel_thread_anyring<CTA_SIZE, 5> <<< GRID_SIZE, CTA_SIZE, 0, stream>>>(PG_ARGS, d_start_color.raw());
            }

            count_uncolored_and_num_colors_kernel<MAX_GRID_SIZE, false> <<< 1, MAX_GRID_SIZE, 0, stream>>>(d_new_color.raw(), d_num_colors.raw(), max_uncolored_rows, d_num_uncolored_block.raw(), d_num_uncolored.raw(), d_done.raw(), h_done, d_start_color.raw(), GRID_SIZE);
        }

        // Throttle every 4 iteration
        if (iteration % 4 == 0)
        {
            cudaEventRecord(throttle_event);
        }
        else
        {
            cudaEventSynchronize(throttle_event);
        };

        iteration++;

        cudaCheckError();
    }

    typedef typename Matrix_h::IVector IVector_h;
    IVector_h new_color(1);
    cudaEventDestroy(throttle_event);
    this->m_num_colors = d_num_colors[0];
    int num_uncolored = d_num_uncolored[0];
    int prev_num_uncolored = 0;

    // Fallback path if # colors exceeds 64, instead of picking the smallest unassigned color,
    // it sets its color to max neighbor color + 1
    if (this->m_num_colors == 64)
    {
        // Choose highest unassigned color among the neighbors
        for ( int num_uncolored = num_rows ; num_uncolored > max_uncolored_rows && prev_num_uncolored != num_uncolored ; )
        {
            prev_num_uncolored = num_uncolored;
            new_color[0] = 0; // Host to device transfer
            color_kernel_thread_fallback<CTA_SIZE> <<< GRID_SIZE, CTA_SIZE, 0, stream>>>(
                num_rows,
                A.row_offsets.raw(),
                A.col_indices.raw(),
                this->m_num_colors,
                amgx::thrust::raw_pointer_cast (&new_color.front()),
                this->m_row_colors.raw() );
            cudaCheckError();
            num_uncolored = (int) amgx::thrust::count_if( this->m_row_colors.begin(), this->m_row_colors.begin() + num_rows, is_zero() );

            if (new_color[0]) { this->m_num_colors++; }

            cudaCheckError();
        }
    }

#if 0
    device_vector_alloc<int> error_found( 1, 0 );
    dbg_check_coloring_kernel<CTA_SIZE, 32> <<< GRID_SIZE, CTA_SIZE>>>(
        num_rows,
        A.row_offsets.raw(),
        A.col_indices.raw(),
        this->m_row_colors.raw(),
        NULL,
        amgx::thrust::raw_pointer_cast( &error_found.front() ) );
    cudaCheckError();

    if ( error_found[0] != 0 )
    {
        std::cout << "INVALID COLORING !!! Two neighbors have the same color!!!" << std::endl;
    }

#endif
    amgx::thrust::global_thread_handle::cudaFreeHost(h_done);
    A.setView(oldView);
}


#define AMGX_CASE_LINE(CASE) template class Parallel_Greedy_Matrix_Coloring_Base<TemplateMode<CASE>::Type>;
AMGX_FORALL_BUILDS(AMGX_CASE_LINE)
#undef AMGX_CASE_LINE

#define AMGX_CASE_LINE(CASE) template class Parallel_Greedy_Matrix_Coloring<TemplateMode<CASE>::Type>;
AMGX_FORALL_BUILDS(AMGX_CASE_LINE)
#undef AMGX_CASE_LINE

} // end namespace amgx

