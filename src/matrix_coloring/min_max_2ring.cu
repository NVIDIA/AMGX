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

#include <algorithm>

#include <basic_types.h>
#include <util.h>
#include <error.h>
#include <types.h>
#include <matrix_coloring/min_max_2ring.h>
#include <cusp/format.h>
#include <cusp/copy.h>
#include <cusp/detail/random.h>
#include <thrust/count.h>
#include <thrust/extrema.h>
#include <sm_utils.inl>

#define COLORING_DEBUG 1


// Pseudo-random number generator
namespace amgx
{

static __host__ __device__ unsigned int hash_function(unsigned int a, unsigned int seed, unsigned int rows = 0)
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

template< int CTA_SIZE, int WARP_SIZE >
__global__
void count_gtlt_kernel( const int A_num_rows,
                        const int *__restrict A_rows,
                        const int *__restrict A_cols,
                        const int *__restrict A_colors,
                        int *A_gtlt_count,
                        const int seed)
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

        if ( row_color != 0 )
        {
            if ( lane_id == 0 )
            {
                A_gtlt_count[row_id] = -1;
            }

            continue;
        }

        // Hash my row id.
        int row_hash = hash_function( row_id, seed );
        // The number of vertices that are greater/smaller than me.
        int gt_count = 0, lt_count = 0;
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

            // Each thread hashes its column id.
            int col_hash = hash_function( col_id, seed );
            // Get the color of the column.
            int col_color = -1;

            if ( row_it < row_end && col_id < A_num_rows)
            {
                col_color = A_colors[col_id];
            }

            // Threads determine if they are greater than the row hash.
            int gt_pred = col_color == 0 && col_hash > row_hash;
            int lt_pred = col_color == 0 && col_hash < row_hash;
            // Count greater/smaller neighbors.
            gt_count += __popc( utils::ballot( gt_pred ) );
            lt_count += __popc( utils::ballot( lt_pred ) );
        }

        // The warp leader stores the result.
        int my_gtlt_count = (gt_count << 16) | lt_count;

        if ( lane_id == 0 )
        {
            A_gtlt_count[row_id] = my_gtlt_count;
        }
    }
}


template< int CTA_SIZE, int WARP_SIZE, bool LATE_REJECTION >
__global__
void color_kernel( const int A_num_rows, const int *A_rows, const int *A_cols, const int *A_gtlt_count, const int current_color, const int weakness_bound, int *A_colors )
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

        if ( row_color != 0 ) // Already colored!!!
        {
            continue;
        }

        // The number of vertices that are greater/smaller than me.
        int row_gtlt_count = A_gtlt_count[row_id];
        // Split gtlt_count into 2.
        int row_gt_count = row_gtlt_count >> 16;
        int row_lt_count = row_gtlt_count  & 0xffff;

        // Min-max algorithm.
        if ( row_gt_count == 0 && row_lt_count == 0 )
        {
            if ( lane_id == 0 )
            {
                A_colors[row_id] = current_color + (row_id & 1);
            }

            continue;
        }

        if ( row_gt_count == 0 )
        {
            if ( lane_id == 0 )
            {
                A_colors[row_id] = current_color;
            }

            continue;
        }

        if ( row_lt_count == 0 )
        {
            if ( lane_id == 0 )
            {
                A_colors[row_id] = current_color + 1;
            }

            continue;
        }

        // Do we skip it.
        int candidate = 1;
        // Predicates. Is a vertex min/max.
        int is_max_vertex = row_gt_count <= weakness_bound;
        int is_min_vertex = row_lt_count <= weakness_bound;
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

            // Get the color of the column (it could help late rejection).
            if ( LATE_REJECTION )
            {
                int col_color = -1;

                if ( row_it < row_end && col_id < A_num_rows)
                {
                    col_color = A_colors[col_id];
                }

                // Late rejection test.
                if ( col_color == current_color )
                {
                    is_max_vertex = 0;
                }

                if ( col_color == current_color + 1 )
                {
                    is_min_vertex = 0;
                }
            }

            // Get the gt/lt count.
            int col_gtlt_count = -1;

            if ( row_it < row_end && col_id < A_num_rows )
            {
                col_gtlt_count = A_gtlt_count[col_id];
            }

            // Split gtlt_count into 2. col_gt/lt_count == -1 if already colored.
            int col_gt_count = col_gtlt_count >> 16;
            int col_lt_count = col_gtlt_count  & 0xffff;
            // Threads determine if they are greater than the row hash.

            if ( col_gtlt_count != -1 )
            {
                is_max_vertex &= col_gt_count > row_gt_count || (col_gt_count == row_gt_count && col_id <= row_id);
                is_min_vertex &= col_lt_count > row_lt_count || (col_lt_count == row_lt_count && col_id >= row_id);
            }
        }

        // The warp leader stores the result.
        if ( candidate && utils::all( is_max_vertex ) )
        {
            if ( lane_id == 0 )
            {
                A_colors[row_id] = current_color;
            }

            continue;
        }

        if ( candidate && utils::all( is_min_vertex ) )
        {
            if ( lane_id == 0 )
            {
                A_colors[row_id] = current_color + 1;
            }

            continue;
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

            if ( row_it < row_end && col_id < A_num_rows)
            {
                col_color = A_colors[col_id];
            }

            // Is there something wrong ??
            if ( row_id != col_id && row_color == col_color && row_color != 0)
            {
                if ( A_gtlt_count != NULL && !error_found[0] )
                {
                    //printf( "row_id=%d, row_color=%d, col_id=%d, col_color=%d\n", row_id, row_color, col_id, col_color );
                    /*
                    // The number of vertices that are greater/smaller than me.
                    int row_gtlt_count = A_gtlt_count[row_id];
                    int row_gt_count = row_gtlt_count >> 16;
                    int row_lt_count = row_gtlt_count  & 0xffff;
                    printf( "row_gt_count=%d, row_gt_count=%d\n", row_gt_count, row_lt_count );

                    int col_gtlt_count = A_gtlt_count[col_id];
                    int col_gt_count = col_gtlt_count >> 16;
                    int col_lt_count = col_gtlt_count  & 0xffff;
                    printf( "col_gt_count=%d, col_gt_count=%d\n", col_gt_count, col_lt_count ); */
                }

                error_found[0] = 1;
                atomicAdd(error_found + 1, 1);
            }
        }
    }
}

// ---------------------------
// Methods
// ---------------------------

template< class T_Config >
Min_Max_2Ring_Matrix_Coloring_Base<T_Config>::Min_Max_2Ring_Matrix_Coloring_Base( AMG_Config &cfg, const std::string &cfg_scope) : MatrixColoring<T_Config>(cfg, cfg_scope)
{
    if ( this->m_coloring_level != 1 )
    {
        FatalError( "Not implemented for coloring_level != 1", AMGX_ERR_NOT_SUPPORTED_TARGET );
    }

    if ( cfg.AMG_Config::template getParameter<IndexType>("determinism_flag", "default") )
    {
        m_uncolored_fraction = 0.0;
    }
    else
    {
        m_uncolored_fraction = cfg.AMG_Config::template getParameter<double>("max_uncolored_percentage", cfg_scope);
    }

    m_weakness_bound = cfg.AMG_Config::template getParameter<int>( "weakness_bound", cfg_scope );
    m_late_rejection = cfg.AMG_Config::template getParameter<int>( "late_rejection", cfg_scope ) != 0;
}



#if !NEW_COLORER_TESTS

// Block version
template< AMGX_VecPrecision V, AMGX_MatPrecision M, AMGX_IndPrecision I >
void
Min_Max_2Ring_Matrix_Coloring<TemplateConfig<AMGX_device, V, M, I> >::colorMatrix( Matrix_d &A )
{
    ViewType oldView = A.currentView();
    this->m_row_colors.resize( A.row_offsets.size() - 1 );

    if  (this->m_halo_coloring == SYNC_COLORS) { A.setView(ALL); }
    else { A.setViewExterior(); }

    const int num_rows = A.get_num_rows();
    const int max_uncolored_rows = static_cast<int>( this->m_uncolored_fraction * num_rows );
    const int CTA_SIZE = 128;
    const int NUM_WARPS_PER_CTA = CTA_SIZE / 32;
    const int GRID_SIZE = std::min( 2048, (num_rows + NUM_WARPS_PER_CTA - 1) / NUM_WARPS_PER_CTA );
    this->m_num_colors = 1;
    thrust::fill( this->m_row_colors.begin(), this->m_row_colors.end(), 0 );
    cudaCheckError();
    device_vector_alloc<int> gtlt_count( num_rows );

    for ( int num_uncolored = num_rows ; num_uncolored > max_uncolored_rows ; )
    {
        count_gtlt_kernel<CTA_SIZE, 32> <<< GRID_SIZE, CTA_SIZE>>>(
            num_rows,
            A.row_offsets.raw(),
            A.col_indices.raw(),
            this->m_row_colors.raw(),
            thrust::raw_pointer_cast( &gtlt_count.front() ), 0);
        cudaCheckError();

        if ( this->m_late_rejection )
            color_kernel<CTA_SIZE, 32, true> <<< GRID_SIZE, CTA_SIZE>>>(
                num_rows,
                A.row_offsets.raw(),
                A.col_indices.raw(),
                thrust::raw_pointer_cast( &gtlt_count.front() ),
                this->m_num_colors,
                this->m_weakness_bound,
                this->m_row_colors.raw() );
        else
            color_kernel<CTA_SIZE, 32, false> <<< GRID_SIZE, CTA_SIZE>>>(
                num_rows,
                A.row_offsets.raw(),
                A.col_indices.raw(),
                thrust::raw_pointer_cast( &gtlt_count.front() ),
                this->m_num_colors,
                this->m_weakness_bound,
                this->m_row_colors.raw() );

        cudaCheckError();
#if 0
        device_vector_alloc<int> error_found( 1, 0 );
        dbg_check_coloring_kernel<CTA_SIZE, 32> <<< GRID_SIZE, CTA_SIZE>>>(
            num_rows,
            A.row_offsets.raw(),
            A.col_indices.raw(),
            this->m_row_colors.raw(),
            thrust::raw_pointer_cast( &gtlt_count.front() ),
            thrust::raw_pointer_cast( &error_found.front() ) );
        cudaCheckError();
#endif
        this->m_num_colors += 2;
        num_uncolored = (int) thrust::count_if( this->m_row_colors.begin(), this->m_row_colors.begin() + num_rows, is_zero() );
        cudaCheckError();
    }

    this->m_num_colors = thrust_wrapper::reduce( this->m_row_colors.begin(), this->m_row_colors.begin() + num_rows, 0, thrust::maximum<int>() ) + 1;
    cudaCheckError();
#if 0
    device_vector_alloc<int> error_found( 1, 0 );
    dbg_check_coloring_kernel<CTA_SIZE, 32> <<< GRID_SIZE, CTA_SIZE>>>(
        num_rows,
        A.row_offsets.raw(),
        A.col_indices.raw(),
        this->m_row_colors.raw(),
        NULL,
        thrust::raw_pointer_cast( &error_found.front() ) );
    cudaCheckError();

    if ( error_found[0] != 0 )
    {
        std::cout << "INVALID COLORING !!! Two neighbors have the same color!!!" << std::endl;
    }

#endif
    A.setView(oldView);
}


#else

template< int CTA_SIZE, int WARP_SIZE >
__global__
void
dbg_coloring_histogram_kernel( int *colors_count, const int A_num_rows, const int *A_colors, const int n_colors )
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    __shared__ unsigned int color_counts[256];

    if (threadIdx.x < n_colors)
    {
        color_counts[threadIdx.x] = 0;
    }

    __syncthreads();

    for (; tid < A_num_rows; tid += blockDim.x * gridDim.x)
    {
        int col = A_colors[tid];
        atomicAdd(color_counts + col, 1);
    }

    __syncthreads();

    if (threadIdx.x < n_colors)
    {
        atomicAdd(colors_count + threadIdx.x, color_counts[threadIdx.x]);
    }
}

template< AMGX_VecPrecision V, AMGX_MatPrecision M, AMGX_IndPrecision I >
void
Min_Max_2Ring_Matrix_Coloring<TemplateConfig<AMGX_device, V, M, I> >::debug_coloring(Matrix_d &A, int step)
{
#if COLORING_DEBUG
    const int num_rows = A.get_num_rows();
    const int CTA_SIZE = 128;
    const int NUM_WARPS_PER_CTA = CTA_SIZE / 32;
    const int GRID_SIZE = std::min( 2048, (num_rows + NUM_WARPS_PER_CTA - 1) / NUM_WARPS_PER_CTA );
    int num_uncolored = (int) thrust::count_if( this->m_row_colors.begin(), this->m_row_colors.begin() + num_rows, is_zero() );
    cudaCheckError();
    int maxr = A.row_offsets[1] - A.row_offsets[0];
    /*for(int i=2;i<=num_rows;i++)
    {
     int d=A.row_offsets[i]-A.row_offsets[i-1];
     if(d>maxr)
     {
        maxr=d;
     }
    }*/
    device_vector_alloc<int> error_found( 2, 0 );
    error_found[0] = 0;
    error_found[1] = 0;
    dbg_check_coloring_kernel<CTA_SIZE, 32> <<< GRID_SIZE, CTA_SIZE>>>(
        num_rows,
        A.row_offsets.raw(),
        A.col_indices.raw(),
        this->m_row_colors.raw(),
        0,//thrust::raw_pointer_cast( &gtlt_count.front() ),
        thrust::raw_pointer_cast( &error_found.front() ) );
    cudaCheckError();
    {
        device_vector_alloc<int> color_histogram(this->m_num_colors + 1);

        for (int i = 0; i < color_histogram.size(); i++)
        {
            color_histogram[i] = 0;
        }

        dbg_coloring_histogram_kernel<CTA_SIZE, 32> <<< GRID_SIZE, CTA_SIZE>>>( thrust::raw_pointer_cast(&color_histogram.front()), num_rows, this->m_row_colors.raw(), this->m_num_colors + 1);
        cudaCheckError();

        for (int i = 0; i < color_histogram.size(); i++)
        {
            std::cout << step << "\t" << "H[" << i << "] = " << color_histogram[i] << std::endl;
        }

        std::cout << step << "\t" << "Errors=" << error_found[1] << std::endl;
        std::cout << step << "\t" << "Uncolored=" << num_uncolored << std::endl;
        std::cout << step << "\t" << "Num colors=" << this->m_num_colors << "/" << maxr << std::endl;
    }
#endif
}

template< int CTA_SIZE, int WARP_SIZE, bool LATE_REJECTION >
__global__
void color_kernel_greedy( const int A_num_rows, const int *A_rows, const int *A_cols, const int *A_gtlt_count, const int current_color, const int weakness_bound, int *A_colors )
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
        int dice = hash_function(row_id, 19881988, 0);
        int row_color = A_colors[row_id];

        if ( row_color != 0 ) // Already colored!!!
        {
            continue;
        }

        // The number of vertices that are greater/smaller than me.
        int row_gtlt_count = A_gtlt_count[row_id];
        // Split gtlt_count into 2.
        int row_gt_count = row_gtlt_count >> 16;
        int row_lt_count = row_gtlt_count  & 0xffff;
        // Do we skip it.
        int candidate = 1;
        // Predicates. Is a vertex min/max.
        int is_max_vertex = true;//row_gt_count <= weakness_bound;
        int is_min_vertex = true;//row_lt_count <= weakness_bound;
        // Iterators over my row.
        int row_begin = A_rows[row_id  ];
        int row_end   = A_rows[row_id + 1];
        unsigned long long used_colors = 0ull;

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

            if ( row_it < row_end && col_id < A_num_rows )
            {
                int col_color = A_colors[col_id];

                if ( col_color > 0 )
                {
                    used_colors |= (1ull << (64 - col_color));
                }
            }

            // Get the gt/lt count.
            int col_gtlt_count = -1;

            if ( row_it < row_end && col_id < A_num_rows )
            {
                col_gtlt_count = A_gtlt_count[col_id];
            }

            // Split gtlt_count into 2. col_gt/lt_count == -1 if already colored.
            int col_gt_count = col_gtlt_count >> 16;
            int col_lt_count = col_gtlt_count  & 0xffff;

            // Threads determine if they are greater than the row hash.
            //if( col_gt_count != -1 )
            //  is_max_vertex &= col_gt_count < row_gt_count || (col_gt_count == row_gt_count && col_id < row_id);
            //if( col_lt_count != -1 )
            //  is_min_vertex &= col_lt_count /*>*/ < row_lt_count || (col_lt_count == row_lt_count && col_id > row_id);
            if ( col_gtlt_count != -1 )
            {
                is_max_vertex &= col_gt_count > row_gt_count || (col_gt_count == row_gt_count && col_id >= row_id);
                is_min_vertex &= col_lt_count > row_lt_count || (col_lt_count == row_lt_count && col_id >= row_id);
            }
        }

        is_min_vertex = false;
        //reduce used colors bit by bit.
#pragma unroll
        for (int i = WARP_SIZE / 2; i >= 1; i /= 2)
        {
            int tmp_hi = __double2hiint( __longlong_as_double( used_colors ) );
            int tmp_lo = __double2hiint( __longlong_as_double( used_colors ) );
            tmp_hi = utils::shfl_xor(tmp_hi, i, WARP_SIZE);
            tmp_lo = utils::shfl_xor(tmp_lo, i, WARP_SIZE);
            long long tmp = __double_as_longlong(__hiloint2double(tmp_hi, tmp_lo));
            used_colors |= tmp;
        }

        int my_color = 64 - utils::bfind( ~used_colors );

        if (my_color <= 0) { my_color = 1; }

        /*int sets=0;
        for(int c=1; c<=64; c++)
        {
            unsigned long long int b = (1ull << (64-c));
            if((~used_colors & b) && sets < (dice % 3)-1)
            {
                sets++;
                my_color = c;
            }
        }*/

        // The warp leader stores the result.
        if ( candidate && utils::all( is_max_vertex ) )
        {
            if ( lane_id == 0 )
            {
                A_colors[row_id] = my_color;
            }

            continue;
        }

        if ( candidate && utils::all( is_min_vertex ) )
        {
            if ( lane_id == 0 )
            {
                A_colors[row_id] = my_color + 1;
            }

            continue;
        }
    }
}


template< int CTA_SIZE, int WARP_SIZE>
__global__
void color_kernel_reassign_tail_thread( const int A_num_rows, const int *A_rows, const int *A_cols, int num_colors, int first_tail_color, int *A_colors )
{
    const int NUM_THREADS_PER_GRID = gridDim.x * CTA_SIZE;
    // Row identifier.
    int row_id = blockIdx.x * CTA_SIZE + threadIdx.x;

    // Iterate over rows.
    for ( ; row_id < A_num_rows ; row_id += NUM_THREADS_PER_GRID )
    {
        unsigned long long used_colors = 0ull;
        int row_color = A_colors[row_id];

        if ( row_color < first_tail_color ) // Already colored!!!
        {
            continue;
        }

        int row_hash = hash_function(row_id, 198719871987L, 0);
        int row_begin = A_rows[row_id  ];
        int row_end   = A_rows[row_id + 1];

        for ( int row_it = row_begin; row_it < row_end ; row_it++)
        {
            int col_id = A_cols[row_it];

            if (col_id >= A_num_rows) { continue; }

            int col_hash = hash_function( col_id, 1987, 0 );
            int col_color = A_colors[col_id];
        }
    }
}

__global__ void unassign_color(int *A_colors, int num_rows)
{
    for (int tid = threadIdx.x + blockIdx.x * blockDim.x; tid < num_rows; tid += gridDim.x * blockDim.x)
    {
        if (tid % 9 == 0)
        {
            A_colors[tid] = 0;
        }
    }
}



// Block version
template< AMGX_VecPrecision V, AMGX_MatPrecision M, AMGX_IndPrecision I >
void
Min_Max_2Ring_Matrix_Coloring<TemplateConfig<AMGX_device, V, M, I> >::color_step( Matrix_d &A, int seed )
{
    ViewType oldView = A.currentView();
    this->m_row_colors.resize( A.row_offsets.size() - 1 );

    if  (this->m_halo_coloring == SYNC_COLORS) { A.setView(ALL); }
    else { A.setViewExterior(); }

    const int num_rows = A.get_num_rows();
    const int max_uncolored_rows = static_cast<int>( this->m_uncolored_fraction * num_rows );
    const int CTA_SIZE = 128;
    const int NUM_WARPS_PER_CTA = CTA_SIZE / 32;
    const int GRID_SIZE = std::min( 2048, (num_rows + NUM_WARPS_PER_CTA - 1) / NUM_WARPS_PER_CTA );
    device_vector_alloc<int> gtlt_count( num_rows );

    for ( int num_uncolored = num_rows ; num_uncolored > max_uncolored_rows ; )
    {
        count_gtlt_kernel<CTA_SIZE, 32> <<< GRID_SIZE, CTA_SIZE>>>(
            num_rows,
            A.row_offsets.raw(),
            A.col_indices.raw(),
            this->m_row_colors.raw(),
            thrust::raw_pointer_cast( &gtlt_count.front() ),
            seed);
        cudaCheckError();

        if ( this->m_late_rejection )
            color_kernel_greedy<CTA_SIZE, 32, true> <<< GRID_SIZE, CTA_SIZE>>>(
                num_rows,
                A.row_offsets.raw(),
                A.col_indices.raw(),
                thrust::raw_pointer_cast( &gtlt_count.front() ),
                this->m_num_colors,
                this->m_weakness_bound,
                this->m_row_colors.raw() );
        else
            color_kernel_greedy<CTA_SIZE, 32, false> <<< GRID_SIZE, CTA_SIZE>>>(
                num_rows,
                A.row_offsets.raw(),
                A.col_indices.raw(),
                thrust::raw_pointer_cast( &gtlt_count.front() ),
                this->m_num_colors,
                this->m_weakness_bound,
                this->m_row_colors.raw() );

        cudaCheckError();
        this->m_num_colors += 2;
        num_uncolored = (int) thrust::count_if( this->m_row_colors.begin(), this->m_row_colors.begin() + num_rows, is_zero() );
        cudaCheckError();
    }

    this->m_num_colors = thrust_wrapper::reduce( this->m_row_colors.begin(), this->m_row_colors.begin() + num_rows, 0, thrust::maximum<int>() ) + 1;
}




// Block version
template< AMGX_VecPrecision V, AMGX_MatPrecision M, AMGX_IndPrecision I >
void
Min_Max_2Ring_Matrix_Coloring<TemplateConfig<AMGX_device, V, M, I> >::color_matrix( Matrix_d &A )
{
    ViewType oldView = A.currentView();
    this->m_row_colors.resize( A.row_offsets.size() - 1 );

    if  (this->m_halo_coloring == SYNC_COLORS) { A.setView(ALL); }
    else { A.setViewExterior(); }

    this->m_num_colors = 1;
    thrust::fill( this->m_row_colors.begin(), this->m_row_colors.end(), 0 );
    cudaCheckError();
    color_step(A, 0);

    A.setView(oldView);
}


#endif


#define AMGX_CASE_LINE(CASE) template class Min_Max_2Ring_Matrix_Coloring_Base<TemplateMode<CASE>::Type>;
AMGX_FORALL_BUILDS(AMGX_CASE_LINE)
#undef AMGX_CASE_LINE

#define AMGX_CASE_LINE(CASE) template class Min_Max_2Ring_Matrix_Coloring<TemplateMode<CASE>::Type>;
AMGX_FORALL_BUILDS(AMGX_CASE_LINE)
#undef AMGX_CASE_LINE

} // end namespace amgx

