// SPDX-FileCopyrightText: 2011 - 2024 NVIDIA CORPORATION. All Rights Reserved.
//
// SPDX-License-Identifier: BSD-3-Clause

#include <basic_types.h>
#include <util.h>
#include <error.h>
#include <types.h>
#include <matrix_coloring/min_max.h>
#include <cusp/format.h>
#include <cusp/copy.h>
#include <cusp/detail/random.h>
#include <thrust/count.h>
#include <thrust/extrema.h>

#include <sm_utils.inl>

#define MAX(a, b) (((a) > (b)) ? (a) : (b))

//#define USE_EXPERIMENTAL_MIN_MAX
#define USE_EXPERIMENTAL_NEIGHBORS

// Pseudo-random number generator
namespace amgx
{

__host__ __device__ unsigned int hash_function(unsigned int a, unsigned int seed)
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

// Kernel to color the rows of the matrix, using min-max approach
#ifdef USE_EXPERIMENTAL_MIN_MAX
template< typename IndexType, int kCtaSize >
__global__
void colorRowsKernel(const IndexType *A_offsets,
                     const IndexType *A_column_indices,
                     IndexType *row_colors,
                     const int current_color,
                     const int num_rows)
{
    const int num_rows_per_grid = gridDim.x * ( kCtaSize / 32 );
    const int lane_id = threadIdx.x % 32;

    for ( int row_id = ( blockIdx.x * kCtaSize + threadIdx.x ) / 32 ; row_id < num_rows ; row_id += num_rows_per_grid )
    {
        int my_row_color = row_colors[row_id];

        if ( my_row_color != 0 )
        {
            continue;
        }

        int hash_i = hash_function(row_id, 0);
        bool max_i = true;
        bool min_i = true;
        int row_it  = A_offsets[row_id  ];
        int row_end = A_offsets[row_id + 1];

        for ( row_it += lane_id ; utils::any(row_it < row_end) ; row_it += 32 )
        {
            int j = row_it < row_end ? A_column_indices[row_it] : -1;
            int hash_j = hash_function(j, 0);
            int row_color_j = -1;

            if ( row_it < row_end && hash_i != hash_j && j < num_rows ) //TODO: is this correct? (ignore edges thru bndry
            {
                row_color_j = row_colors[j];
            }

            // There is an uncolored neighbour that is greater
            max_i &= utils::all( hash_i > hash_j || ( row_color_j != 0 && row_color_j != current_color ) );
            // There is an uncolored neighbour that is smaller
            min_i &= utils::all( hash_i < hash_j || ( row_color_j != 0 && row_color_j != current_color + 1 ) );
        }

        if ( lane_id == 0 && (max_i || min_i) )
        {
            row_colors[row_id] = current_color + (min_i ? 1 : 0);
        }
    }
}
#else
template <typename IndexType>
__global__
void colorRowsKernel(const IndexType *A_offsets, const IndexType *A_column_indices, IndexType *row_colors,
                     const int current_color, const int num_rows)
{
    for ( int i = threadIdx.x + blockDim.x * blockIdx.x; i < num_rows ; i += gridDim.x * blockDim.x )
    {
        int my_row_color = 0;

        // skip if already colored
        if (row_colors[i] == 0)
        {
            int hash_i = hash_function(i, 0);
            bool max_i, min_i;
            max_i = true;
            min_i = true;
            int row_start = A_offsets[i];
            int row_end = A_offsets[i + 1];

            for (int r = row_start; r < row_end; r++)
            {
                int j = A_column_indices[r];

                if (j >= num_rows) { continue; }

                int hash_j = hash_function(j, 0);
                int row_color_j = row_colors[j];

                // There is an uncolored neighbour that is greater
                if ( hash_j > hash_i && (row_color_j == 0 || row_color_j == current_color) )
                {
                    max_i = false;
                }

                // There is an uncolored neighbour that is smaller
                if (hash_j < hash_i && (row_color_j == 0 || row_color_j == current_color + 1) )
                {
                    min_i = false;
                }
            }

            if (max_i)
            {
                my_row_color = current_color;
            }
            else if (min_i)
            {
                my_row_color = current_color + 1;
            }

            if (my_row_color != 0)
            {
                row_colors[i] = my_row_color;
            }
        }
    }
}
#endif

#ifdef USE_EXPERIMENTAL_NEIGHBORS

template< int CTA_SIZE, int NUM_THREADS_PER_ROW >
__global__ __launch_bounds__( CTA_SIZE )
void
find_min_max_neighbors_kernel( const int *__restrict A_offsets,
                               const int *__restrict A_column_indices,
                               const int *__restrict row_colors,
                               int *__restrict max_hash_array,
                               int *__restrict min_hash_array,
                               const int current_color,
                               const int num_rows )
{
    const int NUM_ROWS_PER_CTA = CTA_SIZE / NUM_THREADS_PER_ROW;
    const int warp_id = threadIdx.x / NUM_THREADS_PER_ROW;
    const int lane_id = threadIdx.x % NUM_THREADS_PER_ROW;

    for ( int row_id = blockIdx.x * NUM_ROWS_PER_CTA + warp_id ; utils::any(row_id < num_rows) ; row_id += gridDim.x * NUM_ROWS_PER_CTA )
    {
        int max_hash = INT_MIN;
        int min_hash = INT_MAX;

        if ( lane_id == 0 && row_id < num_rows )
        {
            int hash_i = hash_function(row_id, 0);

            if ( row_colors[row_id] == 0 )
            {
                max_hash = hash_i;
                min_hash = hash_i;
            }
        }

        int row_it = 0, row_end = 0;

        if ( row_id < num_rows )
        {
            row_it  = A_offsets[row_id + 0];
            row_end = A_offsets[row_id + 1];
        }

        for ( row_it += lane_id ; utils::any(row_it < row_end) ; row_it += NUM_THREADS_PER_ROW )
        {
            int a_col_id = -1;

            if ( row_it < row_end )
            {
                a_col_id = A_column_indices[row_it];
            }

            int hash_j = 0;
            int col_color = -1;

            if ( a_col_id != -1 && a_col_id < num_rows )
            {
                hash_j    = hash_function(a_col_id, 0);
                col_color = row_colors[a_col_id];
            }

            // There is an uncolored neighbour that is greater
            if ( col_color == 0 && hash_j > max_hash )
            {
                max_hash = hash_j;
            }

            // There is an uncolored neighbour that is smaller
            if ( col_color == 0 && hash_j < min_hash )
            {
                min_hash = hash_j;
            }
        }

#pragma unroll
        for ( int mask = NUM_THREADS_PER_ROW / 2 ; mask > 0 ; mask >>= 1 )
        {
            min_hash = min( min_hash, utils::shfl_xor( min_hash, mask ) );
            max_hash = max( max_hash, utils::shfl_xor( max_hash, mask ) );
        }

        if ( row_id < num_rows && lane_id == 0 )
        {
            max_hash_array[row_id] = max_hash;
            min_hash_array[row_id] = min_hash;
        }
    }
}

#else

template <typename IndexType>
__global__
void FindMaxMinNeighboursKernel(const IndexType *A_offsets, const IndexType *A_column_indices, IndexType *row_colors, IndexType *max_hash_array, IndexType *min_hash_array,
                                const int current_color, const int num_rows)
{
    for ( int i = threadIdx.x + blockDim.x * blockIdx.x; i < num_rows ; i += gridDim.x * blockDim.x )
    {
        int max_hash;
        int min_hash;
        int hash_i = hash_function(i, 0);
        int row_col = row_colors[i];
        max_hash = (row_col == 0) ? hash_i : INT_MIN;
        min_hash = (row_col == 0) ? hash_i : INT_MAX;
        int row_start = A_offsets[i];
        int row_end = A_offsets[i + 1];

        for (int r = row_start; r < row_end; r++)
        {
            int j = A_column_indices[r];

            if (j >= num_rows) { continue; }

            int hash_j = hash_function(j, 0);
            int row_color_j = row_colors[j];
            // Only consider if neighbour is not colored
            // There is an uncolored neighbour that is greater
            max_hash = (row_color_j == 0 && hash_j > max_hash) ? hash_j : max_hash;
            // There is an uncolored neighbour that is smaller
            min_hash = (row_color_j == 0 && hash_j < min_hash) ? hash_j : min_hash;
        }

        max_hash_array[i] = max_hash;
        min_hash_array[i] = min_hash;
    }
}

#endif

template <typename IndexType>
__global__
void colorRowsRingTwoKernel(const IndexType *A_offsets, const IndexType *A_column_indices, IndexType *row_colors, const IndexType *max_hash_array, const IndexType *min_hash_array,
                            const int current_color, const int num_rows)
{
    for ( int i = threadIdx.x + blockDim.x * blockIdx.x; i < num_rows ; i += gridDim.x * blockDim.x )
    {
        int my_row_color = 0;

        // skip if already colored
        if (row_colors[i] == 0)
        {
            int hash_i = hash_function(i, 0);
            int max_hash = max_hash_array[i];
            int min_hash = min_hash_array[i];
            bool max_i, min_i;
            max_i = true;
            min_i = true;

            // Check if I was a max or min in previous step
            if ( hash_i != max_hash && hash_i != min_hash)
            {
                continue;
            }

            if (hash_i != min_hash) { min_i = false; }

            if (hash_i != max_hash) { max_i = false; }

            int row_start = A_offsets[i];
            int row_end = A_offsets[i + 1];

            for (int r = row_start; r < row_end; r++)
            {
                int j = A_column_indices[r];

                if (j >=  num_rows) { continue; }

                int min_hash_j = min_hash_array[j];
                int max_hash_j = max_hash_array[j];
                // There is a neighbour that has a uncolored neighbour with larger hash
                max_i = (max_hash_j > max_hash && max_hash_j != INT_MIN) ? false : max_i;
                // There is a neighbour that has a uncolored neighbour with smaller hash
                min_i = (min_hash_j < min_hash && min_hash_j != INT_MAX) ? false : min_i;
            }

            if (max_i && min_i)
            {
                my_row_color = (i % 2) ? current_color : current_color + 1;
            }
            else if (max_i)
            {
                my_row_color = current_color;
            }
            else if (min_i)
            {
                my_row_color = current_color + 1;
            }

            if (my_row_color != 0)
            {
                row_colors[i] = my_row_color;
            }
        }
    }
}



// ---------------------------
// Methods
// ---------------------------

template<class T_Config>
MinMaxMatrixColoringBase<T_Config>::MinMaxMatrixColoringBase(AMG_Config &cfg, const std::string &cfg_scope) : MatrixColoring<T_Config>(cfg, cfg_scope)
{
    if (cfg.AMG_Config::template getParameter<IndexType>("determinism_flag", "default"))
    {
        m_uncolored_fraction = 0;
    }
    else
    {
        m_uncolored_fraction = cfg.AMG_Config::template getParameter<double>("max_uncolored_percentage", cfg_scope);
    }
}



// Block version
template <AMGX_VecPrecision t_vecPrec, AMGX_MatPrecision t_matPrec, AMGX_IndPrecision t_indPrec>
void MinMaxMatrixColoring<TemplateConfig<AMGX_device, t_vecPrec, t_matPrec, t_indPrec> >::colorMatrixOneRing(Matrix_d &A)
{
    const int num_rows =  A.get_num_rows();
    int max_uncolored_rows = (int) (this->m_uncolored_fraction * ((ValueType) num_rows));
    IndexType *row_colors_ptr = this->m_row_colors.raw();
    const int threads_per_block = 256;
    const int warps_per_block = threads_per_block / 32;
    const int num_blocks = std::min( AMGX_GRID_MAX_SIZE, (num_rows + warps_per_block - 1) / warps_per_block );
    this->m_num_colors = 1;
    thrust_wrapper::fill<AMGX_device>(this->m_row_colors.begin(), this->m_row_colors.end(), 0);
    cudaCheckError();

    for ( int num_uncolored = num_rows; num_uncolored > max_uncolored_rows ; )
    {
#ifdef USE_EXPERIMENTAL_MIN_MAX
        colorRowsKernel<IndexType, threads_per_block> <<< num_blocks, threads_per_block, 0, amgx::thrust::global_thread_handle::get_stream()>>>(
            A.row_offsets.raw(),
            A.col_indices.raw(),
            row_colors_ptr,
            this->m_num_colors,
            num_rows);
            cudaCheckError();
#else
        colorRowsKernel<IndexType> <<< num_blocks, threads_per_block, 0, amgx::thrust::global_thread_handle::get_stream()>>>(A.row_offsets.raw(), A.col_indices.raw(), row_colors_ptr, this->m_num_colors, num_rows);
        cudaCheckError();
#endif
        cudaCheckError();
        this->m_num_colors += 2;
        num_uncolored = (int) amgx::thrust::count_if( this->m_row_colors.begin(), this->m_row_colors.begin() + num_rows, is_zero() );
        cudaCheckError();
    }

    this->m_num_colors = thrust_wrapper::reduce<AMGX_device>( this->m_row_colors.begin(), this->m_row_colors.begin() + num_rows, 0, amgx::thrust::maximum<int>() ) + 1;
    cudaCheckError();
}

// Block version
template <class T_Config>
void MinMaxMatrixColoringBase<T_Config>::colorMatrix(Matrix<T_Config> &A)
{
    ViewType oldView = A.currentView();
    this->m_row_colors.resize(A.row_offsets.size() - 1);

    if  (this->m_halo_coloring == SYNC_COLORS) { A.setView(ALL); }
    else { A.setViewExterior(); }

    if (this->m_coloring_level == 0)
    {
        FatalError("Calling coloring scheme but coloring level==0", AMGX_ERR_NOT_SUPPORTED_TARGET);
    }
    else if (this->m_coloring_level == 1)
    {
        this->colorMatrixOneRing(A);
    }
    else if (this->m_coloring_level == 2)
    {
        this->colorMatrixTwoRing(A);
    }
    else
    {
        FatalError("Min-max coloring does not support coloring_level > 2", AMGX_ERR_NOT_SUPPORTED_TARGET);
    }

    A.setView(oldView);
}

template <AMGX_VecPrecision t_vecPrec, AMGX_MatPrecision t_matPrec, AMGX_IndPrecision t_indPrec>
void MinMaxMatrixColoring<TemplateConfig<AMGX_device, t_vecPrec, t_matPrec, t_indPrec> >::colorMatrixTwoRing(Matrix_d &A)
{
    const int num_rows = A.get_num_rows();
    const int max_uncolored_rows = (int) (this->m_uncolored_fraction * ((ValueType) num_rows));
    const int threads_per_block = 256;
#ifdef USE_EXPERIMENTAL_NEIGHBORS
    const int half_warps_per_block = threads_per_block / 16;
    const int num_blocks = std::min( AMGX_GRID_MAX_SIZE, (num_rows + half_warps_per_block - 1) / half_warps_per_block );
#else
    const int warps_per_block = threads_per_block / 32;
    const int num_blocks = std::min( AMGX_GRID_MAX_SIZE, (num_rows + warps_per_block - 1) / warps_per_block );
#endif
    this->m_num_colors = 1;
    thrust_wrapper::fill<AMGX_device>(this->m_row_colors.begin(), this->m_row_colors.end(), 0);
    cudaCheckError();
    IVector max_hash_array(num_rows);
    IVector min_hash_array(num_rows);
    float avg_nnz_per_row = A.get_num_nz() / float(num_rows);

    for ( int num_uncolored = num_rows; num_uncolored > max_uncolored_rows ; )
    {
        // Each vertex checks its neighbours and store the max and min values of its neighbours
#ifdef USE_EXPERIMENTAL_NEIGHBORS
        if ( avg_nnz_per_row < 8.0f )
        {
            find_min_max_neighbors_kernel<threads_per_block, 2> <<< num_blocks, threads_per_block>>>( A.row_offsets.raw(), A.col_indices.raw(), this->m_row_colors.raw(), max_hash_array.raw(), min_hash_array.raw(), this->m_num_colors, num_rows );
            cudaCheckError();
        }
        else if ( avg_nnz_per_row < 16.0f )
        {
            find_min_max_neighbors_kernel<threads_per_block, 4> <<< num_blocks, threads_per_block>>>( A.row_offsets.raw(), A.col_indices.raw(), this->m_row_colors.raw(), max_hash_array.raw(), min_hash_array.raw(), this->m_num_colors, num_rows );
            cudaCheckError();
        }
        else // if( avg_nnz_per_row < 32.0f )
        {
            find_min_max_neighbors_kernel<threads_per_block, 8> <<< num_blocks, threads_per_block>>>( A.row_offsets.raw(), A.col_indices.raw(), this->m_row_colors.raw(), max_hash_array.raw(), min_hash_array.raw(), this->m_num_colors, num_rows );
            cudaCheckError();
        }

#else
        FindMaxMinNeighboursKernel <<< num_blocks, threads_per_block>>>(A.row_offsets.raw(), A.col_indices.raw(), this->m_row_colors.raw(), max_hash_array.raw(), min_hash_array.raw(), this->m_num_colors, num_rows);
        cudaCheckError();
#endif
        // Each vertex checks if its still the min or max
        colorRowsRingTwoKernel<IndexType> <<< num_blocks, threads_per_block>>>(A.row_offsets.raw(), A.col_indices.raw(), this->m_row_colors.raw(), max_hash_array.raw(), min_hash_array.raw(), this->m_num_colors, num_rows);
        cudaCheckError();
        this->m_num_colors += 2;
        num_uncolored = (int) amgx::thrust::count_if( this->m_row_colors.begin(), this->m_row_colors.begin() + num_rows, is_zero() );
        cudaCheckError();
    }

    this->m_num_colors = thrust_wrapper::reduce<AMGX_device>( this->m_row_colors.begin(), this->m_row_colors.begin() + num_rows, 0, amgx::thrust::maximum<int>() ) + 1;
    cudaCheckError();
#if 0
    std::cout << "Num colors=" << this->m_num_colors << std::endl;
    // Print coloring statistics
    //for (int i=0;i<this->m_num_colors;i++)
    //  std::cout << "color= " << i << ", num_rows = " << A.offsets_rows_per_color[i+1] - A.offsets_rows_per_color[i] << std::endl;
#endif
}

template <AMGX_VecPrecision t_vecPrec, AMGX_MatPrecision t_matPrec, AMGX_IndPrecision t_indPrec>
void MinMaxMatrixColoring<TemplateConfig<AMGX_host, t_vecPrec, t_matPrec, t_indPrec> >::colorMatrixOneRing(Matrix_h &A)
{
    FatalError("Haven't implemented MinMax matrix coloring for host format", AMGX_ERR_NOT_SUPPORTED_TARGET);
}

template <AMGX_VecPrecision t_vecPrec, AMGX_MatPrecision t_matPrec, AMGX_IndPrecision t_indPrec>
void MinMaxMatrixColoring<TemplateConfig<AMGX_host, t_vecPrec, t_matPrec, t_indPrec> >::colorMatrixTwoRing(Matrix_h &A)
{
    FatalError("Haven't implemented MinMax matrix coloring for host format", AMGX_ERR_NOT_SUPPORTED_TARGET);
}


#define AMGX_CASE_LINE(CASE) template class MinMaxMatrixColoringBase<TemplateMode<CASE>::Type>;
AMGX_FORALL_BUILDS(AMGX_CASE_LINE)
#undef AMGX_CASE_LINE

#define AMGX_CASE_LINE(CASE) template class MinMaxMatrixColoring<TemplateMode<CASE>::Type>;
AMGX_FORALL_BUILDS(AMGX_CASE_LINE)
#undef AMGX_CASE_LINE


} // end namespace amgx

