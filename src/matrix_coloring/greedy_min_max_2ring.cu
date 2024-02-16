// SPDX-FileCopyrightText: 2011 - 2024 NVIDIA CORPORATION. All Rights Reserved.
//
// SPDX-License-Identifier: BSD-3-Clause

#include <algorithm>
#include <basic_types.h>
#include <util.h>
#include <error.h>
#include <types.h>
#include <matrix_coloring/greedy_min_max_2ring.h>
#include <cusp/format.h>
#include <cusp/copy.h>
#include <cusp/detail/random.h>
#include <thrust/count.h>
#include <thrust/extrema.h>
#include <sm_utils.inl>
#include <thrust/remove.h>
#include  <matrix_coloring/coloring_utils.h>

// Pseudo-random number generator
namespace amgx
{

static __host__ __device__ __forceinline__ unsigned int hash_function(unsigned int a, unsigned int seed, unsigned int rows = 0)
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

// ---------------------------
// Kernels: Any ring parallel greedy with color buffers and max id propagation.
// ---------------------------


/*
 * This kernel computes the index of the node with maximum hash in a neighborhood and the colors used in a vertex neighborhood.
 * Since both operation are associative (max and or), it is possible to iterate the operation propagating the values across neigborhoods (first, second and so on rings).
 *
 * The N-th iteration computes, for each vertex:
 * 1. The ID of the uncolored vertex with ID of the vertex having max hash inside the N-ring neighborhood.
 *    If all neighbors in the N-hood are colored, saves -1 (this allows work efficiency).
 * 2. The buffer of used colors in the n-ring neighborhood.
 *
 * And for that, it uses data from previous iteration.
 *
 * For avoiding the waste of work, once any n-ring neighborhood is all colored, it starts getting ignored:
 *  no new MAX HASH can be found there and the set of used colors for that ring will not change.
 */


template< bool USE_EXT, int CTA_SIZE, int WARP_SIZE >
__global__
void find_max_neighbor_kernel_and_propagate_used_colors( const int A_num_rows,
        const int *A_rows,
        const int *A_cols,
        const int *__restrict A_colors,

        int *level_n0_max_id_in, //for multiple iterations. load the ID of the neighbor with max hash in n-1 ring hood
        int *level_n1_out_max_id,//for storing the ID of the n-ring neighbor with max hash (in the n-ring hood)

        unsigned long long int *level_n0_used_colors_in, //for multiple iterations: load used color buffers from n-1 level
        unsigned long long int *level_n1_used_colors_out,//for storing the used color buffers of the n-level neighborhood of each vertex

        //int* interesting,
        const int currentSafeColor, //starts assigning colors from this. multiple of 64
        const int seed)
{
    const int NUM_WARPS_PER_CTA  = CTA_SIZE / WARP_SIZE;
    const int NUM_WARPS_PER_GRID = gridDim.x * NUM_WARPS_PER_CTA;
    // Thread coordinates.
    const int warp_id = subwarp<WARP_SIZE>::subwarp_id();
    const int lane_id = subwarp<WARP_SIZE>::sublane_id();
    // Row identifier.
    int row_id = blockIdx.x * NUM_WARPS_PER_CTA + warp_id;

    // Iterate over the rows of the matrix.
    for ( ; row_id < A_num_rows ; row_id += NUM_WARPS_PER_GRID )
    {
        const int n1_max_id = level_n1_out_max_id[row_id];

        if (n1_max_id == -1)
        {
            //-1 means that all the vertices in this n-ring
            //level neighborhood were colored during the previous coloring epochs.
            //Since nothing could have changed, just skip:
            //save work and use the color buffers computed in last step.
            continue;
        }

        //If the old max 'n1_max_id' is still uncolored, it still will be the max in this iteration.
        //Plus, nothing else would get colored in its n1-hood, since 'n1_max_id' is the max.
        if (A_colors[n1_max_id] <= 0)
        {
            continue;
        }

        int row_color = A_colors[row_id];
        // Iterators over my row.
        int row_begin = A_rows[row_id  ];
        int row_end   = A_rows[row_id + 1];
        unsigned long long used_colors = 0ull;

        //For multiple iterations: load the used color buffer from previous iter.
        if (USE_EXT)
        {
            used_colors = level_n0_used_colors_in[row_id];
        }

        if (row_color - currentSafeColor > 0) //if the color belongs to the current buffer [currentSafeColor,currentSafeColor+64)
        {
            used_colors |= (1ull << (64 - row_color + currentSafeColor)); //add my color to buffer
        }

        // Hash my row id.
        int max_hash = hash_function( row_id, seed, 0 );
        int max_hash_id = row_id; //id of the neighbor with max hash, initialized with myself

        //If already colored, don't count me for finding max
        if (row_color != 0)
        {
            max_hash_id = -1; //-1 means no max
            max_hash = 0;//LONG_LONG_MIN;
        }

        //iterate the neighbors
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
            int col_hash = hash_function( col_id, seed, 0);
            // Get the color of the column.
            int col_color = -1;

            if ( row_it < row_end && col_id < A_num_rows)
            {
                col_color = A_colors[col_id];

                if ( col_color - currentSafeColor > 0 ) //if in the current buffer..
                {
                    used_colors |= (1ull << (64 - col_color + currentSafeColor)); //..add it to buffer
                }

                if (USE_EXT)
                {
                    //combine used_colors buffer with the ones computed in previous iterations
                    used_colors |= level_n0_used_colors_in[col_id];
                    col_id = level_n0_max_id_in[col_id];

                    if (col_id < 0) //row and its k-1 level neigbhors all colored
                    {
                        col_hash = 0;//LONG_LONG_MIN;
                        col_color = -1;
                    }
                    else
                    {
                        col_hash  = hash_function(col_id, seed, 0); //col hash is the max hash of its neighbors in n-1 level
                        col_color = 0; //there is an uncolored neighbor
                    }
                }
            }

            //check if any neighbor has a greater max id.
            if (col_color == 0 && (max_hash_id < 0  || col_hash > max_hash || (col_hash == max_hash && col_id >= max_hash_id)))
            {
                max_hash_id = col_id;
                max_hash = col_hash;
            }
        }

        //BEGIN: Reduce used_colors/max_hashes amongst subwarps
#pragma unroll

        for (int i = WARP_SIZE / 2; i >= 1; i /= 2)
        {
            int tmp    = utils::shfl_xor(max_hash, i, WARP_SIZE); //ss_max_hash[threadIdx.x+i];
            int tmp_id = utils::shfl_xor(max_hash_id, i, WARP_SIZE);

            if (tmp_id >= 0 && (max_hash_id < 0 || tmp > max_hash || (tmp == max_hash && tmp_id >= max_hash_id)))
            {
                max_hash = tmp;
                max_hash_id = tmp_id;
            }

            //
            int tmp_hi = __double2hiint( __longlong_as_double( used_colors ) );
            int tmp_lo = __double2loint( __longlong_as_double( used_colors ) );
            tmp_hi = utils::shfl_xor(tmp_hi, i, WARP_SIZE);
            tmp_lo = utils::shfl_xor(tmp_lo, i, WARP_SIZE);
            long long tmpu = __double_as_longlong(__hiloint2double(tmp_hi, tmp_lo));
            used_colors |= tmpu;
        }

        //END: Reduce used_colors/max_hashes amongst subwarps

        //The subwarp leader stores the result.
        if ( lane_id == 0 )
        {
            //if max_hash_id=1, will skip work next time: every n-ring neighbor is colored
            level_n1_out_max_id[row_id] = max_hash_id;
            level_n1_used_colors_out[row_id] = used_colors;
        }
    }
}


/*
 * This kernel colors rows using a greedy strategy and the used colors buffer from propagate iterations.
 */
template< int CTA_SIZE, int WARP_SIZE >
__global__
void color_kernel_greedy_onlymax(
    const int A_num_rows, const int *A_rows, const int *A_cols,
    const int *max_hash_id_in_neighborhood,

    const int seed,
    int *A_colors,
    const unsigned long long *used_colors_in,
    int *finished_colors_flag,
    const int currentSafeColor)
{
    const int NUM_WARPS_PER_CTA  = CTA_SIZE / WARP_SIZE;
    const int NUM_WARPS_PER_GRID = gridDim.x * NUM_WARPS_PER_CTA;
    // Thread coordinates.
    const int warp_id = subwarp<WARP_SIZE>::subwarp_id();
    const int lane_id = subwarp<WARP_SIZE>::sublane_id();
    // Row identifier.
    int row_id = blockIdx.x * NUM_WARPS_PER_CTA + warp_id;

    // Iterate over the rows of the matrix.
    for ( ; row_id < A_num_rows ; row_id += NUM_WARPS_PER_GRID )
    {
        //int dice = hash_function(row_id, 19881988, 0);
        int row_color = A_colors[row_id];

        if ( row_color != 0 ) // Already colored!!!
        {
            continue;
        }

        //int row_hash = hash_function(row_id,seed,0);
        // The number of vertices that are greater/smaller than me.
        int max_id = max_hash_id_in_neighborhood[row_id];

        if (max_id != row_id) //was not a max in its own neighborhood
        {
            continue;
        }

        int max_hash = hash_function(max_id, seed, 0);
        //A_colors[row_id] = 1;
        //continue;
        bool is_max_vertex = true;
        //bool all_neighbors_colored = true;
        // Iterators over my row.
        int row_begin = A_rows[row_id  ];
        int row_end   = A_rows[row_id + 1];
        unsigned long long used_colors = 0ull;
        used_colors |= used_colors_in[row_id];

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

            // Filter out used colors
            if ( row_it < row_end && col_id < A_num_rows)
            {
                //int col_color = A_colors[col_id];
                //if(col_color == 0) all_neighbors_colored = false;
                used_colors |= used_colors_in[col_id];
                int nn_max_id = max_hash_id_in_neighborhood[col_id];
                int nn_max = hash_function(nn_max_id, seed, 0);
                is_max_vertex &= (max_hash > nn_max || (max_hash == nn_max && row_id >= nn_max_id));
            }

            if (!is_max_vertex) { break; }
        }

        //reduce used colors bit by bit.
#pragma unroll

        for (int i = WARP_SIZE / 2; i >= 1; i /= 2)
        {
            int tmp_hi = __double2hiint( __longlong_as_double( used_colors ) );
            int tmp_lo = __double2loint( __longlong_as_double( used_colors ) );
            tmp_hi = utils::shfl_xor(tmp_hi, i, WARP_SIZE);
            tmp_lo = utils::shfl_xor(tmp_lo, i, WARP_SIZE);
            long long tmp = __double_as_longlong(__hiloint2double(tmp_hi, tmp_lo));
            used_colors |= tmp;
        }

        int my_color_1 = 64 - utils::bfind( ~used_colors );

        if (__popc(used_colors) >= 64 || my_color_1 > 64 || my_color_1 <= 0)
        {
            if (lane_id == 0)
            {
                finished_colors_flag[0] = 1;
            }

            is_max_vertex = 0;
        }

        my_color_1 += currentSafeColor;

        if ( subwarp<WARP_SIZE>::all( is_max_vertex ) )
        {
            if ( lane_id == 0 )
            {
                A_colors[row_id] = my_color_1;
            }

            continue;
        }
    }
}


/*
 *
 * One ring parallel greedy with greater hash count metric instead of raw hash metric. Similar to MIN_MAX_2RING
 *
 * */
template< int CTA_SIZE, int WARP_SIZE, bool USE_EXTERNAL_METRIC >
__global__
void count_gtlt_kernel( const int A_num_rows,
                        const int *A_rows,
                        const int *A_cols,
                        const int *__restrict A_colors,
                        int *A_gtlt_count,
                        const int seed,
                        const int *metric_in)
{
    const int NUM_WARPS_PER_CTA  = CTA_SIZE / WARP_SIZE;
    const int NUM_WARPS_PER_GRID = gridDim.x * NUM_WARPS_PER_CTA;
    // Thread coordinates.
    const int warp_id = subwarp<WARP_SIZE>::subwarp_id();
    const int lane_id = subwarp<WARP_SIZE>::sublane_id();
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

        // The number of vertices that are greater/smaller than me.
        int gt_count = 0, lt_count = 0;
        // Iterators over my row.
        int row_begin = A_rows[row_id  ];
        int row_end   = A_rows[row_id + 1];
        // Hash my row id.
        int row_cols = row_end - row_begin;
        int row_hash = hash_function( row_id, seed, row_cols );

        if (USE_EXTERNAL_METRIC)
        {
            row_hash = metric_in[row_id] >> 16; //gt count
        }

        for ( ; row_begin < row_end ; row_begin += WARP_SIZE )
        {
            // Iterator.
            int row_it = row_begin + lane_id;
            // Get the column index (if the iterator is valid).
            int col_id = -1;
            int col_cols = 1;

            if ( row_it < row_end )
            {
                col_id = A_cols[row_it];
#if USE_ROWS
                col_cols = A_rows[col_id + 1] - A_rows[col_id];
#endif
            }

            // Each thread hashes its column id.
            int col_hash = hash_function( col_id, seed, col_cols);
            // Get the color of the column.
            int col_color = -1;

            if ( row_it < row_end && col_id < A_num_rows)
            {
                col_color = A_colors[col_id];

                if (USE_EXTERNAL_METRIC)
                {
                    col_hash = metric_in[col_id] >> 16; //gt count
                }
            }

            // Threads determine if they are greater than the row hash.
            int gt_pred = col_color == 0 && col_hash > row_hash;
            int lt_pred = col_color == 0 && col_hash < row_hash;
            // Count greater/smaller neighbors.
            gt_count += __popc( subwarp<WARP_SIZE>::masked_ballot( gt_pred ) );//subwarp<WARP_SIZE>::masked_ballot
            lt_count += __popc( subwarp<WARP_SIZE>::masked_ballot( lt_pred ) );
        }

        // The warp leader stores the result.
        int my_gtlt_count = (gt_count << 16) | lt_count;

        if ( lane_id == 0 )
        {
            A_gtlt_count[row_id] = my_gtlt_count;
        }
    }
}
template< int CTA_SIZE, int WARP_SIZE, bool LOAD_EXTERNAL_USED_COLORS >
__global__
void color_kernel_greedy_gtlt(
    const int A_num_rows, const int *A_rows, const int *A_cols,
    const int *A_gtlt_count, const int current_color,
    const int weakness_bound, const int max_color_to_use,
    int *A_colors,
    const int min_degree, const unsigned long long *used_colors_in)
{
    const int NUM_WARPS_PER_CTA  = CTA_SIZE / WARP_SIZE;
    const int NUM_WARPS_PER_GRID = gridDim.x * NUM_WARPS_PER_CTA;
    // Thread coordinates.
    const int warp_id = subwarp<WARP_SIZE>::subwarp_id();
    const int lane_id = subwarp<WARP_SIZE>::sublane_id();
    // Row identifier.
    int row_id = blockIdx.x * NUM_WARPS_PER_CTA + warp_id;

    // Iterate over the rows of the matrix.
    for ( ; row_id < A_num_rows ; row_id += NUM_WARPS_PER_GRID )
    {
        //int dice = hash_function(row_id, 19881988, 0);
        int row_color = A_colors[row_id];

        if ( row_color > 0 ) // Already colored!!!
        {
            continue;
        }

        if ( row_color < 0 )
        {
            A_colors[row_id] = 0; //discovered

            if (row_id != 0) { continue; } //process next time
        }

        // The number of vertices that are greater/smaller than me.
        int row_gtlt_count = A_gtlt_count[row_id];
        // Split gtlt_count into 2.
        int row_gt_count = row_gtlt_count >> 16;
        int row_lt_count = row_gtlt_count  & 0xffff;
        // Do we skip it.
        int candidate = 1;
        // Predicates. Is a vertex min/max.
        int is_max_vertex = row_gt_count <= weakness_bound;
        int is_min_vertex = row_lt_count <= weakness_bound;
        // Iterators over my row.
        int row_begin = A_rows[row_id  ];
        int row_end   = A_rows[row_id + 1];
        int degree = row_end - row_begin;

        if (degree < min_degree && min_degree > 0)
        {
            continue;
        }

        unsigned long long used_colors = 0ull;

        if (LOAD_EXTERNAL_USED_COLORS)
        {
            used_colors |= used_colors_in[row_id];
        }

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

            if (LOAD_EXTERNAL_USED_COLORS)
            {
                used_colors |= used_colors_in[col_id];
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

        //is_max_vertex = row_gt_count==0;
        //is_min_vertex = false;
        //reduce used colors bit by bit.
#pragma unroll

        for (int i = WARP_SIZE / 2; i >= 1; i /= 2)
        {
            int tmp_hi = __double2hiint( __longlong_as_double( used_colors ) );
            int tmp_lo = __double2loint( __longlong_as_double( used_colors ) );
            tmp_hi = utils::shfl_xor(tmp_hi, i, WARP_SIZE);
            tmp_lo = utils::shfl_xor(tmp_lo, i, WARP_SIZE);
            long long tmp = __double_as_longlong(__hiloint2double(tmp_hi, tmp_lo));
            used_colors |= tmp;
        }

        int my_color_1 = 0;
        int my_color_2 = 0;
        int free_colors = __popc(used_colors);
        my_color_1 = 64 - utils::bfind( ~used_colors );
        used_colors |= (1ull << (64 - my_color_1));
        my_color_2 = 64 - utils::bfind( ~used_colors );
        free_colors = 0;

        if ( candidate && subwarp<WARP_SIZE>::all( is_max_vertex ) && my_color_1 < max_color_to_use)
        {
            if ( lane_id == 0 )
            {
                A_colors[row_id] = my_color_1;
            }

            continue;
        }

        if ( candidate && subwarp<WARP_SIZE>::all( is_min_vertex ) && free_colors >= 2)
        {
            if ( lane_id == 0 )
            {
                A_colors[row_id] = my_color_2;
            }

            continue;
        }
    }
}

struct is_zero
{
    __host__ __device__
    bool operator()(int x)
    {
        return x == 0;
    }
};
struct is_minus_one
{
    __host__ __device__
    bool operator()(int x)
    {
        return x == -1;
    }
};

// ---------------------------
// Methods
// ---------------------------

template< class T_Config >
Greedy_Min_Max_2Ring_Matrix_Coloring_Base<T_Config>::Greedy_Min_Max_2Ring_Matrix_Coloring_Base( AMG_Config &cfg, const std::string &cfg_scope) : MatrixColoring<T_Config>(cfg, cfg_scope)
{
    if ( this->m_coloring_level > 5 || this->m_coloring_level < 0)
    {
        FatalError( "Not implemented for coloring_level != 1", AMGX_ERR_NOT_SUPPORTED_TARGET );
    }

    if ( cfg.AMG_Config::template getParameter<IndexType>("determinism_flag", "default") == 1 )
    {
        m_uncolored_fraction = 0.0;
    }
    else
    {
        m_uncolored_fraction = cfg.AMG_Config::template getParameter<double>("max_uncolored_percentage", cfg_scope);
    }

    m_weakness_bound = cfg.AMG_Config::template getParameter<int>( "weakness_bound", cfg_scope );
    m_late_rejection = cfg.AMG_Config::template getParameter<int>( "late_rejection", cfg_scope ) != 0;
    m_coloring_try_remove_last_color = cfg.AMG_Config::template getParameter<int>( "coloring_try_remove_last_colors", cfg_scope );
    m_coloring_custom_arg = cfg.AMG_Config::template getParameter<std::string>( "coloring_custom_arg", cfg_scope );
}

// Block version
template< AMGX_VecPrecision V, AMGX_MatPrecision M, AMGX_IndPrecision I >
template<int NNZ>
void
Greedy_Min_Max_2Ring_Matrix_Coloring<TemplateConfig<AMGX_device, V, M, I> >::color_step(
    Matrix_d &A, int seed, int max_color )
{
    const int num_rows = A.get_num_rows();
    const int max_uncolored_rows = static_cast<int>( this->m_uncolored_fraction * num_rows );
    const int WARP_SIZE = NNZ;
    const int CTA_SIZE = 256;
    const int NUM_WARPS_PER_CTA = CTA_SIZE / WARP_SIZE;
    const int GRID_SIZE = std::min( 2048, (num_rows + NUM_WARPS_PER_CTA - 1) / NUM_WARPS_PER_CTA );
    const int NUM_WARPS_PER_CTA_1 = CTA_SIZE;
    const int GRID_SIZE_1 = std::min( 2048, (num_rows + NUM_WARPS_PER_CTA_1 - 1) / NUM_WARPS_PER_CTA_1 );
    device_vector_alloc<int> finished_colors_flag(1);
    finished_colors_flag[0] = 0;
    int *finished_colors_flag_ptr = amgx::thrust::raw_pointer_cast( &finished_colors_flag.front() );
    device_vector_alloc<int> gtlt_count;
    std::vector<device_vector_alloc<int> > max_id_per_level(this->m_coloring_level);
    std::vector<device_vector_alloc<unsigned long long int> > used_colors_per_level(this->m_coloring_level);

    if (this->m_coloring_level == 1)
    {
        gtlt_count.resize(num_rows);
    }
    else
    {
        for (int i = 1; i < this->m_coloring_level; i++)
        {
            max_id_per_level[i].resize(num_rows);
            used_colors_per_level[i].resize(num_rows);
        }
    }

    int prev_uncolored = num_rows;
    int currentSafeColor = 0;
    int iter = 0;
    int changes_seed = 0;
    int num_color_box = 64;

    for ( int num_uncolored = num_rows ; num_uncolored > max_uncolored_rows; iter++)
    {
        if (this->m_coloring_level == 1)
        {
            //parallel greedy with greater hash count metric instead of raw hash
            count_gtlt_kernel<CTA_SIZE, WARP_SIZE, false> <<< GRID_SIZE, CTA_SIZE>>>(
                num_rows,
                A.row_offsets.raw(),
                A.col_indices.raw(),
                this->m_row_colors.raw(),
                amgx::thrust::raw_pointer_cast( &gtlt_count.front() ),
                seed
                , 0);
            cudaCheckError();
            color_kernel_greedy_gtlt<CTA_SIZE, WARP_SIZE, false> <<< GRID_SIZE, CTA_SIZE>>>(
                num_rows,
                A.row_offsets.raw(),
                A.col_indices.raw(),
                amgx::thrust::raw_pointer_cast( &gtlt_count.front() ),
                this->m_num_colors,
                this->m_weakness_bound, max_color,
                this->m_row_colors.raw(), 0, 0);
            cudaCheckError();
        }
        else
        {
            //Any ring parallel greedy with propagation
            for (int i = 0; i < this->m_coloring_level - 1; i++)
            {
                if (i == 0)
                    find_max_neighbor_kernel_and_propagate_used_colors<false, CTA_SIZE, WARP_SIZE> <<< GRID_SIZE, CTA_SIZE>>>(num_rows,
                            A.row_offsets.raw(),
                            A.col_indices.raw(),
                            this->m_row_colors.raw(),
                            0, // in
                            amgx::thrust::raw_pointer_cast(max_id_per_level[i + 1].data()),  //out
                            0, // in
                            amgx::thrust::raw_pointer_cast(used_colors_per_level[i + 1].data()),  //out
                            //amgx::thrust::raw_pointer_cast(interesting_in.data()),
                            currentSafeColor,
                            seed);
                else
                    find_max_neighbor_kernel_and_propagate_used_colors<true, CTA_SIZE, WARP_SIZE> <<< GRID_SIZE, CTA_SIZE>>>(num_rows,
                            A.row_offsets.raw(),
                            A.col_indices.raw(),
                            this->m_row_colors.raw(),
                            amgx::thrust::raw_pointer_cast(max_id_per_level[i  ].data()),  // in
                            amgx::thrust::raw_pointer_cast(max_id_per_level[i + 1].data()),  //out
                            amgx::thrust::raw_pointer_cast(used_colors_per_level[i  ].data()),  // in
                            amgx::thrust::raw_pointer_cast(used_colors_per_level[i + 1].data()),  //out
                            //amgx::thrust::raw_pointer_cast(interesting_in.data()),
                            currentSafeColor,
                            seed);

                cudaCheckError();
            }

            color_kernel_greedy_onlymax<CTA_SIZE, WARP_SIZE> <<< GRID_SIZE, CTA_SIZE>>>( num_rows,
                    A.row_offsets.raw(), A.col_indices.raw(),
                    amgx::thrust::raw_pointer_cast( max_id_per_level[this->m_coloring_level - 1].data()), //in: max_id_tmp
                    seed,
                    this->m_row_colors.raw(),
                    (unsigned long long *)amgx::thrust::raw_pointer_cast( used_colors_per_level[this->m_coloring_level - 1].data() ),
                    finished_colors_flag_ptr,
                    currentSafeColor);
            cudaCheckError();
        }

        int nnz_per_row = float(A.get_num_nz()) / A.get_num_rows();

        if (iter > std::min((int)std::pow(nnz_per_row, this->m_coloring_level), 63)) //estimate number of epochs required for coloring
        {
            num_uncolored = (int) amgx::thrust::count_if( this->m_row_colors.begin(), this->m_row_colors.begin() + num_rows, is_zero() );
            cudaCheckError();

            if (num_uncolored == prev_uncolored)
            {
                max_color++;

                if (this->m_coloring_level >= 2)
                {
                    //printf("%d Fail with numcolor %d %d\n",seed,max_color,num_uncolored);
                    if (changes_seed < 0)
                    {
                        seed++;
                        changes_seed++;
                    }
                    else
                    {
                        changes_seed = 0;
                        currentSafeColor += num_color_box;

                        //reset buffers
                        for (int i = 1; i < this->m_coloring_level; i++)
                        {
                            thrust_wrapper::fill<AMGX_device>(max_id_per_level[i].begin(), max_id_per_level[i]    .end(), 0);
                            thrust_wrapper::fill<AMGX_device>(used_colors_per_level[i].begin(), used_colors_per_level[i].end(), 0);
                        }

                        cudaCheckError();
                        //printf("opening new color box %d\n",currentSafeColor);
                    }
                }
            }

            prev_uncolored = num_uncolored;
            //printf("%d\t%d\t%d\n", iter, 0, num_uncolored);
        }
    }

    this->m_num_colors = thrust_wrapper::reduce<AMGX_device>( this->m_row_colors.begin(), this->m_row_colors.begin() + num_rows, 0, amgx::thrust::maximum<int>() ) + 1;
    cudaCheckError();
}



// Block version
template< AMGX_VecPrecision V, AMGX_MatPrecision M, AMGX_IndPrecision I >
void
Greedy_Min_Max_2Ring_Matrix_Coloring<TemplateConfig<AMGX_device, V, M, I> >::colorMatrix( Matrix_d &A )
{
    ViewType oldView = A.currentView();
    this->m_row_colors.resize( A.row_offsets.size() - 1 );

    if  (this->m_halo_coloring == SYNC_COLORS) { A.setView(ALL); }
    else { A.setViewExterior(); }

    this->m_num_colors = 1;
    thrust_wrapper::fill<AMGX_device>( this->m_row_colors.begin(), this->m_row_colors.end(), 0 );
    cudaCheckError();
    float avg_nnz_per_row = A.get_num_nz() / float(A.get_num_rows());
    //Choose subwarp (group of threads processing a row) size
    {
        if ( avg_nnz_per_row < 4.0f )
        {
            color_step<1>(A, 0, 64);
        }
        else if ( avg_nnz_per_row < 8.0f )
        {
            color_step<1>(A, 0, 64);
        }
        else if (avg_nnz_per_row < 32)
        {
            color_step<4>(A, 0, 64);
        }
        else
        {
            color_step<32>(A, 0, 64);
        }
    }
    A.setView(oldView);
}


#define AMGX_CASE_LINE(CASE) template class Greedy_Min_Max_2Ring_Matrix_Coloring_Base<TemplateMode<CASE>::Type>;
AMGX_FORALL_BUILDS(AMGX_CASE_LINE)
#undef AMGX_CASE_LINE

#define AMGX_CASE_LINE(CASE) template class Greedy_Min_Max_2Ring_Matrix_Coloring<TemplateMode<CASE>::Type>;
AMGX_FORALL_BUILDS(AMGX_CASE_LINE)
#undef AMGX_CASE_LINE

} // end namespace amgx
