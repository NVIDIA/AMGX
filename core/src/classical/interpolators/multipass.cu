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

#include <classical/interpolators/multipass.h>
#include <classical/interpolators/common.h>
#include <amgx_timer.h>
#include <basic_types.h>
#include <types.h>
#include <cutil.h>
#include <util.h>
#include <sm_utils.inl>
#include <fstream>
#include <set>
#include <vector>
#include <algorithm>
#include <sort.h>
#include <assert.h>
#include <misc.h>
#include <sstream>
#include <csr_multiply.h>
#include <hash_workspace.h>
#include <thrust/count.h>
#include <thrust/scan.h>
#include <thrust/iterator/transform_iterator.h>
#include <thrust/iterator/counting_iterator.h>

namespace amgx
{

using std::set;
using std::vector;

template < typename IndexType, typename ValueType >
__global__
void check_P_values(IndexType *offsets, ValueType *val, int size, int flag)
{
    for (int tidx = threadIdx.x + blockIdx.x * blockDim.x; tidx < size; tidx += blockDim.x * gridDim.x)
    {
        // Simply cast
        for (int i = offsets[tidx]; i < offsets[tidx + 1]; i++)
        {
            if (isnan(val[i])) { printf("%d, row = %d, i=%d\n", flag, tidx, i); }
        }
    }
}

template <typename IndexType, typename KeyType>
__global__
void fill_cf_map_global(const IndexType *cf_map, KeyType *cf_map_global, IndexType num_owned_fine_pts)
{
    for (int tidx = threadIdx.x + blockIdx.x * blockDim.x; tidx < num_owned_fine_pts ; tidx += blockDim.x * gridDim.x)
    {
        // Simply cast
        cf_map_global[tidx] = (KeyType) cf_map[tidx];
    }
}

template <typename IndexType, typename KeyType>
__global__
void fill_P_col_indices(const KeyType *global_cols, IndexType *cols, IndexType num_owned_fine_pts)
{
    for (int tidx = threadIdx.x + blockIdx.x * blockDim.x; tidx < num_owned_fine_pts ; tidx += blockDim.x * gridDim.x)
    {
        // Simply cast
        cols[tidx] = (IndexType) global_cols[tidx];
    }
}



template <typename IndexType, typename KeyType>
__global__
void createCfMapGlobal(const IndexType *cf_map, KeyType *cf_map_global, const KeyType my_part_offset, const IndexType num_owned_fine_pts)
{
    for (int tidx = threadIdx.x + blockIdx.x * blockDim.x; tidx < num_owned_fine_pts ; tidx += blockDim.x * gridDim.x)
    {
        // Only renumber the interior points
        if (cf_map[tidx] >= 0 && tidx < num_owned_fine_pts)
        {
            // Simply shift
            cf_map_global[tidx] = (KeyType) cf_map[tidx] /* my_local_id */ + my_part_offset;
        }
        else
        {
            cf_map_global[tidx] = -1;
        }
    }
}



template<typename IndexType, int cta_size >
__global__
void initializeAssignedArray(const IndexType *cf_map, IndexType *assigned, const IndexType *A_rows, const IndexType *A_cols, const bool *s_con, IndexType *C_hat_offsets, int A_num_rows)
{
    const int nWarps = cta_size / 32;
    // The coordinates of the thread inside the CTA/warp.
    const int warpId = threadIdx.x / 32;
    const int laneId = threadIdx.x % 32;

    // Loop over rows of A.
    for ( int aRowId = blockIdx.x * nWarps + warpId ; aRowId < A_num_rows ; aRowId += gridDim.x * nWarps )
    {
        int coarse_fine_id = cf_map[aRowId];

        if ( coarse_fine_id >= 0)
        {
            assigned[aRowId] = 0;
        }
        else if (coarse_fine_id == FINE)
        {
            // Load A row IDs.
            int aColIt  = A_rows[aRowId  ];
            int aColEnd = A_rows[aRowId + 1];
            int coarse_count = 0;

            // Iterate over the columns of A.
            for ( aColIt += laneId ; utils::any( aColIt < aColEnd ) ; aColIt += 32 )
            {
                // Columns of A maps to rows of B. Each thread of the warp loads its A-col/B-row ID.
                int aColId = aColIt < aColEnd ? A_cols[aColIt] : -1;
                // Is it an off-diagonal element.
                bool is_off_diagonal = aColIt < aColEnd && aColId != aRowId;
                // Is it fine.
                bool is_coarse  = is_off_diagonal && cf_map[aColId] >= 0;
                // Is it strongly connected ?
                bool is_coarse_strongly_connected = s_con[aColIt] && is_coarse;
                // Push coarse and strongly connected nodes in the set.
                coarse_count += __popc( utils::ballot( is_coarse_strongly_connected ) );
            }

            // If fine point connected to coarse point, assigned = 1,
            if (coarse_count)
            {
                assigned[aRowId] = 1;
                C_hat_offsets[aRowId] = coarse_count;
            }
        }
        else // Strong fine
        {
            assigned[aRowId] = -1;
        }
    }
}

template<typename IndexType, int cta_size >
__global__
void fillAssignedArray(IndexType *assigned, const IndexType *A_rows, const IndexType *A_cols, const bool *s_con, int A_num_rows, int pass)
{
    const int nWarps = cta_size / 32;
    // The coordinates of the thread inside the CTA/warp.
    const int warpId = threadIdx.x / 32;
    const int laneId = threadIdx.x % 32;

    // Loop over rows of A.
    for ( int aRowId = blockIdx.x * nWarps + warpId ; aRowId < A_num_rows ; aRowId += gridDim.x * nWarps )
    {
        if ( assigned[aRowId] == -1)
        {
            // Load A row IDs.
            int aColIt  = A_rows[aRowId  ];
            int aColEnd = A_rows[aRowId + 1];
            int assigned_count = 0;

            // Iterate over the columns of A.
            for ( aColIt += laneId ; utils::any( aColIt < aColEnd ) ; aColIt += 32 )
            {
                // Columns of A maps to rows of B. Each thread of the warp loads its A-col/B-row ID.
                int aColId = aColIt < aColEnd ? A_cols[aColIt] : -1;
                // Is it an off-diagonal element.
                bool is_off_diagonal = aColIt < aColEnd && aColId != aRowId;
                // Is it assigned
                bool is_assigned = is_off_diagonal && assigned[aColId] == pass - 1;
                // Is it strongly connected ?
                bool is_assigned_strongly_connected = s_con[aColIt] && is_assigned;
                // Push coarse and strongly connected nodes in the set.
                assigned_count += __popc( utils::ballot( is_assigned_strongly_connected ) );

                if (assigned_count) { break; }
            }

            // If one of the strong neighbor has been assigned, set assigned to current pass number
            if (assigned_count)
            {
                assigned[aRowId] = pass;
            }
        }
    }
}


/*
 * remove all duplicates from a vector
 */
template <typename T>
void removeDuplicates(vector<T> &vec)
{
    std::sort(vec.begin(), vec.end());
    vec.erase(std::unique(vec.begin(), vec.end()), vec.end());
}

/*
 * find the value at a particular column of a row if it exists
 */
template <typename IVector, typename VVector>
typename VVector::value_type findValue(const IVector &columns,
                                       const VVector &values,
                                       const int row_begin, const int row_end, const int column_needed)
{
    typename VVector::value_type v = 0;

    // NAIVE
    for (int k = row_begin; k < row_end; k++)
    {
        int kcol = columns[k];

        if (kcol == column_needed)
        {
            v = values[k];
        }
    }

    return v;
}



/*************************************************************************
* create the interpolation matrix P
************************************************************************/
template <AMGX_VecPrecision t_vecPrec, AMGX_MatPrecision t_matPrec, AMGX_IndPrecision t_indPrec>
void Multipass_Interpolator<TemplateConfig<AMGX_host, t_vecPrec, t_matPrec, t_indPrec> >::generateInterpolationMatrix_1x1(Matrix_h &A,
        IntVector &cf_map,
        BVector &s_con,
        IntVector &scratch,
        Matrix_h &P,
        void *amg)
{
    FatalError("multipass interpolation not implemented on host\n", AMGX_ERR_NOT_IMPLEMENTED);
} // end multipass interpolator


///////////////////////////////////////////////////////////////////////////////

// DEVICE CODE

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

namespace multipass_sm20
{

#include <sm_utils.inl>
#include <hash_containers_sm20.inl> // Included inside the namespace to solve name colisions.

__device__ __forceinline__ int get_work( volatile int *offsets, int *queue, int warp_id )
{
    if ( utils::lane_id() == 0 )
    {
        offsets[warp_id] = atomicAdd( queue, 1 );
    }

    return offsets[warp_id];
}

} // namespace multipass_sm20

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

namespace multipass_sm35
{

#include <sm_utils.inl>
#include <hash_containers_sm35.inl> // Included inside the namespace to solve name colisions.

__device__ __forceinline__ int get_work( int *queue, int warp_id )
{
#if __CUDA_ARCH__ >= 300
    int offset = -1;

    if ( utils::lane_id() == 0 )
    {
        offset = atomicAdd( queue, 1 );
    }

    return utils::shfl( offset, 0 );
#else
    return 0;
#endif
}

} // namespace multipass_sm35

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

namespace multipass
{

#include <sm_utils.inl>

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template< int CTA_SIZE, int WARP_SIZE >
__global__ __launch_bounds__( CTA_SIZE )
void
estimate_c_hat_size_kernel( const int A_num_rows,
                            const int *A_rows,
                            const int *A_cols,
                            const bool *s_con,
                            const int *assigned,
                            int *C_hat_offsets,
                            int pass )
{
#if __CUDA_ARCH__ < 300
    __shared__ volatile int s_mem[CTA_SIZE];
#endif
    const int NUM_WARPS_PER_CTA = CTA_SIZE / WARP_SIZE;
    // Number of items per grid.
    const int NUM_WARPS_PER_GRID = gridDim.x * NUM_WARPS_PER_CTA;
    // The coordinates of the thread inside the CTA/warp.
    const int warp_id = utils::warp_id( );
    const int lane_id = utils::lane_id( );
    // First threads load the row IDs of A needed by the CTA...
    volatile int a_row_id = blockIdx.x * NUM_WARPS_PER_CTA + warp_id;

    // Loop over rows of A.
    for ( ; a_row_id < A_num_rows ; a_row_id += NUM_WARPS_PER_GRID )
    {
        // Skip coarse rows and fine rows touching a coarse point
        if (assigned[a_row_id] != pass)
        {
            continue;
        }

        // Load A row IDs.
        int a_col_begin = A_rows[a_row_id  ];
        int a_col_end   = A_rows[a_row_id + 1];
        // The number of elements.
        int my_count = 0;

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
            // assigned == pass-1
            bool is_assigned = is_off_diagonal && (assigned[a_col_id] == pass - 1);
            // Is it a coarse, strongly-connected column and already assigned?
            bool is_assigned_strongly_connected = is_strongly_connected && is_assigned;
            // If so, add their C_hat_offset value to my_count
            my_count += (is_assigned_strongly_connected ? C_hat_offsets[a_col_id] : 0);
        }

        // Do reduction
#if __CUDA_ARCH__ >= 300
#pragma unroll

        for ( int mask = WARP_SIZE / 2 ; mask > 0 ; mask >>= 1 )
        {
            my_count += utils::shfl_xor( my_count, mask );
        }

#else
        s_mem[threadIdx.x] = my_count;
#pragma unroll

        for ( int offset = WARP_SIZE / 2 ; offset > 0 ; offset >>= 1 )
            if ( lane_id < offset )
            {
                s_mem[threadIdx.x] = my_count += s_mem[threadIdx.x + offset];
            }

#endif

        // Write result -- RACE CONDITION!
        if (lane_id == 0)
        {
            C_hat_offsets[a_row_id] = my_count;
        }
    }
}

template< int CTA_SIZE, int WARP_SIZE, typename KeyType >
__global__ __launch_bounds__( CTA_SIZE )
void
compute_c_hat_first_pass_kernel( int A_num_rows,
                                 const int *__restrict A_rows,
                                 const int *__restrict A_cols,
                                 const bool *__restrict s_con,
                                 const KeyType *__restrict cf_map_global,
                                 const int *__restrict C_hat_start,
                                 int *__restrict C_hat_size,
                                 KeyType *__restrict C_hat,
                                 int *__restrict assigned)
{
    const int NUM_WARPS_PER_CTA = CTA_SIZE / WARP_SIZE;
    const int NUM_WARPS_PER_GRID = gridDim.x * NUM_WARPS_PER_CTA;
#if __CUDA_ARCH__ >= 300
#else
    // Shared memory to store where to load from.
    __shared__ volatile int s_rows[2 * NUM_WARPS_PER_CTA];
#endif
    // The coordinates of the thread inside the CTA/warp.
    const int warp_id = utils::warp_id( );
    const int lane_id = utils::lane_id( );
    // First threads load the row IDs of A needed by the CTA...
    int a_row_id = blockIdx.x * NUM_WARPS_PER_CTA + warp_id;

    // Loop over rows of A.
    for ( ; a_row_id < A_num_rows ; a_row_id += NUM_WARPS_PER_GRID)
    {
        // Skip if not in current pass
        if (assigned[a_row_id] != 1)
        {
            continue;
        }

        // Load the range of the row.
#if __CUDA_ARCH__ >= 300
        int a_col_tmp = -1;

        if ( lane_id < 2 )
        {
            a_col_tmp = A_rows[a_row_id + lane_id];
        }

        int a_col_begin = utils::shfl( a_col_tmp, 0 );
        int a_col_end   = utils::shfl( a_col_tmp, 1 );
#else

        if ( lane_id < 2 )
        {
            s_rows[2 * warp_id + lane_id] = A_rows[a_row_id + lane_id];
        }

        int a_col_begin = s_rows[2 * warp_id + 0];
        int a_col_end   = s_rows[2 * warp_id + 1];
#endif
        int count = 0;
        int c_col_it = C_hat_start[a_row_id];

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
            // assigned == pass -1
            bool is_assigned = is_off_diagonal && (assigned[a_col_id] == 0);
            // Is it an assigned==pass -1 and strongly-connected column.
            bool is_assigned_strongly_connected = is_strongly_connected && is_assigned;
            // We collect fine and strongly-collected columns.
            int vote = utils::ballot( is_assigned_strongly_connected );
            int dest = __popc( vote & utils::lane_mask_lt() );

            if ( is_assigned_strongly_connected )
            {
                //if (cf_map[a_col_id] < 0)
                //  printf("error in multipass 1\n");
                C_hat[c_col_it + count + dest] = cf_map_global[a_col_id];
            }

            count += __popc( vote );
        }

        if ( lane_id == 0 )
        {
            C_hat_size[a_row_id] = count;
        }
    }
}


///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template< int NUM_THREADS_PER_ROW, int CTA_SIZE, int SMEM_SIZE, int WARP_SIZE, typename KeyType >
__global__ __launch_bounds__( CTA_SIZE )
void
compute_c_hat_kernel( int A_num_rows,
                      const int *__restrict A_rows,
                      const int *__restrict A_cols,
                      const bool *__restrict s_con,
                      const int *__restrict C_hat_start,
                      int *__restrict C_hat_size,
                      KeyType *__restrict C_hat,
                      int *__restrict C_hat_pos,
                      int *__restrict assigned,
                      int gmem_size,
                      KeyType *g_keys,
                      int *wk_work_queue,
                      int *wk_status,
                      int pass )
{
    const int NUM_WARPS = CTA_SIZE / WARP_SIZE;
    const int NUM_LOADED_ROWS = WARP_SIZE / NUM_THREADS_PER_ROW;
    // Shared memory to vote.
    __shared__ volatile int s_b_row_ids[CTA_SIZE];
    // The hash keys stored in shared memory.
    __shared__ volatile KeyType s_keys[NUM_WARPS * SMEM_SIZE];
#if __CUDA_ARCH__ >= 300
#else
    // Shared memory to acquire work.
    __shared__ volatile int s_offsets[NUM_WARPS];
    // Shared memory to store where to load from.
    __shared__ volatile int s_rows[2 * NUM_WARPS];
#endif
    // The coordinates of the thread inside the CTA/warp.
    const int warp_id = utils::warp_id( );
    const int lane_id = utils::lane_id( );
    // Constants.
    const int lane_id_div_num_threads = lane_id / NUM_THREADS_PER_ROW;
    const int lane_id_mod_num_threads = lane_id % NUM_THREADS_PER_ROW;
    // First threads load the row IDs of A needed by the CTA...
    int a_row_id = blockIdx.x * NUM_WARPS + warp_id;
    // Create local storage for the set.
#if __CUDA_ARCH__ >= 300
    multipass_sm35::Hash_set<KeyType, SMEM_SIZE, 4, WARP_SIZE> set( &s_keys[warp_id * SMEM_SIZE], &g_keys[a_row_id * gmem_size], gmem_size );
#else
    multipass_sm20::Hash_set<KeyType, SMEM_SIZE, 4, WARP_SIZE> set( &s_keys[warp_id * SMEM_SIZE], &g_keys[a_row_id * gmem_size], gmem_size );
#endif
    // Loop over rows of A.
#if __CUDA_ARCH__ >= 300

    for ( ; a_row_id < A_num_rows ; a_row_id = multipass_sm35::get_work( wk_work_queue, warp_id ) )
#else
    for ( ; a_row_id < A_num_rows ; a_row_id = multipass_sm20::get_work( s_offsets, wk_work_queue, warp_id ) )
#endif
    {
        // Skip if not in current pass
        if (assigned[a_row_id] != pass)
        {
            continue;
        }

        // Clear the set.
        set.clear();
        // Load the range of the row.
#if __CUDA_ARCH__ >= 300
        int a_col_tmp = -1;

        if ( lane_id < 2 )
        {
            a_col_tmp = A_rows[a_row_id + lane_id];
        }

        int a_col_begin = utils::shfl( a_col_tmp, 0 );
        int a_col_end   = utils::shfl( a_col_tmp, 1 );
#else

        if ( lane_id < 2 )
        {
            s_rows[2 * warp_id + lane_id] = A_rows[a_row_id + lane_id];
        }

        int a_col_begin = s_rows[2 * warp_id + 0];
        int a_col_end   = s_rows[2 * warp_id + 1];
#endif

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
            // Check if assigned == pass -1
            bool is_assigned = is_off_diagonal && (assigned[a_col_id] == pass - 1);
            // Check if we have a strong connection to the point
            bool is_strongly_connected = s_con[a_col_it];
            // Check if strong connection to assigned == pass-1
            bool is_assigned_strongly_connected = is_assigned && is_strongly_connected;
            // We collect columns for which assigned == pass - 1
            int vote = utils::ballot( is_assigned_strongly_connected);
            int dest = __popc( vote & utils::lane_mask_lt() );

            if ( is_assigned_strongly_connected)
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
                    // this loops over elements of C_hat for row uniform_b_row_id
                    b_col_it  = C_hat_start[uniform_b_row_id];
                    b_col_end = b_col_it + C_hat_size[uniform_b_row_id];
                }

                // _iterate over the range of columns of B.
                for ( b_col_it += lane_id_mod_num_threads ; utils::any(b_col_it < b_col_end) ; b_col_it += NUM_THREADS_PER_ROW )
                {
                    // The ID of the interpolatory point
                    KeyType  b_col_id = -1;

                    if ( b_col_it < b_col_end )
                    {
                        b_col_id = C_hat[b_col_it];
                    }

                    // Is it valid
                    is_off_diagonal = b_col_it < b_col_end; //  && b_col_id != local_k;
                    // Push node to set
                    KeyType b_item = -1;

                    if ( is_off_diagonal )
                    {
                        b_item = b_col_id;
                    }

                    // Insert the interpolatory point to the set
                    set.insert( b_item, wk_status );
                }
            }
        }

        int c_col_it = C_hat_start[a_row_id];
        int count = set.store_with_positions( &C_hat[c_col_it], &C_hat_pos[c_col_it] );

        if ( lane_id == 0 )
        {
            C_hat_size[a_row_id] = count;
        }
    }
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template< typename Value_type, int CTA_SIZE, int SMEM_SIZE, int WARP_SIZE, typename KeyType >
__global__ __launch_bounds__( CTA_SIZE )
void
compute_interp_weight_first_pass_kernel( const int A_num_rows,
        const int *__restrict A_rows,
        const int *__restrict A_cols,
        const Value_type *__restrict A_vals,
        const KeyType *__restrict cf_map_global,
        const bool *__restrict s_con,
        const Value_type *__restrict diag,
        const int *__restrict P_rows,
        KeyType *P_cols,
        Value_type *P_vals,
        const int *__restrict assigned)
{
    const int NUM_WARPS_PER_CTA = CTA_SIZE / 32;
    const int NUM_WARPS_PER_GRID = gridDim.x * NUM_WARPS_PER_CTA;
#if __CUDA_ARCH__ >= 300
#else
    // Shared memory to store where to load from.
    __shared__ volatile int s_rows[2 * NUM_WARPS_PER_CTA];
    // Shared memory for broadcast.
    __shared__ volatile Value_type s_n_values[CTA_SIZE];
    __shared__ volatile Value_type s_c_values[CTA_SIZE];
#endif
    // The coordinates of the thread inside the CTA/warp.
    const int warp_id = utils::warp_id();
    const int lane_id = utils::lane_id();
    // First threads load the row IDs of A needed by the CTA...
    int a_row_id = blockIdx.x * NUM_WARPS_PER_CTA + warp_id;

    // Loop over rows of A.
    for ( ; a_row_id < A_num_rows ; a_row_id += NUM_WARPS_PER_GRID)
    {
        // Only do work if assigned[row_id] == pass
        KeyType coarse_fine_id = cf_map_global[a_row_id];

        // If coarse rows
        if ( assigned[a_row_id] == 0 )
        {
            if ( lane_id == 0 )
            {
                int p_row_it = P_rows[a_row_id];
                P_cols[p_row_it] = coarse_fine_id;
                P_vals[p_row_it] = Value_type( 1 );
            }
        }
        // If strong fine
        else if ( assigned[a_row_id] == -1 )
        {
            // Do nothing
        }
        // strongly connected to a coarse point
        else if (assigned[a_row_id] == 1)
        {
            // Load A row IDs.
#if __CUDA_ARCH__ >= 300
            int a_col_tmp = -1;

            if ( lane_id < 2 )
            {
                a_col_tmp = A_rows[a_row_id + lane_id];
            }

            int a_col_it  = utils::shfl( a_col_tmp, 0 );
            int a_col_end = utils::shfl( a_col_tmp, 1 );
#else

            if ( lane_id < 2 )
            {
                s_rows[2 * warp_id + lane_id] = A_rows[a_row_id + lane_id];
            }

            int a_col_it  = s_rows[2 * warp_id + 0];
            int a_col_end = s_rows[2 * warp_id + 1];
#endif
#if __CUDA_ARCH__ >= 300
            int p_col_tmp = -1;

            if ( lane_id < 2 )
            {
                p_col_tmp = P_rows[a_row_id + lane_id];
            }

            int p_col_it  = utils::shfl( p_col_tmp, 0 );
            int p_col_end = utils::shfl( p_col_tmp, 1 );
#else

            if ( lane_id < 2 )
            {
                s_rows[2 * warp_id + lane_id] = P_rows[a_row_id + lane_id];
            }

            int p_col_it  = s_rows[2 * warp_id + 0];
            int p_col_end = s_rows[2 * warp_id + 1];
#endif
            // Weak value.
            Value_type sum_N(0), sum_C(0), alfa(0);
            int count = 0;

            // Iterate over the columns of A.
            for ( a_col_it += lane_id ; utils::any( a_col_it < a_col_end ) ; a_col_it += WARP_SIZE )
            {
                // Columns of A maps to rows of B. Each thread of the warp loads its A-col/B-row ID.
                int a_col_id(-1);
                Value_type a_value(0);

                if ( a_col_it < a_col_end )
                {
                    a_col_id = A_cols[a_col_it];
                    a_value  = A_vals[a_col_it];
                }

                // Is it an off-diagonal element.
                bool is_off_diagonal = a_col_it < a_col_end && a_col_id != a_row_id;
                // Is it a strongly-connected column.
                bool is_strongly_connected = is_off_diagonal && s_con[a_col_it];
                // Is it a weakly connected node.
                //bool is_weakly_connected = is_off_diagonal && !is_strongly_connected;
                // Is it isolated.
                bool is_strong_fine = is_off_diagonal && cf_map_global[a_col_id] == STRONG_FINE;
                // Is it a fine and strongly-connected column.
                bool is_assigned_strongly_connected = is_strongly_connected && assigned[a_col_id] == 0;

                // Update the weak value.
                if ( is_off_diagonal && !is_strong_fine)
                {
                    sum_N += a_value;
                }

                if ( is_assigned_strongly_connected )
                {
                    sum_C += a_value;
                }

                // We collect fine and strongly-collected columns.
                int vote = utils::ballot( is_assigned_strongly_connected );
                int dest = __popc( vote & utils::lane_mask_lt() );

                if ( is_assigned_strongly_connected )
                {
                    P_cols[p_col_it + count + dest] = cf_map_global[a_col_id];
                    P_vals[p_col_it + count + dest] = a_value;
                }

                count += __popc( vote );
            }

            // We're done with that row of A, now reduce sum_N and sum_C
#if __CUDA_ARCH__ >= 300
#pragma unroll

            for ( int mask = WARP_SIZE / 2 ; mask > 0 ; mask >>= 1 )
            {
                sum_C += utils::shfl_xor( sum_C, mask );
                sum_N += utils::shfl_xor( sum_N, mask );
            }

            if ( lane_id == 0 )
            {
                double div = (fabs(sum_C * diag[a_row_id]) < 1e-10) ? 1. : sum_C * diag[a_row_id];
                alfa = -sum_N / div;
            }

            alfa = utils::shfl( alfa, 0 );
#else
            s_c_values[threadIdx.x] = sum_C;
            s_n_values[threadIdx.x] = sum_N;
#pragma unroll

            for ( int offset = WARP_SIZE / 2 ; offset > 0 ; offset >>= 1 )
                if ( lane_id < offset )
                {
                    s_c_values[threadIdx.x] = sum_C += s_c_values[threadIdx.x + offset];
                    s_n_values[threadIdx.x] = sum_N += s_n_values[threadIdx.x + offset];
                }

            sum_C = s_c_values[warp_id * WARP_SIZE];
            sum_N = s_n_values[warp_id * WARP_SIZE];

            if ( lane_id == 0 )
            {
                if (fabs(sum_C * diag[a_row_id]) < 1e-10) { printf("Dividing by zero\n"); }

                double div = (fabs(sum_C * diag[a_row_id]) < 1e-10) ? 1. : sum_C * diag[a_row_id];
                alfa = -sum_N / div;
                // alfa = -sum_N/(sum_C*diag[a_row_id]);
                s_c_values[threadIdx.x] = alfa;
            }

            alfa = s_c_values[warp_id * WARP_SIZE];
#endif

            // Scale the value of P
            for ( p_col_it += lane_id ; utils::any( p_col_it < p_col_end ) ; p_col_it += WARP_SIZE )
            {
                // Columns of A maps to rows of B. Each thread of the warp loads its A-col/B-row ID.
                if ( p_col_it < p_col_end )
                {
                    P_vals[p_col_it] *= alfa;
                }
            }
        } // if (assigned[a_row_id] == 1)
    } // Loop over rows
}



template< typename Value_type, int CTA_SIZE, int SMEM_SIZE, int WARP_SIZE, typename KeyType >
__global__ __launch_bounds__( CTA_SIZE )
void
compute_interp_weight_kernel( const int A_num_rows,
                              const int *__restrict A_rows,
                              const int *__restrict A_cols,
                              const Value_type *__restrict A_vals,
                              const int *__restrict cf_map,
                              const bool *__restrict s_con,
                              const KeyType *__restrict C_hat,
                              const int *__restrict C_hat_pos,
                              const int *__restrict C_hat_start,
                              const int *__restrict C_hat_size,
                              const Value_type *__restrict diag,
                              const int *__restrict P_rows,
                              KeyType *P_cols,
                              Value_type *P_vals,
                              const int gmem_size,
                              KeyType *g_keys,
                              Value_type *g_vals,
                              int *wk_work_queue,
                              const int *__restrict assigned,
                              int pass)
{
#if 1
    const int NUM_WARPS = CTA_SIZE / 32;
    // The hash keys stored in shared memory.
    __shared__ volatile KeyType s_keys[NUM_WARPS * SMEM_SIZE];
    // A shared location where threads propose a row of B to load.
    __shared__ volatile int s_b_row_ids[CTA_SIZE];
    // A shared location where threads store a value of B to load.
    __shared__ volatile Value_type s_b_values[CTA_SIZE];
#if __CUDA_ARCH__ >= 300
    // The hash values stored in shared memory.
    __shared__ volatile multipass_sm35::Word s_vote[NUM_WARPS * SMEM_SIZE / 4];
#else
    // Shared memory to acquire work.
    __shared__ volatile int s_offsets[NUM_WARPS];
    // Shared memory to store where to load from.
    __shared__ volatile int s_rows[2 * NUM_WARPS];
    // Shared memory to store the values in the hash table.
    __shared__ Value_type s_vals[NUM_WARPS * SMEM_SIZE];
    __shared__ volatile Value_type s_n_values[CTA_SIZE];
    __shared__ volatile Value_type s_c_values[CTA_SIZE];
#endif
    // The coordinates of the thread inside the CTA/warp.
    const int warp_id = utils::warp_id();
    const int lane_id = utils::lane_id();
    // First threads load the row IDs of A needed by the CTA...
    volatile int a_row_id = blockIdx.x * NUM_WARPS + warp_id;
    // Create local storage for the set.
#if __CUDA_ARCH__ >= 300
    multipass_sm35::Hash_map<KeyType, Value_type, SMEM_SIZE, 4, WARP_SIZE> map( &s_keys[warp_id * SMEM_SIZE],
            &g_keys[a_row_id * gmem_size],
            &s_vote[warp_id * SMEM_SIZE / 4],
            &g_vals[a_row_id * gmem_size],
            gmem_size );
#else
    multipass_sm20::Hash_map<KeyType, Value_type, SMEM_SIZE, 4, WARP_SIZE> map( &s_keys[warp_id * SMEM_SIZE],
            &g_keys[a_row_id * gmem_size],
            &s_vals[warp_id * SMEM_SIZE],
            &g_vals[a_row_id * gmem_size],
            gmem_size );
#endif
    // Loop over rows of A.
#if __CUDA_ARCH__ >= 300

    for ( ; a_row_id < A_num_rows ; a_row_id = multipass_sm35::get_work( wk_work_queue, warp_id ) )
#else
    for ( ; a_row_id < A_num_rows ; a_row_id = multipass_sm20::get_work( s_offsets, wk_work_queue, warp_id ) )
#endif
    {
        // Only do work if assigned[row_id] == pass
        if (assigned[a_row_id] != pass)
        {
            continue;
        }

        // Clear the table.
        map.clear();
        // Rebuild C_hat.
        int c_hat_it  = C_hat_start[a_row_id];
        int c_hat_end = c_hat_it + C_hat_size[a_row_id];
        map.load( c_hat_end - c_hat_it, &C_hat[c_hat_it], &C_hat_pos[c_hat_it] );
        // Load A row IDs.
#if __CUDA_ARCH__ >= 300
        int a_col_tmp = -1;

        if ( lane_id < 2 )
        {
            a_col_tmp = A_rows[a_row_id + lane_id];
        }

        int a_col_it  = utils::shfl( a_col_tmp, 0 );
        int a_col_end = utils::shfl( a_col_tmp, 1 );
#else

        if ( lane_id < 2 )
        {
            s_rows[2 * warp_id + lane_id] = A_rows[a_row_id + lane_id];
        }

        int a_col_it  = s_rows[2 * warp_id + 0];
        int a_col_end = s_rows[2 * warp_id + 1];
#endif
        // sums
        Value_type sum_N(0), sum_C(0), alfa(0);

        // Iterate over the columns of A.
        for ( a_col_it += lane_id ; utils::any( a_col_it < a_col_end ) ; a_col_it += WARP_SIZE )
        {
            // Columns of A maps to rows of B. Each thread of the warp loads its A-col/B-row ID.
            int a_col_id(-1);
            Value_type a_value(0);

            if ( a_col_it < a_col_end )
            {
                a_col_id = A_cols[a_col_it];
                a_value  = A_vals[a_col_it];
            }

            // Is it an off-diagonal element.
            bool is_off_diagonal = a_col_it < a_col_end && a_col_id != a_row_id;
            // Is it a strongly-connected column.
            bool is_strongly_connected = is_off_diagonal && s_con[a_col_it];
            // Is it isolated.
            bool is_strong_fine = is_off_diagonal && cf_map[a_col_id] == STRONG_FINE;
            // Is it a fine and strongly-connected column.
            bool is_assigned_strongly_connected = is_strongly_connected && assigned[a_col_id] == pass - 1;
            // Is it a weakly connected node.
            bool is_weakly_connected = is_off_diagonal && !is_assigned_strongly_connected;

            // Update the weak value.
            if ( is_weakly_connected && !is_strong_fine)
            {
                sum_N += a_value;
            }

            // We collect fine and strongly-collected columns.
            int vote = utils::ballot( is_assigned_strongly_connected );
            int dest = __popc( vote & utils::lane_mask_lt() );

            if ( is_assigned_strongly_connected )
            {
                s_b_row_ids[warp_id * WARP_SIZE + dest] = a_col_id;
                s_b_values[warp_id * WARP_SIZE + dest] = a_value;
            }

            int num_rows = __popc( vote );

            // For each warp, we have up to 32 rows of B to proceed.
            for ( int k = 0 ; k < num_rows ; ++k )
            {
                // Threads in the warp proceeds columns of B in the range [b_col_it, bCol_end).
                int b_row_id = s_b_row_ids[warp_id * WARP_SIZE + k];
                Value_type uniform_val = s_b_values[warp_id * WARP_SIZE + k];
                // TODO: make sure we have better memory accesses. b_colBegin rather than bCol_it because we iterate twice.
                int b_col_begin = P_rows[b_row_id  ];
                int b_col_end   = P_rows[b_row_id + 1];

                // _iterate over the range of columns of B.
                for ( int b_col_it = b_col_begin + lane_id ; utils::any( b_col_it < b_col_end ) ; b_col_it += WARP_SIZE )
                {
                    // The ID of the column.
                    KeyType b_col_id(-1);
                    Value_type tmp(0);

                    if ( b_col_it < b_col_end )
                    {
                        b_col_id = P_cols[b_col_it];
                        tmp = P_vals[b_col_it] * uniform_val;
                        sum_C += tmp;
                        sum_N += tmp;
                    }

                    // Update C_hat values. If the value is not in C_hat it will be skipped (true parameter).
                    // Need to get fine level id for b_col_id
                    // if (a_row_id == 8 && b_col_id == 3) printf("updating 8,3: %lg\n",tmp);
                    bool found = map.update( b_col_id, tmp );
                    bool any_found = (__popc(utils::ballot(found)));
                }
            }
        }

        // We're done with that row of A, now reduce sum_N and sum_C
#if __CUDA_ARCH__ >= 300
#pragma unroll

        for ( int mask = WARP_SIZE / 2 ; mask > 0 ; mask >>= 1 )
        {
            sum_C += utils::shfl_xor( sum_C, mask );
            sum_N += utils::shfl_xor( sum_N, mask );
        }

        if ( lane_id == 0 )
        {
            double div = (fabs(sum_C * diag[a_row_id]) < 1e-10) ? 1. : sum_C * diag[a_row_id];
            alfa = -sum_N / div;
            // alfa = -sum_N/(sum_C*diag[a_row_id]);
        }

        alfa = utils::shfl( alfa, 0 );
#else
        s_c_values[threadIdx.x] = sum_C;
        s_n_values[threadIdx.x] = sum_N;
#pragma unroll

        for ( int offset = WARP_SIZE / 2 ; offset > 0 ; offset >>= 1 )
            if ( lane_id < offset )
            {
                s_c_values[threadIdx.x] = sum_C += s_c_values[threadIdx.x + offset];
                s_n_values[threadIdx.x] = sum_N += s_n_values[threadIdx.x + offset];
            }

        sum_C = s_c_values[warp_id * WARP_SIZE];
        sum_N = s_n_values[warp_id * WARP_SIZE];

        if ( lane_id == 0 )
        {
            double div = (fabs(sum_C * diag[a_row_id]) < 1e-10) ? 1. : sum_C * diag[a_row_id];
            alfa = -sum_N / div;
            s_c_values[threadIdx.x] = alfa;
        }

        alfa = s_c_values[warp_id * WARP_SIZE];
#endif
#if __CUDA_ARCH__ >= 300
        int p_col_tmp = -1;

        if ( lane_id < 2 )
        {
            p_col_tmp = P_rows[a_row_id + lane_id];
        }

        int p_col_it  = utils::shfl( p_col_tmp, 0 );
        int p_col_end = utils::shfl( p_col_tmp, 1 );
#else

        if ( lane_id < 2 )
        {
            s_rows[2 * warp_id + lane_id] = P_rows[a_row_id + lane_id];
        }

        int p_col_it  = s_rows[2 * warp_id + 0];
        int p_col_end = s_rows[2 * warp_id + 1];
#endif
        map.store_keys_scale_values( p_col_end - p_col_it, &P_cols[p_col_it], alfa, &P_vals[p_col_it] );
    }

#endif
}

} // namespace multipass

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template< typename T >
__device__ __forceinline__
bool
sign( T x ) { return x >= T(0); }

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

/*
 * grab the diagonal of the matrix
 */
template <typename IndexType, typename ValueType>
__global__
void getDiagonalKernel(const IndexType *offsets, const IndexType *column_indices,
                       const ValueType *values, const IndexType numRows, ValueType *diagonal)
{
    for (int tIdx = threadIdx.x + blockDim.x * blockIdx.x; tIdx < numRows; tIdx += blockDim.x * gridDim.x)
    {
        const int offset = offsets[tIdx];
        const int numj = offsets[tIdx + 1] - offset;

        for (int j = offset; j < offset + numj; j++)
        {
            int jcol = column_indices[j];

            if (tIdx == jcol)
            {
                diagonal[tIdx] = values[j];
            }
        }
    }
}
/*************************************************************************
 * Implementing Extended+i algorithm from \S 4.5 of:
 * "Distance-two interpolation for parallel algebraic multigrid"
 * Reference [4] on wiki
 ************************************************************************/
template <AMGX_VecPrecision t_vecPrec, AMGX_MatPrecision t_matPrec, AMGX_IndPrecision t_indPrec>
Multipass_Interpolator<TemplateConfig<AMGX_device, t_vecPrec, t_matPrec, t_indPrec> >::Multipass_Interpolator(AMG_Config &cfg, const std::string &cfg_scope) : Base(cfg, cfg_scope)
{}

template <AMGX_VecPrecision t_vecPrec, AMGX_MatPrecision t_matPrec, AMGX_IndPrecision t_indPrec>
Multipass_Interpolator<TemplateConfig<AMGX_device, t_vecPrec, t_matPrec, t_indPrec> >::~Multipass_Interpolator()
{}

enum { WARP_SIZE = 32, GRID_SIZE = 128, SMEM_SIZE = 128 };


struct is_less_than_zero
{
    __host__ __device__
    bool operator()(int x)
    {
        return x < 0;
    }
};

struct is_strong_fine
{
    __host__ __device__
    bool operator()(int x)
    {
        return x == -3;
    }
};

template <AMGX_VecPrecision t_vecPrec, AMGX_MatPrecision t_matPrec, AMGX_IndPrecision t_indPrec>
void Multipass_Interpolator<TemplateConfig<AMGX_device, t_vecPrec, t_matPrec, t_indPrec> >::generateInterpolationMatrix_1x1(Matrix_d &A,
        IntVector &cf_map,
        BVector &s_con,
        IntVector &scratch,
        Matrix_d &P,
        void *amg_ptr )
{
    // Implementation based on paper "On long range interpolation operators for aggressive coarsening" by Ulrike Meier Yang, section 4.3
    const int blockSize = 256;
    typedef typename Matrix_d::index_type IndexType;
    typedef typename Matrix_d::value_type ValueType;
    // get raw pointers to data
    const IndexType *Aoffsets = A.row_offsets.raw();
    const IndexType *Acolumn_indices = A.col_indices.raw();
    const ValueType *Avalues = A.values.raw();
    const IndexType Anum_rows = (int) A.get_num_rows();
    int *cf_map_ptr = cf_map.raw();
    bool *s_con_ptr = s_con.raw();
    int size_two_ring;
    int size_one_ring;

    if (A.is_matrix_distributed())
    {
        int offset;
        A.getOffsetAndSizeForView(ALL, &offset, &size_two_ring);
        A.getOffsetAndSizeForView(FULL, &offset, &size_one_ring);
    }
    else
    {
        size_one_ring = A.get_num_rows();
        size_two_ring = A.get_num_rows();
    }

    int numBlocksOneRing = min( 4096, (int) (size_one_ring + blockSize - 1) / blockSize );
    int numBlocksTwoRing = min( 4096, (int) (size_two_ring + blockSize - 1) / blockSize );
    int numBlocks = min( 4096, (int) (A.get_num_rows() + blockSize - 1) / blockSize );
    // ----------------------------------------------------------
    // First fill out the assigned array and count # of passes
    // ----------------------------------------------------------
    IntVector assigned( size_one_ring, -1);
    IntVector C_hat_start( size_one_ring + 1, 0 );
    // assigned[i] = -1: unassigned point
    // assigned[i] = 0: coarse point
    // assigned[i] = 1: fine point directly connected to coarse point
    // assigned[i] = 2: fine point distance 2 away from strongly connected coarse point
    // assigned[i] = ...
    // Initialize assigned array by marking the coarse and fine point directly connected to coarse points
    const int cta_size = 256;
    const int nWarps = cta_size / 32;
    int grid_size = min( 4096, (int) (A.get_num_rows() + nWarps - 1) / nWarps);
    initializeAssignedArray<IndexType, cta_size> <<< grid_size, cta_size>>>(cf_map.raw(), assigned.raw(), A.row_offsets.raw(), A.col_indices.raw(), s_con.raw(), C_hat_start.raw(), Anum_rows);
    cudaCheckError();
    // Count the number of passes and fill assigned for pass > 1
    int pass = 2;
    int max_num_passes = 10;
    // Count the number of unassigned nodes by checking which nodes have assigned[i] = -1
    IndexType num_unassigned = thrust::count_if(assigned.begin(), assigned.end(), is_less_than_zero());
    cudaCheckError();
    IndexType num_strong_fine = thrust::count_if(cf_map.begin(), cf_map.end(), is_strong_fine());
    cudaCheckError();
    IndexType num_unassigned_max = num_unassigned - num_strong_fine;

    // Exchange 1 ring of assigned array
    if (A.is_matrix_distributed())
    {
        assigned.dirtybit = 1;
        A.manager->exchange_halo(assigned, assigned.tag);
        // get max across all processes
        A.manager->getComms()->all_reduce_max(num_unassigned, num_unassigned_max);
    }

    // printf("initial unassigned: %d\n",num_unassigned_max);

    // Fill the assigned array for the other nodes, going one pass at the time
    while (num_unassigned_max && pass < max_num_passes)
    {
        fillAssignedArray<IndexType, cta_size> <<< grid_size, cta_size>>>(assigned.raw(), A.row_offsets.raw(), A.col_indices.raw(), s_con.raw(), Anum_rows, pass);
        num_unassigned = thrust::count_if(assigned.begin(), assigned.begin() + A.get_num_rows(), is_less_than_zero());
        cudaCheckError();
        num_unassigned_max = num_unassigned - num_strong_fine;

        if (A.is_matrix_distributed())
        {
            assigned.dirtybit = 1;
            A.manager->exchange_halo(assigned, assigned.tag);
            // get max across all processes
            A.manager->getComms()->all_reduce_max(num_unassigned, num_unassigned_max);
        }

        // printf("after pass %d, %d unassigned\n",pass, num_unassigned_max);
        pass++;
    }

    cudaCheckError();
    // Check that we haven't reached maximum number of passes
    //if (pass == max_num_passes)
    //FatalError("Multipass intepolator reached maximum number of passes, exiting\n",AMGX_ERR_NOT_IMPLEMENTED);
    int num_passes = pass;
    // -------------------------------------------------------------------------------------
    // Create array cf_map_global, which assigns a global index to all owned coarse points
    // -------------------------------------------------------------------------------------
    I64Vector_d cf_map_global(size_one_ring);
    DistributedArranger<TConfig_d> *prep = NULL;
    I64Vector_d P_col_indices_global;
    int coarsePoints;

    if (A.is_matrix_distributed())
    {
        prep = new DistributedArranger<TConfig_d>;
        int num_owned_fine_pts = A.get_num_rows();
        int num_owned_coarse_pts = thrust::count_if(cf_map.begin(), cf_map.begin() + num_owned_fine_pts, is_non_neg());
        cudaCheckError();
        int num_halo_coarse_pts = thrust::count_if(cf_map.begin() + num_owned_fine_pts, cf_map.end(), is_non_neg());
        cudaCheckError();
        coarsePoints = num_owned_coarse_pts + num_halo_coarse_pts;
        // Partially initialize the distributed manager of matrix P, using num_owned_coarse_pts
        prep->initialize_manager(A, P, num_owned_coarse_pts);
        int my_rank = A.manager->global_id();
        const int cta_size = 128;
        const int grid_size = std::min( 4096, (num_owned_fine_pts + cta_size - 1) / cta_size);
        // Fill the cf_map_global array, which will assign for each coarse pt, the global index of that coarse pt
        createCfMapGlobal<IndexType, int64_t> <<< grid_size, cta_size>>>(cf_map.raw(), cf_map_global.raw(), P.manager->part_offsets_h[my_rank], num_owned_fine_pts);
        cudaCheckError();
        // Exchange the cf_map_global so that we know the coarse global id of halo nodes in one ring
        cf_map_global.dirtybit = 1;
        A.manager->exchange_halo(cf_map_global, cf_map_global.tag);
    }
    else
    {
        coarsePoints = (int) thrust::count_if(cf_map.begin(), cf_map.end(), is_non_neg());
        cudaCheckError();
        const int cta_size = 128;
        const int grid_size = std::min( 4096, (A.get_num_rows() + cta_size - 1) / cta_size);
        fill_cf_map_global <<< grid_size, cta_size>>>(cf_map.raw(), cf_map_global.raw(), A.get_num_rows());
        cudaCheckError();
    }

    // ----------------------------------------------------------
    // Create an upper bound for the length of each row P
    // ----------------------------------------------------------
    Hash_Workspace<TConfig_d, int64_t> exp_wk;
    {
        const int CTA_SIZE  = 256;
        const int NUM_WARPS = CTA_SIZE / WARP_SIZE;
        int work_offset = GRID_SIZE * NUM_WARPS;
        cudaMemcpy( exp_wk.get_work_queue(), &work_offset, sizeof(int), cudaMemcpyHostToDevice );

        // Do one ring exchange of C_hat_start, which have been filled for nodes with assigned=1
        if (A.is_matrix_distributed())
        {
            C_hat_start.dirtybit = 1;
            A.manager->exchange_halo(C_hat_start, C_hat_start.tag);
        }

        for (int i = 2; i < num_passes; i++)
        {
            multipass::estimate_c_hat_size_kernel< CTA_SIZE, WARP_SIZE> <<< 2048, CTA_SIZE>>>(
                A.get_num_rows(),
                A.row_offsets.raw(),
                A.col_indices.raw(),
                s_con.raw(),
                assigned.raw(),
                C_hat_start.raw(),
                i);

            if (A.is_matrix_distributed())
            {
                // TODO: This could be improved by only doing the exchange for points in current pass
                C_hat_start.dirtybit = 1;
                A.manager->exchange_halo(C_hat_start, C_hat_start.tag);
            }
        }

        cudaCheckError();
    }
    // Do an exclusive scan on C_hat_start to compute compute upper bound on row offsets
    thrust::exclusive_scan( C_hat_start.begin( ), C_hat_start.end( ), C_hat_start.begin( ) );
    cudaCheckError();
    // Now C_hat_start contains the offsets
    // Create a temporary manager that can be used to exchange halo information on values and/or column indices, where there's more than 1 value associated with each node
    Matrix<TConfig_d> C_hat_matrix(1, 1, 1, 0);

    // Create manager to exchange C_hat data
    if (A.is_matrix_distributed())
    {
        prep->createTempManager(C_hat_matrix, A, C_hat_start);
    }

    // ----------------------------------------------------------
    // Count the actual number of nonzeroes in each row of P
    // ----------------------------------------------------------
    // Allocate memory to store columns/values.
    int nVals = C_hat_start[C_hat_start.size() - 1];
    // C_hat stores the global indices of the coarse nodes that each fine point depends on
    I64Vector_d C_hat( nVals, -1 );
    // C_hat_pos is an array used by the hash set structure (check with Julien for more details)
    IntVector C_hat_pos( nVals );
    // C_hat_size stores the number of coarse points on which each fine point depends on
    IntVector C_hat_size( size_one_ring + 1, 0 );
    {
        const int CTA_SIZE  = 256;
        const int NUM_WARPS = CTA_SIZE / WARP_SIZE;
        int avg_nz_per_row = A.get_num_nz() / A.get_num_rows();
        // This will fill C_hat for nodes with assigned = 1 (for fine points strongly connected to coarse points)
        multipass::compute_c_hat_first_pass_kernel< CTA_SIZE, WARP_SIZE, int64_t> <<< GRID_SIZE, CTA_SIZE>>>(
            A.get_num_rows(),
            A.row_offsets.raw(),
            A.col_indices.raw(),
            s_con.raw(),
            cf_map_global.raw(),
            C_hat_start.raw(),
            C_hat_size.raw(),
            C_hat.raw(),
            assigned.raw()
        );
        cudaCheckError();

        // Here exchange C_hat_size and C_hat for one ring neighbors
        if (A.is_matrix_distributed())
        {
            //TODO: We could probably reduce the amount of data exchanged here, since we only need to exchange for recently updated nodes
            C_hat.dirtybit = 1;
            C_hat_matrix.manager->exchange_halo(C_hat, C_hat.tag);
            C_hat_size.dirtybit = 1;
            A.manager->exchange_halo(C_hat_size, C_hat_size.tag);
        }

        for (int i = 2; i < num_passes; i++)
        {
            // In case the hash table is not big enough, run multiple attemps
            int attempt = 0;

            for ( bool done = false ; !done && attempt < 10 ; ++attempt )
            {
                // Double the amount of GMEM (if needed).
                if ( attempt > 0 )
                {
                    exp_wk.expand();
                }

                // Reset the status. TODO: Launch async copies.
                int status = 0;
                cudaMemcpy( exp_wk.get_status(), &status, sizeof(int), cudaMemcpyHostToDevice );
                // Compute the set C_hat.
                int work_offset = GRID_SIZE * NUM_WARPS;
                cudaMemcpy( exp_wk.get_work_queue(), &work_offset, sizeof(int), cudaMemcpyHostToDevice );
                // Run the computation.
                multipass::compute_c_hat_kernel< 8, CTA_SIZE, SMEM_SIZE, WARP_SIZE, int64_t> <<< GRID_SIZE, CTA_SIZE>>>(
                    A.get_num_rows(),
                    A.row_offsets.raw(),
                    A.col_indices.raw(),
                    s_con.raw(),
                    C_hat_start.raw(),
                    C_hat_size.raw(),
                    C_hat.raw(),
                    C_hat_pos.raw(),
                    assigned.raw(),
                    exp_wk.get_gmem_size(),
                    exp_wk.get_keys(),
                    exp_wk.get_work_queue(),
                    exp_wk.get_status(),
                    i);

                if (A.is_matrix_distributed())
                {
                    //TODO: We could probably reduce the amount of data exchanged here, since we only need to exchange for recently updated nodes
                    C_hat.dirtybit = 1;
                    C_hat_matrix.manager->exchange_halo(C_hat, C_hat.tag);
                    C_hat_size.dirtybit = 1;
                    A.manager->exchange_halo(C_hat_size, C_hat_size.tag);
                }

                cudaCheckError();
                // Read the result from count_non_zeroes.
                cudaMemcpy( &status, exp_wk.get_status(), sizeof(int), cudaMemcpyDeviceToHost );
                done = status == 0;
            }
        }
    }
    // At this stage, for each fine node (even in 1 ring halo), we have the global indices
    // of the interpolatory coarse points
    // count the number of non-zeros in the interpolation matrix,
    // For coarse point, nonzeros = 1
    // for strong fine, nonzeros = 0
    // for other points, nonzeros[i] = C_hat_size[i]
    IntVector nonZeroOffsets(size_one_ring + 1);
    IntVector nonZerosPerRow(size_one_ring);
    nonZerosPerRowSizeKernel <<< numBlocks, blockSize>>>(Anum_rows, cf_map_ptr, C_hat_size.raw(),
            nonZerosPerRow.raw());
    cudaCheckError();

    // Do a one ring halo exchange to know the lengths of the neighbors row
    if (A.is_matrix_distributed())
    {
        nonZerosPerRow.dirtybit = 1;
        A.manager->exchange_halo(nonZerosPerRow, nonZerosPerRow.tag);
    }

    // get total number of non-zeros with reduction
    int nonZeros = thrust::reduce(nonZerosPerRow.begin(), nonZerosPerRow.end());
    cudaCheckError();
    // get the offsets with an exclusive scan
    thrust::exclusive_scan(nonZerosPerRow.begin(), nonZerosPerRow.end(), nonZeroOffsets.begin());
    cudaCheckError();
    nonZeroOffsets[nonZeroOffsets.size() - 1] = nonZeros;
    // resize P
    P.resize(0, 0, 0, 1);
    P.addProps(CSR);
    // TODO: is the number of columns used anywhere?
    P.resize(size_one_ring, coarsePoints, nonZeros, 1);
    cudaCheckError();
    P_col_indices_global.resize(P.col_indices.size());
    Matrix<TConfig_d> P_temp(1, 1, 1, 0);

    if (A.is_matrix_distributed())
    {
        prep->createTempManager(P_temp, A, nonZeroOffsets);
    }

    IndexType *Poffsets = P.row_offsets.raw();
    IndexType *Pcolumn_indices = P.col_indices.raw();
    ValueType *Pvalues = P.values.raw();
    // copy nonzero offsets to the P matrix
    thrust::copy(nonZeroOffsets.begin(), nonZeroOffsets.end(), P.row_offsets.begin());
    cudaCheckError();
    // grab the diagonal terms
    VVector diag(size_one_ring);
    ValueType *diag_ptr = diag.raw();
    {
        find_diag_kernel_indexed_dia <<< numBlocksOneRing, blockSize>>>(
            size_one_ring,
            A.diag.raw(),
            A.values.raw(),
            diag.raw());
    }
    cudaCheckError();
    // Fill P.col_indices_global, P.values
    {
        const int CTA_SIZE  = 256;
        const int NUM_WARPS = CTA_SIZE / WARP_SIZE;
        int avg_nz_per_row = A.get_num_nz() / A.get_num_rows();
        // Compute the set C_hat.
        int work_offset = GRID_SIZE * NUM_WARPS;
        cudaMemcpy( exp_wk.get_work_queue(), &work_offset, sizeof(int), cudaMemcpyHostToDevice );
        // Run the computation.
        typedef typename MatPrecisionMap<t_matPrec>::Type Value_type;
        multipass::compute_interp_weight_first_pass_kernel<Value_type, CTA_SIZE, SMEM_SIZE, WARP_SIZE, int64_t> <<< GRID_SIZE, CTA_SIZE>>>(
            A.get_num_rows(),
            A.row_offsets.raw(),
            A.col_indices.raw(),
            A.values.raw(),
            cf_map_global.raw(),
            s_con.raw(),
            diag.raw(),
            P.row_offsets.raw(),
            P_col_indices_global.raw(),
            P.values.raw(),
            assigned.raw());
        cudaCheckError();

        if (A.is_matrix_distributed())
        {
            //TODO: We could probably reduce the amount of data exchanged here, since we only need to exchange for recently updated nodes
            P_col_indices_global.dirtybit = 1;
            P_temp.manager->exchange_halo(P_col_indices_global, P_col_indices_global.tag);
            P.values.dirtybit = 1;
            P_temp.manager->exchange_halo(P.values, P.values.tag);
        }

        for (int i = 2; i < num_passes; i++)
        {
            cudaMemcpy( exp_wk.get_work_queue(), &work_offset, sizeof(int), cudaMemcpyHostToDevice );
            typedef typename MatPrecisionMap<t_matPrec>::Type Value_type;
            multipass::compute_interp_weight_kernel<Value_type, CTA_SIZE, SMEM_SIZE, WARP_SIZE, int64_t> <<< GRID_SIZE, CTA_SIZE>>>(
                A.get_num_rows(),
                A.row_offsets.raw(),
                A.col_indices.raw(),
                A.values.raw(),
                cf_map.raw(),
                s_con.raw(),
                C_hat.raw(),
                C_hat_pos.raw(),
                C_hat_start.raw(),
                C_hat_size.raw(),
                diag.raw(),
                P.row_offsets.raw(),
                P_col_indices_global.raw(),
                P.values.raw(),
                exp_wk.get_gmem_size(),
                exp_wk.get_keys(),
                exp_wk.get_vals(),
                exp_wk.get_work_queue(),
                assigned.raw(),
                i);

            if (A.is_matrix_distributed())
            {
                //TODO: This is exchanging too much data
                P_col_indices_global.dirtybit = 1;
                P_temp.manager->exchange_halo(P_col_indices_global, P_col_indices_global.tag);
                P.values.dirtybit = 1;
                P_temp.manager->exchange_halo(P.values, P.values.tag);
            }

            cudaCheckError();
        }

        cudaCheckError();
    }

    if (!A.is_matrix_distributed())
    {
        const int cta_size = 128;
        const int grid_size = std::min( 4096, ( (int) P.col_indices.size() + cta_size - 1) / cta_size);
        // Simply cast from int64 to int
        fill_P_col_indices <<< grid_size, cta_size>>>(P_col_indices_global.raw(), P.col_indices.raw(), (int) P.col_indices.size());
        cudaCheckError();
    }
    else
    {
        // Resize P to only include the owned rows and columns
        int P_num_nnz = P.row_offsets[P.get_num_rows()];
        P.resize(A.get_num_rows(), P.manager->halo_offsets[0], P_num_nnz, 1);
        // Using the global col indices, identify neighbors for P.manager, fill B2L_maps, etc...
        prep->initialize_manager_from_global_col_indices(P, P_col_indices_global);
        // Create the list of interior and boundary nodes (this is necessary if doing masked SpMV operations)
        prep->createRowsLists(P, true);
    }

    // Delete the temporary distributed arranger
    if (prep != NULL) { delete prep; }
}

template< class T_Config>
void Multipass_InterpolatorBase<T_Config>::generateInterpolationMatrix(Matrix<T_Config> &A,
        IntVector &cf_map,
        BVector &s_con,
        IntVector &scratch,
        Matrix<T_Config> &P,
        void *amg)
{
    P.set_initialized(0);
    ViewType oldView = A.currentView();
    A.setView(OWNED);

    if (A.get_block_size() == 1)
    {
        generateInterpolationMatrix_1x1(A, cf_map, s_con, scratch, P, amg);
    }
    else
    {
        FatalError("Unsupported dimensions for multipass interpolator", AMGX_ERR_NOT_SUPPORTED_BLOCKSIZE);
    }

    A.setView(oldView);
    //P.computeDiagonal();
    P.set_initialized(1);
#if 0
    P.printToFile("P.out", "", -1, -1);
#endif
}

#define AMGX_CASE_LINE(CASE) template class Multipass_InterpolatorBase<TemplateMode<CASE>::Type>;
AMGX_FORALL_BUILDS(AMGX_CASE_LINE)
#undef AMGX_CASE_LINE

#define AMGX_CASE_LINE(CASE) template class Multipass_Interpolator<TemplateMode<CASE>::Type>;
AMGX_FORALL_BUILDS(AMGX_CASE_LINE)
#undef AMGX_CASE_LINE

} // namespace amgx
