// SPDX-FileCopyrightText: 2011 - 2024 NVIDIA CORPORATION. All Rights Reserved.
//
// SPDX-License-Identifier: BSD-3-Clause

#include <classical/interpolators/multipass.h>
#include <classical/interpolators/common.h>
#include <device_properties.h>
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
#include <thrust_wrapper.h>

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
                bool is_coarse_strongly_connected = aColIt < aColEnd && s_con[aColIt] && is_coarse;
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

template <typename IndexType, int cta_size, int group_size>
__global__
void initializeAssignedArray_opt(const IndexType *cf_map, IndexType *assigned, const IndexType *A_rows, const IndexType *A_cols, const bool *s_con, IndexType *C_hat_offsets, int A_num_rows)
{
    const int ngroups = cta_size / group_size;

    // The coordinates of the thread inside the CTA/group.
    const int group_id = threadIdx.x / group_size;
    const int lane_id = threadIdx.x % group_size;

    int a_row_id = blockIdx.x * ngroups + group_id;

    bool active = true;
    if(a_row_id > A_num_rows)
    {
        active = false;
    }

    int coarse_fine_id = UNASSIGNED;

    if(active)
    {
        coarse_fine_id = cf_map[a_row_id];

        if(coarse_fine_id >= 0)
        {
            if(lane_id == 0)
            {
                assigned[a_row_id] = 0;
            }

            active = false;
        }
        else if(coarse_fine_id != FINE) // Strong fine
        {
            if(lane_id == 0)
            {
                assigned[a_row_id] = -1;
            }

            active = false;
        }
    }

    active = (active && coarse_fine_id == FINE);

    constexpr int warp_size = 32;
    unsigned int base_mask = 0xFFFFFFFF >> (warp_size - group_size);
    int ngroups_in_warp = warp_size / group_size;
    int group_id_in_warp = group_id % ngroups_in_warp;
    unsigned int group_mask = base_mask << (group_id_in_warp*group_size);

    // Load A row IDs.
    int aColIt  = A_rows[a_row_id  ];
    int aColEnd = A_rows[a_row_id + 1];

    // Iterate over the columns of A.
    int coarse_count = 0;
    for ( aColIt += lane_id ; utils::any( active && aColIt < aColEnd ) ; aColIt += group_size )
    {
        bool is_coarse_strongly_connected = false;

        if(active)
        {
            // Columns of A maps to rows of B. Each thread of the warp loads its A-col/B-row ID.
            int aColId = aColIt < aColEnd ? A_cols[aColIt] : -1;

            // Is it an off-diagonal element.
            bool is_off_diagonal = aColIt < aColEnd && aColId != a_row_id;

            // Is it fine.
            bool is_coarse  = is_off_diagonal && cf_map[aColId] >= 0;

            // Is it strongly connected ?
            is_coarse_strongly_connected = s_con[aColIt] && is_coarse;
        }

        // Count any of the threads in the group that had strong connections
        int coarse_exist_flags = utils::ballot(is_coarse_strongly_connected);
        if(lane_id == 0)
        {
            coarse_count += __popc(coarse_exist_flags & group_mask);
        }
    }

    // If fine point connected to coarse point, assigned = 1,
    if (active && coarse_count > 0)
    {
        if(lane_id == 0)
        {
            assigned[a_row_id] = 1;
            C_hat_offsets[a_row_id] = coarse_count;
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
                bool is_assigned_strongly_connected = aColIt < aColEnd && s_con[aColIt] && is_assigned;
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

template < typename IndexType, int cta_size, int group_size >
__global__
void fillAssignedArray_opt(IndexType *assigned, const IndexType *A_rows, const IndexType *A_cols, const bool *s_con, int A_num_rows, int pass)
{
    // The coordinates of the thread inside the CTA/groups
    const int group_id = threadIdx.x / group_size;
    const int lane_id = threadIdx.x % group_size;
    const int ngroups_per_block = cta_size / group_size;

    int a_row_id = blockIdx.x * ngroups_per_block + group_id;

    bool active = true;
    if (a_row_id >= A_num_rows || assigned[a_row_id] != -1)
    {
        active = false;
    }

    // Mask for the subgroup
    constexpr int warp_size = 32;
    unsigned int base_mask = 0xFFFFFFFF >> (warp_size - group_size);
    int ngroups_in_warp = warp_size / group_size;
    int group_id_in_warp = group_id % ngroups_in_warp;
    unsigned int group_mask = base_mask << (group_id_in_warp*group_size);

    // Load A row IDs.
    int a_col_begin = 0;
    int a_col_end   = 0;

    if(active)
    {
        a_col_begin = A_rows[a_row_id  ];
        a_col_end   = A_rows[a_row_id + 1];
    }

    // Iterate over the columns of A.
    for (int a_col_it = a_col_begin + lane_id ; utils::any(active && a_col_it < a_col_end); a_col_it += group_size)
    {
        bool is_assigned_strongly_connected = false;

        if(active)
        {
            // Columns of A maps to rows of B. Each thread of the group loads its A-col/B-row ID.
            int a_col_id = a_col_it < a_col_end ? A_cols[a_col_it] : -1;

            // Is it an off-diagonal element.
            bool is_off_diagonal = a_col_it < a_col_end && a_col_id != a_row_id;

            // Is it assigned
            bool is_assigned = is_off_diagonal && assigned[a_col_id] == pass - 1;

            // Is it strongly connected ?
            is_assigned_strongly_connected = s_con[a_col_it] && is_assigned;
        }

        int group_strongly_connected = utils::ballot(is_assigned_strongly_connected);

        // Push coarse and strongly connected nodes in the set.
        if(active && (group_strongly_connected & group_mask))
        {
            if(lane_id == 0)
            {
                assigned[a_row_id] = pass;
            }

            active = false;
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
        Matrix_h &P)
{
    FatalError("multipass interpolation not implemented on host\n", AMGX_ERR_NOT_IMPLEMENTED);
} // end multipass interpolator


///////////////////////////////////////////////////////////////////////////////

// DEVICE CODE

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

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
#pragma unroll

        for ( int mask = WARP_SIZE / 2 ; mask > 0 ; mask >>= 1 )
        {
            my_count += utils::shfl_xor( my_count, mask );
        }

        // Write result -- RACE CONDITION!
        if (lane_id == 0)
        {
            C_hat_offsets[a_row_id] = my_count;
        }
    }
}

template< int CTA_SIZE, int group_size >
__global__ __launch_bounds__( CTA_SIZE )
void
estimate_c_hat_size_kernel_opt( const int A_num_rows,
                            const int *A_rows,
                            const int *A_cols,
                            const bool *s_con,
                            const int *assigned,
                            int *C_hat_offsets,
                            int pass )
{
    const int group_id = threadIdx.x / group_size;
    const int lane_id = threadIdx.x % group_size;

    int ngroups = CTA_SIZE / group_size;
    int a_row_id = blockIdx.x * ngroups + group_id;

    // Skip coarse rows and fine rows touching a coarse point
    bool active = true;
    if (a_row_id >= A_num_rows || assigned[a_row_id] != pass)
    {
        active = false;
    }

    int my_count = 0;

    // Load A row IDs.
    if(active)
    {
        int a_col_begin = A_rows[a_row_id  ];
        int a_col_end   = A_rows[a_row_id + 1];

        // Iterate over the columns of A to build C_hat.
        for (int a_col_it = a_col_begin + lane_id; a_col_it < a_col_end; a_col_it += group_size )
        {
            // Columns of A maps to rows of B. Each thread of the warp loads its A-col/B-row ID.
            int a_col_id = A_cols[a_col_it];

            // Is it an off-diagonal element.
            bool is_off_diagonal = a_col_id != a_row_id;

            // Is it strongly connected ?
            bool is_strongly_connected = is_off_diagonal && s_con[a_col_it];

            // assigned == pass-1
            bool is_assigned = is_off_diagonal && (assigned[a_col_id] == pass - 1);

            // Is it a coarse, strongly-connected column and already assigned?
            bool is_assigned_strongly_connected = is_strongly_connected && is_assigned;

            // If so, add their C_hat_offset value to my_count
            my_count += (is_assigned_strongly_connected ? C_hat_offsets[a_col_id] : 0);
        }
    }

#pragma unroll
    for(int i = group_size / 2; i > 0; i /= 2)
    {
        my_count += utils::shfl_down(my_count, i);
    }

    if (active && lane_id == 0)
    {
        C_hat_offsets[a_row_id] = my_count;
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
        int a_col_tmp = -1;

        if ( lane_id < 2 )
        {
            a_col_tmp = A_rows[a_row_id + lane_id];
        }

        int a_col_begin = utils::shfl( a_col_tmp, 0 );
        int a_col_end   = utils::shfl( a_col_tmp, 1 );

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

    template< int CTA_SIZE, int WARP_SIZE, int group_size, typename KeyType>
__global__ __launch_bounds__( CTA_SIZE )
    void
compute_c_hat_first_pass_kernel_opt( int A_num_rows,
        const int *__restrict A_rows,
        const int *__restrict A_cols,
        const bool *__restrict s_con,
        const KeyType *__restrict cf_map_global,
        const int *__restrict C_hat_start,
        int *__restrict C_hat_size,
        KeyType *__restrict C_hat,
        int *__restrict assigned)
{
    const int ngroups = CTA_SIZE / group_size;

    // The coordinates of the thread inside the CTA/warp.
    const int group_id = threadIdx.x / group_size;
    const int lane_id = threadIdx.x % group_size;

    // First threads load the row IDs of A needed by the CTA...
    int a_row_id = blockIdx.x * ngroups + group_id;

    bool active = true;

    // Skip if not in current pass
    if (a_row_id >= A_num_rows || assigned[a_row_id] != 1)
    {
        active = false;
    }

    int a_col_begin = 0;
    int a_col_end   = 0;
    int c_col_it = 0;

    if(active)
    {
        a_col_begin = A_rows[a_row_id];
        a_col_end   = A_rows[a_row_id + 1];
        c_col_it = C_hat_start[a_row_id];
    }

    unsigned int base_mask = 0xFFFFFFFF >> (WARP_SIZE - group_size);
    int ngroups_in_warp = WARP_SIZE / group_size;
    int group_id_in_warp = group_id % ngroups_in_warp;
    unsigned int group_mask = base_mask << (group_id_in_warp*group_size);

    // _iterate over the columns of A to build C_hat.
    int count = 0;
    for ( int a_col_it = a_col_begin + lane_id ; utils::any(active && a_col_it < a_col_end) ; a_col_it += group_size )
    {
        bool is_assigned_strongly_connected = false;

        // Columns of A maps to rows of B. Each thread of the warp loads its A-col/B-row ID.
        int a_col_id = -1;

        if(active)
        {
            if (a_col_it < a_col_end)
            {
                a_col_id = A_cols[a_col_it];
            }

            // Is it an off-diagonal element.
            bool is_off_diagonal = (a_col_it < a_col_end) && (a_col_id != a_row_id);

            // Is it strongly connected ?
            bool is_strongly_connected = is_off_diagonal && s_con[a_col_it];

            // assigned == pass -1
            bool is_assigned = is_off_diagonal && (assigned[a_col_id] == 0);

            // Is it an assigned==pass -1 and strongly-connected column.
            is_assigned_strongly_connected = is_strongly_connected && is_assigned;
        }

        // We collect fine and strongly-collected columns.
        int all_vote = utils::ballot( is_assigned_strongly_connected );
        int group_vote = all_vote & group_mask;
        int dest = __popc( group_vote & utils::lane_mask_lt() );

        if ( active && is_assigned_strongly_connected )
        {
            C_hat[c_col_it + count + dest] = cf_map_global[a_col_id];
        }

        count += __popc( group_vote );
    }

    if ( active && lane_id == 0 )
    {
        C_hat_size[a_row_id] = count;
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
    __shared__ int s_b_row_ids[CTA_SIZE];
    // The hash keys stored in shared memory.
    __shared__ KeyType s_keys[NUM_WARPS * SMEM_SIZE];
    // The coordinates of the thread inside the CTA/warp.
    const int warp_id = utils::warp_id( );
    const int lane_id = utils::lane_id( );
    // Constants.
    const int lane_id_div_num_threads = lane_id / NUM_THREADS_PER_ROW;
    const int lane_id_mod_num_threads = lane_id % NUM_THREADS_PER_ROW;
    // First threads load the row IDs of A needed by the CTA...
    int a_row_id = blockIdx.x * NUM_WARPS + warp_id;
    // Create local storage for the set.
    Hash_set<KeyType, SMEM_SIZE, 4, WARP_SIZE> set( &s_keys[warp_id * SMEM_SIZE], &g_keys[a_row_id * gmem_size], gmem_size );
    // Loop over rows of A.
    for ( ; a_row_id < A_num_rows ; a_row_id = get_work( wk_work_queue, warp_id ) )
    {
        // Skip if not in current pass
        if (assigned[a_row_id] != pass)
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
            // Check if assigned == pass -1
            bool is_assigned = is_off_diagonal && (assigned[a_col_id] == pass - 1);
            // Check if we have a strong connection to the point
            bool is_strongly_connected = is_active && s_con[a_col_it];
            // Check if strong connection to assigned == pass-1
            bool is_assigned_strongly_connected = is_assigned && is_strongly_connected;
            // We collect columns for which assigned == pass - 1
            int vote = utils::ballot( is_assigned_strongly_connected);
            int dest = __popc( vote & utils::lane_mask_lt() );

            if ( is_assigned_strongly_connected)
            {
                s_b_row_ids[warp_id * WARP_SIZE + dest] = a_col_id;
            }

            __syncwarp();

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

template <int GROUP_SIZE, int CTA_SIZE, int HASH_SIZE, class KeyType >
__global__ __launch_bounds__( CTA_SIZE )
void
compute_c_hat_kernel_opt( int assigned_set_size,
                      const int *__restrict A_rows,
                      const int *__restrict A_cols,
                      const bool *__restrict s_con,
                      const int *__restrict C_hat_start,
                      int *__restrict C_hat_size,
                      KeyType *__restrict C_hat,
                      int *__restrict C_hat_pos,
                      int *__restrict assigned,
                      int *__restrict assigned_set,
                      int pass )
{
    constexpr KeyType SLOT_VACANT = -1;

    // Group indices
    constexpr int ngroups = CTA_SIZE / GROUP_SIZE;
    const int group_id = threadIdx.x / GROUP_SIZE;
    const int lane_id = threadIdx.x % GROUP_SIZE;

    const int gid = blockIdx.x * ngroups + group_id;

    // Dynamic sized shared memory
    extern __shared__ int s[];

    // Block-level hash container storage
    KeyType* key_s = (KeyType*)s;

    //KeyType* data_s = (KeyType*)&key_s[ngroups*HASH_SIZE];
    int* col_ind_s = (int*)&key_s[ngroups*HASH_SIZE];

    // Group-level hash containers
    KeyType* key_group_s = &key_s[group_id*HASH_SIZE];
    //KeyType* data_group_s = &data_s[group_id*HASH_SIZE];

    // Initialise the keys and values.
#pragma unroll
    for(int i = threadIdx.x; i < ngroups*HASH_SIZE; i += CTA_SIZE)
    {
        key_s[i] = SLOT_VACANT; // Inserted keys will be in range [0,N]
        //data_s[i] = 0; // We will sum into values
    }

    if(lane_id == 0)
    {
        col_ind_s[group_id] = 0;
    }

    __syncthreads();

    int a_row_id = -1;

    if(gid < assigned_set_size)
    {
        // One row of A per group
        a_row_id = assigned_set[gid];

        // Iterate over the columns of A to build C_hat.
        for ( int a_col_it = A_rows[a_row_id] + lane_id ; a_col_it < A_rows[a_row_id + 1]; a_col_it += GROUP_SIZE )
        {
            int a_col_id = A_cols[a_col_it];

            // Is it an off-diagonal element.
            bool is_off_diagonal = a_col_id != a_row_id;

            // Check if assigned == pass -1
            bool is_assigned = is_off_diagonal && (assigned[a_col_id] == pass - 1);

            // Check if we have a strong connection to the point
            bool is_strongly_connected = s_con[a_col_it];

            // Check if strong connection to assigned == pass-1
            bool is_assigned_strongly_connected = is_assigned && is_strongly_connected;

            if(!is_assigned_strongly_connected) continue;

            // Threads in the warp proceeds columns of B in the range [bColIt, bColEnd).
            int b_col_beg  = C_hat_start[a_col_id];

            // Iterate over the range of columns of B.
            for (int b_col_it = b_col_beg; b_col_it < b_col_beg + C_hat_size[a_col_id]; ++b_col_it)
            {
                KeyType key = C_hat[b_col_it];

                int hash = key % HASH_SIZE;

                while(true)
                {
                    // XXX You can just avoid this and directly atomicCAS, which is faster?
                    int curr_key = key_group_s[hash];
                    if(curr_key == key) break;
                    
                    if(curr_key == SLOT_VACANT)
                    {
                        KeyType old_key = (KeyType)atomicCAS((unsigned long long*)&key_group_s[hash], (unsigned long long)SLOT_VACANT, (unsigned long long)key);
                        if(old_key == SLOT_VACANT || old_key == key)
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
    if(gid < assigned_set_size)
    {
        // Store the results.
        int c_col_it = C_hat_start[a_row_id];

#pragma unroll
        for(int i = lane_id; i < HASH_SIZE; i += GROUP_SIZE)
        {
            KeyType key = key_group_s[i];
            if(key != SLOT_VACANT)
            {
                int ind = atomicAdd(&col_ind_s[group_id], 1);
                C_hat[c_col_it + ind] = key;
                //C_hat_pos[c_col_it + ind] = i;
            }
        }
    }

    __syncthreads();

    if(gid < assigned_set_size)
    {
        if(lane_id == 0)
        {
            C_hat_size[a_row_id] = col_ind_s[group_id];
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
            int a_col_tmp = -1;

            if ( lane_id < 2 )
            {
                a_col_tmp = A_rows[a_row_id + lane_id];
            }

            int a_col_it  = utils::shfl( a_col_tmp, 0 );
            int a_col_end = utils::shfl( a_col_tmp, 1 );

            int p_col_tmp = -1;

            if ( lane_id < 2 )
            {
                p_col_tmp = P_rows[a_row_id + lane_id];
            }

            int p_col_it  = utils::shfl( p_col_tmp, 0 );
            int p_col_end = utils::shfl( p_col_tmp, 1 );

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
#pragma unroll
            for ( int mask = WARP_SIZE / 2 ; mask > 0 ; mask >>= 1 )
            {
                sum_C += utils::shfl_xor( sum_C, mask );
                sum_N += utils::shfl_xor( sum_N, mask );
            }

            if ( lane_id == 0 )
            {
                // NOTE this matches the check for zero in HYPRE
                double div = (fabs(sum_C * diag[a_row_id]) == 0.0) ? 1. : sum_C * diag[a_row_id];
                alfa = -sum_N / div;
            }

            alfa = utils::shfl( alfa, 0 );

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
compute_interp_weight_first_pass_kernel_opt( const int A_num_rows,
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
            int a_col_tmp = -1;

            if ( lane_id < 2 )
            {
                a_col_tmp = A_rows[a_row_id + lane_id];
            }

            int a_col_it  = utils::shfl( a_col_tmp, 0 );
            int a_col_end = utils::shfl( a_col_tmp, 1 );

            int p_col_tmp = -1;

            if ( lane_id < 2 )
            {
                p_col_tmp = P_rows[a_row_id + lane_id];
            }

            int p_col_it  = utils::shfl( p_col_tmp, 0 );
            int p_col_end = utils::shfl( p_col_tmp, 1 );

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
#pragma unroll
            for ( int mask = WARP_SIZE / 2 ; mask > 0 ; mask >>= 1 )
            {
                sum_C += utils::shfl_xor( sum_C, mask );
                sum_N += utils::shfl_xor( sum_N, mask );
            }

            if ( lane_id == 0 )
            {
                // NOTE this matches the check for zero in HYPRE
                double div = (fabs(sum_C * diag[a_row_id]) == 0.0) ? 1. : sum_C * diag[a_row_id];
                alfa = -sum_N / div;
            }

            alfa = utils::shfl( alfa, 0 );

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
compute_interp_weight_kernel_opt( const int A_num_rows,
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
                              const int *__restrict assigned_set,
                              int pass)
{
    const int NUM_WARPS = CTA_SIZE / 32;
    // The hash keys stored in shared memory.
    __shared__ KeyType s_keys[NUM_WARPS * SMEM_SIZE];
    // A shared location where threads propose a row of B to load.
    __shared__ int s_b_row_ids[CTA_SIZE];
    // A shared location where threads store a value of B to load.
    __shared__ Value_type s_b_values[CTA_SIZE];
    // The hash values stored in shared memory.
    __shared__ Value_type s_vals[NUM_WARPS * SMEM_SIZE];
    // The coordinates of the thread inside the CTA/warp.
    const int warp_id = utils::warp_id();
    const int lane_id = utils::lane_id();
    // First threads load the row IDs of A needed by the CTA...
    int a = blockIdx.x * NUM_WARPS + warp_id;
    // Create local storage for the set.
    Hash_map<KeyType, Value_type, SMEM_SIZE, 4, WARP_SIZE> map( &s_keys[warp_id * SMEM_SIZE],
            &g_keys[a * gmem_size],
            &s_vals[warp_id * SMEM_SIZE],
            &g_vals[a * gmem_size],
            gmem_size );
    // Loop over rows of A.
    for ( ; a < A_num_rows ; a += gridDim.x*NUM_WARPS )
    {
        int a_row_id = assigned_set[a];

        // Clear the table.
        map.clear();
        // Rebuild C_hat.
        int c_hat_it  = C_hat_start[a_row_id];
        int c_hat_end = c_hat_it + C_hat_size[a_row_id];
        map.load( c_hat_end - c_hat_it, &C_hat[c_hat_it], &C_hat_pos[c_hat_it] );
        // Load A row IDs.
        int a_col_tmp = -1;

        if ( lane_id < 2 )
        {
            a_col_tmp = A_rows[a_row_id + lane_id];
        }

        int a_col_it  = utils::shfl( a_col_tmp, 0 );
        int a_col_end = utils::shfl( a_col_tmp, 1 );

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
#pragma unroll
        for ( int mask = WARP_SIZE / 2 ; mask > 0 ; mask >>= 1 )
        {
            sum_C += utils::shfl_xor( sum_C, mask );
            sum_N += utils::shfl_xor( sum_N, mask );
        }

        if ( lane_id == 0 )
        {
            // NOTE this matches the check for zero in HYPRE
            double div = (fabs(sum_C * diag[a_row_id]) == 0.0) ? 1. : sum_C * diag[a_row_id];
            alfa = -sum_N / div;
            // alfa = -sum_N/(sum_C*diag[a_row_id]);
        }

        alfa = utils::shfl( alfa, 0 );

        int p_col_tmp = -1;

        if ( lane_id < 2 )
        {
            p_col_tmp = P_rows[a_row_id + lane_id];
        }

        int p_col_it  = utils::shfl( p_col_tmp, 0 );
        int p_col_end = utils::shfl( p_col_tmp, 1 );

        map.store_keys_scale_values( p_col_end - p_col_it, &P_cols[p_col_it], alfa, &P_vals[p_col_it] );
    }
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
    const int NUM_WARPS = CTA_SIZE / 32;
    // The hash keys stored in shared memory.
    __shared__ KeyType s_keys[NUM_WARPS * SMEM_SIZE];
    // A shared location where threads propose a row of B to load.
    __shared__ int s_b_row_ids[CTA_SIZE];
    // A shared location where threads store a value of B to load.
    __shared__ Value_type s_b_values[CTA_SIZE];
    // The hash values stored in shared memory.
    __shared__ Value_type s_vals[NUM_WARPS * SMEM_SIZE];
    // The coordinates of the thread inside the CTA/warp.
    const int warp_id = utils::warp_id();
    const int lane_id = utils::lane_id();
    // First threads load the row IDs of A needed by the CTA...
    int a_row_id = blockIdx.x * NUM_WARPS + warp_id;
    // Create local storage for the set.
    Hash_map<KeyType, Value_type, SMEM_SIZE, 4, WARP_SIZE> map( &s_keys[warp_id * SMEM_SIZE],
            &g_keys[a_row_id * gmem_size],
            &s_vals[warp_id * SMEM_SIZE],
            &g_vals[a_row_id * gmem_size],
            gmem_size );

    // Loop over rows of A.
    for ( ; a_row_id < A_num_rows ; a_row_id = get_work( wk_work_queue, warp_id ) )
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
        int a_col_tmp = -1;

        if ( lane_id < 2 )
        {
            a_col_tmp = A_rows[a_row_id + lane_id];
        }

        int a_col_it  = utils::shfl( a_col_tmp, 0 );
        int a_col_end = utils::shfl( a_col_tmp, 1 );

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

            __syncwarp();

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
#pragma unroll
        for ( int mask = WARP_SIZE / 2 ; mask > 0 ; mask >>= 1 )
        {
            sum_C += utils::shfl_xor( sum_C, mask );
            sum_N += utils::shfl_xor( sum_N, mask );
        }

        if ( lane_id == 0 )
        {
            // NOTE this matches the check for zero in HYPRE
            double div = (fabs(sum_C * diag[a_row_id]) == 0.0) ? 1. : sum_C * diag[a_row_id];
            alfa = -sum_N / div;
            // alfa = -sum_N/(sum_C*diag[a_row_id]);
        }

        alfa = utils::shfl( alfa, 0 );

        int p_col_tmp = -1;

        if ( lane_id < 2 )
        {
            p_col_tmp = P_rows[a_row_id + lane_id];
        }

        int p_col_it  = utils::shfl( p_col_tmp, 0 );
        int p_col_end = utils::shfl( p_col_tmp, 1 );

        map.store_keys_scale_values( p_col_end - p_col_it, &P_cols[p_col_it], alfa, &P_vals[p_col_it] );
    }
}


} // namespace multipass

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
{
    this->m_use_opt_kernels = cfg.getParameter<int>("use_opt_kernels", "default");
}

template <AMGX_VecPrecision t_vecPrec, AMGX_MatPrecision t_matPrec, AMGX_IndPrecision t_indPrec>
Multipass_Interpolator<TemplateConfig<AMGX_device, t_vecPrec, t_matPrec, t_indPrec> >::~Multipass_Interpolator()
{}

enum { WARP_SIZE = 32, GRID_SIZE = 1024, SMEM_SIZE = 128 };


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

__global__ void set_assigned_in_pass(int n, int pass, int* assigned, int* assigned_in_pass, int* assigned_offs)
{
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    if(i >= n) return;

    assigned_in_pass[i] = (assigned[i] == pass) ? 1 : 0;

    if(i == 0) 
    {
        assigned_offs[0] = 0;
    }
}

__global__ void assigned_set_fill(int n, int* assigned_in_pass, int* assigned_offs, int* assigned_set)
{
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    if(i >= n) return;

    if(assigned_in_pass[i]) 
    { 
        assigned_set[assigned_offs[i]] = i; 
    } 
}

template <AMGX_VecPrecision V, AMGX_MatPrecision M, AMGX_IndPrecision I>
template <int hash_size, int group_size, class KeyType>
void Multipass_Interpolator<TemplateConfig<AMGX_device, V, M, I>>::compute_c_hat_opt_dispatch(
        const Matrix_d &A,
        const bool *s_con,
        const int *C_hat_start,
        int *C_hat_size,
        KeyType *C_hat,
        int *C_hat_pos,
        int *assigned,
        IntVector &assigned_set,
        int pass )
{
    typedef typename IndPrecisionMap<I>::Type Index_type;

    constexpr int cta_size = 128;
    constexpr int ngroups = cta_size / group_size;

    const int grid_size = assigned_set.size() / ngroups + 1;

    cudaDeviceProp deviceProps = getDeviceProperties();
    size_t max_shmem_size = deviceProps.sharedMemPerMultiprocessor;

    constexpr int shmem_size =
        (sizeof(KeyType)+sizeof(Index_type))*ngroups*hash_size + group_size; 

    if(shmem_size > max_shmem_size) 
    { 
        FatalError("In compute_values_opt the requested hash size is larger than max.\n", 
                AMGX_ERR_NOT_IMPLEMENTED); 
    } 

    cudaFuncSetAttribute(multipass::compute_c_hat_kernel_opt 
            <group_size, cta_size, hash_size, KeyType>, 
            cudaFuncAttributeMaxDynamicSharedMemorySize, shmem_size); 

    multipass::compute_c_hat_kernel_opt
        <group_size, cta_size, hash_size, KeyType> 
        <<<grid_size, cta_size, shmem_size>>>(
            assigned_set.size(),
            A.row_offsets.raw(),
            A.col_indices.raw(),
            s_con,
            C_hat_start,
            C_hat_size,
            C_hat,
            C_hat_pos,
            assigned,
            assigned_set.raw(),
            pass);
}

template <AMGX_VecPrecision t_vecPrec, AMGX_MatPrecision t_matPrec, AMGX_IndPrecision t_indPrec>
void Multipass_Interpolator<TemplateConfig<AMGX_device, t_vecPrec, t_matPrec, t_indPrec> >::generateInterpolationMatrix_1x1(Matrix_d &A,
        IntVector &cf_map,
        BVector &s_con,
        IntVector &scratch,
        Matrix_d &P)
{
    nvtxRange nvr_gim(__func__);

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

    int numBlocksOneRing = std::min( 4096, (int) (size_one_ring + blockSize - 1) / blockSize );
    int numBlocksTwoRing = std::min( 4096, (int) (size_two_ring + blockSize - 1) / blockSize );
    int numBlocks = std::min( 4096, (int) (A.get_num_rows() + blockSize - 1) / blockSize );
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
    int grid_size = std::min( 4096, (int) (A.get_num_rows() + nWarps - 1) / nWarps);
    if(this->m_use_opt_kernels)
    {
        constexpr int group_size = 8;
        int ngroups_per_block = cta_size / group_size;
        int grid_size = A.get_num_rows() / ngroups_per_block + 1;
        initializeAssignedArray_opt<IndexType, cta_size, group_size> <<< grid_size, cta_size>>>(cf_map.raw(), assigned.raw(), A.row_offsets.raw(), A.col_indices.raw(), s_con.raw(), C_hat_start.raw(), Anum_rows);
    }
    else
    {
        initializeAssignedArray<IndexType, cta_size> <<< grid_size, cta_size>>>(cf_map.raw(), assigned.raw(), A.row_offsets.raw(), A.col_indices.raw(), s_con.raw(), C_hat_start.raw(), Anum_rows);
    }
    cudaCheckError();
    // Count the number of passes and fill assigned for pass > 1
    int pass = 2;
    int max_num_passes = 10;
    // Count the number of unassigned nodes by checking which nodes have assigned[i] = -1
    IndexType num_unassigned = thrust_wrapper::count_if<AMGX_device>(assigned.begin(), assigned.end(), is_less_than_zero());
    cudaCheckError();
    IndexType num_strong_fine = thrust_wrapper::count_if<AMGX_device>(cf_map.begin(), cf_map.end(), is_strong_fine());
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

    // Fill the assigned array for the other nodes, going one pass at the time
    while (num_unassigned_max && pass < max_num_passes)
    {
        if(this->m_use_opt_kernels)
        {
            constexpr int group_size = 8;
            int ngroups_per_block = cta_size / group_size;
            int nblocks = A.get_num_rows() / ngroups_per_block + 1;
            fillAssignedArray_opt<IndexType, cta_size, group_size> <<< nblocks, cta_size>>>(assigned.raw(), A.row_offsets.raw(), A.col_indices.raw(), s_con.raw(), Anum_rows, pass);
        }
        else
        {
            fillAssignedArray<IndexType, cta_size> <<< grid_size, cta_size>>>(assigned.raw(), A.row_offsets.raw(), A.col_indices.raw(), s_con.raw(), Anum_rows, pass);
        }
        num_unassigned = thrust_wrapper::count_if<AMGX_device>(assigned.begin(), assigned.begin() + A.get_num_rows(), is_less_than_zero());
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
        int num_owned_coarse_pts = thrust_wrapper::count_if<AMGX_device>(cf_map.begin(), cf_map.begin() + num_owned_fine_pts, is_non_neg());
        cudaCheckError();
        int num_halo_coarse_pts = thrust_wrapper::count_if<AMGX_device>(cf_map.begin() + num_owned_fine_pts, cf_map.end(), is_non_neg());
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
        coarsePoints = (int) thrust_wrapper::count_if<AMGX_device>(cf_map.begin(), cf_map.end(), is_non_neg());
        cudaCheckError();
        const int cta_size = 128;
        const int grid_size = std::min( 4096, (A.get_num_rows() + cta_size - 1) / cta_size);
        fill_cf_map_global <<< grid_size, cta_size>>>(cf_map.raw(), cf_map_global.raw(), A.get_num_rows());
        cudaCheckError();
    }

    // ----------------------------------------------------------
    // Create an upper bound for the length of each row P
    // ----------------------------------------------------------
    Hash_Workspace<TConfig_d, int64_t> exp_wk(true, GRID_SIZE);
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
            if(this->m_use_opt_kernels)
            {
                constexpr int group_size = 8;
                int ngroups_per_block = CTA_SIZE / group_size;
                int nblocks = A.get_num_rows() / ngroups_per_block + 1;
                multipass::estimate_c_hat_size_kernel_opt< CTA_SIZE, group_size> <<< nblocks, CTA_SIZE>>>(
                        A.get_num_rows(),
                        A.row_offsets.raw(),
                        A.col_indices.raw(),
                        s_con.raw(),
                        assigned.raw(),
                        C_hat_start.raw(),
                        i);
            }
            else
            {
                multipass::estimate_c_hat_size_kernel< CTA_SIZE, WARP_SIZE> <<< 2048, CTA_SIZE>>>(
                        A.get_num_rows(),
                        A.row_offsets.raw(),
                        A.col_indices.raw(),
                        s_con.raw(),
                        assigned.raw(),
                        C_hat_start.raw(),
                        i);
            }

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
    thrust_wrapper::exclusive_scan<AMGX_device>( C_hat_start.begin( ), C_hat_start.end( ), C_hat_start.begin( ) );
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
        if(this->m_use_opt_kernels)
        {
            constexpr int group_size = 8;
            int ngroups_per_block = CTA_SIZE / group_size;
            int nblocks = A.get_num_rows() / ngroups_per_block + 1;

            // This will fill C_hat for nodes with assigned = 1 (for fine points strongly connected to coarse points)
            multipass::compute_c_hat_first_pass_kernel_opt< CTA_SIZE, WARP_SIZE, group_size, int64_t> <<< nblocks, CTA_SIZE>>>(
                A.get_num_rows(),
                A.row_offsets.raw(),
                A.col_indices.raw(),
                s_con.raw(),
                cf_map_global.raw(),
                C_hat_start.raw(),
                C_hat_size.raw(),
                C_hat.raw(),
                assigned.raw());

            cudaCheckError();
        }
        else
        {
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
        }

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
            if(false && this->m_use_opt_kernels)
            {
                I64Vector_d C_hat_(C_hat);
                IntVector C_hat_pos_(C_hat_pos);
                IntVector C_hat_size_(C_hat_size);

                IntVector assigned_in_pass(assigned.size());
                IntVector assigned_offs(assigned.size()+1);

                int nthreads = 128;
                int nblocks = assigned.size() / nthreads + 1;
                set_assigned_in_pass<<<nblocks, nthreads>>>(assigned.size(), i, assigned.raw(), assigned_in_pass.raw(), assigned_offs.raw());

                thrust_wrapper::inclusive_scan<AMGX_device>(assigned_in_pass.begin(), assigned_in_pass.end(), assigned_offs.begin()+1);

                int nassigned_in_pass = assigned_offs[assigned_offs.size()-1] ;
                if(nassigned_in_pass > 0)
                {
                    IntVector assigned_set(nassigned_in_pass);
                    assigned_set_fill<<<nblocks, nthreads>>>(assigned.size(), assigned_in_pass.raw(), assigned_offs.raw(), assigned_set.raw());

                    int max_c_hat_size = thrust::reduce(C_hat_size_.begin(), C_hat_size_.end(), -1, thrust::maximum<int>());

                    int hash_size = pow(2, ceil(log2(max_c_hat_size)));

                    switch(hash_size)
                    {
                        case 1:
                        case 2:
                        case 4:
                        case 8:
                        case 16:
                        case 32:
                            compute_c_hat_opt_dispatch<32, 32>(
                                    A, s_con.raw(), C_hat_start.raw(),
                                    C_hat_size_.raw(), C_hat_.raw(), C_hat_pos_.raw(), assigned.raw(),
                                    assigned_set, i);
                            break;
                        case 64:
                            compute_c_hat_opt_dispatch<64, 32>(
                                    A, s_con.raw(), C_hat_start.raw(),
                                    C_hat_size_.raw(), C_hat_.raw(), C_hat_pos_.raw(), assigned.raw(),
                                    assigned_set, i);
                            break;
                        case 128:
                            compute_c_hat_opt_dispatch<128, 32>(
                                    A, s_con.raw(), C_hat_start.raw(),
                                    C_hat_size_.raw(), C_hat_.raw(), C_hat_pos_.raw(), assigned.raw(),
                                    assigned_set, i);
                            break;
                        case 256:
                            compute_c_hat_opt_dispatch<256, 32>(
                                    A, s_con.raw(), C_hat_start.raw(),
                                    C_hat_size_.raw(), C_hat_.raw(), C_hat_pos_.raw(), assigned.raw(),
                                    assigned_set, i);
                            break;
                        case 512:
                            compute_c_hat_opt_dispatch<512, 32>(
                                    A, s_con.raw(), C_hat_start.raw(),
                                    C_hat_size_.raw(), C_hat_.raw(), C_hat_pos_.raw(), assigned.raw(),
                                    assigned_set, i);
                            break;
                        default:
                            FatalError("Size of C_hat in a single row too large, solve with original solvers.", AMGX_ERR_INTERNAL);
                    }
                }

                cudaDeviceSynchronize();
                cudaCheckError();

#if 0
                if (A.is_matrix_distributed())
                {
                    //TODO: We could probably reduce the amount of data exchanged here, since we only need to exchange for recently updated nodes
                    C_hat_.dirtybit = 1;
                    C_hat_matrix.manager->exchange_halo(C_hat_, C_hat_.tag);
                    C_hat_size_.dirtybit = 1;
                    A.manager->exchange_halo(C_hat_size_, C_hat_size_.tag);
                }
#endif
            }
            else
            {
            }

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
    int nonZeros = thrust_wrapper::reduce<AMGX_device>(nonZerosPerRow.begin(), nonZerosPerRow.end());
    cudaCheckError();
    // get the offsets with an exclusive scan
    thrust_wrapper::exclusive_scan<AMGX_device>(nonZerosPerRow.begin(), nonZerosPerRow.end(), nonZeroOffsets.begin());
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
    amgx::thrust::copy(nonZeroOffsets.begin(), nonZeroOffsets.end(), P.row_offsets.begin());
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

    const int CTA_SIZE  = 256;
    const int NUM_WARPS = CTA_SIZE / WARP_SIZE;
    int work_offset = GRID_SIZE * NUM_WARPS;
    typedef typename MatPrecisionMap<t_matPrec>::Type Value_type;

    // Fill P.col_indices_global, P.values
    if(this->m_use_opt_kernels)
    {
        int grid_size = A.get_num_rows() / NUM_WARPS + 1;

        multipass::compute_interp_weight_first_pass_kernel_opt
            <Value_type, CTA_SIZE, SMEM_SIZE, WARP_SIZE, int64_t>
            <<< grid_size, CTA_SIZE>>>(
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
    }
    else
    {
        cudaMemcpy( exp_wk.get_work_queue(), &work_offset, sizeof(int), cudaMemcpyHostToDevice );

        // Compute the set C_hat.
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
    }

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
        if(this->m_use_opt_kernels)
        {
            IntVector assigned_in_pass(assigned.size());
            IntVector assigned_offs(assigned.size() + 1);

            int nthreads = 128;
            int nblocks = assigned.size() / nthreads + 1;
            set_assigned_in_pass<<<nblocks, nthreads>>>(assigned.size(), i, assigned.raw(), assigned_in_pass.raw(), assigned_offs.raw());
            thrust_wrapper::inclusive_scan<AMGX_device>(assigned_in_pass.begin(), assigned_in_pass.end(), assigned_offs.begin()+1);

            IntVector assigned_set(assigned_offs[assigned_offs.size()-1]);
            assigned_set_fill<<<nblocks, nthreads>>>(assigned.size(), assigned_in_pass.raw(), assigned_offs.raw(), assigned_set.raw());

            Hash_Workspace<TConfig_d, int64_t> exp_wk2(true, 4096);
            multipass::compute_interp_weight_kernel_opt<Value_type, CTA_SIZE, SMEM_SIZE, WARP_SIZE, int64_t> <<< 4096, CTA_SIZE>>>(
                    assigned_set.size(),
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
                    exp_wk2.get_gmem_size(),
                    exp_wk2.get_keys(),
                    exp_wk2.get_vals(),
                    exp_wk2.get_work_queue(),
                    assigned.raw(),
                    assigned_set.raw(),
                    i);
        }
        else
        {
            cudaMemcpy( exp_wk.get_work_queue(), &work_offset, sizeof(int), cudaMemcpyHostToDevice );
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
        }

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
        Matrix<T_Config> &P)
{
    P.set_initialized(0);
    ViewType oldView = A.currentView();
    A.setView(OWNED);

    if (A.get_block_size() == 1)
    {
        generateInterpolationMatrix_1x1(A, cf_map, s_con, scratch, P);
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
