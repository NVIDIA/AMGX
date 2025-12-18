// SPDX-FileCopyrightText: 2011 - 2025 NVIDIA CORPORATION. All Rights Reserved.
//
// SPDX-License-Identifier: BSD-3-Clause

#include <thrust/count.h>
#include <aggregation/selectors/parallel_greedy_selector.h>
#include <sm_utils.inl>

namespace amgx
{
namespace aggregation
{

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

__host__ __device__ int hash_function(int a, int seed = 17)
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
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template< typename Value_type, int NUM_THREADS_PER_ROW, int CTA_SIZE, int WARP_SIZE >
__global__
void compute_edge_weights_diag( const int  A_num_rows,
                                const int  block_size,
                                const int  num_owned,
                                const int *__restrict A_rows,
                                const int *__restrict A_cols,
                                const int *__restrict A_diag,
                                const Value_type *__restrict A_vals,
                                Value_type *__restrict A_edge_weights )
{
    const int NUM_WARPS_PER_CTA = CTA_SIZE  / WARP_SIZE;
    // The coordinates of the thread inside the CTA/warp.
    const int warp_id = utils::warp_id();
    const int lane_id = utils::lane_id();
    // Shared memory to broadcast columns.
    __shared__ volatile int s_bcast_col[CTA_SIZE];
    // Shared memory to store columns ji.
    __shared__ volatile Value_type s_bcast_ji[CTA_SIZE];
    // Constants.
    // const int lane_id_div_num_threads = lane_id / NUM_THREADS_PER_ROW;
    // const int lane_id_mod_num_threads = lane_id % NUM_THREADS_PER_ROW;
    // First threads load the row IDs of A needed by the CTA...
    int a_row_id = blockIdx.x * NUM_WARPS_PER_CTA + warp_id;

    // Loop over rows of A.
    for ( ; a_row_id < A_num_rows ; a_row_id += gridDim.x * NUM_WARPS_PER_CTA )
    {
        // The diagonal of (i,i).
        Value_type row_diag = A_vals[block_size * A_diag[a_row_id]];
        // Load the range of the row.
        int a_col_begin = A_rows[a_row_id + 0];
        int a_col_end   = A_rows[a_row_id + 1];

        // Iterate over the columns of A.
        for ( ; a_col_begin < a_col_end ; a_col_begin += WARP_SIZE )
        {
            // Each lane works on a different column.
            const int a_col_it = a_col_begin + lane_id;
            // Is it an active thread.
            int is_active = a_col_it < a_col_end;
            // Active nodes load column ids.
            int a_col_id = -1;

            if ( is_active )
            {
                a_col_id = A_cols[a_col_it];
            }

            s_bcast_col[threadIdx.x] = a_col_id;
            // Reset ji coefficients.
            s_bcast_ji[threadIdx.x] = Value_type(0);

            // Iterate over columns to find ji coefficients if they exist.
            for ( int k = 0, num_cols = __popc(utils::ballot(is_active)) ; k < num_cols ; ++k )
            {
                const int uniform_col_id = s_bcast_col[warp_id * WARP_SIZE + k];

                // Skip columns whose id is greater than num_owned.
                if ( uniform_col_id == -1 || uniform_col_id == a_row_id || uniform_col_id >= num_owned )
                {
                    continue;
                }

                // The bounds of the row.
                int b_row_begin = A_rows[uniform_col_id + 0];
                int b_row_end   = A_rows[uniform_col_id + 1];

                // Iterate over the row.
                for ( int not_found = 1 ; not_found && b_row_begin < b_row_end ; b_row_begin += WARP_SIZE )
                {
                    const int b_row_it = b_row_begin + lane_id;
                    // Load the column id.
                    int b_col_id = -1;

                    if ( b_row_it < b_row_end )
                    {
                        b_col_id = A_cols[b_row_it];
                    }

                    // Has anyone found the column.
                    not_found = utils::all( b_col_id != a_row_id );

                    // If someone found the column, ask it to load the value.
                    if ( b_col_id == a_row_id )
                    {
                        s_bcast_ji[warp_id * WARP_SIZE + k] = A_vals[block_size * b_row_it];
                    }
                }
            }

            // Deactivate "invalid" threads.
            is_active = is_active && a_col_id < num_owned;
            // The diagonal value associated with the column.
            Value_type col_diag(0);

            if ( is_active )
            {
                col_diag = A_vals[block_size * A_diag[a_col_id]];
            }

            // The value of the column.
            Value_type col_val(0);

            if ( is_active )
            {
                col_val = A_vals[block_size * a_col_it];
            }

            // Compute the denominator.
            Value_type den = max( abs(row_diag), abs(col_diag) );
            // Compute the weight of the edge.
            Value_type weight(0);

            if ( den != Value_type(0) )
            {
                weight = Value_type(0.5) * (abs(col_val) + abs(s_bcast_ji[threadIdx.x])) / den;
            }

            if ( is_active )
            {
                A_edge_weights[a_col_it] = weight;
            }
        }
    }
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template< int CTA_SIZE, int WARP_SIZE >
__global__
void compute_ring_leader( const int A_num_rows,
                          const int num_owned,
                          const int *__restrict A_rows,
                          const int *__restrict A_cols,
                          const int *__restrict is_aggregated,
                          const int *__restrict in_leader_id,
                          const int *__restrict in_leader_hash,
                          int *__restrict out_leader_id,
                          int *__restrict out_leader_hash )
{
    const int NUM_WARPS_PER_CTA = CTA_SIZE  / WARP_SIZE;
    // The coordinates of the thread inside the CTA/warp.
    const int warp_id = utils::warp_id();
    const int lane_id = utils::lane_id();
    // First threads load the row IDs of A needed by the CTA...
    int a_row_id = blockIdx.x * NUM_WARPS_PER_CTA + warp_id;

    // Loop over rows of A.
    for ( ; a_row_id < A_num_rows ; a_row_id += gridDim.x * NUM_WARPS_PER_CTA )
    {
        // Skip already aggregated vertices.
        if ( is_aggregated[a_row_id] )
        {
            continue;
        }

        // Load the range of the row.
        int a_col_begin = A_rows[a_row_id + 0];
        int a_col_end   = A_rows[a_row_id + 1];
        // The max hash in my ith ring.
        int my_min_id   = A_num_rows;
        int my_max_hash = INT_MIN;

        if ( in_leader_id )
        {
            my_min_id   = in_leader_id  [a_row_id];
            my_max_hash = in_leader_hash[a_row_id];
        }
        else
        {
            my_min_id   = a_row_id;
            my_max_hash = hash_function(a_row_id);
        }

        // Iterate over the columns of A.
        for ( ; a_col_begin < a_col_end ; a_col_begin += WARP_SIZE )
        {
            // Each lane works on a different column.
            const int a_col_it = a_col_begin + lane_id;
            // Is it an active thread.
            int is_active = a_col_it < a_col_end;
            // Active nodes load column ids.
            int a_col_id = -1;

            if ( is_active )
            {
                a_col_id = A_cols[a_col_it];
            }

            // Compute the hash value if needed.
            int col_min_id   = A_num_rows;
            int col_max_hash = INT_MIN;

            if ( in_leader_id )
            {
                if ( is_active && !is_aggregated[a_col_id] )
                {
                    col_min_id   = in_leader_id  [a_col_id];
                    col_max_hash = in_leader_hash[a_col_id];
                }
            }
            else
            {
                if ( is_active && !is_aggregated[a_col_id] )
                {
                    col_min_id   = a_col_id;
                    col_max_hash = hash_function(a_col_id);
                }
            }

            // Update the max_hash if needed.
            if ( is_active && (my_max_hash < col_max_hash || (my_max_hash == col_max_hash && my_min_id >= col_min_id)) )
            {
                my_min_id   = col_min_id;
                my_max_hash = col_max_hash;
            }
        }

        // Reduce the max hash.
#pragma unroll

        for ( int mask = WARP_SIZE / 2 ; mask > 0 ; mask >>= 1 )
        {
            int other_min_id   = utils::shfl_xor(my_min_id, mask);
            int other_max_hash = utils::shfl_xor(my_max_hash, mask);

            if ( other_max_hash > my_max_hash || (other_max_hash == my_max_hash && other_min_id < my_min_id) )
            {
                my_min_id   = other_min_id;
                my_max_hash = other_max_hash;
            }
        }

        // The 1st thread stores the result.
        if ( lane_id == 0 )
        {
            out_leader_id  [a_row_id] = my_min_id;
            out_leader_hash[a_row_id] = my_max_hash;
        }
    }
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template< typename Value_type, int CTA_SIZE, int WARP_SIZE >
__global__
void build_aggregates( const int num_rings,
                       const int A_num_rows,
                       const int num_owned,
                       const int *__restrict A_rows,
                       const int *__restrict A_cols,
                       int *__restrict is_aggregated,
                       const int *__restrict in_leader_id,
                       const int *__restrict in_leader_hash,
                       const Value_type *__restrict edge_weights,
                       int *__restrict num_aggregates,
                       int *__restrict aggregates,
                       int *__restrict num_unaggregated)
{
    const int NUM_WARPS_PER_CTA = CTA_SIZE  / WARP_SIZE;
    // The coordinates of the thread inside the CTA/warp.
    const int warp_id = utils::warp_id();
    const int lane_id = utils::lane_id();
    // First threads load the row IDs of A needed by the CTA...
    int a_row_id = blockIdx.x * NUM_WARPS_PER_CTA + warp_id;

    // Loop over rows of A.
    for ( ; a_row_id < A_num_rows ; a_row_id += gridDim.x * NUM_WARPS_PER_CTA )
    {
        // Skip already aggregated vertices.
        if ( is_aggregated[a_row_id] )
        {
            continue;
        }

        // The max hash in my ith ring.
        int my_hash = hash_function(a_row_id), my_max_hash = in_leader_hash[a_row_id];

        // Skip if I'm not a local king.
        if ( my_hash < my_max_hash || (my_hash == my_max_hash && a_row_id > in_leader_id[a_row_id]) )
        {
            continue;
        }

        // We start at the row.
        int curr_row = a_row_id;

        // Iterate to form the aggregate.
        for ( int aggregate_size = 1, aggregate_id = -1 ; aggregate_size <= num_rings ; ++aggregate_size )
        {
            // Load the range of the row.
            int a_col_begin = A_rows[curr_row + 0];
            int a_col_end   = A_rows[curr_row + 1];
            // The max weight.
            int max_id = -1;
            float max_weight = -1.0f;
            // The max aggregated weight.
            int max_aggregated_id = -1;
            float max_aggregated_weight = -1.0f;

            // Iterate over the columns of A.
            for ( ; a_col_begin < a_col_end ; a_col_begin += WARP_SIZE )
            {
                // Each lane works on a different column.
                const int a_col_it = a_col_begin + lane_id;
                // Is it an active thread.
                int is_active = a_col_it < a_col_end;
                // Active nodes load column ids.
                int a_col_id = -1;
                float a_col_weight = -1.0f;

                if ( is_active )
                {
                    a_col_id = A_cols[a_col_it];
                    a_col_weight = static_cast<float>(edge_weights[a_col_it]);
                }

                // Is the column aggregated?
                int is_col_aggregated = is_active && is_aggregated[a_col_id];

                // Select the column if it's not aggregated and its weight is better.
                if ( is_active && !is_col_aggregated && a_col_weight > max_weight )
                {
                    max_id     = a_col_id;
                    max_weight = a_col_weight;
                }

                // Update max aggregated weight if needed.
                if ( is_active && is_col_aggregated && a_col_weight > max_aggregated_weight )
                {
                    max_aggregated_id     = a_col_id;
                    max_aggregated_weight = a_col_weight;
                }
            }

            // Is there a valid max_id.
            int valid_max_id = utils::any(max_id != -1, utils::activemask());

            // No valid max id?
            if ( !valid_max_id && aggregate_size > 1 )
            {
                break;
            }

            // Find the max id.
            if ( !valid_max_id )
            {
                max_id     = max_aggregated_id;
                max_weight = max_aggregated_weight;
            }

#pragma unroll
            for ( int mask = WARP_SIZE / 2 ; mask > 0 ; mask >>= 1 )
            {
                int   other_max_id     = utils::shfl_xor(max_id, mask);
                float other_max_weight = utils::shfl_xor(max_weight, mask);

                if ( other_max_weight > max_weight )
                {
                    max_id     = other_max_id;
                    max_weight = other_max_weight;
                }
            }

            // We know it's a singleton so merge with an existing aggregate.
            if ( !valid_max_id )
            {
                if ( lane_id == 0 && max_id != -1 )
                {
                    is_aggregated[curr_row] = 1;
                    aggregates[curr_row] = aggregates[max_id];
                    atomicAdd(num_unaggregated, -1);
                }

                break;
            }

            // It's not a singleton but the 1st vertex in the aggregate.
            if ( lane_id == 0 && aggregate_size == 1 )
            {
                aggregate_id = atomicAdd(num_aggregates, 1);
                is_aggregated[a_row_id] = 1;
                aggregates[a_row_id] = aggregate_id;
                atomicAdd(num_unaggregated, -1);
            }

            // Set the aggregate of the winner.
            if ( lane_id == 0 )
            {
                aggregates[max_id] = aggregate_id;
                is_aggregated[max_id] = 1;
                atomicAdd(num_unaggregated, -1);
            }

            // Set the next row to consider.
            curr_row = max_id;
        }
    }
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

enum { WARP_SIZE = 32 };

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template< AMGX_VecPrecision V, AMGX_MatPrecision M, AMGX_IndPrecision I >
void
ParallelGreedySelector<TemplateConfig<AMGX_device, V, M, I> >::setAggregates_1x1( const MatrixType &A, IVector &aggregates, IVector &aggregates_global, int &num_aggregates )
{}

template< AMGX_VecPrecision V, AMGX_MatPrecision M, AMGX_IndPrecision I >
void
ParallelGreedySelector<TemplateConfig<AMGX_device, V, M, I> >::setAggregates_common_sqblocks( const MatrixType &A, IVector &aggregates, IVector &aggregates_global, int &num_aggregates )
{
    const int num_rings = 4;
    // The number of rows of A.
    const int num_rows = A.get_num_rows();
    // The size of the block.
    const int block_size = A.get_block_dimx() * A.get_block_dimy();
    // The number of threads per CTA.
    const int CTA_SIZE = 128;
    // The number of warps per CTA.
    const int NUM_WARPS_PER_CTA = CTA_SIZE / WARP_SIZE;
    // The number of CTAs in a grid where each thread is independent.
    const int THREAD_GRID_SIZE = (A.get_num_rows() + CTA_SIZE - 1) / CTA_SIZE;
    // The number of CTAs in a grid where each warp proceeds a row.
    const int WARP_GRID_SIZE = (A.get_num_rows() + NUM_WARPS_PER_CTA - 1) / NUM_WARPS_PER_CTA;
    // Edge weights.
    device_vector_alloc<ValueType> edge_weights(A.get_num_nz());
    // Compute edge weights.
    compute_edge_weights_diag<ValueType, 32, CTA_SIZE, WARP_SIZE> <<< WARP_GRID_SIZE, CTA_SIZE>>>(
        num_rows,
        block_size,
        num_rows, // It should be num_owned!!!
        A.row_offsets.raw(),
        A.col_indices.raw(),
        A.diag.raw(),
        A.values.raw(),
        amgx::thrust::raw_pointer_cast( &edge_weights.front() ) );
    cudaCheckError();
    // Make sure there's enough room to store aggregates.
    aggregates.resize(num_rows);
    // The number of aggregates.
    device_vector_alloc<int> dev_num_aggregates(1, 0), dev_num_unaggregated(1, num_rows);
    // Is a vertex already aggregated.
    device_vector_alloc<int> is_aggregated(num_rows, 0);
    // Hash values.
    device_vector_alloc<int> ring_leader_id0_array(num_rows);
    device_vector_alloc<int> ring_leader_id1_array(num_rows);
    device_vector_alloc<int> ring_leader_hash0_array(num_rows);
    device_vector_alloc<int> ring_leader_hash1_array(num_rows);
    // Is there any unaggregated vertex?
    int num_unaggregated = num_rows;

    // Iterate until all vertices are aggregated.
    while ( num_unaggregated > 0 )
    {
        int *ring_leader_id0 = amgx::thrust::raw_pointer_cast(&ring_leader_id0_array.front());
        int *ring_leader_id1 = amgx::thrust::raw_pointer_cast(&ring_leader_id1_array.front());
        int *ring_leader_hash0 = amgx::thrust::raw_pointer_cast(&ring_leader_hash0_array.front());
        int *ring_leader_hash1 = amgx::thrust::raw_pointer_cast(&ring_leader_hash1_array.front());

        // Count N-ring of vertices.
        for ( int i = 0 ; i < 2 * num_rings ; ++i )
        {
            compute_ring_leader<CTA_SIZE, WARP_SIZE> <<< WARP_GRID_SIZE, CTA_SIZE>>>(
                num_rows,
                num_rows, // It should be num_owned!!!
                A.row_offsets.raw(),
                A.col_indices.raw(),
                amgx::thrust::raw_pointer_cast(&is_aggregated.front()),
                i == 0 ? NULL : ring_leader_id0,
                i == 0 ? NULL : ring_leader_hash0,
                ring_leader_id1,
                ring_leader_hash1);
            cudaCheckError();
            std::swap(ring_leader_id0, ring_leader_id1);
            std::swap(ring_leader_hash0, ring_leader_hash1);
        }

        // Perform the assignments to aggregates.
        build_aggregates<ValueType, CTA_SIZE, WARP_SIZE> <<< WARP_GRID_SIZE, CTA_SIZE>>>(
            num_rings,
            num_rows,
            num_rows, // It should be num_owned!!!
            A.row_offsets.raw(),
            A.col_indices.raw(),
            amgx::thrust::raw_pointer_cast(&is_aggregated.front()),
            ring_leader_id0,
            ring_leader_hash0,
            amgx::thrust::raw_pointer_cast(&edge_weights.front()),
            amgx::thrust::raw_pointer_cast(&dev_num_aggregates.front()),
            aggregates.raw(),
            amgx::thrust::raw_pointer_cast(&dev_num_unaggregated.front()));
        cudaCheckError();
        // Number of aggregated vertices.
        num_unaggregated = dev_num_unaggregated[0];
    }

    // The number of aggregates.
    num_aggregates = dev_num_aggregates[0];
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template< AMGX_VecPrecision V, AMGX_MatPrecision M, AMGX_IndPrecision I >
void
ParallelGreedySelector<TemplateConfig<AMGX_host, V, M, I> >::setAggregates_1x1( const MatrixType &A, IVector &aggregates, IVector &aggregates_global, int &num_aggregates )
{}

template< AMGX_VecPrecision V, AMGX_MatPrecision M, AMGX_IndPrecision I >
void
ParallelGreedySelector<TemplateConfig<AMGX_host, V, M, I> >::setAggregates_common_sqblocks( const MatrixType &A, IVector &aggregates, IVector &aggregates_global, int &num_aggregates )
{}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template< class T_Config >
ParallelGreedySelector_Base<T_Config>::ParallelGreedySelector_Base( AMG_Config &cfg, const std::string &cfg_scope ) :
    Selector<T_Config>()
{
}

template< class T_Config >
void ParallelGreedySelector_Base<T_Config>::setAggregates( MatrixType &A, IVector &aggregates, IVector &aggregates_global, int &num_aggregates )
{
    setAggregates_common_sqblocks( A, aggregates, aggregates_global, num_aggregates );
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#define AMGX_CASE_LINE(CASE) template class ParallelGreedySelector_Base<TemplateMode<CASE>::Type>;
AMGX_FORALL_BUILDS(AMGX_CASE_LINE)
#undef AMGX_CASE_LINE

#define AMGX_CASE_LINE(CASE) template class ParallelGreedySelector<TemplateMode<CASE>::Type>;
AMGX_FORALL_BUILDS(AMGX_CASE_LINE)
#undef AMGX_CASE_LINE

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace aggregation
} // namespace amgx
