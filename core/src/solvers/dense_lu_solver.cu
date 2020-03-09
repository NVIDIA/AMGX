/* Copyright (c) 2013-2017, NVIDIA CORPORATION. All rights reserved.
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
#include <basic_types.h>
#include <util.h>
#include <ld_functions.h>
#include <logger.h>
#include <thrust/logical.h>
#include <profile.h>
#include <sm_utils.inl>
#include <texture.h>
#include <typeinfo>

#include "solvers/dense_lu_solver.h"
#include "solvers/block_common_solver.h"

#include "amgx_types/util.h"

#include <algorithm>

//trick to get nvcc to print a defined value
//(__CUDA_ARCH__) @ compile time:
//
namespace //unnamed namespace
{
struct X
{
    private:
        X(void) {}
};

template<int N>
struct Print_N_atCompile
{
    X array[N];
};
}//end unnamed namespace

namespace amgx
{
namespace dense_lu_solver
{

enum { WARP_SIZE = 32, CTA_SIZE = 128 };

//
// supporting kernels
//
template< typename T,  typename Tv, int WARP_SIZE >
__global__
void csr_to_dense_kernel(
    const int num_rows,
    const int num_cols,
    const int block_num_rows,
    const int block_num_cols,
    const int *__restrict A_csr_rows,
    const int *__restrict A_csr_cols,
    const T   *__restrict A_csr_vals,
    const int *__restrict A_csr_diag,
    T   *__restrict A_dense,
    const int lda)
{
    // Note:
    // To handle block CSR, the sparsity pattern csr_rows and csr_cols only store
    // the typical csr info assuming each block is a scalar.
    // The values in csr_vals has all entries in the blocks, using row major to
    // store the block. So we need the number of entries in each block as stride.
    const int block_mxn = block_num_rows * block_num_cols;
    // Each lane copies one entry in a block and iterate through row sparsity pattern.
    // Essentially one warp per row-block. For 4x4, we have 16 working threads per warp.
    const int lane_id = threadIdx.x % WARP_SIZE;
    // find the (row,col) local to a block
    const int block_row = lane_id / block_num_cols;
    const int block_col = lane_id % block_num_cols;

    // These are wasted threads per warp
    if ( block_row >= block_num_rows ) { return; }

    // The first row to consider. One row per warp.
    int row = (blockIdx.x * blockDim.x + threadIdx.x) / WARP_SIZE;
    const int row_offset = blockDim.x * gridDim.x / WARP_SIZE;

    for ( ; row < num_rows ; row += row_offset )
    {
        int dense_row = row * block_num_rows + block_row;
        // Iterate over each row and copy the elements into col-major A_dense
        int row_end = A_csr_rows[row + 1];

        for (int row_it = A_csr_rows[row]; row_it < row_end ; ++row_it )
        {
            int col = A_csr_cols[row_it];

            if ( col >= num_rows ) { continue; } // Skip entries corresponding to halo

            int dense_col = col * block_num_cols + block_col;
            A_dense[dense_col * lda + dense_row] = A_csr_vals[block_mxn * row_it + lane_id];
        }

        // copy diagonal block
        if ( A_csr_diag )
        {
            int diag_it = A_csr_diag[row];
            int dense_col = row * block_num_cols + block_col; // diag means row=col
            A_dense[dense_col * lda + dense_row] = A_csr_vals[block_mxn * diag_it + lane_id];
        }
    }
}


template< int N, bool ROW_MAJOR, int WARP_SIZE, typename Value_type >
static __device__ __forceinline__
Value_type reduce_distributed_vectors( Value_type x, int is_leader, unsigned int active_mask )
{
    if ( N & (N - 1) )
    {
#pragma unroll

        for ( int i = 1 ; i < N ; ++i )
        {
            Value_type other_x = utils::shfl_down( x, ROW_MAJOR ? i : N * i, WARP_SIZE, active_mask );

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
            x += utils::shfl_xor( x, ROW_MAJOR ? i : N * i, WARP_SIZE, active_mask );
        }
    }

    return x;
}



template< typename Matrix_type, typename Vector_type, int N, int CTA_SIZE,
          int WARP_SIZE, bool ROW_MAJOR, bool HAS_EXTERNAL_DIAG >
__global__
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 700
__launch_bounds__( CTA_SIZE, 12 )
#elif defined(__CUDA_ARCH__)
__launch_bounds__( CTA_SIZE, 12 )
#endif
void b_minus_A_halo_x( const int *__restrict A_rows,
                       const int *__restrict A_cols,
                       const Matrix_type *__restrict A_vals,
                       const int *__restrict A_diag,
                       const Vector_type *x,
                       const Vector_type *b,
                       Vector_type *new_rhs,
                       const int num_owned_rows)
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
    int a_row_it = num_owned_rows;

    if ( is_active )
    {
        a_row_it = blockIdx.x * NUM_ITEMS_PER_CTA + warp_id * NUM_ITEMS_PER_WARP + lane_id_div_NxN;
    }

    unsigned int active_mask = utils::ballot(is_active);

    // Iterate over the rows of the matrix. One warp per row.
    for ( ; a_row_it < num_owned_rows; a_row_it += NUM_ITEMS_PER_GRID )
    {
        int a_row_id = a_row_it;
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

            // Determine if the column is halo column
            int a_col_is_valid = (a_col_id != -1) && (a_col_id >= num_owned_rows);
            // Count the number of active columns.
            // int vote =  __ballot(aColId != -1);
            // The number of iterations.
            // int nCols = max( __popc( vote & 0x0000ffff ), __popc( vote & 0xffff0000 ) );

            // Loop over columns. We compute 8 columns per iteration.
            for ( int k = 0 ; k < NxN ; k += N )
            {
                int my_k = k + lane_id_mod_NxN_div_N;
                // Load N blocks of X.
                int uniform_a_col_id = utils::shfl( a_col_id, shfl_offset + my_k, WARP_SIZE, active_mask);
                int uniform_a_col_is_valid = utils::shfl( a_col_is_valid, shfl_offset + my_k, WARP_SIZE, active_mask );
                Vector_type my_x(0);

                if ( uniform_a_col_id != -1 && uniform_a_col_is_valid)
                {
                    my_x = __cachingLoad(&x[N * uniform_a_col_id + lane_id_mod_NxN_mod_N]);
                    //printf("loading entry %d, num_rows = %d, my_x = %f\n",uniform_a_col_id,num_owned_rows,my_x);
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

        Vector_type my_Einv = (lane_id_mod_NxN == 0 || lane_id_mod_NxN == 5 || lane_id_mod_NxN == 10 || lane_id_mod_NxN == 15) ? 1. : 0.;
        // Reduce bmAx terms.
        int is_leader = lane_id_mod_NxN_div_N == 0;

        if ( ROW_MAJOR )
        {
            is_leader = lane_id_mod_NxN_mod_N == 0;
        }

        my_bmAx = reduce_distributed_vectors<N, ROW_MAJOR, WARP_SIZE>( my_bmAx, is_leader, active_mask );

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
        my_bmAx = reduce_distributed_vectors<N, ROW_MAJOR, WARP_SIZE>( my_bmAx, is_leader, active_mask );

        // Store the results.
        if ( ROW_MAJOR )
        {
            if ( lane_id_mod_NxN_mod_N == 0 )
            {
                new_rhs[N * a_row_id + lane_id_mod_NxN_div_N] = my_bmAx;
            }
        }
        else
        {
            if ( lane_id_mod_NxN_div_N == 0 )
            {
                new_rhs[N * a_row_id + lane_id_mod_NxN_mod_N] = my_bmAx;
            }
        }
    }
}


////////////////////////////////////////////////////////////////////////////////////////////////////////////



template< typename Vector_type, typename Matrix_type, int N>
void distributed_rhs_mod_dispatch( const int *__restrict A_rows,
                                   const int *__restrict A_cols,
                                   const Matrix_type *__restrict A_vals,
                                   const int *__restrict A_diag,
                                   const Vector_type *x,
                                   const Vector_type *b,
                                   Vector_type *new_rhs,
                                   const int num_owned_rows,
                                   const int row_major,
                                   const int has_external_diag
                                 )
{
    const int NUM_WARPS_PER_CTA = CTA_SIZE / WARP_SIZE;
    // Squared N.
    const int NxN = N * N;
    // Number of items per warp.
    const int NUM_ROWS_PER_WARP = WARP_SIZE / NxN;
    // Number of items computer per CTA.
    const int NUM_ROWS_PER_CTA = NUM_ROWS_PER_WARP * NUM_WARPS_PER_CTA;
    // The number of threads to launch.
    const int grid_size = std::min( 4096, (num_owned_rows + NUM_ROWS_PER_CTA - 1) / NUM_ROWS_PER_CTA );
    // Branch to the correct kernel call.
    int code = 2 * (row_major ? 1 : 0) + (has_external_diag ? 1 : 0);

    switch ( code )
    {
        case 0: // Column-major, no external diagonal.
            b_minus_A_halo_x<Matrix_type, Vector_type, N, CTA_SIZE, WARP_SIZE, false, false> <<< grid_size, CTA_SIZE>>>(
                A_rows,
                A_cols,
                A_vals,
                A_diag,
                x,
                b,
                new_rhs,
                num_owned_rows
            );
            break;

        case 1: // Column-major, external diagonal.
            b_minus_A_halo_x<Matrix_type, Vector_type, N, CTA_SIZE, WARP_SIZE, false, true> <<< grid_size, CTA_SIZE>>>(
                A_rows,
                A_cols,
                A_vals,
                A_diag,
                x,
                b,
                new_rhs,
                num_owned_rows
            );
            break;

        case 2: // Row-major, no external diagonal.
            b_minus_A_halo_x<Matrix_type, Vector_type, N, CTA_SIZE, WARP_SIZE, true, false> <<< grid_size, CTA_SIZE>>>(
                A_rows,
                A_cols,
                A_vals,
                A_diag,
                x,
                b,
                new_rhs,
                num_owned_rows
            );
            break;

        case 3: // Row-major, external diagonal.
            b_minus_A_halo_x<Matrix_type, Vector_type, N, CTA_SIZE, WARP_SIZE, true, true> <<< grid_size, CTA_SIZE>>>(
                A_rows,
                A_cols,
                A_vals,
                A_diag,
                x,
                b,
                new_rhs,
                num_owned_rows
            );
            break;

        default:
            FatalError( "Internal error", AMGX_ERR_NOT_IMPLEMENTED );
    }

    cudaCheckError();
}


template< AMGX_VecPrecision V, AMGX_MatPrecision M, AMGX_IndPrecision I>
void DenseLUSolver<TemplateConfig<AMGX_device, V, M, I> >::
distributed_rhs_mod(
    const Vector_d &x,
    const Vector_d &b,
    Vector_d &new_rhs)
{
    const Matrix_d *A = dynamic_cast<Matrix_d *>(Base::m_A);

    switch (A->get_block_dimy())
    {
        case 1:
            distributed_rhs_mod_dispatch<Vector_data, Matrix_data, 1>(A->row_offsets.raw(), A->col_indices.raw(), A->values.raw(), A->diag.raw(), x.raw(), b.raw(), new_rhs.raw(), A->get_num_rows(), A->getBlockFormat() == ROW_MAJOR, A->hasProps(DIAG));
            break;

        case 2:
            distributed_rhs_mod_dispatch<Vector_data, Matrix_data, 2>(A->row_offsets.raw(), A->col_indices.raw(), A->values.raw(), A->diag.raw(), x.raw(), b.raw(), new_rhs.raw(), A->get_num_rows(), A->getBlockFormat() == ROW_MAJOR, A->hasProps(DIAG));
            break;

        case 3:
            distributed_rhs_mod_dispatch<Vector_data, Matrix_data, 3>(A->row_offsets.raw(), A->col_indices.raw(), A->values.raw(), A->diag.raw(), x.raw(), b.raw(), new_rhs.raw(), A->get_num_rows(), A->getBlockFormat() == ROW_MAJOR, A->hasProps(DIAG));
            break;

        case 4:
            distributed_rhs_mod_dispatch<Vector_data, Matrix_data, 4>(A->row_offsets.raw(), A->col_indices.raw(), A->values.raw(), A->diag.raw(), x.raw(), b.raw(), new_rhs.raw(), A->get_num_rows(), A->getBlockFormat() == ROW_MAJOR, A->hasProps(DIAG));
            break;

        case 5:
            distributed_rhs_mod_dispatch<Vector_data, Matrix_data, 5>(A->row_offsets.raw(), A->col_indices.raw(), A->values.raw(), A->diag.raw(), x.raw(), b.raw(), new_rhs.raw(), A->get_num_rows(), A->getBlockFormat() == ROW_MAJOR, A->hasProps(DIAG));
            break;
    }
}

// copy non zero elements only
template< AMGX_VecPrecision V, AMGX_MatPrecision M, AMGX_IndPrecision I >
void DenseLUSolver<TemplateConfig<AMGX_device, V, M, I> >::
csr_to_dense(
)
{
    const Matrix_d *A = dynamic_cast<Matrix_d *>(Base::m_A);
    const int block_size = 256;
    const int num_warps  = block_size / WARP_SIZE;
    const int grid_size = std::min(4096, (A->get_num_rows() + num_warps - 1) / num_warps);
    cudaStream_t stream = thrust::global_thread_handle::get_stream();
    csr_to_dense_kernel<Matrix_data, Vector_data, WARP_SIZE> <<< grid_size, block_size, 0, stream>>>(
        A->get_num_rows(),
        A->get_num_cols(),
        A->get_block_dimx(),
        A->get_block_dimy(),
        A->row_offsets.raw(),
        A->col_indices.raw(),
        A->values.raw(),
        A->hasProps(DIAG) ? A->diag.raw() : NULL,
        m_dense_A,
        m_lda
    );
    cudaCheckError();
}



template< AMGX_VecPrecision V, AMGX_MatPrecision M, AMGX_IndPrecision I >
void DenseLUSolver<TemplateConfig<AMGX_device, V, M, I> >::cudense_getrf()
{
    int wsize = 0;
    cusolverStatus_t status1 = cusolverDnXgetrf_bufferSize(m_cuds_handle,
                               m_num_rows,
                               m_num_cols,
                               m_dense_A,
                               m_lda,
                               &wsize);

    if ( status1 != CUSOLVER_STATUS_SUCCESS)
    {
        FatalError( "Failed kernel in DenseLU", AMGX_ERR_INTERNAL);
    }

    cudaCheckError();
    allocMem(m_trf_wspace, wsize, false);
    status1 = cusolverDnXgetrf(m_cuds_handle, m_num_rows, m_num_cols,
                               m_dense_A, m_lda, m_trf_wspace, m_ipiv, m_cuds_info);

    if ( status1 != CUSOLVER_STATUS_SUCCESS)
    {
        FatalError( "failed kernel in Dense LU is likely due to invalid input parameters",
                    AMGX_ERR_INTERNAL);
    }
    else
    {
        int t_info;
        cudaMemcpy(&t_info, m_cuds_info, sizeof(int), cudaMemcpyDefault);

        if (t_info != 0)
        {
            FatalError( "Fail to get info from cudense", AMGX_ERR_INTERNAL);
        }
        else
        {
            // We follow the standard established by Lapack and used in cudense.
            if (t_info > 0)
            {
                FatalError( "Dense LU factorization failed due to a singular matrix", AMGX_ERR_INTERNAL);
            }
            else if (t_info < 0)
            {
                FatalError( "Invalid input parameter(s) to dense LU", AMGX_ERR_INTERNAL);
            }
        }
    }

    cudaCheckError();

    if (m_trf_wspace)
    {
        thrust::global_thread_handle::cudaFreeAsync(m_trf_wspace);
        m_trf_wspace = 0;
    }
}


template< AMGX_VecPrecision V, AMGX_MatPrecision M, AMGX_IndPrecision I >
void DenseLUSolver<TemplateConfig<AMGX_device, V, M, I> >::cudense_getrs( Vector_d &x )
{
    //Solve L*X = RHS
    cusolverStatus_t status = cusolverDnXgetrs(m_cuds_handle,
                              CUBLAS_OP_N,
                              m_num_rows,
                              1,
                              m_dense_A,
                              m_lda,
                              m_ipiv,
                              (Matrix_data *)(x.raw()),
                              m_num_rows,
                              m_cuds_info);

    if (status != CUSOLVER_STATUS_SUCCESS)
    {
        FatalError( "cuSolver trsv failed to solve Lx=rhs", AMGX_ERR_INTERNAL);
    }

    cudaCheckError();
}


template< AMGX_VecPrecision V, AMGX_MatPrecision M, AMGX_IndPrecision I >
template< class DataType, class IndexType >
void
DenseLUSolver<TemplateConfig<AMGX_device, V, M, I> >::
allocMem(DataType *&ptr,
         IndexType numEntry,
         bool initToZero)
{
    if ( ptr != NULL ) { thrust::global_thread_handle::cudaFreeAsync(ptr); }

    cudaCheckError();
    size_t sz = numEntry * sizeof(DataType);
    thrust::global_thread_handle::cudaMalloc((void **)&ptr, sz);
    cudaCheckError();

    if (initToZero)
    {
        cudaMemset(ptr, 0x0, sz);
        cudaCheckError();
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////


template< AMGX_VecPrecision V, AMGX_MatPrecision M, AMGX_IndPrecision I >
DenseLUSolver<TemplateConfig<AMGX_device, V, M, I> >::
DenseLUSolver(AMG_Config &cfg,
              const std::string &cfg_scope,
              ThreadManager *tmng)
    : Solver<Config_d>(cfg, cfg_scope, tmng),
      m_cuds_handle(0),
      m_num_rows(0),
      m_num_cols(0),
      m_lda(0),
      m_dense_A(0),
      m_ipiv(0),
      m_cuds_info(0),
      m_trf_wspace(0)
{
    // Allocate a handle for cudense
    cusolverStatus_t status = cusolverDnCreate(&m_cuds_handle);

    if ( status != CUSOLVER_STATUS_SUCCESS )
    {
        FatalError( "Could not create the CUDENSE handle", AMGX_ERR_CUDA_FAILURE );
    }

    // Allocate a handle for cublas
    cublasStatus_t cublasStatus = cublasCreate(&m_cublas_handle);

    if ( cublasStatus != CUBLAS_STATUS_SUCCESS )
    {
        FatalError( "Could not create the CUBLAS handle", AMGX_ERR_CUDA_FAILURE );
    }

    // Define the cudense stream.
    status = cusolverDnSetStream(m_cuds_handle, thrust::global_thread_handle::get_stream());

    if ( status != CUSOLVER_STATUS_SUCCESS )
    {
        FatalError( "Could not set the stream for CUDENSE", AMGX_ERR_CUDA_FAILURE );
    }

    // Make sure we don't run more than 1 iteration.
    this->set_max_iters(1);
    allocMem(m_cuds_info, sizeof(int), false);
}


template< AMGX_VecPrecision V, AMGX_MatPrecision M, AMGX_IndPrecision I >
DenseLUSolver<TemplateConfig<AMGX_device, V, M, I> >::~DenseLUSolver()
{
    if (m_cuds_handle) { cusolverDnDestroy(m_cuds_handle); }

    if (m_cublas_handle) { cublasDestroy(m_cublas_handle); }

    if (m_dense_A)
    {
        thrust::global_thread_handle::cudaFreeAsync(m_dense_A);
    }

    if (m_ipiv)
    {
        thrust::global_thread_handle::cudaFreeAsync(m_ipiv);
    }

    if (m_trf_wspace)
    {
        thrust::global_thread_handle::cudaFreeAsync(m_trf_wspace);
    }

    if (m_cuds_info)
    {
        thrust::global_thread_handle::cudaFreeAsync(m_cuds_info);
    }

    cudaCheckError();
}


template< AMGX_VecPrecision V, AMGX_MatPrecision M, AMGX_IndPrecision I >
void DenseLUSolver<TemplateConfig<AMGX_device, V, M, I> >::solve_init(Vector_d &, Vector_d &, bool)
{}


template< AMGX_VecPrecision V, AMGX_MatPrecision M, AMGX_IndPrecision I >
void DenseLUSolver<TemplateConfig<AMGX_device, V, M, I> >::solve_finalize(Vector_d &, Vector_d &)
{}


template< AMGX_VecPrecision V, AMGX_MatPrecision M, AMGX_IndPrecision I >
void DenseLUSolver<TemplateConfig<AMGX_device, V, M, I> >::
solver_setup(bool reuse_matrix_structure)
{
    // This is probably not much.
    Matrix_d *A = dynamic_cast<Matrix_d *>(Base::m_A);

    if (!A)
    {
        FatalError("DenseLUSolver only works with explicit matrices",
                   AMGX_ERR_INTERNAL);
    }

    ViewType oldView = A->currentView();
    A->setViewExterior();
    m_num_rows = A->get_num_rows() * A->get_block_dimx();
    // don't use A->get_num_cols() because A is rectangular.
    // Only the diagonal block owned by this rank is factored.
    m_num_cols = A->get_num_rows() * A->get_block_dimy();
    m_lda      = m_num_rows; // col-major
    // Allocate mem for cudense pivoting sequence.
    allocMem(m_ipiv, m_num_rows, false);
    // Allocate memory to store the dense A and initialize to zero.
    allocMem(m_dense_A, m_num_cols * m_lda, true);
    csr_to_dense(); // copy sparse A to dense_A
    cudense_getrf(); // do LU factor
    A->setView(oldView);
}

// There is one subtle point here:
// We only do LU on the diagonal blocks associated with each rank.
// Halo is used to update the right-hand-side (RHS) vector.
// For multi GPU cases, this is essentially block Jacobi. Since the block size
// is the size of each partition, this is OK trade-off between accuracy and runtime.
template< AMGX_VecPrecision V, AMGX_MatPrecision M, AMGX_IndPrecision I >
bool DenseLUSolver<TemplateConfig<AMGX_device, V, M, I> >::
solve_iteration(Vector_d &rhs,
                Vector_d &x,
                bool xIsZero)
{
    Matrix_d *A = dynamic_cast<Matrix_d *>(Base::m_A);
    ViewType oldView = A->currentView();
    A->setViewExterior();

    if ((!A->is_matrix_singleGPU()) && (!xIsZero))
    {
        // Modify rhs to include contribution from halo nodes
        // i.e. new_rhs = b - A_halo*x;
        // Note: dense_lu solver doesn't support latency hiding
        A->manager->exchange_halo_async(x, x.tag);
        A->manager->exchange_halo_wait(x, x.tag);
        Vector_d new_rhs(rhs.size());
        distributed_rhs_mod(x, rhs, new_rhs);
        thrust::copy(new_rhs.begin(), new_rhs.begin() + m_num_rows, x.begin());
        cudaCheckError();
    }
    else
    {
        x.copy(rhs);
    }

    cudense_getrs(x); // triangular solves
    //Speculative send of x vector
    x.dirtybit = 1;
    A->setView(oldView);
    return true; // direct solver always converges
}


#define AMGX_CASE_LINE(CASE) template class DenseLUSolver<TemplateMode<CASE>::Type>;
AMGX_FORALL_BUILDS(AMGX_CASE_LINE)
//  AMGX_FORCOMPLEX_BUILDS(AMGX_CASE_LINE)
#undef AMGX_CASE_LINE

} // namespace dense_lu
} // namespace amgx

