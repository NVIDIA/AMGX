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

#include <cublas_v2.h>
#include "unit_test.h"
#include "matrix.h"
#include "blas.h"
#include "multiply.h"
#include "solvers/dense_lu_solver.h"

namespace amgx
{

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template< typename T >
__global__ void split_l_and_u(int n, const T *lu, int lda, T *l, T *u )
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i >= n || j >= n)
    {
        return;
    }

    T l_ij = i == j ? T(1) : T(0), u_ij = T(0);
    T lu_ij = lu[i * lda + j];

    if (i <= j)
    {
        u_ij = lu_ij;
    }
    else
    {
        l_ij = lu_ij;
    }

    l[i * lda + j] = l_ij;
    u[i * lda + j] = u_ij;
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

DECLARE_UNITTEST_BEGIN(DenseLUSolverTest_Base);

std::string base_keywords()
{
    return "dense_lu";
}

template< typename Matrix >
void make_identity(Matrix &A)
{
    typedef typename Matrix::TConfig Matrix_config;
    typedef typename Matrix_config::template setMemSpace<AMGX_host>::Type Config_h;
    typedef typename Config_h::template setVecPrec<AMGX_vecInt>::Type IVector_config_h;
    typedef Vector<Config_h> FVector_h;
    typedef Vector<IVector_config_h> IVector_h;
    const int num_rows = A.get_num_rows();
    IVector_h row_offsets(num_rows + 1), col_indices(num_rows);

    for ( int i = 0 ; i < num_rows ; ++i )
    {
        row_offsets[i] = i;
        col_indices[i] = i;
    }

    row_offsets.back() = num_rows;
    FVector_h values(num_rows, 1.0);
    A.row_offsets.copy(row_offsets);
    A.col_indices.copy(col_indices);
    A.values.     copy(values);
}

template< typename Matrix_h, typename Matrix_data >
void csr_to_dense(const Matrix_h &A_h, Matrix_data *dense_A_h, int lda)
{
    for ( int i = 0 ; i < A_h.get_num_rows() ; ++i )
        for ( int j = A_h.row_offsets[i] ; j < A_h.row_offsets[i + 1] ; ++j )
        {
            dense_A_h[i * lda + A_h.col_indices[j]] = A_h.values[j];
        }
}

template< typename Matrix_data >
void l_times_u(int n, Matrix_data *lu_d, int lda)
{
    Matrix_data *l_d, *u_d;
    cudaMallocAsync((void **) &l_d, n * lda * sizeof(Matrix_data), 0);
    UNITTEST_ASSERT_EQUAL(cudaGetLastError(), cudaSuccess);
    UNITTEST_ASSERT_EQUAL(cudaStreamSynchronize(0), cudaSuccess);
    cudaMallocAsync((void **) &u_d, n * lda * sizeof(Matrix_data), 0);
    UNITTEST_ASSERT_EQUAL(cudaGetLastError(), cudaSuccess);
    UNITTEST_ASSERT_EQUAL(cudaStreamSynchronize(0), cudaSuccess);
    // Split LU.
    dim3 block_dim(16, 16);
    dim3 grid_dim((n + block_dim.x - 1) / block_dim.x, (n + block_dim.y - 1) / block_dim.y);
    split_l_and_u <<< grid_dim, block_dim>>>(n, lu_d, lda, l_d, u_d);
    cudaError_t status = cudaDeviceSynchronize();
    UNITTEST_ASSERT_EQUAL(status, cudaSuccess);
    // LxU = LU.
    Matrix_data one(1), zero(0);
    cublasHandle_t cublas_handle;
    cublasStatus_t cublas_status;
    cublas_status = cublasCreate(&cublas_handle);
    UNITTEST_ASSERT_EQUAL(cublas_status, CUBLAS_STATUS_SUCCESS);
    cublas_status = cublasGemm(cublas_handle,
                               CUBLAS_OP_N,
                               CUBLAS_OP_N,
                               n, n, n,
                               &one, l_d, lda,
                               u_d, lda,
                               &zero, lu_d, lda);
    UNITTEST_ASSERT_EQUAL(cublas_status, CUBLAS_STATUS_SUCCESS);
    cublas_status = cublasDestroy(cublas_handle);
    UNITTEST_ASSERT_EQUAL(cublas_status, CUBLAS_STATUS_SUCCESS);
    cudaFreeAsync(l_d, 0);
    UNITTEST_ASSERT_EQUAL(cudaGetLastError(), cudaSuccess);
    UNITTEST_ASSERT_EQUAL(cudaStreamSynchronize(0), cudaSuccess);
    cudaFreeAsync(u_d, 0);
    UNITTEST_ASSERT_EQUAL(cudaGetLastError(), cudaSuccess);
    UNITTEST_ASSERT_EQUAL(cudaStreamSynchronize(0), cudaSuccess);
}

template< typename Matrix, typename Matrix_data >
void check_lu_product(const Matrix &A_h, int n, Matrix_data *lu_d, int lda)
{
    // Compute LxU.
    l_times_u(n, lu_d, lda);
    // Copy LxU to the host.
    Matrix_data *lu_h = new Matrix_data[n * lda];
    cudaMemcpy(lu_h, lu_d, n * lda * sizeof(Matrix_data), cudaMemcpyDeviceToHost);
    UNITTEST_ASSERT_EQUAL(cudaGetLastError(), cudaSuccess);
    // Make sure LxU equals A.
    Matrix_data *dense_A_h = new Matrix_data[n * lda];
    std::memset(dense_A_h, 0x0, n * lda * sizeof(Matrix_data));
    csr_to_dense(A_h, dense_A_h, lda);

    // Compare the matrices.
    for ( int i = 0 ; i < n ; ++i )
        for ( int j = 0 ; j < n ; ++j )
        {
            UNITTEST_ASSERT_EQUAL(lu_h[i * lda + j], dense_A_h[i * lda + j]);
        }

    delete[] dense_A_h;
    delete[] lu_h;
}

DECLARE_UNITTEST_END(DenseLUSolverTest_Base);

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

DECLARE_UNITTEST_BEGIN_EXTD(DenseLUSolverTest_Factorization_Id_32, DenseLUSolverTest_Base<T_Config>);

void run()
{
    // Make sure the solver produces A=LU where A is the identity matrix.
    typedef typename T_Config::template setMemSpace<AMGX_host>::Type Config_h;
    typedef Vector<Config_h> FVector_h;
    typedef typename TConfig::MatPrec Matrix_data;
    const int N = 32;
    Matrix<T_Config> A(N, N, N, CSR);
    A.set_initialized(0);
    this->make_identity(A);
    A.set_initialized(1);
    AMG_Config cfg;
    dense_lu_solver::DenseLUSolver<T_Config> solver(cfg, "", NULL);
    solver.setup(A, false);
    FVector_h m(N * N);
    cudaMemcpy(m.raw(), solver.get_dense_A(), N * N * sizeof(Matrix_data), cudaMemcpyDeviceToHost);
    UNITTEST_ASSERT_EQUAL(cudaGetLastError(), cudaSuccess);

    for ( int i = 0 ; i < N ; ++i )
        for ( int j = 0 ; j < N ; ++j )
        {
            UNITTEST_ASSERT_EQUAL(m[i * solver.get_lda() + j], i == j ? Matrix_data(1) : Matrix_data(0));
        }
}

DECLARE_UNITTEST_END(DenseLUSolverTest_Factorization_Id_32)

DenseLUSolverTest_Factorization_Id_32<TemplateMode<AMGX_mode_dDDI>::Type> DenseLUSolverTest_Factorization_Id_32_dDDI;
DenseLUSolverTest_Factorization_Id_32<TemplateMode<AMGX_mode_dFFI>::Type> DenseLUSolverTest_Factorization_Id_32_dFFI;

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

DECLARE_UNITTEST_BEGIN_EXTD(DenseLUSolverTest_Factorization_Id_256, DenseLUSolverTest_Base<T_Config>);

void run()
{
    // Make sure the solver produces A=LU where A is the identity matrix.
    typedef typename T_Config::template setMemSpace<AMGX_host>::Type Config_h;
    typedef Vector<Config_h> FVector_h;
    typedef typename TConfig::MatPrec Matrix_data;
    const int N = 256;
    Matrix<T_Config> A(N, N, N, CSR);
    A.set_initialized(0);
    this->make_identity(A);
    A.set_initialized(1);
    AMG_Config cfg;
    dense_lu_solver::DenseLUSolver<T_Config> solver(cfg, "", NULL);
    solver.setup(A, false);
    FVector_h m(N * N);
    cudaMemcpy(m.raw(), solver.get_dense_A(), N * N * sizeof(Matrix_data), cudaMemcpyDeviceToHost);
    UNITTEST_ASSERT_EQUAL(cudaGetLastError(), cudaSuccess);

    for ( int i = 0 ; i < N ; ++i )
        for ( int j = 0 ; j < N ; ++j )
        {
            UNITTEST_ASSERT_EQUAL(m[i * solver.get_lda() + j], i == j ? Matrix_data(1) : Matrix_data(0));
        }
}

DECLARE_UNITTEST_END(DenseLUSolverTest_Factorization_Id_256)

DenseLUSolverTest_Factorization_Id_256<TemplateMode<AMGX_mode_dDDI>::Type> DenseLUSolverTest_Factorization_Id_256_dDDI;
DenseLUSolverTest_Factorization_Id_256<TemplateMode<AMGX_mode_dFFI>::Type> DenseLUSolverTest_Factorization_Id_256_dFFI;

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

DECLARE_UNITTEST_BEGIN_EXTD(DenseLUSolverTest_Solve_Poisson3D, DenseLUSolverTest_Base<T_Config>);

void run()
{
    typedef typename T_Config::MatPrec Matrix_data;
    typedef typename T_Config::template setMemSpace<AMGX_device>::Type Config_d;
    typedef typename T_Config::template setMemSpace<AMGX_host>  ::Type Config_h;
    typedef Matrix<Config_d> Matrix_d;
    typedef Vector<Config_d> Vector_d;
    typedef Matrix<Config_h> Matrix_h;
    typedef Vector<Config_h> Vector_h;
    Matrix_h A_h;
    Vector_h x_h, b_h;
    
    A_h.set_initialized(0);
    generatePoissonForTest(A_h, 1, 0, 27, 16, 16, 16);
    A_h.set_initialized(1);
    
    AMG_Config cfg;
    cfg.parseParameterString("monitor_residual=1");
    Matrix_d A_d(A_h);
    dense_lu_solver::DenseLUSolver<T_Config> solver(cfg, "default", NULL);
    solver.setup(A_d, false);
    const int n = A_h.get_num_rows();
    Vector_d b_d(n), x_d(n), r_d(n);
    thrust_wrapper::fill<AMGX_device>(b_d.begin(), b_d.end(), Matrix_data(1));
    solver.solve(b_d, x_d, false);
    solver.compute_residual(b_d, x_d, r_d);
    Vector_h resid_nrm(1);
    solver.compute_norm(r_d, resid_nrm);
    Vector_h rhs_nrm(1);
    solver.compute_norm(b_d, rhs_nrm);
    double relError = resid_nrm[0] / rhs_nrm[0];

    if ( T_Config::matPrec == AMGX_matDouble )
    {
        UNITTEST_ASSERT_EQUAL_TOL(resid_nrm[0], 0.0, 1.0e-12);
    }
    else
    {
        UNITTEST_ASSERT_EQUAL_TOL(resid_nrm[0], 0.0f, 1.0e-6f);
    }
}

DECLARE_UNITTEST_END(DenseLUSolverTest_Solve_Poisson3D)

DenseLUSolverTest_Solve_Poisson3D<TemplateMode<AMGX_mode_dDDI>::Type> DenseLUSolverTest_Solve_Poisson3D_dDDI;

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

DECLARE_UNITTEST_BEGIN_EXTD(DenseLUSolverTest_Solve_Id_32, DenseLUSolverTest_Base<T_Config>);

void run()
{
    typedef typename T_Config::template setMemSpace<AMGX_host>::Type Config_h;
    typedef Vector<Config_h> FVector_h;
    typedef typename TConfig::MatPrec Matrix_data;
    const int N = 32;
    Matrix<T_Config> A(N, N, N, CSR);
    A.set_initialized(0);
    this->make_identity(A);
    A.set_initialized(1);
    AMG_Config cfg;
    dense_lu_solver::DenseLUSolver<T_Config> solver(cfg, "", NULL);
    solver.setup(A, false);
    FVector_h b_h(N);

    for ( int i = 0 ; i < N ; ++i )
    {
        b_h[i] = Matrix_data(rand()) / RAND_MAX;
    }

    Vector<T_Config> b(b_h), x(N), r(N);
    solver.solve(b, x, false);
    UNITTEST_ASSERT_EQUAL(x, b);
}

DECLARE_UNITTEST_END(DenseLUSolverTest_Solve_Id_32)

DenseLUSolverTest_Solve_Id_32<TemplateMode<AMGX_mode_dDDI>::Type> DenseLUSolverTest_Solve_Id_32_dDDI;
DenseLUSolverTest_Solve_Id_32<TemplateMode<AMGX_mode_dFFI>::Type> DenseLUSolverTest_Solve_Id_32_dFFI;

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

DECLARE_UNITTEST_BEGIN_EXTD(DenseLUSolverTest_Solve_Id_256, DenseLUSolverTest_Base<T_Config>);

void run()
{
    typedef typename T_Config::template setMemSpace<AMGX_host>::Type Config_h;
    typedef Vector<Config_h> FVector_h;
    typedef typename TConfig::MatPrec Matrix_data;
    const int N = 256;
    Matrix<T_Config> A(N, N, N, CSR);
    A.set_initialized(0);
    this->make_identity(A);
    A.set_initialized(1);
    AMG_Config cfg;
    dense_lu_solver::DenseLUSolver<T_Config> solver(cfg, "", NULL);
    solver.setup(A, false);
    FVector_h b_h(N);

    for ( int i = 0 ; i < N ; ++i )
    {
        b_h[i] = Matrix_data(rand()) / RAND_MAX;
    }

    Vector<T_Config> b(b_h), x(N);
    solver.solve(b, x, false);
    UNITTEST_ASSERT_EQUAL(x, b);
}

DECLARE_UNITTEST_END(DenseLUSolverTest_Solve_Id_256)

DenseLUSolverTest_Solve_Id_256<TemplateMode<AMGX_mode_dDDI>::Type> DenseLUSolverTest_Solve_Id_256_dDDI;
DenseLUSolverTest_Solve_Id_256<TemplateMode<AMGX_mode_dFFI>::Type> DenseLUSolverTest_Solve_Id_256_dFFI;

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace amgx

