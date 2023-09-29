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

#include <solvers/polynomial_solver.h>
#include <solvers/block_common_solver.h>
#include <blas.h>
#include <string.h>
#include <cutil.h>
#include <multiply.h>
#include <miscmath.h>

#ifdef _WIN32
#pragma warning (push)
#pragma warning (disable : 4244 4267 4521)
#endif
#include <cusp/relaxation/polynomial.h>
#include <cusp/detail/spectral_radius.h>
#ifdef _WIN32
#pragma warning (pop)
#endif

#include <matrix_cusp.h>


namespace amgx
{
namespace polynomial_solver
{


template <typename IndexType, typename ValueTypeA, typename ValueTypeB>
__global__
void aux_norm1_csr(const IndexType *row_offsets, const IndexType *column_indices, const IndexType *dia_values, const ValueTypeA *nonzero_values,
                   const int num_rows, int bsize, ValueTypeB *row_sum)
{
    int tid = threadIdx.x + blockDim.x * blockIdx.x;
    int bsize_sq = bsize * bsize;

    while (tid < num_rows)
    {
        ValueTypeB tmp_sum = 0;
        int idx_i = tid / bsize;
        int offset_i = tid % bsize;

        for (int i = 0; i < bsize; i++)
        {
            tmp_sum += fabs(nonzero_values[dia_values[idx_i] * bsize_sq + offset_i * bsize + i]);
        }

        for (int j = row_offsets[idx_i]; j < row_offsets[idx_i + 1]; j++)
        {
            // Compute edge weight
            for (int i = 0; i < bsize; i++)
            {
                tmp_sum += fabs(nonzero_values[j * bsize_sq + offset_i * bsize + i]);
            }
        }

        row_sum[tid] = tmp_sum / fabs(nonzero_values[dia_values[idx_i] * bsize_sq + offset_i * bsize + offset_i]);
        tid += gridDim.x * blockDim.x;
    }
}

template <typename IndexType, typename ValueTypeA, typename ValueTypeB>
__global__
void get_diaginv(const IndexType *dia_idx, const ValueTypeA *nonzero_values,
                 const int num_rows, int bsize, ValueTypeB *Dinv)
{
    int tid = threadIdx.x + blockDim.x * blockIdx.x;
    int bsize_sq = bsize * bsize;

    while (tid < num_rows)
    {
        int idx_i = tid / bsize;
        int offset_i = tid % bsize;
        ValueTypeB diag = nonzero_values[dia_idx[idx_i] * bsize_sq + offset_i * bsize + offset_i];
        Dinv[tid] = ValueTypeB(1) / (isNotCloseToZero(diag) ? diag : epsilon(diag) );
        tid += gridDim.x * blockDim.x;
    }
}


// Constructor
template<class T_Config>
PolynomialSolverBase<T_Config>::PolynomialSolverBase( AMG_Config &cfg, const std::string &cfg_scope) : Solver<T_Config>( cfg, cfg_scope), R(0)
{
    ndeg0 = cfg.AMG_Config::template getParameter<int>("kpz_order", cfg_scope);
}

// Destructor
template<class T_Config>
PolynomialSolverBase<T_Config>::~PolynomialSolverBase()
{
}

template<class T_Config>
void
PolynomialSolverBase<T_Config>::printSolverParameters() const
{
    std::cout << "kpz_order = " << this->ndeg0 << std::endl;
}


// Solver setup
template<class T_Config>
void
PolynomialSolverBase<T_Config>::solver_setup(bool reuse_matrix_structure)
{
    this->m_explicit_A = dynamic_cast<Matrix<T_Config>*>(this->m_A);

    if (!this->m_explicit_A)
    {
        FatalError("PolynomialSolver only works with explicit matrices", AMGX_ERR_INTERNAL);
    }

    if (ndeg0 == 0) { ndeg0 = 6; }

    int N = this->m_explicit_A->get_num_rows() * this->m_explicit_A->get_block_dimy();
    ValueTypeA  mu0, mu1, smu0, smu1;
    const IndexType *A_row_offsets_ptr = this->m_explicit_A->row_offsets.raw();
    const IndexType *A_column_indices_ptr = this->m_explicit_A->col_indices.raw();
    const IndexType *A_dia_ptr = this->m_explicit_A->diag.raw();
    const ValueTypeA *A_nonzero_values_ptr = this->m_explicit_A->values.raw();
    VVector row_sum(N);
    ValueTypeB *row_sum_ptr = row_sum.raw();
    const int threads_per_block = 512;
    const int num_blocks = std::min((N - 1) / threads_per_block + 1, AMGX_GRID_MAX_SIZE);
    aux_norm1_csr <<< num_blocks, threads_per_block>>>(A_row_offsets_ptr, A_column_indices_ptr, A_dia_ptr, A_nonzero_values_ptr,
            N, this->m_explicit_A->get_block_dimy(), row_sum_ptr);
    cudaCheckError();
    mu0 = nrmmax(row_sum);
    cudaCheckError();

    if (mu0 == 0) { mu0 = nrmmax(this->m_explicit_A->values); }

    cudaCheckError();
    mu0 = 1.0 / mu0;
    mu1 = 4.0 * mu0; // default set 8;
    smu0 =  sqrt(mu0);
    smu1 = sqrt(mu1);
    k[1] = (mu0 + mu1) / 2.0;
    k[2] = (smu0 + smu1) * (smu0 + smu1) / 2.0;
    k[3] = mu0 * mu1;
    k[4] = 2.0 * k[3] / k[2]; // 4.0*mu0*mu1/(sqrt(mu0)+sqrt(mu1))/(sqrt(mu0)+sqrt(mu1));
    k[5] = (mu1 - 2.0 * smu0 * smu1 + mu0) / (mu1 + 2.0 * smu0 * smu1 + mu0); // square of (sqrt(kappa)-1)/(sqrt(kappa)+1);

    if (this->m_explicit_A->get_block_size() == 1)
    {
        MatrixCusp<T_Config, cusp::csr_format> wA((Matrix<T_Config> *) &*this->m_explicit_A);
        ValueTypeA rho = cusp::detail::ritz_spectral_radius_symmetric(wA, 8);
        cudaCheckError();
        cusp::array1d<ValueTypeA, cusp::host_memory> coeffs;
        cusp::relaxation::detail::chebyshev_polynomial_coefficients(rho, coeffs);
        cudaCheckError();
        poly = cusp::relaxation::polynomial<ValueTypeA, typename Matrix<T_Config>::memory_space > (wA, coeffs);
        cudaCheckError();
    }

    R.resize(N);
    V0.resize(N);
    V.resize(N);
    Rbar.resize(N);
    Sn.resize(N);
    Dinv.resize(N);
    R.set_block_dimy(this->m_explicit_A->get_block_dimy());
    R.set_block_dimx(1);
    V0.set_block_dimy(this->m_explicit_A->get_block_dimy());
    V0.set_block_dimx(1);
    V.set_block_dimy(this->m_explicit_A->get_block_dimy());
    V.set_block_dimx(1);
    Rbar.set_block_dimy(this->m_explicit_A->get_block_dimy());
    Rbar.set_block_dimx(1);
    Sn.set_block_dimy(this->m_explicit_A->get_block_dimy());
    Sn.set_block_dimx(1);
    Dinv.set_block_dimy(this->m_explicit_A->get_block_dimy());
    Dinv.set_block_dimx(1);
    ValueTypeB *Dinv_ptr = Dinv.raw();
    get_diaginv <<< num_blocks, threads_per_block>>>(A_dia_ptr, A_nonzero_values_ptr, N, this->m_explicit_A->get_block_dimy(), Dinv_ptr);
    cudaCheckError();
}

//
template<class T_Config>
void
PolynomialSolverBase<T_Config>::solve_init( VVector &b, VVector &x, bool xIsZero )
{
}


// Solve one iteration
template<class T_Config>
bool
PolynomialSolverBase<T_Config>::solve_iteration( VVector &b, VVector &x, bool xIsZero )
{
    smooth_common_sqblocks( *this->m_explicit_A, b, x );
    return this->converged( b, x );
}

template<class T_Config>
void
PolynomialSolverBase<T_Config>::solve_finalize( VVector &b, VVector &x )
{
}

template<class T_Config>
struct poly_smooth
{
    static void poly_postsmooth(const Matrix<T_Config> &A, const Vector<T_Config> &B, Vector<T_Config> &C, cusp::relaxation::polynomial<typename T_Config::MatPrec, typename T_Config::MemSpace> &poly)
    {
        FatalError("Mixed precision is not supported for scalar matrix type", AMGX_ERR_NOT_IMPLEMENTED);
    }
    static void poly_presmooth(const Matrix<T_Config> &A, const Vector<T_Config> &B, Vector<T_Config> &C, cusp::relaxation::polynomial<typename T_Config::MatPrec, typename T_Config::MemSpace> &poly)
    {
        FatalError("Mixed precision is not supported for scalar matrix type", AMGX_ERR_NOT_IMPLEMENTED);
    }
};

template<AMGX_MemorySpace t_memSpace, AMGX_IndPrecision t_indInt>
struct poly_smooth<TemplateConfig<t_memSpace, AMGX_vecDouble, AMGX_matDouble, t_indInt>>
{
    static void poly_postsmooth(const Matrix<TemplateConfig<t_memSpace, AMGX_vecDouble, AMGX_matDouble, t_indInt> > &A, const Vector<TemplateConfig<t_memSpace, AMGX_vecDouble, AMGX_matDouble, t_indInt> > &B, Vector<TemplateConfig<t_memSpace, AMGX_vecDouble, AMGX_matDouble, t_indInt> > &C, cusp::relaxation::polynomial<double, typename MemorySpaceMap<t_memSpace>::Type> &poly)
    {
        MatrixCusp<TemplateConfig<t_memSpace, AMGX_vecDouble, AMGX_matDouble, t_indInt>, cusp::csr_format> wA((Matrix<TemplateConfig<t_memSpace, AMGX_vecDouble, AMGX_matDouble, t_indInt> > *) &A);
        poly.postsmooth(wA, B, C);
        cudaCheckError();
    }

    static void poly_presmooth(const Matrix<TemplateConfig<t_memSpace, AMGX_vecDouble, AMGX_matDouble, t_indInt> > &A, const Vector<TemplateConfig<t_memSpace, AMGX_vecDouble, AMGX_matDouble, t_indInt> > &B, Vector<TemplateConfig<t_memSpace, AMGX_vecDouble, AMGX_matDouble, t_indInt> > &C, cusp::relaxation::polynomial<double, typename MemorySpaceMap<t_memSpace>::Type> &poly)
    {
        MatrixCusp<TemplateConfig<t_memSpace, AMGX_vecDouble, AMGX_matDouble, t_indInt>, cusp::csr_format> wA((Matrix<TemplateConfig<t_memSpace, AMGX_vecDouble, AMGX_matDouble, t_indInt> > *) &A);
        poly.presmooth(wA, B, C);
        cudaCheckError();
    }
};

template<AMGX_MemorySpace t_memSpace, AMGX_IndPrecision t_indInt>
struct poly_smooth<TemplateConfig<t_memSpace, AMGX_vecFloat, AMGX_matFloat, t_indInt>>
{
    static void poly_postsmooth(const Matrix<TemplateConfig<t_memSpace, AMGX_vecFloat, AMGX_matFloat, t_indInt> > &A, const Vector<TemplateConfig<t_memSpace, AMGX_vecFloat, AMGX_matFloat, t_indInt> > &B, Vector<TemplateConfig<t_memSpace, AMGX_vecFloat, AMGX_matFloat, t_indInt> > &C, cusp::relaxation::polynomial<float, typename MemorySpaceMap<t_memSpace>::Type> &poly)
    {
        MatrixCusp<TemplateConfig<t_memSpace, AMGX_vecFloat, AMGX_matFloat, t_indInt>, cusp::csr_format> wA((Matrix<TemplateConfig<t_memSpace, AMGX_vecFloat, AMGX_matFloat, t_indInt> > *) &A);
        poly.postsmooth(wA, B, C);
        cudaCheckError();
    }

    static void poly_presmooth(const Matrix<TemplateConfig<t_memSpace, AMGX_vecFloat, AMGX_matFloat, t_indInt> > &A, const Vector<TemplateConfig<t_memSpace, AMGX_vecFloat, AMGX_matFloat, t_indInt> > &B, Vector<TemplateConfig<t_memSpace, AMGX_vecFloat, AMGX_matFloat, t_indInt> > &C, cusp::relaxation::polynomial<float, typename MemorySpaceMap<t_memSpace>::Type> &poly)
    {
        MatrixCusp<TemplateConfig<t_memSpace, AMGX_vecFloat, AMGX_matFloat, t_indInt>, cusp::csr_format> wA((Matrix<TemplateConfig<t_memSpace, AMGX_vecFloat, AMGX_matFloat, t_indInt> > *) &A);
        poly.presmooth(wA, B, C);
        cudaCheckError();
    }
};

template <AMGX_VecPrecision t_vecPrec, AMGX_MatPrecision t_matPrec, AMGX_IndPrecision t_indPrec>
void PolynomialSolver<TemplateConfig<AMGX_host, t_vecPrec, t_matPrec, t_indPrec> >::smooth_1x1(const Matrix_h &A, const VVector &B, VVector &C)
{
    FatalError("Unsupported on host for Polynomial smoother", AMGX_ERR_NOT_SUPPORTED_BLOCKSIZE);
}
template <AMGX_VecPrecision t_vecPrec, AMGX_MatPrecision t_matPrec, AMGX_IndPrecision t_indPrec>
void PolynomialSolver<TemplateConfig<AMGX_device, t_vecPrec, t_matPrec, t_indPrec> >::smooth_1x1(const Matrix_d &A, const VVector &B, VVector &C)
{
    if (A.hasProps(DIAG))
    {
        FatalError("Unsupported separate diag", AMGX_ERR_NOT_IMPLEMENTED);
    }

    if (A.get_block_size() != 1)
    {
        FatalError("Unsupported block size for PolynomialSolver", AMGX_ERR_NOT_SUPPORTED_BLOCKSIZE);
    }

    poly_smooth<TemplateConfig<AMGX_device, t_vecPrec, t_matPrec, t_indPrec>>::poly_postsmooth(A, B, C, this->poly);
}

template <AMGX_VecPrecision t_vecPrec, AMGX_MatPrecision t_matPrec, AMGX_IndPrecision t_indPrec>
void PolynomialSolver<TemplateConfig<AMGX_host, t_vecPrec, t_matPrec, t_indPrec> >::smooth_common_sqblocks(Matrix_h &A, const VVector &b, VVector &x)
{
    FatalError("Unsupported on host for Polynomial smoother", AMGX_ERR_NOT_SUPPORTED_BLOCKSIZE);
}
template <AMGX_VecPrecision t_vecPrec, AMGX_MatPrecision t_matPrec, AMGX_IndPrecision t_indPrec>
void PolynomialSolver<TemplateConfig<AMGX_device, t_vecPrec, t_matPrec, t_indPrec> >::smooth_common_sqblocks(Matrix_d &A, const VVector &b, VVector &x)
{
    //0, get residule : R = b - Ax
    multiply( A, x, this->R );
    cusp::blas::axpby( b, this->R, this->R, ValueTypeB(1), ValueTypeB(-1) );
    //1, set up rbar
    cusp::blas::xmy(this->Dinv, this->R, this->Rbar);
    //2, set up V and V0
    multiply( A, this->Rbar, this->V );
    cusp::blas::xmy(this->Dinv, this->V, this->V);
    this->V0 = this->Rbar;
    cusp::blas::scal(this->V0, this->k[1]);
    cusp::blas::axpby(this->Rbar, this->V, this->V, this->k[2], this->k[3]);

    //3, iterate to get v_{j+1}
    for (int i = 0; i < this->ndeg0; i++)
    {
        multiply( A, this->V, this->Rbar );
        cusp::blas::axpby( this->R, this->Rbar, this->Rbar, ValueTypeB(1), ValueTypeB(-1) );
        cusp::blas::xmy(this->Dinv, this->Rbar, this->Rbar);
        cusp::blas::axpbypcz(this->Rbar, this->V, this->V0, this->Sn,
                             this->k[4], this->k[5] + 1.0, -1.0 * this->k[5]);
        // V0 = V
        this->V0 = this->V;
        // V = V+Sn
        cusp::blas::copy(this->Sn, this->V);
    }

    //4, update solution
    cusp::blas::axpy(this->V, x, ValueTypeB(1));
}


template <AMGX_VecPrecision t_vecPrec, AMGX_MatPrecision t_matPrec, AMGX_IndPrecision t_indPrec>
void PolynomialSolver<TemplateConfig<AMGX_host, t_vecPrec, t_matPrec, t_indPrec> >::smooth_with_0_initial_guess_1x1(const Matrix_h &A, const VVector &b, VVector &x)
{
    FatalError("Unsupported on host for Polynomial smoother", AMGX_ERR_NOT_SUPPORTED_BLOCKSIZE);
}
template <AMGX_VecPrecision t_vecPrec, AMGX_MatPrecision t_matPrec, AMGX_IndPrecision t_indPrec>
void PolynomialSolver<TemplateConfig<AMGX_device, t_vecPrec, t_matPrec, t_indPrec> >::smooth_with_0_initial_guess_1x1(const Matrix_d &A, const VVector &b, VVector &x)
{
    if (A.hasProps(DIAG))
    {
        FatalError("Unsupported separate diag", AMGX_ERR_NOT_IMPLEMENTED);
    }

    if (A.get_block_size() != 1)
    {
        FatalError("Unsupported block size for PolynomialSolver", AMGX_ERR_NOT_SUPPORTED_BLOCKSIZE);
    }

    poly_smooth<TemplateConfig<AMGX_device, t_vecPrec, t_matPrec, t_indPrec>>::poly_presmooth(A, b, x, this->poly);
}

/****************************************
 * Explict instantiations
 ***************************************/
#define AMGX_CASE_LINE(CASE) template class PolynomialSolverBase<TemplateMode<CASE>::Type>;
AMGX_FORALL_BUILDS(AMGX_CASE_LINE)
#undef AMGX_CASE_LINE

#define AMGX_CASE_LINE(CASE) template class PolynomialSolver<TemplateMode<CASE>::Type>;
AMGX_FORALL_BUILDS(AMGX_CASE_LINE)
#undef AMGX_CASE_LINE




} // namespace polynomial_solver
} // namespace amgx
