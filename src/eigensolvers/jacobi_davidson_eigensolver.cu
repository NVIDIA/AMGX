// SPDX-FileCopyrightText: 2013 - 2025 NVIDIA CORPORATION. All Rights Reserved.
//
// SPDX-License-Identifier: BSD-3-Clause

#include <eigensolvers/jacobi_davidson_eigensolver.h>
#include <eigensolvers/multivector_operations.h>
#include <blas.h>
#include <multiply.h>
#include <norm.h>
#include <eigensolvers/qr.h>
#include <amgx_cusparse.h>
#include <amgx_cublas.h>
#include <amgx_lapack.h>
#include <solvers/solver.h>
#include <operators/solve_operator.h>
#include <operators/deflated_multiply_operator.h>

namespace amgx
{

template <class TConfig>
JacobiDavidson_EigenSolver<TConfig>::JacobiDavidson_EigenSolver(AMG_Config &cfg, const std::string &cfg_scope)
    : Base(cfg, cfg_scope), m_cfg(cfg)
{
    if (this->m_which != EIG_LARGEST)
    {
        FatalError("JacobiDavidson: can only compute largest eigenpair.", AMGX_ERR_CONFIGURATION);
    }
}

template <class TConfig>
JacobiDavidson_EigenSolver<TConfig>::~JacobiDavidson_EigenSolver()
{
    delete m_operator;
    free_allocated();
}

template <class TConfig>
void JacobiDavidson_EigenSolver<TConfig>::free_allocated()
{
}

template <class TConfig>
void JacobiDavidson_EigenSolver<TConfig>::solver_setup()
{
    Operator<TConfig> &A = *this->m_A;
    ViewType oldView = A.currentView();
    A.setViewExterior();
    const int N = A.get_num_cols() * A.get_block_dimy();
    m_x.resize(N);
    m_Ax.resize(N);
    m_d.resize(N);
    m_work.resize(N);
    std::vector<VVector *> allocated_vectors;
    allocated_vectors.push_back(&m_x);
    allocated_vectors.push_back(&m_Ax);
    allocated_vectors.push_back(&m_d);
    allocated_vectors.push_back(&m_work);
    int current_tag = 100;

    for (int i = 0; i < allocated_vectors.size(); ++i)
    {
        VVector *v = allocated_vectors[i];
        v->tag = current_tag++;
        v->set_block_dimy(A.get_block_dimy());
        v->set_block_dimx(1);
        v->dirtybit = 1;
        v->delayed_send = 1;
        v->set_num_rows(N);
        v->set_num_cols(1);
        v->set_lda(N);
    }

    int max_iters = this->m_max_iters;
    int rows = A.get_num_rows();
    int cols = A.get_num_cols();
    m_V.resize(cols * max_iters);
    m_V.set_lda(cols);
    m_V.set_num_rows(rows);
    m_V.tag = current_tag++;
    m_V.set_block_dimx(1);
    m_V.set_block_dimy(1);
    m_V.dirtybit = 1;
    m_V.delayed_send = 1;
    m_AV.resize(cols * max_iters);
    m_AV.set_lda(cols);
    m_AV.set_num_rows(rows);
    m_AV.tag = current_tag++;
    m_AV.set_block_dimx(1);
    m_AV.set_block_dimy(1);
    m_AV.dirtybit = 1;
    m_AV.delayed_send = 1;
    m_H.resize(max_iters * max_iters);
    m_H.set_lda(max_iters);
    m_H.set_num_rows(max_iters);
    m_H.set_block_dimx(1);
    m_H.set_block_dimy(1);
    m_H.dirtybit = 1;
    m_H.delayed_send = 1;
    m_s.resize(max_iters);
    m_s.set_lda(max_iters);
    m_s.set_num_rows(max_iters);
    m_s.set_num_cols(1);
    m_s.set_block_dimx(1);
    m_s.set_block_dimy(1);
    m_s.dirtybit = 1;
    m_s.delayed_send = 1;
    m_subspace_eigenvalues.resize(max_iters);
    m_deflated_operator = new DeflatedMultiplyOperator<TConfig>(A);
    Solver<TConfig> *solver = SolverFactory<TConfig>::allocate(m_cfg, "default", "solver");
    SolveOperator<TConfig> *solve_op = new SolveOperator<TConfig>(*m_deflated_operator, *solver);
//        SolveOperator<TConfig>* solve_op = new SolveOperator<TConfig>(A, *solver);
    solve_op->setup();
    m_operator = solve_op;
    A.setView(oldView);
}
template <class TConfig>
void JacobiDavidson_EigenSolver<TConfig>::solver_pagerank_setup(VVector &a)
{
    if (this->m_which == EIG_PAGERANK)
    {
        printf("FATAL ERROR : PageRank is only supported in POWER_ITERATION / SINGLE_ITERATION solvers \n");
    }
}
template <class TConfig>
void JacobiDavidson_EigenSolver<TConfig>::solve_init(VVector &x)
{
    Operator<TConfig> &A = *this->m_A;
    ViewType oldView = A.currentView();
    A.setViewExterior();
    int offset, size;
    A.getOffsetAndSizeForView(A.getViewExterior(), &offset, &size);
    copy(x, m_x, offset, size);
    A.setView(oldView);
    // Initialize subspace dimension.
    set_subspace_size(0);
}

template <typename TConfig>
void JacobiDavidson_EigenSolver<TConfig>::orthonormalize(VVector &V)
{
    Matrix<TConfig> *pA = dynamic_cast< Matrix<TConfig>* > (this->m_A);
    Matrix<TConfig> &A = *pA;
    HouseholderQR<TConfig> qr(A);
    qr.QR_decomposition(V);
}

template <typename TConfig>
void JacobiDavidson_EigenSolver<TConfig>::set_subspace_size(int new_size)
{
    m_V.set_num_cols(new_size);
    m_AV.set_num_cols(new_size);
    m_H.set_num_rows(new_size);
    m_H.set_num_cols(new_size);
    m_s.set_num_rows(new_size);
}

template <class TConfig>
bool JacobiDavidson_EigenSolver<TConfig>::solve_iteration(VVector &x)
{
    Matrix<TConfig> *pA = dynamic_cast< Matrix<TConfig>* > (this->m_A);
    Matrix<TConfig> &A = *pA;
    ViewType oldView = A.currentView();
    A.setViewExterior();
    int offset, size;
    A.getOffsetAndSizeForView(A.getViewExterior(), &offset, &size);
    int iter = this->m_curr_iter;
    int subspace_size = iter + 1;
    // mu = (x' * A * x) / (x' * x);
    multiply(A, m_x, m_Ax);
    ValueTypeVec xtAx = dot(A, m_x, m_Ax);
    ValueTypeVec xtx = dot(A, m_x, m_x);
    ValueTypeVec mu = xtAx / xtx;
    m_deflated_operator->set_x(m_x);
    m_deflated_operator->set_mu(mu);
    m_deflated_operator->set_workspace(m_work);
    // r = (A - mu * I) * x;
    axpy(m_x, m_Ax, -mu, offset, size);
    ValueTypeVec residual_norm = get_norm(A, m_Ax, L2);

    if (iter != 0)
    {
        this->m_residuals.push_back(residual_norm / fabs(m_lambda));

        if (residual_norm < this->m_tolerance * fabs(m_lambda))
        {
            this->m_eigenvalues.push_back(m_lambda);
            return true;
        }
    }

    // B = (I - x * x') * (A - mu * I) * (I - x * x');
    // d = gmres(B, -r);
    scal(m_Ax, ValueTypeVec(-1), offset, size);
    m_operator->apply(m_Ax, m_d);
    set_subspace_size(subspace_size);
    // V = [V d];
    ValueTypeVec *dst_ptr = m_V.raw() + m_V.get_lda() * (m_V.get_num_cols() - 1);
    Cublas::copy(size, m_d.raw(), 1, dst_ptr, 1);
    // [V, ~] = qr(V, 0);
    orthonormalize(m_V);
    // H = V' * A * V;
    Cusparse::csrmm(1, A, m_V, 0, m_AV);

    if (!A.is_matrix_singleGPU())
    {
        distributed_gemm_TN(1, m_V, m_AV, 0, m_H, A);
    }
    else
    {
        Cublas::gemm(1, m_V, m_AV, 0, m_H, true, false);
    }

    // [s, theta] = eigs(H, 1);
    Lapack<TConfig>::syevd(m_H, m_subspace_eigenvalues);
    // Eigenvalues are stored in ascending order.
    m_lambda = m_subspace_eigenvalues[subspace_size - 1];
    // Extract in variable s the eigenvector corresponding to
    // largest eigenvalue.
    ValueTypeVec *src_ptr = m_H.raw() + (subspace_size - 1) * m_H.get_lda();
    Cublas::copy(subspace_size, src_ptr, 1, m_s.raw(), 1);
    ValueTypeVec alpha = 1.;
    ValueTypeVec beta = 0.;
    // x = V * s;
    Cublas::gemv(false, m_V.get_num_rows(), m_V.get_num_cols(), &alpha, m_V.raw(), m_V.get_lda(),
                 m_s.raw(), 1, &beta, m_x.raw(), 1);
    m_x.dirtybit = 1;
    A.setView(oldView);
    return false;
}

template <class TConfig>
void JacobiDavidson_EigenSolver<TConfig>::solve_finalize()
{
}

// Explicit template instantiation.
#define AMGX_CASE_LINE(CASE) template class JacobiDavidson_EigenSolver<TemplateMode<CASE>::Type>;
AMGX_FORALL_BUILDS(AMGX_CASE_LINE)
#undef AMGX_CASE_LINE

};
