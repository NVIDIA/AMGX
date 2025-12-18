// SPDX-FileCopyrightText: 2013 - 2025 NVIDIA CORPORATION. All Rights Reserved.
//
// SPDX-License-Identifier: BSD-3-Clause

#include <eigensolvers/lobpcg_eigensolver.h>
#include <solvers/solver.h>
#include <blas.h>
#include <multiply.h>
#include <norm.h>
#include <amgx_cusparse.h>
#include <cublas_v2.h>
#include <amgx_lapack.h>

//#define AMGX_LOBPCG_USE_SOLVER 1
/*

function [ eigenvec, eigenval ] = lobpcg_impl(X, A, M)
    [N, m] = size(X);

    maxiters = 100;
    % Normalize X.
    X = qr(X, 0);
    P = zeros(N, m);

    for k = 0:maxiters-1
        mu = dot(X, X) / dot(X, A * X);
        R = X - mu * A * X;
        %W = T * R;
        W = R;
        W = qr(W, 0);

        % Perform Rayleigh-Ritz procedure.

        % Compute symmetric Gram matrices.
        if k > 0
            P = qr(P, 0);
            Z = [W, X, P];
        else
            Z = [W, X];
        end

        gramA = Z' * A * Z;
        gramB = Z' * Z;

        % Solve generalized eigenvalue problem.
        [Y, lambda] = eigs(gramA, gramB, m);

        % Compute Ritz vectors.
        Yw = Y(1:m, :);
        Yx = Y(m+1:2*m, :);
        if k > 0
            Yp = Y(2*m+1:end, :);
        else
            Yp = zeros(m, m);
        end
        X = W * Yw + X * Yx + P * Yp;
        P = W * Yw + P * Yp;
    end
    eigenval = lambda;
    eigenvec = X;
end

*/

namespace amgx
{


template <class TConfig>
LOBPCG_EigenSolver<TConfig>::LOBPCG_EigenSolver(AMG_Config &cfg,
        const std::string &cfg_scope)
    : Base(cfg, cfg_scope), m_subspace_size(1), m_solver(0)
{
    // LOBPCG can also compute smallest eigenvalue but convergence seems terrible.
    if (this->m_which != EIG_LARGEST)
    {
        FatalError("LOBPCG: can only compute largest eigenpair.", AMGX_ERR_CONFIGURATION);
    }

#ifdef AMGX_LOBPCG_USE_SOLVER
    m_solver = SolverFactory<TConfig>::allocate(cfg, "default", "solver");
#endif
    m_work.resize(4096);
}

template <class TConfig>
LOBPCG_EigenSolver<TConfig>::~LOBPCG_EigenSolver()
{
    free_allocated();
    delete m_solver;
}


template <class TConfig>
void LOBPCG_EigenSolver<TConfig>::free_allocated()
{
}

template <class TConfig>
void LOBPCG_EigenSolver<TConfig>::solver_setup()
{
    Operator<TConfig> &A = *this->m_A;
    ViewType oldView = A.currentView();
    A.setViewExterior();
    int offset, size;
    A.getOffsetAndSizeForView(A.getViewExterior(), &offset, &size);
    int rows = A.get_num_rows();
    std::vector<VVector *> allocated_vectors;
    const int N = static_cast<int>(A.get_num_cols() * A.get_block_dimy());
    X.resize(N);
    AX.resize(N);
    W.resize(N);
#ifdef AMGX_LOBPCG_USE_SOLVER
    W_cpy.resize(N);
#endif
    AW.resize(N);
    P.resize(N);
    AP.resize(N);
    free_allocated();
    allocated_vectors.push_back(&X);
    allocated_vectors.push_back(&AX);
    allocated_vectors.push_back(&W);
#ifdef AMGX_LOBPCG_USE_SOLVER
    allocated_vectors.push_back(&W_cpy);
#endif
    allocated_vectors.push_back(&AW);
    allocated_vectors.push_back(&P);
    allocated_vectors.push_back(&AP);
    int start_tag = 100;

    for (int i = 0; i < allocated_vectors.size(); ++i)
    {
        VVector *v = allocated_vectors[i];
        v->tag = start_tag + i;
        v->set_block_dimy(A.get_block_dimy());
        v->set_block_dimx(1);
        v->dirtybit = 1;
        v->delayed_send = 1;
    }

#ifdef AMGX_LOBPCG_USE_SOLVER
    m_solver->setup(A, false);
#endif
    A.setView(oldView);
}

template <class TConfig>
void LOBPCG_EigenSolver<TConfig>::solver_pagerank_setup(VVector &a)
{
    if (this->m_which == EIG_PAGERANK)
    {
        printf("FATAL ERROR : PageRank is only supported in POWER_ITERATION / SINGLE_ITERATION solvers \n");
    }
}

template <class TConfig>
void LOBPCG_EigenSolver<TConfig>::solve_init(VVector &x)
{
    Operator<TConfig> &A = *this->m_A;
    ViewType oldView = A.currentView();
    A.setViewExterior();
    int offset, size;
    A.getOffsetAndSizeForView(A.getViewExterior(), &offset, &size);
    copy(x, X, offset, size);
    scal(X, 1 / get_norm(A, X, this->m_norm_type), offset, size);
    //multiply(A, X, AX);
    A.apply(X, AX);
    // Initial lambda.
    m_lambda = dot(A, X, AX);
    A.setView(oldView);
}

template <class TConfig>
bool LOBPCG_EigenSolver<TConfig>::solve_iteration(VVector &x)
{
    Operator<TConfig> &A = *this->m_A;
    ViewType oldView = A.currentView();
    A.setViewExterior();
    int offset, size;
    A.getOffsetAndSizeForView(A.getViewExterior(), &offset, &size);
    int rows = A.get_num_rows();
    int k = this->m_curr_iter;
    axpby(AX, X, W, ValueTypeVec(1), -m_lambda, offset, size);
    ValueTypeVec residual_norm = get_norm(A, W, L2);
    this->m_residuals.push_back(residual_norm / fabs(m_lambda));

    if (residual_norm < this->m_tolerance * fabs(m_lambda))
    {
        this->m_eigenvalues.push_back(m_lambda);
        return true;
    }

#ifdef AMGX_LOBPCG_USE_SOLVER
    copy(W, W_cpy, offset, size);
    AMGX_STATUS solve_status = m_solver->solve(W_cpy, W, false);
#endif
    // Normalize W.
    scal(W, 1 / get_norm(A, W, L2), offset, size);
    A.apply(W, AW);

    if (k > 0)
    {
        ValueTypeVec normP = get_norm(A, P, L2);
        scal(P, 1 / normP, offset, size);
        scal(AP, 1 / normP, offset, size);
    }

    ValueTypeVec XAW = dot(A, X, AW);
    ValueTypeVec WAW = dot(A, W, AW);
    ValueTypeVec XBW = dot(A, X, W);
    ValueTypeVec XAP, WAP, PAP, XBP, WBP;

    if (k > 0)
    {
        XAP = dot(A, X, AP);
        WAP = dot(A, W, AP);
        PAP = dot(A, P, AP);
        XBP = dot(A, X, P);
        WBP = dot(A, W, P);
    }
    else
    {
        XAP = WAP = PAP = XBP = WBP = 0;
    }

    Vector_h gramA(3 * 3);
    gramA.set_num_rows(3);
    gramA.set_num_cols(3);
    gramA.set_lda(3);
    gramA[0 * 3 + 0] = m_lambda;
    gramA[0 * 3 + 1] = XAW;
    gramA[0 * 3 + 2] = XAP;
    gramA[1 * 3 + 0] =      XAW;
    gramA[1 * 3 + 1] = WAW;
    gramA[1 * 3 + 2] = WAP;
    gramA[2 * 3 + 0] =      XAP;
    gramA[2 * 3 + 1] = WAP;
    gramA[2 * 3 + 2] = PAP;
    Vector_h gramB(3 * 3);
    gramB.set_num_rows(3);
    gramB.set_num_cols(3);
    gramB.set_lda(3);
    gramB[0 * 3 + 0] =        1;
    gramB[0 * 3 + 1] = XBW;
    gramB[0 * 3 + 2] = XBP;
    gramB[1 * 3 + 0] =      XBW;
    gramB[1 * 3 + 1] =   1;
    gramB[1 * 3 + 2] = WBP;
    gramB[2 * 3 + 0] =      XBP;
    gramB[2 * 3 + 1] = WBP;
    gramB[2 * 3 + 2] =   1;
    Vector_h Y(3);
    Lapack<TConfig_h>::sygv(gramA, gramB, Y, m_work);
    // Get largest eigenvalue.
    m_lambda = Y[2];

    // Compute Ritz vectors.
    if (k > 0)
    {
        ValueTypeVec Yx = gramA[2 * 3 + 0];
        ValueTypeVec Yw = gramA[2 * 3 + 1];
        ValueTypeVec Yp = gramA[2 * 3 + 2];
        axpby(W, P, P, Yw, Yp, offset, size);
        axpby(AW, AP, AP, Yw, Yp, offset, size);
        axpby(X, P, X, Yx, ValueTypeVec(1), offset, size);
        axpby(AX, AP, AX, Yx, ValueTypeVec(1), offset, size);
    }
    else
    {
        // To get the same results as MATLAB: negate Yx and Yw.
        ValueTypeVec Yx = gramA[2 * 3 + 0];
        ValueTypeVec Yw = gramA[2 * 3 + 1];
        fill(P, 0, offset, size);
        axpy(W, P, Yw, offset, size);
        fill(AP, 0, offset, size);
        axpy(AW, AP, Yw, offset, size);
        scal(X, Yx, offset, size);
        axpy(P, X, ValueTypeVec(1.), offset, size);
        scal(AX, Yx, offset, size);
        axpy(AP, AX, ValueTypeVec(1.), offset, size);
    }

    A.setView(oldView);
    return false;
}

template <class TConfig>
void LOBPCG_EigenSolver<TConfig>::solve_finalize()
{
}

// Explicit template instantiation.
#define AMGX_CASE_LINE(CASE) template class LOBPCG_EigenSolver<TemplateMode<CASE>::Type>;
AMGX_FORALL_BUILDS(AMGX_CASE_LINE)
#undef AMGX_CASE_LINE

};
