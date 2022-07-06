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

#include <eigensolvers/subspace_iteration_eigensolver.h>
#include <eigensolvers/multivector_operations.h>
#include <blas.h>
#include <multiply.h>
#include <norm.h>
#include <eigensolvers/qr.h>
#include <amgx_cusparse.h>
#include <amgx_cublas.h>
#include <amgx_lapack.h>
#include <algorithm>

namespace amgx
{

template <class TConfig>
SubspaceIteration_EigenSolver<TConfig>::SubspaceIteration_EigenSolver(AMG_Config &cfg, const std::string &cfg_scope)
    : Base(cfg, cfg_scope)
{
    if (this->m_which != EIG_LARGEST)
    {
        FatalError("Subspace Iteration: can only compute largest eigenpair.", AMGX_ERR_CONFIGURATION);
    }

    if (TConfig::memSpace == AMGX_host)
    {
        FatalError("Host is unsupported for Subspace iteration.", AMGX_ERR_NOT_SUPPORTED_TARGET);
    }
    else
    {
        m_wanted_count = cfg.getParameter<int>("eig_wanted_count", cfg_scope);
        m_subspace_size = cfg.getParameter<int>("eig_subspace_size", cfg_scope);

        // If subspace dimension was not set, use heuristic from:
        // http://web.eecs.utk.edu/~dongarra/etemplates/node99.html
        if (m_subspace_size == -1)
        {
            m_subspace_size = std::min(2 * m_wanted_count, m_wanted_count + 8);
        }

        if (m_subspace_size < m_wanted_count)
        {
            FatalError("Subspace Iteration: subspace dimensions is too small for the number of sought eigenvalues", AMGX_ERR_CONFIGURATION);
        }
    }
}

template <class TConfig>
SubspaceIteration_EigenSolver<TConfig>::~SubspaceIteration_EigenSolver()
{
}

template <typename TConfig>
void SubspaceIteration_EigenSolver<TConfig>::orthonormalize(VVector &V)
{
    Matrix<TConfig> *pA = dynamic_cast< Matrix<TConfig>* > (this->m_A);
    HouseholderQR<TConfig> qr(*pA);
    qr.QR_decomposition(V);
}

template <class TConfig>
void SubspaceIteration_EigenSolver<TConfig>::solver_setup()
{
    Operator<TConfig> &A = *this->m_A;
    ViewType oldView = A.currentView();
    A.setViewExterior();
    int rows = A.get_num_rows();
    int cols = A.get_num_cols();
    int subspace_size = m_subspace_size;
    srand(42);
    Vector_h x(cols * subspace_size);

    for (int i = 0; i < cols; ++i)
        for (int j = 0; j < subspace_size; ++j)
        {
            x[j * cols + i] = rand() / ValueTypeVec(RAND_MAX);
        }

    // Vectors need halo space, so use the number of columns for
    // allocation.  In distributed settings, rows indicate the
    // number of interior and boundary nodes while columns indicates the number
    // of interior, boundary and halo nodes.
    m_X = x;
    m_X.set_lda(cols);
    m_X.set_num_rows(rows);
    m_X.set_num_cols(subspace_size);
    m_X.set_block_dimx(1);
    m_X.set_block_dimy(1);
    m_V.resize(cols * subspace_size);
    m_V.set_lda(cols);
    m_V.set_num_rows(rows);
    m_V.set_num_cols(subspace_size);
    m_V.tag = 1;
    m_V.set_block_dimx(1);
    m_V.set_block_dimy(1);
    m_R.resize(cols * subspace_size);
    m_R.set_lda(cols);
    m_R.set_num_rows(rows);
    m_R.set_num_cols(subspace_size);
    m_R.set_block_dimx(1);
    m_R.set_block_dimy(1);
    m_H.resize(subspace_size * subspace_size);
    m_H.set_lda(subspace_size);
    m_H.set_num_rows(subspace_size);
    m_H.set_num_cols(subspace_size);
    m_H.set_block_dimx(1);
    m_H.set_block_dimy(1);
    A.setView(oldView);
}

template <class TConfig>
void SubspaceIteration_EigenSolver<TConfig>::solver_pagerank_setup(VVector &a)
{
    if (this->m_which == EIG_PAGERANK)
    {
        printf("FATAL ERROR : PageRank is only supported in POWER_ITERATION / SINGLE_ITERATION solvers \n");
    }
}

template <class TConfig>
void SubspaceIteration_EigenSolver<TConfig>::solve_init(VVector &x)
{
}

// Pseudo-MATLAB code of the algorithm:
// while true
//   [V, ~] = qr(X, 0);
//   X = A * V;
//   H = V ' * X;
//   R = X - V * H';
//   if converged(R)
//     break
//   end
template <class TConfig>
bool SubspaceIteration_EigenSolver<TConfig>::solve_iteration(VVector &x)
{
    Matrix<TConfig> *pA = dynamic_cast< Matrix<TConfig>* > (this->m_A);
    Matrix<TConfig> &A = *pA;
    ViewType oldView = A.currentView();
    A.setViewExterior();
    int offset, size;
    A.getOffsetAndSizeForView(A.getViewExterior(), &offset, &size);
    int rows = A.get_num_rows();
    int cols = A.get_num_cols();
    // [V, ~] = qr(X, 0);
    copy(m_X, m_V);
    orthonormalize(m_V);
    // X = A * V;
    Cusparse::csrmm(1, A, m_V, 0, m_X);

    // H = V ' * X;
    if (!A.is_matrix_singleGPU())
    {
        distributed_gemm_TN(1, m_V, m_X, 0, m_H, A);
    }
    else
    {
        Cublas::gemm(1, m_V, m_X, 0, m_H, true, false);
    }

    // R = X - V * H';
    // First step: R = X
    copy(m_X, m_R);
    // Second step: R -= V * H';
    Cublas::gemm(-1, m_V, m_H, 1, m_R, false, true);
    // Check convergence.
    Vector_h column_norms;
    multivector_column_norms(m_R, column_norms, A);
    ValueTypeVec max_norm = 0.;

    // Only check convergence of the largest eigenvalues.
    for (int i = 0; i < m_wanted_count; ++i)
    {
        max_norm = std::max(max_norm, column_norms[i]);
    }

    A.setView(oldView);

    if (this->m_curr_iter == 0)
    {
        m_initial_residual = max_norm;
        return false;
    }

    // Compute relative reduction of residual.
    if (max_norm / m_initial_residual < this->m_tolerance)
    {
        Vector_h eigenvalues(m_subspace_size);
        Vector_h H_host = m_H;
        Lapack<TConfig_h>::geev(H_host, eigenvalues);
        std::sort(eigenvalues.begin(), eigenvalues.end(),
                  std::greater<ValueTypeVec>());

        // Only keep the largest eigenvalues.
        for (int i = 0; i < m_wanted_count; ++i)
        {
            this->m_eigenvalues.push_back(eigenvalues[i]);
        }

        return true;
    }
    else
    {
        return false;
    }
}

template <class TConfig>
void SubspaceIteration_EigenSolver<TConfig>::solve_finalize()
{
}

// Explicit template instantiation.
template class SubspaceIteration_EigenSolver<TConfigGeneric_d>;
template class SubspaceIteration_EigenSolver<TConfigGeneric_h>;

};
