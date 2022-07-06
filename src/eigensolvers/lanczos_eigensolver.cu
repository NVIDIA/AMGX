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

#include <eigensolvers/lanczos_eigensolver.h>
#include <blas.h>
#include <multiply.h>
#include <norm.h>
#include <cmath>
#include <amgx_lapack.h>

namespace amgx
{

template <class TConfig>
Lanczos_EigenSolver<TConfig>::Lanczos_EigenSolver(AMG_Config &cfg, const std::string &cfg_scope)
    : Base(cfg, cfg_scope)
{
}

template <class TConfig>
Lanczos_EigenSolver<TConfig>::~Lanczos_EigenSolver()
{
    free_allocated();
}

template <class TConfig>
void Lanczos_EigenSolver<TConfig>::free_allocated()
{
}

template <class TConfig>
void Lanczos_EigenSolver<TConfig>::solver_setup()
{
    Operator<TConfig> &A = *this->m_A;
    ViewType oldView = A.currentView();
    A.setViewExterior();
    free_allocated();
    int max_iters = this->m_max_iters;
    m_diagonal.resize(max_iters);
    m_subdiagonal.resize(max_iters);
    m_diagonal_tmp.resize(max_iters);
    m_subdiagonal_tmp.resize(max_iters);
    m_ritz_eigenvectors.resize(max_iters * max_iters);
    m_ritz_eigenvectors.set_lda(max_iters);
    int dwork_size = (3 * max_iters * max_iters) / 2 + 3 * max_iters;
    m_dwork.resize(dwork_size);
    // We need three storage vectors: the current vector and the two previous ones.
    const int N = static_cast<int>(A.get_num_cols() * A.get_block_dimy());
    m_v.resize(N);
    m_v_prev.resize(N);
    m_w.resize(N);
    std::vector<VVector *> allocated_vectors;
    allocated_vectors.push_back(&m_v);
    allocated_vectors.push_back(&m_v_prev);
    allocated_vectors.push_back(&m_w);
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

    A.setView(oldView);
}
template <class TConfig>
void Lanczos_EigenSolver<TConfig>::solver_pagerank_setup(VVector &a)
{
    if (this->m_which == EIG_PAGERANK)
    {
        printf("FATAL ERROR : PageRank is only supported in POWER_ITERATION / SINGLE_ITERATION solvers \n");
    }
}
template <class TConfig>
void Lanczos_EigenSolver<TConfig>::solve_init(VVector &x)
{
    Operator<TConfig> &A = *this->m_A;
    ViewType oldView = A.currentView();
    A.setViewExterior();
    int offset, size;
    A.getOffsetAndSizeForView(A.getViewExterior(), &offset, &size);
    // We copy the input vector since we need halo space allocated for m_v.
    copy(x, m_v, offset, size);
    // v = x / norm(X)
    scal(m_v, ValueTypeVec(1) / get_norm(A, m_v, L2), offset, size);
    // v_prev is initialized to zero.
    fill(m_v_prev, ValueTypeVec(0), offset, size);
    // beta is initialized to zero.
    m_beta = ValueTypeVec(0);
    A.setView(oldView);
}

template <class TConfig>
bool Lanczos_EigenSolver<TConfig>::solve_iteration(VVector &x)
{
    Operator<TConfig> &A = *this->m_A;
    ViewType oldView = A.currentView();
    A.setViewExterior();
    int offset, size;
    A.getOffsetAndSizeForView(A.getViewExterior(), &offset, &size);
    this->m_eigenvalues.clear();
    int i = this->m_curr_iter;
    int subspace_size = i + 1;
    // w = A * v
    //multiply(A, m_v, m_w);
    A.apply(m_v, m_w);
    // alpha = w' * v
    ValueTypeVec alpha = dot(A, m_w, m_v);
    // T(i, i) = alpha
    (m_diagonal)[i] = alpha;

    if (i > 0)
    {
        // T(i - 1, i) = T(i, i - 1) = beta
        (m_subdiagonal)[i - 1] = m_beta;
    }

    // w = w - alpha * v - beta * v_prev
    axpbypcz(m_w, m_v, m_v_prev, m_w, ValueTypeVec(1), -alpha, -m_beta, offset, size);
    // beta = norm(w)
    m_beta = get_norm(A, m_w, L2);
    // Check convergence of the method by approximating the residual of the Ritz pair.
    // See http://web.eecs.utk.edu/~dongarra/etemplates/node103.html.
    // MATLAB code to approximate the residuals:
    // [eigenvec, eigenval] = eigs(T, 1);
    // residual_norm = abs(beta * eigenvec(end))
    // Copy diagonal to a temporary array since input arrays are modified by MAGMA.
    copy(m_diagonal, m_diagonal_tmp, 0, subspace_size);
    copy(m_subdiagonal, m_subdiagonal_tmp, 0, subspace_size);
    // Compute eigenvalues and eigenvectors of symmetric tridiagonal matrix.
    Lapack<TConfig_h>::stedx(m_diagonal_tmp, m_subdiagonal_tmp, m_ritz_eigenvectors, subspace_size, m_dwork);
    // Get last component of largest Ritz eigenvector.
    ValueTypeVec last_ritz_vector = m_ritz_eigenvectors[i * m_ritz_eigenvectors.get_lda() + i];
    ValueTypeVec residual_norm = std::abs(last_ritz_vector * m_beta);
    ValueTypeVec lambda = m_diagonal_tmp[i];
    this->m_residuals.push_back(residual_norm / std::abs(lambda));

    for (int iter = i; iter >= 0; iter--)
    {
        this->m_eigenvalues.push_back(m_diagonal_tmp[iter]);
    }

    if (residual_norm < this->m_tolerance * std::abs(lambda))
    {
        //this->m_eigenvalues.push_back(lambda);
        return true;
    }

    // We need to update the previous vectors, we don't move data, instead we only change the pointers.
    VVector saved_v_prev = m_v_prev;
    m_v_prev = m_v;
    // v = w / beta
    m_v = m_w;
    scal(m_v, ValueTypeVec(1) / m_beta, offset, size);
    m_w = saved_v_prev;
    A.setView(oldView);
    return false;
}

template <class TConfig>
void Lanczos_EigenSolver<TConfig>::solve_finalize()
{
}

// Explicit template instantiation.
template class Lanczos_EigenSolver<TConfigGeneric_d>;
template class Lanczos_EigenSolver<TConfigGeneric_h>;

};

