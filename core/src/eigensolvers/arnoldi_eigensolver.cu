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

#include <eigensolvers/arnoldi_eigensolver.h>
#include <blas.h>
#include <multiply.h>
#include <norm.h>
#include <amgx_lapack.h>

namespace amgx
{

template <class TConfig>
Arnoldi_EigenSolver<TConfig>::Arnoldi_EigenSolver(AMG_Config &cfg, const std::string &cfg_scope)
    : Base(cfg, cfg_scope)
{
    if (this->m_which != EIG_LARGEST)
    {
        FatalError("Arnoldi: can only compute largest eigenpair.", AMGX_ERR_CONFIGURATION);
    }
}

template <class TConfig>
Arnoldi_EigenSolver<TConfig>::~Arnoldi_EigenSolver()
{
    free_allocated();
}

template <class TConfig>
void Arnoldi_EigenSolver<TConfig>::free_allocated()
{
    for (int i = 0; i < m_V_vectors.size(); ++i)
    {
        delete m_V_vectors[i];
    }

    m_V_vectors.clear();
}

template <class TConfig>
void Arnoldi_EigenSolver<TConfig>::solver_setup()
{
    Operator<TConfig> &A = *this->m_A;
    ViewType oldView = A.currentView();
    free_allocated();
    m_krylov_size = this->m_max_iters;
    m_ritz_eigenvalues.resize(m_krylov_size);
    m_ritz_eigenvectors.resize(m_krylov_size * m_krylov_size);
    m_ritz_eigenvectors.set_lda(m_krylov_size);
    m_H.resize(m_krylov_size * m_krylov_size);
    m_H.set_lda(m_krylov_size);
    m_H.set_num_rows(0);
    m_H.set_num_cols(0);
    m_H_tmp.resize(m_krylov_size * m_krylov_size);
    m_H_tmp.set_lda(m_krylov_size);
    m_H_tmp.set_num_rows(0);
    m_H_tmp.set_num_cols(0);
    m_V_vectors.resize(m_krylov_size + 1, 0);
    int N = A.get_num_cols() * A.get_block_dimy();

    // Allocate memory needed for iterating.
    for (int i = 0; i < m_V_vectors.size(); ++i)
    {
        m_V_vectors[i] = new VVector(N);
        m_V_vectors[i]->set_block_dimy(A.get_block_dimy());
        m_V_vectors[i]->set_block_dimx(1);
        m_V_vectors[i]->dirtybit = 1;
        m_V_vectors[i]->delayed_send = 1;
        m_V_vectors[i]->tag = i;
    }

    A.setView(oldView);
}
template <class TConfig>
void Arnoldi_EigenSolver<TConfig>::solver_pagerank_setup(VVector &a)
{
    if (this->m_which == EIG_PAGERANK)
    {
        printf("FATAL ERROR : PageRank is only supported in POWER_ITERATION / SINGLE_ITERATION solvers \n");
    }
}
template <class TConfig>
void Arnoldi_EigenSolver<TConfig>::solve_init(VVector &x)
{
    Operator<TConfig> &A = *this->m_A;
    ViewType oldView = A.currentView();
    A.setViewExterior();
    int offset, size;
    A.getOffsetAndSizeForView(A.getViewExterior(), &offset, &size);
    copy(x, *m_V_vectors[0], offset, size);
    ValueTypeVec inv_norm = ValueTypeVec(1.) / get_norm(A, *m_V_vectors[0], L2);
    scal(*m_V_vectors[0], inv_norm, offset, size);
    A.setView(oldView);
}

template <class TConfig>
bool Arnoldi_EigenSolver<TConfig>::solve_iteration(VVector &x)
{
    Operator<TConfig> &A = *this->m_A;
    ViewType oldView = A.currentView();
    A.setViewExterior();
    int offset, size;
    A.getOffsetAndSizeForView(A.getViewExterior(), &offset, &size);
    int i = this->m_curr_iter;
    // V(i + 1) = A * V(i)
    //multiply(A, *m_V_vectors[i], *m_V_vectors[i + 1]);
    A.apply(*m_V_vectors[i], *m_V_vectors[i + 1]);

    for (int k = 0; k <= i; ++k)
    {
        // H(k, i) = <V(i + 1), V(k)>
        ValueTypeVec Hki = dot(A, *m_V_vectors[i + 1], *m_V_vectors[k]);
        m_H[i * m_H.get_lda() + k] = Hki;
        // V(i + 1) -= H(k, i) * V(k)
        axpy(*m_V_vectors[k], *m_V_vectors[i + 1], -Hki, offset, size);
    }

    if (i > 0)
    {
        m_H[(i - 1) * m_H.get_lda() + i] = m_beta;
    }

    m_H.set_num_rows(i + 1);
    m_H.set_num_cols(i + 1);
    // beta = norm(V(i + 1))
    m_beta = get_norm(A, *m_V_vectors[i + 1], L2);
    // Check convergence of the method by approximating the residual of the Ritz pair.
    // See http://web.eecs.utk.edu/~dongarra/etemplates/node216.html
    // MATLAB code to approximate the residuals:
    // [eigenvec, eigenval] = eigs(H, 1);
    // residual_norm = abs(beta * eigenvec(end))
    copy(m_H, m_H_tmp);
    m_H_tmp.set_num_rows(i + 1);
    m_H_tmp.set_num_cols(i + 1);
    Lapack<TConfig_h>::geev(m_H_tmp, m_ritz_eigenvalues, m_ritz_eigenvectors);
    int index_max = 0;
    ValueTypeVec lambda = m_ritz_eigenvalues[0];

    for (int j = 0; j < i + 1; ++j)
    {
        if (lambda < m_ritz_eigenvalues[j])
        {
            index_max = j;
            lambda = m_ritz_eigenvalues[j];
        }
    }

    m_ritz_eigenvectors.set_num_rows(m_krylov_size);
    m_ritz_eigenvectors.set_num_cols(m_krylov_size);
    // Get last component of largest Ritz eigenvector.
    ValueTypeVec last_ritz_vector = m_ritz_eigenvectors[index_max * m_ritz_eigenvectors.get_lda() + i];
    ValueTypeVec residual_norm = std::abs(last_ritz_vector * m_beta);
    this->m_residuals.push_back(residual_norm / std::abs(lambda));

    if (residual_norm < this->m_tolerance * std::abs(lambda))
    {
        this->m_eigenvalues.push_back(lambda);
        return true;
    }

    // V(i + 1) = V(i + 1) / beta
    scal(*m_V_vectors[i + 1], ValueTypeVec(1) / m_beta, offset, size);
    A.setView(oldView);
    return false;
}

template <class TConfig>
void Arnoldi_EigenSolver<TConfig>::solve_finalize()
{
}

// Explicit template instantiation.
#define AMGX_CASE_LINE(CASE) template class Arnoldi_EigenSolver<TemplateMode<CASE>::Type>;
AMGX_FORALL_BUILDS(AMGX_CASE_LINE)
#undef AMGX_CASE_LINE

};
