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

#include <solvers/bicgstab_solver.h>
#include <blas.h>
#include <cusp/blas.h>

namespace amgx
{

template<class T_Config>
BiCGStab_Solver<T_Config>::BiCGStab_Solver( AMG_Config &cfg, const std::string &cfg_scope ) :
    Solver<T_Config>( cfg, cfg_scope ),
    m_rho(0),
    m_p(0),
    m_s(0),
    m_t(0),
    m_v(0),
    m_r_tilde(0),
    m_s_norm()
{}

template<class T_Config>
BiCGStab_Solver<T_Config>::~BiCGStab_Solver()
{}

template<class T_Config>
void
BiCGStab_Solver<T_Config>::solver_setup(bool reuse_matrix_structure)
{
    ViewType oldView = this->m_A->currentView();
    this->m_A->setView(OWNED);
    // The number of elements in temporary vectors.
    const int N = static_cast<int>( this->m_A->get_num_cols() * this->m_A->get_block_dimy() );
    const int N_owned = static_cast<int>( this->m_A->get_num_rows() * this->m_A->get_block_dimy() );
    // The size of temporary vectors if they have already been allocated.
    const int old_N = m_p.size();

    // Allocate memory needed for iterating.
    if ( (int) m_p.size() != N )
    {
        m_p.resize(N);
        m_s.resize(N);
        m_t.resize(N_owned); //this has no halo
        m_v.resize(N_owned); //this has no halo
        m_r_tilde.resize(N_owned); //this has no halo
    }

    m_p.tag = this->tag * 100 + 3;
    m_s.tag = this->tag * 100 + 4;
    m_p.set_block_dimy(this->m_A->get_block_dimy());
    m_p.set_block_dimx(1);
    m_s.set_block_dimy(this->m_A->get_block_dimy());
    m_s.set_block_dimx(1);
    m_t.set_block_dimy(this->m_A->get_block_dimy());
    m_t.set_block_dimx(1);
    m_v.set_block_dimy(this->m_A->get_block_dimy());
    m_v.set_block_dimx(1);
    m_r_tilde.set_block_dimy(this->m_A->get_block_dimy());
    m_r_tilde.set_block_dimx(1);
    // Make sure s_nrm has the right size.
    m_s_norm.resize( this->m_A->get_block_dimy() );
    this->m_A->setView(oldView);
}

template<class T_Config>
void
BiCGStab_Solver<T_Config>::solve_init(  VVector &b, VVector &x, bool xIsZero )
{
    int size, offset;
    this->m_A->getOffsetAndSizeForView(OWNED, &offset, &size);

    // Make sure the residual has previously been computed.
    //assert( this->m_r && this->m_r->size() == x.size() ); //This is no longer true, the two sizes may not be equal
    if ( ( this->m_r == NULL ) || ( this->m_r->size() != m_t.size() ) )
    {
        FatalError("Make sure the residual has previously been computed.", AMGX_ERR_UNKNOWN);
    }

    // r_tilde = r. TODO: use a random vector as r_star.
    amgx::thrust::copy(this->m_r->begin(), this->m_r->end(), m_r_tilde.begin()); //these guys should have the same size (OWNED)
    cudaCheckError();
    // Set rho to <r, r_tilde>.
    m_rho = dot(*this->m_A, m_r_tilde, *this->m_r);
    // p = r.
    amgx::thrust::copy(this->m_r->begin(), this->m_r->end(), m_p.begin());
    cudaCheckError();
    m_p.dirtybit = 1;
}

//launches a single standard cycle
template<class T_Config>
bool
BiCGStab_Solver<T_Config>::solve_iteration( VVector &b, VVector &x, bool xIsZero )
{
    Operator<T_Config> &A = *this->m_A;
    ViewType oldView = A.currentView();
    A.setViewExterior();
    int size, offset;
    A.getOffsetAndSizeForView(OWNED, &offset, &size);
    // v = Ap.
    A.apply(m_p, m_v);
    // Update alpha: alpha = rho / dot(r_tilde, v).
    ValueTypeB denom = dot(A, m_r_tilde, m_v);
    ValueTypeB alpha = m_rho / denom;
    // Compute s = r - alpha*v.
    axpby( *this->m_r, m_v, m_s, ValueTypeB(1), -alpha, offset, size );
    m_s.dirtybit = 1;

    // Early exit if norm(s) is small enough...
    if ( this->m_monitor_convergence && this->compute_norm_and_converged( m_s, m_s_norm ) )
    {
        axpby( x, m_p, x, ValueTypeB(1), alpha, offset, size );
        x.dirtybit = 1;
        this->compute_residual( b, x );
        this->compute_norm();
        return true;
    }

    // t = As.
    A.apply( m_s, m_t );
    // Update omega: omega = <t,s> / <t,t>.
    ValueTypeB nom = dot(A, m_t, m_s);
    denom = dot(A, m_t, m_t);
    /*if (A.manager.neighbors.size()>0) {
      //global reduction
      reduce_own[0] = nom;
      reduce_own[1] = denom;
      A.manager.comms->global_reduce(reduce, reduce_own, A, 1);
      denom = 0;
      nom = 0;
      for (int j = 0; j < values.size(); j++) {
        nom += reduce[j][0];
        denom += reduce[j][1];
      }
    }*/
    ValueTypeB omega = nom / denom;
    // Update x: x = x + alpha*p + omega*s.
    axpbypcz( x, m_p, m_s, x, ValueTypeB(1), alpha, omega, offset, size );
    x.dirtybit = 1;
    // Update r: r = s - omega*t.
    axpby( m_s, m_t, *this->m_r, ValueTypeB(1), -omega, offset, size );
    this->m_r->dirtybit = 1;

    // Do we converge ?
    if ( this->m_monitor_convergence && this->compute_norm_and_converged() )
    {
        return true;
    }

    // Early exit: last iteration, no need to prepare the next one.
    if ( this->is_last_iter() )
    {
        return !this->m_monitor_convergence;
    }

    // Prepare next iteration: Update beta and rho.
    ValueTypeB rho_new = dot(A, m_r_tilde, *this->m_r);
    ValueTypeB beta = (rho_new / m_rho) * (alpha / omega);
    m_rho = rho_new;
    // Update p: p = r + beta*p - beta*omega*v.
    axpbypcz( *this->m_r, m_p, m_v, m_p, ValueTypeB(1), beta, -beta * omega, offset, size );
    m_p.dirtybit = 1;
    A.setView(oldView);
    // Return.
    return !this->m_monitor_convergence;
}

template<class T_Config>
void
BiCGStab_Solver<T_Config>::solve_finalize( VVector &b, VVector &x )
{}

/****************************************
 * Explict instantiations
 ***************************************/
#define AMGX_CASE_LINE(CASE) template class BiCGStab_Solver<TemplateMode<CASE>::Type>;
AMGX_FORALL_BUILDS(AMGX_CASE_LINE)
#undef AMGX_CASE_LINE

} // namespace amgx
