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

#include <solvers/pbicgstab_solver.h>
#include <blas.h>
#include <memory_info.h>

namespace amgx
{

template<class T_Config>
PBiCGStab_Solver<T_Config>::PBiCGStab_Solver( AMG_Config &cfg, const std::string &cfg_scope ) :
    Solver<T_Config>( cfg, cfg_scope ),
    m_rho(0)
{
    std::string solverName, new_scope, tmp_scope;
    cfg.getParameter<std::string>( "preconditioner", solverName, cfg_scope, new_scope );

    if (solverName.compare("NOSOLVER") == 0)
    {
        no_preconditioner = true;
        m_preconditioner = NULL;
    }
    else
    {
        no_preconditioner = false;
        m_preconditioner = SolverFactory<T_Config>::allocate( cfg, cfg_scope, "preconditioner" );
    }
}

template<class T_Config>
PBiCGStab_Solver<T_Config>::~PBiCGStab_Solver()
{
    /*
    if (this->m_A != NULL) {
      ViewType oldView = this->m_A->currentView();
      this->m_A->setViewExterior();
      this->m_A->setView(oldView);
    }
    */
    if (!no_preconditioner) { delete m_preconditioner; }
}

template<class T_Config>
void
PBiCGStab_Solver<T_Config>::printSolverParameters() const
{
    if (!no_preconditioner)
    {
        std::cout << "preconditioner: "
                  << this->m_preconditioner->getName()
                  << " with scope name: "
                  << this->m_preconditioner->getScope()
                  << std::endl;
    }
}

template<class T_Config>
void
PBiCGStab_Solver<T_Config>::solver_setup(bool reuse_matrix_structure)
{
    ViewType oldView = this->m_A->currentView();
    this->m_A->setViewExterior();
    // The number of elements in temporary vectors.
    //const int N_owned = static_cast<int>( this->m_A->get_num_rows() * this->m_A->get_block_dimy() );
    const int N_owned = static_cast<int>( this->m_A->get_num_cols() * this->m_A->get_block_dimy() );
    const int N_all = static_cast<int>( this->m_A->get_num_cols() * this->m_A->get_block_dimy() );
    // Allocate memory needed for iterating.
    m_p.resize(N_all);
    m_Mp.resize(N_all);
    m_s.resize(N_all);
    m_Ms.resize(N_all);
    m_t.resize(N_owned);
    m_v.resize(N_owned);
    m_r_tilde.resize(N_owned);
    m_p.set_block_dimy(this->m_A->get_block_dimy());
    m_p.set_block_dimx(1);
    m_p.dirtybit = 1;
    m_p.delayed_send = 1;
    m_p.tag = this->tag * 100 + 3;
    m_Mp.set_block_dimy(this->m_A->get_block_dimy());
    m_Mp.set_block_dimx(1);
    m_Mp.dirtybit = 1;
    m_Mp.delayed_send = 1;
    m_Mp.tag = this->tag * 100 + 4;
    m_s.set_block_dimy(this->m_A->get_block_dimy());
    m_s.set_block_dimx(1);
    m_s.dirtybit = 1;
    m_s.delayed_send = 1;
    m_s.tag = this->tag * 100 + 5;
    m_Ms.set_block_dimy(this->m_A->get_block_dimy());
    m_Ms.set_block_dimx(1);
    m_Ms.dirtybit = 1;
    m_Ms.delayed_send = 1;
    m_Ms.tag = this->tag * 100 + 6;
    m_t.set_block_dimy(this->m_A->get_block_dimy());
    m_t.set_block_dimx(1);
    m_t.dirtybit = 1;
    m_t.tag = this->tag * 100 + 7;
    m_v.set_block_dimy(this->m_A->get_block_dimy());
    m_v.set_block_dimx(1);
    m_v.dirtybit = 1;
    m_v.tag = this->tag * 100 + 8;
    m_r_tilde.set_block_dimy(this->m_A->get_block_dimy());
    m_r_tilde.set_block_dimx(1);
    m_r_tilde.dirtybit = 1;
    m_r_tilde.tag = this->tag * 100 + 9;
    // Make sure m_s_norm has the right size.
    m_s_norm.resize( this->m_A->get_block_dimy() );

    // Setup the preconditioner.
    if (!no_preconditioner)
    {
        m_preconditioner->setup( *this->m_A, reuse_matrix_structure );
    }

    this->m_A->setView(oldView);
}

template<class T_Config>
void
PBiCGStab_Solver<T_Config>::solve_init( VVector &b, VVector &x, bool xIsZero )
{
    int offset, size;
    this->m_A->getOffsetAndSizeForView(this->m_A->getViewExterior(), &offset, &size);
    // Make sure the residual has previously been computed.
    assert( (this->m_r) && (this->m_r->size() == size * this->m_r->get_block_size()) );
    // r_tilde = r. TODO: use a random vector as r_star.
    copy( *this->m_r, m_r_tilde, offset, size );
    // Set rho to <r, r_tilde>.
    m_rho = dot(*this->m_A, m_r_tilde, *this->m_r);
    // p = r.
    copy( *this->m_r, m_p, offset, size );
}

//launches a single standard cycle
template<class T_Config>
AMGX_STATUS
PBiCGStab_Solver<T_Config>::solve_iteration( VVector &b, VVector &x, bool xIsZero )
{
    AMGX_STATUS conv_stat = AMGX_ST_NOT_CONVERGED;

    Operator<T_Config> &A = *this->m_A;
    ViewType oldView = A.currentView();
    A.setViewExterior();
    int offset, size;
    A.getOffsetAndSizeForView(A.getViewExterior(), &offset, &size);

    //printf("-1. pass in: %f\n", reduction1);

    //  A.setView(oldView);
    if (no_preconditioner)
    {
        copy(m_p, m_Mp, offset, size);
    }
    else
    {
        m_p.delayed_send = 1;
        m_Mp.delayed_send = 1;
        m_preconditioner->solve( m_p, m_Mp, true );
        m_p.delayed_send = 1;
        m_Mp.delayed_send = 1;
    }

    //  A.setViewExterior();
    // v = AMp.
    A.apply( m_Mp, m_v );
    // Update alpha: alpha = rho / dot(r_tilde, v).
    ValueTypeB reduction = dot(A, m_r_tilde, m_v);
    ValueTypeB alpha(0);

    if ( reduction != ValueTypeB(0) )
    {
        alpha = m_rho / reduction;
    }

    //printf("0. m_rho: %f\n", m_rho);
    //printf("1. alpha: %f\n", alpha);
    // Compute s = r - alpha*v.
    axpby( *this->m_r, m_v, m_s, ValueTypeB(1), -alpha, offset, size );

    // Early exit if norm(s) is small enough...
    if ( this->m_monitor_convergence &&
         isDone( ( conv_stat = this->compute_norm_and_converged( m_s, m_s_norm ) ) ) )
    {
        axpby( x, m_Mp, x, ValueTypeB(1), alpha, offset, size );
        this->compute_residual( b, x );
        this->compute_norm();
        A.setView(oldView);
        return conv_stat;
    }

    //  A.setView(oldView);
    // Precondition s.
    if (no_preconditioner)
    {
        copy(m_s, m_Ms, offset, size);
    }
    else
    {
        m_s.delayed_send = 1;
        m_Ms.delayed_send = 1;
        m_preconditioner->solve( m_s, m_Ms, true );
        m_s.delayed_send = 1;
        m_Ms.delayed_send = 1;
    }

    //  A.setViewExterior();
    // t = AMs.
    A.apply( m_Ms, m_t );
    // Update omega: omega = <t,s> / <t,t>.
    reduction = dot(A, m_t, m_t);
    ValueTypeB omega = dot(A, m_t, m_s);

    if ( reduction == ValueTypeB(0) )
    {
        omega = ValueTypeB(0);
    }
    else
    {
        omega = omega / reduction;
    }

    //printf("2. omega: %f\n", omega);
    // Update x: x = x + alpha*Mp + omega*Ms.
    axpbypcz( x, m_Mp, m_Ms, x, ValueTypeB(1), alpha, omega, offset, size );
    // Update r: r = s - omega*t.
    axpby( m_s, m_t, *this->m_r, ValueTypeB(1), -omega, offset, size );

    // Do we converge ?
    if ( this->m_monitor_convergence &&
         isDone( ( conv_stat = this->compute_norm_and_converged() ) ) )
    {
        A.setView(oldView);
        return conv_stat;
    }

    // Early exit: last iteration, no need to prepare the next one.
    if ( this->is_last_iter() )
    {
        A.setView(oldView);
        return this->m_monitor_convergence ? AMGX_ST_NOT_CONVERGED : AMGX_ST_CONVERGED;
    }

    // Prepare next iteration: Update beta and rho.
    ValueTypeB rho_new = dot(A, m_r_tilde, *this->m_r);
    //printf("3. rho_new: %f\n", rho_new);
    ValueTypeB beta(0);

    if ( m_rho != ValueTypeB(0) && omega != ValueTypeB(0) )
    {
        beta = (rho_new / m_rho) * (alpha / omega);
    }

    m_rho = rho_new;
    //printf("4. beta: %f\n", beta);
    // Update p: p = r + beta*p - beta*omega*v.
    axpbypcz( *this->m_r, m_p, m_v, m_p, ValueTypeB(1), beta, -beta * omega, offset, size );
    // Return.
    A.setView(oldView);
    return this->m_monitor_convergence ? AMGX_ST_NOT_CONVERGED : AMGX_ST_CONVERGED;
}

template<class T_Config>
void
PBiCGStab_Solver<T_Config>::solve_finalize( VVector &b, VVector &x )
{}

/****************************************
* Explict instantiations
***************************************/
#define AMGX_CASE_LINE(CASE) template class PBiCGStab_Solver<TemplateMode<CASE>::Type>;
AMGX_FORALL_BUILDS(AMGX_CASE_LINE)
#undef AMGX_CASE_LINE

} // namespace amgx
