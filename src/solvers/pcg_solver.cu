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

#include <solvers/pcg_solver.h>
#include <blas.h>
#include <util.h>

namespace amgx
{

// Constructor
template< class T_Config>
PCG_Solver<T_Config>::PCG_Solver( AMG_Config &cfg, const std::string &cfg_scope) :
    Solver<T_Config>( cfg, cfg_scope),
    m_buffer_N(0)
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
PCG_Solver<T_Config>::~PCG_Solver()
{
    if (!no_preconditioner) { delete m_preconditioner; }
}

template<class T_Config>
void
PCG_Solver<T_Config>::solver_setup(bool reuse_matrix_structure)
{
    AMGX_CPU_PROFILER( "PCG_Solver::solver_setup " );
    ViewType oldView = this->m_A->currentView();
    this->m_A->setViewExterior();

    if (!no_preconditioner)
    {
        m_preconditioner->setup(*this->m_A, reuse_matrix_structure);
    }

    // The number of elements in temporary vectors.
    this->m_buffer_N = static_cast<int>( this->m_A->get_num_cols() * this->m_A->get_block_dimy() );
    // Allocate memory needed for iterating.
    m_p.resize( this->m_buffer_N );
    m_z.resize( this->m_buffer_N );
    m_Ap.resize( this->m_buffer_N );
    m_p.set_block_dimy(this->m_A->get_block_dimy());
    m_p.set_block_dimx(1);
    m_p.dirtybit = 1;
    m_p.delayed_send = 1;
    m_p.tag = this->tag * 100 + 1;
    m_Ap.set_block_dimy(this->m_A->get_block_dimy());
    m_Ap.set_block_dimx(1);
    m_Ap.dirtybit = 1;
    m_Ap.delayed_send = 1;
    m_Ap.tag = this->tag * 100 + 2;
    m_z.set_block_dimy(this->m_A->get_block_dimy());
    m_z.set_block_dimx(1);
    m_z.dirtybit = 1;
    m_z.delayed_send = 1;
    m_z.tag = this->tag * 100 + 3;
    // Setup the preconditionner
    this->m_A->setView(oldView);
}

template<class T_Config>
void
PCG_Solver<T_Config>::solve_init( VVector &b, VVector &x, bool xIsZero )
{
    AMGX_CPU_PROFILER( "PCG_Solver::solve_init " );
    Operator<T_Config> &A = *this->m_A;
    ViewType oldView = A.currentView();
    A.setViewExterior();
    int offset, size;
    A.getOffsetAndSizeForView(A.getViewExterior(), &offset, &size);

    // Run one iteration of preconditioner with zero initial guess
    if (no_preconditioner)
    {
        copy(*this->m_r, m_z, offset, size);
    }
    else
    {
        m_z.delayed_send = 1;
        this->m_r->delayed_send = 1;
        m_preconditioner->solve( *this->m_r, m_z, true );
        m_z.delayed_send = 1;
        this->m_r->delayed_send = 1;
    }

    copy( m_z, m_p, offset, size );
    m_r_z = dot(A, *this->m_r, m_z);
    A.setView(oldView);
}

template<class T_Config>
AMGX_STATUS
PCG_Solver<T_Config>::solve_iteration( VVector &b, VVector &x, bool xIsZero )
{
    AMGX_CPU_PROFILER( "PCG_Solver::solve_iteration " );

    AMGX_STATUS conv_stat = AMGX_ST_NOT_CONVERGED;

    Operator<T_Config> &A = *this->m_A;
    ViewType oldView = A.currentView();
    A.setViewExterior();
    int offset, size;
    A.getOffsetAndSizeForView(A.getViewExterior(), &offset, &size);
    // Ap = A * p. Krylov iteration.
    A.apply(m_p, m_Ap);
    /*
    //print
    ValueTypeB lnrm2p= dotc(m_p, m_p, offset, size);
    ValueTypeB nrm2p = dot(A, m_p, m_p );
    ValueTypeB lnrm2q= dotc(m_Ap, m_Ap, offset, size);
    ValueTypeB nrm2q = dot(A, m_Ap, m_Ap );
    printf("||p||^2=%f (%f), ||q||^2=%f (%f)\n",nrm2p,lnrm2p,nrm2q,lnrm2q);
    */
    // alpha = <r,z>/<y,p>
    ValueTypeB dot_App = dot(A, m_Ap, m_p );
    ValueTypeB alpha(0);

    if ( dot_App != ValueTypeB(0) )
    {
        alpha = m_r_z / dot_App;
    }

    // x = x + alpha * p.
    axpy( m_p, x, alpha, offset, size );
    // r = r - alpha * Ap.
    axpy( m_Ap, *this->m_r, -alpha, offset, size );

    // Do we converge ?
    if ( this->m_monitor_convergence &&
         isDone( ( conv_stat= this->compute_norm_and_converged() ) ) )
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

    // Run one iteration of preconditioner with zero initial guess
    if (no_preconditioner)
    {
        copy(*this->m_r, m_z, offset, size);
    }
    else
    {
        m_z.delayed_send = 1;
        this->m_r->delayed_send = 1;
        m_preconditioner->solve( *this->m_r, m_z, true );
        m_z.delayed_send = 1;
        this->m_r->delayed_send = 1;
    }

    // Store m_r_z.
    ValueTypeB rz_old = m_r_z;
    // rz = <r, z>.
    m_r_z = dot( A, *this->m_r, m_z );
    // beta <- <r_{i+1},z_{i+1}>/<r,z>
    ValueTypeB beta(0);

    if ( rz_old != ValueTypeB(0) )
    {
        beta = m_r_z / rz_old;
    }

    // p += z + beta*p
    axpby( m_z, m_p, m_p, ValueTypeB( 1 ), beta, offset, size);
    // No convergence so far.
    A.setView(oldView);
    return this->m_monitor_convergence ? AMGX_ST_NOT_CONVERGED : AMGX_ST_CONVERGED;
}

template<class T_Config>
void
PCG_Solver<T_Config>::solve_finalize( VVector &b, VVector &x )
{}

template<class T_Config>
void
PCG_Solver<T_Config>::printSolverParameters() const
{
    if (!no_preconditioner)
    {
        std::cout << "preconditioner: " << this->m_preconditioner->getName()
                  << " with scope name: "
                  << this->m_preconditioner->getScope() << std::endl;
    }
}

/****************************************
 * Explict instantiations
 ***************************************/
#define AMGX_CASE_LINE(CASE) template class PCG_Solver<TemplateMode<CASE>::Type>;
AMGX_FORALL_BUILDS(AMGX_CASE_LINE)
#undef AMGX_CASE_LINE

} // namespace amgx
