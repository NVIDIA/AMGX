// SPDX-FileCopyrightText: 2013 - 2024 NVIDIA CORPORATION. All Rights Reserved.
//
// SPDX-License-Identifier: BSD-3-Clause

#include <solvers/cg_solver.h>
#include <blas.h>

namespace amgx
{


// Destructor
template<class T_Config>
CG_Solver<T_Config>::~CG_Solver()
{}

template<class T_Config>
void
CG_Solver<T_Config>::solver_setup(bool reuse_matrix_structure)
{
    ViewType oldView = this->m_A->currentView();
    this->m_A->setViewExterior();
    const int N = static_cast<int>( this->m_A->get_num_cols() * this->m_A->get_block_dimy() );
    // Allocate memory needed for iterating.
    m_p .resize(N);
    m_Ap.resize(N);
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
    this->m_A->setView(oldView);
}

template<class T_Config>
void
CG_Solver<T_Config>::solve_init( VVector &b, VVector &x, bool xIsZero )
{
    Operator<T_Config> &A = *this->m_A;
    ViewType oldView = A.currentView();
    A.setViewExterior();
    int offset, size;
    A.getOffsetAndSizeForView(A.getViewExterior(), &offset, &size);
    assert( this->m_r && this->m_r->size() == x.size() );
    copy( *this->m_r, m_p, offset, size );
    m_r_r = dot(*this->m_A, *this->m_r, *this->m_r);
    A.setView(oldView);
}

// launches a single standard cycle
template<class T_Config>
AMGX_STATUS
CG_Solver<T_Config>::solve_iteration( VVector &b, VVector &x, bool xIsZero )
{
    AMGX_STATUS conv_stat = AMGX_ST_NOT_CONVERGED;

    Operator<T_Config> &A = *this->m_A;
    ViewType oldView = A.currentView();
    A.setViewExterior();
    int offset, size;
    A.getOffsetAndSizeForView(A.getViewExterior(), &offset, &size);
    // Ap = A * p. Krylov iteration.
    A.apply(m_p, m_Ap);
    // alpha = <r,z>/<y,p>
    ValueTypeB alpha =  m_r_r / dot( A, m_Ap, m_p );
    // x = x + alpha * p.
    axpy( m_p, x, alpha, offset, size );
    // r = r - alpha * Ap.
    axpy( m_Ap, *this->m_r, -alpha, offset, size );

    // Do we converge ?
    if ( this->m_monitor_convergence &&
         isDone( (conv_stat = this->compute_norm_and_converged()) ) )
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

    ValueTypeB rr_old = m_r_r;
    // rz = <r, z>
    m_r_r = dot( A, *this->m_r, *this->m_r );
    // beta <- <r_{i+1},z_{i+1}>/<r,z>
    ValueTypeB beta = m_r_r / rr_old;
    // p += r + beta*p
    axpby( *this->m_r, m_p, m_p, ValueTypeB(1), beta, offset, size);
    // No convergence so far.
    A.setView(oldView);
    return this->m_monitor_convergence ? AMGX_ST_NOT_CONVERGED : AMGX_ST_CONVERGED;
}

template<class T_Config>
void
CG_Solver<T_Config>::solve_finalize( VVector &b, VVector &x )
{}

/****************************************
 * Explict instantiations
 ***************************************/
#define AMGX_CASE_LINE(CASE) template class CG_Solver<TemplateMode<CASE>::Type>;
AMGX_FORALL_BUILDS(AMGX_CASE_LINE)
#undef AMGX_CASE_LINE

} // namespace amgx
