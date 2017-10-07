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
bool
CG_Solver<T_Config>::solve_iteration( VVector &b, VVector &x, bool xIsZero )
{
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
    if ( this->m_monitor_convergence && this->compute_norm_and_converged() )
    {
        A.setView(oldView);
        return true;
    }

    // Early exit: last iteration, no need to prepare the next one.
    if ( this->is_last_iter() )
    {
        A.setView(oldView);
        return !this->m_monitor_convergence;
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
    return !this->m_monitor_convergence;
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
