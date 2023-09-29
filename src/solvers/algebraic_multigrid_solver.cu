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

#include <amg.h>
#include <solvers/solver.h>
#include <solvers/algebraic_multigrid_solver.h>
#include <util.h>

namespace amgx
{

template<class T_Config>
AlgebraicMultigrid_Solver<T_Config>::AlgebraicMultigrid_Solver( AMG_Config &cfg, const std::string &cfg_scope, ThreadManager *tmng ):
    Solver<T_Config>(cfg, cfg_scope, tmng),
    m_amg(cfg, cfg_scope)
{
    SetThreadManager(tmng);
    m_amg.allocate_fine_level();
}

template<class T_Config>
void
AlgebraicMultigrid_Solver<T_Config>::printSolverParameters() const
{
    m_amg.printSettings();
}

template<class T_Config>
AlgebraicMultigrid_Solver<T_Config>::~AlgebraicMultigrid_Solver()
{
}

template <class T_Config>
void AlgebraicMultigrid_Solver<T_Config >::solver_setup(bool reuse_matrix_structure)
{
    m_A = dynamic_cast<Matrix<T_Config>*>(Base::m_A);

    if (!m_A)
    {
        FatalError("AlgebraicMultigrid only works with explicit matrices", AMGX_ERR_INTERNAL);
    }

    m_amg.setup(*m_A);
}

template<class T_Config>
void
AlgebraicMultigrid_Solver<T_Config>::solve_init( VVector &b, VVector &x, bool xIsZero )
{
    m_amg.solve_init( b, x, xIsZero );
}

template<class T_Config>
bool
AlgebraicMultigrid_Solver<T_Config>::solve_iteration( VVector &b, VVector &x, bool xIsZero )
{
    m_amg.solve_iteration( b, x );
    return this->converged( b, x );
}

template<class T_Config>
void
AlgebraicMultigrid_Solver<T_Config>::solve_finalize( VVector &b, VVector &x )
{}

template<class T_Config>
void
AlgebraicMultigrid_Solver<T_Config>::print_grid_stats()
{
    m_amg.printGridStatistics();
}

template<class T_Config>
void
AlgebraicMultigrid_Solver<T_Config>::print_grid_stats2()
{
    m_amg.printGridStatistics2();
}

template<class T_Config>
void
AlgebraicMultigrid_Solver<T_Config>::print_vis_data()
{
    m_amg.printCoarsePoints();
    m_amg.printConnections();
}

/****************************************
* Explict instantiations
***************************************/
#define AMGX_CASE_LINE(CASE) template class AlgebraicMultigrid_Solver<TemplateMode<CASE>::Type>;
AMGX_FORALL_BUILDS(AMGX_CASE_LINE)
AMGX_FORCOMPLEX_BUILDS(AMGX_CASE_LINE)
#undef AMGX_CASE_LINE

} // namespace amgx
