// SPDX-FileCopyrightText: 2013 - 2024 NVIDIA CORPORATION. All Rights Reserved.
//
// SPDX-License-Identifier: BSD-3-Clause

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
AMGX_STATUS
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
