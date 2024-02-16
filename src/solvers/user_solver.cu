// SPDX-FileCopyrightText: 2011 - 2024 NVIDIA CORPORATION. All Rights Reserved.
//
// SPDX-License-Identifier: BSD-3-Clause

#include <solvers/user_solver.h>
#include <cassert>

namespace amgx
{

template<class T_Config>
void
User_Solver<T_Config>::solver_setup(bool reuse_matrix_structure)
{
}

//launches a single standard cycle
template<class T_Config>
AMGX_STATUS
User_Solver<T_Config>::solve_iteration( VVector &b, VVector &x, bool xIsZero )
{
    assert( callback );
    callback( *this->m_A, b, x );
    return this->converged( b, x );
};

/****************************************
 * Explict instantiations
 ***************************************/
#define AMGX_CASE_LINE(CASE) template class User_Solver<TemplateMode<CASE>::Type>;
AMGX_FORALL_BUILDS(AMGX_CASE_LINE)
#undef AMGX_CASE_LINE

} // namespace amgx
