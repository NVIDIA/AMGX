// SPDX-FileCopyrightText: 2011 - 2025 NVIDIA CORPORATION. All Rights Reserved.
//
// SPDX-License-Identifier: BSD-3-Clause

#include <solvers/dummy_solver.h>
#include <cassert>

#include "amgx_types/util.h"

namespace amgx
{

template<class T_Config>
void
Dummy_Solver<T_Config>::solver_setup(bool reuse_matrix_structure)
{
}

//launches a single standard cycle
template<class T_Config>
AMGX_STATUS
Dummy_Solver<T_Config>::solve_iteration( VVector &b, VVector &x, bool xIsZero )
{
    if (xIsZero)
    {
        thrust_wrapper::fill<T_Config::memSpace>(x.begin(), x.end(), types::util<ValueTypeB>::get_zero());
        cudaCheckError();
    }

    return (this->converged(b, x));
};

/****************************************
 * Explict instantiations
 ***************************************/
#define AMGX_CASE_LINE(CASE) template class Dummy_Solver<TemplateMode<CASE>::Type>;
AMGX_FORALL_BUILDS(AMGX_CASE_LINE)
AMGX_FORCOMPLEX_BUILDS(AMGX_CASE_LINE)
#undef AMGX_CASE_LINE

} // namespace amgx
