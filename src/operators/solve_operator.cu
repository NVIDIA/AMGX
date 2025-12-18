// SPDX-FileCopyrightText: 2013 - 2025 NVIDIA CORPORATION. All Rights Reserved.
//
// SPDX-License-Identifier: BSD-3-Clause

namespace amgx
{

template <class T_Config> class Operator;

}

#include <operators/solve_operator.h>
#include <solvers/solver.h>
#include <blas.h>

#include "amgx_types/util.h"

namespace amgx
{

template <typename TConfig>
SolveOperator<TConfig>::~SolveOperator()
{
    delete m_solver;
}

template <typename TConfig>
void SolveOperator<TConfig>::apply(const Vector<TConfig> &v, Vector<TConfig> &res, ViewType view)
{
    Operator<TConfig> &A = *this->m_A;
    Vector<TConfig> &v_ = const_cast<Vector<TConfig>&>(v);
    int offset, size;
    A.getOffsetAndSizeForView(view, &offset, &size);
    // Fill initial solution with 0s before solving.
    fill(res, types::util<typename Vector<TConfig>::value_type>::get_zero(), offset, size);
    AMGX_STATUS solve_status = m_solver->solve(v_, res, false);

    if (solve_status != AMGX_ST_CONVERGED)
    {
        FatalError("OperatorSolve: solver did not converge.", AMGX_ERR_CONFIGURATION);
    }
}

template <typename TConfig>
void SolveOperator<TConfig>::setup()
{
    assert(m_A);
    assert(m_solver);
    m_solver->setup(*m_A, false);
}

#define AMGX_CASE_LINE(CASE) template class SolveOperator<TemplateMode<CASE>::Type>;
AMGX_FORALL_BUILDS(AMGX_CASE_LINE)
AMGX_FORCOMPLEX_BUILDS(AMGX_CASE_LINE)
#undef AMGX_CASE_LINE

}
