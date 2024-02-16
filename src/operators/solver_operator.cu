// SPDX-FileCopyrightText: 2013 - 2024 NVIDIA CORPORATION. All Rights Reserved.
//
// SPDX-License-Identifier: BSD-3-Clause

namespace amgx
{

template <class T_Config> class Operator;

}

#include <operators/solver_operator.h>
#include <blas.h>

namespace amgx
{

#define AMGX_CASE_LINE(CASE) template class SolverOperator<TemplateMode<CASE>::Type>;
AMGX_FORALL_BUILDS(AMGX_CASE_LINE)
AMGX_FORCOMPLEX_BUILDS(AMGX_CASE_LINE)
#undef AMGX_CASE_LINE

}
