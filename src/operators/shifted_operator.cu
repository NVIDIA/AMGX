// SPDX-FileCopyrightText: 2013 - 2024 NVIDIA CORPORATION. All Rights Reserved.
//
// SPDX-License-Identifier: BSD-3-Clause

namespace amgx
{

template <class T_Config> class Operator;

}

#include <operators/shifted_operator.h>
#include <blas.h>

namespace amgx
{

template <typename TConfig>
void ShiftedOperator<TConfig>::apply(const Vector<TConfig> &v, Vector<TConfig> &res, ViewType view)
{
    Operator<TConfig> &A = *m_A;
    int offset, size;
    A.getOffsetAndSizeForView(view, &offset, &size);
    A.apply(v, res, OWNED);
    axpy(v, res, m_shift, offset, size);
}

#define AMGX_CASE_LINE(CASE) template class ShiftedOperator<TemplateMode<CASE>::Type>;
AMGX_FORALL_BUILDS(AMGX_CASE_LINE)
AMGX_FORCOMPLEX_BUILDS(AMGX_CASE_LINE)
#undef AMGX_CASE_LINE

}
