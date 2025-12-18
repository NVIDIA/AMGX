// SPDX-FileCopyrightText: 2013 - 2025 NVIDIA CORPORATION. All Rights Reserved.
//
// SPDX-License-Identifier: BSD-3-Clause

namespace amgx
{

template <class T_Config> class Operator;

}

#include <operators/deflated_multiply_operator.h>
#include <blas.h>

namespace amgx
{

template <typename TConfig>
void DeflatedMultiplyOperator<TConfig>::apply(const Vector<TConfig> &v, Vector<TConfig> &res, ViewType view)
{
    Operator<TConfig> &A = *m_A;
    int offset, size;
    A.getOffsetAndSizeForView(view, &offset, &size);
    copy(v, *m_work, offset, size);
    ValueTypeVec xtv = dot(A, *m_x, *m_work);
    axpy(*m_x, *m_work, types::util<ValueTypeVec>::invert(xtv), offset, size);
    A.apply(*m_work, res, OWNED);
    axpy(*m_work, res, types::util<ValueTypeVec>::invert(m_mu), offset, size);
    ValueTypeVec xtres = dot(A, *m_x, res);
    axpy(*m_x, res, types::util<ValueTypeVec>::invert(xtres), offset, size);
}

#define AMGX_CASE_LINE(CASE) template class DeflatedMultiplyOperator<TemplateMode<CASE>::Type>;
AMGX_FORALL_BUILDS(AMGX_CASE_LINE)
AMGX_FORCOMPLEX_BUILDS(AMGX_CASE_LINE)
#undef AMGX_CASE_LINE

}
