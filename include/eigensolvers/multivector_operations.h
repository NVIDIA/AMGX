// SPDX-FileCopyrightText: 2013 - 2025 NVIDIA CORPORATION. All Rights Reserved.
//
// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#include <matrix.h>
#include <vector.h>

namespace amgx
{


// res = alpha * lhs * rhs + beta * res.
template <typename TConfig>
void
distributed_gemm_TN(typename TConfig::VecPrec alpha, const Vector<TConfig> &lhs,
                    const Vector<TConfig> &rhs,
                    typename TConfig::VecPrec beta, Vector<TConfig> &res,
                    const Operator<TConfig> &A);

template <typename TConfig>
void
multivector_column_norms(const Vector<TConfig> &v,
                         Vector<typename TConfig::template setMemSpace<AMGX_host>::Type> &results,
                         const Operator<TConfig> &A);

}
