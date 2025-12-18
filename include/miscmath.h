// SPDX-FileCopyrightText: 2011 - 2025 NVIDIA CORPORATION. All Rights Reserved.
//
// SPDX-License-Identifier: BSD-3-Clause

#pragma once

namespace amgx
{
template <class Matrix>  typename Matrix::value_type estimate_largest_eigen_value(Matrix &A);
}

#include <norm.h>
#include <multiply.h>
#include <blas.h>

namespace amgx
{

template <class Matrix>
typename Matrix::value_type estimate_largest_eigen_value(Matrix &A)
{
    typedef typename Matrix::TConfig TConfig;
    typedef typename Matrix::value_type ValueTypeA;
    typedef typename TConfig::VecPrec ValueTypeB;
    typedef Vector<TConfig> VVector;
    VVector x(A.get_num_rows()), y(A.get_num_rows());
    fill(x, 1);

    for (int i = 0; i < 20; i++)
    {
        ValueTypeB Lmax = get_norm(A, x, LMAX);
        scal(x, ValueTypeB(1) / Lmax);
        multiply(A, x, y);
        x.swap(y);
    }

    ValueTypeB retval = get_norm(A, x, L2) / get_norm(A, y, L2);
    return retval;
}

} // namespace amgx
