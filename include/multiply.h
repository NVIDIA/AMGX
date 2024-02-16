// SPDX-FileCopyrightText: 2011 - 2024 NVIDIA CORPORATION. All Rights Reserved.
//
// SPDX-License-Identifier: BSD-3-Clause

#pragma once
#include <basic_types.h>
#include <vector.h>
#include <matrix.h>

namespace amgx
{

//computes C=A*B
template <class TConfig>
void multiply(Matrix<TConfig> &A, Vector<TConfig> &B, Vector<TConfig> &C, ViewType view = OWNED);

template <class TConfig>
void multiply_masked(Matrix<TConfig> &A, Vector<TConfig> &B, Vector<TConfig> &C, typename Matrix<TConfig>::IVector &mask, ViewType view = OWNED);

template <class MatrixA, class Vector>
void multiply_with_mask(MatrixA &A, Vector &B, Vector &C);

template <class MatrixA, class Vector>
void multiply_with_mask_restriction(MatrixA &A, Vector &B, Vector &C, MatrixA &P);


//computes C=A*B
template <class TConfig>
void multiplyMM(const Matrix<TConfig> &A, const Matrix<TConfig> &B, Matrix<TConfig> &C);

} // namespace amgx
