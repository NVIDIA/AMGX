// SPDX-FileCopyrightText: 2008 - 2024 NVIDIA CORPORATION. All Rights Reserved.
//
// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#include <cusp/detail/config.h>

namespace cusp
{
namespace precond
{
namespace detail
{

/*  Compute a strength of connection matrix using the standard symmetric measure.
 *  An off-diagonal connection A[i,j] is strong iff::
 *
 *     abs(A[i,j]) >= theta * sqrt( abs(A[i,i]) * abs(A[j,j]) )
 *
 *  With the default threshold (theta = 0.0) all connections are strong.
 *
 *  Note: explicit diagonal entries are always considered strong.
 */
template <typename Matrix1, typename Matrix2>
void symmetric_strength_of_connection(const Matrix1& A, Matrix2& S, const double theta = 0.0);

} // end namepace detail
} // end namespace precond
} // end namespace cusp

#include <cusp/precond/detail/strength.inl>

