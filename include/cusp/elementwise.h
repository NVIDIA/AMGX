// SPDX-FileCopyrightText: 2008 - 2025 NVIDIA CORPORATION. All Rights Reserved.
//
// SPDX-License-Identifier: BSD-3-Clause

/*! \file elementwise.h
 *  \brief Elementwise operations on matrices.
 */

#pragma once

#include <cusp/detail/config.h>

namespace cusp
{

/*! \addtogroup algorithms Algorithms
 *  \ingroup algorithms
 *  \{
 */

//// Uses Matrix1::value_type(0) and Matrix2::value_type(0) for values not present
//template <typename Matrix1,
//          typename Matrix2,
//          typename Matrix3,
//          typename BinaryFunction>
//void transform_elementwise(const Matrix1& A,
//                           const Matrix2& B,
//                                 Matrix3& C,
//                                 BinaryFunction op);

/*! \p add : Compute the sum of two matrices
 */
template <typename Matrix1,
          typename Matrix2,
          typename Matrix3>
void add(const Matrix1& A,
         const Matrix2& B,
               Matrix3& C);

/*! \p add : Compute the difference of two matrices
 */
template <typename Matrix1,
          typename Matrix2,
          typename Matrix3>
void subtract(const Matrix1& A,
              const Matrix2& B,
                    Matrix3& C);
/*! \}
 */

} // end namespace cusp

#include <cusp/detail/elementwise.inl>

