// SPDX-FileCopyrightText: 2008 - 2025 NVIDIA CORPORATION. All Rights Reserved.
//
// SPDX-License-Identifier: BSD-3-Clause

#pragma once

namespace cusp
{
namespace detail
{
namespace device
{

//template <typename Matrix1,
//          typename Matrix2,
//          typename Matrix3,
//          typename BinaryFunction>
//void transform_elementwise(const Matrix1& A,
//                           const Matrix2& B,
//                                 Matrix3& C,
//                                 BinaryFunction op);

template <typename Matrix1,
          typename Matrix2,
          typename Matrix3>
void add(const Matrix1& A,
         const Matrix2& B,
               Matrix3& C);

template <typename Matrix1,
          typename Matrix2,
          typename Matrix3>
void subtract(const Matrix1& A,
              const Matrix2& B,
                    Matrix3& C);

} // end namespace device
} // end namespace detail
} // end namespace cusp

#include <cusp/detail/device/elementwise.inl>

