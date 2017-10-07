/*
 *  Copyright 2008-2009 NVIDIA Corporation
 *
 *  Licensed under the Apache License, Version 2.0 (the "License");
 *  you may not use this file except in compliance with the License.
 *  You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 *  Unless required by applicable law or agreed to in writing, software
 *  distributed under the License is distributed on an "AS IS" BASIS,
 *  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  See the License for the specific language governing permissions and
 *  limitations under the License.
 */

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

