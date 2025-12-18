// SPDX-FileCopyrightText: 2008 - 2025 NVIDIA CORPORATION. All Rights Reserved.
//
// SPDX-License-Identifier: BSD-3-Clause

#include <cusp/detail/dispatch/elementwise.h>

namespace cusp
{

//template <typename Matrix1,
//          typename Matrix2,
//          typename Matrix3,
//          typename BinaryFunction>
//void transform_elementwise(const Matrix1& A,
//                           const Matrix2& B,
//                                 Matrix3& C,
//                                 BinaryFunction op)
//{
//    cusp::detail::dispatch::transform_elementwise(A, B, C, op,
//            typename Matrix1::memory_space(),
//            typename Matrix2::memory_space(),
//            typename Matrix3::memory_space());
//}

template <typename Matrix1,
          typename Matrix2,
          typename Matrix3>
void add(const Matrix1& A,
         const Matrix2& B,
               Matrix3& C)
{
    CUSP_PROFILE_SCOPED();

    // TODO replace with cusp::detail::assert_same_dimensions(A,B);
    if(A.num_rows != B.num_rows || A.num_cols != B.num_cols)
        throw cusp::invalid_input_exception("array dimensions do not match");

    cusp::detail::dispatch::add(A, B, C,
            typename Matrix1::memory_space(),
            typename Matrix2::memory_space(),
            typename Matrix3::memory_space());

}

template <typename Matrix1,
          typename Matrix2,
          typename Matrix3>
void subtract(const Matrix1& A,
              const Matrix2& B,
                    Matrix3& C)
{
    CUSP_PROFILE_SCOPED();

    // TODO replace with cusp::detail::assert_same_dimensions(A,B);
    if(A.num_rows != B.num_rows || A.num_cols != B.num_cols)
        throw cusp::invalid_input_exception("array dimensions do not match");
    
    cusp::detail::dispatch::subtract(A, B, C,
            typename Matrix1::memory_space(),
            typename Matrix2::memory_space(),
            typename Matrix3::memory_space());
}

} // end namespace cusp

