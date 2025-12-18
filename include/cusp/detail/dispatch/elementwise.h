// SPDX-FileCopyrightText: 2008 - 2025 NVIDIA CORPORATION. All Rights Reserved.
//
// SPDX-License-Identifier: BSD-3-Clause

#include <cusp/array1d.h>

#include <cusp/detail/host/elementwise.h>
#include <cusp/detail/device/elementwise.h>

#include <thrust/functional.h>

namespace cusp
{
namespace detail
{
namespace dispatch
{

////////////////
// Host Paths //
////////////////
//template <typename Matrix1,
//          typename Matrix2,
//          typename Matrix3,
//          typename BinaryFunction>
//void transform_elementwise(const Matrix1& A,
//                           const Matrix2& B,
//                                 Matrix3& C,
//                                 BinaryFunction op,
//                           cusp::host_memory,
//                           cusp::host_memory,
//                           cusp::host_memory)
//{
//    cusp::detail::host::transform_elementwise(A, B, C, op);
//}

template <typename Matrix1,
          typename Matrix2,
          typename Matrix3>
void add(const Matrix1& A,
         const Matrix2& B,
               Matrix3& C,
         cusp::host_memory,
         cusp::host_memory,
         cusp::host_memory)
{
    typedef typename Matrix3::value_type ValueType;
    cusp::detail::host::transform_elementwise(A, B, C, amgx::thrust::plus<ValueType>());
}

template <typename Matrix1,
          typename Matrix2,
          typename Matrix3>
void subtract(const Matrix1& A,
              const Matrix2& B,
                    Matrix3& C,
              cusp::host_memory,
              cusp::host_memory,
              cusp::host_memory)
{
    typedef typename Matrix3::value_type ValueType;
    cusp::detail::host::transform_elementwise(A, B, C, amgx::thrust::minus<ValueType>());
}

//////////////////
// Device Paths //
//////////////////
//template <typename Matrix1,
//          typename Matrix2,
//          typename Matrix3,
//          typename BinaryFunction>
//void transform_elementwise(const Matrix1& A,
//                           const Matrix2& B,
//                                 Matrix3& C,
//                                 BinaryFunction op,
//                           cusp::device_memory,
//                           cusp::device_memory,
//                           cusp::device_memory)
//{
//    cusp::detail::device::transform_elementwise(A, B, C, op);
//}

template <typename Matrix1,
          typename Matrix2,
          typename Matrix3>
void add(const Matrix1& A,
         const Matrix2& B,
               Matrix3& C,
         cusp::device_memory,
         cusp::device_memory,
         cusp::device_memory)
{
    cusp::detail::device::add(A, B, C);
}

template <typename Matrix1,
          typename Matrix2,
          typename Matrix3>
void subtract(const Matrix1& A,
              const Matrix2& B,
                    Matrix3& C,
              cusp::device_memory,
              cusp::device_memory,
              cusp::device_memory)
{
    cusp::detail::device::subtract(A, B, C);
}

} // end namespace dispatch
} // end namespace detail
} // end namespace cusp

