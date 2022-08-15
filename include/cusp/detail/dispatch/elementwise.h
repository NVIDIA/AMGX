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
    cusp::detail::host::transform_elementwise(A, B, C, thrust::plus<ValueType>());
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
    cusp::detail::host::transform_elementwise(A, B, C, thrust::minus<ValueType>());
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

