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

#include <cusp/detail/host/multiply.h>
#include <cusp/detail/device/multiply.h>

namespace cusp
{
namespace detail
{
namespace dispatch
{

////////////////
// Host Paths //
////////////////
template <typename LinearOperator,
          typename MatrixOrVector1,
          typename MatrixOrVector2>
void multiply(const LinearOperator&  A,
              const MatrixOrVector1& B,
                    MatrixOrVector2& C,
              cusp::host_memory,
              cusp::host_memory,
              cusp::host_memory)
{
    cusp::detail::host::multiply(A, B, C);
}

//////////////////
// Device Paths //
//////////////////
template <typename LinearOperator,
          typename MatrixOrVector1,
          typename MatrixOrVector2>
void multiply(const LinearOperator&  A,
              const MatrixOrVector1& B,
                    MatrixOrVector2& C,
              cusp::device_memory,
              cusp::device_memory,
              cusp::device_memory)
{
    cusp::detail::device::multiply(A, B, C);
}

} // end namespace dispatch
} // end namespace detail
} // end namespace cusp

