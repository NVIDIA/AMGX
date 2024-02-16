// SPDX-FileCopyrightText: 2008 - 2024 NVIDIA CORPORATION. All Rights Reserved.
//
// SPDX-License-Identifier: BSD-3-Clause

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

