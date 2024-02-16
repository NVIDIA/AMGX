// SPDX-FileCopyrightText: 2008 - 2024 NVIDIA CORPORATION. All Rights Reserved.
//
// SPDX-License-Identifier: BSD-3-Clause

#include <cusp/detail/dispatch/multiply.h>

#include <cusp/linear_operator.h>
#include <thrust/detail/type_traits.h>

namespace cusp
{
namespace detail
{

template <typename LinearOperator,
          typename MatrixOrVector1,
          typename MatrixOrVector2>
void multiply(LinearOperator&  A,
              MatrixOrVector1& B,
              MatrixOrVector2& C,
              cusp::unknown_format)
{
  // user-defined LinearOperator
  A(B,C);
}

template <typename LinearOperator,
          typename MatrixOrVector1,
          typename MatrixOrVector2>
void multiply(LinearOperator&  A,
              MatrixOrVector1& B,
              MatrixOrVector2& C,
              cusp::known_format)
{
  // built-in format
  cusp::detail::dispatch::multiply(A, B, C,
                                   typename LinearOperator::memory_space(),
                                   typename MatrixOrVector1::memory_space(),
                                   typename MatrixOrVector2::memory_space());
}

} // end namespace detail

template <typename LinearOperator,
          typename MatrixOrVector1,
          typename MatrixOrVector2>
void multiply(LinearOperator&  A,
              MatrixOrVector1& B,
              MatrixOrVector2& C)
{
  CUSP_PROFILE_SCOPED();

  // TODO check that dimensions are compatible

  typedef typename LinearOperator::value_type   ValueType;
  typedef typename LinearOperator::memory_space MemorySpace;

  cusp::detail::multiply(A, B, C,
                         typename LinearOperator::format());
}

} // end namespace cusp

