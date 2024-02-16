// SPDX-FileCopyrightText: 2008 - 2024 NVIDIA CORPORATION. All Rights Reserved.
//
// SPDX-License-Identifier: BSD-3-Clause

/*! \file linear_operator.h
 *  \brief Abstract interface for iterative solvers
 */

#pragma once

#include <cusp/detail/config.h>

#include <cusp/format.h>
#include <cusp/blas.h>
#include <cusp/exception.h>
#include <cusp/detail/matrix_base.h>

namespace cusp
{

template <typename ValueType, typename MemorySpace, typename IndexType=int>
class linear_operator : public cusp::detail::matrix_base<IndexType,ValueType,MemorySpace,cusp::unknown_format>
{
  typedef cusp::detail::matrix_base<IndexType,ValueType,MemorySpace,cusp::unknown_format> Parent;
 public:
  linear_operator()
      : Parent() {}

  linear_operator(IndexType num_rows, IndexType num_cols)
      : Parent(num_rows, num_cols) {}

  linear_operator(IndexType num_rows, IndexType num_cols, IndexType num_entries)
      : Parent(num_rows, num_cols, num_entries) {}
}; // linear_operator

template <typename ValueType, typename MemorySpace, typename IndexType=int>
class identity_operator : public linear_operator<ValueType,MemorySpace,IndexType>
{       
    typedef linear_operator<ValueType,MemorySpace> Parent;
    public:

    identity_operator() 
        : Parent() {}
    
    identity_operator(IndexType num_rows, IndexType num_cols)
        : Parent(num_rows, num_cols) {}

    template <typename VectorType1,
              typename VectorType2>
    void operator()(const VectorType1& x, VectorType2& y) const
    {
        cusp::blas::copy(x, y);
    }
}; // identity_operator

} // end namespace cusp

