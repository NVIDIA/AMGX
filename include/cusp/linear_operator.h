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

