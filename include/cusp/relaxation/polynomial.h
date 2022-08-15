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

/*! \file polynomial.h
 *  \brief polynomial relaxation.
 */

#pragma once

#include <cusp/detail/config.h>

#include <cusp/linear_operator.h>

namespace cusp
{
namespace relaxation
{

template <typename ValueType, typename MemorySpace>
class polynomial
{
    cusp::array1d<ValueType, host_memory> default_coefficients;
    cusp::array1d<ValueType, MemorySpace> residual;
    cusp::array1d<ValueType, MemorySpace> h;
    cusp::array1d<ValueType, MemorySpace> y;

public:
    polynomial();

    template <typename MatrixType, typename VectorType>
    polynomial(const MatrixType& A, const VectorType& coefficients);

    // ignores initial x
    template<typename MatrixType, typename VectorType1, typename VectorType2>
    void presmooth(const MatrixType& A, const VectorType1& b, VectorType2& x);
   
    // smooths initial x
    template<typename MatrixType, typename VectorType1, typename VectorType2>
    void postsmooth(const MatrixType& A, const VectorType1& b, VectorType2& x);

    template <typename MatrixType, typename VectorType1, typename VectorType2>
    void operator()(const MatrixType& A, const VectorType1& b, VectorType2& x) const;

    template <typename MatrixType, typename VectorType1, typename VectorType2, typename VectorType3>
    void operator()(const MatrixType& A, const VectorType1& b, VectorType2& x, VectorType3& coeffients);
};

} // end namespace relaxation
} // end namespace cusp

#include <cusp/relaxation/detail/polynomial.inl>

