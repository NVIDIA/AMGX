// SPDX-FileCopyrightText: 2008 - 2024 NVIDIA CORPORATION. All Rights Reserved.
//
// SPDX-License-Identifier: BSD-3-Clause

/*! \file jacobi.h
 *  \brief Jacobi relaxation.
 */

#pragma once

#include <cusp/detail/config.h>

#include <cusp/linear_operator.h>

namespace cusp
{
namespace relaxation
{

template <typename ValueType, typename MemorySpace>
class jacobi
{
    ValueType default_omega;
    cusp::array1d<ValueType,MemorySpace> diagonal;
    cusp::array1d<ValueType,MemorySpace> temp;

public:
    jacobi();

    template <typename MatrixType>
    jacobi(const MatrixType& A, ValueType omega=1.0);
    
    // ignores initial x
    template<typename MatrixType, typename VectorType1, typename VectorType2>
    void presmooth(const MatrixType& A, const VectorType1& b, VectorType2& x);
   
    // smooths initial x
    template<typename MatrixType, typename VectorType1, typename VectorType2>
    void postsmooth(const MatrixType& A, const VectorType1& b, VectorType2& x);

    template <typename MatrixType, typename VectorType1, typename VectorType2>
    void operator()(const MatrixType& A, const VectorType1& b, VectorType2& x);
        
    template <typename MatrixType, typename VectorType1, typename VectorType2>
    void operator()(const MatrixType& A, const VectorType1& b, VectorType2& x, ValueType omega);
};

} // end namespace relaxation
} // end namespace cusp

#include <cusp/relaxation/detail/jacobi.inl>

