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


#pragma once

#include <cusp/format.h>
#include <cusp/coo_matrix.h>

#include <cusp/detail/device/detail/coo.h>

namespace cusp
{
namespace detail
{
namespace device
{
namespace dispatch
{

/////////
// COO //
/////////
    
template <typename Matrix1,
          typename Matrix2,
          typename Matrix3>
void add(const Matrix1& A,
         const Matrix2& B,
               Matrix3& C,
         cusp::coo_format,
         cusp::coo_format,
         cusp::coo_format)
{
    cusp::detail::device::detail::coo_add(A, B, C);
}

template <typename Matrix1,
          typename Matrix2,
          typename Matrix3>
void subtract(const Matrix1& A,
              const Matrix2& B,
                    Matrix3& C,
              cusp::coo_format,
              cusp::coo_format,
              cusp::coo_format)
{
    cusp::detail::device::detail::coo_subtract(A, B, C);
}


///////////
// Array //
///////////

template <typename Matrix1,
          typename Matrix2,
          typename Matrix3>
void add(const Matrix1& A,
         const Matrix2& B,
               Matrix3& C,
         cusp::array2d_format,
         cusp::array2d_format,
         cusp::array2d_format)
{
    typedef typename Matrix3::value_type ValueType;

    C.resize(A.num_rows, A.num_cols);

    thrust::transform(A.values.values.begin(), A.values.values.end(),
                      B.values.values.begin(),
                      C.values.values.begin(),
                      thrust::plus<ValueType>());
}

template <typename Matrix1,
          typename Matrix2,
          typename Matrix3>
void subtract(const Matrix1& A,
              const Matrix2& B,
                    Matrix3& C,
              cusp::array2d_format,
              cusp::array2d_format,
              cusp::array2d_format)
{
    typedef typename Matrix3::value_type ValueType;

    C.resize(A.num_rows, A.num_cols);

    thrust::transform(A.values.values.begin(), A.values.values.end(),
                      B.values.values.begin(),
                      C.values.values.begin(),
                      thrust::minus<ValueType>());
}

/////////////
// Default //
/////////////

template <typename Matrix1,
          typename Matrix2,
          typename Matrix3,
          typename MatrixType1,
          typename MatrixType2,
          typename MatrixType3>
void add(const Matrix1& A,
         const Matrix2& B,
               Matrix3& C,
         MatrixType1,
         MatrixType2,
         MatrixType3)
{
    typedef typename Matrix1::index_type IndexType1;
    typedef typename Matrix2::index_type IndexType2;
    typedef typename Matrix3::index_type IndexType3;
    typedef typename Matrix1::value_type ValueType1;
    typedef typename Matrix2::value_type ValueType2;
    typedef typename Matrix3::value_type ValueType3;

    cusp::coo_matrix<IndexType1,ValueType1,cusp::device_memory> A_(A);
    cusp::coo_matrix<IndexType2,ValueType2,cusp::device_memory> B_(B);
    cusp::coo_matrix<IndexType3,ValueType3,cusp::device_memory> C_;

    cusp::detail::device::add(A_, B_, C_);

    cusp::convert(C_, C);
}

template <typename Matrix1,
          typename Matrix2,
          typename Matrix3,
          typename MatrixType1,
          typename MatrixType2,
          typename MatrixType3>
void subtract(const Matrix1& A,
              const Matrix2& B,
                    Matrix3& C,
              MatrixType1,
              MatrixType2,
              MatrixType3)
{
    typedef typename Matrix1::index_type IndexType1;
    typedef typename Matrix2::index_type IndexType2;
    typedef typename Matrix3::index_type IndexType3;
    typedef typename Matrix1::value_type ValueType1;
    typedef typename Matrix2::value_type ValueType2;
    typedef typename Matrix3::value_type ValueType3;

    cusp::coo_matrix<IndexType1,ValueType1,cusp::device_memory> A_(A);
    cusp::coo_matrix<IndexType2,ValueType2,cusp::device_memory> B_(B);
    cusp::coo_matrix<IndexType3,ValueType3,cusp::device_memory> C_;

    cusp::detail::device::subtract(A_, B_, C_);

    cusp::convert(C_, C);
}

} // end namespace dispatch



/////////////////
// Entry Point //
/////////////////

template <typename Matrix1,
          typename Matrix2,
          typename Matrix3>
void add(const Matrix1& A,
         const Matrix2& B,
               Matrix3& C)
{
    cusp::detail::device::dispatch::add(A, B, C,
            typename Matrix1::format(),
            typename Matrix2::format(),
            typename Matrix3::format());
}

template <typename Matrix1,
          typename Matrix2,
          typename Matrix3>
void subtract(const Matrix1& A,
              const Matrix2& B,
                    Matrix3& C)
{
    cusp::detail::device::dispatch::subtract(A, B, C,
            typename Matrix1::format(),
            typename Matrix2::format(),
            typename Matrix3::format());
}

} // end namespace device
} // end namespace detail
} // end namespace cusp

