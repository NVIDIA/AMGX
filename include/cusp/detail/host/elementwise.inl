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
#include <cusp/csr_matrix.h>
#include <cusp/detail/host/detail/csr.h>

namespace cusp
{
namespace detail
{
namespace host
{
namespace dispatch
{

/////////
// COO //
/////////

// TODO add COO version of transform_elementwise
//template <typename Matrix1,
//          typename Matrix2,
//          typename Matrix3,
//          typename BinaryFunction>
//void transform_elementwise(const Matrix1& A,
//                           const Matrix2& B,
//                                 Matrix3& C,
//                                 BinaryFunction op,
//                           cusp::coo_format,
//                           cusp::coo_format,
//                           cusp::coo_format)
//{
//    cusp::detail::host::detail::transform_elementwise_coo(A, B, C, op); 
//}


/////////
// CSR //
/////////

template <typename Matrix1,
          typename Matrix2,
          typename Matrix3,
          typename BinaryFunction>
void transform_elementwise(const Matrix1& A,
                           const Matrix2& B,
                                 Matrix3& C,
                                 BinaryFunction op,
                           cusp::csr_format,
                           cusp::csr_format,
                           cusp::csr_format)
{
    cusp::detail::host::detail::csr_transform_elementwise(A, B, C, op); 
}

///////////
// Array //
///////////

template <typename Matrix1,
          typename Matrix2,
          typename Matrix3,
          typename BinaryFunction>
void transform_elementwise(const Matrix1& A,
                           const Matrix2& B,
                                 Matrix3& C,
                                 BinaryFunction op,
                           cusp::array2d_format,
                           cusp::array2d_format,
                           cusp::array2d_format)
{
    C.resize(A.num_rows, A.num_cols);

    for(size_t i = 0; i < A.num_rows; i++)
        for(size_t j = 0; j < A.num_cols; j++)
            C(i,j) = op(A(i,j), B(i,j));
}

/////////////
// Default //
/////////////

template <typename Matrix1,
          typename Matrix2,
          typename Matrix3,
          typename BinaryFunction>
void transform_elementwise(const Matrix1& A,
                           const Matrix2& B,
                                 Matrix3& C,
                                 BinaryFunction op,
                           sparse_format,
                           sparse_format,
                           sparse_format)
{
    typedef typename Matrix1::index_type IndexType1;
    typedef typename Matrix2::index_type IndexType2;
    typedef typename Matrix3::index_type IndexType3;
    typedef typename Matrix1::value_type ValueType1;
    typedef typename Matrix2::value_type ValueType2;
    typedef typename Matrix3::value_type ValueType3;

    cusp::csr_matrix<IndexType1,ValueType1,cusp::host_memory> A_(A);
    cusp::csr_matrix<IndexType2,ValueType2,cusp::host_memory> B_(B);
    cusp::csr_matrix<IndexType3,ValueType3,cusp::host_memory> C_;

    cusp::detail::host::transform_elementwise(A_, B_, C_, op);

    cusp::convert(C_, C);
}

} // end namespace dispatch


/////////////////
// Entry Point //
/////////////////

template <typename Matrix1,
          typename Matrix2,
          typename Matrix3,
          typename BinaryFunction>
void transform_elementwise(const Matrix1& A,
                           const Matrix2& B,
                                 Matrix3& C,
                                 BinaryFunction op)
{
    cusp::detail::host::dispatch::transform_elementwise(A, B, C, op,
            typename Matrix1::format(),
            typename Matrix2::format(),
            typename Matrix3::format());
}

} // end namespace host
} // end namespace detail
} // end namespace cusp

