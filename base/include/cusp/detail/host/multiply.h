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

#include <cusp/detail/functional.h>

#ifdef INTEL_MKL_SPBLAS
#include <cusp/detail/host/spmv_mkl.h>
#else
#include <cusp/detail/host/spmv.h>
#endif

#include <cusp/detail/host/detail/coo.h>
#include <cusp/detail/host/detail/csr.h>

namespace cusp
{
namespace detail
{
namespace host
{

//////////////////////////////////
// Dense Matrix-Vector Multiply //
//////////////////////////////////
template <typename Matrix,
          typename Vector1,
          typename Vector2>
void multiply(const Matrix&  A,
              const Vector1& B,
                    Vector2& C,
              cusp::array2d_format,
              cusp::array1d_format,
              cusp::array1d_format)
{
    typedef typename Vector2::value_type ValueType;

    for(size_t i = 0; i < A.num_rows; i++)
    {
        ValueType sum = 0;
        for(size_t j = 0; j < A.num_cols; j++)
        {
            sum += A(i,j) * B[j];
        }
        C[i] = sum;
    }
}

///////////////////////////////////
// Sparse Matrix-Vector Multiply //
///////////////////////////////////
template <typename Matrix,
          typename Vector1,
          typename Vector2>
void multiply(const Matrix&  A,
              const Vector1& B,
                    Vector2& C,
              cusp::coo_format,
              cusp::array1d_format,
              cusp::array1d_format)
{
    cusp::detail::host::spmv_coo(A, B, C);
}

template <typename Matrix,
          typename Vector1,
          typename Vector2>
void multiply(const Matrix&  A,
              const Vector1& B,
                    Vector2& C,
              cusp::csr_format,
              cusp::array1d_format,
              cusp::array1d_format)
{
    cusp::detail::host::spmv_csr(A, B, C);
}

template <typename Matrix,
          typename Vector1,
          typename Vector2>
void multiply(const Matrix&  A,
              const Vector1& B,
                    Vector2& C,
              cusp::dia_format,
              cusp::array1d_format,
              cusp::array1d_format)
{
    cusp::detail::host::spmv_dia(A, B, C);
}

template <typename Matrix,
          typename Vector1,
          typename Vector2>
void multiply(const Matrix&  A,
              const Vector1& B,
                    Vector2& C,
              cusp::ell_format,
              cusp::array1d_format,
              cusp::array1d_format)
{
    cusp::detail::host::spmv_ell(A, B, C);
}

template <typename Matrix,
          typename Vector1,
          typename Vector2>
void multiply(const Matrix&  A,
              const Vector1& B,
                    Vector2& C,
              cusp::hyb_format,
              cusp::array1d_format,
              cusp::array1d_format)
{
    typedef typename Vector2::value_type ValueType;

    cusp::detail::host::spmv_ell(A.ell, B, C);
    cusp::detail::host::spmv_coo(A.coo, B, C, thrust::identity<ValueType>(), thrust::multiplies<ValueType>(), thrust::plus<ValueType>());
}

////////////////////////////////////////
// Sparse Matrix-BlockVector Multiply //
////////////////////////////////////////
//// TODO implement this w/ repeated SpMVs and then specialize
//template <typename Matrix,
//          typename Vector1,
//          typename Vector2>
//void multiply(const Matrix&  A,
//              const Vector1& B,
//                    Vector2& C,
//              cusp::detail::sparse_format,
//              cusp::detail::array2d_format,
//              cusp::detail::array2d_format)
//{
//}

////////////////////////////////////////
// Dense Matrix-Matrix Multiplication //
////////////////////////////////////////
template <typename Matrix1,
          typename Matrix2,
          typename Matrix3>
void multiply(const Matrix1&  A,
              const Matrix2& B,
                    Matrix3& C,
              cusp::array2d_format,
              cusp::array2d_format,
              cusp::array2d_format)
{
    typedef typename Matrix3::value_type ValueType;

    C.resize(A.num_rows, B.num_cols);

    for(size_t i = 0; i < C.num_rows; i++)
    {
        for(size_t j = 0; j < C.num_cols; j++)
        {
            ValueType v = 0;

            for(size_t k = 0; k < A.num_cols; k++)
                v += A(i,k) * B(k,j);
            
            C(i,j) = v;
        }
    }
}

/////////////////////////////////////////
// Sparse Matrix-Matrix Multiplication //
/////////////////////////////////////////
template <typename Matrix1,
          typename Matrix2,
          typename Matrix3>
void multiply(const Matrix1& A,
              const Matrix2& B,
                    Matrix3& C,
              cusp::coo_format,
              cusp::coo_format,
              cusp::coo_format)
{
    cusp::detail::host::detail::spmm_coo(A,B,C);
}

template <typename Matrix1,
          typename Matrix2,
          typename Matrix3>
void multiply(const Matrix1& A,
              const Matrix2& B,
                    Matrix3& C,
              cusp::csr_format,
              cusp::csr_format,
              cusp::csr_format)
{
    cusp::detail::host::detail::spmm_csr(A,B,C);
}

template <typename Matrix1,
          typename Matrix2,
          typename Matrix3>
void multiply(const Matrix1& A,
              const Matrix2& B,
                    Matrix3& C,
              cusp::sparse_format,
              cusp::sparse_format,
              cusp::sparse_format)
{
    // other formats use CSR * CSR
    cusp::csr_matrix<typename Matrix1::index_type,typename Matrix1::value_type,cusp::host_memory> A_(A);
    cusp::csr_matrix<typename Matrix2::index_type,typename Matrix2::value_type,cusp::host_memory> B_(B);
    cusp::csr_matrix<typename Matrix3::index_type,typename Matrix3::value_type,cusp::host_memory> C_;

    cusp::detail::host::detail::spmm_csr(A_,B_,C_);

    cusp::convert(C_, C);
}
  
/////////////////
// Entry Point //
/////////////////
template <typename Matrix,
          typename MatrixOrVector1,
          typename MatrixOrVector2>
void multiply(const Matrix&  A,
              const MatrixOrVector1& B,
                    MatrixOrVector2& C)
{
  cusp::detail::host::multiply(A, B, C,
                               typename Matrix::format(),
                               typename MatrixOrVector1::format(),
                               typename MatrixOrVector2::format());
}

} // end namespace host
} // end namespace detail
} // end namespace cusp

