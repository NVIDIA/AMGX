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

// SpMV
#include <cusp/detail/device/spmv/coo_flat.h>
#include <cusp/detail/device/spmv/csr_vector.h>
#include <cusp/detail/device/spmv/dia.h>
#include <cusp/detail/device/spmv/ell.h>
#include <cusp/detail/device/spmv/hyb.h>

// SpMM
#include <cusp/detail/device/spmm/coo.h>

namespace cusp
{
namespace detail
{
namespace device
{

//////////////////////////////////
// Dense Matrix-Vector Multiply //
//////////////////////////////////
//// TODO implement this for both row and column-major ordering
//template <typename Matrix,
//          typename Vector1,
//          typename Vector2>
//void multiply(const Matrix&  A,
//              const Vector1& B,
//                    Vector2& C,
//              cusp::array2d_format,
//              cusp::array1d_format,
//              cusp::array1d_format)
//{
//}

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
#ifdef CUSP_USE_TEXTURE_MEMORY    
    cusp::detail::device::spmv_coo_flat_tex(A, thrust::raw_pointer_cast(&B[0]), thrust::raw_pointer_cast(&C[0]));
#else
    cusp::detail::device::spmv_coo_flat(A, thrust::raw_pointer_cast(&B[0]), thrust::raw_pointer_cast(&C[0]));
#endif    
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
#ifdef CUSP_USE_TEXTURE_MEMORY    
    cusp::detail::device::spmv_csr_vector_tex(A, thrust::raw_pointer_cast(&B[0]), thrust::raw_pointer_cast(&C[0]));
#else
    cusp::detail::device::spmv_csr_vector(A, thrust::raw_pointer_cast(&B[0]), thrust::raw_pointer_cast(&C[0]));
#endif    
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
#ifdef CUSP_USE_TEXTURE_MEMORY    
    cusp::detail::device::spmv_dia_tex(A, thrust::raw_pointer_cast(&B[0]), thrust::raw_pointer_cast(&C[0]));
#else
    cusp::detail::device::spmv_dia(A, thrust::raw_pointer_cast(&B[0]), thrust::raw_pointer_cast(&C[0]));
#endif    
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
#ifdef CUSP_USE_TEXTURE_MEMORY    
    cusp::detail::device::spmv_ell_tex(A, thrust::raw_pointer_cast(&B[0]), thrust::raw_pointer_cast(&C[0]));
#else
    cusp::detail::device::spmv_ell(A, thrust::raw_pointer_cast(&B[0]), thrust::raw_pointer_cast(&C[0]));
#endif    
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
#ifdef CUSP_USE_TEXTURE_MEMORY    
    cusp::detail::device::spmv_hyb_tex(A, thrust::raw_pointer_cast(&B[0]), thrust::raw_pointer_cast(&C[0]));
#else
    cusp::detail::device::spmv_hyb(A, thrust::raw_pointer_cast(&B[0]), thrust::raw_pointer_cast(&C[0]));
#endif    
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
//              cusp::sparse_format,
//              cusp::array2d_format,
//              cusp::array2d_format)
//{
//}

////////////////////////////////////////
// Dense Matrix-Matrix Multiplication //
////////////////////////////////////////
// TODO implement
//template <typename Matrix1,
//          typename Matrix2,
//          typename Matrix3>
//void multiply(const Matrix1& A,
//              const Matrix2& B,
//                    Matrix3& C,
//              cusp::array2d_format,
//              cusp::array2d_format,
//              cusp::array2d_format)
//{
//}

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
    cusp::detail::device::spmm_coo(A,B,C);
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
    // other formats use COO * COO
    cusp::coo_matrix<typename Matrix1::index_type,typename Matrix1::value_type,cusp::device_memory> A_(A);
    cusp::coo_matrix<typename Matrix2::index_type,typename Matrix2::value_type,cusp::device_memory> B_(B);
    cusp::coo_matrix<typename Matrix3::index_type,typename Matrix3::value_type,cusp::device_memory> C_;

    cusp::detail::device::spmm_coo(A_,B_,C_);

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
    cusp::detail::device::multiply(A, B, C,
            typename Matrix::format(),
            typename MatrixOrVector1::format(),
            typename MatrixOrVector2::format());
}

} // end namespace device
} // end namespace detail
} // end namespace cusp

