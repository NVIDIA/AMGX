// SPDX-FileCopyrightText: 2008 - 2025 NVIDIA CORPORATION. All Rights Reserved.
//
// SPDX-License-Identifier: BSD-3-Clause

/*! \file diagonal.inl
 *  \brief Inline file for diagonal.h
 */

#include <cusp/blas.h>
#include <cusp/detail/format_utils.h>

#include <thrust/functional.h>
#include <thrust/transform.h>

namespace cusp
{
namespace precond
{
namespace detail
{
    template <typename T>
        struct reciprocal
    {
        __host__ __device__
        T operator()(const T& v)
        {
            return T(1.0) / v;
        }
    };

} // end namespace detail


// constructor
template <typename ValueType, typename MemorySpace>
    template<typename MatrixType>
    diagonal<ValueType,MemorySpace>
    ::diagonal(const MatrixType& A)
        : linear_operator<ValueType,MemorySpace>(A.num_rows, A.num_cols, A.num_rows)
    {
        // extract the main diagonal
        cusp::detail::extract_diagonal(A, diagonal_reciprocals);
    
        // invert the entries
        thrust_wrapper::transform(diagonal_reciprocals.begin(), diagonal_reciprocals.end(),
                          diagonal_reciprocals.begin(), detail::reciprocal<ValueType>());
    }
        
// linear operator
template <typename ValueType, typename MemorySpace>
    template <typename VectorType1, typename VectorType2>
    void diagonal<ValueType, MemorySpace>
    ::operator()(const VectorType1& x, VectorType2& y) const
    {
        cusp::blas::xmy(diagonal_reciprocals, x, y);
    }

} // end namespace precond
} // end namespace cusp

