// SPDX-FileCopyrightText: 2008 - 2025 NVIDIA CORPORATION. All Rights Reserved.
//
// SPDX-License-Identifier: BSD-3-Clause

#include <cusp/convert.h>

#include <cusp/detail/format_utils.h>

#include <thrust/sort.h>
#include <thrust/iterator/zip_iterator.h>

namespace cusp
{

//////////////////
// Constructors //
//////////////////
        
// construct from a different matrix
template <typename IndexType, typename ValueType, class MemorySpace>
template <typename MatrixType>
coo_matrix<IndexType,ValueType,MemorySpace>
    ::coo_matrix(const MatrixType& matrix)
    {
        cusp::convert(matrix, *this);
    }

////////////////////////////////
// Container Member Functions //
////////////////////////////////
        
// assignment from another matrix
template <typename IndexType, typename ValueType, class MemorySpace>
template <typename MatrixType>
    coo_matrix<IndexType,ValueType,MemorySpace>&
    coo_matrix<IndexType,ValueType,MemorySpace>
    ::operator=(const MatrixType& matrix)
    {
        cusp::convert(matrix, *this);
        
        return *this;
    }

// sort matrix elements by row index
template <typename IndexType, typename ValueType, class MemorySpace>
    void
    coo_matrix<IndexType,ValueType,MemorySpace>
    ::sort_by_row(void)
    {
        cusp::detail::sort_by_row(row_indices, column_indices, values);
    }

// sort matrix elements by row index
template <typename IndexType, typename ValueType, class MemorySpace>
    void
    coo_matrix<IndexType,ValueType,MemorySpace>
    ::sort_by_row_and_column(void)
    {
        cusp::detail::sort_by_row_and_column(row_indices, column_indices, values);
    }

// determine whether matrix elements are sorted by row index
template <typename IndexType, typename ValueType, class MemorySpace>
    bool
    coo_matrix<IndexType,ValueType,MemorySpace>
    ::is_sorted_by_row(void)
    {
        return amgx::thrust::is_sorted(row_indices.begin(), row_indices.end());
    }

// determine whether matrix elements are sorted by row and column index
template <typename IndexType, typename ValueType, class MemorySpace>
    bool
    coo_matrix<IndexType,ValueType,MemorySpace>
    ::is_sorted_by_row_and_column(void)
    {
        return amgx::thrust::is_sorted
            (amgx::thrust::make_zip_iterator(amgx::thrust::make_tuple(row_indices.begin(), column_indices.begin())),
             amgx::thrust::make_zip_iterator(amgx::thrust::make_tuple(row_indices.end(),   column_indices.end())));
    }

///////////////////////////
// View Member Functions //
///////////////////////////

// sort matrix elements by row index
template <typename Array1, typename Array2, typename Array3, typename IndexType, typename ValueType, typename MemorySpace>
    void
    coo_matrix_view<Array1,Array2,Array3,IndexType,ValueType,MemorySpace>
    ::sort_by_row(void)
    {
        cusp::detail::sort_by_row(row_indices, column_indices, values);
    }

// sort matrix elements by row index
template <typename Array1, typename Array2, typename Array3, typename IndexType, typename ValueType, typename MemorySpace>
    void
    coo_matrix_view<Array1,Array2,Array3,IndexType,ValueType,MemorySpace>
    ::sort_by_row_and_column(void)
    {
        cusp::detail::sort_by_row_and_column(row_indices, column_indices, values);
    }

// determine whether matrix elements are sorted by row index
template <typename Array1, typename Array2, typename Array3, typename IndexType, typename ValueType, typename MemorySpace>
    bool
    coo_matrix_view<Array1,Array2,Array3,IndexType,ValueType,MemorySpace>
    ::is_sorted_by_row(void)
    {
        return amgx::thrust::is_sorted(row_indices.begin(), row_indices.end());
    }

// determine whether matrix elements are sorted by row and column index
template <typename Array1, typename Array2, typename Array3, typename IndexType, typename ValueType, typename MemorySpace>
    bool
    coo_matrix_view<Array1,Array2,Array3,IndexType,ValueType,MemorySpace>
    ::is_sorted_by_row_and_column(void)
    {
        return amgx::thrust::is_sorted
            (amgx::thrust::make_zip_iterator(amgx::thrust::make_tuple(row_indices.begin(), column_indices.begin())),
             amgx::thrust::make_zip_iterator(amgx::thrust::make_tuple(row_indices.end(),   column_indices.end())));
    }

} // end namespace cusp

