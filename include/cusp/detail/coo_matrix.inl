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
        return thrust::is_sorted(row_indices.begin(), row_indices.end());
    }

// determine whether matrix elements are sorted by row and column index
template <typename IndexType, typename ValueType, class MemorySpace>
    bool
    coo_matrix<IndexType,ValueType,MemorySpace>
    ::is_sorted_by_row_and_column(void)
    {
        return thrust::is_sorted
            (thrust::make_zip_iterator(thrust::make_tuple(row_indices.begin(), column_indices.begin())),
             thrust::make_zip_iterator(thrust::make_tuple(row_indices.end(),   column_indices.end())));
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
        return thrust::is_sorted(row_indices.begin(), row_indices.end());
    }

// determine whether matrix elements are sorted by row and column index
template <typename Array1, typename Array2, typename Array3, typename IndexType, typename ValueType, typename MemorySpace>
    bool
    coo_matrix_view<Array1,Array2,Array3,IndexType,ValueType,MemorySpace>
    ::is_sorted_by_row_and_column(void)
    {
        return thrust::is_sorted
            (thrust::make_zip_iterator(thrust::make_tuple(row_indices.begin(), column_indices.begin())),
             thrust::make_zip_iterator(thrust::make_tuple(row_indices.end(),   column_indices.end())));
    }

} // end namespace cusp

