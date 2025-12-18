// SPDX-FileCopyrightText: 2008 - 2025 NVIDIA CORPORATION. All Rights Reserved.
//
// SPDX-License-Identifier: BSD-3-Clause

#include <cusp/convert.h>
#include <cusp/detail/utils.h>

namespace cusp
{

//////////////////
// Constructors //
//////////////////
        
// construct from a different matrix
template <typename IndexType, typename ValueType, class MemorySpace>
template <typename MatrixType>
ell_matrix<IndexType,ValueType,MemorySpace>
    ::ell_matrix(const MatrixType& matrix)
    {
        cusp::convert(matrix, *this);
    }

//////////////////////
// Member Functions //
//////////////////////

// assignment from another matrix
template <typename IndexType, typename ValueType, class MemorySpace>
template <typename MatrixType>
    ell_matrix<IndexType,ValueType,MemorySpace>&
    ell_matrix<IndexType,ValueType,MemorySpace>
    ::operator=(const MatrixType& matrix)
    {
        cusp::convert(matrix, *this);
        
        return *this;
    }

///////////////////////////
// Convenience Functions //
///////////////////////////

template <typename Array1,
          typename Array2>
ell_matrix_view<Array1,Array2>
make_ell_matrix_view(size_t num_rows,
                     size_t num_cols,
                     size_t num_entries,
                     Array1 column_indices,
                     Array2 values)
{
  return ell_matrix_view<Array1,Array2>
    (num_rows, num_cols, num_entries,
     column_indices, values);
}

template <typename Array1,
          typename Array2,
          typename IndexType,
          typename ValueType,
          typename MemorySpace>
ell_matrix_view<Array1,Array2,IndexType,ValueType,MemorySpace>
make_ell_matrix_view(const ell_matrix_view<Array1,Array2,IndexType,ValueType,MemorySpace>& m)
{
  return ell_matrix_view<Array1,Array2,IndexType,ValueType,MemorySpace>(m);
}
    
template <typename IndexType, typename ValueType, class MemorySpace>
typename ell_matrix<IndexType,ValueType,MemorySpace>::view
make_ell_matrix_view(ell_matrix<IndexType,ValueType,MemorySpace>& m)
{
  return make_ell_matrix_view
    (m.num_rows, m.num_cols, m.num_entries,
     cusp::make_array2d_view(m.column_indices),
     cusp::make_array2d_view(m.values));
}

template <typename IndexType, typename ValueType, class MemorySpace>
typename ell_matrix<IndexType,ValueType,MemorySpace>::const_view
make_ell_matrix_view(const ell_matrix<IndexType,ValueType,MemorySpace>& m)
{
  return make_ell_matrix_view
    (m.num_rows, m.num_cols, m.num_entries,
     cusp::make_array2d_view(m.column_indices),
     cusp::make_array2d_view(m.values));
}

} // end namespace cusp

