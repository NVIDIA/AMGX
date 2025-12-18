// SPDX-FileCopyrightText: 2008 - 2025 NVIDIA CORPORATION. All Rights Reserved.
//
// SPDX-License-Identifier: BSD-3-Clause

#pragma once

namespace cusp
{
namespace detail
{

template <typename OffsetArray, typename IndexArray>
void offsets_to_indices(const OffsetArray& offsets, IndexArray& indices);

template <typename IndexArray, typename OffsetArray>
void indices_to_offsets(const IndexArray& indices, OffsetArray& offsets);
    
template <typename MatrixType, typename ArrayType>
void extract_diagonal(const MatrixType& A, ArrayType& output);

template <typename Array1, typename Array2, typename Array3>
void sort_by_row(Array1& rows, Array2& columns, Array3& values);

template <typename Array1, typename Array2, typename Array3>
void sort_by_row_and_column(Array1& rows, Array2& columns, Array3& values);
    
} // end namespace detail
} // end namespace cusp

#include <cusp/detail/format_utils.inl>

