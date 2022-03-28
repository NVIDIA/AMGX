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

