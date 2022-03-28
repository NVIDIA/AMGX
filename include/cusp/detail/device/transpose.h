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

#include <cusp/format.h>
// TODO replace with detail/array2d_utils.h or something
#include <cusp/array2d.h>
#include <cusp/coo_matrix.h>
#include <cusp/csr_matrix.h>

#include <cusp/detail/utils.h>
#include <cusp/detail/format_utils.h>

#include <thrust/functional.h>
#include <thrust/gather.h>
#include <thrust/sequence.h>
#include <thrust/sort.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/transform_iterator.h>
#include <thrust/iterator/zip_iterator.h>

namespace cusp
{
namespace detail
{
namespace device
{

// COO format
template <typename MatrixType1,   typename MatrixType2>
void transpose(const MatrixType1& A, MatrixType2& At,
               cusp::coo_format,
               cusp::coo_format)
{
    At.resize(A.num_cols, A.num_rows, A.num_entries);

    cusp::copy(A.row_indices,    At.column_indices);
    cusp::copy(A.column_indices, At.row_indices);
    cusp::copy(A.values,         At.values);

    At.sort_by_row();
}


// CSR format
template <typename MatrixType1,   typename MatrixType2>
void transpose(const MatrixType1& A, MatrixType2& At,
               cusp::csr_format,
               cusp::csr_format)
{
    typedef typename MatrixType2::index_type   IndexType2;
    typedef typename MatrixType2::memory_space MemorySpace2;

    At.resize(A.num_cols, A.num_rows, A.num_entries);

    cusp::detail::offsets_to_indices(A.row_offsets, At.column_indices);
    cusp::copy(A.values, At.values);

    cusp::array1d<IndexType2,MemorySpace2> At_row_indices(A.column_indices);

    cusp::detail::sort_by_row(At_row_indices, At.column_indices, At.values);
    
    cusp::detail::indices_to_offsets(At_row_indices, At.row_offsets);
}


} // end namespace device
} // end namespace detail
} // end namespace cusp

