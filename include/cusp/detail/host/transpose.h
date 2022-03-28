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
namespace host
{

// COO format
template <typename MatrixType1,   typename MatrixType2>
void transpose(const MatrixType1& A, MatrixType2& At,
               cusp::coo_format,
               cusp::coo_format)
{
    At.resize(A.num_cols, A.num_rows, A.num_entries);

    typedef typename MatrixType2::index_type   IndexType;
    
    cusp::array1d<IndexType,cusp::host_memory> starting_pos(A.num_cols+1, 0);

    if( A.num_entries > 0 )
    {
	for( size_t i = 0; i < A.num_entries; i++ )
        {
	   IndexType col = A.column_indices[i];
	   starting_pos[col+1]++;
        }

	for( size_t i = 1; i < A.num_cols+1; i++ )
	   starting_pos[i] += starting_pos[i-1];

	for( size_t i = 0; i < A.num_entries; i++ )
        {
	   IndexType col = A.column_indices[i];
           IndexType j = starting_pos[col]++;

	   At.row_indices[j] = A.column_indices[i];
	   At.column_indices[j] = A.row_indices[i];
	   At.values[j] = A.values[i];
        }
    }
}

// CSR format
template <typename MatrixType1,   typename MatrixType2>
void transpose(const MatrixType1& A, MatrixType2& At,
               cusp::csr_format,
               cusp::csr_format)
{
    typedef typename MatrixType2::index_type   IndexType;

    At.resize(A.num_cols, A.num_rows, A.num_entries);

    if( A.num_entries > 0 )
    {
	for( size_t i = 0; i < At.num_rows+1; i++ )
	   At.row_offsets[i] = 0;

	for( size_t i = 0; i < At.num_entries; i++ )
	{
	   IndexType col = A.column_indices[i];
	   At.row_offsets[col+1]++;
	}

	for( size_t i = 1; i < At.num_rows+1; i++ )
	   At.row_offsets[i] += At.row_offsets[i-1];

	cusp::array1d<IndexType,cusp::host_memory> starting_pos( At.row_offsets );

	for( size_t row = 0; row < A.num_rows; row++ )
	{
	   IndexType row_start = A.row_offsets[row];
	   IndexType row_end   = A.row_offsets[row+1];

	   for( IndexType i = row_start; i < row_end; i++ )
           {
	      IndexType col = A.column_indices[i];
              IndexType j   = starting_pos[col]++;

	      At.column_indices[j] = row;
	      At.values[j] = A.values[i];
           }
	}
    }
}

} // end namespace host
} // end namespace detail
} // end namespace cusp

