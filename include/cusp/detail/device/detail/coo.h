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

#include <cusp/array1d.h>
#include <cusp/detail/format_utils.h>

#include <thrust/functional.h>
#include <thrust/reduce.h>
#include <thrust/inner_product.h>
#include <thrust/iterator/zip_iterator.h>

namespace cusp
{
namespace detail
{
namespace device
{
namespace detail
{

// simplified implementation for plus and minus operators
template <typename Matrix1,
          typename Matrix2,
          typename Matrix3,
          typename UnaryFunction>
void coo_elementwise_transform_simple(const Matrix1& A,
                                      const Matrix2& B,
                                            Matrix3& C,
                                      UnaryFunction op)
{
    typedef typename Matrix3::index_type   IndexType;
    typedef typename Matrix3::value_type   ValueType;
    typedef typename Matrix3::memory_space MemorySpace;
  
    IndexType A_nnz = A.num_entries;
    IndexType B_nnz = B.num_entries;

    if (A_nnz == 0 && B_nnz == 0)
    {
      C.resize(A.num_rows, A.num_cols, 0);
      return;
    }

    cusp::array1d<IndexType,MemorySpace> rows(A_nnz + B_nnz);
    cusp::array1d<IndexType,MemorySpace> cols(A_nnz + B_nnz);
    cusp::array1d<ValueType,MemorySpace> vals(A_nnz + B_nnz);

    thrust::copy(A.row_indices.begin(),    A.row_indices.end(),    rows.begin());
    thrust::copy(B.row_indices.begin(),    B.row_indices.end(),    rows.begin() + A_nnz);
    thrust::copy(A.column_indices.begin(), A.column_indices.end(), cols.begin());
    thrust::copy(B.column_indices.begin(), B.column_indices.end(), cols.begin() + A_nnz);
    thrust::copy(A.values.begin(),         A.values.end(),         vals.begin());

    // apply transformation to B's values 
    thrust::transform(B.values.begin(), B.values.end(), vals.begin() + A_nnz, op);

    // sort by (I,J)
    cusp::detail::sort_by_row_and_column(rows, cols, vals);

    // compute unique number of nonzeros in the output
    IndexType C_nnz = thrust::inner_product(thrust::make_zip_iterator(thrust::make_tuple(rows.begin(), cols.begin())),
                                            thrust::make_zip_iterator(thrust::make_tuple(rows.end (),  cols.end()))   - 1,
                                            thrust::make_zip_iterator(thrust::make_tuple(rows.begin(), cols.begin())) + 1,
                                            IndexType(1),
                                            thrust::plus<IndexType>(),
                                            thrust::not_equal_to< thrust::tuple<IndexType,IndexType> >());

    // allocate space for output
    C.resize(A.num_rows, A.num_cols, C_nnz);

    // sum values with the same (i,j)
    thrust::reduce_by_key(thrust::make_zip_iterator(thrust::make_tuple(rows.begin(), cols.begin())),
                          thrust::make_zip_iterator(thrust::make_tuple(rows.end(),   cols.end())),
                          vals.begin(),
                          thrust::make_zip_iterator(thrust::make_tuple(C.row_indices.begin(), C.column_indices.begin())),
                          C.values.begin(),
                          thrust::equal_to< thrust::tuple<IndexType,IndexType> >(),
                          thrust::plus<ValueType>());
}

template <typename Matrix1,
          typename Matrix2,
          typename Matrix3>
void coo_add(const Matrix1& A,
             const Matrix2& B,
                   Matrix3& C)
{
    typedef typename Matrix2::value_type ValueType;

    coo_elementwise_transform_simple(A, B, C, thrust::identity<ValueType>());
}

template <typename Matrix1,
          typename Matrix2,
          typename Matrix3>
void coo_subtract(const Matrix1& A,
                  const Matrix2& B,
                        Matrix3& C)
{
    typedef typename Matrix2::value_type ValueType;

    coo_elementwise_transform_simple(A, B, C, thrust::negate<ValueType>());
}


} // end namespace detail
} // end namespace device
} // end namespace detail
} // end namespace cusp

