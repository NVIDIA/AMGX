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

#include <cusp/detail/device/arch.h>
#include <cusp/detail/device/utils.h>
#include <cusp/detail/device/dereference.h>

#include <thrust/iterator/iterator_traits.h>

namespace cusp
{
namespace detail
{
namespace device
{
namespace cuda
{

template <int BLOCK_SIZE,
          typename SizeType,
          typename IndexIterator1,
          typename IndexIterator2,
          typename ValueIterator1,
          typename ValueIterator2,
          typename ValueIterator3,
          typename ValueIterator4,
          typename BinaryFunction1,
          typename BinaryFunction2>
__launch_bounds__(BLOCK_SIZE,1)
__global__
void spmv_csr_scalar_kernel(SizeType        num_rows,
                            IndexIterator1  row_offsets,
                            IndexIterator2  column_indices,
                            ValueIterator1  values,
                            ValueIterator2  x, 
                            ValueIterator3  y,
                            ValueIterator4  z,
                            BinaryFunction1 combine,
                            BinaryFunction2 reduce)
{
  typedef typename thrust::iterator_value<IndexIterator1>::type IndexType1;
  typedef typename thrust::iterator_value<IndexIterator2>::type IndexType2;
  typedef typename thrust::iterator_value<ValueIterator1>::type ValueType1;
  typedef typename thrust::iterator_value<ValueIterator2>::type ValueType2;
  typedef typename thrust::iterator_value<ValueIterator3>::type ValueType3;
  typedef typename thrust::iterator_value<ValueIterator4>::type ValueType4;

  const SizeType thread_id = BLOCK_SIZE * blockIdx.x + threadIdx.x;
  const SizeType grid_size = BLOCK_SIZE * gridDim.x;

  for(SizeType i = thread_id; i < num_rows; i += grid_size)
  {
    IndexIterator1 r0 = row_offsets; r0 += i;      IndexType1 row_start = CUSP_DEREFERENCE(r0); // row_offsets[i]
    IndexIterator1 r1 = row_offsets; r1 += i + 1;  IndexType1 row_end   = CUSP_DEREFERENCE(r1); // row_offsets[i + 1]
    ValueIterator3 y0 = y;           y0 += i;      ValueType4 sum       = CUSP_DEREFERENCE(y0); // sum = y[i]

    for (IndexType2 jj = row_start; jj < row_end; jj++)
    {
      IndexIterator2 c0 = column_indices; c0 += jj;  IndexType2 j    = CUSP_DEREFERENCE(c0);  // j    = column_indices[jj]
      ValueIterator1 v0 = values;         v0 += jj;  ValueType1 A_ij = CUSP_DEREFERENCE(v0);  // A_ij = values[jj]
      ValueIterator2 x0 = x;              x0 += j;   ValueType2 x_j  = CUSP_DEREFERENCE(x0);  // x_j  = x[j]

      sum = reduce(sum, combine(A_ij, x_j));                                                                     // sum += A_ij * x_j
    }

    ValueIterator4 z0 = z; z0 += i;  CUSP_DEREFERENCE(z0) = sum;                                // z[i] = sum
  }
}

    
template <typename SizeType,
          typename IndexIterator1,
          typename IndexIterator2,
          typename ValueIterator1,
          typename ValueIterator2,
          typename ValueIterator3,
          typename ValueIterator4,
          typename BinaryFunction1,
          typename BinaryFunction2>
void spmv_csr_scalar(SizeType        num_rows,
                     IndexIterator1  row_offsets,
                     IndexIterator2  column_indices,
                     ValueIterator1  values,
                     ValueIterator2  x, 
                     ValueIterator3  y,
                     ValueIterator4  z,
                     BinaryFunction1 combine,
                     BinaryFunction2 reduce)
{
    const SizeType block_size = 256;
    const SizeType max_blocks = cusp::detail::device::arch::max_active_blocks(spmv_csr_scalar_kernel<block_size, SizeType, IndexIterator1, IndexIterator2, ValueIterator1, ValueIterator2, ValueIterator3, ValueIterator4, BinaryFunction1, BinaryFunction2>, block_size, (size_t) 0);
    const SizeType num_blocks = std::min(max_blocks, DIVIDE_INTO(num_rows, block_size));
    
    spmv_csr_scalar_kernel<block_size><<<num_blocks, block_size>>>
        (num_rows,
         row_offsets, column_indices, values,
         x, y, z,
         combine, reduce);
}

} // end namespace cuda
} // end namespace device
} // end namespace detail
} // end namespace cusp

