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

// XXX this code sometimes fails on CUDA r4.1 for an unknown reason

#include <cusp/array1d.h>
#include <cusp/detail/device/arch.h>
#include <cusp/detail/device/utils.h>
#include <cusp/detail/device/dereference.h>

#include <thrust/copy.h>
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
          typename IndexType,
          typename ValueType,
          typename BinaryFunction>
          __forceinline__
          __device__
void scan_by_key(const IndexType * rows, ValueType * vals, BinaryFunction binary_op)
{
    const IndexType row = rows[threadIdx.x];
          ValueType val = vals[threadIdx.x];

    if (BLOCK_SIZE >   1) { if(threadIdx.x >=   1 && row == rows[threadIdx.x -   1 ]) { val = binary_op(vals[threadIdx.x -   1], val); } __syncthreads(); vals[threadIdx.x] = val; __syncthreads(); }
    if (BLOCK_SIZE >   2) { if(threadIdx.x >=   2 && row == rows[threadIdx.x -   2 ]) { val = binary_op(vals[threadIdx.x -   2], val); } __syncthreads(); vals[threadIdx.x] = val; __syncthreads(); }
    if (BLOCK_SIZE >   4) { if(threadIdx.x >=   4 && row == rows[threadIdx.x -   4 ]) { val = binary_op(vals[threadIdx.x -   4], val); } __syncthreads(); vals[threadIdx.x] = val; __syncthreads(); }
    if (BLOCK_SIZE >   8) { if(threadIdx.x >=   8 && row == rows[threadIdx.x -   8 ]) { val = binary_op(vals[threadIdx.x -   8], val); } __syncthreads(); vals[threadIdx.x] = val; __syncthreads(); }
    if (BLOCK_SIZE >  16) { if(threadIdx.x >=  16 && row == rows[threadIdx.x -  16 ]) { val = binary_op(vals[threadIdx.x -  16], val); } __syncthreads(); vals[threadIdx.x] = val; __syncthreads(); }
    if (BLOCK_SIZE >  32) { if(threadIdx.x >=  32 && row == rows[threadIdx.x -  32 ]) { val = binary_op(vals[threadIdx.x -  32], val); } __syncthreads(); vals[threadIdx.x] = val; __syncthreads(); }
    if (BLOCK_SIZE >  64) { if(threadIdx.x >=  64 && row == rows[threadIdx.x -  64 ]) { val = binary_op(vals[threadIdx.x -  64], val); } __syncthreads(); vals[threadIdx.x] = val; __syncthreads(); }
    if (BLOCK_SIZE > 128) { if(threadIdx.x >= 128 && row == rows[threadIdx.x - 128 ]) { val = binary_op(vals[threadIdx.x - 128], val); } __syncthreads(); vals[threadIdx.x] = val; __syncthreads(); }
    if (BLOCK_SIZE > 256) { if(threadIdx.x >= 256 && row == rows[threadIdx.x - 256 ]) { val = binary_op(vals[threadIdx.x - 256], val); } __syncthreads(); vals[threadIdx.x] = val; __syncthreads(); }  
    if (BLOCK_SIZE > 512) { if(threadIdx.x >= 512 && row == rows[threadIdx.x - 512 ]) { val = binary_op(vals[threadIdx.x - 512], val); } __syncthreads(); vals[threadIdx.x] = val; __syncthreads(); }  
}

template <int BLOCK_SIZE,
          typename IndexType,
          typename ValueType,
          typename BinaryFunction,
          typename SizeType>
          __forceinline__
          __device__
void scan_by_key(const IndexType * rows, ValueType * vals, BinaryFunction binary_op, SizeType N)
{
    const IndexType row = rows[threadIdx.x];
          ValueType val = vals[threadIdx.x];

    if (BLOCK_SIZE >   1) { if(threadIdx.x < N && threadIdx.x >=   1 && row == rows[threadIdx.x -   1 ]) { val = binary_op(vals[threadIdx.x -   1], val); } __syncthreads(); vals[threadIdx.x] = val; __syncthreads(); }
    if (BLOCK_SIZE >   2) { if(threadIdx.x < N && threadIdx.x >=   2 && row == rows[threadIdx.x -   2 ]) { val = binary_op(vals[threadIdx.x -   2], val); } __syncthreads(); vals[threadIdx.x] = val; __syncthreads(); }
    if (BLOCK_SIZE >   4) { if(threadIdx.x < N && threadIdx.x >=   4 && row == rows[threadIdx.x -   4 ]) { val = binary_op(vals[threadIdx.x -   4], val); } __syncthreads(); vals[threadIdx.x] = val; __syncthreads(); }
    if (BLOCK_SIZE >   8) { if(threadIdx.x < N && threadIdx.x >=   8 && row == rows[threadIdx.x -   8 ]) { val = binary_op(vals[threadIdx.x -   8], val); } __syncthreads(); vals[threadIdx.x] = val; __syncthreads(); }
    if (BLOCK_SIZE >  16) { if(threadIdx.x < N && threadIdx.x >=  16 && row == rows[threadIdx.x -  16 ]) { val = binary_op(vals[threadIdx.x -  16], val); } __syncthreads(); vals[threadIdx.x] = val; __syncthreads(); }
    if (BLOCK_SIZE >  32) { if(threadIdx.x < N && threadIdx.x >=  32 && row == rows[threadIdx.x -  32 ]) { val = binary_op(vals[threadIdx.x -  32], val); } __syncthreads(); vals[threadIdx.x] = val; __syncthreads(); }
    if (BLOCK_SIZE >  64) { if(threadIdx.x < N && threadIdx.x >=  64 && row == rows[threadIdx.x -  64 ]) { val = binary_op(vals[threadIdx.x -  64], val); } __syncthreads(); vals[threadIdx.x] = val; __syncthreads(); }
    if (BLOCK_SIZE > 128) { if(threadIdx.x < N && threadIdx.x >= 128 && row == rows[threadIdx.x - 128 ]) { val = binary_op(vals[threadIdx.x - 128], val); } __syncthreads(); vals[threadIdx.x] = val; __syncthreads(); }
    if (BLOCK_SIZE > 256) { if(threadIdx.x < N && threadIdx.x >= 256 && row == rows[threadIdx.x - 256 ]) { val = binary_op(vals[threadIdx.x - 256], val); } __syncthreads(); vals[threadIdx.x] = val; __syncthreads(); }  
    if (BLOCK_SIZE > 512) { if(threadIdx.x < N && threadIdx.x >= 512 && row == rows[threadIdx.x - 512 ]) { val = binary_op(vals[threadIdx.x - 512], val); } __syncthreads(); vals[threadIdx.x] = val; __syncthreads(); }  
}


template <int BLOCK_SIZE,
          typename SizeType,
          typename ValueIterator4,
          typename BinaryFunction2,
          typename OutputIterator1,
          typename OutputIterator2>
__launch_bounds__(BLOCK_SIZE,1)
__global__
void spmv_coo_kernel_postprocess(SizeType        num_outputs,
                                 ValueIterator4  z,
                                 BinaryFunction2 reduce,
                                 OutputIterator1 row_carries,
                                 OutputIterator2 val_carries)
{
  typedef typename thrust::iterator_value<OutputIterator1>::type IndexType;
  typedef typename thrust::iterator_value<OutputIterator2>::type ValueType;

  __shared__ IndexType rows[BLOCK_SIZE + 1];
  __shared__ ValueType vals[BLOCK_SIZE + 1];
  
  __shared__ IndexType row_carry;
  __shared__ ValueType val_carry;

  __syncthreads();

  SizeType base = 0;

  row_carries += threadIdx.x;
  val_carries += threadIdx.x;

  while(base + BLOCK_SIZE <= num_outputs)
  {
    rows[threadIdx.x] = CUSP_DEREFERENCE(row_carries);
    vals[threadIdx.x] = CUSP_DEREFERENCE(val_carries);

    // carry in
    if(threadIdx.x == 0 && base != 0)
    {
      if (row_carry == rows[0])
      {
        vals[0] = reduce(val_carry, vals[0]);
      }
      else
      {
        // row terminates in previous unit
        ValueIterator4 zj = z + row_carry;
        CUSP_DEREFERENCE(zj) = reduce(CUSP_DEREFERENCE(zj), val_carry);
      }
    }

    __syncthreads();

    scan_by_key<BLOCK_SIZE>(rows, vals, reduce);

    if (threadIdx.x == BLOCK_SIZE - 1)
    {
      row_carry = rows[BLOCK_SIZE - 1];
      val_carry = vals[BLOCK_SIZE - 1];
    }
    else
    {
      if(rows[threadIdx.x] != rows[threadIdx.x + 1])
      {
        // row terminates in current unit
        ValueIterator4 zj = z + rows[threadIdx.x];
        CUSP_DEREFERENCE(zj) = reduce(CUSP_DEREFERENCE(zj), vals[threadIdx.x]);
      }
    }

    base        += BLOCK_SIZE;
    row_carries += BLOCK_SIZE;
    val_carries += BLOCK_SIZE;

    __syncthreads();
  }

  if(base < num_outputs)
  {
    SizeType offset_end = num_outputs - base;

    if (threadIdx.x < offset_end)
    {
      rows[threadIdx.x] = CUSP_DEREFERENCE(row_carries);
      vals[threadIdx.x] = CUSP_DEREFERENCE(val_carries);
    }

    // carry in
    if(threadIdx.x == 0 && base != 0)
    {
      if (row_carry == rows[0])
      {
        vals[0] = reduce(val_carry, vals[0]);
      }
      else
      {
        // row terminates in previous unit
        ValueIterator4 zj = z + row_carry;
        CUSP_DEREFERENCE(zj) = reduce(CUSP_DEREFERENCE(zj), val_carry);
      }
    }

    __syncthreads();

    scan_by_key<BLOCK_SIZE>(rows, vals, reduce, offset_end);
    
    if(threadIdx.x < offset_end)
    {
      if(threadIdx.x == offset_end - 1 || rows[threadIdx.x] != rows[threadIdx.x + 1])
      {
        // row terminates in current unit or we're at the end of input
        ValueIterator4 zj = z + rows[threadIdx.x];
        CUSP_DEREFERENCE(zj) = reduce(CUSP_DEREFERENCE(zj), vals[threadIdx.x]);
      }
    }
  }
}

template <int BLOCK_SIZE,
          int K,
          typename SizeType,
          typename IndexIterator1,
          typename IndexIterator2,
          typename ValueIterator1,
          typename ValueIterator2,
          typename ValueIterator4,
          typename BinaryFunction1,
          typename BinaryFunction2,
          typename OutputIterator1,
          typename OutputIterator2>
__launch_bounds__(BLOCK_SIZE,1)
__global__
void spmv_coo_kernel(SizeType        num_entries,
                     SizeType        interval_size,
                     IndexIterator1  row_indices,
                     IndexIterator2  column_indices,
                     ValueIterator1  values,
                     ValueIterator2  x, 
                     ValueIterator4  z,
                     BinaryFunction1 combine,
                     BinaryFunction2 reduce,
                     OutputIterator1 row_carries,
                     OutputIterator2 val_carries)
{
  typedef typename thrust::iterator_value<IndexIterator1>::type IndexType1;
  typedef typename thrust::iterator_value<IndexIterator2>::type IndexType2;
  typedef typename thrust::iterator_value<ValueIterator1>::type ValueType1;
  typedef typename thrust::iterator_value<ValueIterator2>::type ValueType2;
  typedef typename thrust::iterator_value<ValueIterator4>::type ValueType4;

  __shared__ IndexType1 rows[K][BLOCK_SIZE + 1];
  __shared__ ValueType4 vals[K][BLOCK_SIZE + 1];

  __shared__ IndexType1 row_carry;
  __shared__ ValueType4 val_carry;

  __syncthreads(); // is this really necessary?

  SizeType interval_begin = interval_size * blockIdx.x;
  SizeType interval_end   = thrust::min(interval_begin + interval_size, num_entries);

  SizeType unit_size = K * BLOCK_SIZE;
 
  SizeType base = interval_begin;

  row_indices    += base;
  column_indices += base;
  values         += base;

  // process full units
  while(base + unit_size <= interval_end)
  {
    // read data
    for(int k = 0; k < K; k++)
    {
      int offset = BLOCK_SIZE * k + threadIdx.x;

      IndexIterator1 _i   = row_indices    + offset;
      IndexIterator2 _j   = column_indices + offset;
      ValueIterator1 _Aij = values         + offset;

      ValueIterator2 _xj = x + CUSP_DEREFERENCE(_j);

      rows[offset % K][offset / K] = CUSP_DEREFERENCE(_i);
      vals[offset % K][offset / K] = combine(CUSP_DEREFERENCE(_Aij), CUSP_DEREFERENCE(_xj));
    }

    // carry in
    if (threadIdx.x == 0 && base != interval_begin)
    {
      if (row_carry == rows[0][0])
      {
        // row continues into this unit
        vals[0][0] = reduce(val_carry, vals[0][0]);
      }
      else
      {
        // row terminates in previous unit
        ValueIterator4 _zj = z + row_carry;
        CUSP_DEREFERENCE(_zj) = reduce(CUSP_DEREFERENCE(_zj), val_carry);
      }
    }
      
    __syncthreads();

    // process local values
    for(int k = 1; k < K; k++)
    {
      if (rows[k][threadIdx.x] == rows[k - 1][threadIdx.x])
        vals[k][threadIdx.x] = reduce(vals[k - 1][threadIdx.x], vals[k][threadIdx.x]);
    }

    __syncthreads();

    // process across block
    scan_by_key<BLOCK_SIZE>(rows[K - 1], vals[K - 1], reduce);

    if (threadIdx.x == 0)
    {
      // update carry and sentinel value
      row_carry = rows[0][BLOCK_SIZE] = rows[K - 1][BLOCK_SIZE - 1];
      val_carry = vals[0][BLOCK_SIZE] = vals[K - 1][BLOCK_SIZE - 1];
    }
    else
    {
      // update local values
      for(int k = 0; k < K - 1; k++)
      {
        IndexType1 row = rows[K - 1][threadIdx.x - 1];
        ValueType4 val = vals[K - 1][threadIdx.x - 1];

        if(rows[k][threadIdx.x] == row)
        {
          vals[k][threadIdx.x] = reduce(val, vals[k][threadIdx.x]);
        }
      }
    }

    __syncthreads();

    // write data
    for(int k = 0; k < K; k++)
    {
      int offset = BLOCK_SIZE * k + threadIdx.x;

      if (rows[offset % K][offset / K] != rows[(offset + 1) % K][(offset + 1) / K])
      {
        // row terminates
        ValueIterator4 _zj = z + rows[offset % K][offset / K];
        CUSP_DEREFERENCE(_zj) = reduce(CUSP_DEREFERENCE(_zj), vals[offset % K][offset / K]);
      }
    }

    // advance iterators
    base           += unit_size;
    row_indices    += unit_size;
    column_indices += unit_size;
    values         += unit_size;

    __syncthreads();
  }

  // process partial unit
  if(base < interval_end)
  {
    int offset_end = interval_end - base;

    // read data
    for(int k = 0; k < K; k++)
    {
      int offset = BLOCK_SIZE * k + threadIdx.x;

      if (offset < offset_end)
      {
        IndexIterator1 _i   = row_indices    + offset;
        IndexIterator2 _j   = column_indices + offset;
        ValueIterator1 _Aij = values         + offset;

        ValueIterator2 _xj = x + CUSP_DEREFERENCE(_j);

        rows[offset % K][offset / K] = CUSP_DEREFERENCE(_i);
        vals[offset % K][offset / K] = combine(CUSP_DEREFERENCE(_Aij), CUSP_DEREFERENCE(_xj));
      }
    }
    
    // carry in
    if (threadIdx.x == 0 && base != interval_begin)
    {
      if (row_carry == rows[0][0])
      {
        // row continues into this unit
        vals[0][0] = reduce(val_carry, vals[0][0]);
      }
      else
      {
        // row terminates in previous unit
        ValueIterator4 _zj = z + row_carry;
        CUSP_DEREFERENCE(_zj) = reduce(CUSP_DEREFERENCE(_zj), val_carry);
      }
    }
      
    __syncthreads();

    // process local values
    for(int k = 1; k < K; k++)
    {
      int offset = K * threadIdx.x + k;

      if (offset < offset_end)
      {
        if (rows[k][threadIdx.x] == rows[k - 1][threadIdx.x])
          vals[k][threadIdx.x] = reduce(vals[k - 1][threadIdx.x], vals[k][threadIdx.x]);
      }
    }

    __syncthreads();

    // process across block
    scan_by_key<BLOCK_SIZE>(rows[K - 1], vals[K - 1], reduce, offset_end);

    if (threadIdx.x == 0)
    {
      // update carry and sentinel value
      row_carry = rows[offset_end % K][offset_end / K] = rows[(offset_end - 1) % K][(offset_end - 1) / K];
      val_carry = vals[offset_end % K][offset_end / K] = vals[(offset_end - 1) % K][(offset_end - 1) / K];
    }
    else
    {
      // update local values
      for(int k = 0; k < K - 1; k++)
      {
        int offset = K * threadIdx.x + k;

        if (offset < offset_end)
        {
          IndexType1 row = rows[K - 1][threadIdx.x - 1];
          ValueType4 val = vals[K - 1][threadIdx.x - 1];

          if(rows[k][threadIdx.x] == row)
          {
            vals[k][threadIdx.x] = reduce(val, vals[k][threadIdx.x]);
          }
        }
      }
    }

    __syncthreads();

    // write data
    for(int k = 0; k < K; k++)
    {
      int offset = BLOCK_SIZE * k + threadIdx.x;
       
      if (offset < offset_end)
      {
        if (rows[offset % K][offset / K] != rows[(offset + 1) % K][(offset + 1) / K])
        {
          // row terminates
          ValueIterator4 _zj = z + rows[offset % K][offset / K];
          CUSP_DEREFERENCE(_zj) = reduce(CUSP_DEREFERENCE(_zj), vals[offset % K][offset / K]);
        }
      }
    }
  }

  __syncthreads();

  if (threadIdx.x == 0)
  {
    // write interval carry out
    row_carries += blockIdx.x;
    val_carries += blockIdx.x;

    CUSP_DEREFERENCE(row_carries) = row_carry;
    CUSP_DEREFERENCE(val_carries) = val_carry;
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
void spmv_coo(SizeType        num_rows,
              SizeType        num_entries,
              IndexIterator1  row_indices,
              IndexIterator2  column_indices,
              ValueIterator1  values,
              ValueIterator2  x, 
              ValueIterator3  y,
              ValueIterator4  z,
              BinaryFunction1 combine,
              BinaryFunction2 reduce)
{
  typedef typename thrust::iterator_value<IndexIterator2>::type IndexType2;
  typedef typename thrust::iterator_value<ValueIterator4>::type ValueType4;
  typedef typename cusp::array1d<IndexType2, cusp::device_memory>::iterator OutputIterator1;
  typedef typename cusp::array1d<ValueType4, cusp::device_memory>::iterator OutputIterator2;

  thrust::copy(y, y + num_rows, z);

  if (num_entries == 0) return;

  // TODO pick K and/or block_size based on sizeof(ValueType4)
  const SizeType K          = 3;
  const SizeType block_size = 128;
  const SizeType unit_size  = K * block_size;
  const SizeType max_blocks = cusp::detail::device::arch::max_active_blocks(spmv_coo_kernel<block_size, K, SizeType, IndexIterator1, IndexIterator2, ValueIterator1, ValueIterator2, ValueIterator4, BinaryFunction1, BinaryFunction2, OutputIterator1, OutputIterator2>, block_size, (size_t) 0);

  thrust::pair<SizeType, SizeType> splitting = cusp::detail::device::uniform_splitting<SizeType>(num_entries, unit_size, max_blocks);
  const SizeType interval_size = splitting.first;
  const SizeType num_blocks    = splitting.second;

  cusp::array1d<IndexType2, cusp::device_memory> row_carries(num_blocks);
  cusp::array1d<ValueType4, cusp::device_memory> val_carries(num_blocks);

  // note: we don't need to pass y
  spmv_coo_kernel<block_size, K><<<num_blocks, block_size>>>
    (num_entries,
     interval_size,
     row_indices, column_indices, values,
     x, z,
     combine, reduce,
     row_carries.begin(), val_carries.begin());

  // process the per-interval results
  spmv_coo_kernel_postprocess<block_size><<<1,block_size>>>
    (num_blocks, z, reduce,
     row_carries.begin(), val_carries.begin());
}

} // end namespace cuda
} // end namespace device
} // end namespace detail
} // end namespace cusp

