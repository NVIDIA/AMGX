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

#include <cusp/ell_matrix.h>

#include <cusp/detail/device/common.h>
#include <cusp/detail/device/utils.h>
#include <cusp/detail/device/texture.h>

#include <thrust/functional.h>
#include <thrust/experimental/arch.h>

// SpMV kernel for the ELLPACK/ITPACK matrix format.

namespace cusp
{
namespace detail
{
namespace device
{

template <bool UseCache,
          typename IndexType,
          typename ValueType,
          typename UnaryFunction,
          typename BinaryFunction1,
          typename BinaryFunction2>
__global__ 
void spmv_ell_kernel(const IndexType num_rows, 
                     const IndexType num_cols, 
                     const IndexType num_cols_per_row,
                     const IndexType stride,
                     const IndexType * Aj,
                     const ValueType * Ax, 
                     const ValueType * x, 
                           ValueType * y,
                     UnaryFunction   initialize,
                     BinaryFunction1 combine,
                     BinaryFunction2 reduce)
{
    const IndexType invalid_index = cusp::ell_matrix<IndexType, ValueType, cusp::device_memory>::invalid_index;

    const IndexType thread_id = blockDim.x * blockIdx.x + threadIdx.x;
    const IndexType grid_size = gridDim.x * blockDim.x;

    for(IndexType row = thread_id; row < num_rows; row += grid_size)
    {
        ValueType sum = initialize(y[row]);

        IndexType offset = row;

        for(IndexType n = 0; n < num_cols_per_row; n++)
        {
            const IndexType col = Aj[offset];

            if (col != invalid_index)
            {
                const ValueType A_ij = Ax[offset];
                sum = reduce(sum, combine(A_ij, fetch_x<UseCache>(col, x)));
            }

            offset += stride;
        }

        y[row] = sum;
    }
}


template <bool UseCache, typename IndexType, typename ValueType>
void __spmv_ell(const cusp::ell_matrix<IndexType,ValueType,cusp::device_memory>& ell, 
                const ValueType * x, 
                      ValueType * y)
{
    const unsigned int BLOCK_SIZE = 256;
    const unsigned int MAX_BLOCKS = MAX_THREADS / BLOCK_SIZE;
//    const unsigned int MAX_BLOCKS = thrust::experimental::arch::max_active_blocks(spmv_ell_kernel<IndexType, ValueType, UseCache>, BLOCK_SIZE, (size_t) 0);
    const unsigned int NUM_BLOCKS = std::min(MAX_BLOCKS, DIVIDE_INTO(ell.num_rows, BLOCK_SIZE));

    const IndexType stride              = ell.column_indices.num_rows;
    const IndexType num_entries_per_row = ell.column_indices.num_cols;
    
    if (UseCache)
        bind_x(x);

    spmv_ell_kernel<UseCache> <<<NUM_BLOCKS, BLOCK_SIZE>>>
        (ell.num_rows, ell.num_cols,
         num_entries_per_row, stride,
         thrust::raw_pointer_cast(&ell.column_indices.values[0]), 
         thrust::raw_pointer_cast(&ell.values.values[0]),
         x, y,
         thrust::identity<ValueType>(), thrust::multiplies<ValueType>(), thrust::plus<ValueType>());

    if (UseCache)
        unbind_x(x);
}

template <typename IndexType, typename ValueType>
void spmv_ell(const cusp::ell_matrix<IndexType,ValueType,cusp::device_memory>& ell, 
              const ValueType * x, 
                    ValueType * y)
{
    __spmv_ell<false>(ell, x, y);
}

template <typename IndexType, typename ValueType>
void spmv_ell_tex(const cusp::ell_matrix<IndexType,ValueType,cusp::device_memory>& ell, 
                  const ValueType * x, 
                        ValueType * y)
{
    __spmv_ell<true>(ell, x, y);
}

template <typename IndexType, typename ValueType>
void spmv(const cusp::ell_matrix<IndexType,ValueType,cusp::device_memory>& ell, 
          const ValueType * x, 
                ValueType * y)
{
    spmv_ell(ell, x, y);
}

template <typename IndexType, typename ValueType>
void spmv_tex(const cusp::ell_matrix<IndexType,ValueType,cusp::device_memory>& ell, 
              const ValueType * x, 
                    ValueType * y)
{
    spmv_ell_tex(ell, x, y);
}

} // end namespace device
} // end namespace detail
} // end namespace cusp

