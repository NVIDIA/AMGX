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
#include <cusp/detail/device/common.h>
#include <cusp/detail/device/utils.h>
#include <cusp/detail/device/texture.h>

#include <thrust/device_ptr.h>

// SpMV kernel for the ELLPACK/ITPACK matrix format.

namespace cusp
{
namespace detail
{
namespace device
{

template <typename IndexType, typename ValueType, size_t BLOCK_SIZE, bool UseCache>
__launch_bounds__(BLOCK_SIZE,1)
__global__ void
spmv_ell_kernel(const IndexType num_rows, 
                const IndexType num_cols, 
                const IndexType num_cols_per_row,
                const IndexType pitch,
                const IndexType * Aj,
                const ValueType * Ax, 
                const ValueType * x, 
                      ValueType * y)
{
    const IndexType invalid_index = cusp::ell_matrix<IndexType, ValueType, cusp::device_memory>::invalid_index;

    const IndexType thread_id = blockDim.x * blockIdx.x + threadIdx.x;
    const IndexType grid_size = gridDim.x * blockDim.x;

    for(IndexType row = thread_id; row < num_rows; row += grid_size)
    {
        ValueType sum = 0;

        IndexType offset = row;

        for(IndexType n = 0; n < num_cols_per_row; n++)
        {
            const IndexType col = Aj[offset];

            if (col != invalid_index)
            {
                const ValueType A_ij = Ax[offset];
                sum += A_ij * fetch_x<UseCache>(col, x);
            }

            offset += pitch;
        }

        y[row] = sum;
    }
}


template <bool UseCache,
          typename Matrix,
          typename ValueType>
void __spmv_ell(const Matrix&    A, 
                const ValueType* x, 
                      ValueType* y)
{
    typedef typename Matrix::index_type IndexType;

    const size_t BLOCK_SIZE = 256;
    const size_t MAX_BLOCKS = cusp::detail::device::arch::max_active_blocks(spmv_ell_kernel<IndexType,ValueType,BLOCK_SIZE,UseCache>, BLOCK_SIZE, (size_t) 0);
    const size_t NUM_BLOCKS = std::min<size_t>(MAX_BLOCKS, DIVIDE_INTO(A.num_rows, BLOCK_SIZE));

    const IndexType pitch               = A.column_indices.pitch;
    const IndexType num_entries_per_row = A.column_indices.num_cols;

    // TODO generalize this
    assert(A.column_indices.pitch == A.values.pitch);
    
    if (UseCache)
        bind_x(x);

    spmv_ell_kernel<IndexType,ValueType,BLOCK_SIZE,UseCache> <<<NUM_BLOCKS, BLOCK_SIZE>>>
        (A.num_rows, A.num_cols,
         num_entries_per_row, pitch,
         thrust::raw_pointer_cast(&A.column_indices.values[0]), 
         thrust::raw_pointer_cast(&A.values.values[0]),
         x, y);

    if (UseCache)
        unbind_x(x);
}

template <typename Matrix,
          typename ValueType>
void spmv_ell(const Matrix&    A, 
              const ValueType* x, 
                    ValueType* y)
{
    __spmv_ell<false>(A, x, y);
}

template <typename Matrix,
          typename ValueType>
void spmv_ell_tex(const Matrix&    A,
                  const ValueType* x, 
                        ValueType* y)
{
    __spmv_ell<true>(A, x, y);
}

} // end namespace device
} // end namespace detail
} // end namespace cusp

