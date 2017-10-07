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
#include <cusp/detail/device/texture.h>

#include <thrust/device_ptr.h>

namespace cusp
{
namespace detail
{
namespace device
{

////////////////////////////////////////////////////////////////////////
// CSR SpMV kernels based on a scalar model (one thread per row)
///////////////////////////////////////////////////////////////////////
//
// spmv_csr_scalar_device
//   Straightforward translation of standard CSR SpMV to CUDA
//   where each thread computes y[i] = A[i,:] * x 
//   (the dot product of the i-th row of A with the x vector)
//
// spmv_csr_scalar_tex_device
//   Same as spmv_csr_scalar_device, except x is accessed via texture cache.
//

template <bool UseCache,
          typename IndexType,
          typename ValueType>
__global__ void
spmv_csr_scalar_kernel(const IndexType num_rows,
                       const IndexType * Ap, 
                       const IndexType * Aj, 
                       const ValueType * Ax, 
                       const ValueType * x, 
                             ValueType * y)
{
    const IndexType thread_id = blockDim.x * blockIdx.x + threadIdx.x;
    const IndexType grid_size = gridDim.x * blockDim.x;

    for(IndexType row = thread_id; row < num_rows; row += grid_size)
    {
        const IndexType row_start = Ap[row];
        const IndexType row_end   = Ap[row+1];
        
        ValueType sum = 0;
    
        for (IndexType jj = row_start; jj < row_end; jj++)
            sum += Ax[jj] * fetch_x<UseCache>(Aj[jj], x);       

        y[row] = sum;
    }
}

    
template <bool UseCache,
          typename Matrix,
          typename ValueType>
void __spmv_csr_scalar(const Matrix&    A,
                       const ValueType* x, 
                             ValueType* y)
{
    typedef typename Matrix::index_type IndexType;

    const size_t BLOCK_SIZE = 256;
    const size_t MAX_BLOCKS = cusp::detail::device::arch::max_active_blocks(spmv_csr_scalar_kernel<UseCache, IndexType, ValueType>, BLOCK_SIZE, (size_t) 0);
    const size_t NUM_BLOCKS = std::min(MAX_BLOCKS, DIVIDE_INTO(A.num_rows, BLOCK_SIZE));
    
    if (UseCache)
        bind_x(x);

    spmv_csr_scalar_kernel<UseCache,IndexType,ValueType> <<<NUM_BLOCKS, BLOCK_SIZE>>> 
        (A.num_rows,
         thrust::raw_pointer_cast(&A.row_offsets[0]),
         thrust::raw_pointer_cast(&A.column_indices[0]),
         thrust::raw_pointer_cast(&A.values[0]),
         x, y);

    if (UseCache)
        unbind_x(x);
}

template <typename Matrix,
          typename ValueType>
void spmv_csr_scalar(const Matrix&    A,
                     const ValueType* x, 
                           ValueType* y)
{
    __spmv_csr_scalar<false>(A, x, y);
}

template <typename Matrix,
          typename ValueType>
void spmv_csr_scalar_tex(const Matrix&    A,
                         const ValueType* x, 
                               ValueType* y)
{
    __spmv_csr_scalar<true>(A, x, y);
}

} // end namespace device
} // end namespace detail
} // end namespace cusp

