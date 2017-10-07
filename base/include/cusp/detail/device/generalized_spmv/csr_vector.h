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

#include <cusp/csr_matrix.h>

#include <cusp/detail/device/common.h>
#include <cusp/detail/device/utils.h>
#include <cusp/detail/device/texture.h>

#include <thrust/experimental/arch.h>

namespace cusp
{
namespace detail
{
namespace device
{

//////////////////////////////////////////////////////////////////////////////
// CSR SpMV kernels based on a vector model (one warp per row)
//////////////////////////////////////////////////////////////////////////////
//
// spmv_csr_vector_device
//   Each row of the CSR matrix is assigned to a warp.  The warp computes
//   y[i] = A[i,:] * x, i.e. the dot product of the i-th row of A with 
//   the x vector, in parallel.  This division of work implies that 
//   the CSR index and data arrays (Aj and Ax) are accessed in a contiguous
//   manner (but generally not aligned).  On GT200 these accesses are
//   coalesced, unlike kernels based on the one-row-per-thread division of 
//   work.  Since an entire 32-thread warp is assigned to each row, many 
//   threads will remain idle when their row contains a small number 
//   of elements.  This code relies on implicit synchronization among 
//   threads in a warp.
//
// spmv_csr_vector_tex_device
//   Same as spmv_csr_vector_tex_device, except that the texture cache is 
//   used for accessing the x vector.
//  
//  Note: THREADS_PER_VECTOR must be one of [2,4,8,16,32]


template <typename IndexType, typename ValueType, unsigned int VECTORS_PER_BLOCK, unsigned int THREADS_PER_VECTOR, bool UseCache>
__global__ void
spmv_csr_vector_kernel(const IndexType num_rows,
                       const IndexType * Ap, 
                       const IndexType * Aj, 
                       const ValueType * Ax, 
                       const ValueType * x, 
                             ValueType * y)
{
    __shared__ volatile ValueType sdata[(VECTORS_PER_BLOCK + 1) * THREADS_PER_VECTOR];      // padded to avoid reduction conditionals
    __shared__ volatile IndexType ptrs[VECTORS_PER_BLOCK][2];
    
    const IndexType THREADS_PER_BLOCK = VECTORS_PER_BLOCK * THREADS_PER_VECTOR;

    const IndexType thread_id   = THREADS_PER_BLOCK * blockIdx.x + threadIdx.x;    // global thread index
    const IndexType thread_lane = threadIdx.x & (THREADS_PER_VECTOR - 1);          // thread index within the vector
    const IndexType vector_id   = thread_id   /  THREADS_PER_VECTOR;               // global vector index
    const IndexType vector_lane = threadIdx.x /  THREADS_PER_VECTOR;               // vector index within the block
    const IndexType num_vectors = VECTORS_PER_BLOCK * gridDim.x;                   // total number of active vectors

    for(IndexType row = vector_id; row < num_rows; row += num_vectors)
    {
        // use two threads to fetch Ap[row] and Ap[row+1]
        // this is considerably faster than the straightforward version
        if(thread_lane < 2)
            ptrs[vector_lane][thread_lane] = Ap[row + thread_lane];

        const IndexType row_start = ptrs[vector_lane][0];                   //same as: row_start = Ap[row];
        const IndexType row_end   = ptrs[vector_lane][1];                   //same as: row_end   = Ap[row+1];

        // initialize local sum
        ValueType sum = 0;
        
        // accumulate local sums
        for(IndexType jj = row_start + thread_lane; jj < row_end; jj += THREADS_PER_VECTOR)
            sum += Ax[jj] * fetch_x<UseCache>(Aj[jj], x);

        // store local sum in shared memory
        sdata[threadIdx.x] = sum;
        
        // reduce local sums to row sum
        if (THREADS_PER_VECTOR > 16) sdata[threadIdx.x] = sum = sum + sdata[threadIdx.x + 16];
        if (THREADS_PER_VECTOR >  8) sdata[threadIdx.x] = sum = sum + sdata[threadIdx.x +  8];
        if (THREADS_PER_VECTOR >  4) sdata[threadIdx.x] = sum = sum + sdata[threadIdx.x +  4];
        if (THREADS_PER_VECTOR >  2) sdata[threadIdx.x] = sum = sum + sdata[threadIdx.x +  2];
        if (THREADS_PER_VECTOR >  1) sdata[threadIdx.x] = sum = sum + sdata[threadIdx.x +  1];
       
//// Alternative method (slightly slower)
//      if (THREADS_PER_VECTOR > 16)  sdata[threadIdx.x] += sdata[threadIdx.x + 16];
//      if (THREADS_PER_VECTOR >  8)  sdata[threadIdx.x] += sdata[threadIdx.x +  8];
//      if (THREADS_PER_VECTOR >  4)  sdata[threadIdx.x] += sdata[threadIdx.x +  4];
//      if (THREADS_PER_VECTOR >  2)  sdata[threadIdx.x] += sdata[threadIdx.x +  2];
//      if (THREADS_PER_VECTOR >  1)  sdata[threadIdx.x] += sdata[threadIdx.x +  1];

        // first thread writes the result
        if (thread_lane == 0)
            y[row] += sdata[threadIdx.x];
    }
}

template <bool UseCache, typename IndexType, typename ValueType>
void __spmv_csr_vector(const csr_matrix<IndexType,ValueType,cusp::device_memory>& csr, 
                       const ValueType * x, 
                             ValueType * y)
{
    const unsigned int THREADS_PER_BLOCK  = 128;
    const unsigned int THREADS_PER_VECTOR = 32;
    const unsigned int VECTORS_PER_BLOCK  = THREADS_PER_BLOCK / THREADS_PER_VECTOR;

    //const unsigned int MAX_BLOCKS = MAX_THREADS / THREADS_PER_BLOCK;
    const unsigned int MAX_BLOCKS = thrust::experimental::arch::max_active_blocks(spmv_csr_vector_kernel<IndexType, ValueType, VECTORS_PER_BLOCK, THREADS_PER_VECTOR, UseCache>, THREADS_PER_BLOCK, (size_t) 0);
    const unsigned int NUM_BLOCKS = std::min(MAX_BLOCKS, DIVIDE_INTO(csr.num_rows, VECTORS_PER_BLOCK));
    
    if (UseCache)
        bind_x(x);

    spmv_csr_vector_kernel<IndexType, ValueType, VECTORS_PER_BLOCK, THREADS_PER_VECTOR, UseCache> <<<NUM_BLOCKS, THREADS_PER_BLOCK>>> 
        (csr.num_rows,
         thrust::raw_pointer_cast(&csr.row_offsets[0]),
         thrust::raw_pointer_cast(&csr.column_indices[0]),
         thrust::raw_pointer_cast(&csr.values[0]),
         x, y);

    if (UseCache)
        unbind_x(x);
}

template <typename IndexType, typename ValueType>
void spmv_csr_vector(const cusp::csr_matrix<IndexType,ValueType,cusp::device_memory>& csr, 
                     const ValueType * x, 
                           ValueType * y)
{
    __spmv_csr_vector<false>(csr, x, y);
}

template <typename IndexType, typename ValueType>
void spmv_csr_vector_tex(const csr_matrix<IndexType,ValueType,cusp::device_memory>& csr, 
                         const ValueType * x, 
                               ValueType * y)
{
    __spmv_csr_vector<true>(csr, x, y);
}

} // end namespace device
} // end namespace detail
} // end namespace cusp

