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

#include <thrust/extrema.h>

#include <cusp/detail/device/arch.h>
#include <cusp/detail/device/common.h>
#include <cusp/detail/device/utils.h>
#include <cusp/detail/device/texture.h>
#include <cusp/detail/device/spmv/coo_serial.h>

#include <thrust/device_ptr.h>

// Note: Unlike the other kernels this kernel implements y += A*x

namespace cusp
{
namespace detail
{
namespace device
{

// segmented reduction in shared memory
template <typename IndexType, typename ValueType>
__device__ ValueType segreduce_warp(const IndexType thread_lane, IndexType row, ValueType val, IndexType * rows, ValueType * vals)
{
    rows[threadIdx.x] = row;
    vals[threadIdx.x] = val;

    if( thread_lane >=  1 && row == rows[threadIdx.x -  1] ) { vals[threadIdx.x] = val = val + vals[threadIdx.x -  1]; } 
    if( thread_lane >=  2 && row == rows[threadIdx.x -  2] ) { vals[threadIdx.x] = val = val + vals[threadIdx.x -  2]; }
    if( thread_lane >=  4 && row == rows[threadIdx.x -  4] ) { vals[threadIdx.x] = val = val + vals[threadIdx.x -  4]; }
    if( thread_lane >=  8 && row == rows[threadIdx.x -  8] ) { vals[threadIdx.x] = val = val + vals[threadIdx.x -  8]; }
    if( thread_lane >= 16 && row == rows[threadIdx.x - 16] ) { vals[threadIdx.x] = val = val + vals[threadIdx.x - 16]; }

    return val;
}

template <typename IndexType, typename ValueType>
__device__ void segreduce_block(const IndexType * idx, ValueType * val)
{
    ValueType left = 0;
    if( threadIdx.x >=   1 && idx[threadIdx.x] == idx[threadIdx.x -   1] ) { left = val[threadIdx.x -   1]; } __syncthreads(); val[threadIdx.x] += left; left = 0; __syncthreads();  
    if( threadIdx.x >=   2 && idx[threadIdx.x] == idx[threadIdx.x -   2] ) { left = val[threadIdx.x -   2]; } __syncthreads(); val[threadIdx.x] += left; left = 0; __syncthreads();
    if( threadIdx.x >=   4 && idx[threadIdx.x] == idx[threadIdx.x -   4] ) { left = val[threadIdx.x -   4]; } __syncthreads(); val[threadIdx.x] += left; left = 0; __syncthreads();
    if( threadIdx.x >=   8 && idx[threadIdx.x] == idx[threadIdx.x -   8] ) { left = val[threadIdx.x -   8]; } __syncthreads(); val[threadIdx.x] += left; left = 0; __syncthreads();
    if( threadIdx.x >=  16 && idx[threadIdx.x] == idx[threadIdx.x -  16] ) { left = val[threadIdx.x -  16]; } __syncthreads(); val[threadIdx.x] += left; left = 0; __syncthreads();
    if( threadIdx.x >=  32 && idx[threadIdx.x] == idx[threadIdx.x -  32] ) { left = val[threadIdx.x -  32]; } __syncthreads(); val[threadIdx.x] += left; left = 0; __syncthreads();  
    if( threadIdx.x >=  64 && idx[threadIdx.x] == idx[threadIdx.x -  64] ) { left = val[threadIdx.x -  64]; } __syncthreads(); val[threadIdx.x] += left; left = 0; __syncthreads();
    if( threadIdx.x >= 128 && idx[threadIdx.x] == idx[threadIdx.x - 128] ) { left = val[threadIdx.x - 128]; } __syncthreads(); val[threadIdx.x] += left; left = 0; __syncthreads();
    if( threadIdx.x >= 256 && idx[threadIdx.x] == idx[threadIdx.x - 256] ) { left = val[threadIdx.x - 256]; } __syncthreads(); val[threadIdx.x] += left; left = 0; __syncthreads();
}


//////////////////////////////////////////////////////////////////////////////
// COO SpMV kernel which flattens data irregularity (segmented reduction)
//////////////////////////////////////////////////////////////////////////////
//
// spmv_coo_flat
//   The input coo_matrix must be sorted by row.  Columns within each row
//   may appear in any order and duplicate entries are also acceptable.
//   This sorted COO format is easily obtained by expanding the row pointer
//   of a CSR matrix (csr.Ap) into proper row indices and then copying 
//   the arrays containing the CSR column indices (csr.Aj) and nonzero values
//   (csr.Ax) verbatim.  A segmented reduction is used to compute the per-row
//   sums.
//
// spmv_coo_flat_tex
//   Same as spmv_coo_flat, except that the texture cache is 
//   used for accessing the x vector.
//


// spmv_coo_flat_kernel
//
// In this kernel each warp processes an interval of the nonzero values.
// For example, if the matrix contains 128 nonzero values and there are
// two warps and interval_size is 64, then the first warp (warp_id == 0)
// will process the first set of 64 values (interval [0, 64)) and the 
// second warp will process // the second set of 64 values 
// (interval [64, 128)).  Note that the  number of nonzeros is not always
// a multiple of 32 (the warp size) or 32 * the number of active warps,
// so the last active warp will not always process a "full" interval of
// interval_size.
// 
// The first thread in each warp (thread_lane == 0) has a special role:
// it is responsible for keeping track of the "carry" values from one
// iteration to the next.  The carry values consist of the row index and
// partial sum from the previous batch of 32 elements.  In the example 
// mentioned before with two warps and 128 nonzero elements, the first
// warp iterates twice and looks at the carry of the first iteration to
// decide whether to include this partial sum into the current batch.
// Specifically, if a row extends over a 32-element boundary, then the
// partial sum is carried over into the new 32-element batch.  If,
// on the other hand, the _last_ row index of the previous batch (the carry)
// differs from the _first_ row index of the current batch (the row
// read by the thread with thread_lane == 0), then the partial sum
// is written out to memory.
//
// Each warp iterates over its interval, processing 32 elements at a time.
// For each batch of 32 elements, the warp does the following
//  1) Fetch the row index, column index, and value for a matrix entry.  These
//     values are loaded from I[n], J[n], and V[n] respectively.
//     The row entry is stored in the shared memory array idx.
//  2) Fetch the corresponding entry from the input vector.  Specifically, for a 
//     nonzero entry (i,j) in the matrix, the thread must load the value x[j]
//     from memory.  We use the function fetch_x to control whether the texture
//     cache is used to load the value (UseCache == True) or whether a normal
//     global load is used (UseCache == False).
//  3) The matrix value A(i,j) (which was stored in V[n]) is multiplied by the 
//     value x[j] and stored in the shared memory array val.
//  4) The first thread in the warp (thread_lane == 0) considers the "carry"
//     row index and either includes the carried sum in its own sum, or it
//     updates the output vector (y) with the carried sum.
//  5) With row indices in the shared array idx and sums in the shared array
//     val, the warp conducts a segmented scan.  The segmented scan operation
//     looks at the row entries for each thread (stored in idx) to see whether
//     two values belong to the same segment (segments correspond to matrix rows).
//     Consider the following example which consists of 3 segments
//     (note: this example uses a warp size of 16 instead of the usual 32)
//
//           0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15   # thread_lane
//     idx [ 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2]  # row indices
//     val [ 4, 6, 5, 0, 8, 3, 2, 8, 3, 1, 4, 9, 2, 5, 2, 4]  # A(i,j) * x(j)
//
//     After the segmented scan the result will be
//
//     val [ 4,10,15,15,23,26, 2,10,13,14, 4,13,15,20,22,26]  # A(i,j) * x(j)
//
//  6) After the warp computes the segmented scan operation 
//     each thread except for the last (thread_lane == 31) looks
//     at the row index of the next thread (threadIdx.x + 1) to 
//     see if the segment ends here, or continues into the 
//     next thread.  The thread at the end of the segment writes
//     the sum into the output vector (y) at the corresponding row
//     index.
//  7) The last thread in each warp (thread_lane == 31) writes
//     its row index and partial sum into the designated spote in the
//     carry_idx and carry_val arrays.  The carry arrays are indexed
//     by warp_lane which is a number in [0, BLOCK_SIZE / 32).
//  
//  These steps are repeated until the warp reaches the end of its interval.
//  The carry values at the end of each interval are written to arrays 
//  temp_rows and temp_vals, which are processed by a second kernel.
//
template <typename IndexType, typename ValueType, unsigned int BLOCK_SIZE, bool UseCache>
__launch_bounds__(BLOCK_SIZE,1)
__global__ void
spmv_coo_flat_kernel(const IndexType num_nonzeros,
                     const IndexType interval_size,
                     const IndexType * I, 
                     const IndexType * J, 
                     const ValueType * V, 
                     const ValueType * x, 
                           ValueType * y,
                           IndexType * temp_rows,
                           ValueType * temp_vals)
{
    __shared__ volatile IndexType rows[48 *(BLOCK_SIZE/32)];
    __shared__ volatile ValueType vals[BLOCK_SIZE];

    const IndexType thread_id   = BLOCK_SIZE * blockIdx.x + threadIdx.x;                         // global thread index
    const IndexType thread_lane = threadIdx.x & (WARP_SIZE-1);                                   // thread index within the warp
    const IndexType warp_id     = thread_id   / WARP_SIZE;                                       // global warp index

    const IndexType interval_begin = warp_id * interval_size;                                    // warp's offset into I,J,V
    const IndexType interval_end   = thrust::min(interval_begin + interval_size, num_nonzeros);  // end of warps's work

    const IndexType idx = 16 * (threadIdx.x/32 + 1) + threadIdx.x;                               // thread's index into padded rows array

    rows[idx - 16] = -1;                                                                         // fill padding with invalid row index

    if(interval_begin >= interval_end)                                                           // warp has no work to do 
        return;

    if (thread_lane == 31)
    {
        // initialize the carry in values
        rows[idx] = I[interval_begin]; 
	vals[threadIdx.x] = ValueType(0);
    }
  
    for(IndexType n = interval_begin + thread_lane; n < interval_end; n += WARP_SIZE)
    {
        IndexType row = I[n];                                         // row index (i)
        ValueType val = V[n] * fetch_x<UseCache>(J[n], x);            // A(i,j) * x(j)
        
        if (thread_lane == 0)
        {
            if(row == rows[idx + 31])
                val += vals[threadIdx.x + 31];                        // row continues
            else
                y[rows[idx + 31]] += vals[threadIdx.x + 31];  // row terminated
        }
        
        rows[idx]         = row;
        vals[threadIdx.x] = val;

        if(row == rows[idx -  1]) { vals[threadIdx.x] = val = val + vals[threadIdx.x -  1]; } 
        if(row == rows[idx -  2]) { vals[threadIdx.x] = val = val + vals[threadIdx.x -  2]; }
        if(row == rows[idx -  4]) { vals[threadIdx.x] = val = val + vals[threadIdx.x -  4]; }
        if(row == rows[idx -  8]) { vals[threadIdx.x] = val = val + vals[threadIdx.x -  8]; }
        if(row == rows[idx - 16]) { vals[threadIdx.x] = val = val + vals[threadIdx.x - 16]; }

        if(thread_lane < 31 && row != rows[idx + 1])
            y[row] += vals[threadIdx.x];                                            // row terminated
    }

    if(thread_lane == 31)
    {
        // write the carry out values
        temp_rows[warp_id] = rows[idx];
        temp_vals[warp_id] = vals[threadIdx.x];
    }
}


// The second level of the segmented reduction operation
template <typename IndexType, typename ValueType, unsigned int BLOCK_SIZE>
__launch_bounds__(BLOCK_SIZE,1)
__global__ void
spmv_coo_reduce_update_kernel(const IndexType num_warps,
                              const IndexType * temp_rows,
                              const ValueType * temp_vals,
                                    ValueType * y)
{
    __shared__ IndexType rows[BLOCK_SIZE + 1];    
    __shared__ ValueType vals[BLOCK_SIZE + 1];    

    const IndexType end = num_warps - (num_warps & (BLOCK_SIZE - 1));

    if (threadIdx.x == 0)
    {
        rows[BLOCK_SIZE] = (IndexType) -1;
        vals[BLOCK_SIZE] = (ValueType)  0;
    }
    
    __syncthreads();

    IndexType i = threadIdx.x;

    while (i < end)
    {
        // do full blocks
        rows[threadIdx.x] = temp_rows[i];
        vals[threadIdx.x] = temp_vals[i];

        __syncthreads();

        segreduce_block(rows, vals);

        if (rows[threadIdx.x] != rows[threadIdx.x + 1])
            y[rows[threadIdx.x]] += vals[threadIdx.x];

        __syncthreads();

        i += BLOCK_SIZE; 
    }

    if (end < num_warps){
        if (i < num_warps){
            rows[threadIdx.x] = temp_rows[i];
            vals[threadIdx.x] = temp_vals[i];
        } else {
            rows[threadIdx.x] = (IndexType) -1;
            vals[threadIdx.x] = (ValueType)  0;
        }

        __syncthreads();
   
        segreduce_block(rows, vals);

        if (i < num_warps)
            if (rows[threadIdx.x] != rows[threadIdx.x + 1])
                y[rows[threadIdx.x]] += vals[threadIdx.x];
    }
}


template <bool UseCache,
          bool InitializeY,
          typename Matrix,
          typename ValueType>
void __spmv_coo_flat(const Matrix&    A, 
                     const ValueType* x, 
                           ValueType* y)
{
    typedef typename Matrix::index_type IndexType;

    const IndexType * I = thrust::raw_pointer_cast(&A.row_indices[0]);
    const IndexType * J = thrust::raw_pointer_cast(&A.column_indices[0]);
    const ValueType * V = thrust::raw_pointer_cast(&A.values[0]);

    if (InitializeY)
        thrust::fill(thrust::device_pointer_cast(y), thrust::device_pointer_cast(y) + A.num_rows, ValueType(0));

    if(A.num_entries == 0)
    {
        // empty matrix
        return;
    }
    else if (A.num_entries < static_cast<size_t>(WARP_SIZE))
    {
        // small matrix
        spmv_coo_serial_kernel<IndexType,ValueType> <<<1,1>>>
            (A.num_entries, I, J, V, x, y);
        return;
    }

    const unsigned int BLOCK_SIZE = 256;
    const unsigned int MAX_BLOCKS = cusp::detail::device::arch::max_active_blocks(spmv_coo_flat_kernel<IndexType, ValueType, BLOCK_SIZE, UseCache>, BLOCK_SIZE, (size_t) 0);
    const unsigned int WARPS_PER_BLOCK = BLOCK_SIZE / WARP_SIZE;

    const unsigned int num_units  = A.num_entries / WARP_SIZE; 
    const unsigned int num_warps  = std::min(num_units, WARPS_PER_BLOCK * MAX_BLOCKS);
    const unsigned int num_blocks = DIVIDE_INTO(num_warps, WARPS_PER_BLOCK);
    const unsigned int num_iters  = DIVIDE_INTO(num_units, num_warps);
    
    const unsigned int interval_size = WARP_SIZE * num_iters;

    const IndexType tail = num_units * WARP_SIZE; // do the last few nonzeros separately (fewer than WARP_SIZE elements)

    const unsigned int active_warps = (interval_size == 0) ? 0 : DIVIDE_INTO(tail, interval_size);

    if (UseCache)
        bind_x(x);

    cusp::array1d<IndexType,cusp::device_memory> temp_rows(active_warps);
    cusp::array1d<ValueType,cusp::device_memory> temp_vals(active_warps);

    spmv_coo_flat_kernel<IndexType, ValueType, BLOCK_SIZE, UseCache> <<<num_blocks, BLOCK_SIZE>>>
        (tail, interval_size, I, J, V, x, y,
         thrust::raw_pointer_cast(&temp_rows[0]), thrust::raw_pointer_cast(&temp_vals[0]));

    spmv_coo_reduce_update_kernel<IndexType, ValueType, BLOCK_SIZE> <<<1, BLOCK_SIZE>>>
        (active_warps, thrust::raw_pointer_cast(&temp_rows[0]), thrust::raw_pointer_cast(&temp_vals[0]), y);
    
    spmv_coo_serial_kernel<IndexType,ValueType> <<<1,1>>>
        (A.num_entries - tail, I + tail, J + tail, V + tail, x, y);

    if (UseCache)
        unbind_x(x);
}

template <typename Matrix,
          typename ValueType>
void spmv_coo_flat(const Matrix& A,
                   const ValueType * x, 
                         ValueType * y)
{ 
    __spmv_coo_flat<false, true>(A, x, y);
}


template <typename Matrix,
          typename ValueType>
void spmv_coo_flat_tex(const Matrix& A,
                       const ValueType * x, 
                             ValueType * y)
{ 
    __spmv_coo_flat<true, true>(A, x, y);
}


} // end namespace device
} // end namespace detail
} // end namespace cusp

