/* Copyright (c) 2011-2017, NVIDIA CORPORATION. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *  * Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *  * Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *  * Neither the name of NVIDIA CORPORATION nor the names of its
 *    contributors may be used to endorse or promote products derived
 *    from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
 * EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 * PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
 * CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 * EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 * PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
 * OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#pragma once

#include <basic_types.h>
#include <matrix.h>
#include <vector.h>
#include <texture.h>
#include <cusp/detail/device/arch.h>
#include <cusp/detail/device/utils.h>

#include <algorithm>

namespace amgx
{

template <typename IndexType, typename matValType, typename inVecType, typename outVecType,
          unsigned int VECTORS_PER_BLOCK, unsigned int THREADS_PER_VECTOR, typename state>
__launch_bounds__(VECTORS_PER_BLOCK *THREADS_PER_VECTOR, 1)
__global__ void
csr_generic_spmv_vector_kernel(const IndexType num_rows,
                               const IndexType *Ap,
                               const IndexType *Aj,
                               const matValType *Ax,
                               const inVecType   *x,
                               outVecType *y,
                               const state       st)
{
    const int SDATA_SIZE = ( VECTORS_PER_BLOCK * THREADS_PER_VECTOR + THREADS_PER_VECTOR / 2 ) * sizeof( outVecType );
    __shared__ char sdata_storage[SDATA_SIZE];  // padded to avoid reduction conditionals
    outVecType *sdata = reinterpret_cast<outVecType *>( sdata_storage );
    __shared__ volatile IndexType ptrs[VECTORS_PER_BLOCK][2];
    const IndexType THREADS_PER_BLOCK = VECTORS_PER_BLOCK * THREADS_PER_VECTOR;
    const IndexType thread_id   = THREADS_PER_BLOCK * blockIdx.x + threadIdx.x;    // global thread index
    const IndexType thread_lane = threadIdx.x & (THREADS_PER_VECTOR - 1);          // thread index within the vector
    const IndexType vector_id   = thread_id   /  THREADS_PER_VECTOR;               // global vector index
    const IndexType vector_lane = threadIdx.x /  THREADS_PER_VECTOR;               // vector index within the block
    const IndexType num_vectors = VECTORS_PER_BLOCK * gridDim.x;                   // total number of active vectors

    for (IndexType row = vector_id; row < num_rows; row += num_vectors)
    {
        // use two threads to fetch Ap[row] and Ap[row+1]
        // this is considerably faster than the straightforward version
        if (thread_lane < 2)
        {
            ptrs[vector_lane][thread_lane] = Ap[row + thread_lane];
        }

        const IndexType row_start = ptrs[vector_lane][0];                   //same as: row_start = Ap[row];
        const IndexType row_end   = ptrs[vector_lane][1];                   //same as: row_end   = Ap[row+1];
        // initialize local sum
        outVecType sum = 0;

        if (THREADS_PER_VECTOR == 32 && row_end - row_start > 32)
        {
            // ensure aligned memory access to Aj and Ax
            IndexType jj = row_start - (row_start & (THREADS_PER_VECTOR - 1)) + thread_lane;

            // accumulate local sums
            if (jj >= row_start && jj < row_end)
            {
                sum = outVecType::reduce_op(
                          sum,
                          outVecType::binary_op(Ax[jj], x, row, Aj[jj], st)
                      );
            }

            // accumulate local sums
            for (jj += THREADS_PER_VECTOR; jj < row_end; jj += THREADS_PER_VECTOR)
            {
                sum = outVecType::reduce_op(sum, outVecType::binary_op(Ax[jj], x, row, Aj[jj], st));
            }
        }
        else
        {
            // accumulate local sums
            for (IndexType jj = row_start + thread_lane; jj < row_end; jj += THREADS_PER_VECTOR)
            {
                sum = outVecType::reduce_op(sum, outVecType::binary_op(Ax[jj], x, row, Aj[jj], st));
            }
        }

        // store local sum in shared memory
        sdata[threadIdx.x] = sum;
        __syncthreads();

        // reduce local sums to row sum
        if (THREADS_PER_VECTOR > 16)
        {
            sdata[threadIdx.x] = sum = outVecType::reduce_op(sum, sdata[threadIdx.x + 16]);
        }

        __syncthreads();

        if (THREADS_PER_VECTOR >  8)
        {
            sdata[threadIdx.x] = sum = outVecType::reduce_op(sum, sdata[threadIdx.x +  8]);
        }

        __syncthreads();

        if (THREADS_PER_VECTOR >  4)
        {
            sdata[threadIdx.x] = sum = outVecType::reduce_op(sum, sdata[threadIdx.x +  4]);
        }

        __syncthreads();

        if (THREADS_PER_VECTOR >  2)
        {
            sdata[threadIdx.x] = sum = outVecType::reduce_op(sum, sdata[threadIdx.x +  2]);
        }

        __syncthreads();

        if (THREADS_PER_VECTOR >  1)
        {
            sdata[threadIdx.x] = sum = outVecType::reduce_op(sum, sdata[threadIdx.x +  1]);
        }

        __syncthreads();

        // first thread writes the result
        if (thread_lane == 0)
        {
            y[row] = outVecType::post_op(sdata[threadIdx.x]);
        }
    }
}

template <unsigned int THREADS_PER_VECTOR, typename Matrix, typename inVector, typename outVector,
          typename state>
void __csr_generic_spmv_vector(const Matrix    &A,
                               const inVector  &x,
                               outVector &y,
                               const state     &st)
{
    typedef typename Matrix::index_type IndexType;
    const size_t THREADS_PER_BLOCK  = 128;
    const size_t VECTORS_PER_BLOCK  = THREADS_PER_BLOCK / THREADS_PER_VECTOR;
    const size_t MAX_BLOCKS = cusp::detail::device::arch::max_active_blocks(
                                  csr_generic_spmv_vector_kernel<IndexType, typename Matrix::value_type,
                                  typename inVector::value_type,
                                  typename outVector::value_type,
                                  VECTORS_PER_BLOCK, THREADS_PER_VECTOR, state>,
                                  THREADS_PER_BLOCK, (size_t) 0);
    const size_t NUM_BLOCKS = std::min<size_t>(MAX_BLOCKS, cusp::detail::device::DIVIDE_INTO(A.get_num_rows(), VECTORS_PER_BLOCK));
    csr_generic_spmv_vector_kernel<IndexType, typename Matrix::value_type,
                                   typename inVector::value_type,
                                   typename outVector::value_type, VECTORS_PER_BLOCK, THREADS_PER_VECTOR>
                                   <<< NUM_BLOCKS, THREADS_PER_BLOCK>>>
                                   (
                                       A.get_num_rows(),
                                       A.row_offsets.raw(),
                                       A.col_indices.raw(),
                                       A.values.raw(),
                                       x.raw(),
                                       amgx::thrust::raw_pointer_cast(&y[0]),
                                       st
                                   );
    cudaCheckError();
}

template <AMGX_VecPrecision t_vecPrec, AMGX_VecPrecision t_vecPrec2, AMGX_MatPrecision t_matPrec, AMGX_IndPrecision t_indPrec,
          typename thrust_vector, typename state>
void genericSpmvCSR(const Matrix<TemplateConfig<AMGX_device, t_vecPrec, t_matPrec, t_indPrec> > &A,
                    const Vector<TemplateConfig<AMGX_device, t_vecPrec2, t_matPrec, t_indPrec> > &x,
                    thrust_vector                                                            &y,
                    const state                                                                    &st)
{
    typedef typename Matrix<TemplateConfig<AMGX_device, t_vecPrec, t_matPrec, t_indPrec> >::index_type  IndexType;
    const IndexType nnz_per_row = A.get_num_nz() / A.get_num_rows();

    if (nnz_per_row <=  2) { __csr_generic_spmv_vector< 2>(A, x, y, st); return; }

    if (nnz_per_row <=  4) { __csr_generic_spmv_vector< 4>(A, x, y, st); return; }

    if (nnz_per_row <=  8) { __csr_generic_spmv_vector< 8>(A, x, y, st); return; }

    if (nnz_per_row <= 16) { __csr_generic_spmv_vector<16>(A, x, y, st); return; }

    __csr_generic_spmv_vector<32>(A, x, y, st);
}

template <AMGX_VecPrecision t_vecPrec, AMGX_VecPrecision t_vecPrec2, AMGX_MatPrecision t_matPrec, AMGX_IndPrecision t_indPrec,
          typename thrust_vector, typename state>
void genericSpmvCSR(const Matrix<TemplateConfig<AMGX_host, t_vecPrec, t_matPrec, t_indPrec> > &A,
                    const Vector<TemplateConfig<AMGX_host, t_vecPrec2, t_matPrec, t_indPrec> > &x,
                    thrust_vector                                                        &y,
                    const state                                                                &st)
{
    typedef typename thrust_vector::value_type outType;

    for (int i = 0; i < A.get_num_rows(); i++)
    {
        outType sum = outType(0);

        for (int jj = A.row_offsets[i]; jj < A.row_offsets[i + 1]; jj++)
        {
            sum = outType::reduce_op(sum, outType::binary_op(A.values[jj], x.raw(), i, A.col_indices[jj], st));
        }

        y[i] = outType::post_op(sum);
    }
}

template <typename inType, typename outType1, typename outType2, typename outType3>
__global__
void retrieveThreeKernel(const inType *in, outType1 *out1, outType2 *out2, outType3 *out3, const int num_rows)
{
    for (int idx = threadIdx.x + blockIdx.x * blockDim.x; idx < num_rows; idx += blockDim.x * gridDim.x )
    {
        in[idx].retrieve(out1[idx], out2[idx], out3[idx]);
    }
}

template <typename inVector, AMGX_VecPrecision t_vecPrec1, AMGX_VecPrecision t_vecPrec2,
          AMGX_VecPrecision t_vecPrec3, AMGX_MatPrecision t_matPrec, AMGX_IndPrecision t_indPrec>
void retrieveThreeArguments(const inVector &in,
                            Vector<TemplateConfig<AMGX_device, t_vecPrec1, t_matPrec, t_indPrec> > &out1,
                            Vector<TemplateConfig<AMGX_device, t_vecPrec2, t_matPrec, t_indPrec> > &out2,
                            Vector<TemplateConfig<AMGX_device, t_vecPrec3, t_matPrec, t_indPrec> > &out3)
{
    const int blockSize = 128;
    const int numBlocks = min( (int)(AMGX_GRID_MAX_SIZE), (int)(in.size() / blockSize + 1));
    retrieveThreeKernel <<< numBlocks, blockSize>>>(amgx::thrust::raw_pointer_cast(&in[0]),
            amgx::thrust::raw_pointer_cast(&out1[0]),
            amgx::thrust::raw_pointer_cast(&out2[0]),
            amgx::thrust::raw_pointer_cast(&out3[0]),
            (int)in.size());
    cudaCheckError();
}

template <typename inVector, AMGX_VecPrecision t_vecPrec1, AMGX_VecPrecision t_vecPrec2,
          AMGX_VecPrecision t_vecPrec3, AMGX_MatPrecision t_matPrec, AMGX_IndPrecision t_indPrec>
void retrieveThreeArguments(const inVector &in,
                            Vector<TemplateConfig<AMGX_host, t_vecPrec1, t_matPrec, t_indPrec> > &out1,
                            Vector<TemplateConfig<AMGX_host, t_vecPrec2, t_matPrec, t_indPrec> > &out2,
                            Vector<TemplateConfig<AMGX_host, t_vecPrec3, t_matPrec, t_indPrec> > &out3)
{
    typedef typename inVector::value_type ValueType;

    for (int i = 0; i < (int)in.size(); i++)
    {
        ((ValueType)in[i]).retrieve(out1[i], out2[i], out3[i]);
    }
}

template <typename inType, typename outType1, typename outType2>
__global__
void retrieveTwoKernel(const inType *in, outType1 *out1, outType2 *out2, const int num_rows)
{
    for (int idx = threadIdx.x + blockIdx.x * blockDim.x; idx < num_rows; idx += blockDim.x * gridDim.x )
    {
        in[idx].retrieve(out1[idx], out2[idx]);
    }
}

template <typename inVector, AMGX_VecPrecision t_vecPrecOne, AMGX_VecPrecision t_vecPrecTwo,
          AMGX_MatPrecision t_matPrec, AMGX_IndPrecision t_indPrec>
void retrieveTwoArguments(const inVector &in,
                          Vector<TemplateConfig<AMGX_device, t_vecPrecOne, t_matPrec, t_indPrec> > &out1,
                          Vector<TemplateConfig<AMGX_device, t_vecPrecTwo, t_matPrec, t_indPrec> > &out2)
{
    const int blockSize = 128;
    const int numBlocks = min( (int)(AMGX_GRID_MAX_SIZE), (int)(in.size() / blockSize + 1));
    retrieveTwoKernel <<< numBlocks, blockSize>>>(amgx::thrust::raw_pointer_cast(&in[0]),
            out1.raw(),
            out2.raw(),
            (int)in.size());
    cudaCheckError();
}

template <typename inVector, AMGX_VecPrecision t_vecPrecOne, AMGX_VecPrecision t_vecPrecTwo,
          AMGX_MatPrecision t_matPrec, AMGX_IndPrecision t_indPrec>
void retrieveTwoArguments(const inVector &in,
                          Vector<TemplateConfig<AMGX_host, t_vecPrecOne, t_matPrec, t_indPrec> > &out1,
                          Vector<TemplateConfig<AMGX_host, t_vecPrecTwo, t_matPrec, t_indPrec> > &out2)
{
    typedef typename inVector::value_type ValueType;

    for (int i = 0; i < (int)in.size(); i++)
    {
        ((ValueType)in[i]).retrieve(out1[i], out2[i]);
    }
}

template <typename inType, typename outType>
__global__
void retrieveOneKernel(const inType *in, outType *out, const int num_rows)
{
    for (int idx = threadIdx.x + blockIdx.x * blockDim.x; idx < num_rows; idx += blockDim.x * gridDim.x)
    {
        in[idx].retrieve(out[idx]);
    }
}

template <typename inVector, AMGX_VecPrecision t_vecPrec, AMGX_MatPrecision t_matPrec,
          AMGX_IndPrecision t_indPrec>
void retrieveOneArgument(const inVector &in,
                         Vector<TemplateConfig<AMGX_device, t_vecPrec, t_matPrec, t_indPrec> > &out)
{
    const int blockSize = 128;
    const int numBlocks = min( (int)(AMGX_GRID_MAX_SIZE), (int)(in.size() / blockSize + 1));
    retrieveOneKernel <<< numBlocks, blockSize>>>(amgx::thrust::raw_pointer_cast(&in[0]),
            out.raw(),
            (int)in.size());
    cudaCheckError();
}

template <typename inVector, AMGX_VecPrecision t_vecPrec, AMGX_MatPrecision t_matPrec,
          AMGX_IndPrecision t_indPrec>
void retrieveOneArgument(const inVector &in,
                         Vector<TemplateConfig<AMGX_host, t_vecPrec, t_matPrec, t_indPrec> > &out)
{
    typedef typename inVector::value_type ValueType;

    for (int i = 0; i < (int)in.size(); i++)
    {
        ((ValueType)in[i]).retrieve(out[i]);
    }
}

} // namespace amgx
