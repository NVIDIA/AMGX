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

#include <truncate.h>
#include <specific_spmv.h>
#include <basic_types.h>
#include <util.h>
#include <algorithm>
#include <thrust_wrapper.h>

#include "amgx_types/util.h"

#define USE_EXPERIMENTAL_FACTOR_COPY
// #define USE_EXPERIMENTAL_NLARGEST

namespace amgx
{

__host__ __device__ __forceinline__
double ourHash2(const int i)
{
    unsigned int a = i;
    double result;
    a = (a + 0x7ed55d16) + (a << 12);
    a = (a ^ 0xc761c23c) + (a >> 19);
    a = (a + 0x165667b1) + (a << 5);
    a = (a ^ 0xd3a2646c) + (a << 9);
    a = (a + 0xfd7046c5) + (a << 3);
    a = (a ^ 0xb55a4f09) + (a >> 16);
    result = (a ^ 0x4a51e590) / (double) UINT_MAX;
    return result;
}

template <typename IndexType, typename metricType, typename matrixType>
struct copy_pred
{
    double _trunc_factor;
    metricType *_metric_arr;

    copy_pred() : _trunc_factor(0), _metric_arr(NULL) {};
    copy_pred(const double trunc_factor, metricType *metric_arr) : _trunc_factor(trunc_factor),
        _metric_arr(metric_arr) {};

    __host__ __device__
    bool operator()(const thrust::tuple<IndexType, IndexType, matrixType> &a)
    {
        metricType metric = _metric_arr[thrust::get<0>(a)];

        if (  types::util<matrixType>::abs(thrust::get<2>(a)) >= _trunc_factor * metric) { return true; }

        return false;
    }

    copy_pred<IndexType, metricType, matrixType> &operator=(const copy_pred<IndexType, metricType, matrixType> &a)
    {
        this->_trunc_factor = a._trunc_factor;
        this->_metric_arr = a._metric_arr;
        return *this;
    }
};

template <typename IndexType, typename VectorType, typename MatrixType>
struct scale_op
{
    const VectorType *scale_vec;

    scale_op(const VectorType *s) : scale_vec(s) {};

    __host__ __device__
    thrust::tuple<IndexType, MatrixType> operator()(const thrust::tuple<IndexType, MatrixType> &a)
    {
        const IndexType row = thrust::get<0>(a);
        return thrust::tuple<IndexType, MatrixType>(row, thrust::get<1>(a) * scale_vec[row]);
    }
};

template <typename index_type, typename value_type>
__global__
void scale_kernel(const index_type *row_indices, value_type *values, const value_type *new_row_sums, const value_type *old_row_sums, const int size)
{
    for (int tidx = threadIdx.x + blockIdx.x * blockDim.x; tidx < size; tidx += blockDim.x * gridDim.x)
    {
        int row_index = row_indices[tidx];
        value_type multiplier = ( types::util<value_type>::abs(new_row_sums[row_index]) == 0. ) ? types::util<value_type>::get_one() : (old_row_sums[row_index] / new_row_sums[row_index]);
        values[tidx] = values[tidx] * multiplier;

        if (values[tidx] != values[tidx]) { printf("%d (%d): NaN after scaling\n", tidx, row_index); }
    }
}

template <typename IndexType>
__global__
void get_row_lengths_kernel(const IndexType *offsets, const IndexType num_rows, IndexType *row_lengths,
                            const int max_elmts)
{
    for (int row = threadIdx.x + blockIdx.x * blockDim.x; row < num_rows; row += blockDim.x * gridDim.x)
    {
        row_lengths[row] = min(offsets[row + 1] - offsets[row], max_elmts);
    }
}

// sort array by abs(values), largest element in [0]
template <typename IndexType, typename ValueType>
__device__
void sortByFabs(IndexType *indices, ValueType *values, int elements)
{
    int n = elements;

    do
    {
        int newn = 0;

        for (int i = 1; i < n; i++)
        {
            if (types::util<ValueType>::abs(values[i - 1]) < types::util<ValueType>::abs(values[i]))
            {
                ValueType temp_v = values[i - 1];
                IndexType temp_i = indices[i - 1];
                values[i - 1] = values[i];
                values[i] = temp_v;
                indices[i - 1] = indices[i];
                indices[i] = temp_i;
                newn = i;
            }
        }

        n = newn;
    }
    while (n > 0);
}

#ifdef USE_EXPERIMENTAL_NLARGEST

template <typename IndexType, typename ValueType>
__device__
void insert_new_max(volatile ValueType *Mv, volatile IndexType *Mi, volatile int *T, const ValueType S,
                    const IndexType I, const int thread, const int N)
{
    if (thread < N)
    {
        T[thread] = 1;

        // new value larger - shift my value down
        if (Mv[thread] < S)
        {
            T[thread] = 0;
            Mv[thread + 1] = Mv[thread];
            Mi[thread + 1] = Mi[thread];
            T[thread + 1]  = 1;
        }

        // insert new largest value here
        if (!T[thread])
        {
            Mv[thread] = S;
            Mi[thread] = I;
            // inserted to me -- note to zero out
        }
    }
}

template <typename IndexType, typename ValueType, typename VecValueType,
          int BLOCKSIZE, int THREADS_PER_VECTOR>
__global__
void truncateNandScale_kernel(const IndexType *A_offsets, const IndexType *A_indices, const ValueType *A_values,
                              const IndexType num_rows,
                              const IndexType *At_offsets, IndexType *At_indices, ValueType *At_values,
                              const int max_elmts, const VecValueType *old_row_sums)
{
    const int WARP_SIZE = 32;
    const int VECTORS_PER_WARP = WARP_SIZE / THREADS_PER_VECTOR;
    const int WARPS_PER_BLOCK  = BLOCKSIZE / WARP_SIZE;
    const int N_LARGEST_SIZE   = (THREADS_PER_VECTOR + 1) * VECTORS_PER_WARP * WARPS_PER_BLOCK;
    __shared__ volatile int T[N_LARGEST_SIZE]; // has M array been modified
    __shared__ volatile ValueType Mv[N_LARGEST_SIZE]; // maximum values
    __shared__ volatile int Mi[N_LARGEST_SIZE]; // maximum indices
    __shared__ volatile ValueType S[WARP_SIZE * WARPS_PER_BLOCK]; // temporary holding for currently read values
    __shared__ volatile IndexType I[WARP_SIZE * WARPS_PER_BLOCK]; //               -- " --                indices
    ValueType S_copy;
    IndexType I_copy;
    __shared__ volatile int Ap[WARPS_PER_BLOCK * VECTORS_PER_WARP][2]; // store row offset pointers
    __shared__ volatile ValueType old_row_sum[WARPS_PER_BLOCK * VECTORS_PER_WARP];
    const int num_vectors = blockDim.x / THREADS_PER_VECTOR * gridDim.x;
    const int vector_id = (threadIdx.x + blockIdx.x * blockDim.x) / THREADS_PER_VECTOR;
    const int thread_lane = threadIdx.x & (THREADS_PER_VECTOR - 1);
    const int vector_lane = threadIdx.x / THREADS_PER_VECTOR;

    // initialise relevant shared memory arrays
    for (int i = threadIdx.x; i < N_LARGEST_SIZE; i += blockDim.x)
    {
        Mv[i] = 0.;
        Mi[i] = -1;
        T[i] = 0;
    }

    __syncthreads();

    // outer loop over rows
    for (int row = vector_id; row < num_rows; row += num_vectors)
    {
        // load row offsets into shared memory
        if (thread_lane < 2)
        {
            Ap[vector_lane][thread_lane] = A_offsets[row + thread_lane];
        }

        const int row_start = Ap[vector_lane][0];
        const int row_end   = Ap[vector_lane][1];
        const int row_len   = row_end - row_start;
        const int max_elmts_row = min(At_offsets[row + 1] - At_offsets[row], max_elmts);

        // degenerate case -- simply copy over entire row
        if (row_len <= max_elmts && row_len > 0 && thread_lane < row_len)
        {
            At_indices[At_offsets[row] + thread_lane] = A_indices[row_start + thread_lane];
            At_values[At_offsets[row] + thread_lane] = A_values[row_start + thread_lane];
            continue;
        }

        // loop over chunks of the row
        const int nloops = (row_len < THREADS_PER_VECTOR) ? 1 : (int)ceil((float)row_len / THREADS_PER_VECTOR);

        for (int i = 0; i < nloops; i++)
        {
            // zero out temp shared memory
            S[threadIdx.x] =  0;
            I[threadIdx.x] = -2;

            // read chunk into shared memory
            if (i * THREADS_PER_VECTOR + thread_lane < row_len)
            {
                S[threadIdx.x] = fabs(A_values[row_start + i * THREADS_PER_VECTOR + thread_lane]);
                I[threadIdx.x] = A_indices[row_start + i * THREADS_PER_VECTOR + thread_lane];
            }

            // loop over n' max elements
            for (int j = 0; j < max_elmts_row; j++)
            {
                // first take a copy of this section of the row so we don't re-read it
                S_copy = S[threadIdx.x];
                I_copy = I[threadIdx.x];
                // local max & associated index (save volatile loads)
                ValueType max = S_copy; // S[threadIdx.x];
                IndexType ind = I_copy; // I[threadIdx.x];

                // reduce to get max
                if (THREADS_PER_VECTOR > 16)
                {
                    I[threadIdx.x] = ind = max > S[threadIdx.x + 16] ? ind : I[threadIdx.x + 16];
                    S[threadIdx.x] = max = max > S[threadIdx.x + 16] ? max : S[threadIdx.x + 16];
                }

                if (THREADS_PER_VECTOR > 8)
                {
                    I[threadIdx.x] = ind = max > S[threadIdx.x +  8] ? ind : I[threadIdx.x +  8];
                    S[threadIdx.x] = max = max > S[threadIdx.x +  8] ? max : S[threadIdx.x +  8];
                }

                if (THREADS_PER_VECTOR > 4)
                {
                    I[threadIdx.x] = ind = max > S[threadIdx.x +  4] ? ind : I[threadIdx.x +  4];
                    S[threadIdx.x] = max = max > S[threadIdx.x +  4] ? max : S[threadIdx.x +  4];
                }

                if (THREADS_PER_VECTOR > 2)
                {
                    I[threadIdx.x] = ind = max > S[threadIdx.x +  2] ? ind : I[threadIdx.x +  2];
                    S[threadIdx.x] = max = max > S[threadIdx.x +  2] ? max : S[threadIdx.x +  2];
                }

                if (THREADS_PER_VECTOR >  1)
                {
                    I[threadIdx.x] = ind = max > S[threadIdx.x +  1] ? ind : I[threadIdx.x +  1];
                    S[threadIdx.x] = max = max > S[threadIdx.x +  1] ? max : S[threadIdx.x +  1];
                }

                // maximum now in S_copy[vector_lane*32] -- insert into maximum array
                const int sm_offset = vector_lane * (THREADS_PER_VECTOR + 1);
                insert_new_max(&Mv[sm_offset], &Mi[sm_offset], &T[sm_offset],
                               S[vector_lane * THREADS_PER_VECTOR], I[vector_lane * THREADS_PER_VECTOR], thread_lane,
                               max_elmts);
                S[threadIdx.x] = (I[vector_lane * THREADS_PER_VECTOR] == I_copy) ? 0 : S_copy;
                I[threadIdx.x] = I_copy;
            }
        }

        // get scaling factor for new row
        // copy Mv -> S
        S[threadIdx.x] = 0; // intialise unneeded S array to 0

        if (thread_lane < max_elmts_row)
        {
            S[threadIdx.x] = Mv[vector_lane * (THREADS_PER_VECTOR + 1) + thread_lane];
        }

        // reduce on S
        ValueType sum = fabs(S[threadIdx.x]);

        if (THREADS_PER_VECTOR > 16) { S[threadIdx.x] = sum = sum + S[threadIdx.x + 16]; }

        if (THREADS_PER_VECTOR >  8) { S[threadIdx.x] = sum = sum + S[threadIdx.x + 8]; }

        if (THREADS_PER_VECTOR >  4) { S[threadIdx.x] = sum = sum + S[threadIdx.x + 4]; }

        if (THREADS_PER_VECTOR >  2) { S[threadIdx.x] = sum = sum + S[threadIdx.x + 2]; }

        if (THREADS_PER_VECTOR >  1) { S[threadIdx.x] = sum = sum + S[threadIdx.x + 1]; }

        const ValueType scale = old_row_sums[row] / S[vector_lane * THREADS_PER_VECTOR];

        // copy final scaled results into output buffer
        if (thread_lane < max_elmts_row)
        {
            At_values[At_offsets[row] + thread_lane]  = Mv[vector_lane * (THREADS_PER_VECTOR + 1) + thread_lane] * scale;
            At_indices[At_offsets[row] + thread_lane] = Mi[vector_lane * (THREADS_PER_VECTOR + 1) + thread_lane];
        }
    }
}

#else
// sort array by column index
template <typename IndexType, typename ValueType>
__device__
void sortByIndex(IndexType *indices, ValueType *values, int elements)
{
    int n = elements;

    do
    {
        int newn = 0;

        for (int i = 1; i < n; i++)
        {
            if (indices[i - 1] > indices[i])
            {
                ValueType temp_v = values[i - 1];
                IndexType temp_i = indices[i - 1];
                values[i - 1] = values[i];
                values[i] = temp_v;
                indices[i - 1] = indices[i];
                indices[i] = temp_i;
                newn = i;
            }
        }

        n = newn;
    }
    while (n > 0);
}

template <typename IndexType, typename ValueType>
__global__
void row_sum_kernel(const IndexType *A_offsets, const ValueType *A_values,
                    ValueType *row_sums, const IndexType num_rows, IndexType *row_lengths, const int max_elmts, bool compute_row_length)
{
    for (int tidx = threadIdx.x + blockIdx.x * blockDim.x; tidx < num_rows; tidx += blockDim.x * gridDim.x)
    {
        IndexType A_start = A_offsets[tidx], A_end = A_offsets[tidx + 1];

        if (compute_row_length)
        {
            row_lengths[tidx] = min(A_end - A_start, max_elmts);
        }

        ValueType row_sum = 0.;
        bool found_nans = false;

        for (int i = A_start; i < A_end; i++)
        {
            row_sum += A_values[i];

            if (A_values[i] != A_values[i]) { found_nans = true; }
        }

        if (found_nans)
        {
            printf("row %d has NaNs in input\n", tidx);
        }

        row_sums[tidx] = row_sum;
    }
}

template <typename IndexType, typename ValueType>
__global__
void __launch_bounds__(256)
truncate_kernel(const IndexType *A_offsets, const IndexType *A_col_indices, const ValueType *A_values,
                const IndexType num_rows,
                IndexType *trunc_offsets, IndexType *trunc_col_indices, ValueType *trunc_values,
                const int max_elmts)
{
    // const int tidx = threadIdx.x + blockIdx.x*blockDim.x;
    __shared__ IndexType s_ind[1024]; // assume <= 256 threads per block
    __shared__ ValueType s_val[1024];
    const int smem_start = threadIdx.x * max_elmts;

    // if (tidx < num_rows)
    for (int tidx = threadIdx.x + blockIdx.x * blockDim.x; tidx < num_rows; tidx += blockDim.x * gridDim.x)
    {
        IndexType A_start = A_offsets[tidx], A_end = A_offsets[tidx + 1];
        IndexType trunc_start = trunc_offsets[tidx];
        IndexType A_len = A_end - A_start;

        // copy my entire row if possible
        if (A_len <= max_elmts)
        {
            for (int j = 0; j < A_len; j++)
            {
                trunc_col_indices[trunc_start + j] = A_col_indices[A_start + j];
                trunc_values[trunc_start + j] = A_values[A_start + j];
            }
        }
        else // normal case
        {
            // read first max_elmts elements into shared
            for (int j = 0; j < max_elmts; j++)
            {
                s_ind[smem_start + j] = A_col_indices[A_start + j];
                s_val[smem_start + j] = A_values[A_start + j];
            }

            // sort the initially inserted elements (start = largest)
            sortByFabs(&s_ind[smem_start], &s_val[smem_start], max_elmts);

            // loop over all other elements on the row
            for (int j = A_start + max_elmts; j < A_end; j++)
            {
                // check against current maximums
                for (int i = 0; i < max_elmts; i++)
                {
                    if (types::util<ValueType>::abs(A_values[j]) > types::util<ValueType>::abs(s_val[smem_start + i]))
                    {
                        // shift remaining values down
                        for (int k = max_elmts - 1; k > i; k--)
                        {
                            s_val[smem_start + k] = s_val[smem_start + k - 1];
                            s_ind[smem_start + k] = s_ind[smem_start + k - 1];
                        }

                        // tentatively replace element
                        s_val[smem_start + i] = A_values[j];
                        s_ind[smem_start + i] = A_col_indices[j];
                        break;
                    }
                }
            }

            //sortByIndex(&s_ind[smem_start],&s_val[smem_start],max_elmts);
            // now we have the 4 largest elements (hopefully) -- output
            for (int i = 0; i < max_elmts; i++)
            {
                trunc_col_indices[trunc_start + i] = s_ind[smem_start + i];
                trunc_values[trunc_start + i] = s_val[smem_start + i];
            }
        }
    }
}
#endif

template <typename IndexType, typename ValueType>
__global__
void perturb_kernel(const IndexType *A_offsets, const IndexType *A_col_indices, ValueType *A_values,
                    const IndexType num_rows,
                    const int max_elmts)
{
    for (int tidx = threadIdx.x + blockIdx.x * blockDim.x; tidx < num_rows; tidx += blockDim.x * gridDim.x)
    {
        IndexType A_start = A_offsets[tidx], A_end = A_offsets[tidx + 1];

        // perturb A slightly
        for (int j = A_start; j < A_end; j++)
        {
            A_values[j] += ourHash2(A_col_indices[j]) * A_values[j];
        }
    }
}


// warp-based inclusive scan
// from: Sengupta, Harris, Garland 2008
// idx is index of thread
template <typename IndexType>
__device__
int scan_warp(volatile IndexType *ptr, const unsigned int idx)
{
    const unsigned int lane = idx & 31; // index of thread in warp
    IndexType p = ptr[idx];

    if (lane >=  1) { ptr[idx] = p = ptr[idx -  1] + p; }

    if (lane >=  2) { ptr[idx] = p = ptr[idx -  2] + p; }

    if (lane >=  4) { ptr[idx] = p = ptr[idx -  4] + p; }

    if (lane >=  8) { ptr[idx] = p = ptr[idx -  8] + p; }

    if (lane >= 16) { ptr[idx] = p = ptr[idx - 16] + p; }

    return ptr[idx];
}

template <typename IndexType, typename matValueType, typename vecValueType, int THREADS_PER_BLOCK,
          int THREADS_PER_VECTOR>
__global__
void truncateAndScale_kernel(const IndexType *A_offsets, const IndexType *A_indices,
                             const matValueType *A_values,
                             const IndexType num_rows,
                             IndexType *At_offsets, IndexType *At_indices, matValueType *At_values,
                             const vecValueType *old_row_sums, const vecValueType *new_row_sums,
                             const double truncate_factor, const vecValueType *metric)
{
    const int WARP_SIZE = 32;
    const int WARPS_PER_BLOCK = THREADS_PER_BLOCK / WARP_SIZE;
    const int VECTORS_PER_WARP = WARP_SIZE / THREADS_PER_VECTOR;
    // allocate shared memory for warp-based scans (assume up to 512 threads / block)
    __shared__ volatile int s_scan[WARP_SIZE * WARPS_PER_BLOCK];
    __shared__ volatile int s_A_ptr[WARPS_PER_BLOCK * VECTORS_PER_WARP][2]; // row offsets for A
    __shared__ volatile int s_At_ptr[WARPS_PER_BLOCK * VECTORS_PER_WARP][2]; // row offsets for At
    __shared__ double s_scale[WARPS_PER_BLOCK * VECTORS_PER_WARP]; // scaling factor for each warp
    const int vectors_per_block = THREADS_PER_BLOCK / WARP_SIZE * VECTORS_PER_WARP;
    const int num_vectors = THREADS_PER_BLOCK / THREADS_PER_VECTOR * gridDim.x;
    const int vector_id = (threadIdx.x + blockIdx.x * blockDim.x) / THREADS_PER_VECTOR;
    const int thread_lane = threadIdx.x & (THREADS_PER_VECTOR - 1);
    const int vector_lane = threadIdx.x / THREADS_PER_VECTOR;
    int scan_offset = 0;
    matValueType val = 0., fval = 0.;
    IndexType ind = 0;

    // outer loop over rows
    for (int row = vector_id; row < num_rows; row += num_vectors)
    {
        scan_offset = 0;
        val = 0.;
        fval = 0.; // fabs(val)

        if (threadIdx.x < vectors_per_block && threadIdx.x < num_rows - row) // first threads in first warp
        {
            s_scale[threadIdx.x] = old_row_sums[row + threadIdx.x] / new_row_sums[row + threadIdx.x];
        }

        __syncthreads();

        if (thread_lane < 2)
        {
            s_A_ptr[vector_lane][thread_lane] = A_offsets[row + thread_lane];
            s_At_ptr[vector_lane][thread_lane] = At_offsets[row + thread_lane];
        }

        const int row_start = s_A_ptr[vector_lane][0];
        const int row_end   = s_A_ptr[vector_lane][1];
        const int row_len = row_end - row_start;
        const int At_row_start = s_At_ptr[vector_lane][0];
        // mark & scan over consecutive blocks of 32 threads
        int nloops = (int)ceilf((float)row_len / THREADS_PER_VECTOR);
#pragma unroll 2

        for (int i = 0; i < nloops; i++)
        {
            // initialise shared memory to 0
            s_scan[threadIdx.x] = 0;

            // mark appropriate elements of the row
            if (i * THREADS_PER_VECTOR + thread_lane < row_len)
            {
                val = A_values[row_start + i * THREADS_PER_VECTOR + thread_lane];
                ind = A_indices[row_start + i * THREADS_PER_VECTOR + thread_lane];
                fval = fabs(val);
                s_scan[threadIdx.x] = fval >= truncate_factor * metric[row] ? 1 : 0;
            }

            // now prefix scan over this
            scan_warp(&s_scan[vector_lane * THREADS_PER_VECTOR], thread_lane); // threadIdx.x%32);

            // copy relevant parts of A to At
            if (i * THREADS_PER_VECTOR + thread_lane < row_len)
            {
                if (fval >= truncate_factor * metric[row])
                {
                    // get relevant index in At
                    const int At_index = At_row_start + scan_offset + s_scan[threadIdx.x] - 1;

                    if (s_scale[vector_lane] == 0.)
                    {
                        At_values[At_index] = val;
                    }
                    else
                    {
                        At_values[At_index] = val / s_scale[vector_lane];
                    }

                    At_indices[At_index] = ind;
                }
            }

            // note new scan_offset
            scan_offset = s_scan[vector_lane * THREADS_PER_VECTOR + THREADS_PER_VECTOR - 1];
        }
    }
}

template <typename Matrix, typename Vector, int THREADS_PER_BLOCK, int THREADS_PER_VECTOR>
void truncateAndScale(const Matrix &A, Matrix &A_trunc, const Vector &row_sum, const Vector &new_row_sum,
                      const double trunc_factor, const Vector &metric)
{
    const int num_blocks = std::min(4096, A.get_num_rows());
    truncateAndScale_kernel<typename Matrix::index_type, typename Matrix::value_type,
                            typename Vector::value_type, THREADS_PER_BLOCK, THREADS_PER_VECTOR>
                            <<< num_blocks, 128>>>(A.row_offsets.raw(), A.col_indices.raw(), A.values.raw(),
                                    A.get_num_rows(),
                                    A_trunc.row_offsets.raw(), A_trunc.col_indices.raw(),
                                    A_trunc.values.raw(),
                                    row_sum.raw(), new_row_sum.raw(),
                                    trunc_factor, metric.raw());
    cudaCheckError();
}

// host code
template <AMGX_VecPrecision t_vecPrec, AMGX_MatPrecision t_matPrec, AMGX_IndPrecision t_indPrec>
void Truncate<TemplateConfig<AMGX_host, t_vecPrec, t_matPrec, t_indPrec> >::truncateByFactor(Matrix_h &A,
        const double trunc_factor, const AMGX_TruncateType truncType)
{
    // throw error
    FatalError("Truncation on host not implemented yet", AMGX_ERR_NOT_IMPLEMENTED);
}

template <AMGX_VecPrecision t_vecPrec, AMGX_MatPrecision t_matPrec, AMGX_IndPrecision t_indPrec>
void Truncate<TemplateConfig<AMGX_host, t_vecPrec, t_matPrec, t_indPrec> >::truncateByMaxElements(Matrix_h &A,
        const int max_elmts)
{
    // throw error
    FatalError("Truncation on host not implemented yet", AMGX_ERR_NOT_IMPLEMENTED);
}

// device code
#ifdef _WIN32
template <AMGX_VecPrecision t_vecPrec, AMGX_MatPrecision t_matPrec, AMGX_IndPrecision t_indPrec>
void Truncate<TemplateConfig<AMGX_device, t_vecPrec, t_matPrec, t_indPrec> >::truncateByFactor(Matrix_d &A, const double trunc_factor, const AMGX_TruncateType truncType)
#else
template <AMGX_VecPrecision t_vecPrec, AMGX_MatPrecision t_matPrec, AMGX_IndPrecision t_indPrec>
void Truncate<TemplateConfig<AMGX_device, t_vecPrec, t_matPrec, t_indPrec> >::truncateByFactor(Matrix_d &A, const double trunc_factor, const AMGX_TruncateType truncType) __attribute__((noinline))
#endif
{
    typedef typename TConfig_d::IndPrec index_type;
    typedef typename TConfig_d::VecPrec vec_value_type;
    typedef typename TConfig_d::MatPrec value_type;
    // store the weighting values per row
    Matrix<TConfig_d> A_trunc(0, 0, 0, CSR);
    typedef Vector<TConfig_d> VVector;
    typedef typename Matrix<TConfig_d>::MVector MVector;
    typedef Vector<typename TConfig_d::template setVecPrec<AMGX_vecInt>::Type> IVector;
    VVector row_sum(A.get_num_rows());
    VVector new_row_sum(A.get_num_rows());
    VVector max_coef;
    // point to either row_sum or max_coef depending on type of truncation desired
    // vec_value_type *metric = 0;
    VVector *metric;

    // get either max row coefficient or row sum
    if (truncType == AMGX_TruncateByRowSum)
    {
        absRowSum(A, row_sum);
        metric = &row_sum;
    }
    else if (truncType == AMGX_TruncateByMaxCoefficient)
    {
        max_coef.resize(A.get_num_rows());
        maxCoefAndSum(A, max_coef, row_sum);
        metric = &max_coef;
    }
    else
    {
        FatalError("Truncation type not implemented", AMGX_ERR_NOT_IMPLEMENTED);
    }

    IVector row_counts(A.get_num_rows());

    if (truncType == AMGX_TruncateByRowSum)
    {
        countTruncElements(A, trunc_factor, row_sum, row_counts, new_row_sum);
    }
    else if (truncType == AMGX_TruncateByMaxCoefficient)
    {
        countTruncElements(A, trunc_factor, max_coef, row_counts, new_row_sum);
    }

    cudaCheckError();
    // initial resize (so we can scan into the row_offsets array)
    A_trunc.resize(A.get_num_rows(), A.get_num_cols(), 0);
    cudaCheckError();

    thrust_wrapper::exclusive_scan(row_counts.begin(), row_counts.end(), A_trunc.row_offsets.begin());
    cudaCheckError();
    
    const int nnz = A_trunc.row_offsets[A.get_num_rows() - 1] + row_counts[A.get_num_rows() - 1];
    A_trunc.row_offsets[A.get_num_rows()] = nnz;

    if (nnz == A.get_num_nz()) // early return -- nothing truncated
    {
        return;
    }

    // final resize & get row indices for both matrices
    A_trunc.resize(A.get_num_rows(), A.get_num_cols(), nnz);
#ifndef USE_EXPERIMENTAL_FACTOR_COPY
    A_trunc.addProps(COO);
    A.addProps(COO);
    // copy relevant values from A -> A_trunc
    copy_pred<index_type, vec_value_type, value_type> pred;

    if (truncType == AMGX_TruncateByRowSum)
    {
        pred = copy_pred<index_type, vec_value_type, value_type>(trunc_factor, row_sum.raw());
    }
    else if (truncType == AMGX_TruncateByMaxCoefficient)
    {
        pred = copy_pred<index_type, vec_value_type, value_type>(trunc_factor, max_coef.raw());
    }

    using thrust::make_zip_iterator;
    using thrust::make_tuple;
    thrust::copy_if(
        make_zip_iterator(make_tuple(A.row_indices.begin(), A.col_indices.begin(), A.values.begin())),
        make_zip_iterator(make_tuple(A.row_indices.end(), A.col_indices.end(), A.values.end())),
        make_zip_iterator(make_tuple(A_trunc.row_indices.begin(), A_trunc.col_indices.begin(),
                                     A_trunc.values.begin())),
        pred
    );
    // scale each row of the truncated matrix - per-row scale in new_row_sum
    thrust::transform(row_sum.begin(), row_sum.end(), new_row_sum.begin(), new_row_sum.begin(),
                      thrust::divides<vec_value_type>());
    scale_op<index_type, vec_value_type, value_type> op(new_row_sum.raw());
    thrust::for_each(
        make_zip_iterator(make_tuple(A_trunc.row_indices.begin(), A_trunc.values.begin())),
        make_zip_iterator(make_tuple(A_trunc.row_indices.end(), A_trunc.values.end())),
        op
    );
#else
    const int avg_row_len = (int)ceil(sqrt((float)A.get_num_nz() / A.get_num_rows()));

    if (avg_row_len > 16)
    {
        truncateAndScale<Matrix<TConfig>, Vector<TConfig>, 128, 32>(A, A_trunc, row_sum, new_row_sum,
                trunc_factor, *metric);
    }
    else if (avg_row_len > 8)
    {
        truncateAndScale<Matrix<TConfig>, Vector<TConfig>, 128, 16>(A, A_trunc, row_sum, new_row_sum,
                trunc_factor, *metric);
    }
    else if (avg_row_len > 4)
    {
        truncateAndScale<Matrix<TConfig>, Vector<TConfig>, 128, 8>(A, A_trunc, row_sum, new_row_sum,
                trunc_factor, *metric);
    }
    else
    {
        truncateAndScale<Matrix<TConfig>, Vector<TConfig>, 128, 4>(A, A_trunc, row_sum, new_row_sum,
                trunc_factor, *metric);
    }

#endif
    cudaCheckError();
    // copy truncated matrix to this
    A.copy(A_trunc);
}

#ifdef USE_EXPERIMENTAL_NLARGEST
template <typename Matrix, typename Vector, int BLOCKSIZE, int THREADS_PER_VECTOR>
void truncateNandScale(const Matrix &A, Matrix &A_trunc, const int max_elmts, const Vector &orig_row_sums)
{
    const int nthreads = BLOCKSIZE;
    const int nblocks = std::min(4096, A.get_num_rows() / (nthreads / THREADS_PER_VECTOR) + 1);
    truncateNandScale_kernel<typename Matrix::index_type, typename Matrix::value_type,
                             typename Vector::value_type, BLOCKSIZE, THREADS_PER_VECTOR>
                             <<< nblocks, nthreads>>>(
                                 A.row_offsets.raw(),
                                 A.col_indices.raw(),
                                 A.values.raw(),
                                 A.get_num_rows(),
                                 A_trunc.row_offsets.raw(),
                                 A_trunc.col_indices.raw(),
                                 A_trunc.values.raw(),
                                 max_elmts,
                                 orig_row_sums.raw()
                             );
    cudaCheckError();
}
#endif

template <AMGX_VecPrecision t_vecPrec, AMGX_MatPrecision t_matPrec, AMGX_IndPrecision t_indPrec>
void Truncate<TemplateConfig<AMGX_device, t_vecPrec, t_matPrec, t_indPrec> >::truncateByMaxElements(Matrix_d &A,
        const int max_elmts)
{
    if (max_elmts > 32)   // not supported
    {
        FatalError("Matrix truncation to > 32 elements not supported", AMGX_ERR_BAD_PARAMETERS);
    }

    typedef typename TConfig_d::IndPrec index_type;
    typedef typename TConfig_d::VecPrec vec_value_type;
    typedef typename TConfig_d::MatPrec value_type;
    // store the weighting values per row
    typedef Vector<TConfig_d> VVector;
    typedef typename Matrix<TConfig_d>::MVector MVector;
    typedef Vector<typename TConfig_d::template setVecPrec<AMGX_vecInt>::Type> IVector;
    IVector row_lengths(A.get_num_rows());
    MVector orig_row_sums(A.get_num_rows(), 0);
    MVector new_row_sums(A.get_num_rows(), 0);
    const int blocksize = 128;
    const int num_blocks = A.get_num_rows() / blocksize + 1;

    // get the original row sums of A
    row_sum_kernel <<< 4096, blocksize>>>(A.row_offsets.raw(), A.values.raw(), orig_row_sums.raw(), A.get_num_rows(), row_lengths.raw(), max_elmts, true);
    cudaCheckError();
    // initial definition
    Matrix_d A_trunc(A.get_num_rows(), A.get_num_cols(), 0, CSR); // add CSR prop
    // exclusive scan to get new row structure
    thrust_wrapper::exclusive_scan(row_lengths.begin(), row_lengths.end(), A_trunc.row_offsets.begin());
    int nnz = A_trunc.row_offsets[A.get_num_rows() - 1] + row_lengths[A.get_num_rows() - 1];
    A_trunc.row_offsets[A.get_num_rows()] = nnz;
    // set final size of truncated matrix
    A_trunc.resize(A.get_num_rows(), A.get_num_cols(), nnz);
#ifdef USE_EXPERIMENTAL_NLARGEST

    // run kernel with appropriate vector size
    if (max_elmts > 16)
    {
        truncateNandScale<Matrix<TConfig_d>, Vector<TConfig_d>, 128, 32>(A, A_trunc, max_elmts, orig_row_sums);
    }
    else if (max_elmts > 8)
    {
        truncateNandScale<Matrix<TConfig_d>, Vector<TConfig_d>, 128, 16>(A, A_trunc, max_elmts, orig_row_sums);
    }
    else if (max_elmts > 4)
    {
        truncateNandScale<Matrix<TConfig_d>, Vector<TConfig_d>, 192, 8>(A, A_trunc, max_elmts, orig_row_sums);
    }
    else
    {
        truncateNandScale<Matrix<TConfig_d>, Vector<TConfig_d>, 128, 4>(A, A_trunc, max_elmts, orig_row_sums);
    }

#else
    cudaCheckError();

    truncate_kernel <<< 4096, blocksize>>>(
        A.row_offsets.raw(),
        A.col_indices.raw(),
        A.values.raw(),
        A.get_num_rows(),
        A_trunc.row_offsets.raw(),
        A_trunc.col_indices.raw(),
        A_trunc.values.raw(),
        max_elmts
    );
    cudaCheckError();
    // get the new row sums of A_trunc
    row_sum_kernel <<< 4096, blocksize>>>(A_trunc.row_offsets.raw(), A_trunc.values.raw(), new_row_sums.raw(), A_trunc.get_num_rows(), (index_type *) NULL, 0, false);
    cudaCheckError();
    A_trunc.addProps(COO);
    scale_kernel <<< 4096, blocksize>>>(A_trunc.row_indices.raw(), A_trunc.values.raw(), new_row_sums.raw(), orig_row_sums.raw(), nnz);
    cudaCheckError();
#endif
    cudaCheckError();
    A_trunc.delProps(COO);
    A.set_initialized(0);
    A.copy(A_trunc);
    A.computeDiagonal();
    A.set_initialized(1);
}

// -------------------------------
//  Explict instantiations
// -------------------------------

#define AMGX_CASE_LINE(CASE) template class Truncate<TemplateMode<CASE>::Type>;
AMGX_FORALL_BUILDS(AMGX_CASE_LINE)
AMGX_FORCOMPLEX_BUILDS(AMGX_CASE_LINE)
#undef AMGX_CASE_LINE
} // namespace amgx

