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
#include <error.h>
#include <ld_functions.h>
/**********************************************************
 * Checks for a cuda error and if one exists prints it,
 * the stack trace, and exits
 *********************************************************/

#define cudaSafeCall(x) x;cudaCheckError()

namespace amgx
{

// block_size == 1 for find diag kernels
template<typename IndexType, typename ValueTypeA, typename ValueTypeB>
__global__ void find_diag_kernel(const IndexType num_rows,
                                 const IndexType *Ap,
                                 const IndexType *Aj,
                                 const ValueTypeA *Ax,
                                 ValueTypeB *diag)
{
    IndexType tidx = blockDim.x * blockIdx.x + threadIdx.x;

    for (int ridx = tidx; ridx < num_rows; ridx += blockDim.x * gridDim.x)
    {
        IndexType row_start = Ap[ridx];
        IndexType row_end   = Ap[ridx + 1];

        for (int j = row_start; j < row_end; j++)
        {
            if (Aj[j] == ridx)
            {
                diag[ridx] = Ax[j];
                break;
            }
        }
    }
}

template<typename IndexType, typename ValueTypeA, typename ValueTypeB>
__global__ void find_diag_kernel_indexed_dia(const IndexType num_rows,
        const IndexType *Adia,
        const ValueTypeA *Ax,
        ValueTypeB *diag)
{
    IndexType tidx = blockDim.x * blockIdx.x + threadIdx.x;

    while (tidx < num_rows)
    {
        diag[tidx] = Ax[Adia[tidx]];
        tidx += gridDim.x * blockDim.x;
    }
}

template<typename IndexType, typename ValueTypeA, typename ValueTypeB>
__global__ void find_diag_kernel_invert(const IndexType num_rows,
                                        const IndexType *Ap,
                                        const IndexType *Aj,
                                        const ValueTypeA *Ax,
                                        ValueTypeB *diag)
{
    IndexType tidx = blockDim.x * blockIdx.x + threadIdx.x;

    for (int ridx = tidx; ridx < num_rows; ridx += blockDim.x * gridDim.x)
    {
        IndexType row_start = Ap[ridx];
        IndexType row_end   = Ap[ridx + 1];

        for (int j = row_start; j < row_end; j++)
        {
            if (Aj[j] == ridx)
            {
                diag[ridx] = static_cast<ValueTypeA>(1) / Ax[j];
                break;
            }
        }
    }
}

// block_size == 1 !
template<typename IndexType, typename ValueTypeA, typename ValueTypeB>
__global__ void find_diag_kernel_indexed_dia_invert(const IndexType num_rows,
        const IndexType *Adia,
        const ValueTypeA *Ax,
        ValueTypeB *diag)
{
    IndexType tidx = blockDim.x * blockIdx.x + threadIdx.x;

    while (tidx < num_rows)
    {
        diag[tidx] = static_cast<ValueTypeA>(1) / Ax[Adia[tidx]];
        tidx += gridDim.x * blockDim.x;
    }
}

template <int NUM_ELEMENTS>
__device__ inline void loadAsVector(const float *src, float *dst)
{
    if (NUM_ELEMENTS % 4 == 0)
    {
        float4 *dst_ = reinterpret_cast<float4 *>(dst);
        const float4 *src_ = reinterpret_cast<const float4 *>(src);
#pragma unroll
        for (int i = 0; i < NUM_ELEMENTS / 4; i++)
        {
            dst_[i] = src_[i];
        }
    }
    else if (NUM_ELEMENTS % 2 == 0)
    {
        float2 *dst_ = reinterpret_cast<float2 *>(dst);
        const float2 *src_ = reinterpret_cast<const float2 *>(src);
#pragma unroll
        for (int i = 0; i < NUM_ELEMENTS / 2; i++)
        {
            dst_[i] = src_[i];
        }
    }
    else
    {
#pragma unroll
        for (int i = 0; i < NUM_ELEMENTS; i++)
        {
            dst[i] = src[i];
        }
    }
}

template <int NUM_ELEMENTS>
__device__ inline void loadAsVector(const float *src, double *dst)
{
    if (NUM_ELEMENTS % 4 == 0)
    {
        const float4 *src_ = reinterpret_cast<const float4 *>(src);
        double4 *dst_ = reinterpret_cast<double4 *>(dst);
#pragma unroll
        for (int i = 0; i < NUM_ELEMENTS / 4; i++)
        {
            float4 temp = src_[i];
            dst_[i] = make_double4(temp.x, temp.y, temp.z, temp.w);
        }
    }
    else  if (NUM_ELEMENTS % 2 == 0)
    {
        const float2 *src_ = reinterpret_cast<const float2 *>(src);
        double2 *dst_ = reinterpret_cast<double2 *>(dst);
#pragma unroll
        for (int i = 0; i < NUM_ELEMENTS / 2; i++)
        {
            float2 temp = src_[i];
            dst_[i] = make_double2(temp.x, temp.y);
        }
    }
    else
    {
#pragma unroll
        for (int i = 0; i < NUM_ELEMENTS; i++)
        {
            dst[i] = src[i];
        }
    }
}

template <int NUM_ELEMENTS>
__device__ inline void loadAsVector(const double *src, double *dst)
{
    if (NUM_ELEMENTS % 4 == 0)
    {
        const double4 *src_ = reinterpret_cast<const double4 *>(src);
        double4 *dst_ = reinterpret_cast<double4 *>(dst);
#pragma unroll
        for (int i = 0; i < NUM_ELEMENTS / 4; i++)
        {
            dst_[i] = src_[i];
        }
    }
    else if (NUM_ELEMENTS % 2 == 0)
    {
        const double2 *src_ = reinterpret_cast<const double2 *>(src);
        double2 *dst_ = reinterpret_cast<double2 *>(dst);
#pragma unroll
        for (int i = 0; i < NUM_ELEMENTS / 2; i++)
        {
            dst_[i] = src_[i];
        }
    }
    else
    {
#pragma unroll
        for (int i = 0; i < NUM_ELEMENTS; i++)
        {
            dst[i] = src[i];
        }
    }
}

template <int NUM_ELEMENTS>
__device__ inline void loadAsVector(const cuComplex *src, cuComplex *dst)
{
    if (NUM_ELEMENTS % 2 == 0)
    {
        const float4 *src_ = reinterpret_cast<const float4 *>(src);
        float4 *dst_ = reinterpret_cast<float4 *>(dst);
#pragma unroll
        for (int i = 0; i < NUM_ELEMENTS / 2; i++)
        {
            dst_[i] = src_[i];
        }
    }
    else
    {
#pragma unroll
        for (int i = 0; i < NUM_ELEMENTS; i++)
        {
            dst[i] = src[i];
        }
    }
}

template <int NUM_ELEMENTS>
__device__ inline void loadAsVector(const cuComplex *src, cuDoubleComplex *dst)
{
    if (NUM_ELEMENTS % 2 == 0)
    {
        double4 _t;
        const double4 *src_ = reinterpret_cast<const double4 *>(src);
#pragma unroll
        for (int i = 0; i < NUM_ELEMENTS / 2; i++)
        {
            _t = src_[i];
            dst[i * 2] = make_cuDoubleComplex(_t.x, _t.y);
            dst[i * 2 + 1] = make_cuDoubleComplex(_t.z, _t.w);
        }
    }
    else
    {
#pragma unroll
        for (int i = 0; i < NUM_ELEMENTS; i++)
        {
            dst[i] = make_cuDoubleComplex(src[i].x, src[i].y);
        }
    }
}

template <int NUM_ELEMENTS>
__device__ inline void loadAsVector(const cuDoubleComplex *src, cuDoubleComplex *dst)
{
    if (NUM_ELEMENTS % 2 == 0)
    {
        const double4 *src_ = reinterpret_cast<const double4 *>(src);
        double4 *dst_ = reinterpret_cast<double4 *>(dst);

#pragma unroll
        for (int i = 0; i < NUM_ELEMENTS / 2; i++)
        {
            dst_[i] = src_[i];
        }
    }
    else
    {

#pragma unroll
        for (int i = 0; i < NUM_ELEMENTS; i++)
        {
            dst[i] = src[i];
        }
    }
}

template <class ScalarType>
__global__ void containsNan_kernel( ScalarType *mem, int num, bool *retval)
{
    for (int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < num; idx += blockDim.x * gridDim.x)
    {
        if (isnan(mem[idx]))
        {
            *retval = true;
        }
    }
}

template <class ScalarType> bool containsNan( ScalarType *mem, int num )
{
    int threads = 256;
    int blocks = min(512, (num + threads - 1) / threads);
    bool *d_retval, retval = false;

    if (num > 0)
    {
        cudaMallocAsync(&d_retval, sizeof(bool), 0);
        cudaStreamSynchronize(0);
        cudaMemcpy(d_retval, &retval, sizeof(bool), cudaMemcpyHostToDevice);
        containsNan_kernel <<< blocks, threads>>>(mem, num, d_retval);
        cudaCheckError();
        cudaMemcpy(&retval, d_retval, sizeof(bool), cudaMemcpyDeviceToHost);
        cudaFreeAsync(d_retval, 0);
    }

    return retval;
}

template <class VectorClass>
bool spoolVector(VectorClass vec, char *filename)
{
    FILE *f = fopen (filename, "w");

    if (!f) { return false; }

    std::stringstream out;

    for (unsigned int i = 0; i < vec.size(); i++)
    {
        out << vec[i] << std::endl;
    }

    fprintf(f, out.str().c_str());
    fclose(f);
    return true;
}

} // namespace amgx
