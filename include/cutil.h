// SPDX-FileCopyrightText: 2011 - 2024 NVIDIA CORPORATION. All Rights Reserved.
//
// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#include <basic_types.h>
#include <error.h>
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
    int blocks = std::min(512, (num + threads - 1) / threads);
    bool *d_retval, retval = false;

    if (num > 0)
    {
        amgx::memory::cudaMallocAsync((void**)&d_retval, sizeof(bool));
        cudaMemcpy(d_retval, &retval, sizeof(bool), cudaMemcpyHostToDevice);
        containsNan_kernel <<< blocks, threads>>>(mem, num, d_retval);
        cudaCheckError();
        cudaMemcpy(&retval, d_retval, sizeof(bool), cudaMemcpyDeviceToHost);
        amgx::memory::cudaFreeAsync(d_retval);
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
