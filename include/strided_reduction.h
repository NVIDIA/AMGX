// SPDX-FileCopyrightText: 2013 - 2024 NVIDIA CORPORATION. All Rights Reserved.
//
// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#include <cstdio>
#include "sm_utils.inl"
#include <cutil.h>

#include <global_thread_handle.h>

#include <amgx_types/util.h>

namespace amgx
{
namespace strided_reduction
{

// basic definitions

#ifdef SR_SDF
#error "change SR_SDF shortcut name, macro name conflict"
#endif
#define SR_SDF static __device__ __forceinline__

SR_SDF int warp_binary_count(const bool pred) { return __popc(utils::ballot(pred)); }
template<int I> struct is_pow2   { static const int value = (I & (I - 1)) == 0; };

// detect arch

#define strided_reduction_arch_KEPLER 2
#define strided_reduction_ARCH strided_reduction_arch_KEPLER

//ops
struct op_sum
{
    template<class T>
    __device__ __forceinline__ T compute(const T &a, const T &b)const
    {
        return a + b;
    }
};
struct op_max
{
    template<class T>
    __device__ __forceinline__ T compute(const T &a, const T &b)const
    {
        if (a > b) { return a; }
        else { return b; }
    }
};
struct op_min
{
    template<class T>
    __device__ __forceinline__ T compute(const T &a, const T &b)const
    {
        if (a < b) { return a; }
        else { return b; }
    }
};
struct op_or
{
    template<class T>
    __device__ __forceinline__ T compute(const T &a, const T &b)const
    {
        return a | b;
    }
};
struct op_and
{
    template<class T>
    __device__ __forceinline__ T compute(const T &a, const T &b)const
    {
        return a & b;
    }
};
struct op_xor
{
    template<class T>
    __device__ __forceinline__ T compute(const T &a, const T &b)const
    {
        return a ^ b;
    }
};
//////////////

//warp_loader: utility class for accessing large words per thread coalescing by warp

template<class DERIVED, class TT, int K>
struct loader_base
{
    template<class T>
    SR_SDF void load1(const T *in, const int i0, const int N, T &a)
    {
        DERIVED::load(in, i0, N, &a);
    }
    template<class T>
    SR_SDF void load2(const T *in, const int i0, const int N, T &a, T &b)
    {
        T tmp[2];
        DERIVED::load(in, i0, N, tmp);
        a = tmp[0];
        b = tmp[1];
    }
    template<class T>
    SR_SDF void load4(const T *in, const int i0, const int N, T &a, T &b, T &c, T &d)
    {
        T tmp[4];
        DERIVED::load(in, i0, N, tmp);
        a = tmp[0];
        b = tmp[1];
        c = tmp[1];
        d = tmp[1];
    }

    SR_SDF int align_shift(const int i)
    {
        return 0;
    }

};


// in order to use it for specific type - specialization is required
template<class T, int K> struct warp_loader_not_specialized : public  loader_base< warp_loader_not_specialized<T, K>, T, K >
{
};

template<class T, int K> struct warp_loader : public warp_loader_not_specialized<T, K>
{
};

SR_SDF bool wl_valid(const int i0, const int N)
{
    return true;
}

//specialized loaders
template<> struct warp_loader<int, 4> : public  loader_base< warp_loader<int, 4>, int, 4 >
{
    SR_SDF int align_shift(const int i)
    {
        return i % 4;
    }
    SR_SDF void load(const int *in, const int i0, const int N, int *out)
    {
        int4 i;
        {
            i = __ldcg((int4 *  ) (in + i0));
            out[0] = i.x;
            out[1] = i.y;
            out[2] = i.z;
            out[3] = i.w;
        }
    }
};

template<> struct warp_loader<float, 4> : public  loader_base< warp_loader<float, 4>, float, 4 >
{
    SR_SDF int align_shift(const int i)
    {
        return i % 4;
    }
    SR_SDF void load(const float *in, const int i0, const int N, float *out)
    {
        float4 i;
        {
            i = __ldcg((float4 *  ) (in + i0));
            out[0] = i.x;
            out[1] = i.y;
            out[2] = i.z;
            out[3] = i.w;
        }
    }
};

template<> struct warp_loader<int, 2> : public  loader_base< warp_loader<int, 2>, int, 2 >
{
    SR_SDF int align_shift(const int i)
    {
        return i % 2;
    }
    SR_SDF void load(const int *in, const int i0, const int N, int *out)
    {
        int2 i;
        {
            i = __ldcg((int2 *  ) (in + i0));
            out[0] = i.x;
            out[1] = i.y;
        }
    }
};

template<> struct warp_loader<float, 2> : public  loader_base< warp_loader<float, 2>, float, 2 >
{
    SR_SDF int align_shift(const int i)
    {
        return i % 2;
    }
    SR_SDF void load(const float *in, const int i0, const int N, float *out)
    {
        float2 i;
        {
            i = __ldcg((float2 *  ) (in + i0));
            out[0] = i.x;
            out[1] = i.y;
        }
    }
};

template<class T> struct warp_loader<T, 1> : public  loader_base< warp_loader<T, 1>, T, 1 >
{
    SR_SDF int align_shift(const int i)
    {
        return 0;
    }
    SR_SDF void load(const T *in, const int i0, const int N, T *out)
    {
        if (wl_valid(i0, N)) { out[0] = in[i0]; }
    }
};

//////////////
// utility for converting 64 bit values to 2x32
template<class T> struct datatype_utils {};
template<> struct datatype_utils<double>
{
    __device__ __forceinline__ static double from_hiloint(const int hi, const int lo) { return __hiloint2double(hi, lo);}
    __device__ __forceinline__ static int to_hiint(const double x) { return __double2hiint(x); }
    __device__ __forceinline__ static int to_loint(const double x) { return __double2loint(x); }
};
template<> struct datatype_utils<long long int>
{
    __device__ __forceinline__ static long long int from_hiloint(const int hi, const int lo) { return __double_as_longlong(__hiloint2double(hi, lo));}
    __device__ __forceinline__ static int to_hiint(const long long int x) { return __double2hiint( __longlong_as_double(x)); }
    __device__ __forceinline__ static int to_loint(const long long int x) { return __double2loint( __longlong_as_double(x)); }
};
template<> struct datatype_utils<unsigned long long int>
{
    __device__ __forceinline__ static unsigned long long int from_hiloint(const int hi, const int lo) { return __double_as_longlong(__hiloint2double(hi, lo));}
    __device__ __forceinline__ static int to_hiint(const unsigned long long int x) { return __double2hiint( __longlong_as_double(x)); }
    __device__ __forceinline__ static int to_loint(const unsigned long long int x) { return __double2loint( __longlong_as_double(x)); }
};
/////////////////////////////////////////
//warp util:  in-warp,strided reductions
/////////////////////////////////////////

template<int SIZE_IN_BYTES, int ARCH> struct warputil {};

//KEPLER, 4 bytes values
template<>
struct warputil<4, strided_reduction_arch_KEPLER>
{
    template<int STRIDE, int NVALS, int CTA_SIZE, int WARP_SIZE, class OP, class T> SR_SDF void warp_reduce_stride(T &value, const OP &op)
    {
        if (STRIDE & (STRIDE - 1)) //NPOT
        {
            const int laneId = utils::lane_id();
#pragma unroll

            for (int i = STRIDE; i < NVALS; i *= 2)
            {
                T tmp = utils::shfl(value, laneId + i);

                if (laneId + i < WARP_SIZE)
                {
                    value = op.compute(value,  tmp);
                }
            }
        }
        else
        {
#pragma unroll

            for (int i = NVALS / 2; i >= STRIDE; i /= 2)
            {
                value = op.compute(value,  utils::shfl_xor(value, i));
            }
        }
    }
};

//KEPLER, 8 bytes values
template<>
struct warputil<8, strided_reduction_arch_KEPLER>
{
    template<int STRIDE, int NVALS, int CTA_SIZE, int WARP_SIZE, class OP, class T> SR_SDF void warp_reduce_stride(T &value, const OP &op)
    {
        if (STRIDE & (STRIDE - 1)) //NPOT
        {
            const int laneId = utils::lane_id();
#pragma unroll

            for (int i = STRIDE; i < NVALS; i *= 2)
            {
                int tmp_hi = datatype_utils<T>::to_hiint(value);
                int tmp_lo = datatype_utils<T>::to_loint(value);
                tmp_hi = utils::shfl(tmp_hi, laneId + i);
                tmp_lo = utils::shfl(tmp_lo, laneId + i);
                T tmp = datatype_utils<T>::from_hiloint(tmp_hi, tmp_lo);

                if (laneId + i < WARP_SIZE)
                {
                    value = op.compute(value, tmp);
                }
            }
        }
        else
        {
#pragma unroll

            for (int i = NVALS / 2; i >= STRIDE; i /= 2)
            {
                int v_hi = datatype_utils<T>::to_hiint(value);
                int v_lo = datatype_utils<T>::to_loint(value);
                v_hi = utils::shfl_xor(v_hi, i, WARP_SIZE);
                v_lo = utils::shfl_xor(v_lo, i, WARP_SIZE);
                value = op.compute(value, datatype_utils<T>::from_hiloint(v_hi, v_lo));
            }
        }
    }
};

/////////////////////////////////
// STRUDED REDUCTION KERNEL
/////////////////////////////////
// future architectures: this handles no more than 32 warps per block (current Kepler's limit)
// STRIDE can be in range [1..32]

template<int STRIDE, int CTA_SIZE, int WARP_SIZE, int BLOCKS_PER_THREAD, class OP, class T, class V, class TRANSFORM>
__global__
__launch_bounds__( CTA_SIZE, 2 )
void strided_reduction(const T *X, const int N, V *sums, const TRANSFORM tx = TRANSFORM(), const OP op = OP())
{
    int tid = threadIdx.x + blockDim.x * blockIdx.x;
    const int lane_id = utils::lane_id();
    const int warp_id = utils::warp_id();
    const int dgrid   = gridDim.x * blockDim.x;
    __shared__ V warp_results[STRIDE][CTA_SIZE / WARP_SIZE];
    V x_warp_0 = 0;

    while (utils::any(tid < N))
    {
        V x = types::util<V>::get_zero();

        if (BLOCKS_PER_THREAD == 1)
        {
            if (tid < N ) { x = tx.apply(X[tid]); }
        }
        else
        {
            int addr = tid;

#pragma unroll
            for (int b = 0; b < BLOCKS_PER_THREAD; b++)
            {
                if (addr < N) { x += tx.apply(X[addr]); }

                addr += dgrid;
            }
        }

        warputil<sizeof(V), strided_reduction_ARCH>::template warp_reduce_stride<STRIDE, 32, CTA_SIZE, WARP_SIZE>(x, op);

        if (lane_id < STRIDE) { x_warp_0 += x; }

        tid += dgrid * BLOCKS_PER_THREAD;
    }

    if (lane_id < STRIDE)
    {
        warp_results[(STRIDE & (STRIDE - 1)) == 0 ? (lane_id) : (tid % STRIDE)][warp_id] = x_warp_0;
    }

    __syncthreads();

    //now that each warp has its total...
    //...the first warp computes the sum of the warps,

    if (warp_id == 0)
    {
        V my_result = 0;
#pragma unroll

        for (int k = 0; k < STRIDE; k++)
        {
            V y;
            y = (lane_id < CTA_SIZE / WARP_SIZE) ? warp_results[k][lane_id] : 0;
            warputil<sizeof(V), strided_reduction_ARCH>::template warp_reduce_stride < 1, (is_pow2 < CTA_SIZE / WARP_SIZE >::value ? 1 : 2)*CTA_SIZE / WARP_SIZE, CTA_SIZE, WARP_SIZE > (y, op);

            if (lane_id == k) { my_result = y; }
        }

        if (lane_id < STRIDE) { sums[blockIdx.x * STRIDE + lane_id] = my_result; }
    }
}

template<int STRIDE, int CTA_SIZE, int WARP_SIZE, class T, class OP>
__device__ __forceinline__ void warp_reduce(T &y, const OP &op = OP())
{
    warputil<sizeof(T), strided_reduction_ARCH>::template warp_reduce_stride<STRIDE, WARP_SIZE, CTA_SIZE, WARP_SIZE>(y, op);
}


template<int STRIDE, int CTA_SIZE, int WARP_SIZE, bool load_previous, class T, class OP>
__device__ __forceinline__ void block_reduce(const T x_warp, T *sums, const OP &op = OP(), const int tid = 0)
{
    __shared__ T warp_results[STRIDE][CTA_SIZE / WARP_SIZE];
    const int lane_id = utils::lane_id();;
    const int warp_id = utils::warp_id();

    if (lane_id < STRIDE)
    {
        warp_results[((STRIDE & (STRIDE - 1)) == 0) ? (lane_id) : (tid % STRIDE)][warp_id] = x_warp;
    }

    __syncthreads();

    //now that each warp has its total...
    //...the first warp computes the sum of the warps, using warp scan on block data

    if (warp_id == 0)
    {
        T my_result = 0;
#pragma unroll

        for (int k = 0; k < STRIDE; k++)
        {
            T y;
            y = (lane_id < CTA_SIZE / WARP_SIZE) ? warp_results[k][lane_id] : 0;
            warputil<sizeof(T), strided_reduction_ARCH>::template warp_reduce_stride < 1, (is_pow2 < CTA_SIZE / WARP_SIZE >::value ? 1 : 2)*CTA_SIZE / WARP_SIZE, CTA_SIZE, WARP_SIZE > (y, op);

            if (lane_id == k) { my_result = y; }
        }

        if (lane_id < STRIDE)
        {
            if (load_previous)
            {
                my_result = op.compute(my_result, sums[blockIdx.x * STRIDE + lane_id]);
            }

            sums[blockIdx.x * STRIDE + lane_id] = my_result;
        }
    }
}

template<int STRIDE, int CTA_SIZE, int WARP_SIZE, class T>
__device__ __forceinline__ void block_count(const T x_warp, T *sums, const int tid = 0)
{
    block_reduce<STRIDE, CTA_SIZE, WARP_SIZE, false>(x_warp, sums, op_sum(), tid);
}
//

template<int STRIDE, int CTA_SIZE, int WARP_SIZE, class T, class OP>
__device__ __forceinline__ void block_reduce_atomic(const int x_warp, T *sums, const OP &op = OP(), const int tid = 0)
{
    __shared__ T warp_results[STRIDE][CTA_SIZE / WARP_SIZE];
    const int lane_id = utils::lane_id();;
    const int warp_id = utils::warp_id();

    if (lane_id < STRIDE)
    {
        warp_results[((STRIDE & (STRIDE - 1)) == 0) ? (lane_id) : (tid % STRIDE)][warp_id] = x_warp; //NPOT stride: stride alignment messes
    }

    __syncthreads();

    //now that each warp has its total...
    //...the first warp computes the sum of the warps, using warp scan on block data

    if (warp_id == 0)
    {
        T my_result = 0;
#pragma unroll

        for (int k = 0; k < STRIDE; k++)
        {
            T y;
            y = (lane_id < CTA_SIZE / WARP_SIZE) ? warp_results[k][lane_id] : 0;
            warputil<sizeof(T), strided_reduction_ARCH>::template warp_reduce_stride < 1, (is_pow2 < CTA_SIZE / WARP_SIZE >::value ? 1 : 2)*CTA_SIZE / WARP_SIZE, CTA_SIZE, WARP_SIZE > (y, op);

            if (lane_id == k) { my_result = y; }
        }

        if (lane_id < STRIDE)
        {
            atomicAdd(sums + lane_id, my_result);
        }
    }
}

// kernel which collects reduction results of blocks

template<class T, int STRIDE, int WARP_SIZE, class OP>
__global__ void strided_reduction_collect_partials(T *out_sums, T *partial_sums, const int NBlocks, const OP op = OP())
{
    int tid = threadIdx.x;
    __shared__ T X[STRIDE]; //to handle NPOT

    if (threadIdx.x < STRIDE) { X[threadIdx.x] = 0; }

    while (utils::any(tid < NBlocks * STRIDE))
    {
        T x = (tid < NBlocks * STRIDE) ? partial_sums[tid] : 0;
        warputil<sizeof(T), strided_reduction_ARCH>::template warp_reduce_stride<STRIDE, 32, 32, WARP_SIZE>(x, op);

        if (STRIDE & (STRIDE - 1)) //NPOT, handle the fact that stride messes around
        {
            if (threadIdx.x < STRIDE) { X[tid % STRIDE] = op.compute( X[tid % STRIDE], x); }
        }
        else
        {
            if (threadIdx.x < STRIDE) { X[threadIdx.x] = op.compute(X[threadIdx.x], x); }
        }

        tid  += blockDim.x;
    }

    if (threadIdx.x < STRIDE) { out_sums[threadIdx.x] = X[threadIdx.x]; }
}

template<int STRIDE, class scalar_t, class OP>
void count_block_results(scalar_t *out_host, const int n_blocks, scalar_t *out_d, const OP &op = OP())
{
    strided_reduction_collect_partials<scalar_t, STRIDE, 32, OP> <<< 1, 32, 0, amgx::thrust::global_thread_handle::get_stream()>>>(out_d, out_d, n_blocks);
    cudaCheckError();
    cudaMemcpy(out_host, out_d, STRIDE * sizeof(scalar_t), cudaMemcpyDeviceToHost);
}

template<class scalar_t, class OP>
scalar_t count_block_results_1(const int n_blocks, scalar_t *out_d, const OP &op = OP())
{
    scalar_t ret;
    count_block_results<1, scalar_t, OP>(&ret, n_blocks, out_d, op);
    return ret;
}

template<class T, int STRIDE, class OP>
struct block_count_gatherer
{

    __device__ __forceinline__ block_count_gatherer(T *sets_per_block)
    {
        this->count(sets_per_block);
    }

    __device__ __forceinline__ void count(T *sets_per_block)
    {
        sets_per_block[gridDim.x] = 0;
        __shared__ bool isLastBlockDone;
        __threadfence(); //wait for results to be visible to others, if(threadIdx.x < STRIDE)
        __syncthreads(); //just in case, probably unnecessary

        if (threadIdx.x == 0)
        {
            unsigned int value = atomicInc((unsigned int *)(sets_per_block + gridDim.x), gridDim.x);
            isLastBlockDone = (value == (gridDim.x - 1));
        }

        __syncthreads(); //sync isLastBlockDone
        const int WARP_SIZE = 32;

        if (isLastBlockDone && threadIdx.x < WARP_SIZE) //only one block goes there, first warp computes sums
        {
            int tid = threadIdx.x;
            const int NBlocks = gridDim.x;
            __shared__ T X[STRIDE];

            if (threadIdx.x < STRIDE) { X[threadIdx.x] = 0; }

#pragma unroll 4
            while (utils::any(tid < NBlocks * STRIDE))
            {
                T x = (tid < NBlocks * STRIDE) ? sets_per_block[tid] : 0;
                warputil<sizeof(T), strided_reduction_ARCH>::template warp_reduce_stride<STRIDE, 32, 32, WARP_SIZE>(x, OP());

                if (threadIdx.x < STRIDE) { X[(STRIDE & (STRIDE - 1)) ? tid % STRIDE : threadIdx.x] += x; }

                tid  += 32;
            }

            if (threadIdx.x < STRIDE)
            {
                sets_per_block[threadIdx.x] = 10000000000 + X[threadIdx.x];
            }
        }
    }
};

struct nop_transform
{
    template<class T>
    __device__ __forceinline__ T apply(const T &x) const
    {
        return x;
    }
};

struct square_transform
{
    template<class T>
    __device__ __forceinline__ typename types::PODTypes<T>::type apply(const T &x) const
    {
        typename types::PODTypes<T>::type   t = types::util<T>::abs(x);
        return x * x;
    }

    // specializations for complex datatypes
    __device__ __forceinline__ float apply(const cuComplex &x) const
    {
        return cuCrealf(x)*cuCrealf(x) + cuCimagf(x)*cuCimagf(x);
    }

    __device__ __forceinline__ double apply(const cuDoubleComplex &x) const
    {
        return cuCreal(x)*cuCreal(x) + cuCimag(x)*cuCimag(x);
    }
};

struct fabs_transform
{
    template<class T>
    __device__ __forceinline__ typename types::PODTypes<T>::type apply(const T &x) const
    {
        return types::util<T>::abs(x);
    }
};

// block scan
template< int STRIDE, class scalar_out, class scalar_t, class TRANSFORM>
void launch_strided_reduction(scalar_out *out_host, const scalar_t *in_d, const long long int N, const TRANSFORM tx = TRANSFORM())
{
    const int ideal_cta_size = 1024;
    const int cta_size = ((STRIDE & (STRIDE - 1)) == 0) ? ideal_cta_size : (ideal_cta_size / (32 * STRIDE) > 0 ? ideal_cta_size / (32 * STRIDE) : 1) * (32 * STRIDE);
    const int n_items_per_thread = 16;
    const int n_blocks = std::min(  (long long int) 13 * 2, (N - 1) / (n_items_per_thread * cta_size) + 1   ); //just one wave of blocks
    const int out_size = n_blocks * STRIDE;
    scalar_out *out_d = 0;
    amgx::memory::cudaMallocAsync((void **) &out_d, out_size * sizeof(scalar_out));
    cudaCheckError();
    cudaMemset(out_d, 0, out_size * sizeof(scalar_out));
    cudaFuncSetCacheConfig(strided_reduction<STRIDE, cta_size, 32, 16, op_sum, scalar_t, scalar_out, TRANSFORM>, cudaFuncCachePreferL1);
    strided_reduction<STRIDE, cta_size, 32, 16, op_sum, scalar_t, scalar_out, TRANSFORM> <<< n_blocks, cta_size, 0, amgx::thrust::global_thread_handle::get_stream()>>>(in_d, N, out_d, tx);
    cudaCheckError();
    strided_reduction_collect_partials<scalar_out, STRIDE, 32, op_sum> <<< 1, 32, 0, amgx::thrust::global_thread_handle::get_stream()>>>(out_d, out_d, n_blocks);
    cudaCheckError();
    cudaMemcpy(out_host, out_d, STRIDE * sizeof(scalar_out), cudaMemcpyDeviceToHost);
    amgx::memory::cudaFreeAsync((void *) out_d);
}

template< class scalar_t, class scalar_out, class TRANSFORM, bool real>
struct strided_reduction_dispatch
{
};

template< class scalar_t, class scalar_out, class TRANSFORM>
struct strided_reduction_dispatch<scalar_t, scalar_out, TRANSFORM, true>
{
    static void run(const int bsize, scalar_out *out_host, const scalar_t *in_d, const long long int N, const TRANSFORM tx)
    {
#define strided_reduction_INSTANTIATE(I)else if(bsize == I) launch_strided_reduction<I>(out_host, in_d, N, tx);

        if (0) {}
        strided_reduction_INSTANTIATE(1)
        strided_reduction_INSTANTIATE(2)
        strided_reduction_INSTANTIATE(3)
        strided_reduction_INSTANTIATE(4)
        strided_reduction_INSTANTIATE(5)
        strided_reduction_INSTANTIATE(6)
        strided_reduction_INSTANTIATE(7)
        strided_reduction_INSTANTIATE(8)
        strided_reduction_INSTANTIATE(9)
        strided_reduction_INSTANTIATE(10)
        strided_reduction_INSTANTIATE(11)
        strided_reduction_INSTANTIATE(12)
        strided_reduction_INSTANTIATE(13)
        strided_reduction_INSTANTIATE(14)
        strided_reduction_INSTANTIATE(15)
        strided_reduction_INSTANTIATE(16)
        strided_reduction_INSTANTIATE(17)
        strided_reduction_INSTANTIATE(18)
        strided_reduction_INSTANTIATE(19)
        strided_reduction_INSTANTIATE(20)
        strided_reduction_INSTANTIATE(21)
        strided_reduction_INSTANTIATE(22)
        strided_reduction_INSTANTIATE(23)
        strided_reduction_INSTANTIATE(24)
        strided_reduction_INSTANTIATE(25)
        strided_reduction_INSTANTIATE(26)
        strided_reduction_INSTANTIATE(27)
        strided_reduction_INSTANTIATE(28)
        strided_reduction_INSTANTIATE(29)
        strided_reduction_INSTANTIATE(30)
        strided_reduction_INSTANTIATE(31)
        strided_reduction_INSTANTIATE(32)
        else { FatalError("Strided reduction is not implemented for blocksize > 32", AMGX_ERR_NOT_IMPLEMENTED); }
    }
};

template< class scalar_t, class scalar_out, class TRANSFORM>
struct strided_reduction_dispatch<scalar_t, scalar_out, TRANSFORM, false>
{
    /*static void run(const int bsize, scalar_out *out_host, const scalar_t *in_d, const long long int N, const TRANSFORM tx)
    {
        //@TODO, optimize for complex
        // complex
    }*/
};


template< class scalar_t, class TRANSFORM>
void reduction_generic_dispatch(const int bsize, typename types::PODTypes<scalar_t>::type *out_host, const scalar_t *in_d, const long long int N, const TRANSFORM tx = TRANSFORM())
{
    typedef typename types::PODTypes<scalar_t>::type scalar_out;
    strided_reduction_dispatch< scalar_t, scalar_out, TRANSFORM, true >::run (bsize, out_host, in_d, N, tx);
}


// block binary compaction
__device__ __forceinline__ int warp_exclusive_binary_scan(const bool input)
{
    return __popc(utils::ballot(input) & utils::lane_mask_lt());
}

template<int CTA_SIZE, int WARP_SIZE> __device__ __forceinline__ int block_binary_scan(const bool input, int &total)
{
    const int warpId = utils::warp_id();
    const int laneId = utils::lane_id();
    volatile __shared__ int s_warp_results[CTA_SIZE / WARP_SIZE];
    int thread_result = input + __popc(utils::ballot(input) & utils::lane_mask_lt());

    if (laneId == WARP_SIZE - 1)
    {
        s_warp_results[warpId] = thread_result;
    }

    __syncthreads();

    if (warpId == 0)
    {
#pragma unroll

        for (int i = 0; i < CTA_SIZE / WARP_SIZE - 1; i++)
        {
            s_warp_results[i + 1] += s_warp_results[i];
        }
    }

    __syncthreads();

    if (warpId != 0)
    {
        thread_result += s_warp_results[warpId - 1];
    }

    total = s_warp_results[CTA_SIZE / WARP_SIZE - 1];
    return thread_result;
}

template<int CTA_SIZE, int WARP_SIZE, int ONE_PASS_UNSTABLE> __device__ __forceinline__ void block_binary_compaction
(int *out_count, int *out, int block_id,
 const bool flag, const int global_index)
{
    int total;
    int new_index_in_block = block_binary_scan<CTA_SIZE, WARP_SIZE>(flag, total) - flag; //exclusive
    __shared__ int s_new_indices[CTA_SIZE + 1];

    if (flag)
    {
        s_new_indices[new_index_in_block] = global_index;
    }

    if (ONE_PASS_UNSTABLE)
    {
        __syncthreads();//not necessary, try remove
        __shared__ int offset;

        if (threadIdx.x == 0)
        {
            offset = atomicAdd(out_count, total);
        }

        __syncthreads();

        if (threadIdx.x < total && out)
        {
            out[offset + threadIdx.x] = s_new_indices[threadIdx.x];
        }
    }
    else
    {
        __syncthreads();

        if (threadIdx.x < total && out)
        {
            out[ block_id * CTA_SIZE + threadIdx.x ] = s_new_indices[threadIdx.x] ;
        }

        if (threadIdx.x == 0 && out_count)
        {
            out_count[block_id] = total;
        }
    }
}

} // namespace strided_reduction
} // namespace amgx
