// SPDX-FileCopyrightText: 2008 - 2025 NVIDIA CORPORATION. All Rights Reserved.
//
// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#include <thrust/pair.h>

#define CUDA_SAFE_CALL_NO_SYNC( call) do {                                \
 cudaError err = call;                                                    \
 if( cudaSuccess != err) {                                                \
     fprintf(stderr, "Cuda error in file '%s' in line %i : %s.\n",        \
             __FILE__, __LINE__, cudaGetErrorString( err) );              \
     exit(EXIT_FAILURE);                                                  \
 } } while (0)

#define CUDA_SAFE_CALL( call) do {                                        \
 CUDA_SAFE_CALL_NO_SYNC(call);                                            \
 cudaError err = cudaDeviceSynchronize();                                 \
 if( cudaSuccess != err) {                                                \
     fprintf(stderr, "Cuda error in file '%s' in line %i : %s.\n",        \
             __FILE__, __LINE__, cudaGetErrorString( err) );              \
     exit(EXIT_FAILURE);                                                  \
 } } while (0)


namespace cusp
{
namespace detail
{    
namespace device
{

template <typename Size1, typename Size2>
__host__ __device__ 
Size1 DIVIDE_INTO(Size1 N, Size2 granularity)
{
  return (N + (granularity - 1)) / granularity;
}

template <typename T>
amgx::thrust::pair<T,T> uniform_splitting(const T N, const T granularity, const T max_intervals)
{
    const T grains  = DIVIDE_INTO(N, granularity);

    // one grain per interval
    if (grains <= max_intervals)
        return amgx::thrust::make_pair(granularity, grains);

    // insures that:
    //     num_intervals * interval_size is >= N 
    //   and 
    //     (num_intervals - 1) * interval_size is < N
    const T grains_per_interval = DIVIDE_INTO(grains, max_intervals);
    const T interval_size       = grains_per_interval * granularity;
    const T num_intervals       = DIVIDE_INTO(N, interval_size);

    return amgx::thrust::make_pair(interval_size, num_intervals);
}

} // end namespace device
} // end namespace detail
} // end namespace cusp

