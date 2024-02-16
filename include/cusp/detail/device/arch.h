// SPDX-FileCopyrightText: 2008 - 2024 NVIDIA CORPORATION. All Rights Reserved.
//
// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#include <cusp/detail/config.h>

///TODO: FIX ME:
///
///#if THRUST_VERSION >= 100600
///#include <thrust/system/cuda/detail/arch.h>
///#else
///#include <thrust/detail/backend/cuda/arch.h>
///#endif

#include <cuda_runtime_api.h>

namespace cusp
{
namespace detail
{
namespace device
{
namespace arch
{

template <typename KernelFunction>
size_t max_active_blocks(KernelFunction kernel, const size_t CTA_SIZE, const size_t dynamic_smem_bytes)
{
#if THRUST_VERSION >= 100600
  ///return amgx::thrust::system::cuda::detail::arch::max_active_blocks(kernel, CTA_SIZE, dynamic_smem_bytes);//OLD Thrust
  int numBlocks = 0;
  cudaOccupancyMaxActiveBlocksPerMultiprocessor ( &numBlocks, kernel, CTA_SIZE, dynamic_smem_bytes );//NEW Thrust: THRUST_VERSION >= 100802
  return numBlocks;
#else
  return amgx::thrust::detail::backend::cuda::arch::max_active_blocks(kernel, CTA_SIZE, dynamic_smem_bytes);//Ancient Thrust: This SHOULD trigger error
#endif
}

} // end namespace arch
} // end namespace device
} // end namespace detail
} // end namespace cusp

