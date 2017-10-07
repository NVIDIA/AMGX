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
  ///return thrust::system::cuda::detail::arch::max_active_blocks(kernel, CTA_SIZE, dynamic_smem_bytes);//OLD Thrust
  int numBlocks = 0;
  cudaOccupancyMaxActiveBlocksPerMultiprocessor ( &numBlocks, kernel, CTA_SIZE, dynamic_smem_bytes );//NEW Thrust: THRUST_VERSION >= 100802
  return numBlocks;
#else
  return thrust::detail::backend::cuda::arch::max_active_blocks(kernel, CTA_SIZE, dynamic_smem_bytes);//Ancient Thrust: This SHOULD trigger error
#endif
}

} // end namespace arch
} // end namespace device
} // end namespace detail
} // end namespace cusp

