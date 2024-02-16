// SPDX-FileCopyrightText: 2008 - 2024 NVIDIA CORPORATION. All Rights Reserved.
//
// SPDX-License-Identifier: BSD-3-Clause

/*! \file memory.h
 *  \brief Memory spaces and allocators
 */

#pragma once

#include <cusp/detail/config.h>
#include <thrust/iterator/iterator_traits.h>

namespace cusp
{
    using host_memory = amgx::thrust::host_system_tag;
    using device_memory = amgx::thrust::device_system_tag;
    using any_memory = amgx::thrust::any_system_tag;

  template<typename T, typename MemorySpace>
  struct default_memory_allocator;
  
  template <typename MemorySpace1, typename MemorySpace2=any_memory, typename MemorySpace3=any_memory>
  struct minimum_space;


} // end namespace cusp

#include <cusp/detail/memory.inl>
#include <basic_types.h>

namespace cusp
{
    template <typename T> struct CuspMemMap;
    template <> struct CuspMemMap<host_memory> { static const int value = AMGX_host; };
    template <> struct CuspMemMap<device_memory> { static const int value = AMGX_device; };
    template <> struct CuspMemMap<any_memory> { static const int value = AMGX_host; };
} // end namespace cusp

