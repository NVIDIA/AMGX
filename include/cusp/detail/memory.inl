// SPDX-FileCopyrightText: 2008 - 2025 NVIDIA CORPORATION. All Rights Reserved.
//
// SPDX-License-Identifier: BSD-3-Clause

#include <memory>
#include <type_traits>

#include <thrust/device_allocator.h>
#include <thrust/iterator/iterator_traits.h>

#include <vector_thrust_allocator.h>

#if THRUST_VERSION >= 100600
#include <thrust/device_malloc_allocator.h>
#endif

namespace cusp
{
namespace detail
{

  template <typename MemorySpace1, typename MemorySpace2>
  struct minimum_space_impl {};
  template <typename MemorySpace>
  struct minimum_space_impl<MemorySpace,MemorySpace> { typedef MemorySpace type; };
  template <typename MemorySpace>
  struct minimum_space_impl<any_memory,MemorySpace>  { typedef MemorySpace type; };
  template <typename MemorySpace>
  struct minimum_space_impl<MemorySpace,any_memory>  { typedef MemorySpace type; };
  template <>
  struct minimum_space_impl<any_memory,any_memory>   { typedef any_memory  type; };
  
} // end namespace detail
   
  template<typename T, typename MemorySpace>
   struct default_memory_allocator
      : std::conditional<
          std::is_convertible<MemorySpace, host_memory>::value,
  
          std::allocator<T>,
  
          // XXX add backend-specific allocators here?
  
          typename std::conditional<
            std::is_convertible<MemorySpace, device_memory>::value,
  
            amgx::thrust_amgx_allocator<T, AMGX_device>,
  
            MemorySpace
          >::type
        >
  {};
  
  // TODO replace this with Thrust's minimum_space in 1.4
  template <typename MemorySpace1, typename MemorySpace2, typename MemorySpace3>
  struct minimum_space { typedef typename detail::minimum_space_impl< typename detail::minimum_space_impl<MemorySpace1,MemorySpace2>::type, MemorySpace3>::type type; };

  
} // end namespace cusp

