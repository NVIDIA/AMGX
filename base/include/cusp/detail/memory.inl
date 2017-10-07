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

#include <memory>

#include <thrust/device_allocator.h>
#include <thrust/iterator/iterator_traits.h>

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
      : thrust::detail::eval_if<
          thrust::detail::is_convertible<MemorySpace, host_memory>::value,
  
          thrust::detail::identity_< std::allocator<T> >,
  
          // XXX add backend-specific allocators here?
  
          thrust::detail::eval_if<
            thrust::detail::is_convertible<MemorySpace, device_memory>::value,
  
            thrust::detail::identity_< thrust::device_malloc_allocator<T> >,
  
            thrust::detail::identity_< MemorySpace >
          >
        >
  {};
  
  // TODO replace this with Thrust's minimum_space in 1.4
  template <typename MemorySpace1, typename MemorySpace2, typename MemorySpace3>
  struct minimum_space { typedef typename detail::minimum_space_impl< typename detail::minimum_space_impl<MemorySpace1,MemorySpace2>::type, MemorySpace3>::type type; };

  
} // end namespace cusp

