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
    
#include <cusp/detail/dispatch/convert.h>

#include <cusp/copy.h>

namespace cusp
{
namespace detail
{
  
// same format
template <typename SourceType, typename DestinationType,
          typename T1>
void convert(const SourceType& src, DestinationType& dst,
             T1, T1)
{
  cusp::copy(src, dst);
}

// different formats
template <typename SourceType, typename DestinationType,
          typename T1, typename T2>
void convert(const SourceType& src, DestinationType& dst,
             T1, T2)
{
  cusp::detail::dispatch::convert(src, dst,
      typename SourceType::memory_space(),
      typename DestinationType::memory_space());
}

} // end namespace detail

/////////////////
// Entry Point //
/////////////////
template <typename SourceType, typename DestinationType>
void convert(const SourceType& src, DestinationType& dst)
{
  CUSP_PROFILE_SCOPED();

  cusp::detail::convert(src, dst,
      typename SourceType::format(),
      typename DestinationType::format());
}

} // end namespace cusp

