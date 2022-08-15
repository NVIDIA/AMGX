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
    
#include <cusp/array1d.h>

#include <cusp/copy.h>
#include <cusp/detail/host/convert.h>
#include <cusp/detail/device/convert.h>

namespace cusp
{
namespace detail
{
namespace dispatch
{

///////////////////////
// Host to Host Path //
///////////////////////
template <typename SourceType, typename DestinationType>
void convert(const SourceType& src, DestinationType& dst, cusp::host_memory, cusp::host_memory)
{
    cusp::detail::host::convert(src, dst);
}

/////////////////////////
// Host to Device Path //
/////////////////////////
template <typename SourceType, typename DestinationType>
void convert(const SourceType& src, DestinationType& dst, cusp::host_memory, cusp::device_memory)
{
    // first convert on host, then transfer to device
    typedef typename DestinationType::container DestinationContainerType;
    typedef typename DestinationContainerType::template rebind<cusp::host_memory>::type HostDestinationContainerType;
    
    HostDestinationContainerType tmp;

    cusp::detail::host::convert(src, tmp);

    cusp::copy(tmp, dst);
}

/////////////////////////
// Device to Host Path //
/////////////////////////
template <typename SourceType, typename DestinationType>
void convert(const SourceType& src, DestinationType& dst, cusp::device_memory, cusp::host_memory)
{
    // first transfer to host, then convert on host
    typedef typename SourceType::container SourceContainerType;
    typedef typename SourceContainerType::template rebind<cusp::host_memory>::type HostSourceContainerType;
    
    HostSourceContainerType tmp(src);

    cusp::detail::host::convert(tmp, dst);
}

///////////////////////////
// Device to Device Path //
///////////////////////////
template <typename SourceType, typename DestinationType>
void convert(const SourceType& src, DestinationType& dst, cusp::device_memory, cusp::device_memory)
{
    cusp::detail::device::convert(src, dst);
}

} // end namespace dispatch
} // end namespace detail
} // end namespace cusp

