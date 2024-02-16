// SPDX-FileCopyrightText: 2008 - 2024 NVIDIA CORPORATION. All Rights Reserved.
//
// SPDX-License-Identifier: BSD-3-Clause

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
void convert(const SourceType& src, DestinationType& dst, amgx::thrust::host_system_tag, amgx::thrust::host_system_tag)
{
    cusp::detail::host::convert(src, dst);
}

/////////////////////////
// Host to Device Path //
/////////////////////////
template <typename SourceType, typename DestinationType>
void convert(const SourceType& src, DestinationType& dst, amgx::thrust::host_system_tag, amgx::thrust::device_system_tag)
{
    // first convert on host, then transfer to device
    typedef typename DestinationType::container DestinationContainerType;
    typedef typename DestinationContainerType::template rebind<amgx::thrust::host_system_tag>::type HostDestinationContainerType;
    
    HostDestinationContainerType tmp;

    cusp::detail::host::convert(src, tmp);

    cusp::copy(tmp, dst);
}

/////////////////////////
// Device to Host Path //
/////////////////////////
template <typename SourceType, typename DestinationType>
void convert(const SourceType& src, DestinationType& dst, amgx::thrust::device_system_tag, amgx::thrust::host_system_tag)
{
    // first transfer to host, then convert on host
    typedef typename SourceType::container SourceContainerType;
    typedef typename SourceContainerType::template rebind<amgx::thrust::host_system_tag>::type HostSourceContainerType;
    
    HostSourceContainerType tmp(src);

    cusp::detail::host::convert(tmp, dst);
}

///////////////////////////
// Device to Device Path //
///////////////////////////
template <typename SourceType, typename DestinationType>
void convert(const SourceType& src, DestinationType& dst, amgx::thrust::device_system_tag, amgx::thrust::device_system_tag)
{
    cusp::detail::device::convert(src, dst);
}

} // end namespace dispatch
} // end namespace detail
} // end namespace cusp

