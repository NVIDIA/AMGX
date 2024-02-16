// SPDX-FileCopyrightText: 2008 - 2024 NVIDIA CORPORATION. All Rights Reserved.
//
// SPDX-License-Identifier: BSD-3-Clause

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

