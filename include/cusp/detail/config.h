// SPDX-FileCopyrightText: 2008 - 2025 NVIDIA CORPORATION. All Rights Reserved.
//
// SPDX-License-Identifier: BSD-3-Clause

/*! \file config.h
 *  \brief Defines platform configuration.
 */

#pragma once

#include <thrust/detail/config.h>
#include <thrust/version.h>

// Cusp supports Thrust v1.3 and above
#if THRUST_VERSION < 100500
#error "Thrust v1.5.0 or newer is required"
#endif 

// decorator for deprecated features
#ifdef THRUST_DEPRECATED
#define CUSP_DEPRECATED THRUST_DEPRECATED
#else
// THRUST_DEPRECATED not available in this version, use C++14 attribute or empty macro
#if __cplusplus >= 201402L
#define CUSP_DEPRECATED [[deprecated]]
#else
#define CUSP_DEPRECATED
#endif
#endif

// hooks for profiling
#if defined(CUSP_PROFILE_ENABLED)
// profiling enabled
#define CUSP_PROFILE_SCOPED()  PROFILE_SCOPED()
#define CUSP_PROFILE_DUMP()    cusp::detail::profiler::dump()
#include <cusp/detail/profiler.h>
#else
// profiling disabled
#define CUSP_PROFILE_SCOPED()
#define CUSP_PROFILE_DUMP()
#endif

