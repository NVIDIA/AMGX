// SPDX-FileCopyrightText: 2008 - 2024 NVIDIA CORPORATION. All Rights Reserved.
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
#define CUSP_DEPRECATED THRUST_DEPRECATED

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

