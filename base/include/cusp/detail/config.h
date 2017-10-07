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

