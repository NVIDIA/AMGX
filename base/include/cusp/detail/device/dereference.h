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

#pragma once

#include <cusp/detail/config.h>

#if THRUST_VERSION >= 100600
#include <thrust/detail/raw_reference_cast.h>
#define CUSP_DEREFERENCE(x)  thrust::raw_reference_cast(*x)
#else
#include <thrust/detail/backend/dereference.h>
#define CUSP_DEREFERENCE(x)  thrust::detail::backend::dereference(x)
#endif


