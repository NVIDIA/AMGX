// SPDX-FileCopyrightText: 2008 - 2024 NVIDIA CORPORATION. All Rights Reserved.
//
// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#include <cusp/detail/config.h>

#if THRUST_VERSION >= 100600
#include <thrust/detail/raw_reference_cast.h>
#define CUSP_DEREFERENCE(x)  amgx::thrust::raw_reference_cast(*x)
#else
#include <thrust/detail/backend/dereference.h>
#define CUSP_DEREFERENCE(x)  amgx::thrust::detail::backend::dereference(x)
#endif


