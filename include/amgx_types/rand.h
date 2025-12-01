// SPDX-FileCopyrightText: 2011 - 2025 NVIDIA CORPORATION. All Rights Reserved.
//
// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#include <basic_types.h>
#include <cstdlib>

namespace amgx
{
namespace types
{

template <typename T>
__inline__ __host__ T get_rand();

template <>
__inline__ __host__ float get_rand<float>()
{
    return 1.f * rand() / RAND_MAX;
}

template <>
__inline__ __host__ double get_rand<double>()
{
    return 1.*rand() / RAND_MAX;
}

template <>
__inline__ __host__ cuComplex get_rand<cuComplex>()
{
    return make_cuComplex (1.f * rand() / RAND_MAX, 1.f * rand() / RAND_MAX);
}

template <>
__inline__ __host__ cuDoubleComplex get_rand<cuDoubleComplex>()
{
    return make_cuDoubleComplex (1.*rand() / RAND_MAX, 1.*rand() / RAND_MAX);
}

} // namespace types
} // namespace amgx
