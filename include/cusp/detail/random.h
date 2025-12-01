// SPDX-FileCopyrightText: 2008 - 2025 NVIDIA CORPORATION. All Rights Reserved.
//
// SPDX-License-Identifier: BSD-3-Clause

#pragma once

namespace cusp
{
namespace detail
{

// array view containing random integers
template <typename T>
class random_integers;

// array view containing random real numbers in [0,1)
template <typename T>
class random_reals;

} // end namespace detail
} // end namespace cusp

#include <cusp/detail/random.inl>

