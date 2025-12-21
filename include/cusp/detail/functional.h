// SPDX-FileCopyrightText: 2008 - 2025 NVIDIA CORPORATION. All Rights Reserved.
//
// SPDX-License-Identifier: BSD-3-Clause

/*! \file functional.h
 *  \brief Defines templated functors and traits analogous to what
 *         is found in stl and boost's functional.
 */

#pragma once

#include <cusp/detail/config.h>

#include <thrust/functional.h>

namespace cusp
{
namespace detail
{

template<typename T>
  struct zero_function
{
  __host__ __device__ T operator()(const T &x) const {return T(0);}
}; // end minus

template<typename T>
  struct identity_function
{
  __host__ __device__ const T& operator()(const T &x) const {return x;}
}; // end identity

} // end namespace detail
} // end namespace cusp

