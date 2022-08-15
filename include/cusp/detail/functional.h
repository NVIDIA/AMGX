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
  struct zero_function : public thrust::unary_function<T,T>
{
  __host__ __device__ T operator()(const T &x) const {return T(0);}
}; // end minus

} // end namespace detail
} // end namespace cusp

