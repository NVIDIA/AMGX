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

/*! \file cmath.h
 *  \brief Complex math functions
 */

#pragma once

#include <cusp/detail/config.h>

#include <cmath>

// We need this to make sure code inside cusp:: that calls sqrt() using real numbers
// doesn't try to call the complex sqrt, but the standard sqrt
namespace cusp
{
  template <typename ValueType> 
    __host__ __device__ 
    inline ValueType cos(ValueType x){
    return std::cos(x);
  }
  template <typename ValueType> 
    __host__ __device__ 
    inline ValueType sin(ValueType x){
    return std::sin(x);
  }
  template <typename ValueType> 
    __host__ __device__ 
    inline ValueType tan(ValueType x){
    return std::tan(x);
  }
  template <typename ValueType> 
    __host__ __device__ 
    inline ValueType acos(ValueType x){
    return std::acos(x);
  }
  template <typename ValueType> 
    __host__ __device__ 
    inline ValueType asin(ValueType x){
    return std::asin(x);
  }
  template <typename ValueType> 
    __host__ __device__ 
    inline ValueType atan(ValueType x){
    return std::atan(x);
  }
  template <typename ValueType> 
    __host__ __device__ 
    inline ValueType atan2(ValueType x,ValueType y){
    return std::atan2(x,y);
  }
  template <typename ValueType> 
    __host__ __device__ 
    inline ValueType cosh(ValueType x){
    return std::cosh(x);
  }
  template <typename ValueType> 
    __host__ __device__ 
    inline ValueType sinh(ValueType x){
    return std::sinh(x);
  }
  template <typename ValueType> 
    __host__ __device__ 
    inline ValueType tanh(ValueType x){
    return std::tanh(x);
  }
  template <typename ValueType> 
    __host__ __device__ 
    inline ValueType exp(ValueType x){
    return std::exp(x);
  }
  template <typename ValueType> 
    __host__ __device__ 
    inline ValueType log(ValueType x){
    return std::log(x);
  }
  template <typename ValueType> 
    __host__ __device__ 
    inline ValueType log10(ValueType x){
    return std::log10(x);
  }
  template <typename ValueType, typename ExponentType> 
    __host__ __device__ 
    inline ValueType pow(ValueType x, ExponentType e){
    return std::pow(x,e);
  }
  template <typename ValueType>
    __host__ __device__ 
    inline ValueType sqrt(ValueType x){
    return std::sqrt(x);
  }
  template <typename ValueType>
    __host__ __device__ 
    inline ValueType abs(ValueType x){
    return std::abs(x);
  }
  template <typename ValueType>
    __host__ __device__ 
    inline ValueType conj(ValueType x){
    return x;
  }
}
