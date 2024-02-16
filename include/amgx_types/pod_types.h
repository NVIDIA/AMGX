// SPDX-FileCopyrightText: 2011 - 2024 NVIDIA CORPORATION. All Rights Reserved.
//
// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#include <basic_types.h>

// definition of base type for supported AMGX datatype which we would call PODType

namespace amgx
{

namespace types
{

// plain data traits
template <typename T>
struct PODTypes;

template <>
struct PODTypes <float>
{
    // raw datatype for AMGX datatype
    typedef float type;
    // tconfig vector precision of raw datatype (used in TConfig templates construction)
    static const AMGX_VecPrecision vec_prec = AMGX_vecFloat;
    // number of raw dataitems in AMGX datatype
    static const int pod_items = 1;
};

template <>
struct PODTypes <double>
{
    typedef double type;
    static const AMGX_VecPrecision vec_prec = AMGX_vecDouble;
    static const int pod_items = 1;
};

template <>
struct PODTypes <cuComplex>
{
    typedef float type;
    static const AMGX_VecPrecision vec_prec = AMGX_vecFloat;
    static const int pod_items = 2;
};

template <>
struct PODTypes <cuDoubleComplex>
{
    typedef double type;
    static const AMGX_VecPrecision vec_prec = AMGX_vecDouble;
    static const int pod_items = 2;
};


} // namespace types

} // namespace amgx