/* Copyright (c) 2011-2017, NVIDIA CORPORATION. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *  * Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *  * Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *  * Neither the name of NVIDIA CORPORATION nor the names of its
 *    contributors may be used to endorse or promote products derived
 *    from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
 * EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 * PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
 * CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 * EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 * PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
 * OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

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