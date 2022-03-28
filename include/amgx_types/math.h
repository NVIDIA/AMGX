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

// add operator
__host__ __device__ __inline__ cuComplex operator+(const cuComplex &lhs, const cuComplex &rhs)
{
    return cuCaddf(lhs, rhs);
}
__host__ __device__ __inline__ cuComplex operator+(const cuComplex &lhs, const cuDoubleComplex &rhs)
{
    return make_cuComplex(cuCrealf(lhs) + cuCreal(rhs), cuCimagf(lhs) + cuCimag(rhs));
}
__host__ __device__ __inline__ cuComplex operator+(const cuComplex &lhs, const float &rhs)
{
    return make_cuComplex(cuCrealf(lhs) + rhs, cuCimagf(lhs));
}
__host__ __device__ __inline__ cuComplex operator+(const cuComplex &lhs, const double &rhs)
{
    return make_cuComplex(cuCrealf(lhs) + rhs, cuCimagf(lhs));
}

__host__ __device__ __inline__ cuDoubleComplex operator+(const cuDoubleComplex &lhs, const cuComplex &rhs)
{
    return make_cuDoubleComplex(cuCreal(lhs) + cuCrealf(rhs), cuCimag(lhs) + cuCimagf(rhs));
}
__host__ __device__ __inline__ cuDoubleComplex operator+(const cuDoubleComplex &lhs, const cuDoubleComplex &rhs)
{
    return cuCadd(lhs, rhs);
}
__host__ __device__ __inline__ cuDoubleComplex operator+(const cuDoubleComplex &lhs, const float &rhs)
{
    return make_cuDoubleComplex(cuCreal(lhs) + rhs, cuCimag(lhs));
}
__host__ __device__ __inline__ cuDoubleComplex operator+(const cuDoubleComplex &lhs, const double &rhs)
{
    return make_cuDoubleComplex(cuCreal(lhs) + rhs, cuCimag(lhs));
}

// sub operator
__host__ __device__ __inline__ cuComplex operator-(const cuComplex &lhs, const cuComplex &rhs)
{
    return cuCsubf(lhs, rhs);
}
__host__ __device__ __inline__ cuComplex operator-(const cuComplex &lhs, const cuDoubleComplex &rhs)
{
    return make_cuComplex(cuCrealf(lhs) - cuCreal(rhs), cuCimagf(lhs) - cuCimag(rhs));
}
__host__ __device__ __inline__ cuComplex operator-(const cuComplex &lhs, const float &rhs)
{
    return make_cuComplex(cuCrealf(lhs) - rhs, cuCimagf(lhs));
}
__host__ __device__ __inline__ cuComplex operator-(const cuComplex &lhs, const double &rhs)
{
    return make_cuComplex(cuCrealf(lhs) - rhs, cuCimagf(lhs));
}

__host__ __device__ __inline__ cuDoubleComplex operator-(const cuDoubleComplex &lhs, const cuComplex &rhs)
{
    return make_cuDoubleComplex(cuCreal(lhs) - cuCrealf(rhs), cuCimag(lhs) - cuCimagf(rhs));
}
__host__ __device__ __inline__ cuDoubleComplex operator-(const cuDoubleComplex &lhs, const cuDoubleComplex &rhs)
{
    return cuCsub(lhs, rhs);
}
__host__ __device__ __inline__ cuDoubleComplex operator-(const cuDoubleComplex &lhs, const float &rhs)
{
    return make_cuDoubleComplex(cuCreal(lhs) - rhs, cuCimag(lhs));
}
__host__ __device__ __inline__ cuDoubleComplex operator-(const cuDoubleComplex &lhs, const double &rhs)
{
    return make_cuDoubleComplex(cuCreal(lhs) - rhs, cuCimag(lhs));
}


// multiply operator
__host__ __device__ __inline__ cuComplex operator*(const cuComplex &lhs, const cuComplex &rhs)
{
    return cuCmulf(lhs, rhs);
}
__host__ __device__ __inline__ cuComplex operator*(const cuComplex &lhs, const cuDoubleComplex &rhs)
{
    // basically copy of cuComplex.h cuCmulf
    cuComplex prod;
    prod = make_cuComplex   ((cuCrealf(lhs) * cuCreal(rhs)) -
                             (cuCimagf(lhs) * cuCimag(rhs)),
                             (cuCrealf(lhs) * cuCimag(rhs)) +
                             (cuCimagf(lhs) * cuCreal(rhs)));
    return prod;
}
__host__ __device__ __inline__ cuComplex operator*(const cuComplex &lhs, const float &rhs)
{
    return make_cuComplex(cuCrealf(lhs) * rhs, cuCimagf(lhs) * rhs);
}
__host__ __device__ __inline__ cuComplex operator*(const cuComplex &lhs, const double &rhs)
{
    return make_cuComplex(cuCrealf(lhs) * rhs, cuCimagf(lhs) * rhs);
}


__host__ __device__ __inline__ cuDoubleComplex operator*(const cuDoubleComplex &lhs, const cuComplex &rhs)
{
    cuDoubleComplex prod;
    prod = make_cuDoubleComplex ((cuCreal(lhs) * cuCrealf(rhs)) -
                                 (cuCimag(lhs) * cuCimagf(rhs)),
                                 (cuCreal(lhs) * cuCimagf(rhs)) +
                                 (cuCimag(lhs) * cuCrealf(rhs)));
    return prod;
}
__host__ __device__ __inline__ cuDoubleComplex operator*(const cuDoubleComplex &lhs, const cuDoubleComplex &rhs)
{
    return cuCmul(lhs, rhs);
}
__host__ __device__ __inline__ cuDoubleComplex operator*(const cuDoubleComplex &lhs, const float &rhs)
{
    return make_cuDoubleComplex(cuCreal(lhs) * rhs, cuCimag(lhs));
}
__host__ __device__ __inline__ cuDoubleComplex operator*(const cuDoubleComplex &lhs, const double &rhs)
{
    return make_cuDoubleComplex(cuCreal(lhs) * rhs, cuCimag(lhs));
}

// div operator
__host__ __device__ __inline__ cuComplex operator/(const cuComplex &lhs, const cuComplex &rhs)
{
    return cuCdivf(lhs, rhs);
}
__host__ __device__ __inline__ cuComplex operator/(const cuComplex &lhs, const cuDoubleComplex &rhs)
{
    return cuCdivf(lhs, make_cuComplex(cuCreal(rhs), cuCimag(rhs)));
}
__host__ __device__ __inline__ cuComplex operator/(const cuComplex &lhs, const float &rhs)
{
    return make_cuComplex(cuCrealf(lhs) / rhs, cuCimagf(lhs) / rhs);
}
__host__ __device__ __inline__ cuComplex operator/(const cuComplex &lhs, const double &rhs)
{
    return make_cuComplex(cuCrealf(lhs) / rhs, cuCimagf(lhs) / rhs);
}

__host__ __device__ __inline__ cuDoubleComplex operator/(const cuDoubleComplex &lhs, const cuComplex &rhs)
{
    return cuCdiv(lhs, make_cuDoubleComplex(cuCrealf(rhs), cuCimagf(rhs)));
}
__host__ __device__ __inline__ cuDoubleComplex operator/(const cuDoubleComplex &lhs, const cuDoubleComplex &rhs)
{
    return cuCdiv(lhs, rhs);
}
__host__ __device__ __inline__ cuDoubleComplex operator/(const cuDoubleComplex &lhs, const float &rhs)
{
    return make_cuDoubleComplex(cuCreal(lhs) / rhs, cuCimag(lhs) / rhs);
}
__host__ __device__ __inline__ cuDoubleComplex operator/(const cuDoubleComplex &lhs, const double &rhs)
{
    return make_cuDoubleComplex(cuCreal(lhs) / rhs, cuCimag(lhs) / rhs);
}
