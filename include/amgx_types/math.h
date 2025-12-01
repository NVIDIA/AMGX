// SPDX-FileCopyrightText: 2011 - 2025 NVIDIA CORPORATION. All Rights Reserved.
//
// SPDX-License-Identifier: BSD-3-Clause

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
