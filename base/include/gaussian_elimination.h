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

#define BSIZE 4
#define BSIZE_SQ 16

#include <matrix.h>
#include "amgx_types/math.h"

namespace amgx
{

// solves Ax = b via gaussian elimination
template<class TConfig>
struct GaussianElimination
{
    static void gaussianElimination(const Matrix<TConfig> &A, Vector<TConfig> &x, const Vector<TConfig> &b);
};

// host specialization
template <AMGX_VecPrecision t_vecPrec, AMGX_MatPrecision t_matPrec, AMGX_IndPrecision t_indPrec>
struct GaussianElimination<TemplateConfig<AMGX_host, t_vecPrec, t_matPrec, t_indPrec> >
{
        typedef TemplateConfig<AMGX_host, t_vecPrec, t_matPrec, t_indPrec> TConfig_h;
        typedef Matrix<TConfig_h> Matrix_h;
        typedef Vector<TConfig_h> Vector_h;
        static void gaussianElimination(const Matrix_h &A, Vector_h &x, const Vector_h &b);
    private:
        static void gaussianElimination_4x4_host(const Matrix_h &A, Vector_h &x, const Vector_h &b);
};

// device specialization
template <AMGX_VecPrecision t_vecPrec, AMGX_MatPrecision t_matPrec, AMGX_IndPrecision t_indPrec>
struct GaussianElimination<TemplateConfig<AMGX_device, t_vecPrec, t_matPrec, t_indPrec> >
{
        typedef TemplateConfig<AMGX_device, t_vecPrec, t_matPrec, t_indPrec> TConfig_d;
        typedef Matrix<TConfig_d> Matrix_d;
        typedef Vector<TConfig_d> Vector_d;
        static void gaussianElimination(const Matrix_d &A, Vector_d &x, const Vector_d &b);
    private:
        static void gaussianElimination_4x4_device(const Matrix_d &A, Vector_d &x, const Vector_d &b);
};

// solves Ax = b via gaussian elimination
template<typename IndexType, typename ValueTypeA, typename ValueTypeB>
void gaussianEliminationRowMajor(ValueTypeA **e, ValueTypeB *x, ValueTypeB *b, const IndexType bsize);

template <typename ValueType>
__host__ __device__ inline void gaussianElimination4by4(ValueType (&E)[BSIZE][BSIZE], ValueType(&x)[BSIZE], ValueType(&b)[BSIZE] )
{
    ValueType pivot, ratio, tmp;
    // j=0
    pivot = E[0][0];
    // k=1
    ratio = E[1][0] / pivot;
    b[1] = b[1] - b[0] * ratio;
    E[1][1] = E[1][1] - E[0][1] * ratio;
    E[1][2] = E[1][2] - E[0][2] * ratio;
    E[1][3] = E[1][3] - E[0][3] * ratio;
    // k=2
    ratio = E[2][0] / pivot;
    b[2] = b[2] - b[0] * ratio;
    E[2][1] = E[2][1] - E[0][1] * ratio;
    E[2][2] = E[2][2] - E[0][2] * ratio;
    E[2][3] = E[2][3] - E[0][3] * ratio;
    //k=3
    ratio = E[3][0] / pivot;
    b[3] = b[3] - b[0] * ratio;
    E[3][1] = E[3][1] - E[0][1] * ratio;
    E[3][2] = E[3][1] - E[0][2] * ratio;
    E[3][3] = E[3][1] - E[0][3] * ratio;
    // j=1
    pivot = E[1][1];
    // k=2
    ratio = E[2][1] / pivot;
    b[2] = b[2] - b[1] * ratio;
    E[2][2] = E[2][2] - E[1][2] * ratio;
    E[2][3] = E[2][3] - E[1][3] * ratio;
    // k=3
    ratio = E[3][1] / pivot;
    b[3] = b[3] - b[1] * ratio;
    E[3][2] = E[3][2] - E[1][2] * ratio;
    E[3][3] = E[3][3] - E[1][3] * ratio;
    // j=2
    pivot = E[2][2];
    // k=3
    ratio = E[3][2] / pivot;
    b[3] = b[3] - b[2] * ratio;
    E[3][3] = E[3][3] - E[2][3] * ratio;
    // back substitution
    // j=3
    x[3] = b[3] / E[3][3];
    // j=2
    tmp = E[2][3] * x[3];
    x[2] = (b[2] - tmp) / E[2][2];
    // j=1
    tmp = E[1][2] * x[2] + E[1][3] * x[3];
    x[1] = (b[1] - tmp) / E[1][1];
    // j=0
    tmp = E[0][1] * x[1] + E[0][2] * x[2] + E[0][3] * x[3];
    x[0] = (b[0] - tmp) / E[0][0];
}

} // namespace amgx
