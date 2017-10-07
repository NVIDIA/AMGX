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

/* Truncation */
typedef enum
{
    AMGX_TruncateByRowSum,
    AMGX_TruncateByMaxCoefficient
} AMGX_TruncateType;

#include <matrix.h>
#include <specific_spmv.h>

namespace amgx
{

template <class T_Config>
struct Truncate
{
    static void truncateByFactor(Matrix<T_Config> &A, const double trunc_factor,
                                 const AMGX_TruncateType truncType = AMGX_TruncateByMaxCoefficient);
};

// host specialisation
template <AMGX_VecPrecision t_vecPrec, AMGX_MatPrecision t_matPrec, AMGX_IndPrecision t_indPrec>
struct Truncate<TemplateConfig<AMGX_host, t_vecPrec, t_matPrec, t_indPrec> >
{
    typedef TemplateConfig<AMGX_host, t_vecPrec, t_matPrec, t_indPrec> TConfig;
    DEFINE_VECTOR_TYPES
    typedef Matrix<TConfig_h> Matrix_h;

    static void truncateByFactor(Matrix_h &A, const double trunc_factor,
                                 const AMGX_TruncateType truncType = AMGX_TruncateByMaxCoefficient);

    static void truncateByMaxElements(Matrix_h &A, const int max_elmts = 4);
};

// device specialisation
template <AMGX_VecPrecision t_vecPrec, AMGX_MatPrecision t_matPrec, AMGX_IndPrecision t_indPrec>
struct Truncate<TemplateConfig<AMGX_device, t_vecPrec, t_matPrec, t_indPrec> >
{
    typedef TemplateConfig<AMGX_device, t_vecPrec, t_matPrec, t_indPrec> TConfig;
    DEFINE_VECTOR_TYPES
    typedef Matrix<TConfig_d> Matrix_d;

    static void truncateByFactor(Matrix_d &A, const double trunc_factor,
                                 const AMGX_TruncateType truncType = AMGX_TruncateByMaxCoefficient);

    static void truncateByMaxElements(Matrix_d &A, const int max_elmts = 4);
};

template <AMGX_MatPrecision t_matPrec, AMGX_IndPrecision t_indPrec>
struct Truncate<TemplateConfig<AMGX_device, AMGX_vecComplex, t_matPrec, t_indPrec> >
{
    typedef TemplateConfig<AMGX_device, AMGX_vecComplex, t_matPrec, t_indPrec> TConfig;
    DEFINE_VECTOR_TYPES
    typedef Matrix<TConfig_d> Matrix_d;

    static void truncateByFactor(Matrix_d &A, const double trunc_factor,
                                 const AMGX_TruncateType truncType = AMGX_TruncateByMaxCoefficient)
    {
        FatalError("This type of truncate for complex is not supported yet", AMGX_ERR_NOT_SUPPORTED_TARGET);
    }

    static void truncateByMaxElements(Matrix_d &A, const int max_elmts = 4)
    {
        FatalError("This type of truncate for complex is not supported yet", AMGX_ERR_NOT_SUPPORTED_TARGET);
    }
};

template <AMGX_MatPrecision t_matPrec, AMGX_IndPrecision t_indPrec>
struct Truncate<TemplateConfig<AMGX_device, AMGX_vecDoubleComplex, t_matPrec, t_indPrec> >
{
    typedef TemplateConfig<AMGX_device, AMGX_vecDoubleComplex, t_matPrec, t_indPrec> TConfig;
    DEFINE_VECTOR_TYPES
    typedef Matrix<TConfig_d> Matrix_d;

    static void truncateByFactor(Matrix_d &A, const double trunc_factor,
                                 const AMGX_TruncateType truncType = AMGX_TruncateByMaxCoefficient)
    {
        FatalError("This type of truncate for complex is not supported yet", AMGX_ERR_NOT_SUPPORTED_TARGET);
    }

    static void truncateByMaxElements(Matrix_d &A, const int max_elmts = 4)
    {
        FatalError("This type of truncate for complex is not supported yet", AMGX_ERR_NOT_SUPPORTED_TARGET);
    }
};

} // namespace amgx
