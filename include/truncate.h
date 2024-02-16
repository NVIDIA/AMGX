// SPDX-FileCopyrightText: 2011 - 2024 NVIDIA CORPORATION. All Rights Reserved.
//
// SPDX-License-Identifier: BSD-3-Clause

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
