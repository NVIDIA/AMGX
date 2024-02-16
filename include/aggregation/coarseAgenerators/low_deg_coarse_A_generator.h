// SPDX-FileCopyrightText: 2011 - 2024 NVIDIA CORPORATION. All Rights Reserved.
//
// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#include <aggregation/coarseAgenerators/coarse_A_generator.h>
#include <matrix.h>

namespace amgx
{
namespace aggregation
{

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template< class T_Config >
class LowDegCoarseAGenerator
{};

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template< AMGX_VecPrecision V, AMGX_MatPrecision M, AMGX_IndPrecision I >
class LowDegCoarseAGenerator<TemplateConfig<AMGX_device, V, M, I> > : public CoarseAGenerator<TemplateConfig<AMGX_device, V, M, I> >
{
    public:
        typedef TemplateConfig<AMGX_device, V, M, I> TConfig_d;
        typedef TemplateConfig<AMGX_host,   V, M, I> TConfig_h;
        typedef TemplateConfig<AMGX_device, V, M, I> TConfig;

        typedef Matrix<TConfig_d> Matrix_d;
        typedef Matrix<TConfig_h> Matrix_h;

        typedef typename Matrix_d::value_type ValueType;
        typedef typename Matrix_d::index_type IndexType;
        typedef typename Matrix_d::IVector    IVector;

    public:
        LowDegCoarseAGenerator(AMG_Config &cfg, const std::string &cfg_scope) : CoarseAGenerator<TConfig>()
        {
            m_force_determinism = cfg.getParameter<int>("determinism_flag", cfg_scope) == 2;
        }

        void computeAOperator( const Matrix<TConfig> &A,
                               Matrix<TConfig> &Ac,
                               const IVector &aggregates,
                               const IVector &R_row_offsets,
                               const IVector &R_column_indices,
                               const int num_aggregates );

    private:
        bool m_force_determinism;
};

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template< AMGX_VecPrecision V, AMGX_MatPrecision M, AMGX_IndPrecision I >
class LowDegCoarseAGenerator<TemplateConfig<AMGX_host, V, M, I> > : public CoarseAGenerator<TemplateConfig<AMGX_host, V, M, I> >
{
    public:
        typedef TemplateConfig<AMGX_device, V, M, I> TConfig_d;
        typedef TemplateConfig<AMGX_host,   V, M, I> TConfig_h;
        typedef TemplateConfig<AMGX_host,   V, M, I> TConfig;

        typedef Matrix<TConfig_d> Matrix_d;
        typedef Matrix<TConfig_h> Matrix_h;

        typedef typename Matrix_h::value_type ValueType;
        typedef typename Matrix_h::index_type IndexType;
        typedef typename Matrix_h::IVector    IVector;

    public:
        LowDegCoarseAGenerator(AMG_Config &, const std::string &) : CoarseAGenerator<TConfig>()
        {}

        void computeAOperator( const Matrix<TConfig> &A,
                               Matrix<TConfig> &Ac,
                               const IVector &aggregates,
                               const IVector &R_row_offsets,
                               const IVector &R_column_indices,
                               const int num_aggregates );
};

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template< class T_Config >
class LowDegCoarseAGeneratorFactory : public CoarseAGeneratorFactory<T_Config>
{
    public:
        CoarseAGenerator<T_Config> *create(AMG_Config &cfg, const std::string &cfg_scope)
        {
            return new LowDegCoarseAGenerator<T_Config>(cfg, cfg_scope);
        }
};

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace aggregation
} // namespace amgx
