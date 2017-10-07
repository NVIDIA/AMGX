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
