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
#include <aggregation/selectors/agg_selector.h>
#include <solvers/solver.h>

namespace amgx
{
namespace aggregation
{
namespace adaptive
{

template <class T_Config> class AdaptiveSelector;

template <class T_Config>
class AdaptiveSelectorBase: public Selector<T_Config>
{

    public:
        typedef T_Config TConfig;
        typedef typename T_Config::VecPrec ValueType;
        typedef typename T_Config::IndPrec IndexType;
        typedef typename T_Config::MemSpace MemorySpace;
        typedef typename Matrix<TConfig>::IVector IVector;
        typedef typename Matrix<TConfig>::VVector VVector;
        typedef typename Matrix<TConfig>::MVector MVector;

        //typedefs for the weight matrix and vectors
        typedef TemplateMode<AMGX_mode_dFFI>::Type TConfig_dFFI;//possible issue: the indextype does not match the actual indextype
        typedef typename TConfig_dFFI::template setVecPrec<AMGX_vecInt>::Type ivec_value_type_dFFI;
        typedef Vector<ivec_value_type_dFFI> IVector_dFFI;
        typedef Vector<TConfig_dFFI> VVector_dFFI;


        AdaptiveSelectorBase(AMG_Config &cfg, const std::string &cfg_scope);
        void setAggregates( Matrix<T_Config> &A,
                            IVector &aggregates, IVector &aggregates_global, int &num_aggregates);

    protected:
        Solver<TConfig> *smoother;
};

// specialization for host
template <AMGX_VecPrecision t_vecPrec, AMGX_MatPrecision t_matPrec, AMGX_IndPrecision t_indPrec>
class AdaptiveSelector<TemplateConfig<AMGX_host, t_vecPrec, t_matPrec, t_indPrec> > : public AdaptiveSelectorBase< TemplateConfig<AMGX_host, t_vecPrec, t_matPrec, t_indPrec> >
{
    public:
        typedef TemplateConfig<AMGX_host, t_vecPrec, t_matPrec, t_indPrec> TConfig;
        typedef Matrix<TConfig> Matrix_h;
        typedef typename TConfig::VecPrec ValueType;
        typedef typename TConfig::IndPrec IndexType;
        typedef typename Matrix_h::IVector IVector;
        typedef typename Matrix_h::VVector VVector;
        typedef typename Matrix_h::MVector MVector;

        //typedefs for the weight matrix and vectors
        typedef TemplateMode<AMGX_mode_dFFI>::Type TConfig_dFFI;//possible issue: the indextype does not match the actual indextype
        typedef typename TConfig_dFFI::template setVecPrec<AMGX_vecInt>::Type ivec_value_type_dFFI;
        typedef Vector<ivec_value_type_dFFI> IVector_dFFI;
        typedef Vector<TConfig_dFFI> VVector_dFFI;

        AdaptiveSelector(AMG_Config &cfg, const std::string &cfg_scope) : AdaptiveSelectorBase<TConfig>(cfg, cfg_scope) {}
        void setAggregates_common_sqblocks(const Matrix_h &A,
                                           IVector &aggregates, IVector &aggregates_global, int &num_aggregates, MVector &edge_weights);
};

// specialization for device
template <AMGX_VecPrecision t_vecPrec, AMGX_MatPrecision t_matPrec, AMGX_IndPrecision t_indPrec>
class AdaptiveSelector<TemplateConfig<AMGX_device, t_vecPrec, t_matPrec, t_indPrec> >: public AdaptiveSelectorBase< TemplateConfig<AMGX_device, t_vecPrec, t_matPrec, t_indPrec> >
{
    public:
        typedef TemplateConfig<AMGX_device, t_vecPrec, t_matPrec, t_indPrec> TConfig;
        typedef Matrix<TemplateConfig<AMGX_host, t_vecPrec, t_matPrec, t_indPrec> > Matrix_h;
        typedef Matrix<TConfig> Matrix_d;
        typedef typename TConfig::MatPrec ValueType;
        typedef typename TConfig::IndPrec IndexType;
        typedef typename Matrix_d::IVector IVector;
        typedef typename Matrix_d::VVector VVector;
        typedef typename Matrix_d::MVector MVector;

        AdaptiveSelector(AMG_Config &cfg, const std::string &cfg_scope) : AdaptiveSelectorBase<TConfig>(cfg, cfg_scope) {}

        void setAggregates_common_sqblocks(const Matrix_d &A,
                                           IVector &aggregates, IVector &aggregates_global, int &num_aggregates, MVector &edge_weights);

};

template<class T_Config>
class AdaptiveSelectorFactory : public SelectorFactory<T_Config>
{
    public:
        Selector<T_Config> *create(AMG_Config &cfg, const std::string &cfg_scope) { return new AdaptiveSelector<T_Config>(cfg, cfg_scope); }
};
}
}
}
