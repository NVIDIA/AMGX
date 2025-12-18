// SPDX-FileCopyrightText: 2011 - 2025 NVIDIA CORPORATION. All Rights Reserved.
//
// SPDX-License-Identifier: BSD-3-Clause

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
