// SPDX-FileCopyrightText: 2011 - 2024 NVIDIA CORPORATION. All Rights Reserved.
//
// SPDX-License-Identifier: BSD-3-Clause

#pragma once
#include <aggregation/selectors/agg_selector.h>

namespace amgx
{
namespace aggregation
{
namespace size2_selector
{

template <class T_Config> class Size2Selector;

template <class T_Config>
class Size2SelectorBase: public Selector<T_Config>
{

    public:
        typedef T_Config TConfig;
        typedef typename T_Config::MatPrec ValueType;
        typedef typename T_Config::IndPrec IndexType;
        typedef typename T_Config::MemSpace MemorySpace;
        typedef typename Matrix<T_Config>::IVector IVector;

        Size2SelectorBase(AMG_Config &cfg, const std::string &cfg_scope);
        void setAggregates( Matrix<T_Config> &A,
                            IVector &aggregates, IVector &aggregates_global, int &num_aggregates);

    protected:
//    virtual void setAggregates_1x1(const Matrix<T_Config> &A,
//                     IVector &aggregates, IVector &aggregates_global, int &num_aggregates)=0;
        virtual void setAggregates_common_sqblocks(const Matrix<T_Config> &A,
                IVector &aggregates, IVector &aggregates_global, int &num_aggregates) = 0;
        int max_iterations;
        int deterministic;
        ValueType numUnassigned_tol;
        bool two_phase;
        int m_aggregation_edge_weight_component;
        bool merge_singletons;
        int weight_formula;
};

// specialization for host
template <AMGX_VecPrecision t_vecPrec, AMGX_MatPrecision t_matPrec, AMGX_IndPrecision t_indPrec>
class Size2Selector<TemplateConfig<AMGX_host, t_vecPrec, t_matPrec, t_indPrec> > : public Size2SelectorBase< TemplateConfig<AMGX_host, t_vecPrec, t_matPrec, t_indPrec> >
{
    public:
        typedef TemplateConfig<AMGX_host, t_vecPrec, t_matPrec, t_indPrec> TConfig;
        typedef Matrix<TConfig> Matrix_h;
        typedef typename TConfig::MatPrec ValueType;
        typedef typename TConfig::IndPrec IndexType;
        typedef typename Matrix_h::IVector IVector;

        Size2Selector(AMG_Config &cfg, const std::string &cfg_scope) : Size2SelectorBase<TConfig>(cfg, cfg_scope) {}
    private:

        void setAggregates_1x1(const Matrix_h &A,
                               IVector &aggregates, IVector &aggregates_global, int &num_aggregates);
        void setAggregates_common_sqblocks(const Matrix_h &A,
                                           IVector &aggregates, IVector &aggregates_global, int &num_aggregates);
};

// specialization for device
template <AMGX_VecPrecision t_vecPrec, AMGX_MatPrecision t_matPrec, AMGX_IndPrecision t_indPrec>
class Size2Selector<TemplateConfig<AMGX_device, t_vecPrec, t_matPrec, t_indPrec> >: public Size2SelectorBase< TemplateConfig<AMGX_device, t_vecPrec, t_matPrec, t_indPrec> >
{
    public:
        typedef TemplateConfig<AMGX_device, t_vecPrec, t_matPrec, t_indPrec> TConfig;
        typedef Matrix<TemplateConfig<AMGX_host, t_vecPrec, t_matPrec, t_indPrec> > Matrix_h;
        typedef Matrix<TConfig> Matrix_d;
        typedef typename TConfig::MatPrec ValueType;
        typedef typename TConfig::IndPrec IndexType;
        typedef typename Matrix_d::IVector IVector;

        Size2Selector(AMG_Config &cfg, const std::string &cfg_scope) : Size2SelectorBase<TConfig>(cfg, cfg_scope) {}
    private:
        void setAggregates_1x1(const Matrix_d &A,
                               IVector &aggregates, IVector &aggregates_global, int &num_aggregates);
        void setAggregates_common_sqblocks(const Matrix_d &A,
                                           IVector &aggregates, IVector &aggregates_global, int &num_aggregates);
};

template<class T_Config>
class Size2SelectorFactory : public SelectorFactory<T_Config>
{
    public:
        Selector<T_Config> *create(AMG_Config &cfg, const std::string &cfg_scope) { return new Size2Selector<T_Config>(cfg, cfg_scope); }
};
}
}
}
