// SPDX-FileCopyrightText: 2011 - 2025 NVIDIA CORPORATION. All Rights Reserved.
//
// SPDX-License-Identifier: BSD-3-Clause

#pragma once
#include <aggregation/selectors/agg_selector.h>
#include <matrix.h>

namespace amgx
{
namespace aggregation
{
namespace size4_selector
{

template <class T_Config> class Size4Selector;

template <class T_Config>
class Size4SelectorBase: public Selector<T_Config>
{

    public:
        typedef T_Config TConfig;
        typedef typename T_Config::MatPrec ValueType;
        typedef typename T_Config::IndPrec IndexType;
        typedef typename Matrix<T_Config>::IVector IVector;

        Size4SelectorBase(AMG_Config &cfg, const std::string &cfg_scope);
        void setAggregates( Matrix<T_Config> &A,
                            IVector &aggregates, IVector &aggregates_global, int &num_aggregates);

    protected:
        int deterministic;
        int max_iterations;
        ValueType numUnassigned_tol;
        int m_aggregation_edge_weight_component;
        int weight_formula;
        virtual void setAggregates_common_sqblock(const Matrix<T_Config> &A,
                IVector &aggregates, IVector &aggregates_global, int &num_aggregates) = 0;
};

// specialization for host
template <AMGX_VecPrecision t_vecPrec, AMGX_MatPrecision t_matPrec, AMGX_IndPrecision t_indPrec>
class Size4Selector<TemplateConfig<AMGX_host, t_vecPrec, t_matPrec, t_indPrec> > : public Size4SelectorBase<TemplateConfig<AMGX_host, t_vecPrec, t_matPrec, t_indPrec> >
{
    public:
        typedef TemplateConfig<AMGX_host, t_vecPrec, t_matPrec, t_indPrec> TConfig;
        typedef Matrix< TemplateConfig<AMGX_host, t_vecPrec, t_matPrec, t_indPrec> > Matrix_h;
        typedef Matrix< TemplateConfig<AMGX_device, t_vecPrec, t_matPrec, t_indPrec> > Matrix_d;
        typedef typename Matrix_h::value_type ValueType;
        typedef typename Matrix_h::index_type IndexType;
        typedef typename Matrix_h::IVector IVector;

        Size4Selector(AMG_Config &cfg, const std::string &cfg_scope) : Size4SelectorBase<TConfig>(cfg, cfg_scope) {}
    private:
        void setAggregates_common_sqblock(const Matrix_h &A,
                                          IVector &aggregates, IVector &aggregates_global, int &num_aggregates);
};

// specialization for device
template <AMGX_VecPrecision t_vecPrec, AMGX_MatPrecision t_matPrec, AMGX_IndPrecision t_indPrec>
class Size4Selector<TemplateConfig<AMGX_device, t_vecPrec, t_matPrec, t_indPrec> >: public Size4SelectorBase<TemplateConfig<AMGX_device, t_vecPrec, t_matPrec, t_indPrec> >
{
    public:
        typedef TemplateConfig<AMGX_device, t_vecPrec, t_matPrec, t_indPrec> TConfig;
        typedef Matrix< TemplateConfig<AMGX_host, t_vecPrec, t_matPrec, t_indPrec> > Matrix_h;
        typedef Matrix< TemplateConfig<AMGX_device, t_vecPrec, t_matPrec, t_indPrec> > Matrix_d;
        typedef typename Matrix_d::value_type ValueType;
        typedef typename Matrix_d::index_type IndexType;
        typedef typename Matrix_d::IVector IVector;
        Size4Selector(AMG_Config &cfg, const std::string &cfg_scope) : Size4SelectorBase<TConfig>(cfg, cfg_scope) {}
    private:
        void setAggregates_common_sqblock(const Matrix_d &A,
                                          IVector &aggregates, IVector &aggregates_global, int &num_aggregates);
};

template<class T_Config>
class Size4SelectorFactory : public SelectorFactory<T_Config>
{
    public:
        Selector<T_Config> *create(AMG_Config &cfg, const std::string &cfg_scope) { return new Size4Selector<T_Config>(cfg, cfg_scope); }
};
}
}
}
