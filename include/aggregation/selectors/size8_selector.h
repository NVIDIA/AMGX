// SPDX-FileCopyrightText: 2011 - 2024 NVIDIA CORPORATION. All Rights Reserved.
//
// SPDX-License-Identifier: BSD-3-Clause

#pragma once
#include <aggregation/selectors/agg_selector.h>
#include <matrix.h>

namespace amgx
{
namespace aggregation
{
namespace size8_selector
{

template <class T_Config> class Size8Selector;

template <class T_Config>
class Size8SelectorBase: public Selector<T_Config>
{

    public:
        typedef T_Config TConfig;
        typedef typename T_Config::MatPrec ValueType;
        typedef typename T_Config::IndPrec IndexType;
        typedef typename Matrix<T_Config>::IVector IVector;


        Size8SelectorBase(AMG_Config &cfg, const std::string &cfg_scope);
        void setAggregates(Matrix<T_Config> &A,
                           IVector &aggregates, IVector &aggregates_global, int &num_aggregates);

    protected:
        int deterministic;
        int max_iterations;
        double numUnassigned_tol;
        int m_aggregation_edge_weight_component;
        int weight_formula;
        virtual void setAggregates_common_sqblock(const Matrix<T_Config> &A,
                IVector &aggregates, IVector &aggregates_global, int &num_aggregates) = 0;
};

// specialization for host
template <AMGX_VecPrecision t_vecPrec, AMGX_MatPrecision t_matPrec, AMGX_IndPrecision t_indPrec>
class Size8Selector<TemplateConfig<AMGX_host, t_vecPrec, t_matPrec, t_indPrec> > : public Size8SelectorBase<TemplateConfig<AMGX_host, t_vecPrec, t_matPrec, t_indPrec> >
{
    public:
        typedef TemplateConfig<AMGX_host, t_vecPrec, t_matPrec, t_indPrec> TConfig;
        typedef Matrix< TemplateConfig<AMGX_host, t_vecPrec, t_matPrec, t_indPrec> > Matrix_h;
        typedef Matrix< TemplateConfig<AMGX_device, t_vecPrec, t_matPrec, t_indPrec> > Matrix_d;
        typedef typename Matrix_h::value_type ValueType;
        typedef typename Matrix_h::index_type IndexType;
        typedef typename Matrix_h::IVector IVector;
        typedef typename Matrix<TConfig>::MVector VVector;
        typedef typename TConfig::template setMemSpace<AMGX_host>::Type TConfig_h;
        typedef Vector<typename TConfig_h::template setVecPrec<(AMGX_VecPrecision)AMGX_GET_MODE_VAL(AMGX_MatPrecision, TConfig_h::mode)>::Type> Vector_h; // host vector with Matrix precision

        Size8Selector(AMG_Config &cfg, const std::string &cfg_scope) : Size8SelectorBase<TConfig>(cfg, cfg_scope) {}
    private:
        void setAggregates_common_sqblock(const Matrix_h &A,
                                          IVector &aggregates, IVector &aggregates_global, int &num_aggregates);
};

// specialization for device
template <AMGX_VecPrecision t_vecPrec, AMGX_MatPrecision t_matPrec, AMGX_IndPrecision t_indPrec>
class Size8Selector<TemplateConfig<AMGX_device, t_vecPrec, t_matPrec, t_indPrec> >: public Size8SelectorBase<TemplateConfig<AMGX_device, t_vecPrec, t_matPrec, t_indPrec> >
{
    public:
        typedef TemplateConfig<AMGX_device, t_vecPrec, t_matPrec, t_indPrec> TConfig;
        typedef Matrix< TemplateConfig<AMGX_host, t_vecPrec, t_matPrec, t_indPrec> > Matrix_h;
        typedef Matrix< TemplateConfig<AMGX_device, t_vecPrec, t_matPrec, t_indPrec> > Matrix_d;
        typedef typename Matrix_d::value_type ValueType;
        typedef typename Matrix_d::index_type IndexType;
        typedef typename Matrix_d::IVector IVector;
        typedef typename Matrix<TConfig>::MVector VVector;
        typedef typename TConfig::template setMemSpace<AMGX_host>::Type TConfig_h;
        typedef Vector<typename TConfig_h::template setVecPrec<(AMGX_VecPrecision)AMGX_GET_MODE_VAL(AMGX_MatPrecision, TConfig_h::mode)>::Type> Vector_h; // host vector with Matrix precision
        Size8Selector(AMG_Config &cfg, const std::string &cfg_scope) : Size8SelectorBase<TConfig>(cfg, cfg_scope) {}
    public:
        void setAggregates_common_sqblock(const Matrix_d &A,
                                          IVector &aggregates, IVector &aggregates_global, int &num_aggregates);

        template<int AVG_NUM_COLS_PER_ROW>
        void setAggregates_common_sqblock_avg_specialized(const Matrix_d &A,
                IVector &aggregates, IVector &aggregates_global, int &num_aggregates);

#if AMGX_ENABLE_KERNEL_TESTING
        template<int AVG, class SELECTOR> void checkNewKernels(SELECTOR *s, const typename SELECTOR::Matrix_d &A);
#endif
};

template<class T_Config>
class Size8SelectorFactory : public SelectorFactory<T_Config>
{
    public:
        Selector<T_Config> *create(AMG_Config &cfg, const std::string &cfg_scope) { return new Size8Selector<T_Config>(cfg, cfg_scope); }
};
}
}
}
