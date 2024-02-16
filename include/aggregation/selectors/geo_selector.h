// SPDX-FileCopyrightText: 2011 - 2024 NVIDIA CORPORATION. All Rights Reserved.
//
// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#include <string>
#include<aggregation/selectors/agg_selector.h>
#include <basic_types.h>

namespace amgx
{
namespace aggregation
{

template <class T_Config> class GEO_Selector;

template <class T_Config>
class GEO_SelectorBase: public Selector<T_Config>
{
    public:
        typedef T_Config TConfig;
        typedef typename T_Config::MatPrec ValueType;
        typedef typename T_Config::VecPrec VectorType;
        typedef typename T_Config::IndPrec IndexType;
        typedef typename T_Config::MemSpace MemorySpace;
        typedef typename Matrix<T_Config>::IVector IVector;
        typedef typename Matrix<T_Config>::MVector VVector;
        typedef typename TConfig::template setMemSpace<AMGX_host>::Type TConfig_h;
        typedef Vector<typename TConfig_h::template setVecPrec<(AMGX_VecPrecision)AMGX_GET_MODE_VAL(AMGX_MatPrecision, TConfig_h::mode)>::Type> Vector_h; // host vector with Matrix precision
        typedef typename Vector_h::value_type value_type;

        GEO_SelectorBase(AMG_Config &cfg, const std::string &cfg_scope);
        void setAggregates( Matrix<T_Config> &A,
                            IVector &aggregates, IVector &aggregates_global, int &num_aggregates);

        void interpolateGeoinfo( Matrix<T_Config> &A );

    protected:
        ValueType coef_factor;
        IndexType size_tol;
        int num_origonal;
        int dimension;
        int geo_size;
        ValueType xmax, xmin, ymax, ymin, zmax, zmin;
        VVector *cord_x, *cord_y, *cord_z; // original geo info
        VVector ngeo_x, ngeo_y, ngeo_z; // aggregated from fine levels
        IVector idx_1d;
        virtual void setAggregates_1x1( Matrix<T_Config> &A,
                                        IVector &aggregates, IVector &aggregates_global, int &num_aggregates) = 0;
        virtual void setAggregates_common_sqblocks(const Matrix<T_Config> &A,
                IVector &aggregates, IVector &aggregates_global, int &num_aggregates) = 0;

};

// specialization for host
template <AMGX_VecPrecision t_vecPrec, AMGX_MatPrecision t_matPrec, AMGX_IndPrecision t_indPrec>
class GEO_Selector<TemplateConfig<AMGX_host, t_vecPrec, t_matPrec, t_indPrec> > : public GEO_SelectorBase<TemplateConfig<AMGX_host, t_vecPrec, t_matPrec, t_indPrec> >
{
    public:
        typedef TemplateConfig<AMGX_host, t_vecPrec, t_matPrec, t_indPrec> TConfig;
        typedef TemplateConfig<AMGX_host, t_vecPrec, t_matPrec, t_indPrec> TConfig_h;
        typedef Matrix<TConfig> Matrix_h;
        typedef typename Matrix<TConfig>::MVector VVector;
        typedef typename Matrix<TConfig_h>::MVector VVector_h;
        typedef typename Matrix_h::MVector MVector_h;
        typedef typename TConfig::MatPrec ValueType;
        typedef typename TConfig::IndPrec IndexType;
        typedef typename TConfig::VecPrec VectorType;
        typedef typename Matrix_h::IVector IVector;
        typedef typename Matrix<TConfig_h>::IVector IVector_h;
        GEO_Selector(AMG_Config &cfg, const std::string &cfg_scope) : GEO_SelectorBase<TConfig>(cfg, cfg_scope) {}
    private:

        void setAggregates_1x1( Matrix_h &A,
                                IVector &aggregates,  IVector &aggregates_global, int &num_aggregates);
        void setAggregates_common_sqblocks(const Matrix_h &A,
                                           IVector &aggregates,  IVector &aggregates_global, int &num_aggregates);
};

// specialization for device
template <AMGX_VecPrecision t_vecPrec, AMGX_MatPrecision t_matPrec, AMGX_IndPrecision t_indPrec>
class GEO_Selector<TemplateConfig<AMGX_device, t_vecPrec, t_matPrec, t_indPrec> > : public GEO_SelectorBase<TemplateConfig<AMGX_device, t_vecPrec, t_matPrec, t_indPrec> >
{
    public:
        typedef TemplateConfig<AMGX_device, t_vecPrec, t_matPrec, t_indPrec> TConfig;
        typedef TemplateConfig<AMGX_host, t_vecPrec, t_matPrec, t_indPrec> TConfig_h;
        typedef Matrix<TConfig> Matrix_d;
        typedef typename Matrix<TConfig_h>::MVector MVector_h;
        typedef typename Matrix<TConfig>::MVector VVector;
        typedef typename Matrix<TConfig_h>::MVector VVector_h;
        typedef typename TConfig::MatPrec ValueType;
        typedef typename TConfig::IndPrec IndexType;
        typedef typename TConfig::VecPrec VectorType;
        typedef typename Matrix_d::IVector IVector;
        typedef typename Matrix<TConfig_h>::IVector IVector_h;


        GEO_Selector(AMG_Config &cfg, const std::string &cfg_scope) : GEO_SelectorBase<TConfig>(cfg, cfg_scope) {}
    private:

        void setAggregates_1x1( Matrix_d &A,
                                IVector &aggregates, IVector &aggregates_global, int &num_aggregates);
        void setAggregates_common_sqblocks(const Matrix_d &A,
                                           IVector &aggregates, IVector &aggregates_global, int &num_aggregates);
};

template<class T_Config>
class GEO_SelectorFactory : public SelectorFactory<T_Config>
{
    public:
        Selector<T_Config> *create(AMG_Config &cfg, const std::string &cfg_scope) { return new GEO_Selector<T_Config>(cfg, cfg_scope); }
};
}
}
