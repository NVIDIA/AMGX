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
        typedef Vector<typename TConfig_h::template setVecPrec<(AMGX_VecPrecision)AMGX_MatPrec>::Type> Vector_h; // host vector with Matrix precision
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
