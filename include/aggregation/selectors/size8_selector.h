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
        typedef Vector<typename TConfig_h::template setVecPrec<(AMGX_VecPrecision)AMGX_MatPrec>::Type> Vector_h; // host vector with Matrix precision

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
        typedef Vector<typename TConfig_h::template setVecPrec<(AMGX_VecPrecision)AMGX_MatPrec>::Type> Vector_h; // host vector with Matrix precision
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
