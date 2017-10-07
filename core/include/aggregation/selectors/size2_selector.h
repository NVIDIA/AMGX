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
