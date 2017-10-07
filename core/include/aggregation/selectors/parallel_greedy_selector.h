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

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template< class T_Config >
class ParallelGreedySelector_Base: public Selector<T_Config>
{
        typedef Selector<T_Config> Base;

    public:
        typedef T_Config                     TConfig;
        typedef Matrix<TConfig>              MatrixType;
        typedef typename TConfig::MatPrec    ValueType;
        typedef typename TConfig::IndPrec    IndexType;
        typedef typename TConfig::MemSpace   MemorySpace;
        typedef typename MatrixType::IVector IVector;

        ParallelGreedySelector_Base( AMG_Config &cfg, const std::string &cfg_scope );

        void setAggregates( MatrixType &A, IVector &aggregates, IVector &aggregates_global, int &num_aggregates );

    protected:
        virtual void setAggregates_1x1( const MatrixType &A, IVector &aggregates, IVector &aggregates_global, int &num_aggregates ) = 0;
        virtual void setAggregates_common_sqblocks( const MatrixType &A, IVector &aggregates, IVector &aggregates_global, int &num_aggregates ) = 0;

        // The max size of aggregates (only a wish ;)).
        int m_max_size;
        // Below that threshold we make all nodes candidates.
        float m_candidates_threshold, m_changed_threshold;
        /*
        int max_iterations;
        int deterministic;
        ValueType numUnassigned_tol;
        bool two_phase; */
};

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template< class T_Config >
class ParallelGreedySelector;

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template< AMGX_VecPrecision V, AMGX_MatPrecision M, AMGX_IndPrecision I >
class ParallelGreedySelector<TemplateConfig<AMGX_host, V, M, I> > : public ParallelGreedySelector_Base<TemplateConfig<AMGX_host, V, M, I> >
{
        typedef ParallelGreedySelector_Base<TemplateConfig<AMGX_host, V, M, I> > Base;

    public:
        typedef typename Base::TConfig    TConfig;
        typedef typename Base::MatrixType MatrixType;
        typedef typename Base::ValueType  ValueType;
        typedef typename Base::IndexType  IndexType;
        typedef typename Base::IVector    IVector;

        ParallelGreedySelector( AMG_Config &cfg, const std::string &cfg_scope ) : Base( cfg, cfg_scope ) {}

    protected:
        void setAggregates_1x1( const MatrixType &A, IVector &aggregates, IVector &aggregates_global, int &num_aggregates );
        void setAggregates_common_sqblocks( const MatrixType &A, IVector &aggregates, IVector &aggregates_global, int &num_aggregates );
};

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template< AMGX_VecPrecision V, AMGX_MatPrecision M, AMGX_IndPrecision I >
class ParallelGreedySelector<TemplateConfig<AMGX_device, V, M, I> > : public ParallelGreedySelector_Base<TemplateConfig<AMGX_device, V, M, I> >
{
        typedef ParallelGreedySelector_Base<TemplateConfig<AMGX_device, V, M, I> > Base;

    public:
        typedef typename Base::TConfig    TConfig;
        typedef typename Base::MatrixType MatrixType;
        typedef typename Base::ValueType  ValueType;
        typedef typename Base::IndexType  IndexType;
        typedef typename Base::IVector    IVector;

        ParallelGreedySelector( AMG_Config &cfg, const std::string &cfg_scope ) : Base( cfg, cfg_scope ) {}

    protected:
        void setAggregates_1x1( const MatrixType &A, IVector &aggregates, IVector &aggregates_global, int &num_aggregates );
        void setAggregates_common_sqblocks( const MatrixType &A, IVector &aggregates, IVector &aggregates_global, int &num_aggregates );
};

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template< class T_Config >
class ParallelGreedySelectorFactory : public SelectorFactory<T_Config>
{
    public:
        Selector<T_Config> *create(AMG_Config &cfg, const std::string &cfg_scope)
        {
            return new ParallelGreedySelector<T_Config>( cfg, cfg_scope );
        }
};

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace aggregation
} // namespace amgx
