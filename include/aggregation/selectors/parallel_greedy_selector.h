// SPDX-FileCopyrightText: 2011 - 2024 NVIDIA CORPORATION. All Rights Reserved.
//
// SPDX-License-Identifier: BSD-3-Clause

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
