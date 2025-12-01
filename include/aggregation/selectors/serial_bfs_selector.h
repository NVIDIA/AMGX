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

template <class T_Config> class Serial_BFS_Selector;

template <class T_Config>
class Serial_BFS_Selector : public Selector<T_Config>
{
    public:
        typedef T_Config TConfig;
        typedef typename T_Config::MatPrec ValueType;
        typedef typename T_Config::IndPrec IndexType;
        typedef typename T_Config::MemSpace MemorySpace;
        typedef typename Matrix<T_Config>::IVector IVector;

        // Constructor
        Serial_BFS_Selector(AMG_Config &cfg, const std::string &cfg_scope);

        void setAggregates( Matrix<T_Config> &A,
                            IVector &aggregates, IVector &aggregates_global, int &num_aggregates);

    private:
        int aggregate_size;
        AMG_Config coloring_cfg;
        std::string coloring_cfg_scope;
};

template<class T_Config>
class Serial_BFS_SelectorFactory : public SelectorFactory<T_Config>
{
    public:
        Selector<T_Config> *create(AMG_Config &cfg, const std::string &cfg_scope) { return new Serial_BFS_Selector<T_Config>(cfg, cfg_scope); }
};
}
}
