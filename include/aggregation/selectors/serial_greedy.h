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

template <class T_Config> class SerialGreedySelector;

template <class T_Config>
class SerialGreedySelector : public Selector<T_Config>
{
    public:
        typedef T_Config TConfig;
        typedef typename T_Config::MatPrec ValueType;
        typedef typename T_Config::IndPrec IndexType;
        typedef typename T_Config::MemSpace MemorySpace;
        typedef typename Matrix<T_Config>::IVector IVector;

        // Constructor
        SerialGreedySelector(AMG_Config &cfg, const std::string &cfg_scope);

        void setAggregates( Matrix<T_Config> &A,
                            IVector &aggregates, IVector &aggregates_global, int &num_aggregates);

    private:
        int aggregate_size;
        int edge_weight_component;

};

template<class T_Config>
class SerialGreedySelectorFactory : public SelectorFactory<T_Config>
{
    public:
        Selector<T_Config> *create(AMG_Config &cfg, const std::string &cfg_scope) { return new SerialGreedySelector<T_Config>(cfg, cfg_scope); }
};
}
}
