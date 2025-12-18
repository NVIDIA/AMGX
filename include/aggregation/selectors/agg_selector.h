// SPDX-FileCopyrightText: 2011 - 2025 NVIDIA CORPORATION. All Rights Reserved.
//
// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#include <getvalue.h>
#include <error.h>
#include <basic_types.h>
#include <amg_config.h>
#include <matrix.h>

namespace amgx
{

namespace aggregation
{

template <class T_Config>
class Selector
{
        typedef T_Config TConfig;
        typedef typename T_Config::MatPrec ValueType;
        typedef typename T_Config::IndPrec IndexType;
        typedef typename T_Config::MemSpace MemorySpace;
        typedef typename Matrix<T_Config>::IVector IVector;
        typedef typename Matrix<T_Config>::MVector MVector;

    public:
        virtual void setAggregates(Matrix<T_Config> &A,
                                   IVector &aggregates, IVector &aggregates_global, int &num_aggregates) = 0;
        void printAggregationInfo(const IVector &aggregates, const IVector &aggregates_global, const IndexType num_aggregates) const;

        virtual ~Selector() {}

        void renumberAndCountAggregates(IVector &aggregates, IVector &aggregates_global, const IndexType num_block_rows, IndexType &num_aggregates);
        void assertAggregates( const IVector &aggregates, int numAggregates );
        void assertRestriction( const IVector &R_row_offsets, const IVector &R_col_indices, const IVector &aggregates );
};

template<class T_Config>
class SelectorFactory
{
    public:
        virtual Selector<T_Config> *create(AMG_Config &cfg, const std::string &cfg_scope) = 0;
        virtual ~SelectorFactory() {};

        /********************************************
         * Register a selector class with key "name"
         *******************************************/
        static void registerFactory(std::string name, SelectorFactory<T_Config> *f);

        /********************************************
         * Unregister a selector class with key "name"
         *******************************************/
        static void unregisterFactory(std::string name);

        /********************************************
         * Unregister all the selector classes
         *******************************************/
        static void unregisterFactories( );

        /*********************************************
        * Allocates selector based on cfg
        *********************************************/
        static Selector<T_Config> *allocate(AMG_Config &cfg, const std::string &current_scope);

        typedef typename std::map<std::string, SelectorFactory<T_Config>*>::const_iterator Iterator;

        static Iterator getIterator() { return getFactories().begin(); };
        static bool isIteratorLast(const Iterator &iter) { if ( iter == getFactories().end() ) return true; else return false; };

    private:
        static std::map<std::string, SelectorFactory<T_Config>*> &getFactories( );
};

}

} // namespace amgx


