// SPDX-FileCopyrightText: 2011 - 2024 NVIDIA CORPORATION. All Rights Reserved.
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
class CoarseAGenerator
{
        typedef T_Config TConfig;
        typedef typename T_Config::MatPrec ValueType;
        typedef typename T_Config::IndPrec IndexType;
        typedef typename T_Config::MemSpace MemorySpace;

    public:
        virtual void computeAOperator(const Matrix<T_Config> &A, Matrix<T_Config> &Ac, const typename Matrix<T_Config>::IVector &aggregates, const typename Matrix<T_Config>::IVector &R_row_offsets, const typename Matrix<T_Config>::IVector &R_column_indices, const int num_aggregates) = 0;

        virtual ~CoarseAGenerator() {}

    protected:
        void printNonzeroStats(const typename Matrix<T_Config>::IVector &Ac_row_offsets, const int num_aggregates);

};

template<class T_Config>
class CoarseAGeneratorFactory
{
    public:
        virtual CoarseAGenerator<T_Config> *create(AMG_Config &cfg, const std::string &cfg_scope) = 0;
        virtual ~CoarseAGeneratorFactory() {};

        /********************************************
         * Register a generator class with key "name"
         *******************************************/
        static void registerFactory(std::string name, CoarseAGeneratorFactory<T_Config> *f);

        /********************************************
         * Unregister a CoarseAGenerator class with key "name"
         *******************************************/
        static void unregisterFactory(std::string name);

        /********************************************
         * Unregister all the CoarseAGenerator classes
         *******************************************/
        static void unregisterFactories( );

        /*********************************************
        * Allocates generator based on cfg
        *********************************************/
        static CoarseAGenerator<T_Config> *allocate(AMG_Config &cfg, const std::string &cfg_scope );

        typedef typename std::map<std::string, CoarseAGeneratorFactory<T_Config>*>::const_iterator Iterator;

        static Iterator getIterator() { return getFactories().begin(); };
        static bool isIteratorLast(const Iterator &iter) { if ( iter == getFactories().end() ) return true; else return false; };

    private:
        static std::map<std::string, CoarseAGeneratorFactory<T_Config>*> &getFactories( );
};

}

} // namespace amgx


