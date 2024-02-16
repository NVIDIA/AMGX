// SPDX-FileCopyrightText: 2011 - 2024 NVIDIA CORPORATION. All Rights Reserved.
//
// SPDX-License-Identifier: BSD-3-Clause

#pragma once
namespace amgx
{
template<class TConfig> class Strength;
}

#include <getvalue.h>
#include <error.h>
#include <basic_types.h>
#include <amg_config.h>
#include <matrix.h>

namespace amgx
{

template<class TConfig>
class Strength
{
        typedef typename TConfig::MatPrec ValueType;
        typedef typename TConfig::IndPrec IndexType;
        typedef Vector<typename TConfig::template setVecPrec<AMGX_vecFloat>::Type> FVector;
        typedef Vector<typename TConfig::template setVecPrec<AMGX_vecBool>::Type> BVector;
    public:
        virtual void computeStrongConnectionsAndWeights(Matrix<TConfig> &A,
                BVector &s_con,
                FVector &weights,
                const double max_row_sum) = 0;
        virtual ~Strength() {}
};

template<class TConfig>
class StrengthFactory
{
    public:
        virtual Strength<TConfig> *create(AMG_Config &cfg, const std::string &cfg_scope) = 0;
        virtual ~StrengthFactory() {};

        /********************************************
         * Register a strength class with key "name"
         *******************************************/
        static void registerFactory(std::string name, StrengthFactory<TConfig> *f);

        /********************************************
          * Unregister a strength class with key "name"
          *******************************************/
        static void unregisterFactory(std::string name);

        /********************************************
         * Unregister all the strength classes
         *******************************************/
        static void unregisterFactories( );

        /*********************************************
        * Allocates strength based on cfg
        *********************************************/
        static Strength<TConfig> *allocate(AMG_Config &cfg, const std::string &cfg_scope);

        typedef typename std::map<std::string, StrengthFactory<TConfig>*>::const_iterator Iterator;

        static Iterator getIterator() { return getFactories().begin(); };
        static bool isIteratorLast(const Iterator &iter) { if ( iter == getFactories().end() ) return true; else return false; };

    private:
        static std::map<std::string, StrengthFactory<TConfig>*> &getFactories( );
};
} // namespace amgx
