// SPDX-FileCopyrightText: 2011 - 2024 NVIDIA CORPORATION. All Rights Reserved.
//
// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#include <getvalue.h>
#include <error.h>
#include <basic_types.h>
#include <amg_config.h>
#include <matrix.h>
#include <amg.h>

namespace amgx
{

template <class TConfig>
class Interpolator
{
        typedef typename TConfig::MatPrec ValueType;
        typedef typename TConfig::IndPrec IndexType;
        typedef Vector<typename TConfig::template setVecPrec<AMGX_vecInt>::Type> IVector;
        typedef Vector<typename TConfig::template setVecPrec<AMGX_vecBool>::Type> BVector;

    public:
        virtual void generateInterpolationMatrix(Matrix<TConfig> &A,
                IVector &cf_map,
                BVector &s_con,
                IVector &scratch,
                Matrix<TConfig> &P) = 0;
        virtual ~Interpolator() {}
};

template<class TConfig>
class InterpolatorFactory
{
    public:
        virtual Interpolator<TConfig> *create(AMG_Config &cfg, const std::string &cfg_scope) = 0;
        virtual ~InterpolatorFactory() {};

        /***********************************************
         * Register a interpolator class with key "name"
         **********************************************/
        static void registerFactory(std::string name, InterpolatorFactory<TConfig> *f);

        /********************************************
         * Unregister a interpolator class with key "name"
         *******************************************/
        static void unregisterFactory(std::string name);

        /********************************************
         * Unregister all the interpolator classes
         *******************************************/
        static void unregisterFactories( );

        /**********************************************
        * Allocates interpolators based on cfg
        **********************************************/
        static Interpolator<TConfig> *allocate(AMG_Config &cfg, const std::string &cfg_scope);

        typedef typename std::map<std::string, InterpolatorFactory<TConfig>*>::const_iterator Iterator;

        static Iterator getIterator() { return getFactories().begin(); };
        static bool isIteratorLast(const Iterator &iter) { if ( iter == getFactories().end() ) return true; else return false; };

    private:
        static std::map<std::string, InterpolatorFactory<TConfig>*> &getFactories( );
};

} // namespace amgx
