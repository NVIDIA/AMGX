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

namespace energymin
{

template <class T_Config> class Selector;

template <class TConfig>
class Selector_Base
{
        typedef typename TConfig::MatPrec ValueType;
        typedef typename TConfig::IndPrec IndexType;
        typedef Vector<typename TConfig::template setVecPrec<AMGX_vecInt>::Type> IVector;
        typedef Vector<typename TConfig::template setVecPrec<AMGX_vecFloat>::Type> FVector;
        typedef Vector<typename TConfig::template setVecPrec<AMGX_vecBool>::Type> BVector;
        typedef Vector<typename TConfig::template setVecPrec<AMGX_vecInt>::Type> IntVector;

        typedef typename TConfig::template setVecPrec<AMGX_vecInt64>::Type i64vec_value_type;
        typedef Vector<i64vec_value_type> I64Vector;

    public:
        virtual void markCoarseFinePoints(Matrix<TConfig> &A,
                                          IVector &cf_map,
                                          IndexType &numCoarse) = 0;

        virtual ~Selector_Base() {};
};


// ----------------------------
//  specialization for host
// ----------------------------
template< AMGX_VecPrecision V, AMGX_MatPrecision M, AMGX_IndPrecision I >
class Selector<TemplateConfig<AMGX_host, V, M, I> > : public Selector_Base< TemplateConfig<AMGX_host, V, M, I> >
{
    public:
        typedef TemplateConfig<AMGX_host, V, M, I> TConfig_h;
        typedef typename TConfig_h::MatPrec ValueType;
        typedef typename TConfig_h::IndPrec IndexType;
        typedef Vector<typename TConfig_h::template setVecPrec<AMGX_vecInt>::Type>   IVector;
        typedef Vector<typename TConfig_h::template setVecPrec<AMGX_vecFloat>::Type> FVector;
        typedef Vector<typename TConfig_h::template setVecPrec<AMGX_vecBool>::Type>  BVector;
        typedef Matrix<TConfig_h> Matrix_h;
        typedef Vector<typename TConfig_h::template setVecPrec<AMGX_vecInt>::Type>   IntVector;

        typedef typename TConfig_h::template setVecPrec<AMGX_vecInt64>::Type i64vec_value_type_h;
        typedef Vector<i64vec_value_type_h> I64Vector;

        virtual void markCoarseFinePoints(Matrix<TConfig_h> &A,
                                          IVector &cf_map,
                                          IndexType &numCoarse) = 0;

        virtual ~Selector() {};
};

// ----------------------------
//  specialization for device
// ----------------------------
template< AMGX_VecPrecision V, AMGX_MatPrecision M, AMGX_IndPrecision I >
class Selector<TemplateConfig<AMGX_device, V, M, I> > : public Selector_Base< TemplateConfig<AMGX_device, V, M, I> >
{
    public:
        typedef TemplateConfig<AMGX_device, V, M, I> TConfig_d;
        typedef typename TConfig_d::MatPrec ValueType;
        typedef typename TConfig_d::IndPrec IndexType;
        typedef Vector<typename TConfig_d::template setVecPrec<AMGX_vecInt>::Type>   IVector;
        typedef Vector<typename TConfig_d::template setVecPrec<AMGX_vecFloat>::Type> FVector;
        typedef Vector<typename TConfig_d::template setVecPrec<AMGX_vecBool>::Type>  BVector;
        typedef Matrix<TConfig_d> Matrix_d;
        typedef Vector<typename TConfig_d::template setVecPrec<AMGX_vecInt>::Type>   IntVector;

        typedef typename TConfig_d::template setVecPrec<AMGX_vecInt64>::Type i64vec_value_type_d;
        typedef Vector<i64vec_value_type_d> I64Vector_d;

        virtual void markCoarseFinePoints(Matrix<TConfig_d> &A,
                                          IVector &cf_map,
                                          IndexType &numCoarse) = 0;
};


template<class TConfig>
class SelectorFactory
{
    public:
        virtual Selector<TConfig> *create() = 0;
        virtual ~SelectorFactory() {};

        /********************************************
         * Register a selector class with key "name"
         *******************************************/
        static void registerFactory(std::string name, SelectorFactory<TConfig> *f);

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
        static Selector<TConfig> *allocate(AMG_Config &cfg, const std::string &cfg_scope);

        typedef typename std::map<std::string, SelectorFactory<TConfig>*>::const_iterator Iterator;

        static Iterator getIterator() { return getFactories().begin(); };
        static bool isIteratorLast(const Iterator &iter) { if ( iter == getFactories().end() ) return true; else return false; };

    private:
        static std::map<std::string, SelectorFactory<TConfig>*> &getFactories( );
};

} // namespace energymin

} // namespace amgx


