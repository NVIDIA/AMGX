// SPDX-FileCopyrightText: 2011 - 2024 NVIDIA CORPORATION. All Rights Reserved.
//
// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#include <basic_types.h>
#include <matrix.h>
#include <map>
#include <amg_config.h>
#include <amg.h>

namespace amgx
{

template <class T_config> class Scaler;

enum ScaleDirection {SCALE, UNSCALE};

enum ScaleSide {LEFT, RIGHT};

template <class TConfig>
class Scaler_Base
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
        virtual void setup(Matrix<TConfig> &A) = 0;
        virtual void scaleMatrix(Matrix<TConfig> &A, ScaleDirection scaleOrUnscale) = 0;
        virtual void scaleVector(Vector<TConfig> &x, ScaleDirection scaleOrUnscale, ScaleSide leftOrRight) = 0;
        virtual ~Scaler_Base() {}
};

// ----------------------------
//  specialization for host
// ----------------------------


template< AMGX_VecPrecision V, AMGX_MatPrecision M, AMGX_IndPrecision I >
class Scaler<TemplateConfig<AMGX_host, V, M, I> > : public Scaler_Base< TemplateConfig<AMGX_host, V, M, I> >
{
        typedef TemplateConfig<AMGX_host, V, M, I> TConfig_h;
        typedef typename TConfig_h::MatPrec ValueType;
        typedef typename TConfig_h::IndPrec IndexType;
        typedef Vector<typename TConfig_h::template setVecPrec<AMGX_vecInt>::Type> IVector;
        typedef Vector<typename TConfig_h::template setVecPrec<AMGX_vecFloat>::Type> FVector;
        typedef Vector<typename TConfig_h::template setVecPrec<AMGX_vecBool>::Type> BVector;
        typedef Matrix<TConfig_h> Matrix_h;
        typedef Vector<typename TConfig_h::template setVecPrec<AMGX_vecInt>::Type> IntVector;

        typedef typename TConfig_h::template setVecPrec<AMGX_vecInt64>::Type i64vec_value_type_h;
        typedef Vector<i64vec_value_type_h> I64Vector;

    public:
        virtual void setup(Matrix<TConfig_h> &A);
        virtual void scaleMatrix(Matrix<TConfig_h> &A, ScaleDirection scaleOrUnscale);
        virtual void scaleVector(Vector<TConfig_h> &v, ScaleDirection scaleOrUnscale, ScaleSide leftOrRight);
};

// ----------------------------
//  specialization for device
// ----------------------------

template< AMGX_VecPrecision V, AMGX_MatPrecision M, AMGX_IndPrecision I >
class Scaler<TemplateConfig<AMGX_device, V, M, I> > : public Scaler_Base< TemplateConfig<AMGX_device, V, M, I> >
{
    public:
        typedef TemplateConfig<AMGX_device, V, M, I> TConfig_d;
        typedef typename TConfig_d::MatPrec ValueType;
        typedef typename TConfig_d::IndPrec IndexType;
        typedef Vector<typename TConfig_d::template setVecPrec<AMGX_vecInt>::Type> IVector;
        typedef Vector<typename TConfig_d::template setVecPrec<AMGX_vecFloat>::Type> FVector;
        typedef Vector<typename TConfig_d::template setVecPrec<AMGX_vecBool>::Type> BVector;
        typedef Matrix<TConfig_d> Matrix_d;
        typedef Vector<typename TConfig_d::template setVecPrec<AMGX_vecInt>::Type> IntVector;

        typedef typename TConfig_d::template setVecPrec<AMGX_vecInt64>::Type i64vec_value_type_d;
        typedef Vector<i64vec_value_type_d> I64Vector_d;

        virtual void setup(Matrix<TConfig_d> &A);
        virtual void scaleMatrix(Matrix<TConfig_d> &A, ScaleDirection scaleOrUnscale);
        virtual void scaleVector(Vector<TConfig_d> &v, ScaleDirection scaleOrUnscale, ScaleSide leftOrRight);
};

template <class TConfig>
class ScalerFactory
{
    public:
        virtual Scaler<TConfig> *create(AMG_Config &cfg, const std::string &cfg_scope) = 0;
        virtual ~ScalerFactory() {};

        /********************************************
         * Register a selector class with key "name"
         *******************************************/
        static void registerFactory(std::string name, ScalerFactory<TConfig> *f);

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
        static Scaler<TConfig> *allocate(AMG_Config &cfg, const std::string &cfg_scope);

        typedef typename std::map<std::string, ScalerFactory<TConfig>*>::const_iterator Iterator;

        static Iterator getIterator() { return getFactories().begin(); };
        static bool isIteratorLast(const Iterator &iter) { if ( iter == getFactories().end() ) return true; else return false; };

    private:
        static std::map<std::string, ScalerFactory<TConfig>*> &getFactories( );
};

} // namespace amgx
