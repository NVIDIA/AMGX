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
