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

namespace amgx
{
template< class T_Config > class Cycle;
template< class T_Config > class CycleFactory;
}

#include <types.h>
#include <amg_level.h>
#include <amg.h>
#include <norm.h>

namespace amgx
{

template< class T_Config>
class Cycle_Base
{
    public:
        typedef T_Config  TConfig;
        typedef Vector<T_Config> VVector;
        typedef typename T_Config::MatPrec ValueTypeA;
        typedef typename T_Config::VecPrec ValueTypeB;
        typedef typename T_Config::IndPrec IndexType;
        bool isASolvable( const Matrix<T_Config> &A );
        void solveExactly( const Matrix<T_Config> &A, VVector &x, VVector &b );

        virtual ~Cycle_Base() {};
    private:
        virtual void solveExactly_1x1( const Matrix<T_Config> &A, VVector &x, VVector &b ) = 0;
        virtual void solveExactly_4x4( const Matrix<T_Config> &A, VVector &x, VVector &b ) = 0;
};

template <AMGX_VecPrecision t_vecPrec, AMGX_MatPrecision t_matPrec, AMGX_IndPrecision t_indPrec>
class Cycle < TemplateConfig<AMGX_host, t_vecPrec, t_matPrec, t_indPrec> > : public Cycle_Base< TemplateConfig<AMGX_host, t_vecPrec, t_matPrec, t_indPrec> >
{
    public:
        typedef TemplateConfig<AMGX_host, t_vecPrec, t_matPrec, t_indPrec> TConfig_h;
        typedef TConfig_h TConfig;
        typedef TConfig  T_Config;
        typedef Vector<T_Config> VVector;
        typedef typename T_Config::MatPrec ValueTypeA;
        typedef typename T_Config::VecPrec ValueTypeB;
        typedef typename T_Config::IndPrec IndexType;

        virtual ~Cycle() {};
    private:
        void solveExactly_1x1( const Matrix<T_Config> &A, VVector &x, VVector &b );
        void solveExactly_4x4( const Matrix<T_Config> &A, VVector &x, VVector &b );
};

template <AMGX_VecPrecision t_vecPrec, AMGX_MatPrecision t_matPrec, AMGX_IndPrecision t_indPrec>
class Cycle < TemplateConfig<AMGX_device, t_vecPrec, t_matPrec, t_indPrec> > : public Cycle_Base< TemplateConfig<AMGX_device, t_vecPrec, t_matPrec, t_indPrec> >
{
    public:
        typedef TemplateConfig<AMGX_device, t_vecPrec, t_matPrec, t_indPrec> TConfig_d;
        typedef TConfig_d  TConfig;
        typedef TConfig  T_Config;
        typedef Vector<T_Config> VVector;
        typedef typename T_Config::MatPrec ValueTypeA;
        typedef typename T_Config::VecPrec ValueTypeB;
        typedef typename T_Config::IndPrec IndexType;


        virtual ~Cycle() {};
    private:
        void solveExactly_1x1( const Matrix<T_Config> &A, VVector &x, VVector &b );
        void solveExactly_4x4( const Matrix<T_Config> &A, VVector &x, VVector &b );
};

template< class T_Config >
class CycleFactory
{
        //typedef typename TraitsFromMatrix<Matrix<T_Config> >::Traits MatrixTraits;
        static const AMGX_VecPrecision vecPrec = T_Config::vecPrec;
        static const AMGX_MatPrecision matPrec = T_Config::matPrec;
        static const AMGX_IndPrecision indPrec = T_Config::indPrec;
        typedef AMG<vecPrec, matPrec, indPrec> AMG_Class;
        typedef Vector<T_Config> VVector;

    public:
        virtual Cycle<T_Config> *create( AMG_Class *amg, AMG_Level<T_Config> *level, VVector &b, VVector &c ) = 0;
        virtual ~CycleFactory() {};

        /********************************************
        * Register a cycle class with key "name"
        *******************************************/
        static void registerFactory( std::string name, CycleFactory<T_Config> *f );

        /********************************************
         * Unregister a cycle class with key "name"
         *******************************************/
        static void unregisterFactory(std::string name);

        /********************************************
        * Unregister all the strength classes
        *******************************************/
        static void unregisterFactories( );

        /*********************************************
        * Allocates cycles
        *********************************************/
        static Cycle<T_Config> *allocate( AMG_Class *amg, AMG_Level<T_Config> *level, VVector &b, VVector &c );

        /*********************************************
        * Generates cycles
        *********************************************/
        static void generate( AMG_Class *amg, AMG_Level<T_Config> *level, VVector &b, VVector &c );

        typedef typename std::map<std::string, CycleFactory<T_Config>*>::const_iterator Iterator;

        static Iterator getIterator() { return getFactories().begin(); };
        static bool isIteratorLast(const Iterator &iter) { if ( iter == getFactories().end() ) return true; else return false; };

    private:
        static std::map<std::string, CycleFactory<T_Config> *> &getFactories( );
};
} // namespace amgx
