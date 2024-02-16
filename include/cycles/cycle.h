// SPDX-FileCopyrightText: 2011 - 2024 NVIDIA CORPORATION. All Rights Reserved.
//
// SPDX-License-Identifier: BSD-3-Clause

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
