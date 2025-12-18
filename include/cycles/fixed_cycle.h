// SPDX-FileCopyrightText: 2011 - 2025 NVIDIA CORPORATION. All Rights Reserved.
//
// SPDX-License-Identifier: BSD-3-Clause

#pragma once
#include <basic_types.h>

namespace amgx
{
template< class T_Config, template< AMGX_VecPrecision t_vecPrec, AMGX_MatPrecision t_matPrec, AMGX_IndPrecision t_indPrec > class CycleDispatcher > class FixedCycle;
}

#include <cycles/cycle.h>
#include <amg_level.h>

namespace amgx
{

template< class T_Config, template< AMGX_VecPrecision t_vecPrec, AMGX_MatPrecision t_matPrec, AMGX_IndPrecision t_indPrec > class CycleDispatcher >
class FixedCycle: public Cycle<T_Config>
{
    public:
        static const AMGX_VecPrecision vecPrec = T_Config::vecPrec;
        static const AMGX_MatPrecision matPrec = T_Config::matPrec;
        static const AMGX_IndPrecision indPrec = T_Config::indPrec;
        typedef AMG<vecPrec, matPrec, indPrec> AMG_Class;
        typedef T_Config TConfig;
        typedef Vector<TConfig> VVector;
        typedef typename T_Config::MatPrec ValueTypeA;
        typedef typename T_Config::template setMemSpace<AMGX_host  >::Type TConfig_h;
        typedef Vector<TConfig_h> Vector_h;


        void cycle( AMG_Class *amg, AMG_Level<T_Config> *level, VVector &b, VVector &x );
        virtual ~FixedCycle() {};
};

} // namespace amgx
