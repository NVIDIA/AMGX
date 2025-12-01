// SPDX-FileCopyrightText: 2011 - 2025 NVIDIA CORPORATION. All Rights Reserved.
//
// SPDX-License-Identifier: BSD-3-Clause

#pragma once
namespace amgx
{
template <class T_Config> class CG_Flex_Cycle;
}

#include <cycles/fixed_cycle.h>

namespace amgx
{

template<AMGX_VecPrecision t_vecPrec, AMGX_MatPrecision t_matPrec, AMGX_IndPrecision t_indPrec>
class CG_Flex_CycleDispatcher
{
        typedef TemplateConfig<AMGX_host, t_vecPrec, t_matPrec, t_indPrec> TConfig_h;
        typedef TemplateConfig<AMGX_device, t_vecPrec, t_matPrec, t_indPrec> TConfig_d;
        typedef AMG<t_vecPrec, t_matPrec, t_indPrec> AMG_Class;

    public:
        void dispatch( AMG_Class *amg, AMG_Level<TConfig_h> *level, Vector<TConfig_h> &b, Vector<TConfig_h> &x ) const;
        void dispatch( AMG_Class *amg, AMG_Level<TConfig_d> *level, Vector<TConfig_d> &b, Vector<TConfig_d> &x ) const;
};

template <class T_Config>
class CG_Flex_Cycle: public FixedCycle<T_Config, CG_Flex_CycleDispatcher>
{
        typedef T_Config TConfig;
        static const AMGX_VecPrecision vecPrec = T_Config::vecPrec;
        static const AMGX_MatPrecision matPrec = T_Config::matPrec;
        static const AMGX_IndPrecision indPrec = T_Config::indPrec;
        typedef AMG<vecPrec, matPrec, indPrec> AMG_Class;
        typedef Vector<T_Config> VVector;
    public:
        CG_Flex_Cycle( AMG_Class *amg, AMG_Level<T_Config> *level, VVector &b, VVector &x )
        {
            this->cycle(amg, level, b, x);
        }
};

template<class T_Config>
class CG_Flex_CycleFactory : public CycleFactory<T_Config>
{
    public:
        static const AMGX_VecPrecision vecPrec = T_Config::vecPrec;
        static const AMGX_MatPrecision matPrec = T_Config::matPrec;
        static const AMGX_IndPrecision indPrec = T_Config::indPrec;
        typedef AMG<vecPrec, matPrec, indPrec> AMG_Class;
        typedef Vector<T_Config> VVector;
        Cycle<T_Config> *create( AMG_Class *amg, AMG_Level<T_Config> *level, VVector &b, VVector &x ) { return new CG_Flex_Cycle<T_Config>(amg, level, b, x); }
};

} // namespace amgx
