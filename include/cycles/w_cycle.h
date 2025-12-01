// SPDX-FileCopyrightText: 2011 - 2025 NVIDIA CORPORATION. All Rights Reserved.
//
// SPDX-License-Identifier: BSD-3-Clause

#pragma once

namespace amgx
{
template <class T_Config> class W_Cycle;
}

#include <cycles/fixed_cycle.h>

namespace amgx
{

template< AMGX_VecPrecision t_vecPrec, AMGX_MatPrecision t_matPrec, AMGX_IndPrecision t_indPrec >
class W_CycleDispatcher
{
        typedef TemplateConfig<AMGX_host, t_vecPrec, t_matPrec, t_indPrec> TConfig_h;
        typedef TemplateConfig<AMGX_device, t_vecPrec, t_matPrec, t_indPrec> TConfig_d;
        typedef AMG<t_vecPrec, t_matPrec, t_indPrec> AMG_Class;
    public:
        void dispatch( AMG_Class *amg, AMG_Level<TConfig_h> *level, Vector<TConfig_h> &b, Vector<TConfig_h> &x ) const;
        void dispatch( AMG_Class *amg, AMG_Level<TConfig_d> *level, Vector<TConfig_d> &b, Vector<TConfig_d> &x ) const;
};

template <class T_Config>
class W_Cycle : public FixedCycle<T_Config, W_CycleDispatcher>
{
        typedef Vector<T_Config> VVector;
        static const AMGX_VecPrecision vecPrec = T_Config::vecPrec;
        static const AMGX_MatPrecision matPrec = T_Config::matPrec;
        static const AMGX_IndPrecision indPrec = T_Config::indPrec;
        typedef AMG<vecPrec, matPrec, indPrec> AMG_Class;
    public:
        W_Cycle( AMG_Class *amg, AMG_Level<T_Config> *level, VVector &b, VVector &x )
        {
            this->cycle(amg, level, b, x);
        }
        ~W_Cycle() {};
};

template<class T_Config>
class W_CycleFactory : public CycleFactory<T_Config>
{
        static const AMGX_VecPrecision vecPrec = T_Config::vecPrec;
        static const AMGX_MatPrecision matPrec = T_Config::matPrec;
        static const AMGX_IndPrecision indPrec = T_Config::indPrec;
        typedef AMG<vecPrec, matPrec, indPrec> AMG_Class;
    public:
        typedef Vector<T_Config> VVector;
        Cycle<T_Config> *create( AMG_Class *amg, AMG_Level<T_Config> *level, VVector &b, VVector &x ) { return new W_Cycle<T_Config>(amg, level, b, x); }
};

} // namespace amgx
