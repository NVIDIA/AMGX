// SPDX-FileCopyrightText: 2013 - 2024 NVIDIA CORPORATION. All Rights Reserved.
//
// SPDX-License-Identifier: BSD-3-Clause

#include <cycles/f_cycle.h>
#include <cycles/w_cycle.h>
#include <cycles/v_cycle.h>

namespace amgx
{

template< AMGX_VecPrecision t_vecPrec, AMGX_MatPrecision t_matPrec, AMGX_IndPrecision t_indPrec >
void F_CycleDispatcher<t_vecPrec, t_matPrec, t_indPrec>::dispatch( AMG_Class *amg, AMG_Level<TConfig_h> *level, Vector<TConfig_h> &b, Vector<TConfig_h> &x ) const
{
    W_Cycle<TConfig_h>( amg, level, b, x );
    V_Cycle<TConfig_h>( amg, level, b, x );
}

template< AMGX_VecPrecision t_vecPrec, AMGX_MatPrecision t_matPrec, AMGX_IndPrecision t_indPrec >
void F_CycleDispatcher<t_vecPrec, t_matPrec, t_indPrec>::dispatch( AMG_Class *amg, AMG_Level<TConfig_d> *level, Vector<TConfig_d> &b, Vector<TConfig_d> &x ) const
{
    W_Cycle<TConfig_d>( amg, level, b, x );
    V_Cycle<TConfig_d>( amg, level, b, x );
}

/****************************************
 * Explict instantiations
 ***************************************/
template class F_CycleDispatcher<AMGX_vecDouble, AMGX_matDouble, AMGX_indInt>;
template class F_CycleDispatcher<AMGX_vecFloat, AMGX_matFloat, AMGX_indInt>;
template class F_CycleDispatcher<AMGX_vecDouble, AMGX_matFloat, AMGX_indInt>;

template class F_CycleDispatcher<AMGX_vecComplex, AMGX_matComplex, AMGX_indInt>;
template class F_CycleDispatcher<AMGX_vecDoubleComplex, AMGX_matComplex, AMGX_indInt>;
template class F_CycleDispatcher<AMGX_vecDoubleComplex, AMGX_matDoubleComplex, AMGX_indInt>;

} // namespace amgx
