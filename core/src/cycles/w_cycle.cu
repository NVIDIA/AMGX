/* Copyright (c) 2013-2017, NVIDIA CORPORATION. All rights reserved.
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

#include <cycles/w_cycle.h>

namespace amgx
{

template< AMGX_VecPrecision t_vecPrec, AMGX_MatPrecision t_matPrec, AMGX_IndPrecision t_indPrec >
void W_CycleDispatcher<t_vecPrec, t_matPrec, t_indPrec>::dispatch( AMG_Class *amg, AMG_Level<TConfig_h> *level, Vector<TConfig_h> &b, Vector<TConfig_h> &x ) const
{
    W_Cycle<TConfig_h>( amg, level, b, x );
    W_Cycle<TConfig_h>( amg, level, b, x );
}

template< AMGX_VecPrecision t_vecPrec, AMGX_MatPrecision t_matPrec, AMGX_IndPrecision t_indPrec >
void W_CycleDispatcher<t_vecPrec, t_matPrec, t_indPrec>::dispatch( AMG_Class *amg, AMG_Level<TConfig_d> *level, Vector<TConfig_d> &b, Vector<TConfig_d> &x  ) const
{
    AMGX_CPU_PROFILER( "W_Cycle::dispatch " );
    W_Cycle<TConfig_d>( amg, level, b, x );
    W_Cycle<TConfig_d>( amg, level, b, x );
}

/****************************************
 * Explict instantiations
 ***************************************/
template class W_CycleDispatcher<AMGX_vecDouble, AMGX_matDouble, AMGX_indInt>;
template class W_CycleDispatcher<AMGX_vecFloat, AMGX_matFloat, AMGX_indInt>;
template class W_CycleDispatcher<AMGX_vecDouble, AMGX_matFloat, AMGX_indInt>;

template class W_CycleDispatcher<AMGX_vecComplex, AMGX_matComplex, AMGX_indInt>;
template class W_CycleDispatcher<AMGX_vecDoubleComplex, AMGX_matComplex, AMGX_indInt>;
template class W_CycleDispatcher<AMGX_vecDoubleComplex, AMGX_matDoubleComplex, AMGX_indInt>;

} // namespace amgx
