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
