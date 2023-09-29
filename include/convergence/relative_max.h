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

#include <convergence/convergence.h>

namespace amgx
{

template<class TConfig>
class RelativeMaxConvergence : public Convergence<TConfig>
{
    public:
        static const AMGX_VecPrecision vecPrec = TConfig::vecPrec;
        static const AMGX_MatPrecision matPrec = TConfig::matPrec;
        static const AMGX_IndPrecision indPrec = TConfig::indPrec;
        typedef Vector<TemplateConfig<AMGX_host, vecPrec, matPrec, indPrec> > Vector_h;
        typedef typename TConfig::VecPrec ValueTypeB;
        typedef typename TConfig::template setMemSpace<AMGX_host>::Type TConfig_h;
        typedef typename TConfig::template setMemSpace<AMGX_device>::Type TConfig_d;
        typedef typename types::PODTypes<ValueTypeB>::type PODValueTypeB;
        typedef Vector<typename TConfig::template setVecPrec<types::PODTypes<ValueTypeB>::vec_prec>::Type> PODVec;
        typedef Vector<typename TConfig_h::template setVecPrec<types::PODTypes<ValueTypeB>::vec_prec>::Type> PODVec_h;

        RelativeMaxConvergence(AMG_Config &amg, const std::string &cfg_scope);

        void convergence_init();

        bool convergence_update_and_check(const PODVec_h &nrm, const PODVec_h &nrm_ini);

    private:
        PODVec_h _max_nrm;
};

template<class TConfig>
class RelativeMaxConvergenceFactory : public ConvergenceFactory<TConfig>
{
    public:
        Convergence<TConfig> *create(AMG_Config &cfg, const std::string &cfg_scope) { return new RelativeMaxConvergence<TConfig>(cfg, cfg_scope); }
};

} // end namespace amgx

