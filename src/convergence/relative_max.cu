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

#include <algorithm> // std::max
#include "amg_config.h"
#include "convergence/relative_max.h"
#include <numerical_zero.h>

namespace amgx
{

template<class TConfig>
RelativeMaxConvergence<TConfig>::RelativeMaxConvergence(AMG_Config &cfg, const std::string &cfg_scope) : Convergence<TConfig>(cfg, cfg_scope)
{
}

template<class TConfig>
void RelativeMaxConvergence<TConfig>::convergence_init()
{
    this->setTolerance(this->m_cfg->template getParameter<double>("tolerance", this->m_cfg_scope));
    _max_nrm.clear();
}

template<class TConfig>
AMGX_STATUS RelativeMaxConvergence<TConfig>::convergence_update_and_check(const PODVec_h &nrm, const PODVec_h &nrm_ini)
{
    /*WARNING: this routine assumes that the input vector has all positive values,
               for example, it is the vector of norms (which are always positive).
               Therefore, it does not take absolute value of the elements during
               computations. You should be careful with this assumption. */
    PODValueTypeB eps = (sizeof(PODValueTypeB) == 4) ? AMGX_NUMERICAL_SZERO : AMGX_NUMERICAL_DZERO;

    if (_max_nrm.empty())
    {
        _max_nrm = nrm;
    }
    else
    {
        for (int i = 0; i < nrm.size(); i++)
        {
            _max_nrm[i] = nrm[i] > _max_nrm[i] ? nrm[i] : _max_nrm[i];
        }
    }

    bool res_converged = true;
    bool res_converged_abs = true;

    for (int i = 0; i < nrm.size(); i++)
    {
        //avoid floating point exception by checking division by zero
        bool conv = (_max_nrm[i] <= eps) ?  true : ((nrm[i] / _max_nrm[i]) <= this->m_tolerance);
        res_converged = res_converged && conv ;
        bool conv_abs = nrm[i] <= std::max(_max_nrm[i] * Epsilon_conv<ValueTypeB>::value(), (PODValueTypeB)1e-20);
        res_converged_abs = res_converged_abs && conv_abs ;
    }

    if (res_converged_abs)
    {
        return AMGX_ST_CONVERGED;
    }

    return res_converged ? AMGX_ST_CONVERGED : AMGX_ST_NOT_CONVERGED;
}


/****************************************
 * Explict instantiations
 ***************************************/
#define AMGX_CASE_LINE(CASE) template class RelativeMaxConvergence<TemplateMode<CASE>::Type>;
AMGX_FORALL_BUILDS(AMGX_CASE_LINE)
AMGX_FORCOMPLEX_BUILDS(AMGX_CASE_LINE)
#undef AMGX_CASE_LINE

} // end namespace

