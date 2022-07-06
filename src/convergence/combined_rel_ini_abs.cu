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

#include "amg_config.h"
#include "convergence/combined_rel_ini_abs.h"

#include <algorithm>

namespace amgx
{
template<typename TConfig>
RelativeAbsoluteCombinedConvergence<TConfig>::RelativeAbsoluteCombinedConvergence(AMG_Config &cfg, const std::string &cfg_scope) : Convergence<TConfig>(cfg, cfg_scope)
{
}

template<class TConfig>
void RelativeAbsoluteCombinedConvergence<TConfig>::convergence_init()
{
    this->setTolerance(this->m_cfg->template getParameter<double>("tolerance", this->m_cfg_scope));
    this->m_alt_rel_tolerance = this->m_cfg->template getParameter<double>("alt_rel_tolerance", this->m_cfg_scope);
}

template<class TConfig>
bool RelativeAbsoluteCombinedConvergence<TConfig>::convergence_update_and_check(const PODVec_h &nrm, const PODVec_h &nrm_ini)
{
    bool res_converged = true;
    bool res_converged_abs = true;
    bool res_converged_abs_precision = true;

    for (int i = 0; i < nrm.size(); i++)
    {
        bool conv_abs = nrm[i] < this->m_tolerance;
        res_converged_abs = res_converged_abs && conv_abs;
        bool conv = (nrm[i] / nrm_ini[i] <= this->m_alt_rel_tolerance);
        res_converged = res_converged && conv;
        bool conv_abs_precision = (nrm[i] <= std::max(nrm_ini[i] * Epsilon_conv<ValueTypeB>::value(), (PODValueTypeB)(1e-20)));
        res_converged_abs_precision = res_converged_abs_precision && conv_abs_precision;
    }

    if (res_converged_abs_precision)
    {
        return true;
    }

    return res_converged || res_converged_abs;
}


/****************************************
 * Explict instantiations
 ***************************************/
template class RelativeAbsoluteCombinedConvergence<TConfigGeneric_d>;
template class RelativeAbsoluteCombinedConvergence<TConfigGeneric_h>;

} // end namespace

