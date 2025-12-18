// SPDX-FileCopyrightText: 2011 - 2025 NVIDIA CORPORATION. All Rights Reserved.
//
// SPDX-License-Identifier: BSD-3-Clause

#include <algorithm> // std::max
#include "amg_config.h"
#include "convergence/relative_ini.h"

namespace amgx
{
template<typename TConfig>
RelativeIniConvergence<TConfig>::RelativeIniConvergence(AMG_Config &cfg, const std::string &cfg_scope) : Convergence<TConfig>(cfg, cfg_scope)
{
}

template<class TConfig>
void RelativeIniConvergence<TConfig>::convergence_init()
{
    this->setTolerance(this->m_cfg->template getParameter<double>("tolerance", this->m_cfg_scope));
}

template<class TConfig>
AMGX_STATUS RelativeIniConvergence<TConfig>::convergence_update_and_check(const PODVec_h &nrm, const PODVec_h &nrm_ini)
{
    bool res_converged = true;
    bool res_converged_abs = true;

    PODValueTypeB eps = (PODValueTypeB)(1e-20);

    for (int i = 0; i < nrm.size(); i++)
    {
        bool conv = (nrm_ini[i] <= eps ? true : (nrm[i] / nrm_ini[i] <= this->m_tolerance));
        res_converged = res_converged && conv ;
        bool conv_abs = (nrm[i] <= std::max(nrm_ini[i] * Epsilon_conv<ValueTypeB>::value(), eps));
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
#define AMGX_CASE_LINE(CASE) template class RelativeIniConvergence<TemplateMode<CASE>::Type>;
AMGX_FORALL_BUILDS(AMGX_CASE_LINE)
AMGX_FORCOMPLEX_BUILDS(AMGX_CASE_LINE)
#undef AMGX_CASE_LINE

} // end namespace

