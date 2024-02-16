// SPDX-FileCopyrightText: 2011 - 2024 NVIDIA CORPORATION. All Rights Reserved.
//
// SPDX-License-Identifier: BSD-3-Clause

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
AMGX_STATUS RelativeAbsoluteCombinedConvergence<TConfig>::convergence_update_and_check(const PODVec_h &nrm, const PODVec_h &nrm_ini)
{
    bool res_converged = true;
    bool res_converged_abs = true;
    bool res_converged_abs_precision = true;

    PODValueTypeB eps = (PODValueTypeB)(1e-20);

    for (int i = 0; i < nrm.size(); i++)
    {
        bool conv_abs = nrm[i] < this->m_tolerance;
        res_converged_abs = res_converged_abs && conv_abs;
        bool conv = (nrm_ini[i] <= eps ? true : (nrm[i] / nrm_ini[i] <= this->m_alt_rel_tolerance));
        res_converged = res_converged && conv;
        bool conv_abs_precision = (nrm[i] <= std::max(nrm_ini[i] * Epsilon_conv<ValueTypeB>::value(), eps));
        res_converged_abs_precision = res_converged_abs_precision && conv_abs_precision;
    }

    if (res_converged_abs_precision)
    {
        return AMGX_ST_CONVERGED;
    }

    return (res_converged || res_converged_abs) ? AMGX_ST_CONVERGED : AMGX_ST_NOT_CONVERGED;
}


/****************************************
 * Explict instantiations
 ***************************************/
#define AMGX_CASE_LINE(CASE) template class RelativeAbsoluteCombinedConvergence<TemplateMode<CASE>::Type>;
AMGX_FORALL_BUILDS(AMGX_CASE_LINE)
AMGX_FORCOMPLEX_BUILDS(AMGX_CASE_LINE)
#undef AMGX_CASE_LINE

} // end namespace

