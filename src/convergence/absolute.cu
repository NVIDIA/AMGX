// SPDX-FileCopyrightText: 2011 - 2024 NVIDIA CORPORATION. All Rights Reserved.
//
// SPDX-License-Identifier: BSD-3-Clause

#include "amg_config.h"
#include "convergence/absolute.h"

namespace amgx
{

template<class TConfig>
AbsoluteConvergence<TConfig>::AbsoluteConvergence(AMG_Config &cfg, const std::string &cfg_scope) : Convergence<TConfig>(cfg, cfg_scope)
{
}

template<class TConfig>
void AbsoluteConvergence<TConfig>::convergence_init()
{
    this->m_tolerance = this->m_cfg->AMG_Config::template getParameter<double>("tolerance", this->m_cfg_scope);
}


template<class TConfig>
AMGX_STATUS AbsoluteConvergence<TConfig>::convergence_update_and_check(const PODVec_h &nrm, const PODVec_h &nrm_ini)
{
    bool res_converged = true;

    for (int i = 0; i < nrm.size(); i++)
    {
        bool conv = nrm[i] < this->m_tolerance;
        res_converged = res_converged && conv;
    }

    return res_converged ? AMGX_ST_CONVERGED : AMGX_ST_NOT_CONVERGED;
}

/****************************************
 * Explict instantiations
 ***************************************/
#define AMGX_CASE_LINE(CASE) template class AbsoluteConvergence<TemplateMode<CASE>::Type>;
AMGX_FORALL_BUILDS(AMGX_CASE_LINE)
AMGX_FORCOMPLEX_BUILDS(AMGX_CASE_LINE)
#undef AMGX_CASE_LINE

} // end namespace

