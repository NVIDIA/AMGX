// SPDX-FileCopyrightText: 2011 - 2025 NVIDIA CORPORATION. All Rights Reserved.
//
// SPDX-License-Identifier: BSD-3-Clause

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

