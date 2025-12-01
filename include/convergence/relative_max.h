// SPDX-FileCopyrightText: 2011 - 2025 NVIDIA CORPORATION. All Rights Reserved.
//
// SPDX-License-Identifier: BSD-3-Clause

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

        AMGX_STATUS convergence_update_and_check(const PODVec_h &nrm, const PODVec_h &nrm_ini);

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

