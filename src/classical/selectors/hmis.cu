// SPDX-FileCopyrightText: 2011 - 2025 NVIDIA CORPORATION. All Rights Reserved.
//
// SPDX-License-Identifier: BSD-3-Clause

#include <classical/selectors/hmis.h>
#include <classical/selectors/pmis.h>
#include <classical/selectors/rs.h>
#include <classical/interpolators/common.h>
#include <cutil.h>
#include <util.h>
#include <types.h>

#include<thrust/count.h>

namespace amgx
{

namespace classical
{
/*************************************************************************
 * marks the strongest connected (and indepent) points as coarse
 ************************************************************************/

template <AMGX_VecPrecision t_vecPrec, AMGX_MatPrecision t_matPrec, AMGX_IndPrecision t_indPrec>
void HMIS_Selector<TemplateConfig<AMGX_host, t_vecPrec, t_matPrec, t_indPrec> >
::markCoarseFinePoints_1x1(Matrix_h &A,
                           FVector &weights,
                           const BVector &s_con,
                           IVector &cf_map,
                           IVector &scratch,
                           int cf_map_init)
{
    // First call Ruge Steuben coarsening
    AMG_Config cfg;
    cfg.parseParameterString("use_opt_kernels=0");
    cfg.setParameter("use_opt_kernels", this->m_use_opt_kernels, "default");
    RS_Selector<TConfig_h> *rs_selector = new RS_Selector<TConfig_h>(cfg, "default");
    rs_selector->markCoarseFinePoints(A, weights, s_con, cf_map, scratch, cf_map_init);
    delete rs_selector;
    // Then call pmis selector
    PMIS_Selector<TConfig_h> *pmis_selector = new PMIS_Selector<TConfig_h>(cfg, "default");
    pmis_selector->markCoarseFinePoints(A, weights, s_con, cf_map, scratch, 1);
    delete pmis_selector;
}

/*************************************************************************
 * device kernels
 ************************************************************************/

/*************************************************************************
 * Implementing the HMIS algorith
 ************************************************************************/

template <AMGX_VecPrecision t_vecPrec, AMGX_MatPrecision t_matPrec, AMGX_IndPrecision t_indPrec>
void HMIS_Selector<TemplateConfig<AMGX_device, t_vecPrec, t_matPrec, t_indPrec> >::markCoarseFinePoints_1x1(Matrix_d &A,
        FVector &weights,
        const BVector &s_con,
        IVector &cf_map,
        IVector &scratch,
        int cf_map_init)
{
    // Copy to host
    Matrix_h A_h = A;
    FVector_h weights_h = weights;
    BVector_h s_con_h = s_con;
    IVector_h scratch_h = scratch;
    IVector_h cf_map_h = cf_map;
    // Need to manually pass parameters here
    AMG_Config cfg;
    cfg.parseParameterString("use_opt_kernels=0");
    cfg.setParameter("use_opt_kernels", this->m_use_opt_kernels, "default");
    // Copy matrix to host
    // First call Ruge Steuben coarsening
    RS_Selector<TConfig_h> *rs_selector = new RS_Selector<TConfig_h>(cfg, "default");
    rs_selector->markCoarseFinePoints(A_h, weights_h, s_con_h, cf_map_h, scratch_h, 0);
    delete rs_selector;
    // Copy cf_map result to device
    cf_map = cf_map_h;
    PMIS_Selector<TConfig_d> *pmis_selector = new PMIS_Selector<TConfig_d>(cfg, "default");
    pmis_selector->markCoarseFinePoints(A, weights, s_con, cf_map, scratch, 1);
    delete pmis_selector;
}

template <class T_Config>
void HMIS_SelectorBase< T_Config>::markCoarseFinePoints(Matrix< T_Config> &A,
        FVector &weights,
        const BVector &s_con,
        IVector &cf_map,
        IVector &scratch,
        int cf_map_init)
{
    ViewType oldView = A.currentView();
    A.setView(OWNED);

    if (A.get_block_size() == 1)
    {
        markCoarseFinePoints_1x1(A, weights, s_con, cf_map, scratch, cf_map_init);
    }
    else
    {
        FatalError("Unsupported block size HMIS selector", AMGX_ERR_NOT_SUPPORTED_BLOCKSIZE);
    }

    A.setView(oldView);
}

/****************************************
 * Explict instantiations
 ***************************************/
#define AMGX_CASE_LINE(CASE) template class HMIS_SelectorBase<TemplateMode<CASE>::Type>;
AMGX_FORALL_BUILDS(AMGX_CASE_LINE)
#undef AMGX_CASE_LINE

#define AMGX_CASE_LINE(CASE) template class HMIS_Selector<TemplateMode<CASE>::Type>;
AMGX_FORALL_BUILDS(AMGX_CASE_LINE)
#undef AMGX_CASE_LINE

} // namespace classical

} // namespace amgx
