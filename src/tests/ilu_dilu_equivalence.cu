// SPDX-FileCopyrightText: 2011 - 2024 NVIDIA CORPORATION. All Rights Reserved.
//
// SPDX-License-Identifier: BSD-3-Clause

#include "unit_test.h"
#include "amg_solver.h"
#include <matrix_io.h>
#include "test_utils.h"
#include <multiply.h>
#include<blas.h>
#include <csr_multiply.h>
#include "util.h"
#include "time.h"
#include <sstream>

namespace amgx
{

// parameter is used as test name
DECLARE_UNITTEST_BEGIN(ILU_DILU_equivalence);

void run()
{
    Matrix_h A;
    Vector_h b;
    Vector_h x;
    UNITTEST_ASSERT_TRUE(this->read_system("Public/florida/atmosdd.mtx", A, b, x));
    int n_rows = A.get_num_rows();

    if (x.size() == 0) { x.resize(A.get_num_rows() * A.get_block_dimy(), 0.); }

    b.set_block_dimx(1);
    b.set_block_dimy(A.get_block_dimy());
    x.set_block_dimx(1);
    x.set_block_dimy(A.get_block_dimx());
    Resources res;        // default resources
    {
        MatrixA A_hd = A;
        VVector x_hd, b_hd, x_fin_dilu, x_fin_ilu;
        b_hd = b;
        x_hd = x;
        // copy initial condition
        x_fin_dilu = x_hd;
        x_fin_ilu = x_hd;
        AMG_Configuration cfg_dilu, cfg_ilu;
        UNITTEST_ASSERT_TRUE ( cfg_dilu.parseParameterString("solver=MULTICOLOR_DILU, max_uncolored_percentage=0, coloring_level=1, ilu_sparsity_level=0, max_iters=10, monitor_residual=1, print_solve_stats=1") == AMGX_OK);
        UNITTEST_ASSERT_TRUE ( cfg_ilu.parseParameterString("solver=MULTICOLOR_ILU, max_uncolored_percentage=0, coloring_level=1, reorder_cols_by_color=1, ilu_sparsity_level=0, insert_diag_while_reordering=1, max_iters=10, monitor_residual=1, print_solve_stats=1") == AMGX_OK);
        AMGX_STATUS solve_status = AMGX_ST_CONVERGED;
        AMG_Solver<TConfig> amg_dilu(&res, cfg_dilu);
        amg_dilu.setup(A_hd);
        amg_dilu.solve(b_hd, x_fin_dilu, solve_status);
        AMG_Solver<TConfig> amg_ilu(&res, cfg_ilu);
        amg_ilu.setup(A_hd);
        amg_ilu.solve(b_hd, x_fin_ilu, solve_status);
        UNITTEST_ASSERT_EQUAL_TOL(x_fin_ilu, x_fin_dilu, 1e-10);
    }
}

DECLARE_UNITTEST_END(ILU_DILU_equivalence);


// if you want to be able run this test for all available configs you can write this:
//#define AMGX_CASE_LINE(CASE) TemplateTest <TemplateMode<CASE>::Type>  TemplateTest_##CASE;
//  AMGX_FORALL_BUILDS(AMGX_CASE_LINE)
//#undef AMGX_CASE_LINE

// or run for all device configs
//#define AMGX_CASE_LINE(CASE) TemplateTest <TemplateMode<CASE>::Type>  TemplateTest_##CASE;
//  AMGX_FORALL_BUILDS_DEVICE(AMGX_CASE_LINE)
//#undef AMGX_CASE_LINE

ILU_DILU_equivalence <TemplateMode<AMGX_mode_dDDI>::Type>  ILU_DILU_equivalence_instance_mode_dDDI;
//ILU_DILU_equivalence <TemplateMode<AMGX_mode_dFFI>::Type>  ILU_DILU_equivalence_instance_mode_dFFI;

// or you can specify several desired configs
//TemplateTest <TemplateMode<AMGX_mode_hDFI>::Type>  TemplateTest_hDFI;
//TemplateTest <TemplateMode<AMGX_mode_dDFI>::Type>  TemplateTest_dDFI;


} //namespace amgx
