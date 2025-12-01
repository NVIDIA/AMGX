// SPDX-FileCopyrightText: 2011 - 2025 NVIDIA CORPORATION. All Rights Reserved.
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
DECLARE_UNITTEST_BEGIN(ILU1_in_vs_out_diag);

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
    Matrix_h tempA;
    bool diag_prop = false;
    int block_size = 4;
    tempA.convert( A, ( diag_prop ? DIAG : 0 ) | CSR, block_size, block_size );
    // resize and copy vector values
    Vector_h new_b = b;
    new_b.resize(A.get_num_rows() * block_size);

    for ( int i = 0; i < new_b.size(); i++ )
    {
        new_b[i] = i < b.size() ?  b[i] : 1.0;
    }

    Resources res;        // default resources
    {
        MatrixA A_hd = A;
        VVector x_hd, b_hd, x_fin_in, x_fin_out;
        b_hd = b;
        x_hd = x;
        // copy initial condition
        x_fin_in = x_hd;
        x_fin_out = x_hd;
        AMG_Configuration cfg;
        UNITTEST_ASSERT_TRUE ( cfg.parseParameterString("solver=MULTICOLOR_ILU, reorder_cols_by_color=1, max_uncolored_percentage=0, coloring_level=2, ilu_sparsity_level=1, max_iters=10, insert_diag_while_reordering=1, monitor_residual=1, print_solve_stats=1") == AMGX_OK);
        AMGX_STATUS solve_status = AMGX_ST_CONVERGED;
        AMG_Solver<TConfig> amg(&res, cfg);
        amg.setup(A_hd);
        amg.solve(b_hd, x_fin_out, solve_status);
        UNITTEST_ASSERT_TRUE ( cfg.parseParameterString("solver=MULTICOLOR_ILU, reorder_cols_by_color=1, max_uncolored_percentage=0, coloring_level=2, ilu_sparsity_level=1, max_iters=10, insert_diag_while_reordering=0, monitor_residual=1, print_solve_stats=1") == AMGX_OK);
        A_hd = tempA;
        b_hd = new_b;
        AMG_Solver<TConfig> amg2(&res, cfg);
        amg2.setup(A_hd);
        amg2.solve(b_hd, x_fin_in, solve_status);
        UNITTEST_ASSERT_EQUAL_TOL(x_fin_in, x_fin_out, 1e-10);
    }
}

DECLARE_UNITTEST_END(ILU1_in_vs_out_diag);


// if you want to be able run this test for all available configs you can write this:
//#define AMGX_CASE_LINE(CASE) TemplateTest <TemplateMode<CASE>::Type>  TemplateTest_##CASE;
//  AMGX_FORALL_BUILDS(AMGX_CASE_LINE)
//#undef AMGX_CASE_LINE

// or run for all device configs
//#define AMGX_CASE_LINE(CASE) TemplateTest <TemplateMode<CASE>::Type>  TemplateTest_##CASE;
//  AMGX_FORALL_BUILDS_DEVICE(AMGX_CASE_LINE)
//#undef AMGX_CASE_LINE

ILU1_in_vs_out_diag <TemplateMode<AMGX_mode_dDDI>::Type>  ILU1_in_vs_out_diag_instance_mode_dDDI;

// or you can specify several desired configs
//TemplateTest <TemplateMode<AMGX_mode_hDFI>::Type>  TemplateTest_hDFI;
//TemplateTest <TemplateMode<AMGX_mode_dDFI>::Type>  TemplateTest_dDFI;


} //namespace amgx
