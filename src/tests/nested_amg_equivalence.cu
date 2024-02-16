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
DECLARE_UNITTEST_BEGIN(Nested_AMG_equivalence);

void run()
{
    Matrix_h A;
    Vector_h b;
    Vector_h x;
    generatePoissonForTest(A, 1, 0, 27, 50, 50, 50);
    b.resize(A.get_num_rows(), 1.);
    x.resize(A.get_num_rows(), 0.);
    int n_rows = A.get_num_rows();

    if (x.size() == 0) { x.resize(A.get_num_rows() * A.get_block_dimy(), 0.); }

    b.set_block_dimx(1);
    b.set_block_dimy(A.get_block_dimy());
    x.set_block_dimx(1);
    x.set_block_dimy(A.get_block_dimx());
    Resources res;    // default resources
    {
        MatrixA A_hd = A;
        A_hd.computeDiagonal();
        A_hd.set_initialized(1);
        VVector x_hd, b_hd, x_fin_no_nesting, x_fin_nesting_fine;
        b_hd = b;
        x_hd = x;
        // copy initial condition
        x_fin_no_nesting = x_hd;
        x_fin_nesting_fine = x_hd;
        AMG_Configuration cfg_no_nesting, cfg_nesting_fine;
        // 5 levels of multigrid
        UNITTEST_ASSERT_TRUE ( cfg_no_nesting.parseParameterString(
                                    "{"
                                        "\"config_version\": 2, "
                                        "\"solver\": {"
                                            "\"matrix_coloring_scheme\": \"MIN_MAX\", "
                                            "\"max_uncolored_percentage\": 0.15, "
                                            "\"algorithm\": \"AGGREGATION\", "
                                            "\"obtain_timings\": 1, "
                                            "\"solver\": \"AMG\", "
                                            "\"smoother\": \"BLOCK_JACOBI\", "
                                            "\"coarse_solver\" : \"NOSOLVER\","
                                            "\"print_solve_stats\": 1, "
                                            "\"presweeps\": 1, "
                                            "\"selector\": \"SIZE_2\", "
                                            "\"coarsest_sweeps\": 1, "
                                            "\"max_iters\": 50, "
                                            "\"monitor_residual\": 1, "
                                            "\"scope\": \"main\", "
                                            "\"max_levels\": 5, "
                                            "\"postsweeps\": 1, "
                                            "\"tolerance\": 1e-6, "
                                            "\"print_grid_stats\": 1, "
                                            "\"norm\": \"L1\", "
                                            "\"cycle\": \"V\""
                                        "}"
                                    "}"
                               ) == AMGX_OK);
        // 3 levels of multigrid + 3 levels of multigrid as a coarse solver (5 levels total) - result should be the same as non-nested
        UNITTEST_ASSERT_TRUE ( cfg_nesting_fine.parseParameterString(
                                    "{"
                                        "\"config_version\": 2, "
                                        "\"solver\": {"
                                            "\"matrix_coloring_scheme\": \"MIN_MAX\", "
                                            "\"max_uncolored_percentage\": 0.15, "
                                            "\"algorithm\": \"AGGREGATION\", "
                                            "\"obtain_timings\": 1, "
                                            "\"solver\": \"AMG\", "
                                            "\"smoother\": \"BLOCK_JACOBI\", "
                                            "\"coarse_solver\" : {"
                                                "\"solver\": \"AMG\", "
                                                "\"algorithm\": \"AGGREGATION\", "
                                                "\"smoother\": \"BLOCK_JACOBI\", "
                                                "\"presweeps\": 1, "
                                                "\"postsweeps\": 1, "
                                                "\"selector\": \"SIZE_2\", "
                                                "\"max_iters\": 1, "
                                                "\"coarse_solver\" : \"NOSOLVER\","
                                                "\"scope\": \"lower\", "
                                                "\"max_levels\": 3, "
                                                "\"print_grid_stats\": 1, "
                                                "\"coarsest_sweeps\": 1, "
                                                "\"cycle\": \"V\""
                                            "},"
                                            "\"print_solve_stats\": 1, "
                                            "\"presweeps\": 1, "
                                            "\"selector\": \"SIZE_2\", "
                                            "\"coarsest_sweeps\": 1, "
                                            "\"max_iters\": 50, "
                                            "\"monitor_residual\": 1, "
                                            "\"scope\": \"main\", "
                                            "\"max_levels\": 3, "
                                            "\"postsweeps\": 1, "
                                            "\"tolerance\": 1e-6, "
                                            "\"print_grid_stats\": 1, "
                                            "\"norm\": \"L1\", "
                                            "\"cycle\": \"V\""
                                        "}"
                                    "}"
                               ) == AMGX_OK);

        AMGX_STATUS solve_status = AMGX_ST_CONVERGED;
        AMG_Solver<TConfig> amg_no_nesting(&res, cfg_no_nesting);
        UNITTEST_ASSERT_EQUAL(amg_no_nesting.setup(A_hd), AMGX_OK);
        UNITTEST_ASSERT_EQUAL(amg_no_nesting.solve(b_hd, x_fin_no_nesting, solve_status), AMGX_OK);
        AMG_Solver<TConfig> amg_nesting_fine(&res, cfg_nesting_fine);
        UNITTEST_ASSERT_EQUAL(amg_nesting_fine.setup(A_hd), AMGX_OK);
        UNITTEST_ASSERT_EQUAL(amg_nesting_fine.solve(b_hd, x_fin_nesting_fine, solve_status), AMGX_OK);
        UNITTEST_ASSERT_EQUAL_TOL(x_fin_no_nesting, x_fin_nesting_fine, 1e-10);
    }
}

DECLARE_UNITTEST_END(Nested_AMG_equivalence);


// if you want to be able run this test for all available configs you can write this:
//#define AMGX_CASE_LINE(CASE) TemplateTest <TemplateMode<CASE>::Type>  TemplateTest_##CASE;
//  AMGX_FORALL_BUILDS(AMGX_CASE_LINE)
//#undef AMGX_CASE_LINE

// or run for all device configs
//#define AMGX_CASE_LINE(CASE) TemplateTest <TemplateMode<CASE>::Type>  TemplateTest_##CASE;
//  AMGX_FORALL_BUILDS_DEVICE(AMGX_CASE_LINE)
//#undef AMGX_CASE_LINE

Nested_AMG_equivalence <TemplateMode<AMGX_mode_dDDI>::Type>  Nested_AMG_equivalence_instance_mode_dDDI;
//ILU_DILU_equivalence <TemplateMode<AMGX_mode_dFFI>::Type>  ILU_DILU_equivalence_instance_mode_dFFI;

// or you can specify several desired configs
//TemplateTest <TemplateMode<AMGX_mode_hDFI>::Type>  TemplateTest_hDFI;
//TemplateTest <TemplateMode<AMGX_mode_dDFI>::Type>  TemplateTest_dDFI;


} //namespace amgx
