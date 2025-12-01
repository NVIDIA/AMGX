// SPDX-FileCopyrightText: 2011 - 2025 NVIDIA CORPORATION. All Rights Reserved.
//
// SPDX-License-Identifier: BSD-3-Clause

#include "unit_test.h"
#include "amg_solver.h"
#include "solvers/multicolor_dilu_solver.h"
#include <matrix_io.h>
#include "test_utils.h"
#include <multiply.h>
#include<blas.h>
#include <csr_multiply.h>
#include "util.h"
#include "time.h"

namespace amgx
{

// parameter is used as test name
DECLARE_UNITTEST_BEGIN(ScalarSmootherPoisson);

void check_scalar_smoothers_poisson(int points, int nx, int ny, int nz, std::string smoother_string, ValueTypeB final_residual_tolerance, int n_smoothing_steps, bool diag_comparison)
{
    std::string error_string;
    typedef TemplateConfig<AMGX_host, TConfig::vecPrec, TConfig::matPrec, TConfig::indPrec> TConfig_h;
    typedef Matrix<TConfig_h> Matrix_h;
    typedef Matrix<TConfig> Matrix_hd;
    typedef Vector<TConfig> Vector_hd;
    Matrix_h A;
    Vector_h b;
    Vector_h x;
    A.set_initialized(0);
    A.addProps(CSR);
    MatrixCusp<TConfig_h, cusp::csr_format> wA(&A);

    switch (points)
    {
        case 5:
            cusp::gallery::poisson5pt(wA, nx, ny);
            break;

        case 7:
            cusp::gallery::poisson7pt(wA, nx, ny, nz);
            break;

        case 9:
            cusp::gallery::poisson9pt(wA, nx, ny);
            break;

        case 27:
            cusp::gallery::poisson27pt(wA, nx, ny, nz);
            break;

        default:
            printf("Error invalid number of poisson points specified, valid numbers are 5, 7, 9, 27\n");
    }

    A.computeDiagonal();
    A.set_initialized(1);
    int bsize = A.get_block_dimy();
    int n_rows = A.get_num_rows() * bsize;
    // Fill b with ones
    b.resize(n_rows);
    thrust_wrapper::fill<AMGX_host>(b.begin(), b.end(), 1);
    b.set_block_dimx(1);
    b.set_block_dimy(A.get_block_dimy());
    x.set_block_dimx(1);
    x.set_block_dimy(A.get_block_dimy());
    // -----------------------------------------

    // Initialize x to zeros
    if (x.size() == 0)
    {
        x.resize(n_rows, 0.);
    }

    // Copy to device if necessary
    Matrix_hd A_hd;
    Vector_hd x_ini_hd, x_fin_inside_hd, x_fin_outside_hd, b_hd, r_hd;
    r_hd.resize(n_rows, 0.);
    r_hd.set_block_dimx(1);
    r_hd.set_block_dimy(A.get_block_dimy());
    x_fin_inside_hd.set_block_dimx(1);
    x_fin_inside_hd.set_block_dimy(A.get_block_dimy());
    x_fin_outside_hd.set_block_dimx(1);
    x_fin_outside_hd.set_block_dimy(A.get_block_dimy());
    A_hd = A;
    b_hd = b;
    x_ini_hd = x;
    // Set parameters
    AMG_Config cfg;

    if (smoother_string == "BLOCK_JACOBI")
    {
        cfg.parseParameterString("determinism_flag=1, smoother=BLOCK_JACOBI, coloring_level=1, matrix_coloring_scheme=MIN_MAX, max_uncolored_percentage=0.0, smoother_weight=1.0");
    }
    else if (smoother_string == "MULTICOLOR_GS")
    {
        cfg.parseParameterString("determinism_flag=1, smoother=MULTICOLOR_GS, coloring_level=1, matrix_coloring_scheme=MIN_MAX, max_uncolored_percentage=0.0, smoother_weight=1.0,symmetric_GS=1");
    }
    else if (smoother_string == "MULTICOLOR_DILU")
    {
        cfg.parseParameterString("determinism_flag=1, smoother=MULTICOLOR_DILU, coloring_level=1, matrix_coloring_scheme=MIN_MAX, max_uncolored_percentage=0.0, smoother_weight=1.0");
    }

    const std::string cfg_scope = "default";
    // Color the matrix
    /*
    MatrixColoring<TConfig>* matrix_coloring_scheme;
    if (smoother_string=="MULTICOLOR_GS" || smoother_string=="MULTICOLOR_DILU")
    {
      matrix_coloring_scheme = MatrixColoringFactory<T_Config>::allocate(cfg,cfg_scope,A_hd);
      this->PrintOnFail("matrixColoring creation");
      UNITTEST_ASSERT_TRUE(matrix_coloring_scheme != NULL);
    }
    */
    Solver<TConfig> *smoother = SolverFactory<T_Config>::allocate(cfg, cfg_scope, "smoother");
    this->PrintOnFail("smoother creation");
    UNITTEST_ASSERT_TRUE(smoother != NULL);
    Vector_h nrm, nrm_ini;
    nrm.resize(bsize);
    nrm_ini.resize(bsize);
    // -----------------------------------------------
    // Test 1: Compare diag inside vs outside result
    // -----------------------------------------------
    //std::cout << setprecision(12) << x_ini_hd[0] << std::endl;
    x_fin_inside_hd = x_ini_hd;
    smoother->setup(A_hd, false);
    // Smooth with diag inside
    smoother->set_max_iters(n_smoothing_steps);
    smoother->solve(b_hd, x_fin_inside_hd, false);

    if (diag_comparison)
    {
        // Create matrix with diagonal stored outside
        Matrix_h Adiag;
        Adiag.convert(A, DIAG | CSR, bsize, bsize);
        A_hd = Adiag;
        // Reinitilize the smoother
        smoother = SolverFactory<T_Config>::allocate(cfg, cfg_scope, "smoother");
        smoother->setup(A_hd, false);
        x_fin_outside_hd = x_ini_hd;
        // Smooth with diag inside
        smoother->set_max_iters(n_smoothing_steps);
        smoother->solve(b_hd, x_fin_outside_hd, false);
        // Assert that result is the same wether diag is inside or outside
        error_string = "Difference between diag inside and diag outside, smoother=" + smoother_string;
        UNITTEST_ASSERT_EQUAL_TOL_DESC(error_string.c_str(), x_fin_inside_hd, x_fin_outside_hd, 1e-5);
    }

    // ----------------------------------------------------------------------------
    // Test 2: Check that final residual is finite and less than initial residual
    // ----------------------------------------------------------------------------
    // Compute the initial residual norm
    multiply( A_hd, x_ini_hd, r_hd );
    axpby( b_hd, r_hd, r_hd, ValueTypeB( 1 ), ValueTypeB( -1 ) );
    get_norm( A_hd, r_hd, 1, L2, nrm_ini );
    // Compute the residual norm
    multiply( A_hd, x_fin_inside_hd, r_hd );
    axpby( b_hd, r_hd, r_hd, ValueTypeB( 1 ), ValueTypeB( -1 ) );
    get_norm( A_hd, r_hd, 1, L2, nrm );
    error_string = "nrm >= nrm_ini, smoother=" + smoother_string;
    UNITTEST_ASSERT_TRUE_DESC(error_string.c_str(), nrm[0] < nrm_ini[0]);
    error_string = "nrm >= final_residual_tolerance, smoother=" + smoother_string;
    UNITTEST_ASSERT_TRUE_DESC(error_string.c_str(), nrm[0] < final_residual_tolerance);
    delete smoother;
    //if (smoother_string=="MULTICOLOR_GS" || smoother_string=="MULTICOLOR_DILU") {
    //  delete matrix_coloring_scheme;
    //}
}


void run()
{
    int n_smoothing_steps = 1000;
    ValueTypeB final_residual_tol = 1e-5;
    //TODO: Test fails because multicolor GS fails on matrices with diagonal stored separately
    // Works
    check_scalar_smoothers_poisson(9, 10, 10, 10, "BLOCK_JACOBI", final_residual_tol, n_smoothing_steps, 1);
    // Works
    check_scalar_smoothers_poisson(9, 10, 10, 10, "MULTICOLOR_DILU", final_residual_tol, n_smoothing_steps, 1);
    // Doesnt work
    check_scalar_smoothers_poisson(9, 10, 10, 10, "MULTICOLOR_GS", final_residual_tol, n_smoothing_steps, 1);
}

DECLARE_UNITTEST_END(ScalarSmootherPoisson);


// if you want to be able run this test for all available configs you can write this:
//#define AMGX_CASE_LINE(CASE) TemplateTest <TemplateMode<CASE>::Type>  TemplateTest_##CASE;
//  AMGX_FORALL_BUILDS(AMGX_CASE_LINE)
//#undef AMGX_CASE_LINE

// or run for all device configs
//#define AMGX_CASE_LINE(CASE) TemplateTest <TemplateMode<CASE>::Type>  TemplateTest_##CASE;
//  AMGX_FORALL_BUILDS_DEVICE(AMGX_CASE_LINE)
//#undef AMGX_CASE_LINE

ScalarSmootherPoisson <TemplateMode<AMGX_mode_dDDI>::Type>  ScalarSmootherPoisson_instance_mode_dDDI;
ScalarSmootherPoisson <TemplateMode<AMGX_mode_dFFI>::Type>  ScalarSmootherPoisson_instance_mode_dFFI;

// or you can specify several desired configs
//TemplateTest <TemplateMode<AMGX_mode_hDFI>::Type>  TemplateTest_hDFI;
//TemplateTest <TemplateMode<AMGX_mode_dDFI>::Type>  TemplateTest_dDFI;


} //namespace amgx
