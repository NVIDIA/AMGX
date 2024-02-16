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
DECLARE_UNITTEST_BEGIN(NestedSolvers);

void run_case(Resources *res, MatrixA &A_hd, VVector &b_hd, VVector &x_hd, const std::string &parameter_string, AMGX_ERROR &err)
{
    // this is incorrect resources/obects usage, might generate memory leaks, but for the sake of simplicity let it be
    AMG_Configuration cfg;
    UNITTEST_ASSERT_TRUE(cfg.parseParameterString(parameter_string.c_str()) == AMGX_OK);
    AMG_Solver<TConfig> amg(res, cfg);
    err = amg.setup(A_hd);
}


void run()
{
    int points = 27;
    int nx, ny, nz;
    nx = ny = nz = 20;
    // Create system
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
    b.set_block_dimx(1);
    b.set_block_dimy(bsize);
    x.set_block_dimy(1);
    x.set_block_dimx(bsize);
    // Fill b
    b.resize(n_rows);
    thrust_wrapper::fill<AMGX_host>(b.begin(), b.end(), 1.);
    // Initialize x
    x.resize(n_rows);
    thrust_wrapper::fill<AMGX_host>(x.begin(), x.end(), 0.);
    // Copy to device if necessary
    MatrixA A_hd;
    VVector x_ini_hd, x_fin_inside_hd, x_fin_outside_hd, b_hd, r_hd;
    r_hd.resize(n_rows, 0.);
    r_hd.set_block_dimy(1);
    r_hd.set_block_dimx(bsize);
    A_hd = A;
    b_hd = b;
    x_ini_hd = x;
    AMGX_ERROR err;
    // Set parameters
    std::string error_string;
    std::stringstream parameter_string;
    Resources res;        // default resources
    // Should work fine
    parameter_string << "config_version=2, solver(s1)=FGMRES, s1:preconditioner(jacobi)=BLOCK_JACOBI, jacobi:max_iters=1, s1:max_iters=" << n_rows << ",s1:norm=L2, determinism_flag=1, s1:tolerance=1e-14, s1:gmres_n_restart=" << n_rows << ", s1:convergence=RELATIVE_INI_CORE, s1:monitor_residual=1";
    run_case(&res, A_hd, b_hd, x_ini_hd, parameter_string.str(), err);
    UNITTEST_ASSERT_TRUE(err == AMGX_OK);
    parameter_string.str("");
    parameter_string << " solver=FGMRES, preconditioner=FGMRES, max_iters=100";
    UNITTEST_ASSERT_EXCEPTION_START;
    run_case(&res, A_hd, b_hd, x_ini_hd, parameter_string.str(), err);
    UNITTEST_ASSERT_EXCEPTION_END_AMGX_ERR(AMGX_ERR_BAD_PARAMETERS);
    parameter_string.str("");
    parameter_string << " solver=AMG, smoother=FGMRES, max_iters=100";
    UNITTEST_ASSERT_EXCEPTION_START;
    run_case(&res, A_hd, b_hd, x_ini_hd, parameter_string.str(), err);
    UNITTEST_ASSERT_EXCEPTION_END_AMGX_ERR(AMGX_ERR_BAD_PARAMETERS);
    parameter_string.str("");
    parameter_string << " solver=AMG, coarse_solver=AMG, max_iters=100";
    UNITTEST_ASSERT_EXCEPTION_START
    run_case(&res, A_hd, b_hd, x_ini_hd, parameter_string.str(), err);
    UNITTEST_ASSERT_EXCEPTION_END_AMGX_ERR(AMGX_ERR_BAD_PARAMETERS);
}

DECLARE_UNITTEST_END(NestedSolvers);

NestedSolvers <TemplateMode<AMGX_mode_dDDI>::Type>  NestedSolvers_instance_mode_dDDI;


} //namespace amgx
