// SPDX-FileCopyrightText: 2011 - 2025 NVIDIA CORPORATION. All Rights Reserved.
//
// SPDX-License-Identifier: BSD-3-Clause

#include "unit_test.h"
#include "amg_solver.h"
#include <blas.h>
#include <multiply.h>
#include <norm.h>
#include <sstream>

namespace amgx
{

DECLARE_UNITTEST_BEGIN(GMRESConvergence);

void build_nonsymmetric_system(Matrix_h &A, Vector_h &b, Vector_h &x, bool nonzero_guess)
{
    const int num_rows = 32;
    const int num_nz = 3 * num_rows - 2;
    A.set_initialized(0);
    A.addProps(CSR);
    A.resize(num_rows, num_rows, num_nz);

    int nz = 0;
    A.row_offsets[0] = 0;

    for (int row = 0; row < num_rows; ++row)
    {
        if (row > 0)
        {
            A.col_indices[nz] = row - 1;
            A.values[nz++] = ValueTypeB(-1);
        }

        A.col_indices[nz] = row;
        A.values[nz++] = ValueTypeB(5);

        if (row + 1 < num_rows)
        {
            A.col_indices[nz] = row + 1;
            A.values[nz++] = ValueTypeB(-2);
        }

        A.row_offsets[row + 1] = nz;
    }

    A.computeDiagonal();
    A.set_initialized(1);
    b.set_block_dimx(1);
    b.set_block_dimy(1);
    x.set_block_dimx(1);
    x.set_block_dimy(1);
    b.resize(num_rows);
    x.resize(num_rows);

    for (int row = 0; row < num_rows; ++row)
    {
        ValueTypeB rhs = ValueTypeB(5);
        if (row > 0) { rhs -= ValueTypeB(1); }
        if (row + 1 < num_rows) { rhs -= ValueTypeB(2); }
        b[row] = rhs;
        x[row] = nonzero_guess ? ValueTypeB(0.25) : ValueTypeB(0);
    }
}

void check_convergence(int restart, bool use_preconditioner, bool nonzero_guess)
{
    Resources res;
    Matrix_h A;
    Vector_h b, x;
    build_nonsymmetric_system(A, b, x, nonzero_guess);

    MatrixA A_device = A;
    VVector b_device = b;
    VVector x_device = x;
    VVector residual(A.get_num_rows(), ValueTypeB(0));
    residual.set_block_dimx(1);
    residual.set_block_dimy(1);

    std::stringstream parameters;
    parameters << "config_version=2, solver(main)=GMRES, ";
    if (use_preconditioner)
    {
        parameters << "main:preconditioner(jacobi)=BLOCK_JACOBI, jacobi:max_iters=1, ";
    }
    else
    {
        parameters << "main:preconditioner=NOSOLVER, ";
    }

    parameters << "main:max_iters=200, main:gmres_n_restart=" << restart
               << ", main:norm=L2, main:use_scalar_norm=1"
               << ", main:tolerance=1e-8, main:convergence=RELATIVE_INI_CORE"
               << ", main:monitor_residual=1, determinism_flag=1";

    AMG_Configuration cfg;
    UNITTEST_ASSERT_EQUAL(cfg.parseParameterString(parameters.str().c_str()), AMGX_OK);
    AMG_Solver<TConfig> solver(&res, cfg);
    AMGX_STATUS solve_status = AMGX_ST_NOT_CONVERGED;
    UNITTEST_ASSERT_EQUAL(solver.setup(A_device), AMGX_OK);
    UNITTEST_ASSERT_EQUAL(solver.solve(b_device, x_device, solve_status), AMGX_OK);
    UNITTEST_ASSERT_EQUAL(solve_status, AMGX_ST_CONVERGED);

    multiply(A_device, x_device, residual);
    axpby(b_device, residual, residual, ValueTypeB(1), ValueTypeB(-1));
    Vector_h residual_norm(1), rhs_norm(1);
    get_norm(A_device, residual, 1, L2, residual_norm);
    get_norm(A_device, b_device, 1, L2, rhs_norm);
    this->PrintOnFail("GMRES relative residual (restart %d): %e",
                      restart, residual_norm[0] / rhs_norm[0]);
    UNITTEST_ASSERT_TRUE(residual_norm[0] / rhs_norm[0] < ValueTypeB(1e-5));
}

void run()
{
    const int restarts[] = {4, 32};
    for (int restart : restarts)
    {
        check_convergence(restart, false, false);
        check_convergence(restart, true, false);
        check_convergence(restart, false, true);
        check_convergence(restart, true, true);
    }
}

DECLARE_UNITTEST_END(GMRESConvergence);

GMRESConvergence<TemplateMode<AMGX_mode_dDDI>::Type> GMRESConvergence_dDDI;
GMRESConvergence<TemplateMode<AMGX_mode_dFFI>::Type> GMRESConvergence_dFFI;

DECLARE_UNITTEST_BEGIN(GMRESZeroInitialResidual);

void check_zero_residual(int max_iters)
{
    Resources res;
    Matrix_h A;
    Vector_h b, x;
    const int num_rows = 3;

    A.set_initialized(0);
    A.addProps(CSR);
    A.resize(num_rows, num_rows, num_rows);
    A.row_offsets[0] = 0;
    A.row_offsets[1] = 1;
    A.row_offsets[2] = 2;
    A.row_offsets[3] = 3;

    for (int i = 0; i < num_rows; ++i)
    {
        A.col_indices[i] = i;
        A.values[i] = ValueTypeB(i + 2);
    }

    A.computeDiagonal();
    A.set_initialized(1);
    b.set_block_dimx(1);
    b.set_block_dimy(1);
    x.set_block_dimx(1);
    x.set_block_dimy(1);
    b.resize(num_rows);
    x.resize(num_rows);

    for (int i = 0; i < num_rows; ++i)
    {
        x[i] = ValueTypeB(i + 1);
        b[i] = A.values[i] * x[i];
    }

    MatrixA A_device = A;
    VVector b_device = b;
    VVector x_device = x;
    std::stringstream parameters;
    parameters << "config_version=2, solver(main)=GMRES, main:preconditioner=NOSOLVER"
               << ", main:max_iters=" << max_iters << ", main:gmres_n_restart=10"
               << ", main:norm=L2, main:tolerance=1e-8"
               << ", main:convergence=RELATIVE_INI_CORE, main:monitor_residual=1";

    AMG_Configuration cfg;
    UNITTEST_ASSERT_EQUAL(cfg.parseParameterString(parameters.str().c_str()), AMGX_OK);
    AMG_Solver<TConfig> solver(&res, cfg);
    AMGX_STATUS solve_status = AMGX_ST_NOT_CONVERGED;
    UNITTEST_ASSERT_EQUAL(solver.setup(A_device), AMGX_OK);
    UNITTEST_ASSERT_EQUAL(solver.solve(b_device, x_device, solve_status), AMGX_OK);
    UNITTEST_ASSERT_EQUAL(solve_status, AMGX_ST_CONVERGED);

    Vector_h x_result = x_device;
    for (int i = 0; i < num_rows; ++i)
    {
        UNITTEST_ASSERT_EQUAL(x_result[i], x[i]);
    }
}

void run()
{
    check_zero_residual(10);
    check_zero_residual(1);
}

DECLARE_UNITTEST_END(GMRESZeroInitialResidual);

GMRESZeroInitialResidual<TemplateMode<AMGX_mode_dDDI>::Type> GMRESZeroInitialResidual_dDDI;
GMRESZeroInitialResidual<TemplateMode<AMGX_mode_dFFI>::Type> GMRESZeroInitialResidual_dFFI;

DECLARE_UNITTEST_BEGIN(GMRESIterationLimit);

void run()
{
    Resources res;
    Matrix_h A;
    Vector_h b, x;
    const int num_rows = 10;

    A.set_initialized(0);
    A.addProps(CSR);
    A.resize(num_rows, num_rows, num_rows);
    A.row_offsets[0] = 0;

    for (int i = 0; i < num_rows; ++i)
    {
        A.row_offsets[i + 1] = i + 1;
        A.col_indices[i] = i;
        A.values[i] = ValueTypeB(i + 2);
    }

    A.computeDiagonal();
    A.set_initialized(1);
    b.set_block_dimx(1);
    b.set_block_dimy(1);
    x.set_block_dimx(1);
    x.set_block_dimy(1);
    b.resize(num_rows);
    x.resize(num_rows);
    thrust_wrapper::fill<AMGX_host>(b.begin(), b.end(), ValueTypeB(1));
    thrust_wrapper::fill<AMGX_host>(x.begin(), x.end(), ValueTypeB(0));

    MatrixA A_device = A;
    VVector b_device = b;
    VVector x_device = x;
    AMG_Configuration cfg;
    UNITTEST_ASSERT_EQUAL(
        cfg.parseParameterString(
            "config_version=2, solver(main)=GMRES, main:preconditioner=NOSOLVER, "
            "main:max_iters=1, main:gmres_n_restart=10, main:norm=L2, "
            "main:tolerance=1e-12, main:convergence=RELATIVE_INI_CORE, "
            "main:monitor_residual=1"),
        AMGX_OK);

    AMG_Solver<TConfig> solver(&res, cfg);
    AMGX_STATUS solve_status = AMGX_ST_CONVERGED;
    UNITTEST_ASSERT_EQUAL(solver.setup(A_device), AMGX_OK);
    UNITTEST_ASSERT_EQUAL(solver.solve(b_device, x_device, solve_status), AMGX_OK);
    UNITTEST_ASSERT_EQUAL(solve_status, AMGX_ST_NOT_CONVERGED);
}

DECLARE_UNITTEST_END(GMRESIterationLimit);

GMRESIterationLimit<TemplateMode<AMGX_mode_dDDI>::Type> GMRESIterationLimit_dDDI;
GMRESIterationLimit<TemplateMode<AMGX_mode_dFFI>::Type> GMRESIterationLimit_dFFI;

} // namespace amgx
