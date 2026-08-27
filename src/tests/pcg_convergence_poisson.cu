// SPDX-FileCopyrightText: 2011 - 2025 NVIDIA CORPORATION. All Rights Reserved.
//
// SPDX-License-Identifier: BSD-3-Clause

#include "unit_test.h"
#include "amg_solver.h"
#include "test_utils.h"
#include <blas.h>
#include <matrix_io.h>
#include <multiply.h>
#include <norm.h>
#include <sstream>

namespace amgx
{

DECLARE_UNITTEST_BEGIN(PCGConvergencePoisson);

void check_convergence(int points, bool use_preconditioner, bool use_nonzero_initial_guess)
{
    const int size = 5;
    Resources res;
    Matrix_h A;
    Vector_h b, x;

    A.set_initialized(0);
    A.addProps(CSR);
    MatrixCusp<TConfig_h, cusp::csr_format> wrapped_A(&A);

    switch (points)
    {
        case 5:  cusp::gallery::poisson5pt(wrapped_A, size, size); break;
        case 7:  cusp::gallery::poisson7pt(wrapped_A, size, size, size); break;
        case 9:  cusp::gallery::poisson9pt(wrapped_A, size, size); break;
        case 27: cusp::gallery::poisson27pt(wrapped_A, size, size, size); break;
        default:
            UNITTEST_ASSERT_TRUE_DESC("Unsupported Poisson stencil", false);
            return;
    }

    A.computeDiagonal();
    A.set_initialized(1);

    const int num_rows = A.get_num_rows();
    b.set_block_dimx(1);
    b.set_block_dimy(1);
    x.set_block_dimx(1);
    x.set_block_dimy(1);
    b.resize(num_rows);
    x.resize(num_rows);
    thrust_wrapper::fill<AMGX_host>(b.begin(), b.end(), ValueTypeB(1));
    thrust_wrapper::fill<AMGX_host>(x.begin(), x.end(),
                                    use_nonzero_initial_guess ? ValueTypeB(0.25) : ValueTypeB(0));

    MatrixA A_device = A;
    VVector b_device = b;
    VVector x_device = x;
    VVector residual(num_rows, ValueTypeB(0));
    residual.set_block_dimx(1);
    residual.set_block_dimy(1);

    std::stringstream parameters;
    parameters << "config_version=2, solver(main)=PCG, ";

    if (use_preconditioner)
    {
        parameters << "main:preconditioner(jacobi)=BLOCK_JACOBI, jacobi:max_iters=1, ";
    }
    else
    {
        parameters << "main:preconditioner=NOSOLVER, ";
    }

    parameters << "main:max_iters=" << num_rows
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
    this->PrintOnFail("PCG relative residual for %d-point stencil: %e",
                      points, residual_norm[0] / rhs_norm[0]);
    UNITTEST_ASSERT_TRUE(residual_norm[0] / rhs_norm[0] < ValueTypeB(1e-5));
}

void run()
{
    const int stencils[] = {5, 7, 9, 27};

    for (int points : stencils)
    {
        check_convergence(points, false, false);
        check_convergence(points, true, false);
    }

    check_convergence(5, false, true);
    check_convergence(5, true, true);
    check_convergence(27, false, true);
    check_convergence(27, true, true);
}

DECLARE_UNITTEST_END(PCGConvergencePoisson);

PCGConvergencePoisson<TemplateMode<AMGX_mode_dDDI>::Type> PCGConvergencePoisson_dDDI;
PCGConvergencePoisson<TemplateMode<AMGX_mode_dFFI>::Type> PCGConvergencePoisson_dFFI;

DECLARE_UNITTEST_BEGIN(PCGZeroInitialResidual);

void run()
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
    AMG_Configuration cfg;
    UNITTEST_ASSERT_EQUAL(
        cfg.parseParameterString(
            "config_version=2, solver(main)=PCG, main:preconditioner=NOSOLVER, "
            "main:max_iters=10, main:norm=L2, main:tolerance=1e-8, "
            "main:convergence=RELATIVE_INI_CORE, main:monitor_residual=1"),
        AMGX_OK);

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

DECLARE_UNITTEST_END(PCGZeroInitialResidual);

PCGZeroInitialResidual<TemplateMode<AMGX_mode_dDDI>::Type> PCGZeroInitialResidual_dDDI;
PCGZeroInitialResidual<TemplateMode<AMGX_mode_dFFI>::Type> PCGZeroInitialResidual_dFFI;

DECLARE_UNITTEST_BEGIN(PCGIterationLimit);

void run()
{
    Resources res;
    Matrix_h A;
    Vector_h b, x;
    const int size = 10;

    A.set_initialized(0);
    A.addProps(CSR);
    MatrixCusp<TConfig_h, cusp::csr_format> wrapped_A(&A);
    cusp::gallery::poisson5pt(wrapped_A, size, size);
    A.computeDiagonal();
    A.set_initialized(1);

    const int num_rows = A.get_num_rows();
    b.set_block_dimx(1);
    b.set_block_dimy(1);
    x.set_block_dimx(1);
    x.set_block_dimy(1);
    b.resize(num_rows);
    x.resize(num_rows);

    for (int i = 0; i < num_rows; ++i)
    {
        b[i] = ValueTypeB((i % 7) + 1);
        x[i] = ValueTypeB(0);
    }

    MatrixA A_device = A;
    VVector b_device = b;
    VVector x_device = x;
    AMG_Configuration cfg;
    UNITTEST_ASSERT_EQUAL(
        cfg.parseParameterString(
            "config_version=2, solver(main)=PCG, main:preconditioner=NOSOLVER, "
            "main:max_iters=1, main:norm=L2, main:tolerance=1e-12, "
            "main:convergence=RELATIVE_INI_CORE, main:monitor_residual=1"),
        AMGX_OK);

    AMG_Solver<TConfig> solver(&res, cfg);
    AMGX_STATUS solve_status = AMGX_ST_CONVERGED;
    UNITTEST_ASSERT_EQUAL(solver.setup(A_device), AMGX_OK);
    UNITTEST_ASSERT_EQUAL(solver.solve(b_device, x_device, solve_status), AMGX_OK);
    UNITTEST_ASSERT_EQUAL(solve_status, AMGX_ST_NOT_CONVERGED);
}

DECLARE_UNITTEST_END(PCGIterationLimit);

PCGIterationLimit<TemplateMode<AMGX_mode_dDDI>::Type> PCGIterationLimit_dDDI;
PCGIterationLimit<TemplateMode<AMGX_mode_dFFI>::Type> PCGIterationLimit_dFFI;

} // namespace amgx
