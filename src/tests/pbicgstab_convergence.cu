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

template <class Matrix, class Vector>
void build_pbicgstab_nonsymmetric_system(Matrix &A, Vector &b, Vector &x,
                                         bool nonzero_guess)
{
    typedef typename Matrix::value_type ValueType;
    const int num_rows = 32;
    A.set_initialized(0);
    A.addProps(CSR);
    A.resize(num_rows, num_rows, 3 * num_rows - 2);

    int nz = 0;
    A.row_offsets[0] = 0;

    for (int row = 0; row < num_rows; ++row)
    {
        if (row > 0)
        {
            A.col_indices[nz] = row - 1;
            A.values[nz++] = ValueType(-1);
        }

        A.col_indices[nz] = row;
        A.values[nz++] = ValueType(5);

        if (row + 1 < num_rows)
        {
            A.col_indices[nz] = row + 1;
            A.values[nz++] = ValueType(-2);
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
        ValueType rhs = ValueType(5);
        if (row > 0) { rhs -= ValueType(1); }
        if (row + 1 < num_rows) { rhs -= ValueType(2); }
        b[row] = rhs;
        x[row] = nonzero_guess ? ValueType(0.25) : ValueType(0);
    }
}

DECLARE_UNITTEST_BEGIN(PBiCGStabConvergence);

void check_convergence(bool use_preconditioner, bool nonzero_guess)
{
    Resources res;
    Matrix_h A;
    Vector_h b, x;
    build_pbicgstab_nonsymmetric_system(A, b, x, nonzero_guess);

    MatrixA A_device = A;
    VVector b_device = b;
    VVector x_device = x;
    VVector residual(A.get_num_rows(), ValueTypeB(0));
    residual.set_block_dimx(1);
    residual.set_block_dimy(1);

    std::stringstream parameters;
    parameters << "config_version=2, solver(main)=PBICGSTAB, ";
    if (use_preconditioner)
    {
        parameters << "main:preconditioner(jacobi)=BLOCK_JACOBI, jacobi:max_iters=1, ";
    }
    else
    {
        parameters << "main:preconditioner=NOSOLVER, ";
    }

    parameters << "main:max_iters=100, main:norm=L2, main:use_scalar_norm=1"
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
    this->PrintOnFail("PBiCGStab relative residual: %e",
                      residual_norm[0] / rhs_norm[0]);
    UNITTEST_ASSERT_TRUE(residual_norm[0] / rhs_norm[0] < ValueTypeB(1e-5));
}

void run()
{
    check_convergence(false, false);
    check_convergence(true, false);
    check_convergence(false, true);
    check_convergence(true, true);
}

DECLARE_UNITTEST_END(PBiCGStabConvergence);

PBiCGStabConvergence<TemplateMode<AMGX_mode_dDDI>::Type> PBiCGStabConvergence_dDDI;
PBiCGStabConvergence<TemplateMode<AMGX_mode_dFFI>::Type> PBiCGStabConvergence_dFFI;

DECLARE_UNITTEST_BEGIN(PBiCGStabZeroInitialResidual);

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
            "config_version=2, solver(main)=PBICGSTAB, main:preconditioner=NOSOLVER, "
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

DECLARE_UNITTEST_END(PBiCGStabZeroInitialResidual);

PBiCGStabZeroInitialResidual<TemplateMode<AMGX_mode_dDDI>::Type> PBiCGStabZeroInitialResidual_dDDI;
PBiCGStabZeroInitialResidual<TemplateMode<AMGX_mode_dFFI>::Type> PBiCGStabZeroInitialResidual_dFFI;

DECLARE_UNITTEST_BEGIN(PBiCGStabIterationLimit);

void run()
{
    Resources res;
    Matrix_h A;
    Vector_h b, x;
    build_pbicgstab_nonsymmetric_system(A, b, x, false);
    MatrixA A_device = A;
    VVector b_device = b;
    VVector x_device = x;
    AMG_Configuration cfg;
    UNITTEST_ASSERT_EQUAL(
        cfg.parseParameterString(
            "config_version=2, solver(main)=PBICGSTAB, main:preconditioner=NOSOLVER, "
            "main:max_iters=1, main:norm=L2, main:tolerance=1e-12, "
            "main:convergence=RELATIVE_INI_CORE, main:monitor_residual=1"),
        AMGX_OK);

    AMG_Solver<TConfig> solver(&res, cfg);
    AMGX_STATUS solve_status = AMGX_ST_CONVERGED;
    UNITTEST_ASSERT_EQUAL(solver.setup(A_device), AMGX_OK);
    UNITTEST_ASSERT_EQUAL(solver.solve(b_device, x_device, solve_status), AMGX_OK);
    UNITTEST_ASSERT_EQUAL(solve_status, AMGX_ST_NOT_CONVERGED);
}

DECLARE_UNITTEST_END(PBiCGStabIterationLimit);

PBiCGStabIterationLimit<TemplateMode<AMGX_mode_dDDI>::Type> PBiCGStabIterationLimit_dDDI;
PBiCGStabIterationLimit<TemplateMode<AMGX_mode_dFFI>::Type> PBiCGStabIterationLimit_dFFI;

} // namespace amgx
