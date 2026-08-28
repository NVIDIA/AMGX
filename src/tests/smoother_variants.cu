// SPDX-FileCopyrightText: 2011 - 2025 NVIDIA CORPORATION. All Rights Reserved.
//
// SPDX-License-Identifier: BSD-3-Clause

#include "unit_test.h"
#include "amg_solver.h"
#include "solvers/solver.h"
#include <matrix_io.h>
#include <multiply.h>
#include <blas.h>
#include <norm.h>

#include <cmath>
#include <string>

namespace amgx
{

DECLARE_UNITTEST_BEGIN(SmootherVariants);

void check_smoother(const std::string &solver_name, const std::string &extra_config)
{
    typedef TemplateConfig<AMGX_host, TConfig::vecPrec, TConfig::matPrec, TConfig::indPrec> TConfig_h;
    typedef Matrix<TConfig_h> Matrix_h;
    typedef Vector<TConfig_h> Vector_h;
    typedef Matrix<TConfig> Matrix_d;
    typedef Vector<TConfig> Vector_d;

    Matrix_h A_h;
    A_h.addProps(CSR);
    A_h.set_initialized(0);
    MatrixCusp<TConfig_h, cusp::csr_format> cusp_A(&A_h);
    cusp::gallery::poisson5pt(cusp_A, 16, 16);
    A_h.computeDiagonal();
    A_h.set_initialized(1);

    Matrix_d A = A_h;
    const int rows = A_h.get_num_rows();
    Vector_h b_h(rows, ValueTypeB(1));
    Vector_h zero_h(rows, ValueTypeB(0));
    b_h.set_block_dimx(1);
    b_h.set_block_dimy(1);
    zero_h.set_block_dimx(1);
    zero_h.set_block_dimy(1);

    Vector_d b = b_h;
    Vector_d x = zero_h;
    Vector_d residual(rows, ValueTypeB(0));
    residual.set_block_dimx(1);
    residual.set_block_dimy(1);

    AMG_Config cfg;
    const std::string parameters =
        "solver=" + solver_name +
        ",max_iters=4,monitor_residual=0,determinism_flag=1" + extra_config;
    cfg.parseParameterString(parameters.c_str());

    Solver<TConfig> *smoother = SolverFactory<TConfig>::allocate(cfg, "default", "solver");
    UNITTEST_ASSERT_TRUE_DESC((solver_name + " factory allocation failed").c_str(), smoother != NULL);
    smoother->setup(A, false);
    smoother->solve(b, x, false);

    Vector_h initial_norm(1), final_norm(1);
    get_norm(A, b, 1, L2, initial_norm);
    multiply(A, x, residual);
    axpby(b, residual, residual, ValueTypeB(1), ValueTypeB(-1));
    get_norm(A, residual, 1, L2, final_norm);

    const std::string residual_message = solver_name + " did not reduce the residual in four iterations";
    UNITTEST_ASSERT_TRUE_DESC(residual_message.c_str(),
                              std::isfinite(static_cast<double>(final_norm[0])) &&
                              final_norm[0] < initial_norm[0]);

    // A zero right-hand side and zero initial guess is a useful guard against
    // divisions by zero in spectral estimates and relaxation updates.
    Vector_d zero_b = zero_h;
    Vector_d zero_x = zero_h;
    smoother->solve(zero_b, zero_x, false);
    Vector_h zero_x_h = zero_x;

    for (int i = 0; i < rows; ++i)
    {
        const double value = static_cast<double>(zero_x_h[i]);
        UNITTEST_ASSERT_TRUE_DESC((solver_name + " changed the homogeneous zero solution").c_str(),
                                  std::isfinite(value) && value == 0.0);
    }

    delete smoother;
}

void run()
{
    check_smoother("BLOCK_JACOBI", ",smoother_weight=0.8");
    check_smoother("JACOBI_L1", ",smoother_weight=0.8");
    check_smoother("CHEBYSHEV",
                   ",preconditioner=NOSOLVER,chebyshev_polynomial_order=2,chebyshev_lambda_estimate_mode=2");
}

DECLARE_UNITTEST_END(SmootherVariants);

SmootherVariants<TemplateMode<AMGX_mode_dDDI>::Type> SmootherVariants_instance_mode_dDDI;
SmootherVariants<TemplateMode<AMGX_mode_dFFI>::Type> SmootherVariants_instance_mode_dFFI;

} // namespace amgx
