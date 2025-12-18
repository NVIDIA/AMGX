// SPDX-FileCopyrightText: 2011 - 2025 NVIDIA CORPORATION. All Rights Reserved.
//
// SPDX-License-Identifier: BSD-3-Clause

#include "unit_test.h"
#include "test_utils.h"
#include "util.h"
#include "time.h"
#include "amg_solver.h"
#include "solvers/solver.h"

namespace amgx
{

DECLARE_UNITTEST_BEGIN(SmootherNaNRandom);

void check_nan_block_smoothers_random(int bsize, int rows, bool diag, std::string smoother_string, int n_smoothing_steps)
{
    typedef TemplateConfig<AMGX_host, TConfig::vecPrec, TConfig::matPrec, TConfig::indPrec> TConfig_h;
    Vector<TConfig_h> x_h;
    Matrix<TConfig> A;
    Vector<TConfig> b;
    Vector<TConfig> x;
    this->randomize( 4321 );
    generateMatrixRandomStruct<TConfig>::generate( A, rows, diag, bsize, false );
    random_fill(A);
    int n_rows = A.get_num_rows() * bsize;
    // Fill b with ones
    b.resize(n_rows);
    thrust_wrapper::fill<TConfig::memSpace>(b.begin(), b.end(), 1);
    b.set_block_dimy(A.get_block_dimy());
    // -----------------------------------------
    // Initialize x to zeros
    x.resize(n_rows, 0.);
    AMG_Config cfg;
    std::string parameter_string = "solver=" + smoother_string + ", determinism_flag=1, smoother_weight=1.0, matrix_coloring_scheme=MIN_MAX, max_uncolored_percentage=0.0, coloring_level=1";
    UNITTEST_ASSERT_TRUE( cfg.parseParameterString(const_cast<char *>(parameter_string.c_str())) == AMGX_OK);
    this->PrintOnFail("config creation");
    Solver<TConfig> *smoother = SolverFactory<T_Config>::allocate(cfg, "default", "solver");
    this->PrintOnFail("smoother creation");
    UNITTEST_ASSERT_TRUE(smoother != NULL);
    A.set_initialized(1);
    smoother->setup(A, false);
    smoother->set_max_iters(n_smoothing_steps);
    smoother->solve( b, x, false );
    x_h = x;

    for (int i = 0; i < x_h.size(); i++)
        //if (x_h[i] != x_h[i]){
        //printf("%s, %d, %f\n",smoother_string.c_str(), i, x_h[i]);
    {
        UNITTEST_ASSERT_TRUE(x_h[i] == x_h[i]);
    }

    //}
}

void run()
{
    int n_smoothing_steps = 25;
    check_nan_block_smoothers_random(4, 10, false, "MULTICOLOR_DILU", n_smoothing_steps);
    check_nan_block_smoothers_random(4, 10, false, "MULTICOLOR_GS", n_smoothing_steps);
    check_nan_block_smoothers_random(4, 10, false, "BLOCK_JACOBI", n_smoothing_steps) ;
    check_nan_block_smoothers_random(1, 10, false, "BLOCK_JACOBI", n_smoothing_steps) ;
    check_nan_block_smoothers_random(1, 10, false, "MULTICOLOR_DILU", n_smoothing_steps) ;
    check_nan_block_smoothers_random(1, 10, false, "MULTICOLOR_GS", n_smoothing_steps) ;
}

DECLARE_UNITTEST_END(SmootherNaNRandom);

SmootherNaNRandom <TemplateMode<AMGX_mode_dDDI>::Type>  SmootherNaNRandom_instance_mode_dDDI;
SmootherNaNRandom <TemplateMode<AMGX_mode_dDFI>::Type>  SmootherNaNRandom_instance_mode_dDFI;
SmootherNaNRandom <TemplateMode<AMGX_mode_dFFI>::Type>  SmootherNaNRandom_instance_mode_dFFI;

}
