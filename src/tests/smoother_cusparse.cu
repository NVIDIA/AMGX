// SPDX-FileCopyrightText: 2011 - 2024 NVIDIA CORPORATION. All Rights Reserved.
//
// SPDX-License-Identifier: BSD-3-Clause

#include "unit_test.h"
#include "test_utils.h"
#include "util.h"
#include "time.h"
#include "amg_solver.h"
#include "solvers/solver.h"

// runs smoothers on a matrix with use_bsrxmv=1 and 0. result should be ~the same

namespace amgx
{
DECLARE_UNITTEST_BEGIN(SmootherCusparse);
void check_block_smoothers_random(int bsize, int rows, bool diag, std::string smoother_string, int n_smoothing_steps, bool &done)
{
    done = false;
    typedef TemplateConfig<AMGX_host, TConfig::vecPrec, TConfig::matPrec, TConfig::indPrec> TConfig_h;
    Vector<TConfig_h> x_h;
    Matrix<TConfig> A, A2;
    Vector<TConfig> b, b2;
    Vector<TConfig> x, x2;
    this->randomize( 4321 );
    generateMatrixRandomStruct<TConfig>::generate( A, rows, diag, bsize, false );
    random_fill(A);
    A2 = A;
    int n_rows = A.get_num_rows() * bsize;
    // Fill b with ones
    b.resize(n_rows);
    thrust_wrapper::fill<TConfig::memSpace>(b.begin(), b.end(), 1);
    b.set_block_dimy(A.get_block_dimy());
    b2.resize(n_rows);
    thrust_wrapper::fill<TConfig::memSpace>(b2.begin(), b2.end(), 1);
    b2.set_block_dimy(A.get_block_dimy());
    // -----------------------------------------
    // Initialize x to zeros
    x.resize(n_rows, 0.);
    x2.resize(n_rows, 0.);
    AMG_Config cfg;
    std::string parameter_string2 = "solver=" + smoother_string + ", determinism_flag=1, smoother_weight=1.0, matrix_coloring_scheme=MIN_MAX, coloring_level=1, max_uncolored_percentage=0.0";
    UNITTEST_ASSERT_TRUE( cfg.parseParameterString(const_cast<char *>(parameter_string2.c_str())) == AMGX_OK);;
    const std::string &cfg_scope = "default";
    this->PrintOnFail("config creation");
    Solver<TConfig> *smoother = SolverFactory<T_Config>::allocate(cfg, cfg_scope, "solver");
    this->PrintOnFail("smoother creation");
    UNITTEST_ASSERT_TRUE(smoother != NULL);
    smoother->setup(A, false);
    A.set_initialized(1);
    smoother->set_max_iters(n_smoothing_steps);
    smoother->solve( b, x, false );
    AMG_Config cfg2;
    std::string parameter_string = "solver=" + smoother_string + ", determinism_flag=1, smoother_weight=1.0, matrix_coloring_scheme=MIN_MAX, max_uncolored_percentage=0.0, reorder_cols_by_color=1, coloring_level=1, use_bsrxmv=1";
    UNITTEST_ASSERT_TRUE(cfg2.parseParameterString(const_cast<char *>(parameter_string.c_str())) == AMGX_OK);
    this->PrintOnFail("config creation");
    Solver<TConfig> *smoother2 = SolverFactory<T_Config>::allocate(cfg2, cfg_scope, "solver");
    this->PrintOnFail("smoother creation");
    UNITTEST_ASSERT_TRUE(smoother2 != NULL);
    smoother2->setup(A2, false);
    A2.set_initialized(1);
    smoother2->set_max_iters(n_smoothing_steps);
    smoother2->solve( b, x2, false );
    char buf[60];
    sprintf(buf, "Smoother %s bs %d result inequality", smoother_string.c_str(), bsize);
    UNITTEST_ASSERT_EQUAL_TOL_DESC(buf, x, x2, 0.1);
    done = true;
}

void run()
{
    std::stringstream err_msg;
    int n_smoothing_steps = 5;
    bool done;
    err_msg << "Block jacobi";
    check_block_smoothers_random(4, 10, false, "BLOCK_JACOBI", n_smoothing_steps, done) ;
    check_block_smoothers_random(1, 10, false, "BLOCK_JACOBI", n_smoothing_steps, done) ;
    UNITTEST_ASSERT_TRUE_DESC(err_msg.str().c_str(), done);
    err_msg << "multicolor gs";
    check_block_smoothers_random(4, 10, false, "MULTICOLOR_GS", n_smoothing_steps, done);
    check_block_smoothers_random(1, 10, false, "MULTICOLOR_GS", n_smoothing_steps, done);
    UNITTEST_ASSERT_TRUE_DESC(err_msg.str().c_str(), done);
    err_msg << "multicolro dilu";
    check_block_smoothers_random(4, 10, false, "MULTICOLOR_DILU", n_smoothing_steps, done) ;
    check_block_smoothers_random(1, 10, false, "MULTICOLOR_DILU", n_smoothing_steps, done) ;
    UNITTEST_ASSERT_TRUE_DESC(err_msg.str().c_str(), done);
}

DECLARE_UNITTEST_END(SmootherCusparse);

SmootherCusparse <TemplateMode<AMGX_mode_dDDI>::Type>  SmootherCusparse_instance_mode_dDDI;
SmootherCusparse <TemplateMode<AMGX_mode_dDFI>::Type>  SmootherCusparse_instance_mode_dDFI;
SmootherCusparse <TemplateMode<AMGX_mode_dFFI>::Type>  SmootherCusparse_instance_mode_dFFI;

}
