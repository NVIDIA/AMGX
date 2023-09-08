/* Copyright (c) 2011-2017, NVIDIA CORPORATION. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *  * Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *  * Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *  * Neither the name of NVIDIA CORPORATION nor the names of its
 *    contributors may be used to endorse or promote products derived
 *    from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
 * EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 * PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
 * CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 * EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 * PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
 * OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

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
