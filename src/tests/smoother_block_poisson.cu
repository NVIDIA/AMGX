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
#include "amg_solver.h"
#include "solvers/multicolor_dilu_solver.h"
#include <matrix_io.h>
#include "test_utils.h"
#include <multiply.h>
#include <blas.h>
#include <csr_multiply.h>
#include "util.h"
#include "time.h"

namespace amgx
{

// This tests run same smoother for the matrix and matrix converted to the some blocksize. Both DIAG properties will be tested. Residuall should be ~the same.

// parameter is used as test name
DECLARE_UNITTEST_BEGIN(SmootherBlockPoissonTest);

void check_block_smoothers_poisson(int bsize, int points, int nx, int ny, int nz, string smoother_string, ValueTypeB final_residual_tolerance, int n_smoothing_steps, bool diag)
{
    string error_string;
    typedef TemplateConfig<AMGX_host, TConfig::vecPrec, TConfig::matPrec, TConfig::indPrec> TConfig_h;
    typedef Matrix<TConfig_h> Matrix_h;
    typedef Matrix<TConfig> Matrix_hd;
    typedef Vector<TConfig> Vector_hd;
    Matrix_h scalarA;
    Vector_h scalarb;
    Vector_h scalarx, blockx;
    scalarA.set_initialized(0);
    scalarA.addProps(CSR);
    MatrixCusp<TConfig_h, cusp::csr_format> wA(&scalarA);

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

    scalarA.computeDiagonal();
    scalarA.set_initialized(1);
    // Fill b with ones
    scalarb.resize(scalarA.get_num_rows());
    scalarb.set_block_dimy(scalarA.get_block_dimy());
    thrust_wrapper::fill<AMGX_host>(scalarb.begin(), scalarb.end(), 1);
    // Initialize x to zeros
    scalarx.resize(scalarA.get_num_rows(), 0.);
    // Copy to device if necessary
    Matrix_hd A_hd;
    Vector_hd x_ini_hd, b_hd;
    A_hd = scalarA;
    b_hd = scalarb;
    x_ini_hd = scalarx;
    // Set parameters
    AMG_Config cfg;
    string parameter_string = "solver=" + smoother_string + ", determinism_flag=1, coloring_level=1, matrix_coloring_scheme=MIN_MAX, max_uncolored_percentage=0.15";
    cfg.parseParameterString(const_cast<char *>(parameter_string.c_str()));
    const std::string &cfg_scope = "default";
    // Color the matrix
    //MatrixColoring<TConfig>* matrix_coloring_scheme = MatrixColoringFactory<T_Config>::allocate(cfg,cfg_scope,A_hd);
    //this->PrintOnFail("matrixColoring creation");
    Solver<TConfig> *scalar_smoother = SolverFactory<T_Config>::allocate(cfg, cfg_scope, "solver");
    this->PrintOnFail("scalar smoother creation");
    UNITTEST_ASSERT_TRUE(scalar_smoother != NULL);
    //UNITTEST_ASSERT_TRUE(matrix_coloring_scheme != NULL);
    // -----------------------------------------------
    // Test 1: Compare block vs scalar poisson result
    // -----------------------------------------------
    // smooth scalar with diag inside
    scalar_smoother->setup(A_hd, false);
    scalar_smoother->set_max_iters(n_smoothing_steps);
    scalar_smoother->solve( b_hd, x_ini_hd, false );
    // convert to block matrix
    Matrix_h A;
    A.convert(scalarA, CSR | ( diag ? DIAG : 0 ), bsize, bsize);
    A_hd = A;
    Solver<TConfig> *block_smoother = SolverFactory<T_Config>::allocate(cfg, cfg_scope, "solver");
    this->PrintOnFail("block smoother creation");
    UNITTEST_ASSERT_TRUE(block_smoother != NULL);
    blockx.resize(A.get_num_rows() * A.get_block_dimy(), 0.);
    Vector_hd x_fin_hd = blockx;
    Vector_h b;
    b.resize(A.get_num_rows() * A.get_block_dimy());
    thrust_wrapper::fill<AMGX_host>(b.begin(), b.end(), 1);
    b.set_block_dimy(A.get_block_dimy());
    b_hd = b;
    // smooth block matrix
    block_smoother->setup(A_hd, false);
    block_smoother->set_max_iters(n_smoothing_steps);
    block_smoother->solve(b_hd, x_fin_hd, false);
//  MatrixIO<TConfig>::writeSystemMatrixMarket("A.mtx",A,x_fin_hd);
//  MatrixIO<TConfig>::writeSystemMatrixMarket("sA.mtx",scalarA,x_ini_hd);
    // assert that result is the same
    ostringstream os;
    os << bsize;
    string ssize = os.str();
    error_string = "Difference between scalar and block " + ssize + "x" + ssize + " smoother = " + smoother_string;
    UNITTEST_ASSERT_EQUAL_TOL_DESC(error_string.c_str(), x_fin_hd, x_ini_hd, final_residual_tolerance);
    delete scalar_smoother;
    delete block_smoother;
}


void run()
{
    int n_smoothing_steps = 2000;
    ValueTypeB final_residual_tol = 1e-5;

    // only 3x3 and 4x4 supported
    // only multicolor DILU
    for ( int bsize = 3; bsize < 5; bsize++ )
    {
        // diag inside
        //check_block_smoothers_poisson(bsize,9,bsize*4,bsize*4,bsize*4,"MULTICOLOR_GS",final_residual_tol,n_smoothing_steps,0);
        check_block_smoothers_poisson(bsize, 9, bsize * 4, bsize * 4, bsize * 4, "MULTICOLOR_DILU", final_residual_tol, n_smoothing_steps, 0);
        // diag outside
        //check_block_smoothers_poisson(bsize,9,bsize*4,bsize*4,bsize*4,"MULTICOLOR_GS",final_residual_tol,n_smoothing_steps,1);
        check_block_smoothers_poisson(bsize, 9, bsize * 4, bsize * 4, bsize * 4, "MULTICOLOR_DILU", final_residual_tol, n_smoothing_steps, 1);
    }
}

DECLARE_UNITTEST_END(SmootherBlockPoissonTest);


// if you want to be able run this test for all available configs you can write this:
//#define AMGX_CASE_LINE(CASE) TemplateTest <TemplateMode<CASE>::Type>  TemplateTest_##CASE;
//  AMGX_FORALL_BUILDS(AMGX_CASE_LINE)
//#undef AMGX_CASE_LINE

// or run for all device configs
#define AMGX_CASE_LINE(CASE) SmootherBlockPoissonTest <TemplateMode<CASE>::Type>  SmootherBlockPoissonTest_##CASE;
AMGX_FORALL_BUILDS_DEVICE(AMGX_CASE_LINE)
#undef AMGX_CASE_LINE

//SmootherBlockPoisson <TemplateMode<AMGX_mode_dDDI>::Type>  SmootherBlockPoisson_instance_mode_dDDI;
//SmootherBlockPoisson <TemplateMode<AMGX_mode_dFFI>::Type>  SmootherBlockPoisson_instance_mode_dFFI;

// or you can specify several desired configs
//TemplateTest <TemplateMode<AMGX_mode_hDFI>::Type>  TemplateTest_hDFI;
//TemplateTest <TemplateMode<AMGX_mode_dDFI>::Type>  TemplateTest_dDFI;


} //namespace amgx
