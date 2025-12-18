/* Copyright (c) 2011-2025, NVIDIA CORPORATION. All rights reserved.
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
#include <sstream>

namespace amgx
{

// parameter is used as test name
DECLARE_UNITTEST_BEGIN(FGMRESZeroInitialResidual);

/**
 * Solving system Ax = b with 
 * A =
 * 2.0 0.0
 * 0.0 4.0
 * 
 * b =
 * 0.0
 * 0.0
 * 
 * x =
 * 0.0
 * 0.0
 * 
 * Expecting the solver to report convergence directly since the system is
 * solved with the intial x value.
**/
void run()
{
    Resources res;
    {
        Matrix_h A;
        Vector_h b;
        Vector_h x;
        A.set_initialized(0);
        A.addProps(CSR);

        const int nrows{2};
        const int nnz{2};

        A.resize(nrows, nrows, nnz);
        A.row_offsets[0] = 0;
        A.row_offsets[1] = 1;
        A.row_offsets[2] = nnz;

        A.col_indices[0] = 0;
        A.col_indices[1] = 1;

        A.values[0] = 2.0;
        A.values[1] = 4.0;

        A.set_block_dimx(1);
        A.set_block_dimy(2);

        A.computeDiagonal();
        A.set_initialized(1);
        int bsize = A.get_block_dimy();
        int n_rows = A.get_num_rows() * bsize;
        b.set_block_dimx(1);
        b.set_block_dimy(bsize);
        x.set_block_dimy(1);
        x.set_block_dimx(bsize);
        b.resize(n_rows);
        thrust_wrapper::fill<AMGX_host>(b.begin(), b.end(), 0.);
        x.resize(n_rows);
        thrust_wrapper::fill<AMGX_host>(x.begin(), x.end(), 0.);
        // Copy to device if necessary
        MatrixA A_hd;
        VVector x_ini_hd, b_hd;

        A_hd = A;
        b_hd = b;
        x_ini_hd = x;
        // Set parameters
        std::stringstream parameter_string;
        parameter_string << "config_version=2, solver(s1)=FGMRES, s1:preconditioner(jacobi)=BLOCK_JACOBI, jacobi:max_iters=1, s1:max_iters=" << n_rows << ",s1:norm=L2, determinism_flag=1, s1:tolerance=1e-14, s1:gmres_n_restart=" << n_rows << ", s1:convergence=RELATIVE_INI_CORE, s1:monitor_residual=1, s1:print_solve_stats=1";
        std::string final = parameter_string.str();
        std::string pp(final.size() + 1, '\0');
        std::copy(final.begin(), final.end(), pp.begin());
        AMG_Configuration cfg;
        UNITTEST_ASSERT_TRUE(cfg.parseParameterString(pp.c_str()) == AMGX_OK);
        AMGX_STATUS solve_status = AMGX_ST_CONVERGED;
        AMG_Solver<TConfig> amg(&res, cfg);
        amg.setup(A_hd);
        amg.solve(b_hd, x_ini_hd, solve_status);

        std::string error_string = "expected convergence";
        UNITTEST_ASSERT_TRUE_DESC(error_string.c_str(), solve_status == AMGX_ST_CONVERGED);
    }
}

DECLARE_UNITTEST_END(FGMRESZeroInitialResidual);


// if you want to be able run this test for all available configs you can write this:
//#define AMGX_CASE_LINE(CASE) TemplateTest <TemplateMode<CASE>::Type>  TemplateTest_##CASE;
//  AMGX_FORALL_BUILDS(AMGX_CASE_LINE)
//#undef AMGX_CASE_LINE

// or run for all device configs
//#define AMGX_CASE_LINE(CASE) TemplateTest <TemplateMode<CASE>::Type>  TemplateTest_##CASE;
//  AMGX_FORALL_BUILDS_DEVICE(AMGX_CASE_LINE)
//#undef AMGX_CASE_LINE

FGMRESZeroInitialResidual <TemplateMode<AMGX_mode_dDDI>::Type>  FGMRESZeroInitialResidual_instance_mode_dDDI;
FGMRESZeroInitialResidual <TemplateMode<AMGX_mode_dFFI>::Type>  FGMRESZeroInitialResidual_instance_mode_dFFI;

// or you can specify several desired configs
//TemplateTest <TemplateMode<AMGX_mode_hDFI>::Type>  TemplateTest_hDFI;
//TemplateTest <TemplateMode<AMGX_mode_dDFI>::Type>  TemplateTest_dDFI;


} //namespace amgx
