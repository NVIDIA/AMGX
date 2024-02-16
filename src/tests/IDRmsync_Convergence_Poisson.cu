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
#ifdef AMGX_WITH_MPI
#include <mpi.h>
#endif

namespace amgx
{

// parameter is used as test name
DECLARE_UNITTEST_BEGIN(IDRmsyncConvergencePoisson);

void check_IDRmsync_convergence_poisson(int points,
                                        int nx,
                                        int ny,
                                        int nz,
                                        ValueTypeB residualTol)
{
    Resources res;
    {
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
        // Fill b with random values
        b.resize(n_rows);
        //fillRandom<Vector_h>::fill(b);
        thrust_wrapper::fill<AMGX_host>(b.begin(), b.end(), 1);
        // Initialize x to random values
        x.resize(n_rows);
        //fillRandom<Vector_h>::fill(x);
        thrust_wrapper::fill<AMGX_host>(x.begin(), x.end(), 0.);
        // Copy to device if necessary
        MatrixA A_hd;
        VVector x_ini_hd, x_fin_outside_hd, b_hd, r_hd, x_fin_inside_hd;
        //x_fin_inside_hd.setResources(&res);
        //b_hd.setResources(&res);
        r_hd.resize(n_rows, 0.);
        r_hd.set_block_dimy(1);
        r_hd.set_block_dimx(bsize);
        A_hd = A;
        b_hd = b;
        x_ini_hd = x;
        // Set parameters
        std::string error_string;
        std::stringstream parameter_string;
        parameter_string << "config_version=2, solver(s1)=IDRMSYNC, s1:preconditioner(jacobi)=MULTICOLOR_DILU, jacobi:max_iters=1, jacobi:monitor_residual=1, s1:max_iters=" << n_rows << ",s1:norm=L2, s1:use_scalar_norm=1, s1:tolerance=1e-8, s1:obtain_timings=1,s1:subspace_dim_s=8, s1:convergence=RELATIVE_INI_CORE, s1:monitor_residual=1, s1:print_solve_stats=1";
        std::string final = parameter_string.str();
        std::string pp(final.size() + 1, '\0');
        std::copy(final.begin(), final.end(), pp.begin());
        AMG_Configuration cfg;
        UNITTEST_ASSERT_TRUE(cfg.parseParameterString(pp.c_str()) == AMGX_OK);
        AMGX_STATUS solve_status = AMGX_ST_CONVERGED;
        AMG_Solver<TConfig> amg(&res, cfg);
        // -----------------------------------------------
        // Test 1: Compare diag inside vs outside result
        // -----------------------------------------------
        x_fin_inside_hd = x_ini_hd;
        //printVector("x_ini",x_ini_hd);
        amg.setup(A_hd);
        amg.solve(b_hd, x_fin_inside_hd, solve_status);
        //printVector4"x_fin_inside_hd",x_fin_inside_hd);
        // ----------------------------------------------------------------------------
        // Test 2: Check that final residual is finite and less than
        Vector_h nrm, nrm_ini;
        nrm.resize(bsize);
        nrm_ini.resize(bsize);
        // Compute the initial residual norm
        multiply( A_hd, x_ini_hd, r_hd );
        axpby( b_hd, r_hd, r_hd, ValueTypeB( 1 ), ValueTypeB( -1 ) );
        get_norm( A_hd, r_hd, 1, L2, nrm_ini );
        // Compute the residual norm
        multiply( A_hd, x_fin_inside_hd, r_hd );
        axpby( b_hd, r_hd, r_hd, ValueTypeB( 1 ), ValueTypeB( -1 ) );
        get_norm( A_hd, r_hd, 1, L2, nrm );
        //      std::cout << "ini_norm=" << nrm_ini[0] << std::endl;
        //      std::cout << "final_norm=" << nrm[0] << std::endl;
        //      std::cout << "ratio=" << nrm[0]/nrm_ini[0] << std::endl;
        //      std::cout << "tol=" << residualTol << std::endl;
        error_string = "nrm >= residualTol";
        UNITTEST_ASSERT_TRUE_DESC(error_string.c_str(), nrm[0] / nrm_ini[0] < residualTol);
    }
}


// This test checks that IDRmsync converges in at most N iterations
// (N is number of rows) for scalar Poisson matrices.
void run()
{
#ifdef AMGX_WITH_MPI
    int mpiFlag;
    MPI_Initialized(&mpiFlag);

    if ( !mpiFlag )
    {
        int argc = 1;
        char **argv = NULL;
        MPI_Init( &argc, &argv);
    }

#endif
    ValueTypeB final_residual_tol = IterativeRelTol<ValueTypeB>::get();
    int min_size = 5;
    int max_size = 10;

    for (int i = min_size; i <= max_size; i++)
    {
        // 2D poisson
        check_IDRmsync_convergence_poisson(5, i, i, i, final_residual_tol);
        check_IDRmsync_convergence_poisson(9, i, i, i, final_residual_tol);
    }

    // can not call this because potentially there is a sub-sequent call to this function
    //  MPI_Finalize();
}

DECLARE_UNITTEST_END(IDRmsyncConvergencePoisson);


// if you want to be able run this test for all available configs you can write this:
//#define AMGX_CASE_LINE(CASE) TemplateTest <TemplateMode<CASE>::Type>  TemplateTest_##CASE;
//  AMGX_FORALL_BUILDS(AMGX_CASE_LINE)
//#undef AMGX_CASE_LINE

// or run for all device configs
//#define AMGX_CASE_LINE(CASE) TemplateTest <TemplateMode<CASE>::Type>  TemplateTest_##CASE;
//  AMGX_FORALL_BUILDS_DEVICE(AMGX_CASE_LINE)
//#undef AMGX_CASE_LINE


IDRmsyncConvergencePoisson <TemplateMode<AMGX_mode_dDDI>::Type>  IDRmsyncConvergencePoisson_instance_mode_dDDI;
IDRmsyncConvergencePoisson <TemplateMode<AMGX_mode_dFFI>::Type>  IDRmsyncConvergencePoisson_instance_mode_dFFI;

// or you can specify several desired configs
//TemplateTest <TemplateMode<AMGX_mode_hDFI>::Type>  TemplateTest_hDFI;
//TemplateTest <TemplateMode<AMGX_mode_dDFI>::Type>  TemplateTest_dDFI;


} //namespace amgx
