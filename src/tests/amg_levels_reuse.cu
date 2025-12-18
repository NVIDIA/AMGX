// SPDX-FileCopyrightText: 2011 - 2025 NVIDIA CORPORATION. All Rights Reserved.
//
// SPDX-License-Identifier: BSD-3-Clause

#include <amgx_c.h>
#include <matrix_io.h>
#include "unit_test.h"
#include "amg_solver.h"
#include <matrix_io.h>
#include "test_utils.h"
#include <multiply.h>
#include <blas.h>
#include <csr_multiply.h>
#include "util.h"
#include "cutil.h"
#include "time.h"
#include <sstream>

//#define DEBUG
//#define DEBUGX
namespace amgx
{

// parameter is used as test name
DECLARE_UNITTEST_BEGIN(AmgLevelsReuse);

struct TestCase
{
    std::string config_string;
    bool insert_diagonal;
    bool use_pre_setup;

    TestCase(): use_pre_setup(true), insert_diagonal(false){}
};

std::vector<double> test_main(TestCase &test_case)
{
    bool insert_diagonal = test_case.insert_diagonal;
    // Create matrix arrays from file
    Matrix_h Atemp;
    Vector_h btemp, xtemp, x_final;
    // Make the matrix
    generatePoissonForTest(Atemp, 1, 0, 27, 40, 40, 40);
    btemp.resize(Atemp.get_num_rows());
    for (auto& val: btemp)
        val = 1.0;
    xtemp.resize(Atemp.get_num_rows());

    bool hasDiag = Atemp.hasProps(DIAG);
    // Create row_offsets, col_indices, off_dia_values and dia_values arrays from the matrix just read
    int num_rows = Atemp.get_num_rows();
    int num_nz = Atemp.get_num_nz();
    int bsize_x = Atemp.get_block_dimx();
    int bsize_y = Atemp.get_block_dimy();
    int bsize = bsize_x * bsize_y;
    xtemp.resize(num_rows * bsize_y, 1.);
    std::vector<int> row_offsets(num_rows + 1);
    std::vector<int> col_indices(num_nz);
    std::vector<double> off_dia_values(num_nz * bsize);
    std::vector<double> dia_values;

    if (hasDiag)
    {
        dia_values.resize(num_rows * bsize);
    }

    std::vector<double> x_vec(num_rows * bsize_y);
    std::vector<double> b_vec(num_rows * bsize_x);
    // Fill vectors
    int *raw_row_ptr = Atemp.row_offsets.raw();
    int *raw_col_ptr = Atemp.col_indices.raw();
    double *raw_val_ptr = Atemp.values.raw();

    // Row offsets
    for (int i = 0; i < num_rows + 1; i++)
    {
        row_offsets[i] = raw_row_ptr[i];
    }

    // Column indices
    for (int i = 0; i < num_nz; i++)
    {
        col_indices[i] = raw_col_ptr[i];
    }

    // Off-diagonal values
    for (int i = 0; i < num_nz; i++)
        for (int j = 0; j < bsize; j++)
        {
            off_dia_values[i * bsize + j] = raw_val_ptr[i * bsize + j];
        }

    // Diagonal values
    if (hasDiag)
    {
        for (int i = 0; i < num_rows; i++)
        {
            for (int j = 0; j < bsize; j++)
            {
                dia_values[i * bsize + j] = raw_val_ptr[num_nz * bsize + i * bsize + j];
            }
        }
    }

    srand(1);
    // Random RHS
    double *b_raw_ptr = btemp.raw();

    for (int i = 0; i < num_rows; i++)
        for (int j = 0; j < bsize_x; j++)
        {
            b_vec[i * bsize_x + j] = b_raw_ptr[i * bsize_x + j] + (1.0 * rand() / RAND_MAX);
        }

    //b_vec[i*bsize_x+j] = b_raw_ptr[i*bsize_x+j];
    // Random xvector
    srand(2);
    double *x_raw_ptr = xtemp.raw();

    for (int i = 0; i < num_rows; i++)
        for (int j = 0; j < bsize_y; j++)
        {
            x_vec[i * bsize_y + j] = x_raw_ptr[i * bsize_y + j] + (1.0 * rand() / RAND_MAX);
        }

    //x_vec[i*bsize_y+j] = x_raw_ptr[i*bsize_y+j];
    std::vector<double> x_vec_col = x_vec;
    std::string option_string = test_case.config_string;

    // Insert diagonal
    if (insert_diagonal && hasDiag)
    {
        std::vector<int> new_col_indices( (num_nz + num_rows) );
        std::vector<double> new_off_dia_values( (num_nz + num_rows)*bsize );
        int icount = 0;

        for (int i = 0; i < num_rows; i++)
        {
            for (int j = row_offsets[i]; j < row_offsets[i + 1]; j++)
            {
                int col = col_indices[j];
                new_col_indices[icount] = col;

                for (int k = 0; k < bsize; k++)
                {
                    new_off_dia_values[icount * bsize + k] = off_dia_values[j * bsize + k];
                }

                icount++;
            }

            // Insert diagonal
            new_col_indices[icount] = i;

            for (int k = 0; k < bsize; k++)
            {
                new_off_dia_values[icount * bsize + k] = dia_values[i * bsize + k];
            }

            icount++;
        }

        // increment row_offsets
        for (int i = 0; i < num_rows + 1; i++)
        {
            row_offsets[i] += i;
        }

        off_dia_values = new_off_dia_values;
        col_indices = new_col_indices;
        dia_values.resize(0);
        num_nz += num_rows;
    }

    AMGX_config_handle rsrc_cfg = NULL;
    UNITTEST_ASSERT_EQUAL(AMGX_config_create(&rsrc_cfg, ""), AMGX_OK);
    // Choosing device 0
    int device = 0;
    AMGX_resources_handle rsrc = NULL;
    UNITTEST_ASSERT_EQUAL(AMGX_resources_create(&rsrc, rsrc_cfg, NULL, 1, &device), AMGX_OK);
    AMGX_config_handle cfg;
    UNITTEST_ASSERT_EQUAL(AMGX_config_create( &cfg, option_string.c_str()), AMGX_OK);
    AMGX_matrix_handle matrix;
    UNITTEST_ASSERT_EQUAL(AMGX_matrix_create( &matrix, rsrc, AMGX_mode_dDDI ), AMGX_OK);
    AMGX_solver_handle solver;
    UNITTEST_ASSERT_EQUAL(AMGX_solver_create( &solver, rsrc, AMGX_mode_dDDI, cfg), AMGX_OK);
    AMGX_vector_handle b, x;
    UNITTEST_ASSERT_EQUAL(AMGX_vector_create( &b, rsrc, AMGX_mode_dDDI ), AMGX_OK);
    UNITTEST_ASSERT_EQUAL(AMGX_vector_create( &x, rsrc, AMGX_mode_dDDI ), AMGX_OK);
    int num_setup_iters = 2;

    for (int i_setup = 0; i_setup < num_setup_iters; i_setup++)
    {
#ifdef DEBUG
        std::cout << "outer iteration #" << i_setup << std::endl;
#endif

        // Upload the new matrix and call setup
        if (i_setup == 0)
        {
            if (dia_values.size() != 0)
            {
                UNITTEST_ASSERT_EQUAL(AMGX_matrix_upload_all(matrix, num_rows, num_nz, bsize_x, bsize_y, &row_offsets[0], &col_indices[0], &off_dia_values[0], &dia_values[0]), AMGX_OK);
            }
            else
            {
                UNITTEST_ASSERT_EQUAL(AMGX_matrix_upload_all(matrix, num_rows, num_nz, bsize_x, bsize_y, &row_offsets[0], &col_indices[0], &off_dia_values[0], NULL), AMGX_OK);
            }

            UNITTEST_ASSERT_EQUAL(AMGX_solver_setup( solver, matrix ), AMGX_OK);
        }
        else
        {
            // Perturb the matrix
            /* for (int i=0;i<num_nz;i++)
               for (int j=0;j<bsize;j++)
                 off_dia_values[i*bsize+j] += .001*i_setup;*/

            //// perturb the diagonal
            if (hasDiag)
            {
                for (int i = 0; i < num_rows; i++)
                {
                    for (int j = 0; j < bsize; j++)
                    {
                        dia_values[i * bsize + j] += .001 * i_setup;
                    }
                }
            }

            if (dia_values.size() != 0)
            {
                UNITTEST_ASSERT_EQUAL(AMGX_matrix_replace_coefficients(matrix, num_rows, num_nz, &off_dia_values[0], &dia_values[0]), AMGX_OK);
            }
            else
            {
                UNITTEST_ASSERT_EQUAL(AMGX_matrix_replace_coefficients(matrix, num_rows, num_nz, &off_dia_values[0], NULL), AMGX_OK);
            }

            if (test_case.use_pre_setup)
            {
                UNITTEST_ASSERT_EQUAL(AMGX_solver_resetup( solver, matrix ), AMGX_OK);
            }
            else
            {
                UNITTEST_ASSERT_EQUAL(AMGX_solver_setup( solver, matrix ), AMGX_OK);
            }
        }

        UNITTEST_ASSERT_EQUAL(AMGX_vector_upload( b, num_rows, bsize_y, &b_vec[0]  ), AMGX_OK);
        UNITTEST_ASSERT_EQUAL(AMGX_vector_upload( x, num_rows, bsize_x, &x_vec[0]  ), AMGX_OK);
        UNITTEST_ASSERT_EQUAL(AMGX_solver_solve( solver, b, x ), AMGX_OK);
        UNITTEST_ASSERT_EQUAL(AMGX_vector_download( x, &x_vec[0]  ), AMGX_OK);
    }

#ifdef DEBUGX
    std::cout << "final x_vec" << std::endl;

    for (int i = 0; i < x_vec.size(); i++)
    {
        std::cout << i << " " << x_vec[i] << std::endl;
    }

#endif

    UNITTEST_ASSERT_EQUAL(AMGX_solver_destroy( solver ), AMGX_OK);
    UNITTEST_ASSERT_EQUAL(AMGX_matrix_destroy( matrix ), AMGX_OK);
    UNITTEST_ASSERT_EQUAL(AMGX_vector_destroy( b ), AMGX_OK);
    UNITTEST_ASSERT_EQUAL(AMGX_vector_destroy( x ), AMGX_OK);
    UNITTEST_ASSERT_EQUAL(AMGX_config_destroy( cfg ), AMGX_OK);
    UNITTEST_ASSERT_EQUAL(AMGX_config_destroy( rsrc_cfg ), AMGX_OK);
    UNITTEST_ASSERT_EQUAL(AMGX_resources_destroy( rsrc ), AMGX_OK);
    return x_vec;
}

void run()
{
    SignalHandler::hook();
    AMGX_finalize();
    UnitTest::amgx_intialized = false;
    std::vector<TestCase> test_cases;
    TestCase temp_case;
    std::string base_string;
    base_string = "config_version=2, ";
    base_string += "solver(main_solver)=AMG, ";
    base_string += "main_solver:algorithm=AGGREGATION, ";
    base_string += "main_solver:coarseAgenerator=LOW_DEG,";
    base_string += "main_solver:coloring_level=1,";
    base_string += "main_solver:convergence=RELATIVE_MAX,";
    base_string += "main_solver:cycle=V,";
    base_string += "main_solver:matrix_coloring_scheme=MIN_MAX,";
    base_string += "main_solver:max_levels=21,";
    base_string += "main_solver:norm=L1,";
    base_string += "main_solver:postsweeps=3,";
    base_string += "main_solver:presweeps=0,";
    base_string += "main_solver:selector=SIZE_2,";
    base_string += "main_solver:smoother=BLOCK_JACOBI,";
    base_string += "main_solver:tolerance=0.1,";
    temp_case.insert_diagonal = true;
    int max_reuse_levels = 10;

    for (int i = 0; i < max_reuse_levels; i++)
    {
        std::ostringstream config_string;
        config_string << base_string << ", main_solver:structure_reuse_levels=" << i << ", ";
        temp_case.config_string = config_string.str();
        temp_case.use_pre_setup = false;
        test_cases.push_back(temp_case);
    }

    for (int i = 0; i < test_cases.size(); i++)
    {
        test_cases[i].config_string += ",";
        test_cases[i].config_string += "main_solver:max_iters=2,";
        test_cases[i].config_string += "main_solver:monitor_residual=1,";
        test_cases[i].config_string += "determinism_flag=1,";
        test_cases[i].config_string += "max_uncolored_percentage=0.,";
        test_cases[i].config_string += "main_solver:store_res_history=1,";
        //test_cases[i].config_string += "main_solver:print_solve_stats=1,";
        //test_cases[i].config_string += "main_solver:print_grid_stats=1,";
        test_cases[i].config_string += "main_solver:obtain_timings=1";
    }

    std::vector<double> x_ref;
    std::vector<double> x;

    for (int i = 0; i < test_cases.size(); i++)
    {
#ifdef DEBUG
        std::stringstream fail_msg;
        fail_msg << std::endl <<  "structure_reuse_levels = " << i << std::endl;
        std::cout << fail_msg.str() << std::endl;
#endif
        AMGX_initialize();

        if (i == 0)
        {
            x_ref = test_main(test_cases[i]);
        }
        else
        {
            x = test_main(test_cases[i]);
            std::stringstream fail_msg;
            fail_msg <<  "Different result for test_case, " << std::endl;
            fail_msg <<  " config string = " << test_cases[i].config_string << std::endl;;
            fail_msg <<  " use pre_setup = " << test_cases[i].use_pre_setup << std::endl;
            this->PrintOnFail(fail_msg.str().c_str());
            double zero = 1e-12;

            for (int i = 0; i < x.size(); i++)
            {
                if (abs(x[i]) > zero)
                {
                    UNITTEST_ASSERT_EQUAL_TOL(x[i], x_ref[i], 1e-8);
                }
                else if (abs(x[i]) < zero && abs(x_ref[i]) < zero)
                {
                    UNITTEST_ASSERT_TRUE(true);
                }
            }
        }

        AMGX_finalize();
    }

    AMGX_initialize();
    UnitTest::amgx_intialized = true;
}

DECLARE_UNITTEST_END(AmgLevelsReuse);


// if you want to be able run this test for all available configs you can write this:
//#define AMGX_CASE_LINE(CASE) TemplateTest <TemplateMode<CASE>::Type>  TemplateTest_##CASE;
//  AMGX_FORALL_BUILDS(AMGX_CASE_LINE)
//#undef AMGX_CASE_LINE

// or run for all device configs
//#define AMGX_CASE_LINE(CASE) TemplateTest <TemplateMode<CASE>::Type>  TemplateTest_##CASE;
//  AMGX_FORALL_BUILDS_DEVICE(AMGX_CASE_LINE)
//#undef AMGX_CASE_LINE

AmgLevelsReuse <TemplateMode<AMGX_mode_dDDI>::Type>  AmgLevelsReuse_mode_dDDI;

// or you can specify several desired configs
//TemplateTest <TemplateMode<AMGX_mode_hDFI>::Type>  TemplateTest_hDFI;
//TemplateTest <TemplateMode<AMGX_mode_dDFI>::Type>  TemplateTest_dDFI;

}



