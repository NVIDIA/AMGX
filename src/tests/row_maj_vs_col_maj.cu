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
#include<blas.h>
#include <csr_multiply.h>
#include "util.h"
#include "cutil.h"
#include "time.h"
#include <sstream>

//#define SOLVE_ZERO_INI_GUESS

namespace amgx
{
// parameter is used as test name
DECLARE_UNITTEST_BEGIN(RowMajorVsColMajor);

struct TestCase
{
    std::string file_name;
    std::string config_string;
};

void test_main(TestCase &test_case)
{
    // --------------------------------------
    // Create matrix arrays from file
    // --------------------------------------
    Matrix_h Atemp;
    Vector_h btemp, xtemp;
    // -------------------------------------------
    // Read the matrix
    // -------------------------------------------
    std::string fail_msg = "Cannot open " + test_case.file_name;
    this->PrintOnFail(fail_msg.c_str());
    UNITTEST_ASSERT_TRUE(this->read_system(test_case.file_name.c_str(), Atemp, btemp, xtemp));
    bool hasDiag = Atemp.hasProps(DIAG);
    //printVector("Atemp.values",Atemp.values); //printVector("Atemp.dia",Atemp.diag);
    //std::cout << "hasDiag=" << hasDiag << std::endl;
    // Create row_offsets, col_indices, off_dia_values and dia_values arrays from the matrix just read
    int num_rows = Atemp.get_num_rows();
    int num_nz = Atemp.get_num_nz();
    int bsize_x = Atemp.get_block_dimx();
    int bsize_y = Atemp.get_block_dimy();
    int bsize = bsize_x * bsize_y;
    //if (xtemp.size() == 0) {
    xtemp.resize(num_rows * bsize_y, 1.);
    //}
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

    // RHS
    double *b_raw_ptr = btemp.raw();

    for (int i = 0; i < num_rows; i++)
        for (int j = 0; j < bsize_x; j++)
        {
            b_vec[i * bsize_x + j] = b_raw_ptr[i * bsize_x + j];
        }

    //b_vec[i*bsize_x+j] = b_raw_ptr[i*bsize_x+j]+(1.0*rand()/RAND_MAX);
    // x vector
    double *x_raw_ptr = xtemp.raw();

    for (int i = 0; i < num_rows; i++)
        for (int j = 0; j < bsize_y; j++)
        {
            x_vec[i * bsize_y + j] = x_raw_ptr[i * bsize_y + j];
        }

    //x_vec[i*bsize_y+j] = x_raw_ptr[i*bsize_y+j]+(1.0*rand()/RAND_MAX);
    std::vector<double> x_vec_col = x_vec;
    std::string option_string = test_case.config_string;

    for (int i = 0; i < 2; i++)
    {
        if (i == 1) // Store in COL MAJOR
        {
            std::vector<double> block(bsize);

            for (int i = 0; i < num_nz; i++)
            {
                for (int j = 0; j < bsize; j++)
                {
                    block[j] = off_dia_values[i * bsize + j];
                }

                for (int j = 0; j < bsize; j++)
                {
                    off_dia_values[i * bsize + j] = block[ (j % bsize_x) * bsize_y + (j / bsize_x)];
                }
            }

            if (dia_values.size() != 0)
            {
                for (int i = 0; i < num_rows; i++)
                {
                    for (int j = 0; j < bsize; j++)
                    {
                        block[j] = dia_values[i * bsize + j];
                    }

                    for (int j = 0; j < bsize; j++)
                    {
                        dia_values[i * bsize + j] = block[ (j % bsize_x) * bsize_y + (j / bsize_x)];
                    }
                }
            }

            option_string += ", block_format=COL_MAJOR";
        }

        AMGX_config_handle cfg;
        UNITTEST_ASSERT_EQUAL(AMGX_config_create( &cfg, option_string.c_str()), AMGX_OK);
        AMGX_config_handle rsrc_cfg = NULL;
        UNITTEST_ASSERT_EQUAL(AMGX_config_create(&rsrc_cfg, ""), AMGX_OK);
        // Choosing device 0
        int device = 0;
        AMGX_resources_handle rsrc = NULL;
        UNITTEST_ASSERT_EQUAL(AMGX_resources_create(&rsrc, rsrc_cfg, NULL, 1, &device), AMGX_OK);
        AMGX_matrix_handle matrix;
        AMGX_matrix_create( &matrix, rsrc, AMGX_mode_dDDI );
        AMGX_solver_handle solver;
        AMGX_solver_create( &solver, rsrc, AMGX_mode_dDDI, cfg );
        AMGX_vector_handle b, x;
        AMGX_vector_create( &b, rsrc, AMGX_mode_dDDI );
        AMGX_vector_create( &x, rsrc, AMGX_mode_dDDI );

        // All of these should create the same result
        // Upload to the GPU.

        // ROW MAJOR
        if (dia_values.size() != 0)
        {
            UNITTEST_ASSERT_EQUAL(AMGX_matrix_upload_all(matrix, num_rows, num_nz, bsize_x, bsize_y, &row_offsets[0], &col_indices[0], &off_dia_values[0], &dia_values[0]), AMGX_OK);
        }
        else
        {
            UNITTEST_ASSERT_EQUAL(AMGX_matrix_upload_all(matrix, num_rows, num_nz, bsize_x, bsize_y, &row_offsets[0], &col_indices[0], &off_dia_values[0], NULL), AMGX_OK);
        }

        UNITTEST_ASSERT_EQUAL(AMGX_solver_setup( solver, matrix ), AMGX_OK);
        AMGX_vector_upload( b, num_rows, bsize_y, &b_vec[0]  );
        // Before solve.
#ifdef SOLVE_ZERO_INI_GUESS
        AMGX_vector_set_zero( x, num_rows, bsize_y );
        AMGX_solver_solve_with_0_initial_guess( solver, b, x );
#else

        if (i == 0)
        {
            AMGX_vector_upload( x, num_rows, bsize_x, &x_vec[0]  );
        }
        else
        {
            AMGX_vector_upload( x, num_rows, bsize_x, &x_vec_col[0]  );
        }

        AMGX_solver_solve( solver, b, x );
#endif

        if (i == 0)
        {
            AMGX_vector_download( x, &x_vec[0]  );
        }
        else
        {
            AMGX_vector_download( x, &x_vec_col[0]  );
        }

        AMGX_solver_destroy( solver );
        AMGX_matrix_destroy( matrix );
        AMGX_vector_destroy( b );
        AMGX_vector_destroy( x );
        AMGX_config_destroy( cfg );
        AMGX_config_destroy( rsrc_cfg );
        AMGX_resources_destroy( rsrc );
    }

    fail_msg = "Different results for row major vs col major for matrix: " + test_case.file_name + " with config string: " + test_case.config_string;
    this->PrintOnFail(fail_msg.c_str());
    double zero = 1e-12;

    for (int i = 0; i < x_vec.size(); i++)
    {
        //std::cout << "i=" << i << "x_vec[i] =" << x_vec[i] << std::endl;
        if (abs(x_vec[i]) > zero)
        {
            UNITTEST_ASSERT_EQUAL_TOL(x_vec[i], x_vec_col[i], 1e-8);
        }
        else if (abs(x_vec[i]) < zero && abs(x_vec_col[i]) < zero)
        {
            UNITTEST_ASSERT_TRUE(true);
        }
    }
}

void run()
{
    std::vector<TestCase> test_cases;
    TestCase temp_case;
    // ILU0
    temp_case.config_string = "config_version=2, solver(main_solver)=MULTICOLOR_ILU, block_format=ROW_MAJOR, main_solver:reorder_cols_by_color=1, main_solver:ilu_sparsity_level=0, main_solver:insert_diag_while_reordering=1, main_solver:coloring_level=2";
    temp_case.file_name = "Public/florida/atmosdd.mtx";
    test_cases.push_back(temp_case);
    // ILU1
    temp_case.config_string = "config_version=2, solver(main_solver)=MULTICOLOR_ILU, block_format=ROW_MAJOR, main_solver:reorder_cols_by_color=1, main_solver:ilu_sparsity_level=1, main_solver:insert_diag_while_reordering=1, main_solver:coloring_level=2";
    temp_case.file_name = "Public/florida/atmosdd.mtx";
    test_cases.push_back(temp_case);

    for (int i = 0; i < test_cases.size(); i++)
    {
        test_cases[i].config_string += ",";
        test_cases[i].config_string += "main_solver:max_iters=5,";
        test_cases[i].config_string += "main_solver:monitor_residual=1,";
        test_cases[i].config_string += "determinism_flag=1,";
        test_cases[i].config_string += "max_uncolored_percentage=0.,";
        test_cases[i].config_string += "main_solver:store_res_history=1,";
        test_cases[i].config_string += "main_solver:print_solve_stats=1";
    }

    SignalHandler::hook();
    AMGX_finalize();
    UnitTest::amgx_intialized = false;

    for (int i = 0; i < test_cases.size(); i++)
    {
        AMGX_initialize();
        UnitTest::amgx_intialized = true;
        test_main(test_cases[i]);
        AMGX_finalize();
        UnitTest::amgx_intialized = false;
    }

    AMGX_initialize();
    UnitTest::amgx_intialized = true;
}

DECLARE_UNITTEST_END(RowMajorVsColMajor);


// if you want to be able run this test for all available configs you can write this:
//#define AMGX_CASE_LINE(CASE) TemplateTest <TemplateMode<CASE>::Type>  TemplateTest_##CASE;
//  AMGX_FORALL_BUILDS(AMGX_CASE_LINE)
//#undef AMGX_CASE_LINE

// or run for all device configs
//#define AMGX_CASE_LINE(CASE) TemplateTest <TemplateMode<CASE>::Type>  TemplateTest_##CASE;
//  AMGX_FORALL_BUILDS_DEVICE(AMGX_CASE_LINE)
//#undef AMGX_CASE_LINE

RowMajorVsColMajor <TemplateMode<AMGX_mode_dDDI>::Type>  RowMajorVsColMajor_instance_mode_dDDI;

// or you can specify several desired configs
//TemplateTest <TemplateMode<AMGX_mode_hDFI>::Type>  TemplateTest_hDFI;
//TemplateTest <TemplateMode<AMGX_mode_dDFI>::Type>  TemplateTest_dDFI;


}



