// SPDX-FileCopyrightText: 2011 - 2024 NVIDIA CORPORATION. All Rights Reserved.
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
#define DEBUG
namespace amgx
{

// parameter is used as test name
DECLARE_UNITTEST_BEGIN(ProfileTest);

struct TestCase
{
    std::string file_name;
    std::string config_string;
    bool extract_diagonal;
    bool insert_diagonal;
    bool use_pre_setup;
    bool use_replace;

    TestCase(): use_pre_setup(true), insert_diagonal(false), extract_diagonal(false), use_replace(true) {}
};

std::vector<double> test_main(TestCase &test_case)
{
    bool insert_diagonal = test_case.insert_diagonal;
    bool extract_diagonal = test_case.extract_diagonal;
    // Create matrix arrays from file
    Matrix_h Atemp;
    Vector_h btemp, xtemp, x_final;
    // Read the matrix
    std::string fail_msg = "Cannot open " + test_case.file_name;
    this->PrintOnFail(fail_msg.c_str());
    this->read_system(test_case.file_name.c_str(), Atemp, btemp, xtemp);
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

    // Random xvector
    double *x_raw_ptr = xtemp.raw();

    for (int i = 0; i < num_rows; i++)
        for (int j = 0; j < bsize_y; j++)
        {
            x_vec[i * bsize_y + j] = x_raw_ptr[i * bsize_y + j] + (1.0 * rand() / RAND_MAX);
        }

    std::vector<double> x_vec_col = x_vec;
    std::string option_string = test_case.config_string;

    // Insert diagonal
    if (insert_diagonal)
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

    //Extract diagonal
    if (extract_diagonal)
    {
        std::vector<int> old_col_indices = col_indices;
        std::vector<double> old_off_dia_values = off_dia_values;
        off_dia_values.resize((num_nz - num_rows)*bsize);
        col_indices.resize(num_nz - num_rows);
        dia_values.resize(num_rows * bsize);
        int icount = 0;

        for (int i = 0; i < num_rows; i++)
        {
            for (int j = row_offsets[i]; j < row_offsets[i + 1]; j++)
            {
                int col = old_col_indices[j];

                if (col != i)
                {
                    col_indices[icount] = col;

                    for (int k = 0; k < bsize; k++)
                    {
                        off_dia_values[icount * bsize + k] = old_off_dia_values[j * bsize + k];
                    }

                    icount++;
                }
                else
                {
                    for (int k = 0; k < bsize; k++)
                    {
                        dia_values[i * bsize + k] = old_off_dia_values[j * bsize + k];
                    }
                }
            }
        }

        // decrement row_offsets
        for (int i = 0; i < num_rows + 1; i++)
        {
            row_offsets[i] -= i;
        }

        num_nz -= num_rows;
    }

    AMGX_config_handle cfg;
    AMGX_config_create( &cfg, option_string.c_str());
    AMGX_config_handle rsrc_cfg = NULL;
    UNITTEST_ASSERT_EQUAL(AMGX_config_create(&rsrc_cfg, ""), AMGX_OK);
    // Choosing device 0
    int device = 0;
    AMGX_resources_handle rsrc = NULL;
    UNITTEST_ASSERT_EQUAL(AMGX_resources_create(&rsrc, rsrc_cfg, NULL, 1, &device), AMGX_OK);
    AMGX_matrix_handle matrix;
    AMGX_matrix_create( &matrix, rsrc, AMGX_mode_dDDI );
    AMGX_solver_handle solver;
    AMGX_solver_create( &solver, rsrc, AMGX_mode_dDDI, cfg);
    AMGX_vector_handle b, x;
    AMGX_vector_create( &b, rsrc, AMGX_mode_dDDI );
    AMGX_vector_create( &x, rsrc, AMGX_mode_dDDI );
    int num_setup_iters = 3;

    for (int i_setup = 0; i_setup < num_setup_iters; i_setup++)
    {
        // Upload the new matrix and call setup
        if (i_setup == 0)
        {
            if (dia_values.size() != 0)
            {
                AMGX_matrix_upload_all(matrix, num_rows, num_nz, bsize_x, bsize_y, &row_offsets[0], &col_indices[0], &off_dia_values[0], &dia_values[0]);
            }
            else
            {
                AMGX_matrix_upload_all(matrix, num_rows, num_nz, bsize_x, bsize_y, &row_offsets[0], &col_indices[0], &off_dia_values[0], NULL);
            }

            AMGX_solver_setup( solver, matrix );
        }
        else
        {
            // Perturb the matrix
            //for (int i=0;i<num_nz;i++)
            //  for (int j=0;j<bsize;j++)
            //    off_dia_values[i*bsize+j] -= .0001*abs(rand())/RAND_MAX;

            //// perturb the diagonal
            //if (hasDiag) {
            //  for (int i=0;i<num_rows;i++) {
            //    for (int j=0;j<bsize;j++) {
            //      dia_values[i*bsize+j] += .001*(rand())/RAND_MAX;
            //    }
            //  }
            //}
            if (test_case.use_replace)
            {
                if (dia_values.size() != 0)
                {
                    AMGX_matrix_replace_coefficients(matrix, num_rows, num_nz, &off_dia_values[0], &dia_values[0]);
                }
                else
                {
                    AMGX_matrix_replace_coefficients(matrix, num_rows, num_nz, &off_dia_values[0], NULL);
                }
            }
            else
            {
                if (dia_values.size() != 0)
                {
                    AMGX_matrix_upload_all(matrix, num_rows, num_nz, bsize_x, bsize_y, &row_offsets[0], &col_indices[0], &off_dia_values[0], &dia_values[0]);
                }
                else
                {
                    AMGX_matrix_upload_all(matrix, num_rows, num_nz, bsize_x, bsize_y, &row_offsets[0], &col_indices[0], &off_dia_values[0], NULL);
                }
            }

            if (test_case.use_pre_setup)
            {
                AMGX_solver_resetup( solver, matrix );
            }
            else
            {
                AMGX_solver_setup( solver, matrix );
            }
        }

        // Run several solves.
        const int num_solves = 5;

        for ( int i_solve = 0 ; i_solve < num_solves ; ++i_solve )
        {
            // perturb the rhs
            //for (int i=0;i<num_rows;i++)
            //  for (int j=0;j<bsize_x;j++)
            //    b_vec[i*bsize_x+j] += (1.0*rand()/RAND_MAX);
            AMGX_vector_upload( b, num_rows, bsize_y, &b_vec[0]  );
            // upload the updated x_vector
            AMGX_vector_upload( x, num_rows, bsize_x, &x_vec[0]  );
            AMGX_solver_solve_with_0_initial_guess( solver, b, x );
            AMGX_vector_download( x, &x_vec[0]  );
        }
    }

    AMGX_solver_destroy( solver );
    AMGX_matrix_destroy( matrix );
    AMGX_vector_destroy( b );
    AMGX_vector_destroy( x );
    AMGX_config_destroy( cfg );
    UNITTEST_ASSERT_EQUAL(AMGX_config_destroy( rsrc_cfg ), AMGX_OK);
    UNITTEST_ASSERT_EQUAL(AMGX_resources_destroy( rsrc ), AMGX_OK);
    return x_vec;
}

void run()
{
    TestCase temp_case;
    temp_case.file_name = "Public/florida/atmosmodd.mtx";
    temp_case.config_string = "config_version=2, solver(main_solver)=MULTICOLOR_ILU, main_solver:ilu_sparsity_level=1, main_solver:max_iters=5, block_format=ROW_MAJOR, main_solver:coloring_level=2";
    temp_case.config_string += ",";
    temp_case.config_string += "main_solver:monitor_residual=1,";
    temp_case.config_string += "max_uncolored_percentage=0.,";
    temp_case.config_string += "main_solver:store_res_history=1,";
    temp_case.config_string += "main_solver:print_solve_stats=1,";
    temp_case.config_string += "main_solver:obtain_timings=1";
    temp_case.extract_diagonal = false;
    temp_case.insert_diagonal = false;
    test_main(temp_case);
}

DECLARE_UNITTEST_END(ProfileTest);


// if you want to be able run this test for all available configs you can write this:
//#define AMGX_CASE_LINE(CASE) TemplateTest <TemplateMode<CASE>::Type>  TemplateTest_##CASE;
//  AMGX_FORALL_BUILDS(AMGX_CASE_LINE)
//#undef AMGX_CASE_LINE

// or run for all device configs
//#define AMGX_CASE_LINE(CASE) TemplateTest <TemplateMode<CASE>::Type>  TemplateTest_##CASE;
//  AMGX_FORALL_BUILDS_DEVICE(AMGX_CASE_LINE)
//#undef AMGX_CASE_LINE

ProfileTest <TemplateMode<AMGX_mode_dDDI>::Type>  ProfileTest_instance_mode_dDDI;

// or you can specify several desired configs
//TemplateTest <TemplateMode<AMGX_mode_hDFI>::Type>  TemplateTest_hDFI;
//TemplateTest <TemplateMode<AMGX_mode_dDFI>::Type>  TemplateTest_dDFI;

}



