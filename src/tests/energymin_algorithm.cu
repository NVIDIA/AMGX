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
#include <stdlib.h>     /* srand, rand */

//#define SOLVE_ZERO_INI_GUESS
//#define DEBUG
//#define DEBUGX
namespace amgx
{

// parameter is used as test name
DECLARE_UNITTEST_BEGIN(EnergyminAlgorithmTest);



std::string get_A_input_Dir() { return "Internal/energymin/input_matrix/"; }
std::string get_ext_str() { return ".mtx"; }

void generateInputFilenames(const std::vector<int> &N_range_vec,
                            const std::vector<std::string> &epsilons_str_vec,
                            const std::vector<std::string> &thetas_str_vec,
                            std::vector<std::string> &fnames_str_vec)
{
    for (int nInd = 0; nInd < N_range_vec.size(); nInd++)
    {
        int N = N_range_vec[nInd];

        for ( int eInd = 0; eInd < epsilons_str_vec.size(); eInd++ )
        {
            std::string epsilon_str = epsilons_str_vec[eInd];

            for ( int tInd = 0; tInd < thetas_str_vec.size(); tInd++ )
            {
                std::string theta_str = thetas_str_vec[tInd];
                std::stringstream ss;
                ss << "aniso_matrix_" << N << "x" << N << "_eps_" << epsilon_str << "_theta_" << theta_str;
                fnames_str_vec.push_back(ss.str());
            }
        }
    }
}

void generateAllInputFilenames(std::vector<std::string> &fnames_str_vec)
{
    //fnames_str_vec.push_back("matrix");
    std::vector<int> N_range_vec;
    //N_range_vec.push_back(5);
    //N_range_vec.push_back(10);
    //N_range_vec.push_back(15);
    N_range_vec.push_back(20);
    std::vector<std::string> epsilons_str_vec;
    epsilons_str_vec.push_back("1");
    //epsilons_str_vec.push_back("0_10");
    //epsilons_str_vec.push_back("0_01");
    std::vector<std::string> thetas_str_vec;
    thetas_str_vec.push_back("0");
    //thetas_str_vec.push_back("0_79");
    //thetas_str_vec.push_back("-0_79");
    this->generateInputFilenames(N_range_vec, epsilons_str_vec, thetas_str_vec, fnames_str_vec);
}


void createConfigString(std::string &base_string)
{
    base_string += "main_solver:coloring_level=1,";
    //base_string += "main_solver:convergence=RELATIVE_MAX,";
    base_string += "main_solver:cycle=V,";
    base_string += "main_solver:matrix_coloring_scheme=MIN_MAX,";
    base_string += "main_solver:max_levels=100,";
    base_string += "main_solver:norm=L1,";
    base_string += "main_solver:postsweeps=2,";
    base_string += "main_solver:presweeps=2,";
    base_string += "main_solver:smoother=MULTICOLOR_GS,";
    base_string += "main_solver:tolerance=0.1,";
    base_string += "main_solver:max_iters=100,";
    base_string += "main_solver:monitor_residual=1,";
    //test_cases[i].config_string += "determinism_flag=1,";
    base_string += "max_uncolored_percentage=0.,";
    base_string += "main_solver:store_res_history=1,";
    base_string += "main_solver:obtain_timings=1";
}


struct TestCase
{
    std::string file_name;
    std::string config_string;
    bool extract_diagonal;
    bool insert_diagonal;
    bool use_pre_setup;

    TestCase(): use_pre_setup(true), insert_diagonal(false), extract_diagonal(false) {}
};



std::vector<double> test_main(TestCase &test_case, bool generate_rhs = 0)
{
    bool insert_diagonal = test_case.insert_diagonal;
    bool extract_diagonal = test_case.extract_diagonal;
    // Create matrix arrays from file
    Matrix_h Atemp;
    Vector_h btemp, xtemp, x_final;
    // Read the matrix
    std::string fail_msg = "Cannot open " + test_case.file_name;
    this->PrintOnFail(fail_msg.c_str());

    if (generate_rhs)
    {
        std::string mtxInFileName = UnitTest::get_configuration().data_folder + test_case.file_name;
        //typedef typename T_Config::template setMemSpace<AMGX_host>::Type Config_h;
        UNITTEST_ASSERT_TRUE(MatrixIO<TConfig_h>::readSystem(mtxInFileName.c_str(), Atemp) == AMGX_OK);
        btemp.resize(Atemp.get_num_rows(), 0.);
        xtemp.resize(Atemp.get_num_rows(), 0.);
        //srand(1);
        int dd = 7;

        for (int i = 0; i < btemp.size(); i++)
        {
            //btemp[i] = (float(rand()%100))/100 - 50;
            btemp[i] = (i % dd - dd / 2); //*(10/float(dd));
            //std::cout << " btemp["<<i<<"]=" << btemp[i];
        }
    }
    else
    {
        UNITTEST_ASSERT_TRUE(this->read_system(test_case.file_name.c_str(), Atemp, btemp, xtemp));
    }

    bool hasDiag = Atemp.hasProps(DIAG);
    // Create row_offsets, col_indices, off_dia_values and dia_values arrays from the matrix just read
    int num_rows = Atemp.get_num_rows();
    int num_nz   = Atemp.get_num_nz();
    int bsize_x = Atemp.get_block_dimx();
    int bsize_y = Atemp.get_block_dimy();
    int bsize   = bsize_x * bsize_y;
    xtemp.resize(num_rows * bsize_y, 1.);
    std::vector<int> row_offsets(num_rows + 1);
    std::vector<int> col_indices(num_nz);
    std::vector<double> off_dia_values(num_nz * bsize);
    std::vector<double> dia_values;

    if (hasDiag) { dia_values.resize(num_rows * bsize); }

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
    {
        for (int j = 0; j < bsize; j++)
        {
            off_dia_values[i * bsize + j] = raw_val_ptr[i * bsize + j];
        }
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
    {
        for (int j = 0; j < bsize_x; j++)
        {
            b_vec[i * bsize_x + j] = b_raw_ptr[i * bsize_x + j] + (1.0 * rand() / RAND_MAX);
            //b_vec[i*bsize_x+j] = b_raw_ptr[i*bsize_x+j];
        }
    }

    // Random xvector
    srand(2);
    double *x_raw_ptr = xtemp.raw();

    for (int i = 0; i < num_rows; i++)
    {
        for (int j = 0; j < bsize_y; j++)
        {
            x_vec[i * bsize_y + j] = x_raw_ptr[i * bsize_y + j] + (1.0 * rand() / RAND_MAX);
            //x_vec[i*bsize_y+j] = x_raw_ptr[i*bsize_y+j];
        }
    }

    std::vector<double> x_vec_col = x_vec;
    std::string option_string = test_case.config_string;

    //std::cout << "hasDiag=" << hasDiag << "\n";
    // Insert diagonal
    if (hasDiag && insert_diagonal)
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

    // End Insert diagonal

    // Extract diagonal
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

    // End Extract diagonal
    AMGX_config_handle rsrc_cfg = NULL;
    UNITTEST_ASSERT_EQUAL(AMGX_config_create(&rsrc_cfg, ""), AMGX_OK);
    // Choosing device 0
    int device = 0;
    AMGX_resources_handle rsrc = NULL;
    UNITTEST_ASSERT_EQUAL(AMGX_resources_create(&rsrc, rsrc_cfg, NULL, 1, &device), AMGX_OK);
    std::cout << "\n" << option_string << "\n";
    AMGX_config_handle cfg = NULL;
    UNITTEST_ASSERT_EQUAL(AMGX_config_create( &cfg, option_string.c_str() ), AMGX_OK);
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
                UNITTEST_ASSERT_EQUAL(AMGX_matrix_upload_all( matrix, num_rows, num_nz, bsize_x, bsize_y,
                                      &row_offsets[0], &col_indices[0],
                                      &off_dia_values[0], &dia_values[0] ),
                                      AMGX_OK);
            }
            else
            {
                UNITTEST_ASSERT_EQUAL(AMGX_matrix_upload_all( matrix, num_rows, num_nz, bsize_x, bsize_y,
                                      &row_offsets[0], &col_indices[0],
                                      &off_dia_values[0], NULL ),
                                      AMGX_OK);
            }

            UNITTEST_ASSERT_EQUAL(AMGX_solver_setup( solver, matrix ), AMGX_OK);
        }
        else
        {
            // Perturb the matrix
            //for (int i=0;i<num_nz;i++)
            //  for (int j=0;j<bsize;j++)
            //    off_dia_values[i*bsize+j] += .001*i_setup;

            // perturb the diagonal
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
                UNITTEST_ASSERT_EQUAL(AMGX_matrix_replace_coefficients( matrix, num_rows, num_nz,
                                      &off_dia_values[0], &dia_values[0] ),
                                      AMGX_OK);
            }
            else
            {
                UNITTEST_ASSERT_EQUAL(AMGX_matrix_replace_coefficients( matrix, num_rows, num_nz,
                                      &off_dia_values[0], NULL ),
                                      AMGX_OK);
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

        UNITTEST_ASSERT_EQUAL(AMGX_vector_upload( b, num_rows, bsize_y, &b_vec[0] ), AMGX_OK);
        UNITTEST_ASSERT_EQUAL(AMGX_vector_upload( x, num_rows, bsize_x, &x_vec[0] ), AMGX_OK);
        UNITTEST_ASSERT_EQUAL(AMGX_solver_solve( solver, b, x ), AMGX_OK);
        UNITTEST_ASSERT_EQUAL(AMGX_vector_download( x, &x_vec[0] ), AMGX_OK);
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
    std::vector<std::string> test_files;
    //test_files.push_back("Internal/energymin/input_matrix/aniso_matrix_20x20_eps_1_theta_0_79.mtx");
    this->generateAllInputFilenames(test_files);
    //test_files.push_back("Internal/poisson/poisson27x16x16x16.mtx");
    //test_files.push_back("Internal/poisson/poisson27x50x50x50.mtx");
    std::string base_string, base_string_em, base_string_cl;
    base_string  = "config_version=2,";
    base_string += "solver(main_solver)=AMG,";
    base_string_em = base_string + "main_solver:algorithm=ENERGYMIN,";
    base_string_cl = base_string + "main_solver:algorithm=CLASSICAL,";
    //base_string_em += "main_solver:algorithm=CLASSICAL,";
    //base_string_em += "main_solver:algorithm=AGGREGATION,";
    this->createConfigString(base_string_em);
    this->createConfigString(base_string_cl);
    temp_case.insert_diagonal = true;

    for (int i = 0; i < test_files.size(); i++)
    {
        temp_case.config_string = base_string_cl;
        temp_case.use_pre_setup = false;
        test_cases.push_back(temp_case);
        test_cases.back().file_name = this->get_A_input_Dir() + test_files[i] + this->get_ext_str();
        temp_case.config_string = base_string_em;
        temp_case.use_pre_setup = false;
        test_cases.push_back(temp_case);
        test_cases.back().file_name = this->get_A_input_Dir() + test_files[i] + this->get_ext_str();
    }

    bool generate_rhs = 1;
    std::vector<double> x_ref;
    std::vector<double> x;

    for (int i = 0; i < test_cases.size(); i++)
    {
        AMGX_initialize();

        if (i % 2 == 0)
        {
            x_ref.clear();
            x_ref = test_main(test_cases[i], generate_rhs);
        }
        else
        {
            x.clear();
            x = test_main(test_cases[i], generate_rhs);
            //x_ref.resize(x.size(), 0);
            std::stringstream fail_msg;
            fail_msg <<  "Different result for test_case, " << std::endl;
            fail_msg <<  " config string = " << test_cases[i].config_string << std::endl;;
            fail_msg <<  " use pre_setup = " << test_cases[i].use_pre_setup << std::endl;
            this->PrintOnFail(fail_msg.str().c_str());

            for (int i = 0; i < x.size(); i++)
            {
                UNITTEST_ASSERT_EQUAL_TOL(x[i], x_ref[i], 1.0e-2); //1e-8);
                //std::cout << "\n x["<<i<<"]=" << x[i] << " x_ref["<<i<<"]=" << x_ref[i];
            }
        }

        AMGX_finalize();
    }

    AMGX_initialize();
    UnitTest::amgx_intialized = true;
}

DECLARE_UNITTEST_END(EnergyminAlgorithmTest);


// if you want to be able run this test for all available configs you can write this:
//#define AMGX_CASE_LINE(CASE) TemplateTest <TemplateMode<CASE>::Type>  TemplateTest_##CASE;
//  AMGX_FORALL_BUILDS(AMGX_CASE_LINE)
//#undef AMGX_CASE_LINE

// or run for all device configs
//#define AMGX_CASE_LINE(CASE) TemplateTest <TemplateMode<CASE>::Type>  TemplateTest_##CASE;
//  AMGX_FORALL_BUILDS_DEVICE(AMGX_CASE_LINE)
//#undef AMGX_CASE_LINE

EnergyminAlgorithmTest <TemplateMode<AMGX_mode_dDDI>::Type>  EnergyminAlgorithmTest_dDDI;

// or you can specify several desired configs
//TemplateTest <TemplateMode<AMGX_mode_hDFI>::Type>  TemplateTest_hDFI;
//TemplateTest <TemplateMode<AMGX_mode_dDFI>::Type>  TemplateTest_dDFI;

}



