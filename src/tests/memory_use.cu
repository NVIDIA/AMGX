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
#include<blas.h>
#include <csr_multiply.h>
#include "util.h"
#include "time.h"
#include <sstream>
#include <cublas_v2.h>

#define SOLVE_ZERO_INI_GUESS
//#define DEBUG

namespace amgx
{


struct TestCase
{
    std::string file_name;
    std::string config_string;
    double max_mem_usage;
    size_t max_mem_leak;
};

template <class Handle>
struct CWrapper
{
    AMGX_Mode mode;
    Handle hdl;
};

// parameter is used as test name
DECLARE_UNITTEST_BEGIN(Memory_Use_Base);

std::string base_keywords()
{
    return "stress";
}



size_t get_memory_usage()
{
    size_t free_mem, total_mem;
    cudaMemGetInfo( &free_mem, &total_mem );
    size_t used_mem = total_mem - free_mem;
    //used_mem /= size_t(1024);
    //double used_mem_f = double(used_mem) / 1024.0;
    return used_mem;
}


void check_memory_usage(const char *msg, size_t &mem_before, TestCase &test_case)
{
    std::stringstream mem_msg;
    size_t used_mem = get_memory_usage();
    mem_msg  << "Maximum used memory (" << (int)(used_mem / (1024.*1024.)) << ") exceeds limit set to " << test_case.max_mem_usage << " for matrix " << test_case.file_name << " : " << msg;
    UNITTEST_ASSERT_TRUE_DESC(mem_msg.str().c_str(), (used_mem - mem_before) / (1024.*1024.) < test_case.max_mem_usage);
#ifdef DEBUG
    std::cout << msg << ": " << used_mem / (1024.*1024.) << " Mb" << std::endl;
#endif
}

void launch_test_case(TestCase &test_case)
{
    SignalHandler::hook();
    AMGX_finalize();
    UnitTest::amgx_intialized = false;
    // Empty kernel call to initialize cuda context
    size_t free_mem, total_mem;
    cudaMemGetInfo(&free_mem, &total_mem);
    size_t context_buffer = 10000000;
    amgx::thrust::device_vector<double> test_vector;
    int vec_size = (free_mem - context_buffer) / 8;
    test_vector.resize(vec_size);
    test_vector.clear();
    test_vector.shrink_to_fit();
    // Create and destroy cusparse and cublas handles
    size_t before = get_memory_usage();
    {
        cusparseHandle_t cusparse_handle;
        cusparseCreate(&cusparse_handle);
        cusparseDestroy(cusparse_handle);
        cublasHandle_t cublas_handle;
        cublasCreate(&cublas_handle);
        cublasDestroy(cublas_handle);
    }
    size_t after = get_memory_usage();
    size_t used_mem_before = get_memory_usage();
    mem_test_main(test_case, used_mem_before);
    size_t used_mem_after = get_memory_usage();
    std::stringstream msg;
    msg << "Memory leak: " << (used_mem_after - used_mem_before) << " bytes exceed threshold set to " << test_case.max_mem_leak ;
#ifdef DEBUG
    std::cout << "Mem leak=" << (used_mem_after - used_mem_before) << std::endl;
#endif
    UNITTEST_ASSERT_TRUE_DESC(msg.str().c_str(), (used_mem_after - used_mem_before) <= test_case.max_mem_leak);
    UNITTEST_ASSERT_EQUAL(AMGX_initialize(), AMGX_OK);
    UnitTest::amgx_intialized = true;
}

void mem_test_main(TestCase &test_case, size_t &mem_before)
{
    check_memory_usage("before initialize", mem_before, test_case);
    UNITTEST_ASSERT_EQUAL(AMGX_initialize(), AMGX_OK);
    UnitTest::amgx_intialized = true;
    check_memory_usage("after initialize", mem_before, test_case);
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
    std::stringstream nan_msg;
    nan_msg  << "Nan residual for matrix " << test_case.file_name;
    // Create row_offsets, col_indices, off_dia_values and dia_values arrays from the matrix just read
    int num_rows = Atemp.get_num_rows();
    int num_nz = Atemp.get_num_nz();
    int bsize_x = Atemp.get_block_dimx();
    int bsize_y = Atemp.get_block_dimy();
    int bsize = bsize_x * bsize_y;

    if (xtemp.size() == 0) { xtemp.resize(num_rows * bsize_y, 0.); }

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
    // All of these should create the same result
    std::string option_string = test_case.config_string;
    check_memory_usage("before config create", mem_before, test_case);
    AMGX_config_handle cfg;
    UNITTEST_ASSERT_EQUAL(AMGX_config_create( &cfg, option_string.c_str() ), AMGX_OK);
    AMGX_config_handle rsrc_cfg = NULL;
    UNITTEST_ASSERT_EQUAL(AMGX_config_create(&rsrc_cfg, ""), AMGX_OK);
    check_memory_usage("after config create", mem_before, test_case);
    // Choosing device 0
    check_memory_usage("before resources create", mem_before, test_case);
    int device = 0;
    AMGX_resources_handle rsrc = NULL;
    UNITTEST_ASSERT_EQUAL(AMGX_resources_create(&rsrc, rsrc_cfg, NULL, 1, &device), AMGX_OK);
    check_memory_usage("after resources create", mem_before, test_case);
    double old_max_mem_usage = test_case.max_mem_usage;
#if 1
    // query device pool size
    amgx::CWrapper<AMGX_resources_handle> *c_resources = (amgx::CWrapper<AMGX_resources_handle> *)rsrc;
    test_case.max_mem_usage += ((Resources *)c_resources->hdl)->getPoolSize() / 1024.0 / 1024.0;
    AMGX_matrix_handle matrix;
    check_memory_usage("before matrix create", mem_before, test_case);
    UNITTEST_ASSERT_EQUAL(AMGX_matrix_create( &matrix, rsrc, AMGX_mode_dDDI ), AMGX_OK);
    check_memory_usage("after matrix create", mem_before, test_case);
    AMGX_solver_handle solver;
    check_memory_usage("before solver create", mem_before, test_case);
    UNITTEST_ASSERT_EQUAL(AMGX_solver_create( &solver, rsrc, AMGX_mode_dDDI, cfg), AMGX_OK);
    check_memory_usage("after solver create", mem_before, test_case);
    int num_setup_iters = 3;

    for (int i_setup = 0; i_setup < num_setup_iters; i_setup++)
    {
        // ------------------------------------------
        // Upload the new matrix and call setup
        // ------------------------------------------
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

            check_memory_usage("after matrix upload", mem_before, test_case);
            check_memory_usage("before solver setup", mem_before, test_case);
            UNITTEST_ASSERT_EQUAL(AMGX_solver_setup( solver, matrix ), AMGX_OK);
            check_memory_usage("after solver setup", mem_before, test_case);
        }
        else
        {
            check_memory_usage("before matrix replace", mem_before, test_case);

            if (dia_values.size() != 0)
            {
                UNITTEST_ASSERT_EQUAL(AMGX_matrix_replace_coefficients(matrix, num_rows, num_nz, &off_dia_values[0], &dia_values[0]), AMGX_OK);
            }
            else
            {
                UNITTEST_ASSERT_EQUAL(AMGX_matrix_replace_coefficients(matrix, num_rows, num_nz, &off_dia_values[0], NULL), AMGX_OK);
            }

            check_memory_usage("after matrix replace", mem_before, test_case);
            check_memory_usage("before solver resetup", mem_before, test_case);
            UNITTEST_ASSERT_EQUAL(AMGX_solver_resetup( solver, matrix ), AMGX_OK);
            check_memory_usage("after solver resetup", mem_before, test_case);
        }

        // Run several solves.
        const int num_solves = 5;
        //std::cout << "------------------------------------------" << std::endl;
        //std::cout << "RUNNING " << num_solves << " SOLVE ITERATIONS" << std::endl;
        //std::cout << "------------------------------------------" << std::endl;

        for ( int i_solve = 0 ; i_solve < num_solves ; ++i_solve )
        {
            // Create new RHS
            // Create vectors.
            AMGX_vector_handle b, x;
            check_memory_usage("before vector create", mem_before, test_case);
            UNITTEST_ASSERT_EQUAL(AMGX_vector_create( &b, rsrc, AMGX_mode_dDDI ), AMGX_OK);
            check_memory_usage("after vector create", mem_before, test_case);
            check_memory_usage("before vector create", mem_before, test_case);
            UNITTEST_ASSERT_EQUAL(AMGX_vector_create( &x, rsrc, AMGX_mode_dDDI ), AMGX_OK);
            check_memory_usage("after vector create", mem_before, test_case);
            // Upload to the GPU.
            check_memory_usage("before vector upload", mem_before, test_case);
            UNITTEST_ASSERT_EQUAL(AMGX_vector_upload( b, num_rows, bsize_y, &b_vec[0]  ), AMGX_OK);
            check_memory_usage("after vector upload", mem_before, test_case);
            // Before solve.
#ifdef SOLVE_ZERO_INI_GUESS
            UNITTEST_ASSERT_EQUAL(AMGX_vector_set_zero( x, num_rows, bsize_y ), AMGX_OK);
            check_memory_usage("before solver solve", mem_before, test_case);
            UNITTEST_ASSERT_EQUAL(AMGX_solver_solve_with_0_initial_guess( solver, b, x ), AMGX_OK);
            check_memory_usage("after solver solve", mem_before, test_case);
#else
            check_memory_usage("before vector upload", mem_before, test_case);
            UNITTEST_ASSERT_EQUAL(AMGX_vector_upload( x, num_rows, bsize_x, &x_vec[0]  ), AMGX_OK);
            check_memory_usage("after vector upload", mem_before, test_case);
            check_memory_usage("before solver solve", mem_before, test_case);
            UNITTEST_ASSERT_EQUAL(AMGX_solver_solve( solver, b, x ), AMGX_OK);
            check_memory_usage("after solver solve", mem_before, test_case);
#endif
            // Read the number of iterations.
            int num_iterations = 0;
            UNITTEST_ASSERT_EQUAL(AMGX_solver_get_iterations_number( solver, &num_iterations ), AMGX_OK);
            check_memory_usage("after get iteration number", mem_before, test_case);
            // Read the residuals and check for NaNs
            std::vector<double> res(bsize_y);

            for ( int iter = 0 ; iter < num_iterations ; ++iter )
            {
                for (int j = 0; j < bsize_y; j++)
                {
                    AMGX_solver_get_iteration_residual(solver, iter, j, &res[j]);
                    UNITTEST_ASSERT_TRUE_DESC(nan_msg.str().c_str(), res[j] == res[j]);
                }
            }

            check_memory_usage("after get iteration residual", mem_before, test_case);
            check_memory_usage("before vector destroy", mem_before, test_case);
            UNITTEST_ASSERT_EQUAL(AMGX_vector_destroy( b ), AMGX_OK);
            check_memory_usage("after vector destroy", mem_before, test_case);
            check_memory_usage("before vector destroy", mem_before, test_case);
            UNITTEST_ASSERT_EQUAL(AMGX_vector_destroy( x ), AMGX_OK);
            check_memory_usage("after vector destroy", mem_before, test_case);
        } // Solve iterations
    } // Setup iterations

    check_memory_usage("before solver destroy", mem_before, test_case);
    UNITTEST_ASSERT_EQUAL(AMGX_solver_destroy( solver ), AMGX_OK);
    check_memory_usage("after solver destroy", mem_before, test_case);
    check_memory_usage("before matrix destroy", mem_before, test_case);
    UNITTEST_ASSERT_EQUAL(AMGX_matrix_destroy( matrix ), AMGX_OK);
    check_memory_usage("after matrix destroy", mem_before, test_case);
#endif
    check_memory_usage("before cfg destroy", mem_before, test_case);
    UNITTEST_ASSERT_EQUAL(AMGX_config_destroy( cfg ), AMGX_OK);
    check_memory_usage("after cfg destroy", mem_before, test_case);
    check_memory_usage("before rsrc_cfg destroy", mem_before, test_case);
    UNITTEST_ASSERT_EQUAL(AMGX_config_destroy( rsrc_cfg ), AMGX_OK);
    check_memory_usage("after rsrc_cfg destroy", mem_before, test_case);
    test_case.max_mem_usage = old_max_mem_usage;
    check_memory_usage("before resources destroy", mem_before, test_case);
    UNITTEST_ASSERT_EQUAL(AMGX_resources_destroy( rsrc ), AMGX_OK);
    check_memory_usage("after resources destroy", mem_before, test_case);
    check_memory_usage("before finalize", mem_before, test_case);
    UNITTEST_ASSERT_EQUAL(AMGX_finalize(), AMGX_OK);
    check_memory_usage("after finalize", mem_before, test_case);
}


DECLARE_UNITTEST_END(Memory_Use_Base);

DECLARE_UNITTEST_BEGIN_EXTD(Memory_Use_DILU, Memory_Use_Base<T_Config>);

void run()
{
    TestCase temp_case;
    // List test cases
    temp_case.config_string = "config_version=2, solver(main_solver)=MULTICOLOR_DILU, block_format=ROW_MAJOR, main_solver:coloring_level=2, main_solver:max_iters=3";
    temp_case.file_name = "Public/florida/atmosdd.mtx";
    temp_case.max_mem_usage = 318; // Mb
    temp_case.max_mem_leak = 0; // bytes
    temp_case.config_string += ",";
    temp_case.config_string += "main_solver:print_solve_stats=1,";
    temp_case.config_string += "main_solver:monitor_residual=1,";
    temp_case.config_string += "determinism_flag=1,";
    temp_case.config_string += "max_uncolored_percentage=0.,";
    temp_case.config_string += "main_solver:store_res_history=1";
    Memory_Use_Base<T_Config>::launch_test_case(temp_case);
}

DECLARE_UNITTEST_END(Memory_Use_DILU);



DECLARE_UNITTEST_BEGIN_EXTD(Memory_Use_DILU2, Memory_Use_Base<T_Config>);

void run()
{
    TestCase temp_case;
    temp_case.config_string = "config_version=2, solver(main_solver)=MULTICOLOR_DILU, block_format=ROW_MAJOR, main_solver:insert_diag_while_reordering=1, main_solver:reorder_cols_by_color=1, main_solver:coloring_level=2, main_solver:max_iters=3 ";
    temp_case.file_name = "Public/florida/atmosdd.mtx";
    //  temp_case.max_mem_usage = 338.; // Mb
    temp_case.max_mem_usage = 602.; // Mb
    temp_case.max_mem_leak = 0;
    temp_case.config_string += ",";
    temp_case.config_string += "main_solver:print_solve_stats=1,";
    temp_case.config_string += "main_solver:monitor_residual=1,";
    temp_case.config_string += "determinism_flag=1,";
    temp_case.config_string += "max_uncolored_percentage=0.,";
    temp_case.config_string += "main_solver:store_res_history=1";
    Memory_Use_Base<T_Config>::launch_test_case(temp_case);
}

DECLARE_UNITTEST_END(Memory_Use_DILU2);

DECLARE_UNITTEST_BEGIN_EXTD(Memory_Use_DILU3, Memory_Use_Base<T_Config>);

void run()
{
    TestCase temp_case;
    temp_case.config_string = "config_version=2, solver(main_solver)=MULTICOLOR_DILU, block_format=ROW_MAJOR, main_solver:insert_diag_while_reordering=0, main_solver:reorder_cols_by_color=1, main_solver:coloring_level=2, main_solver:max_iters=3 ";
    temp_case.file_name = "Public/florida/atmosdd.mtx";
    temp_case.max_mem_usage = 338.; // Mb
    temp_case.max_mem_leak = 0;
    temp_case.config_string += ",";
    temp_case.config_string += "main_solver:print_solve_stats=1,";
    temp_case.config_string += "main_solver:monitor_residual=1,";
    temp_case.config_string += "determinism_flag=1,";
    temp_case.config_string += "max_uncolored_percentage=0.,";
    temp_case.config_string += "main_solver:store_res_history=1";
    Memory_Use_Base<T_Config>::launch_test_case(temp_case);
}

DECLARE_UNITTEST_END(Memory_Use_DILU3);

DECLARE_UNITTEST_BEGIN_EXTD(Memory_Use_ILU, Memory_Use_Base<T_Config>);

void run()
{
    TestCase temp_case;
    // ILU0
    temp_case.config_string = "config_version=2, solver(main_solver)=MULTICOLOR_ILU, main_solver:ilu_sparsity_level=0, block_format=ROW_MAJOR, main_solver:insert_diag_while_reordering=1, main_solver:reorder_cols_by_color=1, main_solver:coloring_level=2, main_solver:max_iters=3";
    temp_case.file_name = "Public/florida/atmosdd.mtx";
    //temp_case.max_mem_usage = 518.; // Mb
    temp_case.max_mem_usage = 792.; // Mb
    temp_case.max_mem_leak = 0;
    temp_case.config_string += ",";
    temp_case.config_string += "main_solver:print_solve_stats=1,";
    temp_case.config_string += "main_solver:monitor_residual=1,";
    temp_case.config_string += "determinism_flag=1,";
    temp_case.config_string += "max_uncolored_percentage=0.,";
    temp_case.config_string += "main_solver:store_res_history=1";
    Memory_Use_Base<T_Config>::launch_test_case(temp_case);
}

DECLARE_UNITTEST_END(Memory_Use_ILU);

DECLARE_UNITTEST_BEGIN_EXTD(Memory_Use_ILU2, Memory_Use_Base<T_Config>);

void run()
{
    TestCase temp_case;
    // ILU1
    temp_case.config_string = "config_version=2, solver(main_solver)=MULTICOLOR_ILU, main_solver:ilu_sparsity_level=1, block_format=ROW_MAJOR, main_solver:insert_diag_while_reordering=1, main_solver:reorder_cols_by_color=1, main_solver:coloring_level=2, main_solver:max_iters=3";
    temp_case.file_name = "Public/florida/atmosdd.mtx";
    temp_case.max_mem_usage = 718.; // Mb
    temp_case.max_mem_leak = 0;
    temp_case.config_string += ",";
    temp_case.config_string += "main_solver:print_solve_stats=1,";
    temp_case.config_string += "main_solver:monitor_residual=1,";
    temp_case.config_string += "determinism_flag=1,";
    temp_case.config_string += "max_uncolored_percentage=0.,";
    temp_case.config_string += "main_solver:store_res_history=1";
    Memory_Use_Base<T_Config>::launch_test_case(temp_case);
}

DECLARE_UNITTEST_END(Memory_Use_ILU2);


DECLARE_UNITTEST_BEGIN_EXTD(Memory_Use_ILU3, Memory_Use_Base<T_Config>);

void run()
{
    TestCase temp_case;
    // ILU1
    temp_case.config_string = "config_version=2, solver(main_solver)=MULTICOLOR_ILU, main_solver:ilu_sparsity_level=1, block_format=COL_MAJOR, main_solver:insert_diag_while_reordering=0, main_solver:reorder_cols_by_color=0, main_solver:coloring_level=2, main_solver:max_iters=5";
    temp_case.file_name = "Public/florida/atmosdd.mtx";
    temp_case.max_mem_usage = 1368.; // Mb
    temp_case.max_mem_leak = 0;
    temp_case.config_string += ",";
    temp_case.config_string += "main_solver:print_solve_stats=1,";
    temp_case.config_string += "main_solver:monitor_residual=1,";
    temp_case.config_string += "determinism_flag=1,";
    temp_case.config_string += "max_uncolored_percentage=0.,";
    temp_case.config_string += "main_solver:store_res_history=1";
    Memory_Use_Base<T_Config>::launch_test_case(temp_case);
}

DECLARE_UNITTEST_END(Memory_Use_ILU3);


DECLARE_UNITTEST_BEGIN_EXTD(Memory_Use_ILU4, Memory_Use_Base<T_Config>);

void run()
{
    TestCase temp_case;
    // ILU1
    temp_case.config_string = "config_version=2, solver(main_solver)=MULTICOLOR_ILU, main_solver:ilu_sparsity_level=1, block_format=COL_MAJOR, main_solver:insert_diag_while_reordering=0, main_solver:reorder_cols_by_color=0, main_solver:coloring_level=2, main_solver:max_iters=5";
    temp_case.file_name = "Public/florida/atmosdd.mtx";
    temp_case.max_mem_usage = 5068.; // Mb
    temp_case.max_mem_leak = 0;
    temp_case.config_string += ",";
    temp_case.config_string += "main_solver:print_solve_stats=1,";
    temp_case.config_string += "main_solver:monitor_residual=1,";
    temp_case.config_string += "determinism_flag=1,";
    temp_case.config_string += "max_uncolored_percentage=0.,";
    temp_case.config_string += "main_solver:store_res_history=1";
    Memory_Use_Base<T_Config>::launch_test_case(temp_case);
}

DECLARE_UNITTEST_END(Memory_Use_ILU4);


DECLARE_UNITTEST_BEGIN_EXTD(Memory_Use_atmosmodd_pressure, Memory_Use_Base<T_Config>);

void run()
{
    std::vector<TestCase> test_cases;
    TestCase temp_case;
    std::ostringstream cfg_options;
    cfg_options << "config_version=2,";
    cfg_options << "solver=PBICGSTAB,";
    cfg_options << "max_iters=1,";
    cfg_options << "convergence=RELATIVE_INI_CORE,";
    cfg_options << "tolerance=1.0e-4,";
    cfg_options << "preconditioner(amg_solver)=AMG,";
    cfg_options << "amg_solver:algorithm=CLASSICAL,";
    cfg_options << "amg_solver:coarseAgenerator=THRUST,";
    cfg_options << "amg_solver:strength_threshold=0.25,";
    cfg_options << "amg_solver:max_levels=8,";
    cfg_options << "amg_solver:max_iters=1,";
    cfg_options << "amg_solver:smoother(amg_smoother)=BLOCK_JACOBI,";
    cfg_options << "amg_smoother:relaxation_factor=0.6,";
    cfg_options << "amg_solver:interpolator=D2,";
    cfg_options << "amg_solver:spmm_no_sort=1,";
    cfg_options << "amg_solver:monitor_residual=0,";
    cfg_options << "amg_solver:print_grid_stats=1,";
    cfg_options << "amg_solver:coarsest_sweeps=1,";
    cfg_options << "monitor_residual=1,";
    cfg_options << "print_solve_stats=1,";
    cfg_options << "store_res_history=1";
    temp_case.config_string = cfg_options.str();
    temp_case.file_name = "Public/Florida/atmosmodd.mtx";
    //  temp_case.max_mem_usage = 1068.; // Mb
    temp_case.max_mem_usage = 1306.; // Mb
    temp_case.max_mem_leak = 0.05;
    Memory_Use_Base<T_Config>::launch_test_case(temp_case);
}

DECLARE_UNITTEST_END(Memory_Use_atmosmodd_pressure);

Memory_Use_DILU<TemplateMode<AMGX_mode_dDDI>::Type> Memory_Use_DILU_dDDI;
Memory_Use_DILU2<TemplateMode<AMGX_mode_dDDI>::Type> Memory_Use_DILU2_dDDI;
Memory_Use_DILU3<TemplateMode<AMGX_mode_dDDI>::Type> Memory_Use_DILU3_dDDI;
Memory_Use_ILU<TemplateMode<AMGX_mode_dDDI>::Type> Memory_Use_ILU_dDDI;
Memory_Use_ILU2<TemplateMode<AMGX_mode_dDDI>::Type> Memory_Use_ILU2_dDDI;
Memory_Use_ILU3<TemplateMode<AMGX_mode_dDDI>::Type> Memory_Use_ILU3_dDDI;
//Memory_Use_ILU4<TemplateMode<AMGX_mode_dDDI>::Type> Memory_Use_ILU4_dDDI;
Memory_Use_atmosmodd_pressure<TemplateMode<AMGX_mode_dDDI>::Type> Memory_Use_atmosmodd_pressure_dDDI;


// if you want to be able run this test for all available configs you can write this:
//#define AMGX_CASE_LINE(CASE) TemplateTest <TemplateMode<CASE>::Type>  TemplateTest_##CASE;
//  AMGX_FORALL_BUILDS(AMGX_CASE_LINE)
//#undef AMGX_CASE_LINE

// or run for all device configs
//#define AMGX_CASE_LINE(CASE) TemplateTest <TemplateMode<CASE>::Type>  TemplateTest_##CASE;
//  AMGX_FORALL_BUILDS_DEVICE(AMGX_CASE_LINE)
//#undef AMGX_CASE_LINE

//Memory_Use <TemplateMode<AMGX_mode_dDDI>::Type>  Memory_Use_instance_mode_dDDI;

// or you can specify several desired configs
//TemplateTest <TemplateMode<AMGX_mode_hDFI>::Type>  TemplateTest_hDFI;
//TemplateTest <TemplateMode<AMGX_mode_dDFI>::Type>  TemplateTest_dDFI;


} // namespace amgx



