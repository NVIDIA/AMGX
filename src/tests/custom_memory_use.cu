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
#include "time.h"
#include <sstream>

#define NUM_NL_ITERS    5

//#define DEBUG

namespace amgx
{

template <class Handle>
struct CWrapper
{
    AMGX_Mode mode;
    Handle hdl;
};


// parameter is used as test name
DECLARE_UNITTEST_BEGIN(TestMemoryUse);

std::string base_keywords()
{
    return "stress";
}


struct TestCase
{
    std::vector<std::string> file_names;
    std::string config_string;
    double max_mem_usage;
    double max_mem_leak;
};

double get_memory_usage()
{
    size_t free_mem, total_mem;
    cudaMemGetInfo( &free_mem, &total_mem );
    size_t used_mem = total_mem - free_mem;
    used_mem /= size_t(1024);
    double used_mem_f = double(used_mem) / 1024.0;
    return used_mem_f;
}


void check_memory_usage(const char *msg, double &mem_before, TestCase &test_case, int it)
{
    std::stringstream mem_msg;
    mem_msg  << "Maximum used memory exceeds limit set to "
             << test_case.max_mem_usage << " for matrix " << test_case.file_names[it];
    double used_mem_f = get_memory_usage();
    UNITTEST_ASSERT_TRUE_DESC(mem_msg.str().c_str(), (used_mem_f - mem_before) < test_case.max_mem_usage);
#ifdef DEBUG
    std::cout << msg << ", memory usage =: " << used_mem_f - mem_before << " Mb" << std::endl;
#endif
}

void load_data(TestCase &test_case, double &mem_before, int it, int &num_rows, int &num_nz, int &bsize_x, int &bsize_y, int &bsize,
               std::vector<int> &row_offsets,
               std::vector<int> &col_indices,
               std::vector<double> &off_dia_values,
               std::vector<double> &dia_values,
               std::vector<double> &x_vec,
               std::vector<double> &b_vec)
{
    // --------------------------------------
    // Create matrix arrays from file
    // --------------------------------------
    Matrix_h Atemp;
    Vector_h btemp, xtemp;
    // -------------------------------------------
    // Read the matrix
    // -------------------------------------------
    std::string fail_msg = "Cannot open " + test_case.file_names[it];
    this->PrintOnFail(fail_msg.c_str());
    UNITTEST_ASSERT_TRUE(this->read_system(test_case.file_names[it].c_str(), Atemp, btemp, xtemp));
    bool hasDiag = Atemp.hasProps(DIAG);
    num_rows = Atemp.get_num_rows();
    num_nz = Atemp.get_num_nz();
    bsize_x = Atemp.get_block_dimx();
    bsize_y = Atemp.get_block_dimy();
    bsize = bsize_x * bsize_y;

    // Create row_offsets, col_indices, off_dia_values and dia_values arrays from the matrix just rea
    if (xtemp.size() == 0) { xtemp.resize(num_rows * bsize_y, 0.); }

    row_offsets.resize(num_rows + 1);
    col_indices.resize(num_nz);
    off_dia_values.resize(num_nz * bsize);

    if (hasDiag)
    {
        dia_values.resize(num_rows * bsize);
    }

    x_vec.resize(num_rows * bsize_y);
    b_vec.resize(num_rows * bsize_x);
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
}

void mem_test_main(TestCase &test_case, double &mem_before)
{
    int mat_it = 0;
    check_memory_usage("before initialize", mem_before, test_case, mat_it);
    AMGX_initialize();
    check_memory_usage("after initialize", mem_before, test_case, mat_it);
    AMGX_config_handle rsrc_cfg = NULL;
    UNITTEST_ASSERT_EQUAL(AMGX_config_create(&rsrc_cfg, ""), AMGX_OK);
    // Choosing device 0
    int device = 0;
    AMGX_resources_handle rsrc = NULL;
    UNITTEST_ASSERT_EQUAL(AMGX_resources_create(&rsrc, rsrc_cfg, NULL, 1, &device), AMGX_OK);
    // query device pool size
    amgx::CWrapper<AMGX_resources_handle> *c_resources = (amgx::CWrapper<AMGX_resources_handle> *)rsrc;
    double old_max_mem_usage = test_case.max_mem_usage;
    test_case.max_mem_usage += ((Resources *)c_resources->hdl)->getPoolSize() / 1024.0 / 1024.0;
    check_memory_usage("after resources ", mem_before, test_case, mat_it);
    int num_rows, num_nz;
    int bsize_x, bsize_y, bsize;
    std::vector<int> row_offsets;
    std::vector<int> col_indices;
    std::vector<double> off_dia_values;
    std::vector<double> dia_values;
    std::vector<double> x_vec;
    std::vector<double> b_vec;
    // All of these should create the same result
    std::string option_string = test_case.config_string;
    bool fine_level_recreated = true;
    AMGX_matrix_handle matrix = NULL;
    AMGX_vector_handle b = NULL;
    AMGX_vector_handle x = NULL;

    // run multiple non-linear iterations
    for ( int it = 0; it < NUM_NL_ITERS; it++ )
    {
        mat_it = it % test_case.file_names.size();
        load_data(test_case, mem_before, mat_it, num_rows, num_nz, bsize_x, bsize_y, bsize, row_offsets, col_indices, off_dia_values, dia_values, x_vec, b_vec);

        if ( fine_level_recreated )
        {
            if ( matrix != NULL )
            {
                check_memory_usage("before matrix destroy", mem_before, test_case, mat_it);
                AMGX_matrix_destroy( matrix );
                check_memory_usage("after matrix destroy", mem_before, test_case, mat_it);
            }

            if ( b != NULL )
            {
                check_memory_usage("before vector destroy", mem_before, test_case, mat_it);
                AMGX_vector_destroy( b );
                check_memory_usage("after vector destroy", mem_before, test_case, mat_it);
            }

            if ( x != NULL )
            {
                check_memory_usage("before vector destroy", mem_before, test_case, mat_it);
                AMGX_vector_destroy( x );
                check_memory_usage("after vector destroy", mem_before, test_case, mat_it);
            }

            check_memory_usage("before matrix create", mem_before, test_case, mat_it);
            AMGX_matrix_create( &matrix, rsrc, AMGX_mode_dDDI );
            check_memory_usage("after matrix create", mem_before, test_case, mat_it);
            check_memory_usage("before vector create", mem_before, test_case, mat_it);
            AMGX_vector_create( &b, rsrc, AMGX_mode_dDDI );
            check_memory_usage("after vector create", mem_before, test_case, mat_it);
            check_memory_usage("before vector create", mem_before, test_case, mat_it);
            AMGX_vector_create( &x, rsrc, AMGX_mode_dDDI );
            check_memory_usage("after vector create", mem_before, test_case, mat_it);
        }

        // fill matrix
        check_memory_usage("before matrix upload", mem_before, test_case, mat_it);

        if ( fine_level_recreated )
        {
            AMGX_matrix_upload_all(matrix, num_rows, num_nz, bsize_x, bsize_y, &row_offsets[0], &col_indices[0], &off_dia_values[0], &dia_values[0]);
        }
        else
        {
            AMGX_matrix_replace_coefficients(matrix, num_rows, num_nz, &off_dia_values[0], &dia_values[0]);
        }

        check_memory_usage("after matrix upload", mem_before, test_case, mat_it);
        // fill b
        check_memory_usage("before vector-b upload", mem_before, test_case, mat_it);
        AMGX_vector_upload( b, num_rows, bsize_y, &b_vec[0] );
        check_memory_usage("after vector-b upload", mem_before, test_case, mat_it);
        // fill x
        check_memory_usage("before vector-x upload", mem_before, test_case, mat_it);
        AMGX_vector_upload( x, num_rows, bsize_x, &x_vec[0]  );
        check_memory_usage("after vector-x upload", mem_before, test_case, mat_it);
        double used_mem_before = get_memory_usage();
        AMGX_config_handle cfg;
        check_memory_usage("before config create", mem_before, test_case, mat_it);
        UNITTEST_ASSERT_EQUAL(AMGX_config_create( &cfg, option_string.c_str() ), AMGX_OK);
        check_memory_usage("after config create", mem_before, test_case, mat_it);
        // verbose = 2
        AMGX_solver_handle solver;
        check_memory_usage("before solver create", mem_before, test_case, mat_it);
        AMGX_solver_create( &solver, rsrc, AMGX_mode_dDDI, cfg);
        check_memory_usage("after solver create", mem_before, test_case, mat_it);
        // solver setup
        check_memory_usage("before solver setup", mem_before, test_case, mat_it);
        AMGX_solver_setup( solver, matrix );
        check_memory_usage("after solver setup", mem_before, test_case, mat_it);
        // solver solve
        check_memory_usage("before solver solve", mem_before, test_case, mat_it);
        AMGX_solver_solve( solver, b, x );
        check_memory_usage("after solver solve", mem_before, test_case, mat_it);
        // copy solution vector
        check_memory_usage("before vector copy", mem_before, test_case, mat_it);
        AMGX_vector_download( x, &x_vec[0] );
        check_memory_usage("after vector copy", mem_before, test_case, mat_it);
        // read the number of iterations
        int num_iterations = 0;
        AMGX_solver_get_iterations_number( solver, &num_iterations );
        check_memory_usage("after get iteration number", mem_before, test_case, mat_it);
        // read the residuals and check for NaNs
        double res[] = { 0.0, 0.0, 0.0, 0.0 };
        AMGX_solver_get_iteration_residual( solver, 0, 0, &res[0] );
        AMGX_solver_get_iteration_residual( solver, 0, 1, &res[1] );
        AMGX_solver_get_iteration_residual( solver, 0, 2, &res[2] );
        AMGX_solver_get_iteration_residual( solver, 0, 3, &res[3] );
        check_memory_usage("after get iteration residual", mem_before, test_case, mat_it);
        check_memory_usage("before solver destroy", mem_before, test_case, mat_it);
        AMGX_solver_destroy( solver );
        check_memory_usage("after solver destroy", mem_before, test_case, mat_it);
        check_memory_usage("before cfg destroy", mem_before, test_case, mat_it);
        AMGX_config_destroy( cfg );
        check_memory_usage("after cfg destroy", mem_before, test_case, mat_it);
        double used_mem_after = get_memory_usage();

        if (it == 0)
        {
            int num_rows;
            int block_dimx;
            int block_dimy;
            AMGX_matrix_get_size(matrix, &num_rows, &block_dimx, &block_dimy);
            // get size of the coloring
            check_memory_usage("before coloring resize", mem_before, test_case, it);
            typename Matrix_d::IVector coloring;
            coloring.resize(num_rows * 2);
            check_memory_usage("after coloring resize", mem_before, test_case, it);
            double coloring_size = (get_memory_usage() - used_mem_after);
            check_memory_usage("before coloring clear and shrink-to-fit", mem_before, test_case, it);
            coloring.clear();
            coloring.shrink_to_fit();
            check_memory_usage("after coloring clear and shrink-to-fit", mem_before, test_case, it);
            used_mem_after -= coloring_size;
            // account for memory pool creation
            used_mem_after -= 32;
#ifdef DEBUG
            std::cout << "Coloring size= " << coloring_size << std::endl;
            //std::cout << "num_rows= " << num_rows << std::endl;
#endif
        }

        fine_level_recreated = false;
        double memLeak = used_mem_after - used_mem_before;
        std::stringstream msg;
        msg << "Memory leak after " << it << " iteration: "
            << memLeak << " Mb exceed threshold set to "
            << test_case.max_mem_leak ;
#ifdef DEBUG
        std::cout << "Mem leak = " << memLeak
                  << ", after " << it << " iteration "
                  << std::endl;
                << std::endl;
#endif
        UNITTEST_ASSERT_TRUE_DESC(msg.str().c_str(), memLeak - 100 <= test_case.max_mem_leak);
    }

    // clean-up matrix and vectors
    if ( matrix != NULL )
    {
        check_memory_usage("before matrix destroy", mem_before, test_case, mat_it);
        AMGX_matrix_destroy( matrix );
        check_memory_usage("after matrix destroy", mem_before, test_case, mat_it);
    }

    if ( b != NULL )
    {
        check_memory_usage("before vector destroy", mem_before, test_case, mat_it);
        AMGX_vector_destroy( b );
        check_memory_usage("after vector destroy", mem_before, test_case, mat_it);
    }

    if ( x != NULL )
    {
        check_memory_usage("before vector destroy", mem_before, test_case, mat_it);
        AMGX_vector_destroy( x );
        check_memory_usage("after vector destroy", mem_before, test_case, mat_it);
    }

    test_case.max_mem_usage = old_max_mem_usage;
    check_memory_usage("before resources free", mem_before, test_case, mat_it);
    AMGX_config_destroy(rsrc_cfg);
    AMGX_resources_destroy(rsrc);
    check_memory_usage("after resources free", mem_before, test_case, mat_it);
    check_memory_usage("before finalize", mem_before, test_case, mat_it);
    AMGX_finalize();
    check_memory_usage("after finalize", mem_before, test_case, mat_it);
}

void run()
{
    std::vector<TestCase> test_cases;
    TestCase temp_case;
    // List test cases
    std::ostringstream cfg_options;
    cfg_options << "config_version=1,";
    cfg_options << "algorithm=AGGREGATION,";
    cfg_options << "coarseAgenerator=LOW_DEG,";
    cfg_options << "coloring_level=1,";
    cfg_options << "convergence=RELATIVE_MAX,";
    cfg_options << "cycle=V,";
    cfg_options << "determinism_flag=1,";
    cfg_options << "matrix_coloring_scheme=MIN_MAX,";
    cfg_options << "max_iters=30,";
    cfg_options << "max_levels=21,";
    cfg_options << "min_block_rows=20,";
    cfg_options << "norm=L1,";
    cfg_options << "postsweeps=3,";
    cfg_options << "presweeps=0,";
    cfg_options << "selector=ONE_PHASE_HANDSHAKING,";
    cfg_options << "smoother=MULTICOLOR_DILU,";
    cfg_options << "smoother_weight=0.9,";
    cfg_options << "tolerance=0.1,";
    cfg_options << "monitor_residual=1,";
    cfg_options << "print_solve_stats=1,";
    cfg_options << "store_res_history=1,";
    temp_case.config_string = cfg_options.str();
    temp_case.file_names.clear();
    temp_case.file_names.push_back("Public/florida/atmosdd.mtx");
    temp_case.max_mem_usage = 2066; // Mb
    temp_case.max_mem_leak = 0;
    test_cases.push_back(temp_case);
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

    for (int i = 0; i < test_cases.size(); i++)
    {
        double used_mem_before = get_memory_usage();
        mem_test_main(test_cases[i], used_mem_before);
        double used_mem_after = get_memory_usage();
        double memLeak = used_mem_after - used_mem_before;
#ifdef DEBUG
        std::cout << "Mem leak = " << memLeak
                  << std::endl;
#endif
        std::stringstream msg;
        msg << "Memory leak: " << memLeak
            << " Mb exceed threshold set to " << test_cases[i].max_mem_leak ;
        UNITTEST_ASSERT_TRUE_DESC(msg.str().c_str(), memLeak <= test_cases[i].max_mem_leak);
    }

    AMGX_initialize();
    UnitTest::amgx_intialized = true;
}

DECLARE_UNITTEST_END(TestMemoryUse);


// if you want to be able run this test for all available configs you can write this:
//#define AMGX_CASE_LINE(CASE) TemplateTest <TemplateMode<CASE>::Type>  TemplateTest_##CASE;
//  AMGX_FORALL_BUILDS(AMGX_CASE_LINE)
//#undef AMGX_CASE_LINE

// or run for all device configs
//#define AMGX_CASE_LINE(CASE) TemplateTest <TemplateMode<CASE>::Type>  TemplateTest_##CASE;
//  AMGX_FORALL_BUILDS_DEVICE(AMGX_CASE_LINE)
//#undef AMGX_CASE_LINE

//TestMemoryUse <TemplateMode<AMGX_mode_dDDI>::Type>  TestMemoryUse_instance_mode_dDDI;

// or you can specify several desired configs
//TemplateTest <TemplateMode<AMGX_mode_hDFI>::Type>  TemplateTest_hDFI;
//TemplateTest <TemplateMode<AMGX_mode_dDFI>::Type>  TemplateTest_dDFI;


}



