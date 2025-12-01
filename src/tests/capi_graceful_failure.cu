// SPDX-FileCopyrightText: 2011 - 2025 NVIDIA CORPORATION. All Rights Reserved.
//
// SPDX-License-Identifier: BSD-3-Clause

#include "unit_test.h"
#include "amgx_c.h"
#include "amgxP_c.h"

namespace amgx

{

// This unit test prints a bunch of error messages (when handling expected errors with C API usage) output possibly trashing unit tests framework 

// parameter is used as test name
DECLARE_UNITTEST_BEGIN(CAPIFailure);

void default_output(const char *msg, int length)
{
    printf("%s", msg);
}

void run()
{
    randomize( 666 );
// config routines
    AMGX_config_handle cfg = NULL;
    UNITTEST_ASSERT_NEQUAL(AMGX_config_create_from_file(&cfg, "nonexisting_file.cfg"), AMGX_RC_OK);
    UNITTEST_ASSERT_NEQUAL(AMGX_config_create_from_file_and_string(&cfg, "nonexisting_file.cfg", "determinim_flag=1"), AMGX_RC_OK);
    UNITTEST_ASSERT_NEQUAL(AMGX_config_create_from_file_and_string(&cfg, "nonexisting_file.cfg", (const char *)(NULL)), AMGX_RC_OK);
    UNITTEST_ASSERT_NEQUAL(AMGX_config_create(&cfg, (const char *)(NULL)), AMGX_RC_OK);
    UNITTEST_ASSERT_EQUAL(AMGX_config_create(&cfg, "determinism_flag=1"), AMGX_RC_OK);
    UNITTEST_ASSERT_EQUAL(AMGX_config_destroy(cfg), AMGX_RC_OK);
    UNITTEST_ASSERT_EQUAL(AMGX_config_create(&cfg, "determinism_flag=1"), AMGX_RC_OK);
// matrix&vector create routines
    AMGX_matrix_handle CA = NULL;
    AMGX_vector_handle Cb = NULL;
    AMGX_vector_handle Csoln = NULL;
    //AMGX_vector_handle Cx = NULL;
    Matrix<TConfig_h> A;
    Vector<TConfig_h> b;
    Vector<TConfig_h> x;
    Vector<TConfig_h> soln;
    AMGX_config_handle rsrc_cfg = NULL;
    UNITTEST_ASSERT_NEQUAL(AMGX_config_create(&rsrc_cfg, "bad_STRING = = = -1"), AMGX_RC_OK);
    UNITTEST_ASSERT_EQUAL(AMGX_config_create(&rsrc_cfg, ""), AMGX_RC_OK);
    int device = 0;
    AMGX_resources_handle rsrc = NULL;
    UNITTEST_ASSERT_EQUAL(AMGX_resources_create(&rsrc, rsrc_cfg, NULL, 1, &device), AMGX_RC_OK);
    //UNITTEST_ASSERT_TRUE(AMGX_matrix_destroy(CA) != AMGX_RC_OK); segfault
    UNITTEST_ASSERT_NEQUAL(AMGX_matrix_create(&CA, rsrc, AMGX_Mode(-1)), AMGX_RC_OK);
    UNITTEST_ASSERT_EQUAL(AMGX_matrix_create(&CA, rsrc, AMGX_mode_dDDI), AMGX_RC_OK);
    UNITTEST_ASSERT_NEQUAL(AMGX_vector_create(&Cb, rsrc, AMGX_Mode(-1)), AMGX_RC_OK);
    UNITTEST_ASSERT_EQUAL(AMGX_vector_create(&Cb, rsrc, AMGX_mode_dDDI), AMGX_RC_OK);
    UNITTEST_ASSERT_NEQUAL(AMGX_vector_create(&Csoln, rsrc, AMGX_Mode(-1)), AMGX_RC_OK);
    UNITTEST_ASSERT_EQUAL(AMGX_vector_create(&Csoln, rsrc, AMGX_mode_dDDI), AMGX_RC_OK);

    // write temp matrix
    std::string filename = ".temp_matrix.mtx";
    {
        Matrix<TConfig_h> tA;
        generatePoissonForTest(tA, 1, 0, 27, 20, 20, 20);
        UNITTEST_ASSERT_TRUE(MatrixIO<TConfig_h>::writeSystemMatrixMarket(filename.c_str(), &tA, NULL, NULL));
    }

// matrix manipulations
    UNITTEST_ASSERT_NEQUAL(AMGX_read_system(CA, Cb, Csoln, "nonexisting_file.mtx"), AMGX_RC_OK);
    UNITTEST_ASSERT_EQUAL(AMGX_read_system(CA, Cb, Csoln, filename.c_str()), AMGX_RC_OK);
    UNITTEST_ASSERT_TRUE(read_system(filename.c_str(), A, b, x, true));
    int nrows, blockdimx, blockdimy;
    UNITTEST_ASSERT_EQUAL(AMGX_matrix_get_size(CA, &nrows, &blockdimx, &blockdimy), AMGX_RC_OK); // how should we retrieve numnz?
    UNITTEST_ASSERT_EQUAL(A.get_num_rows(), nrows);
    UNITTEST_ASSERT_EQUAL(A.get_block_dimx(), blockdimx);
    UNITTEST_ASSERT_EQUAL(A.get_block_dimy(), blockdimy);

    UNITTEST_ASSERT_EQUAL(AMGX_matrix_upload_all(CA, nrows, A.get_num_nz(), blockdimx, blockdimy, (int *)(A.row_offsets.raw()), (int *)(A.col_indices.raw()), (void *)(A.values.raw()), (void *)(A.diag.raw())), AMGX_RC_OK);
    UNITTEST_ASSERT_NEQUAL(AMGX_matrix_upload_all(CA, -1, A.get_num_nz(), blockdimx, blockdimy, (int *)(A.row_offsets.raw()), (int *)(A.col_indices.raw()), (void *)(A.values.raw()), (void *)(A.diag.raw())), AMGX_RC_OK);
    UNITTEST_ASSERT_EQUAL(AMGX_matrix_destroy(CA), AMGX_RC_OK);
    UNITTEST_ASSERT_EQUAL(AMGX_matrix_create(&CA, rsrc, AMGX_mode_dDDI), AMGX_RC_OK);
    UNITTEST_ASSERT_EQUAL(AMGX_read_system(CA, Cb, Csoln, filename.c_str()), AMGX_RC_OK);
    UNITTEST_ASSERT_EQUAL( AMGX_matrix_replace_coefficients(CA, nrows, A.get_num_nz(), (void *)(NULL), (void *)(A.values.raw() + A.get_num_nz()*A.get_block_size()) ), AMGX_RC_OK ); // NULLs should be ok
    UNITTEST_ASSERT_EQUAL(AMGX_matrix_destroy(CA), AMGX_RC_OK);
    UNITTEST_ASSERT_NEQUAL(AMGX_read_system(CA, Cb, Csoln, filename.c_str()), AMGX_RC_OK);
    UNITTEST_ASSERT_EQUAL(AMGX_matrix_create(&CA, rsrc, AMGX_mode_dDDI), AMGX_RC_OK);
    UNITTEST_ASSERT_EQUAL(AMGX_read_system(CA, Cb, Csoln, filename.c_str()), AMGX_RC_OK);
    UNITTEST_ASSERT_EQUAL(AMGX_matrix_replace_coefficients(CA, nrows, A.get_num_nz(), (void *)(A.values.raw()), (void *)(A.values.raw() + A.get_num_nz()*A.get_block_size()) ), AMGX_RC_OK );
    UNITTEST_ASSERT_NEQUAL(AMGX_write_system(CA, Cb, NULL, (const char *)(NULL)), AMGX_RC_OK);
    //UNITTEST_ASSERT_NEQUAL(AMGX_write_system(CA, (AMGX_vector_handle)(NULL), "test.mtx"), AMGX_RC_OK);
    UNITTEST_ASSERT_NEQUAL(AMGX_write_system(AMGX_matrix_handle(NULL), Cb, NULL, ".temp_matrix.mtx"), AMGX_RC_OK);
// solver manipulations
    AMGX_solver_handle solver = NULL;
    UNITTEST_ASSERT_NEQUAL(AMGX_solver_create(&solver, rsrc, (AMGX_Mode)(-1), cfg), AMGX_RC_OK);
    UNITTEST_ASSERT_EQUAL(AMGX_solver_create(&solver, rsrc, AMGX_mode_dDDI, cfg), AMGX_RC_OK);
    int niters;
    AMGX_SOLVE_STATUS status;
    AMGX_solver_handle null_solver = NULL;
    double res;
    UNITTEST_ASSERT_EQUAL(AMGX_solver_get_iterations_number(solver, &niters), AMGX_RC_OK);
    UNITTEST_ASSERT_NEQUAL(AMGX_solver_get_iteration_residual(solver, 0, 0, &res), AMGX_RC_OK); // because residual flag wasn't provided
    UNITTEST_ASSERT_EQUAL(AMGX_solver_get_status(solver, &status), AMGX_RC_OK);
    UNITTEST_ASSERT_NEQUAL(AMGX_solver_get_status(null_solver, &status), AMGX_RC_OK);
    UNITTEST_ASSERT_EQUAL(status, AMGX_SOLVE_FAILED);
    UNITTEST_ASSERT_EQUAL(AMGX_solver_destroy(solver), AMGX_RC_OK);
    UNITTEST_ASSERT_EQUAL(AMGX_config_destroy(cfg), AMGX_RC_OK);
    UNITTEST_ASSERT_NEQUAL(AMGX_solver_destroy(null_solver), AMGX_RC_OK);
    UNITTEST_ASSERT_NEQUAL(AMGX_config_destroy(AMGX_config_handle(NULL)), AMGX_RC_OK);
// vector manipulations
    int nrows1, /*bsizex1,*/ bsizey1;
    UNITTEST_ASSERT_NEQUAL(AMGX_vector_set_zero(Cb, -1, blockdimy), AMGX_RC_OK);
    UNITTEST_ASSERT_EQUAL(AMGX_vector_destroy(Cb), AMGX_RC_OK);
    UNITTEST_ASSERT_EQUAL(AMGX_vector_create(&Cb, rsrc, AMGX_mode_dDDI), AMGX_RC_OK);
    UNITTEST_ASSERT_EQUAL(AMGX_vector_set_zero(Cb, 2 * nrows, blockdimy), AMGX_RC_OK);
    UNITTEST_ASSERT_EQUAL(AMGX_vector_get_size(Cb, &nrows1, &bsizey1), AMGX_RC_OK);
    UNITTEST_ASSERT_EQUAL(nrows1, 2 * nrows);
//  UNITTEST_ASSERT_EQUAL(bsizex1, blockdimx);
    UNITTEST_ASSERT_EQUAL(bsizey1, blockdimy);
    UNITTEST_ASSERT_EQUAL(AMGX_vector_upload(Cb, nrows, blockdimy, b.raw()), AMGX_RC_OK);
    UNITTEST_ASSERT_EQUAL(AMGX_vector_destroy(Cb), AMGX_RC_OK);
    UNITTEST_ASSERT_EQUAL(AMGX_matrix_destroy(CA), AMGX_RC_OK);
    UNITTEST_ASSERT_NEQUAL(AMGX_vector_destroy(AMGX_vector_handle(NULL)), AMGX_RC_OK);
    UNITTEST_ASSERT_NEQUAL(AMGX_matrix_destroy(AMGX_matrix_handle(NULL)), AMGX_RC_OK);
    char *version;
    char *date;
    char *time;
    UNITTEST_ASSERT_EQUAL(AMGX_get_build_info_strings(&version, &date, &time), AMGX_RC_OK);
}

DECLARE_UNITTEST_END(CAPIFailure);


// if you want to be able run this test for all available configs you can write this:
//#define AMGX_CASE_LINE(CASE) SampleTest <TemplateMode<CASE>::Type>  TemplateTest_##CASE;
//  AMGX_FORALL_BUILDS(AMGX_CASE_LINE)
//#undef AMGX_CASE_LINE

// or run for all device configs
//#define AMGX_CASE_LINE(CASE) SampleTest <TemplateMode<CASE>::Type>  TemplateTest_##CASE;
//  AMGX_FORALL_BUILDS_DEVICE(AMGX_CASE_LINE)
//#undef AMGX_CASE_LINE

// or you can specify several desired configs
CAPIFailure <TemplateMode<AMGX_mode_dDDI>::Type>  CAPIFailure_dDDI;


} //namespace amgx
