// SPDX-FileCopyrightText: 2011 - 2025 NVIDIA CORPORATION. All Rights Reserved.
//
// SPDX-License-Identifier: BSD-3-Clause

#include "unit_test.h"
#include "amgx_c.h"
#include <vector>

namespace amgx

{

// Writes a system in the binary format and reads it back: every entry of the rhs
// and of the initial guess, including the last one, has to survive the round trip.
DECLARE_UNITTEST_BEGIN(WriteSystemBinaryVectors);

void check_vector(AMGX_vector_handle v, const double *ref, int ref_n)
{
    int n = 0, block_dim = 0;
    UNITTEST_ASSERT_EQUAL(AMGX_vector_get_size(v, &n, &block_dim), AMGX_RC_OK);
    UNITTEST_ASSERT_EQUAL(n * block_dim, ref_n);
    std::vector<double> data(ref_n, 0.);
    UNITTEST_ASSERT_EQUAL(AMGX_vector_download(v, &data[0]), AMGX_RC_OK);

    for (int i = 0; i < ref_n; i++)
    {
        UNITTEST_ASSERT_EQUAL(data[i], ref[i]);
    }
}

void run()
{
    const char *filename = ".temp_matrix.bin";
    const int n = 3, nnz = 7;
    const int row_ptrs[] = {0, 2, 5, 7};
    const int col_indices[] = {0, 1, 0, 1, 2, 1, 2};
    const double values[] = {2., -1., -1., 2., -1., -1., 2.};
    const double ref_b[] = {10., 20., 30.};
    const double ref_x[] = {0.25, 0.5, 0.75};
    AMGX_config_handle cfg = NULL;
    AMGX_resources_handle rsrc = NULL;
    AMGX_matrix_handle A = NULL;
    AMGX_vector_handle b = NULL;
    AMGX_vector_handle x = NULL;
    UNITTEST_ASSERT_EQUAL(AMGX_config_create(&cfg, "matrix_writer=binary"), AMGX_RC_OK);
    UNITTEST_ASSERT_EQUAL(AMGX_resources_create_simple(&rsrc, cfg), AMGX_RC_OK);
    UNITTEST_ASSERT_EQUAL(AMGX_matrix_create(&A, rsrc, AMGX_mode_hDDI), AMGX_RC_OK);
    UNITTEST_ASSERT_EQUAL(AMGX_vector_create(&b, rsrc, AMGX_mode_hDDI), AMGX_RC_OK);
    UNITTEST_ASSERT_EQUAL(AMGX_vector_create(&x, rsrc, AMGX_mode_hDDI), AMGX_RC_OK);
    UNITTEST_ASSERT_EQUAL(AMGX_matrix_upload_all(A, n, nnz, 1, 1, row_ptrs, col_indices, values, NULL), AMGX_RC_OK);
    UNITTEST_ASSERT_EQUAL(AMGX_vector_upload(b, n, 1, ref_b), AMGX_RC_OK);
    UNITTEST_ASSERT_EQUAL(AMGX_vector_upload(x, n, 1, ref_x), AMGX_RC_OK);
    UNITTEST_ASSERT_EQUAL(AMGX_write_system(A, b, x, filename), AMGX_RC_OK);
    UNITTEST_ASSERT_EQUAL(AMGX_vector_destroy(x), AMGX_RC_OK);
    UNITTEST_ASSERT_EQUAL(AMGX_vector_destroy(b), AMGX_RC_OK);
    UNITTEST_ASSERT_EQUAL(AMGX_matrix_destroy(A), AMGX_RC_OK);
    UNITTEST_ASSERT_EQUAL(AMGX_matrix_create(&A, rsrc, AMGX_mode_hDDI), AMGX_RC_OK);
    UNITTEST_ASSERT_EQUAL(AMGX_vector_create(&b, rsrc, AMGX_mode_hDDI), AMGX_RC_OK);
    UNITTEST_ASSERT_EQUAL(AMGX_vector_create(&x, rsrc, AMGX_mode_hDDI), AMGX_RC_OK);
    UNITTEST_ASSERT_EQUAL(AMGX_read_system(A, b, x, filename), AMGX_RC_OK);
    check_vector(b, ref_b, n);
    check_vector(x, ref_x, n);
    UNITTEST_ASSERT_EQUAL(AMGX_vector_destroy(x), AMGX_RC_OK);
    UNITTEST_ASSERT_EQUAL(AMGX_vector_destroy(b), AMGX_RC_OK);
    UNITTEST_ASSERT_EQUAL(AMGX_matrix_destroy(A), AMGX_RC_OK);
    UNITTEST_ASSERT_EQUAL(AMGX_resources_destroy(rsrc), AMGX_RC_OK);
    UNITTEST_ASSERT_EQUAL(AMGX_config_destroy(cfg), AMGX_RC_OK);
}

DECLARE_UNITTEST_END(WriteSystemBinaryVectors);

WriteSystemBinaryVectors <TemplateMode<AMGX_mode_hDDI>::Type>  WriteSystemBinaryVectors_hDDI;

} //namespace amgx
