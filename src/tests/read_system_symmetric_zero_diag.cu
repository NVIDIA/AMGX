// SPDX-FileCopyrightText: 2011 - 2025 NVIDIA CORPORATION. All Rights Reserved.
//
// SPDX-License-Identifier: BSD-3-Clause

#include "unit_test.h"
#include "amgx_c.h"
#include <fstream>
#include <vector>

namespace amgx

{

// Reads a symmetric MatrixMarket file that stores an explicit zero on the diagonal.
// The reader skips explicit zeroes, so the matrix
//    2 -1  0
//   -1  0 -1
//    0 -1  2
// has to come back with 6 nonzeroes and consistent CSR arrays.
DECLARE_UNITTEST_BEGIN(ReadSystemSymmetricZeroDiag);

void run()
{
    const char *filename = ".temp_matrix.mtx";
    {
        std::ofstream fout(filename);
        fout << "%%MatrixMarket matrix coordinate real symmetric" << std::endl;
        fout << "3 3 5" << std::endl;
        fout << "1 1 2.0" << std::endl;
        fout << "2 1 -1.0" << std::endl;
        fout << "2 2 0.0" << std::endl;
        fout << "3 2 -1.0" << std::endl;
        fout << "3 3 2.0" << std::endl;
    }
    const int ref_nnz = 6;
    const int ref_row_ptrs[] = {0, 2, 4, 6};
    const int ref_col_indices[] = {0, 1, 0, 2, 1, 2};
    const double ref_values[] = {2., -1., -1., -1., -1., 2.};
    AMGX_config_handle cfg = NULL;
    AMGX_resources_handle rsrc = NULL;
    AMGX_matrix_handle A = NULL;
    UNITTEST_ASSERT_EQUAL(AMGX_config_create(&cfg, ""), AMGX_RC_OK);
    UNITTEST_ASSERT_EQUAL(AMGX_resources_create_simple(&rsrc, cfg), AMGX_RC_OK);
    UNITTEST_ASSERT_EQUAL(AMGX_matrix_create(&A, rsrc, AMGX_mode_hDDI), AMGX_RC_OK);
    UNITTEST_ASSERT_EQUAL(AMGX_read_system(A, NULL, NULL, filename), AMGX_RC_OK);
    int n = 0, block_dimx = 0, block_dimy = 0, nnz = 0;
    UNITTEST_ASSERT_EQUAL(AMGX_matrix_get_size(A, &n, &block_dimx, &block_dimy), AMGX_RC_OK);
    UNITTEST_ASSERT_EQUAL(AMGX_matrix_get_nnz(A, &nnz), AMGX_RC_OK);
    UNITTEST_ASSERT_EQUAL(n, 3);
    UNITTEST_ASSERT_EQUAL(block_dimx, 1);
    UNITTEST_ASSERT_EQUAL(block_dimy, 1);
    UNITTEST_ASSERT_EQUAL_DESC("number of nonzeroes", nnz, ref_nnz);
    std::vector<int> row_ptrs(n + 1, -1);
    std::vector<int> col_indices(nnz, -1);
    std::vector<double> values(nnz, 0.);
    void *diag = NULL;
    UNITTEST_ASSERT_EQUAL(AMGX_matrix_download_all(A, &row_ptrs[0], &col_indices[0], &values[0], &diag), AMGX_RC_OK);

    for (int i = 0; i <= n; i++)
    {
        UNITTEST_ASSERT_EQUAL_DESC("row offsets", row_ptrs[i], ref_row_ptrs[i]);
    }

    for (int i = 0; i < nnz; i++)
    {
        UNITTEST_ASSERT_EQUAL_DESC("column indices", col_indices[i], ref_col_indices[i]);
        UNITTEST_ASSERT_EQUAL_DESC("values", values[i], ref_values[i]);
    }

    UNITTEST_ASSERT_EQUAL(AMGX_matrix_destroy(A), AMGX_RC_OK);
    UNITTEST_ASSERT_EQUAL(AMGX_resources_destroy(rsrc), AMGX_RC_OK);
    UNITTEST_ASSERT_EQUAL(AMGX_config_destroy(cfg), AMGX_RC_OK);
}

DECLARE_UNITTEST_END(ReadSystemSymmetricZeroDiag);

ReadSystemSymmetricZeroDiag <TemplateMode<AMGX_mode_hDDI>::Type>  ReadSystemSymmetricZeroDiag_hDDI;

} //namespace amgx
