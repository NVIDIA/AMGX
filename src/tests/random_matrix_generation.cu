// SPDX-FileCopyrightText: 2011 - 2025 NVIDIA CORPORATION. All Rights Reserved.
//
// SPDX-License-Identifier: BSD-3-Clause

#include "unit_test.h"
#include <matrix_io.h>
#include "test_utils.h"
#include "util.h"
#include "time.h"

namespace amgx

{

// parameter is used as test name
DECLARE_UNITTEST_BEGIN(RandomMatrix);

void checkMatrix(Matrix_h &A)
{
    IVector vec1, vec2;
    int bsize = A.get_block_dimx();
    UNITTEST_ASSERT_TRUE( (A.get_num_rows() <= 10000) && (A.get_num_rows() > 0) );
    UNITTEST_ASSERT_TRUE( (A.get_num_cols() <= 10000) && (A.get_num_cols() > 0) );
    UNITTEST_ASSERT_EQUAL( A.get_num_cols(), A.get_num_rows() );
    //UNITTEST_ASSERT_TRUE( A.get_num_nz() <= 10*A.get_num_rows() );
    UNITTEST_ASSERT_EQUAL( A.get_block_size(), bsize * bsize ); // only square blocks?
    UNITTEST_ASSERT_EQUAL( A.get_block_dimx(), bsize );
    UNITTEST_ASSERT_EQUAL( A.get_block_dimy(), bsize );
    UNITTEST_ASSERT_EQUAL( A.values.size(), ((A.get_num_nz() + (A.hasProps(DIAG) ? A.get_num_rows() : 1))*bsize * bsize));
    UNITTEST_ASSERT_EQUAL( A.row_offsets.size(), (A.get_num_rows() + 1) );
    UNITTEST_ASSERT_EQUAL( A.col_indices.size(), A.get_num_nz() );
    int num_rows = A.get_num_rows();
    int num_nz = A.get_num_nz();
    UNITTEST_ASSERT_EQUAL( A.row_offsets[0], 0 );
    UNITTEST_ASSERT_EQUAL( A.row_offsets[num_rows], num_nz );

    for (int r = 0; r < num_rows; r++)
    {
        int start = A.row_offsets[r];
        int end = A.row_offsets[r + 1];
        UNITTEST_ASSERT_TRUE( (end <= num_nz) && (end >= 0) );
        UNITTEST_ASSERT_TRUE( (start <= num_nz) && (start >= 0) );

        if (!A.hasProps(DIAG))
        {
            UNITTEST_ASSERT_TRUE( start != end );    // at least one diagonal non-zero
        }

        for (int j = start; j < end; j++)
        {
            UNITTEST_ASSERT_TRUE( (A.col_indices[j] >= 0) && (A.col_indices[j] < num_rows) );

            if (j != start)
            {
                UNITTEST_ASSERT_TRUE( A.col_indices[j] > A.col_indices[j - 1] );
            }

            if ((j == r) && !A.hasProps(DIAG))
            {
                UNITTEST_ASSERT_EQUAL( A.diag[r], j );
            }
        }
    }
}

void run()
{
    Matrix_h A;
    Vector_h b;
    randomize(83);
    generateMatrixRandomStruct<TConfig>::generate(A, 10000, false, max(rand() % 10, 1), false);
    checkMatrix(A);
    generateMatrixRandomStruct<TConfig>::generate(A, 10000, true, max(rand() % 10, 1), false);
    checkMatrix(A);
    generateMatrixRandomStruct<TConfig>::generate(A, 10000, false, max(rand() % 10, 1), true);
    checkMatrix(A);
    generateMatrixRandomStruct<TConfig>::generate(A, 10000, true, max(rand() % 10, 1), true);
    checkMatrix(A);
}

DECLARE_UNITTEST_END(RandomMatrix);

RandomMatrix <TemplateMode<AMGX_mode_hDDI>::Type>  RandomMatrix_hDDI;

} //namespace amgx
