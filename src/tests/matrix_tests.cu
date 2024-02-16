// SPDX-FileCopyrightText: 2011 - 2024 NVIDIA CORPORATION. All Rights Reserved.
//
// SPDX-License-Identifier: BSD-3-Clause

#include "unit_test.h"
#include "matrix.h"

namespace amgx

{

// parameter is used as test name
DECLARE_UNITTEST_BEGIN(MatrixTests);

void run()
{
    Matrix_d d_M1, d_M2, d_I, d_mNULL;
    Matrix_h h_M1, h_M2, h_I, h_mNULL;
    Matrix<TConfig> M1, M2, I, mNULL;
    UNITTEST_ASSERT_EQUAL(d_mNULL.get_num_rows(), 0);
    UNITTEST_ASSERT_EQUAL(d_mNULL.get_num_cols(), 0);
    UNITTEST_ASSERT_EQUAL(d_mNULL.get_num_nz(), 0);
    UNITTEST_ASSERT_EQUAL(h_mNULL.get_num_rows(), 0);
    UNITTEST_ASSERT_EQUAL(h_mNULL.get_num_cols(), 0);
    UNITTEST_ASSERT_EQUAL(h_mNULL.get_num_nz(), 0);
    h_I.addProps(CSR);
    h_I.resize(10, 10, 10);
    UNITTEST_ASSERT_EQUAL(h_I.get_num_rows(), 10);
    UNITTEST_ASSERT_EQUAL(h_I.get_num_cols(), 10);
    UNITTEST_ASSERT_EQUAL(h_I.get_num_nz(), 10);
    UNITTEST_ASSERT_EQUAL(h_I.row_offsets.size(), 11);
    UNITTEST_ASSERT_EQUAL(h_I.values.size(), 11); // because of additional zero for default diagprop = 0
    UNITTEST_ASSERT_EQUAL(h_I.col_indices.size(), 10);

    //create identity matrix
    for (int i = 0; i < 10; i++)
    {
        h_I.row_offsets[i + 1] = i + 1;
    }

    for (int i = 0; i < 10; i++)
    {
        h_I.col_indices[i] = i;
        h_I.values[i] = 1;
    }

    d_I = h_I;
    UNITTEST_ASSERT_EQUAL_DESC("Matrix Host to Device Copy", d_I, h_I);
    h_I = h_mNULL;
    UNITTEST_ASSERT_EQUAL_DESC("Matrix Host to Host Copy", h_I, h_mNULL);
    h_I = d_I;
    UNITTEST_ASSERT_EQUAL_DESC("Matrix Device to Host Copy", d_I, h_I);
    d_M1.copy_async(h_I);
    h_M1.copy_async(d_I);
    d_M2.copy_async(d_I);
    d_M1.sync();
    h_M1.sync();
    d_M2.sync();
    UNITTEST_ASSERT_EQUAL_DESC("Matrix Device to Host Async Copy", h_M1, d_I);
    UNITTEST_ASSERT_EQUAL_DESC("Matrix Host to Device Async Copy", h_I, d_M1);
    UNITTEST_ASSERT_EQUAL_DESC("Matrix Device to Device Async Copy", d_I, d_M2);
    //create a more complex matrix
    h_M1.addProps(CSR);
    h_M1.resize(100, 100, 100 * 101 / 2);

    for (int row = 0; row < 100; row++)
    {
        h_M1.row_offsets[row + 1] = (row + 1) + h_M1.row_offsets[row];

        for (int j = h_M1.row_offsets[row]; j < h_M1.row_offsets[row + 1]; j++)
        {
            int col = j - h_M1.row_offsets[row];
            h_M1.values[j] = col;
            h_M1.col_indices[j] = col;
        }
    }

    M1 = h_M1;
    //save matrix in M2
    M2 = M1;
    //Add COO property and check that the conversion worked
    M1.addProps(COO);
    UNITTEST_ASSERT_EQUAL(M1.row_indices.size(), M1.get_num_nz());
    h_M1 = M1;
    int idx = 0;

    for (int i = 0; i < 100; i++)
    {
        for (int j = 0; j <= i; j++)
        {
            UNITTEST_ASSERT_EQUAL(h_M1.row_indices[idx], i);
            idx++;
        }
    }

    //Remove CSR
    M1.delProps(CSR);
    UNITTEST_ASSERT_EQUAL(M1.row_offsets.size(), 0);
    //add CSR and remove COO
    M1.addProps(CSR);
    M1.delProps(COO);
    UNITTEST_ASSERT_EQUAL(M1.row_indices.size(), 0);
    //this should now equal M2
    UNITTEST_ASSERT_EQUAL(M1, M2);
    UNITTEST_ASSERT_EQUAL(M1.diag.size(), M1.get_num_rows());
    UNITTEST_ASSERT_EQUAL(M1.hasProps(DIAG), false);
    M1.computeDiagonal();
    h_M1 = M1;

    for (int i = 0, d = 0; i < 100; ++i)
    {
        UNITTEST_ASSERT_EQUAL(h_M1.diag[i], d);
        d += i + 2;
    }

    M1.addProps(DIAG);
    UNITTEST_ASSERT_EQUAL(M1.hasProps(DIAG), true);
    M1.computeDiagonal();
    h_M1 = M1;

    for (int i = 0; i < 100; i++)
    {
        int d = i + h_M1.get_num_nz(); //index at the end of the values array
        UNITTEST_ASSERT_EQUAL(h_M1.diag[i], d);
    }

    M1.delProps(DIAG);
    UNITTEST_ASSERT_EQUAL(M1.hasProps(DIAG), false);
    M1.addProps(COO);
    M1.delProps(CSR);
    M1.computeDiagonal();
    h_M1 = M1;

    for (int i = 0, d = 0; i < 100; ++i)
    {
        UNITTEST_ASSERT_EQUAL(h_M1.diag[i], d);
        d += i + 2;
    }

    M1.addProps(DIAG);
    M1.computeDiagonal();
    h_M1 = M1;

    for (int i = 0; i < 100; i++)
    {
        int d = i + h_M1.get_num_nz(); //index at the end of the values array
        UNITTEST_ASSERT_EQUAL(h_M1.diag[i], d);
    }
}

DECLARE_UNITTEST_END(MatrixTests);


// if you want to be able run this test for all available configs you can write this:
#define AMGX_CASE_LINE(CASE) MatrixTests <TemplateMode<CASE>::Type>  MatrixTests_##CASE;
AMGX_FORALL_BUILDS(AMGX_CASE_LINE)
#undef AMGX_CASE_LINE

// or run for all device configs
//#define AMGX_CASE_LINE(CASE) SampleTest <TemplateMode<CASE>::Type>  MatrixTests_##CASE;
//  AMGX_FORALL_BUILDS_DEVICE(AMGX_CASE_LINE)
//#undef AMGX_CASE_LINE

// or you can specify several desired configs
//MatrixTests <TemplateMode<AMGX_mode_dDDI>::Type>  MatrixTests_dDDI;


} //namespace amgx
