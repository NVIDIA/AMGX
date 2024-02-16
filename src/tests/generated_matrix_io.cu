// SPDX-FileCopyrightText: 2011 - 2024 NVIDIA CORPORATION. All Rights Reserved.
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
DECLARE_UNITTEST_BEGIN(GeneratedMatrixIOTest);

void run()
{
    Matrix_h A, An;
    VVector b, x, bn, xn;
    this->randomize( 65 );

    for (int bin = 0; bin < 2; bin++)
    {
        std::string write_format;

        if (bin)
        {
            write_format = "binary";
        }
        else
        {
            write_format = "matrixmarket";
        }

        // separate diag, 1x1
        generateMatrixRandomStruct<TConfig>::generate(A, 10, true, 1, false);
        random_fill(A);
        b.resize(A.get_num_rows());
        random_fill(b);
        UNITTEST_ASSERT_TRUE(MatrixIO<TConfig_h>::writeSystemWithFormat(".temp_matrix.mtx", write_format.c_str(), &A, &b, NULL) == AMGX_OK);
        UNITTEST_ASSERT_TRUE(MatrixIO<TConfig_h>::readSystem(".temp_matrix.mtx", An, bn, xn) == AMGX_OK);
        UNITTEST_ASSERT_EQUAL_DESC("1x1 rhs i/o equality", b, bn);
        // reading rhs only:
        UNITTEST_ASSERT_TRUE(MatrixIO<TConfig_h>::readSystem(".temp_matrix.mtx", An, bn, xn, AMG_Config(), io_config::RHS) == AMGX_OK);
        UNITTEST_ASSERT_TRUE_DESC("1x1 DIAG matrices i/o equality", (equalMatrices<TConfig, TConfig>::check(A, An, false)));
        UNITTEST_ASSERT_EQUAL_DESC("1x1 rhs i/o equality", b, bn);
        // inside diag, 1x1
        generateMatrixRandomStruct<TConfig>::generate(A, 10, false, 1, false);
        random_fill(A);
        b.resize(A.get_num_rows());
        random_fill(b);
        UNITTEST_ASSERT_TRUE(MatrixIO<TConfig_h>::writeSystemWithFormat(".temp_matrix.mtx", write_format.c_str(), &A, &b, NULL) == AMGX_OK);
        UNITTEST_ASSERT_TRUE(MatrixIO<TConfig_h>::readSystem(".temp_matrix.mtx", An, bn, xn) == AMGX_OK);
        UNITTEST_ASSERT_TRUE_DESC("1x1 matrices i/o equality", (equalMatrices<TConfig, TConfig>::check(A, An, false)));
        UNITTEST_ASSERT_EQUAL_DESC("1x1 rhs i/o equality", b, bn);
        // separate diag, 4x4
        generateMatrixRandomStruct<TConfig>::generate(A, 10, true, 4, false);
        random_fill(A);
        b.resize(A.get_num_rows() * 4);
        random_fill(b);
        UNITTEST_ASSERT_TRUE(MatrixIO<TConfig_h>::writeSystemWithFormat(".temp_matrix.mtx", write_format.c_str(), &A, &b, NULL) == AMGX_OK);
        UNITTEST_ASSERT_TRUE(MatrixIO<TConfig_h>::readSystem(".temp_matrix.mtx", An, bn, xn) == AMGX_OK);
        UNITTEST_ASSERT_TRUE_DESC("4x4 DIAG matrices i/o equality", (equalMatrices<TConfig, TConfig>::check(A, An, false)));
        UNITTEST_ASSERT_EQUAL_DESC("4x4 rhs i/o equality", b, bn);
        // inside diag, 4x4
        generateMatrixRandomStruct<TConfig>::generate(A, 10, false, 4, false);
        random_fill(A);
        b.resize(4 * A.get_num_rows());
        random_fill(b);
        UNITTEST_ASSERT_TRUE(MatrixIO<TConfig_h>::writeSystemWithFormat(".temp_matrix.mtx", write_format.c_str(), &A, &b, NULL) == AMGX_OK);
        UNITTEST_ASSERT_TRUE(MatrixIO<TConfig_h>::readSystem(".temp_matrix.mtx", An, bn, xn) == AMGX_OK);
        UNITTEST_ASSERT_TRUE_DESC("4x4 matrices i/o equality", (equalMatrices<TConfig, TConfig>::check(A, An, false)));
        UNITTEST_ASSERT_EQUAL_DESC("4x4 rhs i/o equality", b, bn);
        // save initial guess
        generateMatrixRandomStruct<TConfig>::generate(A, 10, false, 1, false);
        random_fill(A);
        b.resize(A.get_num_rows());
        random_fill(b);
        x.resize(A.get_num_rows());
        random_fill(x);
        UNITTEST_ASSERT_TRUE(MatrixIO<TConfig_h>::writeSystemWithFormat(".temp_matrix.mtx", write_format.c_str(), &A, &b, &x) == AMGX_OK);
        UNITTEST_ASSERT_TRUE(MatrixIO<TConfig_h>::readSystem(".temp_matrix.mtx", An, bn, xn) == AMGX_OK);
        //printMatrixStats(A);
        //printMatrixStats(An);
        UNITTEST_ASSERT_TRUE_DESC("1x1 full system matrix i/o equality", (equalMatrices<TConfig, TConfig>::check(A, An, false)) );
        UNITTEST_ASSERT_EQUAL_DESC("1x1 full system i/o equality", b, bn);
        UNITTEST_ASSERT_EQUAL_TOL_DESC("1x1 full system initial guess equality", x, xn, 1e-10);
        // reading solution only:
        UNITTEST_ASSERT_TRUE(MatrixIO<TConfig_h>::readSystem(".temp_matrix.mtx", An, bn, xn, AMG_Config(), io_config::SOLN) == AMGX_OK);
        UNITTEST_ASSERT_TRUE_DESC("Save initial guess matrix i/o equality", (equalMatrices<TConfig, TConfig>::check(A, An, false)) );
        UNITTEST_ASSERT_EQUAL_DESC("Save initial guess rhs equality", b, bn);
        UNITTEST_ASSERT_EQUAL_TOL_DESC("Save initial guess equality", x, xn, 1e-10);
    }
}

DECLARE_UNITTEST_END(GeneratedMatrixIOTest);

#define AMGX_CASE_LINE(CASE) GeneratedMatrixIOTest <TemplateMode<CASE>::Type>  GeneratedMatrixIOTest_##CASE;
AMGX_FORALL_BUILDS_HOST(AMGX_CASE_LINE)
#undef AMGX_CASE_LINE

} //namespace amgx
