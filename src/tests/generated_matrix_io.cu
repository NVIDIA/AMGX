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
