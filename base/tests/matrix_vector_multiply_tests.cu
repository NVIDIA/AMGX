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
#include "matrix.h"
#include "multiply.h"

namespace amgx

{

// parameter is used as test name
DECLARE_UNITTEST_BEGIN(MatrixVectorMultiplyTests);

void run()
{
    Matrix<TConfig_h> h_I, h_A;
    Matrix<TConfig>   I, mNULL, A, B, C;
    Vector<TConfig_h> h_x, h_one, h_zero, h_BIG, h_b;
    Vector<TConfig>   x, b;
    const int MSIZE = 10;
    h_I.addProps(CSR);
    h_I.resize(MSIZE, MSIZE, MSIZE);
    h_x.resize(MSIZE);
    h_one.resize(MSIZE);
    h_zero.resize(MSIZE);
    h_BIG.resize(MSIZE);
    h_b.resize(MSIZE);

    //create identity matrix
    for (int i = 0; i < h_I.get_num_rows(); i++)
    {
        h_I.row_offsets[i] = i;
    }

    h_I.row_offsets[h_I.get_num_rows()] = h_I.get_num_rows();

    for (int i = 0; i < h_I.get_num_rows(); i++)
    {
        h_I.col_indices[i] = i;
        h_I.values[i] = 1;
        h_one[i] = 1;
        h_zero[i] = 0;
        h_BIG[i] = 99999999;
    }

    h_I.computeDiagonal();
    h_I.set_initialized(1);
    I = h_I;
    x = h_one;
    b = h_BIG;
    A = I;
    multiply(A, x, b);
    UNITTEST_ASSERT_EQUAL_DESC("CSR: b=I*x", x, b);
#if 1
    A.set_initialized(0);
    A.addProps(COO);
    A.delProps(CSR);
    A.set_initialized(1);
    b = h_BIG;
    multiply(A, x, b);
    UNITTEST_ASSERT_EQUAL_DESC("COO: b=I*x", x, b);
#endif
    h_A = I;
    h_A.values[0] = 0;
    A = h_A;
    b = h_BIG;
    multiply(A, x, b);
    h_b = h_one;
    h_b[0] = 0;
    UNITTEST_ASSERT_EQUAL_DESC("CSR: b=A*x (explict zero row)", h_b, b);
#if 1
    A.set_initialized(0);
    A.addProps(COO);
    A.delProps(CSR);
    A.set_initialized(1);
    b = h_BIG;
    multiply(A, x, b);
    UNITTEST_ASSERT_EQUAL_DESC("COO: b=A*x (explict zero row)", h_b, b);
#endif
#if 1
    //Multiply with NULL row
    h_A.set_initialized(0);
    h_A.addProps(CSR);
    h_A.resize(MSIZE, MSIZE, MSIZE - 1);

    //first row should be empty
    for (int i = 0; i < h_A.get_num_rows(); i++)
    {
        h_A.row_offsets[i + 1] = i;
    }

    for (int i = 0; i < h_A.get_num_rows() - 1; i++)
    {
        h_A.col_indices[i] = i + 1;
        h_A.values[i] = 1;
    }

    h_A.set_initialized(1);
    A = h_A;
    b = h_BIG;
    x = h_one;
    multiply(A, x, b);
    UNITTEST_ASSERT_EQUAL_DESC("b=A*x (A contains an empty row)", b, h_b);
#endif
#if 1
    //Multiply by identity with external diagonal
    h_A.set_initialized(0);
    h_A.addProps(CSR);
    h_A.addProps(DIAG);
    h_A.resize(MSIZE, MSIZE, 0);

    //all rows are empty, only diagonal exists
    for (int i = 0; i < h_A.get_num_rows(); i++)
    {
        h_A.row_offsets[i + 1] = 0;
        h_A.diag[i] = 1;
    }

    h_A.set_initialized(1);
    A = h_A;
    b = h_BIG;
    x = h_one;
    multiply(A, x, b);
    UNITTEST_ASSERT_EQUAL_DESC("b=A*x (A contains diagonal only)", b, x);
#endif
    //Multiply by completly dense matrix values=1..N, x= (double)i/N, b=sum_i^N((double)i/N)...
    //no diagonal
    h_A.set_initialized(0);
    h_A.delProps(DIAG);
    h_A.resize(MSIZE, MSIZE, MSIZE * MSIZE);

    for (int i = 0; i < MSIZE; i++)
    {
        h_A.row_offsets[i + 1] = MSIZE + h_A.row_offsets[i];
    }

    h_b = h_zero;

    for (int i = 0, idx = 0; i < MSIZE; i++)
    {
        for (int j = 0; j < MSIZE; j++, idx++)
        {
            h_A.col_indices[idx] = j;
            double v = (double)(j + 1) / MSIZE;
            h_A.values[idx] = v;
            h_b[i] += v;
        }
    }

    h_A.set_initialized(1);
    A = h_A;
    b = h_BIG;
    x = h_one;
    multiply(A, x, b);
    UNITTEST_ASSERT_EQUAL_TOL_DESC("CSR b=A*x ", b, h_b, 1e-5);
#if 1
    h_A.set_initialized(0);
    h_A.addProps(COO);
    h_A.delProps(CSR);
    h_A.set_initialized(1);
    A = h_A;
    b = h_BIG;
    x = h_one;
    multiply(A, x, b);
    UNITTEST_ASSERT_EQUAL_TOL_DESC("COO b=A*x ", b, h_b, 1e-5);
#endif
    //Multiply by completly dense matrix values=1..N, x= (double)i/N, b=sum_i^N((double)i/N)...
    //with diagonal
    h_A.set_initialized(0);
    h_A.addProps(DIAG);
    h_A.addProps(CSR);
    h_A.resize(MSIZE, MSIZE, MSIZE * MSIZE - MSIZE);
    h_A.row_offsets[0] = 0;

    for ( int i = 0; i <= MSIZE; i++ )
    {
        h_A.row_offsets[i + 1] = MSIZE - 1 + h_A.row_offsets[i];
    }

    ValueTypeB dMSIZE = (ValueTypeB)(MSIZE);
    h_b = h_zero;

    for (int i = 0, idx = 0; i < MSIZE; i++)
    {
        for (int j = 0; j < MSIZE; j++)
        {
            ValueTypeB v = (ValueTypeB)(j + 1) / dMSIZE;

            if (i == j)
            {
                h_A.values[h_A.diag[i]] = v;
            }
            else
            {
                h_A.col_indices[idx] = j;
                h_A.values[idx] = v;
                idx++;
            }

            h_b[i] += v;
        }
    }

    h_A.set_initialized(1);
    MatrixIO<TConfig_h>::writeSystemMatrixMarket("spmv1.mtx", &h_A, &h_b, NULL);
    A = h_A;
    b = h_BIG;
    x = h_one;
    multiply(A, x, b);
    UNITTEST_ASSERT_EQUAL_TOL_DESC("CSR+DIAG b=A*x ", b, h_b, 1e-5);
#if 1
    h_A.set_initialized(0);
    h_A.delProps(COO);
    h_A.addProps(COO);
    h_A.delProps(CSR);
    h_A.set_initialized(1);
    A = h_A;
    b = h_BIG;
    x = h_one;
    multiply(A, x, b);
    UNITTEST_ASSERT_EQUAL_TOL_DESC("COO+DIAG b=A*x ", b, h_b, 1e-5);
#endif
    //Multiply by completly dense matrix values=1..N, x= (double)i/N, b=sum_i^N((double)i/N)...
}

DECLARE_UNITTEST_END(MatrixVectorMultiplyTests);


// if you want to be able run this test for all available configs you can write this:
//#define AMGX_CASE_LINE(CASE) MatrixVectorMultiplyTests <TemplateMode<CASE>::Type>  MatrixVectorMultiplyTests_##CASE;
//  AMGX_FORALL_BUILDS(AMGX_CASE_LINE)
//#undef AMGX_CASE_LINE

// or run for all device configs
//#define AMGX_CASE_LINE(CASE) SampleTest <TemplateMode<CASE>::Type>  MatrixVectorMultiplyTests_##CASE;
//  AMGX_FORALL_BUILDS_DEVICE(AMGX_CASE_LINE)
//#undef AMGX_CASE_LINE

// or you can specify several desired configs
MatrixVectorMultiplyTests <TemplateMode<AMGX_mode_dDDI>::Type>  MatrixVectorMultiplyTests_dDDI;
MatrixVectorMultiplyTests <TemplateMode<AMGX_mode_dFFI>::Type>  MatrixVectorMultiplyTests_dFFI;
MatrixVectorMultiplyTests <TemplateMode<AMGX_mode_hDDI>::Type>  MatrixVectorMultiplyTests_hDDI;
MatrixVectorMultiplyTests <TemplateMode<AMGX_mode_hFFI>::Type>  MatrixVectorMultiplyTests_hFFI;


} //namespace amgx
