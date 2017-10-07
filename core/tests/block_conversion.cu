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

#define MAX_BLOCK_SIZE      10
#define MAX_NUM_ROWS        100

// parameter is used as test name
DECLARE_UNITTEST_BEGIN(BlockConversionTest);

// convert NxN -> MxM -> NxN
void test_block_conversion( int N, int M, int rows, bool diag, bool exact )
{
    Matrix<TConfig> A, B, C;

    if ( exact ) { generateMatrixRandomStruct<TConfig>::generateExact( A, rows, diag, N, false ); }
    else { generateMatrixRandomStruct<TConfig>::generate( A, rows, diag, N, false ); }

    random_fill( A );
    B.convert( A, ( diag ? DIAG : 0 ) | CSR, M, M );
    C.convert( B, ( diag ? DIAG : 0 ) | CSR, N, N );
//  UNITTEST_ASSERT_TRUE_DESC( "Matrix equality", (check_equal_tolerance<TConfig, TConfig>::check( C, A, false )) );
    UNITTEST_ASSERT_EQUAL_DESC("Matrix equality", C, A);
}

void run()
{
    this->randomize( 8426 );

    // test NxN -> 1x1 -> NxN
    for ( int N = 1; N <= MAX_BLOCK_SIZE; N++ )
    {
        test_block_conversion( N, 1, MAX_NUM_ROWS, true, false );
        test_block_conversion( N, 1, MAX_NUM_ROWS, false, false );
    }

    // test general NxN -> MxM
    test_block_conversion( 3, 4, 4, true, true );
    test_block_conversion( 3, 4, 4, false, true );
    test_block_conversion( 46, 34, 17, true, true );
    test_block_conversion( 46, 34, 17, false, true );
}

DECLARE_UNITTEST_END(BlockConversionTest);

#define AMGX_CASE_LINE(CASE) BlockConversionTest <TemplateMode<CASE>::Type>  BlockConversionTest_##CASE;
AMGX_FORALL_BUILDS_HOST(AMGX_CASE_LINE)
#undef AMGX_CASE_LINE

} //namespace amgx
