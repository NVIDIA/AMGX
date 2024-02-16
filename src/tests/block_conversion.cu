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
