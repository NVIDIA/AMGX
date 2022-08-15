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
#include "aggregation/selectors/size2_selector.h"
#include "multiply.h"

namespace amgx

{

// parameter is used as test name
DECLARE_UNITTEST_BEGIN(TemplateTest);

void run()
{

  // call randomize to seed random numbers generator
  randomize( 513 );

  {
    Matrix_h A;
    // we can generate some random matrix
    generateMatrixRandomStruct<TConfig_h>::generate(A, 10000, (rand() % 2) == 0 , max(rand() % 10, 1), false);
    // and fill it with random values  
    random_fill(A);
    // we can also fill vectors with random values
    Vector_h b(A.get_num_rows()*A.get_block_dimy()), x(A.get_num_rows()*A.get_block_dimy());
    random_fill(b);

    //writing temp matrix on disk
    MatrixIO<TConfig_h>::writeSystemMatrixMarket(".temp.mtx", &A, &b, NULL);    
  }

  // some custom TConfigs.
  Matrix<typename TConfig_h::template setMatPrec<AMGX_matDouble>::Type> hA_double;
  Vector<typename TConfig_h::template setMatPrec<AMGX_matDouble>::Type> b_double;
  Matrix<typename TConfig_h::template setMatPrec<AMGX_matFloat>::Type> hA_float;
  Vector<typename TConfig_h::template setMatPrec<AMGX_matFloat>::Type> b_float;

  // reading some matrix in matrix market format
  UNITTEST_ASSERT_TRUE_DESC("reading matrix in double", this->read_system(".temp.mtx", hA_double, b_double, b_double, true));
  UNITTEST_ASSERT_TRUE_DESC("reading matrix in float", this->read_system(".temp.mtx", hA_float, b_float, b_float, true));
  
  // we can pass matrices to the ASSERT_EQUAL macro. it performs structure comparison and nnz values comparison
  // check equality with default tolerance. second parameter's double values will be casted to the first parameter's type (float) 
  // in the case of different matrices and vectors - they won't be printed
  UNITTEST_ASSERT_EQUAL(hA_float, hA_double);
  UNITTEST_ASSERT_EQUAL_TOL_DESC("or we can specify tolerance manually. comparing rhs.", b_double, b_float, 1e-5);

  // running amgx routine on the readed matrix
  
  {
    // example of creating configured amgx routine
    AMG_Config cfg;
    cfg.parseParameterString("determinism_flag=1");
    aggregation::Selector<TConfig>* selector = new aggregation::size2_selector::Size2Selector<T_Config>(cfg,"default");

    // example of providing messages to be shown in the case of assert fail
    this->PrintOnFail("If selector is NULL we %s this message\n", "will see");
    UNITTEST_ASSERT_TRUE(selector != NULL);
    UNITTEST_ASSERT_TRUE_DESC("Or we can provide simple message inside the macros. Every macros have NAME_DESC() analog with this additional message", selector != NULL);


    Matrix_d A = hA_double;
    IVector vec1, vec2, vec3;
    int num1 = 0, num2 = -1;
    selector->setAggregates(A, vec1, vec3, num1);
    selector->setAggregates(A, vec2, vec3, num2);
  // check, the result. if values are different - they will be printed to the assert string
    UNITTEST_ASSERT_EQUAL(num1, num2);

    delete selector;
  }

  
  {
    // fast equality check using hash on gpu
    Vector_d tvec;
    tvec.resize(142);

    Vector_d tv1 = tvec;
    Vector_d tv2 = tvec;
    
    UNITTEST_ASSERT_TRUE(check_hashsum(tv1, tv2));

    // this one will fail
    tvec[0] = -666.666;
    tv2 = tvec;  
    tvec[0] = 1147.2013;
    tv1 = tvec;  
    UNITTEST_ASSERT_TRUE(!check_hashsum(tv1, tv2));
    
    // work with matrices as well
    Matrix_d At1 = hA_double;
    Matrix_d At2 = hA_double;
    UNITTEST_ASSERT_TRUE(check_hashsum(At1, At2));

  }


  {
    // we can also check for some exceptions
    UNITTEST_ASSERT_EXCEPTION_START;
    std::vector<int> hb(10);
    hb.at(10000) = 5;
    UNITTEST_ASSERT_EXCEPTION_END_ANY_DESC("assert will fail if code between START and END won't generate any exception");

    Matrix_d A;
    A.set_initialized(1);
    UNITTEST_ASSERT_EXCEPTION_START;
    PrintOnFail("We can also expect some particular exception. If code will reach FatalError() - test will pass\n");
    PrintOnFail("Matrix is initialized, we cannot change it\n");
    A.set_block_dimx(1);
    UNITTEST_ASSERT_EXCEPTION_END(amgx_exception);
  }
}

DECLARE_UNITTEST_END(TemplateTest);


// if you want to be able run this test for all available configs you can write this:
//#define AMGX_CASE_LINE(CASE) SampleTest <TemplateMode<CASE>::Type>  TemplateTest_##CASE;
//  AMGX_FORALL_BUILDS(AMGX_CASE_LINE)                                                                
//#undef AMGX_CASE_LINE

// or run for all device configs
//#define AMGX_CASE_LINE(CASE) SampleTest <TemplateMode<CASE>::Type>  TemplateTest_##CASE;
//  AMGX_FORALL_BUILDS_DEVICE(AMGX_CASE_LINE)
//#undef AMGX_CASE_LINE

// or you can specify several desired configs
TemplateTest <TemplateMode<AMGX_mode_dDDI>::Type>  TemplateTest_dDDI;


} //namespace amgx
