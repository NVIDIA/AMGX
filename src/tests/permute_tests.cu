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
#include "vector.h"
#include "permute.h"
#include "test_utils.h"
#include "util.h"
#include "time.h"

namespace amgx

{

// parameter is used as test name
DECLARE_UNITTEST_BEGIN(PermuteTests);

template <class T> void swap(T &v1, T &v2)
{
    T tmp = v1;
    v1 = v2;
    v2 = tmp;
}


void run()
{
    int SIZE = 4096 * 16;
    IVector v1(SIZE);
    IVector v2(SIZE);
    IVector v3(SIZE);
    IVector p(SIZE);
    IVector_h h_p;
    //initial permutation set to identity
    thrust_wrapper::sequence<IVector::TConfig::memSpace>(p.begin(), p.end());
    thrust_wrapper::sequence<IVector::TConfig::memSpace>(v1.begin(), v1.end());
    permuteVector(v1, v2, p, SIZE);
    UNITTEST_ASSERT_EQUAL(v1, v2);
    unpermuteVector(v2, v3, p, SIZE);
    UNITTEST_ASSERT_EQUAL(v1, v3);
    //reverse permutation
    thrust_wrapper::sequence<IVector::TConfig::memSpace>(p.rbegin(), p.rend());
    thrust_wrapper::sequence<IVector::TConfig::memSpace>(v1.begin(), v1.end());
    permuteVector(v1, v2, p, SIZE);
    thrust_wrapper::sequence<IVector::TConfig::memSpace>(v1.rbegin(), v1.rend());
    UNITTEST_ASSERT_EQUAL(v1, v2);
    thrust_wrapper::sequence<IVector::TConfig::memSpace>(v1.begin(), v1.end());
    unpermuteVector(v2, v3, p, SIZE);
    UNITTEST_ASSERT_EQUAL(v1, v3);
    p.resize(SIZE / 16);
    thrust_wrapper::sequence<IVector::TConfig::memSpace>(p.begin(), p.end());
    v1.set_block_dimx(4);
    v1.set_block_dimy(4);
    v2.set_block_dimx(4);
    v2.set_block_dimy(4);
    v3.set_block_dimx(4);
    v3.set_block_dimy(4);
    thrust_wrapper::sequence<IVector::TConfig::memSpace>(v1.begin(), v1.end());
    permuteVector(v1, v2, p, SIZE);
    UNITTEST_ASSERT_EQUAL(v1, v2);
    unpermuteVector(v2, v3, p, SIZE);
    UNITTEST_ASSERT_EQUAL(v1, v3);
    thrust_wrapper::sequence<IVector::TConfig::memSpace>(p.rbegin(), p.rend());
    thrust_wrapper::sequence<IVector::TConfig::memSpace>(v1.begin(), v1.end());
    permuteVector(v1, v2, p, SIZE);
    UNITTEST_ASSERT_NEQUAL(v1, v2);
    //applying permutation twice will invert itself
    permuteVector(v2, v3, p, SIZE);
    UNITTEST_ASSERT_EQUAL(v1, v3);
    unpermuteVector(v2, v3, p, SIZE);
    UNITTEST_ASSERT_EQUAL(v1, v3);
    h_p = p;
    thrust_wrapper::sequence<AMGX_host>(h_p.begin(), h_p.end());

    //create random permuatation
    for (int j = 0; j < 10; j++)
        for (int i = 0; i < h_p.size(); i++)
        {
            int new_i = rand() % h_p.size();
            swap(h_p[i], h_p[new_i]);
        }

    p = h_p;
    permuteVector(v1, v2, p, SIZE);
    unpermuteVector(v2, v3, p, SIZE);
    UNITTEST_ASSERT_EQUAL(v1, v3);
}

DECLARE_UNITTEST_END(PermuteTests);


// if you want to be able run this test for all available configs you can write this:
//#define AMGX_CASE_LINE(CASE) PermuteTests <TemplateMode<CASE>::Type>  PermuteTests_instance_mode##CASE;
//  AMGX_FORALL_BUILDS(AMGX_CASE_LINE)
//#undef AMGX_CASE_LINE

// or you can specify several desired configs
PermuteTests <TemplateMode<AMGX_mode_dDDI>::Type>  PermuteTests_instance_mode_dDDI;

} //namespace amgx
