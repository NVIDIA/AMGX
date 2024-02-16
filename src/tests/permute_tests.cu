// SPDX-FileCopyrightText: 2011 - 2024 NVIDIA CORPORATION. All Rights Reserved.
//
// SPDX-License-Identifier: BSD-3-Clause

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
