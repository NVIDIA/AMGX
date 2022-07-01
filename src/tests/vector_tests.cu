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
#include "test_utils.h"
#include "util.h"
#include "time.h"

namespace amgx

{

template<class T>
static __global__
void set_raw(T *v, int N)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < N)
    {
        v[i] = i;
    }
}
template<class T>
static __global__
void set_flat(T v, int N)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < N)
    {
        v[i] = i;
    }
}

template<class T>
static __global__
void set_block(T v, int K, int I, int J)
{
    for (int k = threadIdx.x; k < K; k += blockDim.x)
        for (int i = 0; i < I; i++)
            for (int j = 0; j < J; j++)
            {
                v(k, i, j) = k * I * J + i * J + j;
            }
}

// parameter is used as test name
DECLARE_UNITTEST_BEGIN(VectorTests);



void run()
{
    typedef typename Vector_h::index_type index_type;
    typedef typename Vector_h::value_type value_type;
    Vector_h h_v1;
    Vector_h h_v2;
    Vector_d d_v1;
    Vector_d d_v2;
    UNITTEST_ASSERT_TRUE(h_v1.size() == 0);
    UNITTEST_ASSERT_TRUE(h_v1.raw() == 0);
    UNITTEST_ASSERT_TRUE(d_v1.size() == 0);
    UNITTEST_ASSERT_TRUE(d_v1.raw() == 0);
    h_v1.resize(4);
    UNITTEST_ASSERT_TRUE_DESC("vector resize", h_v1.size() == 4);
    h_v1[3] = 1;
    UNITTEST_ASSERT_TRUE_DESC("vector assign value", h_v1[3] == 1);
    h_v1.clear();
    UNITTEST_ASSERT_TRUE_DESC("vector clear", h_v1.empty());
    UNITTEST_ASSERT_TRUE_DESC("vector clear", h_v1.size() == 0);
    d_v1.resize(4);
    UNITTEST_ASSERT_TRUE_DESC("vector resize", d_v1.size() == 4);
    d_v1[3] = 1;
    UNITTEST_ASSERT_TRUE_DESC("vector assign value", d_v1[3] == 1);
    d_v1.clear();
    UNITTEST_ASSERT_TRUE_DESC("vector clear", d_v1.empty());
    UNITTEST_ASSERT_TRUE_DESC("vector clear", d_v1.size() == 0);
    const Vector_d d_one(64, 1);
    const Vector_d d_zero(64, 0);
    const Vector_d h_one(64, 1);
    const Vector_d h_zero(64, 0);
    UNITTEST_ASSERT_TRUE_DESC("device vector copy constructor size", d_one.size() == 64);
    UNITTEST_ASSERT_TRUE_DESC("device vector copy constructor value", d_one[5] == 1);
    UNITTEST_ASSERT_TRUE_DESC("host vector copy constructor size", h_one.size() == 64);
    UNITTEST_ASSERT_TRUE_DESC("host vector copy constructor value", h_one[5] == 1);
    d_v1 = h_one;
    UNITTEST_ASSERT_EQUAL_DESC("vector host to device", d_v1, d_one);
    h_v2 = d_one;
    UNITTEST_ASSERT_EQUAL_DESC("vector device to host", h_v2, h_one);
    d_v1 = d_one;
    d_v2 = d_one;
    h_v1 = h_zero;
    h_v2 = h_zero;
    h_v2.copy_async(d_v1);
    d_v2.copy_async(h_v1);
    h_v2.sync();
    d_v2.sync();
    UNITTEST_ASSERT_EQUAL_DESC("vector device to host async", h_v2, h_one);
    UNITTEST_ASSERT_EQUAL_DESC("vector host to device async", h_v1, d_zero);
    d_v1 = d_zero;
    d_v2 = d_one;
    d_v1.swap(d_v2);
    UNITTEST_ASSERT_EQUAL_DESC("vector swap1", d_v1, d_one);
    UNITTEST_ASSERT_EQUAL_DESC("vector swap2", d_v2, d_zero);
    //set vector to 0,1,2,3,...,N using the raw pointer
    set_raw <<< 1, 64>>>(d_v1.raw(), 64);
    cudaCheckError();
    d_v2 = d_zero;
    d_v2.set_block_dimx(4);
    d_v2.set_block_dimy(4);
    //set vector to 0,1,2,3,...,N using operator[]
    set_flat <<< 1, 64>>>(d_v2.pod(), 64);
    cudaCheckError();
    UNITTEST_ASSERT_EQUAL_DESC("block vector operator[]", d_v1, d_v2);
    d_v2 = d_zero;
    d_v2.set_block_dimx(4);
    d_v2.set_block_dimy(4);
    //set vector to 0,1,2,3,...,N using operator()
    set_block <<< 1, 64>>>(d_v2.pod(), 4, 4, 4);
    cudaCheckError();
    UNITTEST_ASSERT_EQUAL_DESC("block vector operator() 4x4", d_v1, d_v2);
    d_v2 = d_zero;
    d_v2.set_block_dimx(8);
    d_v2.set_block_dimy(2);
    //set vector to 0,1,2,3,...,N using operator()
    set_block <<< 1, 64>>>(d_v2.pod(), 4, 2, 8);
    cudaCheckError();
    UNITTEST_ASSERT_EQUAL_DESC("block vector operator() 2x8", d_v1, d_v2);
    d_v2 = d_zero;
    d_v2.set_block_dimx(2);
    d_v2.set_block_dimy(8);
    //set vector to 0,1,2,3,...,N using operator()
    set_block <<< 1, 64>>>(d_v2.pod(), 4, 8, 2);
    cudaCheckError();
    UNITTEST_ASSERT_EQUAL_DESC("block vector operator() 8x2", d_v1, d_v2);
}

DECLARE_UNITTEST_END(VectorTests);


// if you want to be able run this test for all available configs you can write this:
#define AMGX_CASE_LINE(CASE) VectorTests <TemplateMode<CASE>::Type>  VectorTests_instance_mode##CASE;
AMGX_FORALL_BUILDS(AMGX_CASE_LINE)
#undef AMGX_CASE_LINE

// or you can specify several desired configs
//VectorTests <TemplateMode<AMGX_mode_dDDI>::Type>  VectorTests_instance_mode_dDDI;

} //namespace amgx
