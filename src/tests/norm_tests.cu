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
#include "matrix.h"
#include "norm.h"
#include "ctime"

namespace amgx
{

DECLARE_UNITTEST_BEGIN(NormTests);


void test_get_norm(Vector_h &block_nrm, const Vector_h &vec, const NormType norm_type, int bdim = 1, int offset = 0)
{
    UNITTEST_ASSERT_TRUE_DESC("Only L1 and L2 are supported in this unit test", norm_type == L1 || norm_type == L2);
    block_nrm.resize(bdim, ValueTypeB(0));
    std::vector <typename Vector_h::value_type> norm(bdim, 0.l);

    if (norm_type == L1)
    {
        for (int i = 0; i < (vec.size() / bdim); i++)
            for (int j = 0; j < bdim; j++)
            {
                norm[j] += std::fabs(vec[(offset + i) * bdim + j]);
            }

        for (int j = 0; j < bdim; j++)
        {
            block_nrm[j] = norm[j];
        }
    }
    else if (norm_type == L2)
    {
        for (int i = 0; i < (vec.size() / bdim); i++)
            for (int j = 0; j < bdim; j++)
            {
                norm[j] += vec[(offset + i) * bdim + j] * vec[(offset + i) * bdim + j];
            }

        for (int j = 0; j < bdim; j++)
        {
            block_nrm[j] = sqrt(norm[j]);
        }
    }
}

void check_norm(const int size, const int bdim, const NormType norm_type)
{
    Matrix_h A;
    //Workaround to test large vector sizes:
    A.set_initialized(0);
    A.set_block_dimx(bdim);
    A.set_block_dimy(bdim);
    A.set_num_nz(size);
    A.set_num_rows(size);
    A.set_num_cols(size);
    A.set_initialized(1);
    //Matrix_h A(size,size,size, bdim, bdim, 0);
    //generateMatrixRandomStruct<TConfig_h>::generateExact(A, size_vec, true , bdim, false);
    int offset = 0;
    Vector_h vec(size);
    vec.set_block_dimx(bdim);
    fillRandom<Vector_h>::fill(vec);
    Matrix<TConfig> A_try(A);
    Vector<TConfig> vec_try(vec);
    Vector_h norm_ref(bdim), norm_try(bdim);
    test_get_norm(norm_ref, vec, norm_type, bdim, offset);
    get_norm( A_try, vec_try, bdim, norm_type, norm_try );
    this->PrintOnFail(": error in checking norm %s, blocksize %d, size %d\n", norm_type == L1 ? "L1" : "L2", bdim, size);
    // summing on gpu and host might produce different numbers due to order of summation for L1, tuning numbers a little bit
    UNITTEST_ASSERT_EQUAL_TOL(norm_ref, norm_try, getTolerance<typename Vector_h::value_type>::get()*(norm_type == L1 ? size : 1.)); 
}

void run()
{
    randomize( 10 );
    for (int bsize = 1; bsize <= 10; bsize ++)
    {
        int size = 10000 * bsize;
        check_norm(size, bsize, L1);
        check_norm(size, bsize, L2);
    }
}

DECLARE_UNITTEST_END(NormTests);

#define AMGX_CASE_LINE(CASE) NormTests <TemplateMode<CASE>::Type>  NormTests_##CASE;
AMGX_FORALL_BUILDS(AMGX_CASE_LINE)
#undef AMGX_CASE_LINE
} //namespace amgx
