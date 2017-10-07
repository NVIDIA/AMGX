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
#include <matrix.h>
#include <truncate.h>
#include "test_utils.h"
#include "util.h"
#include <cusp/gallery/poisson.h>

namespace amgx
{

DECLARE_UNITTEST_BEGIN(truncateCountTest);

void run()
{
    typedef Vector<typename TConfig::template setVecPrec<AMGX_vecInt>::Type> IVector;
    const int N = 100;
    Matrix<TConfig> A;
    A.addProps(CSR);
    MatrixCusp<TConfig, cusp::csr_format> Aw(&A);
    cusp::gallery::poisson5pt(Aw, N, N);
    IVector count(A.get_num_rows(), 0);
    VVector x(A.get_num_rows(), 4.), new_row_sum(A.get_num_rows(), 0.);
    const double trunc_factor = 0.5;
    countTruncElements(A, trunc_factor, x, count, new_row_sum);
    int new_count = thrust::reduce(count.begin(), count.end());
    this->PrintOnFail("truncateCountTest: new nnz should = num rows");
    UNITTEST_ASSERT_TRUE(A.get_num_rows() == new_count);

    for (int i = 0; i < new_row_sum.size(); i++)
    {
        this->PrintOnFail("truncateCountTest: new_row_sum[i] should = 4 for all i");
        UNITTEST_ASSERT_TRUE(fabs(new_row_sum[i] - 4.) <= 1e-6);
    }
}

DECLARE_UNITTEST_END(truncateCountTest);

truncateCountTest<TemplateMode<AMGX_mode_dDDI>::Type> truncateCountTest_instance_mode_dDDI;

} // end namespace amgx
