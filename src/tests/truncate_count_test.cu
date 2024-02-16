// SPDX-FileCopyrightText: 2011 - 2024 NVIDIA CORPORATION. All Rights Reserved.
//
// SPDX-License-Identifier: BSD-3-Clause

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
    int new_count = amgx::thrust::reduce(count.begin(), count.end());
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
