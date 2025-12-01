// SPDX-FileCopyrightText: 2013 - 2025 NVIDIA CORPORATION. All Rights Reserved.
//
// SPDX-License-Identifier: BSD-3-Clause

#include <eigensolvers/multivector_operations.h>
#include <blas.h>
#include <amgx_cublas.h>
#include <thrust_wrapper.h>

namespace amgx
{

template <typename TConfig>
void
distributed_gemm_TN(typename TConfig::VecPrec alpha, const Vector<TConfig> &lhs,
                    const Vector<TConfig> &rhs,
                    typename TConfig::VecPrec beta, Vector<TConfig> &res,
                    const Operator<TConfig> &A)
{
    typedef typename TConfig::template setMemSpace<AMGX_host>::Type TConfig_h;
    typedef Vector<TConfig_h> Vector_h;
    typedef typename Vector<TConfig>::value_type ValueTypeVec;

    if (A.is_matrix_singleGPU())
    {
        FatalError("distributed_gemm: matrix is not partitioned.", AMGX_ERR_INTERNAL);
    }

    // Compute local matrix-matrix multiplication.
    Cublas::gemm(alpha, lhs, rhs, beta, res, true, false);
    // Copy result vector to CPU.
    Vector_h h_res = res;
    // Perform distributed reduction of each local result.
    std::vector<Vector_h> gathered;
    A.getManager()->getComms()->global_reduce(gathered, h_res, A, 0);
    thrust_wrapper::fill<AMGX_host>(h_res.begin(), h_res.end(), 0);
    cudaCheckError();

    for (int i = 0; i < gathered.size(); ++i)
    {
        thrust_wrapper::transform<AMGX_host>(gathered[i].begin(), gathered[i].end(),
                          h_res.begin(), h_res.begin(), amgx::thrust::plus<ValueTypeVec>());
    }

    cudaCheckError();
    // Transfer final result to GPU.
    res = h_res;
}

template <typename TConfig>
void
multivector_column_norms(const Vector<TConfig> &v,
                         Vector<typename TConfig::template setMemSpace<AMGX_host>::Type> &results,
                         const Operator<TConfig> &A)
{
    typedef typename TConfig::template setMemSpace<AMGX_host>::Type TConfig_h;
    typedef Vector<TConfig_h> Vector_h;
    typedef typename Vector<TConfig>::value_type ValueTypeVec;
    int rows = v.get_num_rows();
    int cols = v.get_num_cols();
    int lda = v.get_lda();
    // Compute column norm of local vector.
    results.resize(cols);

    for (int i = 0; i < cols; ++i)
    {
        results[i] = nrm2(v, lda * i, rows);
    }

    // Reduce column norms along all GPUs.
    if (!A.is_matrix_singleGPU())
    {
        std::vector<Vector_h> gathered;
        A.getManager()->getComms()->global_reduce(gathered, results, A, 0);
        thrust_wrapper::fill<AMGX_host>(results.begin(), results.end(), 0);
        cudaCheckError();

        for (int i = 0; i < gathered.size(); ++i)
        {
            for (int j = 0; j < cols; ++j)
            {
                ValueTypeVec norm = gathered[i][j];
                results[j] += norm * norm;
            }
        }

        for (int j = 0; j < cols; ++j)
        {
            results[j] = std::sqrt(results[j]);
        }
    }
}

#define AMGX_CASE_LINE(CASE) \
    template void distributed_gemm_TN(typename TemplateMode<CASE>::Type::VecPrec alpha, \
                                      const Vector<TemplateMode<CASE>::Type>& lhs, \
                                      const Vector<TemplateMode<CASE>::Type>& rhs,\
                                      typename TemplateMode<CASE>::Type::VecPrec beta,\
                                      Vector<TemplateMode<CASE>::Type>& res,\
                                      const Operator<TemplateMode<CASE>::Type>& A);
AMGX_FORALL_BUILDS(AMGX_CASE_LINE)
#undef AMGX_CASE_LINE

#define AMGX_CASE_LINE(CASE) \
template void multivector_column_norms(const Vector<TemplateMode<CASE>::Type>&,\
                         Vector<typename TemplateMode<CASE>::Type::template setMemSpace<AMGX_host>::Type>&,\
                                       const Operator<TemplateMode<CASE>::Type>&);
AMGX_FORALL_BUILDS(AMGX_CASE_LINE)
#undef AMGX_CASE_LINE


}
