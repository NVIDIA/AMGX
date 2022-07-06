/* Copyright (c) 2013-2017, NVIDIA CORPORATION. All rights reserved.
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

#include <eigensolvers/multivector_operations.h>
#include <blas.h>
#include <amgx_cublas.h>

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
    thrust::fill(h_res.begin(), h_res.end(), 0);
    cudaCheckError();

    for (int i = 0; i < gathered.size(); ++i)
    {
        thrust::transform(gathered[i].begin(), gathered[i].end(),
                          h_res.begin(), h_res.begin(), thrust::plus<ValueTypeVec>());
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
        thrust::fill(results.begin(), results.end(), 0);
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

template void distributed_gemm_TN(typename TConfigGeneric_d::VecPrec alpha,
        const Vector<TConfigGeneric_d>& lhs,
        const Vector<TConfigGeneric_d>& rhs,
        typename TConfigGeneric_d::VecPrec beta,
        Vector<TConfigGeneric_d>& res,
        const Operator<TConfigGeneric_d>& A);
template void distributed_gemm_TN(typename TConfigGeneric_h::VecPrec alpha,
        const Vector<TConfigGeneric_h>& lhs,
        const Vector<TConfigGeneric_h>& rhs,
        typename TConfigGeneric_h::VecPrec beta,
        Vector<TConfigGeneric_h>& res,
        const Operator<TConfigGeneric_h>& A);


template void multivector_column_norms(const Vector<TConfigGeneric_d>&,
        Vector<typename TConfigGeneric_d::template setMemSpace<AMGX_host>::Type>&,
        const Operator<TConfigGeneric_d>&);
template void multivector_column_norms(const Vector<TConfigGeneric_h>&,
        Vector<typename TConfigGeneric_h::template setMemSpace<AMGX_host>::Type>&,
        const Operator<TConfigGeneric_h>&);


}
