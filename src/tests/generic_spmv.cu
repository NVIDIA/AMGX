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
#include "test_utils.h"
#include "util.h"
#include <amg_config.h>
#include <generic_spmv_csr.h>
#include <specific_spmv.h>
#include <multiply.h>

namespace amgx
{
template <typename IndexType, typename ValueType>
struct CheckGenericSpMV
{
    const IndexType *_A_offsets, *_A_col_indices;
    const ValueType *_A_values;

    CheckGenericSpMV(const IndexType *A_offsets, const IndexType *A_col_indices, const ValueType *A_values) :
        _A_offsets(A_offsets), _A_col_indices(A_col_indices), _A_values(A_values) {};

    __host__ __device__
    thrust::tuple<ValueType, ValueType> operator()(IndexType i)
    {
        IndexType row = i;
        ValueType sum = 0., coef = 0.;

        for (IndexType jj = _A_offsets[row]; jj < _A_offsets[row + 1]; jj++)
        {
            sum += _A_values[jj];
            coef = max(coef, fabs(_A_values[jj]));
        }

        return thrust::tuple<ValueType, ValueType>(coef, sum);
    }
};

template <typename ValueType>
struct getCoef
{
    __host__ __device__
    ValueType operator()(const thrust::tuple<ValueType, ValueType> &a) const
    {
        return thrust::get<0>(a);
    }
};

template <typename ValueType>
struct getSum
{
    __host__ __device__
    ValueType operator()(const thrust::tuple<ValueType, ValueType> &a) const
    {
        return thrust::get<1>(a);
    }
};

template <typename Matrix, typename Vector>
void checkRowSumCoef(const Matrix &A, Vector &max_coef, Vector &row_sums)
{
    typedef typename Matrix::memory_space MemorySpace;
    typedef typename Matrix::index_type IndexType;
    typedef typename Matrix::value_type ValueType;
    cusp::array1d<thrust::tuple<ValueType, ValueType>, MemorySpace> out(A.get_num_rows());
    CheckGenericSpMV<IndexType, ValueType> checker(A.row_offsets.raw(), A.col_indices.raw(), A.values.raw());
    // perform transform on each row
    typedef thrust::counting_iterator<IndexType> c_iter;
    thrust::transform(c_iter(0), c_iter(A.get_num_rows()), out.begin(), checker);
    cudaCheckError();
    // retrieve results
    getCoef<ValueType> getC;
    getSum<ValueType> getS;
    thrust::transform(out.begin(), out.end(), max_coef.begin(), getC);
    thrust::transform(out.begin(), out.end(), row_sums.begin(), getS);
    cudaCheckError();
}

DECLARE_UNITTEST_BEGIN(GenericSpMVTest);

// struct for generic spmvt
template <typename T>
struct spmv
{
    T val;

    __host__ __device__
    spmv() : val(0) {};

    __host__ __device__
    spmv(T v) : val(v) {};

    template <typename valType1, typename valType2, typename indType, typename state>
    __host__ __device__ __forceinline__
    static spmv binary_op(const valType1 &a, const valType2 b, const indType row, const indType col,
                          const state &st)
    {
        return spmv(a * b[col]);
    }

    __host__ __device__ __forceinline__
    static spmv reduce_op(const spmv &a, const spmv &b)
    {
        return spmv(a.val + b.val);
    }

    __host__ __device__ __forceinline__
    static spmv post_op(const spmv &a)
    {
        return spmv(a.val);
    }

    template <typename outType>
    __host__ __device__ __forceinline__
    void retrieve(outType &out) const
    {
        out = val;
    }
};

struct default_state
{
};



template <typename IndexType, typename ValueType>
struct CheckMaxRowCoef
{
    const IndexType *_A_offsets;
    const IndexType *_A_cols;
    const ValueType *_A_vals;
};

void run()
{
    AMG_Config cfg;
    cfg.parseParameterString("");
    // Allocate a matrix
    Matrix<TConfig> A;
    generateMatrixRandomStruct<TConfig>::generate(A, 5000, false, 1, false);
    A.set_initialized(0);
    random_fill(A);
    A.set_initialized(1);
    Vector<TConfig> x(A.get_num_rows());
    random_fill(x);
    Vector<TConfig> y(A.get_num_rows(), 0), y_cusp(A.get_num_rows(), 0);
    // setup and perform generic spmv operation on device
    thrust::device_vector<spmv<typename TConfig::VecPrec> > y_spmv(A.get_num_rows());
    default_state st;
    genericSpmvCSR(A, x, y_spmv, st);
    retrieveOneArgument(y_spmv, y);
    // cusp multiplication for checking
    multiply(A, x, y_cusp);
    typedef typename Vector<TConfig>::value_type ValueType;
    ValueType tol = 1e-4;
    this->PrintOnFail("Generic SpMV: Does not match Cusp for SpMV operation");
    UNITTEST_ASSERT_EQUAL_TOL(y, y_cusp, tol);
    // check a 2 return value call
    // generic spmv
    Vector<TConfig> max_coef(A.get_num_rows()), row_sums(A.get_num_rows());
    // checked values
    Vector<TConfig> mc_test(A.get_num_rows()), rs_test(A.get_num_rows());
    maxCoefAndSum(A, max_coef, row_sums);
    checkRowSumCoef(A, mc_test, rs_test);
    this->PrintOnFail("Generic SpMV: Maximum coefficients do not match");
    UNITTEST_ASSERT_EQUAL_TOL(max_coef, mc_test, tol);
    this->PrintOnFail("Generic SpMV: Row sums do not match");
    UNITTEST_ASSERT_EQUAL_TOL(row_sums, rs_test, tol);
}

DECLARE_UNITTEST_END(GenericSpMVTest);

GenericSpMVTest <TemplateMode<AMGX_mode_hDDI>::Type>  GenericSpMVTest_instance_mode_hDDI;
GenericSpMVTest <TemplateMode<AMGX_mode_hFFI>::Type>  GenericSpMVTest_instance_mode_hFFI;
GenericSpMVTest <TemplateMode<AMGX_mode_dDDI>::Type>  GenericSpMVTest_instance_mode_dDDI;
GenericSpMVTest <TemplateMode<AMGX_mode_dFFI>::Type>  GenericSpMVTest_instance_mode_dFFI;

} // namespace amgx
