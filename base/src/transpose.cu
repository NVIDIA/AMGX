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


#include <transpose.h>
#include <basic_types.h>

#ifdef _WIN32
#pragma warning (push)
#pragma warning (disable : 4244 4267 4521)
#endif
#include <cusp/transpose.h>
#include <matrix_cusp.h>
#ifdef _WIN32
#pragma warning (pop)
#endif

#include <thrust/transform.h>

#include "amgx_types/util.h"

namespace amgx
{

struct conjugate
{
    template <typename T>
    __host__ __device__
    T operator() (const T &x) const
    {
        return types::util<T>::conjugate(x);
    }
};

template <class Matrix>
void transpose(const Matrix &A, Matrix &B)
{
    if (A.get_block_size() != 1 || B.get_block_size() != 1)
    {
        FatalError("Unsupported block size", AMGX_ERR_NOT_SUPPORTED_BLOCKSIZE);
    }

    if (A.get_num_nz() == 0)
    {
        if ( A.diag.size() == 0 )
        {
            FatalError("Invalid matrix (no values)", AMGX_ERR_UNKNOWN);
        }

        B.copy(A);
    }
    else
    {
        B.addProps(CSR);
        MatrixCusp<typename Matrix::TConfig, cusp::csr_format> wA((Matrix *) &A);
        MatrixCusp<typename Matrix::TConfig, cusp::csr_format> wB(&B);
        B.set_allow_recompute_diag(false);
        cusp::transpose(wA, wB);
        B.set_allow_recompute_diag(true);
        cudaCheckError();
        B.computeDiagonal();
        typedef typename Matrix::TConfig::MatPrec ValueTypeA;

        if (types::util<ValueTypeA>::is_complex)
        {
            thrust::transform(B.values.begin(), B.values.end(), B.values.begin(), conjugate());
        }

        B.set_initialized(1);
    }
}


template <class Matrix>
void transpose(const Matrix &A, Matrix &B, int num_rows)
{
    if (A.get_block_size() != 1 || B.get_block_size() != 1)
    {
        FatalError("Unsupported block size", AMGX_ERR_NOT_SUPPORTED_BLOCKSIZE);
    }

    B.addProps(CSR);
    MatrixCusp<typename Matrix::TConfig, cusp::csr_format> wA((Matrix *) &A);
    MatrixCusp<typename Matrix::TConfig, cusp::csr_format> wB(&B);
    B.set_allow_recompute_diag(false);
    // operate on wA / wB
    typedef typename Matrix::index_type   IndexType;
    typedef typename Matrix::value_type   ValueType;
    typedef typename Matrix::memory_space MemorySpace;
    int num_entries = A.row_offsets[num_rows];
    int num_cols = A.get_num_cols();
    // resize matrix
    wB.resize(num_cols, num_rows, num_entries);
    // resize row offsets
    cusp::array1d<IndexType, MemorySpace> wA_row_offsets(wA.row_offsets);
    wA_row_offsets.resize(num_rows + 1);
    cusp::detail::offsets_to_indices(wA_row_offsets, wB.column_indices);
    // resize values
    cusp::array1d<ValueType, MemorySpace> wA_values(wA.values);
    wA_values.resize(num_entries);
    cusp::copy(wA_values, wB.values);
    // resize col indices
    cusp::array1d<IndexType, MemorySpace> wB_row_indices(wA.column_indices);
    wB_row_indices.resize(num_entries);

    if (num_rows > 0 && num_entries > 0)
    {
        cusp::detail::sort_by_row(wB_row_indices, wB.column_indices, wB.values);
        cusp::detail::indices_to_offsets(wB_row_indices, wB.row_offsets);
    }

    B.set_allow_recompute_diag(true);
    cudaCheckError();
    B.computeDiagonal();
    typedef typename Matrix::TConfig::MatPrec ValueTypeA;

    if (types::util<ValueTypeA>::is_complex)
    {
        thrust::transform(B.values.begin(), B.values.end(), B.values.begin(), conjugate());
    }

    B.set_initialized(1);
}


/****************************************
 * Explict instantiations
 ***************************************/
#define AMGX_CASE_LINE(CASE) template void transpose<Matrix<TemplateMode<CASE>::Type> >(const Matrix<TemplateMode<CASE>::Type> &A, Matrix<TemplateMode<CASE>::Type>& B);
AMGX_FORALL_BUILDS(AMGX_CASE_LINE)
AMGX_FORCOMPLEX_BUILDS(AMGX_CASE_LINE)
#undef AMGX_CASE_LINE

#define AMGX_CASE_LINE(CASE) template void transpose<Matrix<TemplateMode<CASE>::Type> >(const Matrix<TemplateMode<CASE>::Type> &A, Matrix<TemplateMode<CASE>::Type>& B, int num_rows);
AMGX_FORALL_BUILDS(AMGX_CASE_LINE)
AMGX_FORCOMPLEX_BUILDS(AMGX_CASE_LINE)
#undef AMGX_CASE_LINE


} // namespace amgx
