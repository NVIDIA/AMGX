// SPDX-FileCopyrightText: 2011 - 2025 NVIDIA CORPORATION. All Rights Reserved.
//
// SPDX-License-Identifier: BSD-3-Clause

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
#include <thrust_wrapper.h>

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
        B.set_allow_recompute_diag(false);

#ifdef ENABLE_CUSPARSE_TRANSPOSE
        Cusparse::transpose(A, B);
#else
        MatrixCusp<typename Matrix::TConfig, cusp::csr_format> wA((Matrix *) &A);
        MatrixCusp<typename Matrix::TConfig, cusp::csr_format> wB(&B);
        cusp::transpose(wA, wB);
#endif

        B.set_allow_recompute_diag(true);
        cudaCheckError();
        B.computeDiagonal();
        typedef typename Matrix::TConfig::MatPrec ValueTypeA;

        if (types::util<ValueTypeA>::is_complex)
        {
            thrust_wrapper::transform<Matrix::TConfig::memSpace>(B.values.begin(), B.values.end(), B.values.begin(), conjugate());
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
    B.set_allow_recompute_diag(false);

#if ENABLE_CUSPARSE_TRANSPOSE
    int num_nz = A.row_offsets[num_rows];
    B.resize(A.get_num_cols(), num_rows, num_nz);
    Cusparse::transpose(A, B, num_rows, num_nz);
#else
    MatrixCusp<typename Matrix::TConfig, cusp::csr_format> wA((Matrix *) &A);
    MatrixCusp<typename Matrix::TConfig, cusp::csr_format> wB(&B);

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
#endif

    B.set_allow_recompute_diag(true);
    cudaCheckError();
    B.computeDiagonal();
    typedef typename Matrix::TConfig::MatPrec ValueTypeA;

    if (types::util<ValueTypeA>::is_complex)
    {
        amgx::thrust::transform(B.values.begin(), B.values.end(), B.values.begin(), conjugate());
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
