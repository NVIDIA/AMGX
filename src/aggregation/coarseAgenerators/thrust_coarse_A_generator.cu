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

// --------------------------------------------------------
// --------------------------------------------------------
#include <aggregation/coarseAgenerators/thrust_coarse_A_generator.h>
#include <thrust/system/detail/generic/reduce_by_key.h>
#include <thrust/remove.h>
#include <thrust/iterator/transform_iterator.h>
#include <error.h>
#include <cutil.h>
#include <types.h>
#include <cusp/detail/format_utils.h>

namespace amgx
{

namespace aggregation
{

typedef thrust::tuple<int, int> tuple_t;

// --------------------
// Kernels
// --------------------

// Kernel to store aggregate I of each fine point index i
template <typename IndexType>
__global__
void iToIKernel(const IndexType *row_offsets, const IndexType *aggregates, IndexType *I, const int num_rows)
{
    for (int tid = blockDim.x * blockIdx.x + threadIdx.x; tid < num_rows; tid += gridDim.x * blockDim.x)
    {
        int agg = aggregates[tid];

        for (int j = row_offsets[tid]; j < row_offsets[tid + 1]; j++)
        {
            I[j] = agg;
        }
    }
}

// Kernel to store aggregate J of each fine point index j
template <typename IndexType>
__global__
void jToJKernel(const IndexType *column_indices, const IndexType *aggregates, IndexType *J, const int num_entries)
{
    for (int tid = blockDim.x * blockIdx.x + threadIdx.x; tid < num_entries; tid += gridDim.x * blockDim.x)
    {
        int j = column_indices[tid];
        J[tid] = aggregates[j];
    }
}

// Constructor
template<class T_Config>
ThrustCoarseAGeneratorBase<T_Config>::ThrustCoarseAGeneratorBase()
{
}

//-----------------------------------------------------
// Method to compute the Galerkin product: A_c=R*A*P
//-----------------------------------------------------

// Method to compute A on DEVICE using csr format
template <AMGX_VecPrecision t_vecPrec, AMGX_MatPrecision t_matPrec, AMGX_IndPrecision t_indPrec>
void ThrustCoarseAGenerator<TemplateConfig<AMGX_device, t_vecPrec, t_matPrec, t_indPrec> >::computeAOperator_1x1(const Matrix_d &A, Matrix_d &Ac, const IVector &aggregates, const IVector &R_row_offsets, const IVector &R_column_indices, const int num_aggregates)
{
    if (A.hasProps(DIAG))
    {
        FatalError("ThrustCoarseAGenerator: unsupported diagonal", AMGX_ERR_NOT_IMPLEMENTED);
    }

    cudaCheckError();
    IVector I(A.get_num_nz(), -1);
    IVector J(A.get_num_nz(), -1);
    VVector V(A.get_num_nz(), -1);
    const int block_size_I = 128;
    const int block_size_J = 256;
    const int num_blocks_I = std::min( AMGX_GRID_MAX_SIZE, (int) ((A.get_num_rows() - 1) / block_size_I + 1) );
    const int num_blocks_J = std::min( AMGX_GRID_MAX_SIZE, (int) ((A.get_num_nz() - 1) / block_size_J + 1) );
    const IndexType *row_offsets_ptr = A.row_offsets.raw();
    const IndexType *column_indices_ptr = A.col_indices.raw();
    const IndexType *aggregates_ptr = aggregates.raw();
    IndexType *I_ptr = I.raw();
    IndexType *J_ptr = J.raw();
    // Kernel to fill array I with aggregates number for fine points i
    iToIKernel <<< num_blocks_I, block_size_I>>>(row_offsets_ptr, aggregates_ptr, I_ptr, (int)A.get_num_rows());
    cudaCheckError();
    // Kernel to fill array J with aggregates number for fine points j
    jToJKernel <<< num_blocks_J, block_size_J>>>(column_indices_ptr, aggregates_ptr, J_ptr, (int)A.get_num_nz());
    cudaCheckError();
    // Copy A.values to V array
    thrust::copy(A.values.begin(), A.values.begin() + A.get_num_nz()*A.get_block_size(), V.begin());
    cudaCheckError();
    // Sort (I,J,V) by rows and columns (I,J)
    cusp::detail::sort_by_row_and_column(I, J, V);
    cudaCheckError();
    // compute unique number of nonzeros in the output
    IndexType NNZ = thrust::inner_product(thrust::make_zip_iterator(thrust::make_tuple(I.begin(), J.begin())),
                                          thrust::make_zip_iterator(thrust::make_tuple(I.end (),  J.end()))   - 1,
                                          thrust::make_zip_iterator(thrust::make_tuple(I.begin(), J.begin())) + 1,
                                          IndexType(0),
                                          thrust::plus<IndexType>(),
                                          thrust::not_equal_to< thrust::tuple<IndexType, IndexType> >()) + 1;
    cudaCheckError();
    // allocate space for coarse matrix Ac
    Ac.addProps(CSR);

    if (A.hasProps(DIAG)) { Ac.addProps(DIAG); }

    if (A.is_matrix_singleGPU())
    {
        Ac.resize(num_aggregates, num_aggregates, NNZ, 1);
    }
    else
    {
        Ac.resize_spare(num_aggregates, num_aggregates, NNZ, A.get_block_dimy(), A.get_block_dimx(), 1.0);

        if (A.hasProps(DIAG)) { Ac.computeDiagonal(); }
    }

    // Reduce by key to fill in Ac.column_indices and Ac.values
    IVector new_row_indices(NNZ, 0);
    thrust::reduce_by_key(thrust::make_zip_iterator(thrust::make_tuple(I.begin(), J.begin())),
        thrust::make_zip_iterator(thrust::make_tuple(I.end(),   J.end())),
        V.begin(),
        thrust::make_zip_iterator(thrust::make_tuple(new_row_indices.begin(), Ac.col_indices.begin())),
        Ac.values.begin(),
        thrust::equal_to< thrust::tuple<IndexType, IndexType> >(),
        thrust::plus<ValueType>());
    cudaCheckError();
    // Convert array new_row_indices to offsets
    cusp::detail::indices_to_offsets(new_row_indices, Ac.row_offsets);
    cudaCheckError();
    I.clear();
    I.shrink_to_fit();
    J.clear();
    J.shrink_to_fit();
    V.clear();
    V.shrink_to_fit();
}

// Method to compute A on HOST using csr format
template <AMGX_VecPrecision t_vecPrec, AMGX_MatPrecision t_matPrec, AMGX_IndPrecision t_indPrec>
void ThrustCoarseAGenerator<TemplateConfig<AMGX_host, t_vecPrec, t_matPrec, t_indPrec> >::computeAOperator_1x1(const Matrix_h &A, Matrix_h &Ac, const IVector &aggregates, const IVector &R_row_offsets, const IVector &R_column_indices, const int num_aggregates)
{
    if (A.hasProps(DIAG))
    {
        FatalError("ThrustCoarseAGenerator: unsupported diagonal", AMGX_ERR_NOT_IMPLEMENTED);
    }

    IVector I(A.get_num_nz(), -1);
    IVector J(A.get_num_nz(), -1);
    VVector V(A.get_num_nz(), -1);
    const IndexType *row_offsets_ptr = A.row_offsets.raw();
    const IndexType *column_indices_ptr = A.col_indices.raw();
    const IndexType *aggregates_ptr = aggregates.raw();
    IndexType *I_ptr = I.raw();
    IndexType *J_ptr = J.raw();

    // Kernel to fill array I with aggregates number for fine points i
    for ( int tid = 0; tid < (int)A.get_num_rows(); tid++ )
    {
        int agg = aggregates_ptr[tid];

        for (int j = row_offsets_ptr[tid]; j < row_offsets_ptr[tid + 1]; j++)
        {
            I_ptr[j] = agg;
        }
    }

    // Kernel to fill array J with aggregates number for fine points j
    for ( int tid = 0; tid < (int)A.get_num_nz(); tid++ )
    {
        int j = column_indices_ptr[tid];
        J_ptr[tid] = aggregates_ptr[j];
    }

    // Copy A.values to V array
    thrust::copy(A.values.begin(), A.values.begin() + A.get_num_nz()*A.get_block_size(), V.begin());
    cudaCheckError();
    // Sort (I,J,V) by rows and columns (I,J)
    cusp::detail::sort_by_row_and_column(I, J, V);
    cudaCheckError();
    // compute unique number of nonzeros in the output
    IndexType NNZ = thrust::inner_product(thrust::make_zip_iterator(thrust::make_tuple(I.begin(), J.begin())),
                                          thrust::make_zip_iterator(thrust::make_tuple(I.end (),  J.end()))   - 1,
                                          thrust::make_zip_iterator(thrust::make_tuple(I.begin(), J.begin())) + 1,
                                          IndexType(0),
                                          thrust::plus<IndexType>(),
                                          thrust::not_equal_to< thrust::tuple<IndexType, IndexType> >()) + 1;
    cudaCheckError();
    // allocate space for coarse matrix Ac
    Ac.addProps(CSR);

    if (A.hasProps(DIAG)) { Ac.addProps(DIAG); }

    if (A.is_matrix_singleGPU())
    {
        Ac.resize(num_aggregates, num_aggregates, NNZ, 1);
    }
    else
    {
        Ac.resize_spare(num_aggregates, num_aggregates, NNZ, A.get_block_dimy(), A.get_block_dimx(), 1.0);

        if (A.hasProps(DIAG)) { Ac.computeDiagonal(); }
    }

    // Reduce by key to fill in Ac.column_indices and Ac.values
    typename Matrix_h::IVector new_row_indices(NNZ, 0);
    thrust::reduce_by_key(thrust::make_zip_iterator(thrust::make_tuple(I.begin(), J.begin())),
        thrust::make_zip_iterator(thrust::make_tuple(I.end(),   J.end())),
        V.begin(),
        thrust::make_zip_iterator(thrust::make_tuple(new_row_indices.begin(), Ac.col_indices.begin())),
        Ac.values.begin(),
        thrust::equal_to< thrust::tuple<IndexType, IndexType> >(),
        thrust::plus<ValueType>());
    cudaCheckError();
    // Convert array new_row_indices to offsets
    cusp::detail::indices_to_offsets(new_row_indices, Ac.row_offsets);
    cudaCheckError();
    I.clear();
    I.shrink_to_fit();
    J.clear();
    J.shrink_to_fit();
    V.clear();
    V.shrink_to_fit();
}

// ------------------------------------------------
template <class T_Config>
void ThrustCoarseAGeneratorBase<T_Config>::computeAOperator(const Matrix<T_Config> &A, Matrix<T_Config> &Ac, const IVector &aggregates, const IVector &R_row_offsets, const IVector &R_column_indices, const int num_aggregates)
{
    Ac.set_initialized(0);

    if (A.get_block_dimx() == 1 && A.get_block_dimy() == 1)
    {
        computeAOperator_1x1( A, Ac, aggregates, R_row_offsets, R_column_indices, num_aggregates );
    }
    else
    {
        FatalError("Unsupported block size for ThrustCoarseAGenerator", AMGX_ERR_NOT_SUPPORTED_BLOCKSIZE);
    }

    if (Ac.is_matrix_singleGPU()) { Ac.computeDiagonal(); }

    Ac.set_initialized(1);
}

// ---------------------------
// Explict instantiations
// ---------------------------
#define AMGX_CASE_LINE(CASE) template class ThrustCoarseAGeneratorBase<TemplateMode<CASE>::Type>;
AMGX_FORALL_BUILDS(AMGX_CASE_LINE)
#undef AMGX_CASE_LINE
#define AMGX_CASE_LINE(CASE) template class ThrustCoarseAGenerator<TemplateMode<CASE>::Type>;
AMGX_FORALL_BUILDS(AMGX_CASE_LINE)
#undef AMGX_CASE_LINE

}
}
