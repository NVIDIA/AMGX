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

//-----------------------------------------------------
// Method to compute the Galerkin product: A_c=R*A*P
// The number of non-zeros per row is computed using Thrust
// The coarse matrix is created by custom kernel
// --------------------------------------------------------

#include <aggregation/coarseAgenerators/hybrid_coarse_A_generator.h>
#include <thrust/system/detail/generic/reduce_by_key.h>
#include <cusp/detail/format_utils.h> //indices_to_offsets
#include <thrust/remove.h>
#include <thrust/extrema.h>
#include <thrust/adjacent_difference.h>
#include <thrust/iterator/transform_iterator.h>
#include <thrust/unique.h>
#include <error.h>
#include <cutil.h>
#include <util.h>
#include <types.h>

namespace amgx
{

namespace aggregation
{

typedef thrust::tuple<int, int> tuple_t;

struct isDiagonal
{
    __host__ __device__
    bool operator()(tuple_t IJ)
    {
        return ( IJ.get<0>() == IJ.get<1>() );
    }

};

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

// Kernel to fill matrix Ac for 4x4 block matrices
template <typename IndexType, typename ValueType, int max_nonzeros_per_row, int threads_per_block, int bsize_sq, int log_bsize_sq>
__global__
void fillA_4by4Blocks_Thrust_Kernel(const IndexType *A_row_offsets, const IndexType *A_column_indices, const IndexType *A_dia_values, const ValueType *A_nonzero_values, const IndexType *R_row_offsets, const IndexType *R_column_indices, const IndexType *Ac_row_offsets, const IndexType *Ac_column_indices, IndexType *Ac_dia_values, ValueType *Ac_nonzero_values, const IndexType *aggregates, const int num_aggregates, const int num_threads)
{
    const int tid = threadIdx.x;
    const int I = blockIdx.x * threads_per_block + tid;
    const int halfwarp_id = tid >> log_bsize_sq;
    const int mat_entry_index = tid  & (bsize_sq - 1);
    volatile __shared__ IndexType current_cell_to_read[threads_per_block / bsize_sq];
    volatile __shared__ IndexType current_cell_to_write[threads_per_block / bsize_sq];
    volatile __shared__ IndexType max_nonzeros_i[threads_per_block];
    volatile __shared__ IndexType max_nonzeros_j[threads_per_block];
    const int size_hash_reg = (2. / 3.) * max_nonzeros_per_row; // 2/3 is arbitrary
    const int size_hash_shared = max_nonzeros_per_row - size_hash_reg;
    IndexType shared_offset = threadIdx.x * size_hash_shared;
    volatile __shared__ IndexType hash_shared[size_hash_shared * threads_per_block];
    IndexType hash_reg[size_hash_reg];
    int Ac_offset_min;
    int Ac_offset_max;
#pragma unroll

    for (int i = 0; i < size_hash_reg; i++)
    {
        hash_reg[i] = -1;
    }

    for (int i = 0; i < size_hash_shared; i++)
    {
        hash_shared[shared_offset + i] = -1;
    }

    // Find maximum number of nonzeros for threads in same halfwarp
    int i_offset_min, i_offset_max;

    if (I < num_aggregates)
    {
        i_offset_min = R_row_offsets[I];
        i_offset_max = R_row_offsets[I + 1];
        max_nonzeros_i[tid] = i_offset_max - i_offset_min;
        Ac_offset_min = Ac_row_offsets[I];
        Ac_offset_max = Ac_row_offsets[I + 1];

        for (int k = Ac_offset_min, icount = 0; k < Ac_offset_max; k++, icount++)
        {
            if (icount < size_hash_reg)
            {
                hash_reg[icount] = Ac_column_indices[k];
            }
            else
            {
                hash_shared[shared_offset + icount - size_hash_reg] = Ac_column_indices[k];
            }
        }
    }
    else
    {
        i_offset_min = 0;
        i_offset_max = 0;
        max_nonzeros_i[tid] = 0;
    }

    int max_loop_i = 0;

    for (int m = 0; m < bsize_sq; m++)
    {
        int max_nonzeros = max_nonzeros_i[halfwarp_id * bsize_sq + m];

        if (max_nonzeros > max_loop_i)
        {
            max_loop_i = max_nonzeros;
        }
    }

    // For all threads that could do useful work
    if (I < num_threads)
    {
        for (int iloop = 0, i_offset = i_offset_min; iloop < max_loop_i; iloop++, i_offset++)
        {
            // Contribution from A_dia_values of fine point i
            int i = (i_offset < i_offset_max)  ? R_column_indices[i_offset] : -1 ;

            // Have threads collaborate to load in coalesced fashion
            for (int m = 0; m < bsize_sq; m++)
            {
                // Write which cell to load in shared memory buffer
                if (mat_entry_index == m)
                {
                    current_cell_to_read[halfwarp_id] = i;
                    current_cell_to_write[halfwarp_id] = I;
                }

                // All threads read from shared which cell to read and write
                int cell_to_read = current_cell_to_read[halfwarp_id];
                int cell_to_write = current_cell_to_write[halfwarp_id];

                // Here all threads in half-warp will take same path
                if (cell_to_read != -1)
                {
                    Ac_nonzero_values[ bsize_sq * Ac_dia_values[cell_to_write] + mat_entry_index] += A_nonzero_values[bsize_sq * A_dia_values[cell_to_read] + mat_entry_index];
                }
            }

            // Contribution from A_nonzero_values of fine point i
            // Find maximum number of nonzeros for threads in same halfwarp_id
            int j_offset_min, j_offset_max;

            if (i != -1)
            {
                j_offset_min = A_row_offsets[i];
                j_offset_max = A_row_offsets[i + 1];
                max_nonzeros_j[tid] = j_offset_max - j_offset_min;
            }
            else
            {
                j_offset_min = 0;
                j_offset_max = 0;
                max_nonzeros_j[tid] = 0;
            }

            int max_loop_j = 0;

            for (int m = 0; m < bsize_sq; m++)
            {
                int max_nonzeros = max_nonzeros_j[halfwarp_id * bsize_sq + m];

                if (max_nonzeros > max_loop_j)
                {
                    max_loop_j = max_nonzeros;
                }
            }

            // Add contribution from nonzero_values of A
            int j_offset_to_read;

            for (int jloop = 0, j_offset = j_offset_min; jloop < max_loop_j; jloop++, j_offset++)
            {
                int J, j, k;

                if (j_offset < j_offset_max)
                {
                    j_offset_to_read = j_offset;
                    j = A_column_indices[j_offset];
                    J = aggregates[j];

                    // Find index k where to store the data and create A_column_indices
                    if (I != J)
                    {
                        // This weird construct is to allow loop unrolling and avoid register spilling (see original version below)
                        int found = 0;
                        k = Ac_offset_min;
#pragma unroll

                        for (int icount = 0; icount < max_nonzeros_per_row; icount++)
                        {
                            if (k < Ac_offset_max)
                            {
                                if (found == 0)
                                {
                                    int Jtemp = (icount < size_hash_reg)  ? hash_reg[icount] : hash_shared[shared_offset + icount - size_hash_reg];

                                    if (J == Jtemp)
                                    {
                                        found = 1;
                                    }

                                    if (found == 0) { k++; }
                                }
                            }
                        }
                    } // if I != J
                } // if j_offset < j_offset_max
                else
                {
                    j_offset_to_read = -1;
                }

                // Have threads collaborate to load in coalesced fashion
                for (int m = 0; m < bsize_sq; m++)
                {
                    // Write which cell to load in shared memory buffer
                    if (mat_entry_index == m)
                    {
                        current_cell_to_read[halfwarp_id] = j_offset_to_read;

                        if (I != J)
                        {
                            current_cell_to_write[halfwarp_id] = -k;
                        }
                        else
                        {
                            current_cell_to_write[halfwarp_id] = I + 1;
                        }
                    }

                    // All threads read from shared which cell to read and write
                    int cell_to_read = current_cell_to_read[halfwarp_id];
                    int cell_to_write = current_cell_to_write[halfwarp_id];

                    if (cell_to_read != -1)
                    {
                        if (cell_to_write <= 0)
                        {
                            Ac_nonzero_values[(-cell_to_write)*bsize_sq + mat_entry_index] += A_nonzero_values[cell_to_read * bsize_sq + mat_entry_index];
                        }
                        else
                        {
                            Ac_nonzero_values[ Ac_dia_values[ cell_to_write - 1 ] * bsize_sq + mat_entry_index] += A_nonzero_values[cell_to_read * bsize_sq + mat_entry_index];
                        }
                    }
                }
            } // j_offset_loop
        } // i_offset_loop
    } // if tid < num_threads
}

// Constructor
template<class T_Config>
HybridCoarseAGeneratorBase<T_Config>::HybridCoarseAGeneratorBase()
{
}

//-----------------------------------------------------
// Method to compute the Galerkin product: A_c=R*A*P
// The number of non-zeros per row is computed using Thrust
// The coarse matrix is created by custom kernel
//-----------------------------------------------------

template <AMGX_VecPrecision t_vecPrec, AMGX_MatPrecision t_matPrec, AMGX_IndPrecision t_indPrec>
void HybridCoarseAGenerator<TemplateConfig<AMGX_device, t_vecPrec, t_matPrec, t_indPrec> >::computeAOperator_4x4(const Matrix<TemplateConfig<AMGX_device, t_vecPrec, t_matPrec, t_indPrec> > &A, Matrix<TemplateConfig<AMGX_device, t_vecPrec, t_matPrec, t_indPrec> > &Ac, const typename Matrix_d::IVector &aggregates, const typename Matrix_d::IVector &R_row_offsets, const typename Matrix_d::IVector &R_column_indices, const int num_aggregates)
{
// supports both DIA properties
    if (!A.hasProps(DIAG))
    {
        FatalError("Hybryd coarser does not support inside diagonal yet\n", AMGX_ERR_NOT_IMPLEMENTED);
    }

    if (A.get_num_nz() == 0)
    {
        FatalError("Hybryd coarser does not work correctly with diagonal matrices\n", AMGX_ERR_NOT_IMPLEMENTED);
    }

    typename Matrix_d::IVector I(A.get_num_nz(), -1);
    typename Matrix_d::IVector J(A.get_num_nz(), -1);
    typedef device_vector_alloc<IndexType> IntVector;
    typedef typename IntVector::iterator IntIterator;
    typedef thrust::tuple< IntIterator, IntIterator> IntIteratorTuple;
    typedef thrust::zip_iterator<IntIteratorTuple> ZipIterator;
    ZipIterator new_end;
    const int block_size_I = 128;
    const int block_size_J = 256;
    const int num_blocks_I = min( AMGX_GRID_MAX_SIZE, (int) ((A.get_num_rows() - 1) / block_size_I + 1));
    const int num_blocks_J = min( AMGX_GRID_MAX_SIZE, (int) ((A.get_num_nz() - 1) / block_size_J + 1));
    const IndexType *A_row_offsets_ptr = A.row_offsets.raw();
    const IndexType *A_column_indices_ptr = A.col_indices.raw();
    const IndexType *A_dia_values_ptr = A.diag.raw();
    const ValueType *A_nonzero_values_ptr = A.values.raw();
    const IndexType *aggregates_ptr = aggregates.raw();
    IndexType *I_ptr = I.raw();
    IndexType *J_ptr = J.raw();
    // Kernel to fill array I with aggregates number for fine points i
    iToIKernel <<< num_blocks_I, block_size_I>>>(A_row_offsets_ptr, aggregates_ptr, I_ptr, (int) A.get_num_rows());
    cudaCheckError();
    // Kernel to fill array J with aggregates number for fine points j
    jToJKernel <<< num_blocks_J, block_size_J>>>(A_column_indices_ptr, aggregates_ptr, J_ptr, (int) A.get_num_nz());
    cudaCheckError();
    // Sort (I,J) by rows and columns (I,J)
    IVector permutation(A.get_num_nz());
    thrust::sequence(permutation.begin(), permutation.end());
    cudaCheckError();
    // compute permutation and sort by (I,J)
    {
        IVector temp(J);
        thrust::stable_sort_by_key(temp.begin(), temp.end(), permutation.begin());
        cudaCheckError();
        temp = I;
        //I = temp;
        thrust_wrapper::gather(permutation.begin(), permutation.end(), temp.begin(), I.begin());
        cudaCheckError();
        thrust::stable_sort_by_key(I.begin(), I.end(), permutation.begin());
        cudaCheckError();
        temp = J;
        //J = temp;
        thrust_wrapper::gather(permutation.begin(), permutation.end(), temp.begin(), J.begin());
        cudaCheckError();
    }
    // Remove duplicate tuples
    new_end = thrust::unique(thrust::make_zip_iterator(thrust::make_tuple(I.begin(), J.begin())),
                             thrust::make_zip_iterator(thrust::make_tuple(I.end(), J.end())), thrust::equal_to < thrust::tuple<IndexType, IndexType> >());
    cudaCheckError();
    IntIteratorTuple endTuple = new_end.get_iterator_tuple();
    I.erase(thrust::get<0>(endTuple), I.end());
    J.erase(thrust::get<1>(endTuple), J.end());
    // Remove diagonal terms
    new_end = thrust::remove_if(thrust::make_zip_iterator(thrust::make_tuple(I.begin(), J.begin())),
                                thrust::make_zip_iterator(thrust::make_tuple(I.end(), J.end())), isDiagonal() );
    cudaCheckError();
    endTuple = new_end.get_iterator_tuple();
    I.erase(thrust::get<0>(endTuple), I.end());
    J.erase(thrust::get<1>(endTuple), J.end());
    int nonzero_blocks = J.size();
    // Resize Ac
    Ac.addProps(CSR);

    if (A.hasProps(DIAG)) { Ac.addProps(DIAG); }

    Ac.resize(num_aggregates, num_aggregates, nonzero_blocks, A.get_block_dimy(), A.get_block_dimx());
    //Ac.resize(num_aggregates,num_aggregates,nonzero_blocks,A.get_block_dimy(),A.get_block_dimx(), 1);
    // Ac.column_indices
    Ac.col_indices = J;
    J.clear();
    J.shrink_to_fit();
    // Convert array new_row_indices to offsets
    cusp::detail::indices_to_offsets(I, Ac.row_offsets);
    cudaCheckError();
    I.resize(Ac.row_offsets.size());
    // Compute the maximum number of nonzeros
    thrust::adjacent_difference(Ac.row_offsets.begin(), Ac.row_offsets.end(), I.begin());
    cudaCheckError();
    const IndexType max_nonzero_per_row = *thrust::max_element(I.begin() + 1, I.end());
    cudaCheckError();
    //std::cout << "max_nonzero_per_row" << max_nonzero_per_row << std::endl;
    I.clear();
    I.shrink_to_fit();
    const IndexType *R_row_offsets_ptr = R_row_offsets.raw();
    const IndexType *R_column_indices_ptr = R_column_indices.raw();
    // Get max_nonzero_per_row
    //const IndexType max_nonzero_per_row = 32;
    const int threads_per_block = 128;
    // Store the column_indices in a register
    // Option to print the number of nonzeros distribution
    //printNonzeroStats(Ac_row_offsets_temp,num_aggregates);
    // Resize Ac and doing exclusive scan on Ac_row_offsets_temp
    IndexType *Ac_row_offsets_ptr = Ac.row_offsets.raw();
    IndexType *Ac_column_indices_ptr = Ac.col_indices.raw();
    IndexType *Ac_dia_values_ptr = Ac.diag.raw();
    ValueType *Ac_nonzero_values_ptr = Ac.values.raw();
    // Now create Ac.dia_values and Ac.nonzero_values
    //thrust::fill(Ac.diag.begin(),Ac.diag.end(),0.);
    thrust::fill(Ac.values.begin(), Ac.values.end(), 0.);
    cudaCheckError();
    // Coalesced version of kernel to fill A
    const int num_threads = ( ( num_aggregates + 15) / 16 ) * 16;
    const int num_blocks2 = ( ( num_threads + threads_per_block - 1) / threads_per_block );

    if (max_nonzero_per_row < 16)
    {
        fillA_4by4Blocks_Thrust_Kernel<IndexType, ValueType, 16, threads_per_block, 16, 4> <<< num_blocks2, threads_per_block>>>(A_row_offsets_ptr, A_column_indices_ptr, A_dia_values_ptr, A_nonzero_values_ptr, R_row_offsets_ptr, R_column_indices_ptr, Ac_row_offsets_ptr, Ac_column_indices_ptr, Ac_dia_values_ptr, Ac_nonzero_values_ptr, aggregates_ptr, num_aggregates, num_threads);
    }
    else if (max_nonzero_per_row < 20)
    {
        fillA_4by4Blocks_Thrust_Kernel<IndexType, ValueType, 20, threads_per_block, 16, 4> <<< num_blocks2, threads_per_block>>>(A_row_offsets_ptr, A_column_indices_ptr, A_dia_values_ptr, A_nonzero_values_ptr, R_row_offsets_ptr, R_column_indices_ptr, Ac_row_offsets_ptr, Ac_column_indices_ptr, Ac_dia_values_ptr, Ac_nonzero_values_ptr, aggregates_ptr, num_aggregates, num_threads);
    }
    else if (max_nonzero_per_row < 24)
    {
        fillA_4by4Blocks_Thrust_Kernel<IndexType, ValueType, 24, threads_per_block, 16, 4> <<< num_blocks2, threads_per_block>>>(A_row_offsets_ptr, A_column_indices_ptr, A_dia_values_ptr, A_nonzero_values_ptr, R_row_offsets_ptr, R_column_indices_ptr, Ac_row_offsets_ptr, Ac_column_indices_ptr, Ac_dia_values_ptr, Ac_nonzero_values_ptr, aggregates_ptr, num_aggregates, num_threads);
    }
    else if (max_nonzero_per_row < 28)
    {
        fillA_4by4Blocks_Thrust_Kernel<IndexType, ValueType, 28, threads_per_block, 16, 4> <<< num_blocks2, threads_per_block>>>(A_row_offsets_ptr, A_column_indices_ptr, A_dia_values_ptr, A_nonzero_values_ptr, R_row_offsets_ptr, R_column_indices_ptr, Ac_row_offsets_ptr, Ac_column_indices_ptr, Ac_dia_values_ptr, Ac_nonzero_values_ptr, aggregates_ptr, num_aggregates, num_threads);
    }
    else if (max_nonzero_per_row < 32)
    {
        fillA_4by4Blocks_Thrust_Kernel<IndexType, ValueType, 32, threads_per_block, 16, 4> <<< num_blocks2, threads_per_block>>>(A_row_offsets_ptr, A_column_indices_ptr, A_dia_values_ptr, A_nonzero_values_ptr, R_row_offsets_ptr, R_column_indices_ptr, Ac_row_offsets_ptr, Ac_column_indices_ptr, Ac_dia_values_ptr, Ac_nonzero_values_ptr, aggregates_ptr, num_aggregates, num_threads);
    }
    else if (max_nonzero_per_row < 36)
    {
        fillA_4by4Blocks_Thrust_Kernel<IndexType, ValueType, 36, threads_per_block, 16, 4> <<< num_blocks2, threads_per_block>>>(A_row_offsets_ptr, A_column_indices_ptr, A_dia_values_ptr, A_nonzero_values_ptr, R_row_offsets_ptr, R_column_indices_ptr, Ac_row_offsets_ptr, Ac_column_indices_ptr, Ac_dia_values_ptr, Ac_nonzero_values_ptr, aggregates_ptr, num_aggregates, num_threads);
    }
    else if (max_nonzero_per_row < 40)
    {
        fillA_4by4Blocks_Thrust_Kernel<IndexType, ValueType, 40, threads_per_block, 16, 4> <<< num_blocks2, threads_per_block>>>(A_row_offsets_ptr, A_column_indices_ptr, A_dia_values_ptr, A_nonzero_values_ptr, R_row_offsets_ptr, R_column_indices_ptr, Ac_row_offsets_ptr, Ac_column_indices_ptr, Ac_dia_values_ptr, Ac_nonzero_values_ptr, aggregates_ptr, num_aggregates, num_threads);
    }
    else if (max_nonzero_per_row < 44)
    {
        fillA_4by4Blocks_Thrust_Kernel<IndexType, ValueType, 44, threads_per_block, 16, 4> <<< num_blocks2, threads_per_block>>>(A_row_offsets_ptr, A_column_indices_ptr, A_dia_values_ptr, A_nonzero_values_ptr, R_row_offsets_ptr, R_column_indices_ptr, Ac_row_offsets_ptr, Ac_column_indices_ptr, Ac_dia_values_ptr, Ac_nonzero_values_ptr, aggregates_ptr, num_aggregates, num_threads);
    }
    else if (max_nonzero_per_row < 48)
    {
        fillA_4by4Blocks_Thrust_Kernel<IndexType, ValueType, 48, threads_per_block, 16, 4> <<< num_blocks2, threads_per_block>>>(A_row_offsets_ptr, A_column_indices_ptr, A_dia_values_ptr, A_nonzero_values_ptr, R_row_offsets_ptr, R_column_indices_ptr, Ac_row_offsets_ptr, Ac_column_indices_ptr, Ac_dia_values_ptr, Ac_nonzero_values_ptr, aggregates_ptr, num_aggregates, num_threads);
    }
    else if (max_nonzero_per_row < 52)
    {
        fillA_4by4Blocks_Thrust_Kernel<IndexType, ValueType, 52, threads_per_block, 16, 4> <<< num_blocks2, threads_per_block>>>(A_row_offsets_ptr, A_column_indices_ptr, A_dia_values_ptr, A_nonzero_values_ptr, R_row_offsets_ptr, R_column_indices_ptr, Ac_row_offsets_ptr, Ac_column_indices_ptr, Ac_dia_values_ptr, Ac_nonzero_values_ptr, aggregates_ptr, num_aggregates, num_threads);
    }
    else if (max_nonzero_per_row < 56)
    {
        fillA_4by4Blocks_Thrust_Kernel<IndexType, ValueType, 56, threads_per_block, 16, 4> <<< num_blocks2, threads_per_block>>>(A_row_offsets_ptr, A_column_indices_ptr, A_dia_values_ptr, A_nonzero_values_ptr, R_row_offsets_ptr, R_column_indices_ptr, Ac_row_offsets_ptr, Ac_column_indices_ptr, Ac_dia_values_ptr, Ac_nonzero_values_ptr, aggregates_ptr, num_aggregates, num_threads);
    }
    else if (max_nonzero_per_row < 60)
    {
        fillA_4by4Blocks_Thrust_Kernel<IndexType, ValueType, 60, threads_per_block, 16, 4> <<< num_blocks2, threads_per_block>>>(A_row_offsets_ptr, A_column_indices_ptr, A_dia_values_ptr, A_nonzero_values_ptr, R_row_offsets_ptr, R_column_indices_ptr, Ac_row_offsets_ptr, Ac_column_indices_ptr, Ac_dia_values_ptr, Ac_nonzero_values_ptr, aggregates_ptr, num_aggregates, num_threads);
    }
    else if (max_nonzero_per_row < 64)
    {
        fillA_4by4Blocks_Thrust_Kernel<IndexType, ValueType, 64, threads_per_block, 16, 4> <<< num_blocks2, threads_per_block>>>(A_row_offsets_ptr, A_column_indices_ptr, A_dia_values_ptr, A_nonzero_values_ptr, R_row_offsets_ptr, R_column_indices_ptr, Ac_row_offsets_ptr, Ac_column_indices_ptr, Ac_dia_values_ptr, Ac_nonzero_values_ptr, aggregates_ptr, num_aggregates, num_threads);
    }
    else if (max_nonzero_per_row < 128)
    {
        fillA_4by4Blocks_Thrust_Kernel<IndexType, ValueType, 128, threads_per_block, 16, 4> <<< num_blocks2, threads_per_block>>>(A_row_offsets_ptr, A_column_indices_ptr, A_dia_values_ptr, A_nonzero_values_ptr, R_row_offsets_ptr, R_column_indices_ptr, Ac_row_offsets_ptr, Ac_column_indices_ptr, Ac_dia_values_ptr, Ac_nonzero_values_ptr, aggregates_ptr, num_aggregates, num_threads);
    }
    else if (max_nonzero_per_row < 256)
    {
        fillA_4by4Blocks_Thrust_Kernel<IndexType, ValueType, 256, threads_per_block, 16, 4> <<< num_blocks2, threads_per_block>>>(A_row_offsets_ptr, A_column_indices_ptr, A_dia_values_ptr, A_nonzero_values_ptr, R_row_offsets_ptr, R_column_indices_ptr, Ac_row_offsets_ptr, Ac_column_indices_ptr, Ac_dia_values_ptr, Ac_nonzero_values_ptr, aggregates_ptr, num_aggregates, num_threads);
    }
    else if (max_nonzero_per_row < 512)
    {
        fillA_4by4Blocks_Thrust_Kernel<IndexType, ValueType, 256, threads_per_block, 16, 4> <<< num_blocks2, threads_per_block>>>(A_row_offsets_ptr, A_column_indices_ptr, A_dia_values_ptr, A_nonzero_values_ptr, R_row_offsets_ptr, R_column_indices_ptr, Ac_row_offsets_ptr, Ac_column_indices_ptr, Ac_dia_values_ptr, Ac_nonzero_values_ptr, aggregates_ptr, num_aggregates, num_threads);
    }
    else
    {
        FatalError("Maximum number of nonzeros is too large", AMGX_ERR_BAD_PARAMETERS);
    }

    cudaCheckError();
}

template <AMGX_VecPrecision t_vecPrec, AMGX_MatPrecision t_matPrec, AMGX_IndPrecision t_indPrec>
void HybridCoarseAGenerator<TemplateConfig<AMGX_host, t_vecPrec, t_matPrec, t_indPrec> >::computeAOperator_4x4(const Matrix<TemplateConfig<AMGX_host, t_vecPrec, t_matPrec, t_indPrec> > &A, Matrix<TemplateConfig<AMGX_host, t_vecPrec, t_matPrec, t_indPrec> > &Ac, const IVector &aggregates, const IVector &R_row_offsets, const IVector &R_column_indices, const int num_aggregates)
{
    FatalError("Host is unsupported for HybridCoarseAGenerator", AMGX_ERR_NOT_SUPPORTED_TARGET);
}

// ------------------------------------------------
template <class T_Config>
void HybridCoarseAGeneratorBase<T_Config>::computeAOperator(const Matrix<T_Config> &A, Matrix<T_Config> &Ac, const IVector &aggregates, const IVector &R_row_offsets, const IVector &R_column_indices, const int num_aggregates)
{
    Ac.set_initialized(0);

    if (A.get_block_dimx() == 4 && A.get_block_dimy() == 4)
    {
        computeAOperator_4x4( A, Ac, aggregates, R_row_offsets, R_column_indices, num_aggregates );
    }
    else
    {
        FatalError("Unsupported block size for HybridCoarseAGenerator", AMGX_ERR_NOT_SUPPORTED_BLOCKSIZE);
    }

    Ac.set_initialized(1);
}

// ---------------------------
// Explict instantiations
// ---------------------------
template class HybridCoarseAGeneratorBase<TConfigGeneric_d>;
template class HybridCoarseAGeneratorBase<TConfigGeneric_h>;
template class HybridCoarseAGenerator<TConfigGeneric_d>;
template class HybridCoarseAGenerator<TConfigGeneric_h>;

}
}
