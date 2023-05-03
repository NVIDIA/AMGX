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


#include <multiply.h>
#include <basic_types.h>
#include <texture.h>
#include <util.h>
#include <cutil.h>

#ifdef _WIN32
#pragma warning (push)
#pragma warning (disable : 4244 4267 4521)
#endif
#include <cusp/multiply.h>
#include <matrix.h>
#include <matrix_cusp.h>
#include <amgx_cusparse.h>
#ifdef _WIN32
#pragma warning (pop)
#endif

#include <sm_utils.inl>

#include <amgx_types/math.h>
#include <amgx_types/util.h>


namespace amgx
{

#define USE_EXPERIMENTAL_4x4

template <class Matrix, class Vector>
class Multiply_1x1;
template <class Matrix, class Vector, class IVector>
class Multiply_1x1_masked;
template <class Matrix, class Vector>
class Multiply_1x1_with_mask;
template <class Matrix, class Vector>
class Multiply_1x1_with_mask_restriction;
template <class Matrix, class Vector>
class Multiply_3x3;
template <class Matrix, class Vector>
class Multiply_4x4;
template <class Matrix, class Vector>
class Multiply_bxb;

template <typename TConfig>
void multiply_block_size(Matrix<TConfig> &A, Vector<TConfig> &B, Vector<TConfig> &C, ViewType view)
{
    typedef Matrix<TConfig> TMatrix;
    typedef Vector<TConfig> TVector;

    if (A.get_block_size() == 1)
    {
        Multiply_1x1<TMatrix, TVector>::multiply_1x1(A, B, C, view);
    }
    else if (A.get_block_dimy() == 3 && A.get_block_dimx() == 3)
    {
        Multiply_3x3<TMatrix, TVector>::multiply_3x3(A, B, C, view);
    }
    else if (A.get_block_dimy() == 4 && A.get_block_dimx() == 4)
    {
        Multiply_4x4<TMatrix, TVector>::multiply_4x4(A, B, C, view);
    }
    else
    {
        Multiply_bxb<TMatrix, TVector>::multiply_bxb(A, B, C, view);
    }
}

template <typename TConfig>
void multiply(Matrix<TConfig> &A, Vector<TConfig> &B, Vector<TConfig> &C, ViewType view)
{
    typedef Matrix<TConfig> TMatrix;
    typedef Vector<TConfig> TVector;

    if (!A.is_initialized())
    {
        FatalError("Trying to multiply uninitialized matrix", AMGX_ERR_BAD_PARAMETERS);
    }

    if (A.get_block_dimx() != B.get_block_dimy())
    {
        std::stringstream ss;
        ss << "Matrix and vector dimensions don't match: A.dimx == " << A.get_block_dimx() << ", B.dimy == " << B.get_block_dimy();
        FatalError(ss.str().c_str(), AMGX_ERR_BAD_PARAMETERS);
    }


    typedef Matrix<TConfig> TMatrix;
    typedef Vector<TConfig> TVector;

    bool latencyHiding = (view == A.getViewExterior() && A.getViewInterior() != A.getViewExterior() && !A.is_matrix_singleGPU() && B.dirtybit != 0);

    if (latencyHiding)
    {
        A.manager->exchange_halo_split_gather(B, B.tag);

        // Multiply interior rows
        multiply_block_size(A, B, C, A.getViewInterior());

        // Finish halo exchange
        A.manager->exchange_halo_split_finish(B, B.tag);

        // Multiply rows with halo dependencies
        ViewType bnd_view = (ViewType)(~(A.getViewInterior()) & A.getViewExterior());
        multiply_block_size(A, B, C, bnd_view);
    }
    else
    {
        if (view != INTERIOR && !A.is_matrix_singleGPU() && B.dirtybit != 0)
        {
            A.manager->exchange_halo_v2(B, B.tag);
        }

        multiply_block_size(A, B, C, view);
    }

    C.dirtybit = 1;
    C.set_block_dimy(A.get_block_dimx());
}

template <class TConfig>
void multiply_masked(Matrix<TConfig> &A, Vector<TConfig> &B, Vector<TConfig> &C, typename Matrix<TConfig>::IVector &mask, ViewType view)
{
    typedef Matrix<TConfig> TMatrix;
    typedef Vector<TConfig> TVector;
    typedef typename Matrix<TConfig>::IVector TIVector;

    if (!A.is_initialized())
    {
        FatalError("Trying to multiply uninitialized matrix", AMGX_ERR_BAD_PARAMETERS);
    }

    if(A.get_block_size() != 1)
    {
        FatalError("Unsupported blocksize for multiply_masked()", AMGX_ERR_BAD_PARAMETERS);
    }

    if (A.get_block_dimx() != B.get_block_dimy())
    {
        std::stringstream ss;
        ss << "Matrix and vector dimensions don't match: A.dimx == " << A.get_block_dimx() << ", B.dimy == " << B.get_block_dimy();
        FatalError(ss.str().c_str(), AMGX_ERR_BAD_PARAMETERS);
    }

    Multiply_1x1_masked<TMatrix, TVector, TIVector>::multiply_1x1_masked(A, B, C, mask, view);
    C.set_block_dimy(A.get_block_dimx());
}

template<class Matrix, class Vector>
void multiply_with_mask(Matrix &A, Vector &B, Vector &C)
{
    if (!A.is_initialized())
    {
        FatalError("Trying to multiply uninitialized matrix", AMGX_ERR_BAD_PARAMETERS);
    }

    if (A.get_block_dimx() != B.get_block_dimy())
    {
        std::stringstream ss;
        ss << "Matrix and vector dimensions don't match: A.dimx == " << A.get_block_dimx() << ", B.dimy == " << B.get_block_dimy();
        FatalError(ss.str().c_str(), AMGX_ERR_BAD_PARAMETERS);
    }

    if (A.get_block_size() == 1)
    {
        Multiply_1x1_with_mask<Matrix, Vector>::multiply_1x1(A, B, C);
    }
    else
    {
        FatalError("multiply with mask not supported for bsize != 1", AMGX_ERR_NOT_IMPLEMENTED);
    }

    C.set_block_dimy(A.get_block_dimx());
    C.dirtybit = 1;
    //if (!A.is_matrix_singleGPU() && C.size() == B.size() && C.delayed_send==0)
    //  A.manager->exchange_halo_async(C, C.tag);
}


template<class Matrix, class Vector>
void multiply_with_mask_restriction(Matrix &A, Vector &B, Vector &C, Matrix &P)
{
    if (!A.is_initialized())
    {
        FatalError("Trying to multiply uninitialized matrix", AMGX_ERR_BAD_PARAMETERS);
    }

    if (A.get_block_dimx() != B.get_block_dimy())
    {
        std::stringstream ss;
        ss << "Matrix and vector dimensions don't match: A.dimx == " << A.get_block_dimx() << ", B.dimy == " << B.get_block_dimy();
        FatalError(ss.str().c_str(), AMGX_ERR_BAD_PARAMETERS);
    }

    if (A.get_block_size() == 1)
    {
        Multiply_1x1_with_mask_restriction<Matrix, Vector>::multiply_1x1(A, B, C, P);
    }
    else
    {
        FatalError("multiply with mask not supported for bsize != 1", AMGX_ERR_NOT_IMPLEMENTED);
    }

    C.set_block_dimy(A.get_block_dimx());
    C.dirtybit = 1;
}




template<class TConfig>
void multiplyMM(const Matrix<TConfig> &A, const Matrix<TConfig> &B, Matrix<TConfig> &C)
{
    if (!A.is_initialized())
    {
        FatalError("Trying to multiply uninitialized matrix", AMGX_ERR_BAD_PARAMETERS);
    }

    if (A.get_block_dimx() != B.get_block_dimx() || A.get_block_dimy() != B.get_block_dimy())
    {
        FatalError("Matrices dimensions do not match", AMGX_ERR_BAD_PARAMETERS);
    }

    if (TConfig::memSpace == AMGX_device)
    {
        FatalError("Error, multiplyMM not implemented on device", AMGX_ERR_BAD_PARAMETERS);
    }
    else
    {
        if (A.get_block_size() != 1)
        {
            FatalError("multiplyMM only works for block_size ==1", AMGX_ERR_NOT_IMPLEMENTED);
        }
        else
        {
            typedef typename TConfig::IndPrec IndexType;
            typedef typename TConfig::MatPrec ValueType;
            typedef typename Matrix<TConfig>::IVector IVector;
            typedef typename Matrix<TConfig>::MVector MVector;
            C.set_initialized(0);
            IndexType num_nonzeros = 0;
            IVector mask(B.get_num_cols(), IndexType (-1));

            // Compute nnz in C (including explicit zeros)
            for (size_t i = 0; i < A.get_num_rows(); i++)
            {
                for (IndexType jj = A.row_offsets[i]; jj < A.row_offsets[i + 1]; jj++)
                {
                    IndexType j = A.col_indices[jj];

                    for (IndexType kk = B.row_offsets[j]; kk < B.row_offsets[j + 1]; kk++)
                    {
                        IndexType k = B.col_indices[kk];

                        if (mask[k] != i)
                        {
                            mask[k] = i;
                            num_nonzeros++;
                        }
                    }
                }
            }

            // Resize output
            C.resize(A.get_num_rows(), B.get_num_cols(), num_nonzeros);
            const IndexType unseen = static_cast<IndexType>(-1);
            const IndexType init   = static_cast<IndexType>(-2);
            // Compute entries of C
            IVector next(B.get_num_cols(), unseen);
            MVector sums(B.get_num_cols(), types::util<ValueType>::get_zero());
            num_nonzeros = 0;
            C.row_offsets[0] = 0;

            for (size_t i = 0; i < A.get_num_rows(); i++)
            {
                IndexType head   = init;
                IndexType length =    0;
                IndexType jj_start = A.row_offsets[i];
                IndexType jj_end   = A.row_offsets[i + 1];

                for (IndexType jj = jj_start; jj < jj_end; jj++)
                {
                    IndexType j = A.col_indices[jj];
                    ValueType v = A.values[jj];
                    IndexType kk_start = B.row_offsets[j];
                    IndexType kk_end   = B.row_offsets[j + 1];

                    for (IndexType kk = kk_start; kk < kk_end; kk++)
                    {
                        IndexType k = B.col_indices[kk];
                        sums[k] = sums[k] + v * B.values[kk];

                        if (next[k] == unseen)
                        {
                            next[k] = head;
                            head  = k;
                            length++;
                        }
                    }
                }

                for (IndexType jj = 0; jj < length; jj++)
                {
                    //if(sums[head] != ValueType(0))
                    //{
                    C.col_indices[num_nonzeros] = head;
                    C.values[num_nonzeros]      = sums[head];
                    num_nonzeros++;
                    //}
                    IndexType temp = head;
                    head = next[head];
                    // clear arrays
                    next[temp] = unseen;
                    sums[temp] = types::util<ValueType>::get_zero();
                }

                C.row_offsets[i + 1] = num_nonzeros;
            }

            // Resize output again since pass2 omits explict zeros
            //C.resize(A.num_rows, B.num_cols, num_nonzeros);
            C.set_initialized(1);
        }
    }
}

// --------------------------------
// KERNELS
// --------------------------------

template<typename IndexType, typename ValueTypeA, typename ValueTypeB, int eighthwarps_per_block, int bsize, int log_bsize, int half_bsize, bool ROW_MAJOR>
__global__
void blockDiaCsrMultiplyKernel(const IndexType *row_offsets,
                               const IndexType *column_indices,
                               const ValueTypeA *nonzero_values,
                               const ValueTypeB *B,
                               ValueTypeB *C,
                               const IndexType num_block_rows,
                               const IndexType row_offset)
{
    const int tid = blockDim.x * blockIdx.x + threadIdx.x;
    int eighthwarp_id = row_offset + (tid >> log_bsize);
    const int block_eighthwarp_id = threadIdx.x >> log_bsize;
    const int vec_entry_index = threadIdx.x & (bsize - 1);
    volatile __shared__ ValueTypeB s_xtemp[ bsize * eighthwarps_per_block ];
    ValueTypeB C_temp;
    int offset, s_offset;
    ValueTypeB temp[bsize];

    while (eighthwarp_id < num_block_rows)
    {
        //i = eighthwarp_id;
        C_temp = types::util<ValueTypeB>::get_zero();
        // Contribution from each nonzero column
        int jmax = row_offsets[eighthwarp_id + 1];

        for (int jind = row_offsets[eighthwarp_id]; jind < jmax; jind++)
        {
            IndexType jcol = column_indices[jind];
            offset = jcol * bsize + vec_entry_index;
            types::util<ValueTypeB>::volcast(__cachingLoad(&B[offset]), s_xtemp + threadIdx.x);
            // Load nonzero_values
            s_offset = block_eighthwarp_id * bsize;

            if (ROW_MAJOR)
            {
                offset = jind * bsize * bsize + vec_entry_index * bsize;
                loadAsVector<bsize>(nonzero_values + offset, temp);
            }
            else
            {
                offset = jind * bsize * bsize + vec_entry_index;
#pragma unroll

                for (int m = 0; m < bsize; m++)
                {
                    types::util<ValueTypeA>::to_uptype(nonzero_values[offset + bsize * m], temp[m]);
                }
            }

#pragma unroll

            for (int m = 0; m < bsize; m++)
            {
                C_temp = C_temp + temp[m] * types::util<ValueTypeB>::volcast(s_xtemp[s_offset + m]);
            }
        }

        C[eighthwarp_id * bsize + vec_entry_index] = C_temp;
        eighthwarp_id += gridDim.x * blockDim.x >> log_bsize;
    }
}

#ifdef USE_EXPERIMENTAL_4x4

template< typename IndexType, typename ValueTypeA, typename ValueTypeB, int CTA_SIZE, bool ROW_MAJOR >
__global__
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 700
__launch_bounds__( CTA_SIZE, 16 )
#elif defined(__CUDA_ARCH__)
__launch_bounds__( CTA_SIZE, 16 )
#endif
void blockDiaCsrMultiplyKernelDiaProps_4x4( const IndexType *row_offsets,
        const IndexType *column_indices,
        const IndexType *dia_ptr,
        const ValueTypeA *nonzero_values,
        const ValueTypeB *B,
        ValueTypeB *C,
        const int num_block_rows,
        const int row_offset )
{
    const int nHalfWarps = CTA_SIZE / 16; // Number of half warps per CTA.
    const int laneId = threadIdx.x % warpSize;
    const int halfWarpId = threadIdx.x / 16;
    const int halfLaneId = threadIdx.x % 16;
    const int halfLaneId_div_4 = halfLaneId / 4;
    const int halfLaneId_mod_4 = halfLaneId % 4;
    const int laneId_div_16 = laneId / 16;
    const int upperHalf = 16 * laneId_div_16;
    // Shared memory needed to exchange X and delta.
    __shared__ volatile ValueTypeB s_mem[CTA_SIZE];
    // Each thread keeps its own pointer to shared memory to avoid some extra computations.
    volatile ValueTypeB *my_s_mem = &s_mem[16 * halfWarpId];

    // Iterate over the rows of the matrix. One warp per two rows.
    for ( int aRowId = blockIdx.x * nHalfWarps + halfWarpId ; aRowId < num_block_rows ; aRowId += gridDim.x * nHalfWarps )
    {
        unsigned int active_mask = utils::activemask();
        // Load one block of B.
        ValueTypeB my_Ax = types::util<ValueTypeB>::get_zero();

        // The diagonal.
        if ( halfLaneId_div_4 == 0 )
        {
            types::util<ValueTypeB>::volcast( B[4 * aRowId + halfLaneId_mod_4], my_s_mem + halfLaneId);
        }

        // Load the diagonal.
        int diagId = dia_ptr[aRowId];
        // Update my values.
        ValueTypeA my_val = nonzero_values[16 * diagId + halfLaneId];

        if ( ROW_MAJOR )
        {
            my_Ax = my_Ax + my_val * types::util<ValueTypeB>::volcast(my_s_mem[halfLaneId_mod_4]);
        }
        else
        {
            my_Ax = my_Ax + my_val * types::util<ValueTypeB>::volcast(my_s_mem[halfLaneId_div_4]);
        }

        // The range of the rows.
        int aColBegin = row_offsets[aRowId + 0];
        int aColEnd   = row_offsets[aRowId + 1];

        // Each warp load column indices of 16 nonzero blocks
        for ( ; utils::any( aColBegin < aColEnd, active_mask ) ; aColBegin += 16 )
        {
            int aColIt = aColBegin + halfLaneId;
            // Get the ID of the column.
            int aColId = -1;

            if ( aColIt < aColEnd )
            {
                aColId = column_indices[aColIt];
            }

            // Count the number of active columns.
            int vote =  utils::ballot(aColId != -1, active_mask);
            // The number of iterations.
            int nCols = max( __popc( vote & 0x0000ffff ), __popc( vote & 0xffff0000 ) );

            // Loop over columns. We compute 8 columns per iteration.
            for ( int k = 0 ; k < nCols ; k += 4 )
            {
                int my_k = k + halfLaneId_div_4;
                // Exchange column indices.
                int waColId = utils::shfl( aColId, upperHalf + my_k, warpSize, active_mask );
                // Load 8 blocks of X if needed.
                ValueTypeB my_x = types::util<ValueTypeB>::get_zero();

                if ( waColId != -1 )
                {
                    my_x = B[4 * waColId + halfLaneId_mod_4];
                }

                types::util<ValueTypeB>::volcast( my_x, my_s_mem + halfLaneId);
                // Load 8 blocks of A.
#pragma unroll

                for ( int i = 0 ; i < 4 ; ++i )
                {
                    int w_aColTmp = aColBegin + k + i, w_aColIt = -1;

                    if ( w_aColTmp < aColEnd )
                    {
                        w_aColIt = w_aColTmp;
                    }

                    ValueTypeA my_val = types::util<ValueTypeA>::get_zero();

                    if ( w_aColIt != -1 )
                    {
                        my_val = nonzero_values[16 * w_aColIt + halfLaneId];
                    }

                    if ( ROW_MAJOR )
                    {
                        my_Ax = my_Ax + my_val * types::util<ValueTypeB>::volcast(my_s_mem[4 * i + halfLaneId_mod_4]);
                    }
                    else
                    {
                        my_Ax = my_Ax + my_val * types::util<ValueTypeB>::volcast(my_s_mem[4 * i + halfLaneId_div_4]);
                    }
                }
            } // Loop over k
        } // Loop over aColIt

        // Reduce bmAx terms.
        if ( ROW_MAJOR )
        {
            my_Ax = my_Ax + utils::shfl_xor( my_Ax, 1, warpSize, active_mask );
            my_Ax = my_Ax + utils::shfl_xor( my_Ax, 2, warpSize, active_mask );
        }
        else
        {
            my_Ax = my_Ax + utils::shfl_xor( my_Ax, 4, warpSize, active_mask );
            my_Ax = my_Ax + utils::shfl_xor( my_Ax, 8, warpSize, active_mask );
        }

        // Store the results.
        if ( ROW_MAJOR )
        {
            if ( halfLaneId_mod_4 == 0 )
            {
                C[4 * aRowId + halfLaneId_div_4] = my_Ax;
            }
        }
        else
        {
            if ( halfLaneId_div_4 == 0 )
            {
                C[4 * aRowId + halfLaneId_mod_4] = my_Ax;
            }
        }
    }
}

#else

template<typename IndexType, typename ValueTypeA, typename ValueTypeB, int eighthwarps_per_block, int bsize, int log_bsize, int half_bsize, int bsize_sq, bool ROW_MAJOR>
__global__
void blockDiaCsrMultiplyKernelDiaProps_4x4(const IndexType *row_offsets,
        const IndexType *column_indices,
        const IndexType *dia_ptr,
        const ValueTypeA *nonzero_values,
        const ValueTypeB *B,
        ValueTypeB *C,
        const int num_block_rows,
        const int row_offset)
{
    const int tid = blockDim.x * blockIdx.x + threadIdx.x;
    int eighthwarp_id = row_offset + (tid >> log_bsize);
    const int block_eighthwarp_id = threadIdx.x >> log_bsize;
    const int vec_entry_index = threadIdx.x & (bsize - 1);
    volatile __shared__ ValueTypeB s_xtemp[ bsize * eighthwarps_per_block ];
    ValueTypeB C_temp;
    int offset, s_offset;

    while (eighthwarp_id < num_block_rows)
    {
        //i = eighthwarp_id;
        C_temp = types::util<ValueTypeB>::get_zero();
        // Contribution from diagonal
        offset = eighthwarp_id * bsize + vec_entry_index;
        types::util<ValueTypeB>::volcast(__cachingLoad(&B[offset]), s_xtemp + threadIdx.x);
        // Load dia_values and do matrix multiply
        s_offset = block_eighthwarp_id * bsize;
        ValueTypeA temp[bsize];

        if (ROW_MAJOR)
        {
            loadAsVector<bsize>(nonzero_values + bsize_sq * dia_ptr[eighthwarp_id] + vec_entry_index * bsize, temp);
        }
        else
        {
#pragma unroll

            for (int m = 0; m < bsize; m++)
            {
                temp[m] = nonzero_values[bsize_sq * dia_ptr[eighthwarp_id] + vec_entry_index + bsize * m];
            }
        }

#pragma unroll

        for (int m = 0; m < bsize; m++)
        {
            C_temp = C_temp + temp[m] * types::util<ValueTypeB>::volcast(s_xtemp[s_offset + m]);
        }

        // Contribution from each nonzero column
        int jmax = row_offsets[eighthwarp_id + 1];

        for (int jind = row_offsets[eighthwarp_id]; jind < jmax; jind++)
        {
            IndexType jcol = column_indices[jind];
            offset = jcol * bsize + vec_entry_index;
            types::util<ValueTypeB>::volcast(__cachingLoad(&B[offset]), s_xtemp + threadIdx.x);
            // Load nonzero_values
            s_offset = block_eighthwarp_id * bsize;

            if (ROW_MAJOR)
            {
                offset = jind * bsize_sq + vec_entry_index * bsize;
                loadAsVector<bsize>(nonzero_values + offset, temp);
            }
            else
            {
                offset = jind * bsize_sq + vec_entry_index;
#pragma unroll

                for (int m = 0; m < bsize; m++)
                {
                    temp[m] = nonzero_values[offset + bsize * m];
                }
            }

#pragma unroll

            for (int m = 0; m < bsize; m++)
            {
                C_temp = C_temp + temp[m] * types::util<ValueTypeB>::volcast(s_xtemp[s_offset + m]);
            }
        }

        C[eighthwarp_id * bsize + vec_entry_index] = C_temp;
        eighthwarp_id += gridDim.x * blockDim.x >> log_bsize;
    }
}

#endif

// implementation for arbitrary block size
template<typename IndexType, typename ValueTypeA, typename ValueTypeB, int blockrows_per_cta, int blockrows_per_warp, int bsize, int diag, bool ROW_MAJOR>
__global__
void blockDiaCsrMultiplyKernelDiaProps(const IndexType *row_offsets,
                                       const IndexType *column_indices,
                                       const IndexType *dia_ptr,
                                       const ValueTypeA *nonzero_values,
                                       const ValueTypeB *B,
                                       ValueTypeB *C,
                                       const int num_block_rows,
                                       const int row_offset)
{
    int warp_id = threadIdx.x / 32;
    int warp_thread_id = threadIdx.x & 31;

    // padding row blocks to fit in a single warp
    if ( warp_thread_id >= blockrows_per_warp * bsize ) { return; }

    // new thread id with padding
    int tid = warp_id * blockrows_per_warp * bsize + warp_thread_id;
    int eighthwarp_id = row_offset + blockIdx.x * blockrows_per_cta + tid / bsize;
    const int block_eighthwarp_id = tid / bsize;
    const int vec_entry_index = tid % bsize;
    const int bsize_sq = bsize * bsize;
    volatile __shared__ ValueTypeB s_xtemp[ bsize * blockrows_per_cta ];
    ValueTypeB C_temp;
    int offset, s_offset;

    while (eighthwarp_id < num_block_rows)
    {
        C_temp = types::util<ValueTypeB>::get_zero();
        ValueTypeB temp[bsize];

        if ( diag )
        {
            // Contribution from diagonal
            offset = eighthwarp_id * bsize + vec_entry_index;
            types::util<ValueTypeB>::volcast( __cachingLoad(&B[offset]), s_xtemp + tid);
            // Load dia_values and do matrix multiply
            s_offset = block_eighthwarp_id * bsize;

            if (ROW_MAJOR)
            {
                loadAsVector<bsize>(nonzero_values + bsize_sq * dia_ptr[eighthwarp_id] + vec_entry_index * bsize, temp);
            }
            else
            {
                offset = dia_ptr[eighthwarp_id] * bsize_sq + vec_entry_index;
#pragma unroll

                for (int m = 0; m < bsize; m++)
                {
                    types::util<ValueTypeA>::to_uptype(nonzero_values[offset + bsize * m], temp[m]);
                }
            }

#pragma unroll

            for (int m = 0; m < bsize; m++)
            {
                C_temp = C_temp + temp[m] * types::util<ValueTypeB>::volcast(s_xtemp[s_offset + m]);
            }
        }

        // Contribution from each nonzero column
        int jmax = row_offsets[eighthwarp_id + 1];

        for (int jind = row_offsets[eighthwarp_id]; jind < jmax; jind++)
        {
            IndexType jcol = column_indices[jind];
            offset = jcol * bsize + vec_entry_index;
            types::util<ValueTypeB>::volcast( __cachingLoad(&B[offset]), s_xtemp + tid);
            // Load nonzero_values
            s_offset = block_eighthwarp_id * bsize;

            if (ROW_MAJOR)
            {
                offset = jind * bsize_sq + vec_entry_index * bsize;
                loadAsVector<bsize>(nonzero_values + offset, temp);
            }
            else
            {
                offset = jind * bsize_sq + vec_entry_index;
#pragma unroll

                for (int m = 0; m < bsize; m++)
                {
                    types::util<ValueTypeA>::to_uptype(nonzero_values[offset + bsize * m], temp[m]);
                }
            }

#pragma unroll

            for (int m = 0; m < bsize; m++)
            {
                C_temp = C_temp + temp[m] * types::util<ValueTypeB>::volcast(s_xtemp[s_offset + m]);
            }
        }

        C[eighthwarp_id * bsize + vec_entry_index] = C_temp;
        eighthwarp_id += gridDim.x * blockrows_per_cta;
    }
}

// --------------------------------------
//  Methods
// -------------------------------------

// Method to perform BSPmV on host using block_dia_csr_matrix format
template <class Matrix, class Vector>
void multiply_common_sqblock_host_diag(const Matrix &A, const Vector &B, Vector &C)
{
    typedef typename Matrix::TConfig TConfig;

    if (TConfig::memSpace == AMGX_device)
    {
        FatalError("Executrion path error: device matrix in host path", AMGX_ERR_NOT_IMPLEMENTED);
    }
    else
    {
        //TODO:: This implementation is very inneficient, Use BLAS
        typedef typename TConfig::IndPrec IndexType;
        typedef typename TConfig::MatPrec ValueType;
        typedef typename Vector::value_type ValueTypeB;
        IndexType bsize = A.get_block_dimy();
        ValueTypeB temp;

        for (int i = 0; i < A.get_num_rows(); i++)
        {
            // Initialize RHS to 0
            for (int m = 0; m < bsize; m++)
            {
                C[i * bsize + m] = types::util<ValueTypeB>::get_zero();
            }

            // Contribution from diagonal blocks
            for (int n = 0; n < bsize; n++)
            {
                temp = B[i * bsize + n];

                for (int m = 0; m < bsize; m++)
                {
                    C[i * bsize + m] = C[i * bsize + m] + A.values[A.diag[i] * bsize * bsize + m * bsize + n] * temp;
                }
            }

            // Contribution from nonzero off-diagonal blocks
            for (int j = A.row_offsets[i]; j < A.row_offsets[i + 1]; j++)
            {
                IndexType jcol = A.col_indices[j];

                for (int n = 0; n < bsize; n++)
                {
                    temp = B[jcol * bsize + n];

                    for (int m = 0; m < bsize; m++)
                    {
                        C[i * bsize + m] = C[i * bsize + m] + A.values[j * bsize * bsize + m * bsize + n] * temp;
                    }
                }
            }
        }
    }
}

template <class Matrix, class Vector>
void multiply_common_sqblock_host_nodiag(const Matrix &A, const Vector &B, Vector &C)
{
    typedef typename Matrix::TConfig TConfig;

    if (TConfig::memSpace == AMGX_device)
    {
        FatalError("Executrion path error: device matrix in host path", AMGX_ERR_NOT_IMPLEMENTED);
    }
    else
    {
        //TODO:: This implementation is very inneficient, Use BLAS
        typedef typename TConfig::IndPrec IndexType;
        typedef typename TConfig::MatPrec ValueType;
        typedef typename Vector::value_type ValueTypeB;
        IndexType bsize = A.get_block_dimy();
        ValueTypeB temp;

        for (int i = 0; i < A.get_num_rows(); i++)
        {
            // Initialize RHS to 0
            for (int m = 0; m < bsize; m++)
            {
                C[i * bsize + m] = types::util<ValueTypeB>::get_zero();
            }

            // Contribution from nonzero blocks
            for (int j = A.row_offsets[i]; j < A.row_offsets[i + 1]; j++)
            {
                IndexType jcol = A.col_indices[j];

                for (int n = 0; n < bsize; n++)
                {
                    temp = B[jcol * bsize + n];

                    for (int m = 0; m < bsize; m++)
                    {
                        C[i * bsize + m] = C[i * bsize + m] + A.values[j * bsize * bsize + m * bsize + n] * temp;
                    }
                }
            }
        }
    }
}


template <class Matrix, class Vector>
class Multiply_1x1
{
    public:
        typedef typename Matrix::TConfig TConfig;
        static void multiply_1x1(Matrix &A, Vector &B, Vector &C, ViewType view)
        {
            if (TConfig::memSpace == AMGX_host)
            {
                if (A.hasProps(DIAG))
                {
                    multiply_common_sqblock_host_diag(A, B, C);
                }
                else
                {
                    multiply_common_sqblock_host_nodiag(A, B, C);
                }
            }
            else
            {
                typedef typename TConfig::VecPrec ValueTypeB;
                Cusparse::bsrmv<TConfig>(types::util<ValueTypeB>::get_one(), A, B, types::util<ValueTypeB>::get_zero(), C, view);
                cudaCheckError();
                //  FatalError("Mixed precision is not supported for scalar matrix type", AMGX_ERR_NOT_IMPLEMENTED);
            }
        }
};

template <class Matrix, class Vector, class IVector>
class Multiply_1x1_masked
{
    public:
        typedef typename Matrix::TConfig TConfig;
        static void multiply_1x1_masked(Matrix &A, Vector &B, Vector &C, IVector mask, ViewType view)
        {
            if (TConfig::memSpace == AMGX_host)
            {
                FatalError("Masked multiply is not supported on host", AMGX_ERR_NOT_IMPLEMENTED);
            }
            else
            {
                typedef typename TConfig::VecPrec ValueTypeB;
                Cusparse::bsrxmv<TConfig>(types::util<ValueTypeB>::get_one(), A, B, types::util<ValueTypeB>::get_zero(), C, mask, view);
                cudaCheckError();
            }
        }
};

template <class Matrix, class Vector>
class Multiply_1x1_with_mask
{
    public:
        typedef typename Matrix::TConfig TConfig;
        static void multiply_1x1(Matrix &A, Vector &B, Vector &C)
        {
            if (TConfig::memSpace == AMGX_host)
            {
                FatalError("multiply with mask not supported on host", AMGX_ERR_NOT_IMPLEMENTED);
            }
            else
            {
                typedef typename TConfig::VecPrec ValueTypeB;
                Cusparse::bsrmv_with_mask<TConfig>(types::util<ValueTypeB>::get_one(), A, B, types::util<ValueTypeB>::get_zero(), C );
                cudaCheckError();
                //  FatalError("Mixed precision is not supported for scalar matrix type", AMGX_ERR_NOT_IMPLEMENTED);
            }
        }
};

template <class Matrix, class Vector>
class Multiply_1x1_with_mask_restriction
{
    public:
        typedef typename Matrix::TConfig TConfig;
        static void multiply_1x1(Matrix &A, Vector &B, Vector &C, Matrix &P)
        {
            if (TConfig::memSpace == AMGX_host)
            {
                FatalError("multiply with mask not supported on host", AMGX_ERR_NOT_IMPLEMENTED);
            }
            else
            {
                typedef typename TConfig::VecPrec ValueTypeB;
                Cusparse::bsrmv_with_mask_restriction<TConfig>(types::util<ValueTypeB>::get_one(), A, B, types::util<ValueTypeB>::get_zero(), C, P);
                cudaCheckError();
                //  FatalError("Mixed precision is not supported for scalar matrix type", AMGX_ERR_NOT_IMPLEMENTED);
            }
        }
};


// Method to perform BSPmV on device using block_dia_csr_matrix format
template <class Matrix, class Vector>
class Multiply_4x4
{
    public:
        typedef typename Matrix::TConfig TConfig;
        static void multiply_4x4(const Matrix &A, const Vector &B, Vector &C, ViewType view)
        {
            if (TConfig::memSpace == AMGX_host)
            {
                if (A.hasProps(DIAG))
                {
                    multiply_common_sqblock_host_diag(A, B, C);
                }
                else
                {
                    multiply_common_sqblock_host_nodiag(A, B, C);
                }
            }
            else
            {
                //TODO: compare with cublas
                typedef typename TConfig::IndPrec IndexType;
                typedef typename TConfig::MatPrec ValueTypeA;
                typedef typename TConfig::VecPrec ValueTypeB;
                int num_rows, offset;
                A.getOffsetAndSizeForView(view, &offset, &num_rows);
                const IndexType *A_row_offsets_ptr = A.row_offsets.raw();
                const IndexType *A_column_indices_ptr = A.col_indices.raw();
                const IndexType *A_dia_ind_ptr = A.diag.raw();
                const ValueTypeA *A_nonzero_values_ptr = A.values.raw();
                const ValueTypeB *B_ptr = B.raw();
                ValueTypeB *C_ptr = C.raw();
                cudaCheckError();
                const unsigned int threads_per_block = 128;
                const int eightwarps_per_block = threads_per_block / 4;
                const int num_warps_per_cta = threads_per_block / 32;
                const int num_blocks =  std::min(AMGX_GRID_MAX_SIZE, (int) (num_rows + num_warps_per_cta - 1) / num_warps_per_cta); // (int) (A.get_num_rows()-1)/eightwarps_per_block + 1;

                if (!A.hasProps(DIAG))
                {
                    if (A.getBlockFormat() == ROW_MAJOR)
                    {
                        cudaFuncSetCacheConfig(blockDiaCsrMultiplyKernel<IndexType, ValueTypeA, ValueTypeB, eightwarps_per_block, 4, 2, 2, true>, cudaFuncCachePreferL1);
                        blockDiaCsrMultiplyKernel<IndexType, ValueTypeA, ValueTypeB, eightwarps_per_block, 4, 2, 2, true> <<< num_blocks, threads_per_block>>>(A_row_offsets_ptr, A_column_indices_ptr, A_nonzero_values_ptr, B_ptr, C_ptr, offset + num_rows, offset);
                    }
                    else
                    {
                        cudaFuncSetCacheConfig(blockDiaCsrMultiplyKernel<IndexType, ValueTypeA, ValueTypeB, eightwarps_per_block, 4, 2, 2, false>, cudaFuncCachePreferL1);
                        blockDiaCsrMultiplyKernel<IndexType, ValueTypeA, ValueTypeB, eightwarps_per_block, 4, 2, 2, false> <<< num_blocks, threads_per_block>>>(A_row_offsets_ptr, A_column_indices_ptr, A_nonzero_values_ptr, B_ptr, C_ptr, offset + num_rows, offset);
                    }

                }
                else
                {

#ifdef USE_EXPERIMENTAL_4x4
                    if ( A.getBlockFormat() == ROW_MAJOR )
                    {
                        blockDiaCsrMultiplyKernelDiaProps_4x4<IndexType, ValueTypeA, ValueTypeB, threads_per_block, true > <<< num_blocks, threads_per_block>>>(A_row_offsets_ptr, A_column_indices_ptr, A_dia_ind_ptr, A_nonzero_values_ptr, B_ptr, C_ptr, offset + num_rows, offset);
                    }
                    else
                    {
                        blockDiaCsrMultiplyKernelDiaProps_4x4<IndexType, ValueTypeA, ValueTypeB, threads_per_block, false > <<< num_blocks, threads_per_block>>>(A_row_offsets_ptr, A_column_indices_ptr, A_dia_ind_ptr, A_nonzero_values_ptr, B_ptr, C_ptr, offset + num_rows, offset);
                    }
#else

                    if (A.getBlockFormat() == ROW_MAJOR)
                    {
                        cudaFuncSetCacheConfig(blockDiaCsrMultiplyKernelDiaProps_4x4<IndexType, ValueTypeA, ValueTypeB, eightwarps_per_block, 4, 2, 2, 16, true>, cudaFuncCachePreferL1);
                        blockDiaCsrMultiplyKernelDiaProps_4x4<IndexType, ValueTypeA, ValueTypeB, eightwarps_per_block, 4, 2, 2, 16, true> <<< num_blocks, threads_per_block>>>(A_row_offsets_ptr, A_column_indices_ptr, A_dia_ind_ptr, A_nonzero_values_ptr, B_ptr, C_ptr, offset + num_rows, offset);
                    }
                    else
                    {
                        cudaFuncSetCacheConfig(blockDiaCsrMultiplyKernelDiaProps_4x4<IndexType, ValueTypeA, ValueTypeB, eightwarps_per_block, 4, 2, 2, 16, false>, cudaFuncCachePreferL1);
                        blockDiaCsrMultiplyKernelDiaProps_4x4<IndexType, ValueTypeA, ValueTypeB, eightwarps_per_block, 4, 2, 2, 16, false> <<< num_blocks, threads_per_block>>>(A_row_offsets_ptr, A_column_indices_ptr, A_dia_ind_ptr, A_nonzero_values_ptr, B_ptr, C_ptr, offset + num_rows, offset);
                    }

#endif
                }
                cudaCheckError();
            }
        }
};

// Method to perform BSPmV on device using block_dia_csr_matrix format
template <class Matrix, class Vector>
class Multiply_bxb
{
    public:
        typedef typename Matrix::TConfig TConfig;
        static void multiply_bxb(Matrix &A, Vector &B, Vector &C, ViewType view)
        {
            if (TConfig::memSpace == AMGX_host)
            {
                if (A.hasProps(DIAG))
                {
                    multiply_common_sqblock_host_diag(A, B, C);
                }
                else
                {
                    multiply_common_sqblock_host_nodiag(A, B, C);
                }
            }
            else
            {
                typedef typename TConfig::VecPrec ValueTypeB;
                Cusparse::bsrmv(types::util<ValueTypeB>::get_one(), A, B, types::util<ValueTypeB>::get_zero(), C, view);
                cudaCheckError();
            }
        }
};


// Method to perform BSPmV on device using block_dia_csr_matrix format
template <class Matrix, class Vector>
class Multiply_3x3
{
    public:
        typedef typename Matrix::TConfig TConfig;
        static void multiply_3x3(const Matrix &A, const Vector &B, Vector &C, ViewType view)
        {
            if (TConfig::memSpace == AMGX_host)
            {
                if (A.hasProps(DIAG))
                {
                    multiply_common_sqblock_host_diag(A, B, C);
                }
                else
                {
                    multiply_common_sqblock_host_nodiag(A, B, C);
                }
            }
            else
            {
                //TODO: compare with cublas
                typedef typename TConfig::IndPrec IndexType;
                typedef typename TConfig::MatPrec ValueTypeA;
                typedef typename TConfig::VecPrec ValueTypeB;
                int num_rows, offset;
                A.getOffsetAndSizeForView(view, &offset, &num_rows);
                const IndexType *A_row_offsets_ptr = A.row_offsets.raw();
                const IndexType *A_column_indices_ptr = A.col_indices.raw();
                const IndexType *A_dia_ind_ptr = A.diag.raw();
                const ValueTypeA *A_nonzero_values_ptr = A.values.raw();
                const ValueTypeB *B_ptr = B.raw();
                ValueTypeB *C_ptr = C.raw();
                cudaCheckError();
                const int threads_per_block = 64 * 3;
                const int blockrows_per_warp = 32 / 3;
                const int blockrows_per_cta = (threads_per_block / 32) * blockrows_per_warp;
                const int num_blocks =  std::min(AMGX_GRID_MAX_SIZE, (int) (num_rows - 1) / blockrows_per_cta + 1); // (int) (A.get_num_rows()-1)/eightwarps_per_block + 1;

                if (!A.hasProps(DIAG))
                {
                    if (A.getBlockFormat() == ROW_MAJOR)
                    {
                        cudaFuncSetCacheConfig(blockDiaCsrMultiplyKernelDiaProps<IndexType, ValueTypeA, ValueTypeB, blockrows_per_cta, blockrows_per_warp, 3, 0, true>, cudaFuncCachePreferL1);
                        blockDiaCsrMultiplyKernelDiaProps<IndexType, ValueTypeA, ValueTypeB, blockrows_per_cta, blockrows_per_warp, 3, 0, true> <<< num_blocks, threads_per_block>>>(A_row_offsets_ptr, A_column_indices_ptr, A_dia_ind_ptr, A_nonzero_values_ptr, B_ptr, C_ptr, offset + num_rows, offset);
                    }
                    else
                    {
                        cudaFuncSetCacheConfig(blockDiaCsrMultiplyKernelDiaProps<IndexType, ValueTypeA, ValueTypeB, blockrows_per_cta, blockrows_per_warp, 3, 0, false>, cudaFuncCachePreferL1);
                        blockDiaCsrMultiplyKernelDiaProps<IndexType, ValueTypeA, ValueTypeB, blockrows_per_cta, blockrows_per_warp, 3, 0, false> <<< num_blocks, threads_per_block>>>(A_row_offsets_ptr, A_column_indices_ptr, A_dia_ind_ptr, A_nonzero_values_ptr, B_ptr, C_ptr, offset + num_rows, offset);
                    }
                }
                else
                {
                    if (A.getBlockFormat() == ROW_MAJOR)
                    {
                        cudaFuncSetCacheConfig(blockDiaCsrMultiplyKernelDiaProps<IndexType, ValueTypeA, ValueTypeB, blockrows_per_cta, blockrows_per_warp, 3, 1, true>, cudaFuncCachePreferL1);
                        blockDiaCsrMultiplyKernelDiaProps<IndexType, ValueTypeA, ValueTypeB, blockrows_per_cta, blockrows_per_warp, 3, 1, true> <<< num_blocks, threads_per_block>>>(A_row_offsets_ptr, A_column_indices_ptr, A_dia_ind_ptr, A_nonzero_values_ptr, B_ptr, C_ptr, offset + num_rows, offset);
                    }
                    else
                    {
                        cudaFuncSetCacheConfig(blockDiaCsrMultiplyKernelDiaProps<IndexType, ValueTypeA, ValueTypeB, blockrows_per_cta, blockrows_per_warp, 3, 1, false>, cudaFuncCachePreferL1);
                        blockDiaCsrMultiplyKernelDiaProps<IndexType, ValueTypeA, ValueTypeB, blockrows_per_cta, blockrows_per_warp, 3, 1, false> <<< num_blocks, threads_per_block>>>(A_row_offsets_ptr, A_column_indices_ptr, A_dia_ind_ptr, A_nonzero_values_ptr, B_ptr, C_ptr, offset + num_rows, offset);
                    }
                }

                cudaCheckError();
            }
        }
};

// -------------------------------
// Explict instantiations
// -------------------------------
#define AMGX_CASE_LINE(CASE) template void multiplyMM(const Matrix<TemplateMode<CASE>::Type>&, const Matrix<TemplateMode<CASE>::Type>&, Matrix<TemplateMode<CASE>::Type>&);
AMGX_FORALL_BUILDS(AMGX_CASE_LINE)
AMGX_FORCOMPLEX_BUILDS(AMGX_CASE_LINE)
#undef AMGX_CASE_LINE

#define AMGX_CASE_LINE(CASE) template void multiply(Matrix<TemplateMode<CASE>::Type> &, Vector<TemplateMode<CASE>::Type>&, Vector<TemplateMode<CASE>::Type> &, ViewType);
AMGX_FORALL_BUILDS(AMGX_CASE_LINE)
AMGX_FORCOMPLEX_BUILDS(AMGX_CASE_LINE)
#undef AMGX_CASE_LINE

#define AMGX_CASE_LINE(CASE) template void multiply_masked(Matrix<TemplateMode<CASE>::Type> &, Vector<TemplateMode<CASE>::Type> &, Vector<TemplateMode<CASE>::Type> &, typename Matrix<TemplateMode<CASE>::Type>::IVector &, ViewType);
AMGX_FORALL_BUILDS(AMGX_CASE_LINE)
AMGX_FORCOMPLEX_BUILDS(AMGX_CASE_LINE)
#undef AMGX_CASE_LINE

#define AMGX_CASE_LINE(CASE) template void multiply_with_mask(Matrix<TemplateMode<CASE>::Type> &, Vector<TemplateMode<CASE>::Type>&, Vector<TemplateMode<CASE>::Type>  &);
AMGX_FORALL_BUILDS(AMGX_CASE_LINE)
AMGX_FORCOMPLEX_BUILDS(AMGX_CASE_LINE)
#undef AMGX_CASE_LINE

#define AMGX_CASE_LINE(CASE) template void multiply_with_mask_restriction(Matrix<TemplateMode<CASE>::Type> &, Vector<TemplateMode<CASE>::Type>&, Vector<TemplateMode<CASE>::Type>  &, Matrix<TemplateMode<CASE>::Type> & );
AMGX_FORALL_BUILDS(AMGX_CASE_LINE)
AMGX_FORCOMPLEX_BUILDS(AMGX_CASE_LINE)
#undef AMGX_CASE_LINE

} // namespace amgx
