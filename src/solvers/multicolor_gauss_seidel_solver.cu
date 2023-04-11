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

#include <string.h>
#include <cutil.h>
#include <miscmath.h>
#include <amgx_cusparse.h>
#include <thrust/copy.h>
#include <solvers/multicolor_gauss_seidel_solver.h>
#include <solvers/block_common_solver.h>
#include <gaussian_elimination.h>
#include <basic_types.h>
#include <util.h>
#include <texture.h>
#include <device_properties.h>
#include <stream.h>

#include "sm_utils.inl"

namespace amgx
{
namespace multicolor_gauss_seidel_solver
{

// -------------------------
//  Kernels
// -------------------------

template<typename IndexType, typename ValueTypeA>
__global__
void setupBlockGSSmooth1x1(const IndexType *row_offsets, const IndexType *column_indices, const ValueTypeA *values, const IndexType *dia_indices, ValueTypeA *Dinv, const int num_rows)
{
    int tid = blockDim.x * blockIdx.x + threadIdx.x;

    while (tid < num_rows)
    {
        //epsilon (val) is a small number with the same sign as val
        ValueTypeA val =  values[dia_indices[tid]];
        Dinv[tid] =  isNotCloseToZero(val) ? ValueTypeA(1) / val :  ValueTypeA(1) / epsilon(val);
        tid += gridDim.x * blockDim.x;
    }
}


template<typename IndexType, typename ValueTypeA, int blockrows_per_cta, int blockrows_per_warp, int bsize, int bsize_sq>
__global__
void setupBlockGSSmooth3by3BlockDiaCsrKernel(const IndexType *row_offsets, const IndexType *column_indices, const ValueTypeA *values, const IndexType *dia_indices, ValueTypeA *Dinv, const int num_block_rows)

{
    int warp_id = threadIdx.x / 32;
    int warp_thread_id = threadIdx.x & 31;

    // padding row blocks to fit in a single warp
    if ( warp_thread_id >= blockrows_per_warp * bsize_sq ) { return; }

    // new thread id with padding
    int tid = warp_id * blockrows_per_warp * bsize_sq + warp_thread_id;
    int cta_blockrow_id = tid / bsize_sq;
    int blockrow_id = blockIdx.x * blockrows_per_cta + cta_blockrow_id;
    const int mat_entry_index = tid - cta_blockrow_id * bsize_sq;
    const int i_ind = mat_entry_index / bsize;
    const int j_ind = mat_entry_index - i_ind * bsize;
    __shared__ ValueTypeA s_Amat[bsize_sq * blockrows_per_cta ];
    ValueTypeA e_out;

    while (blockrow_id < num_block_rows)
    {
        int offset = blockrow_id * bsize_sq + mat_entry_index;
        // Store the diagonal
        e_out = values[bsize_sq * dia_indices[blockrow_id] + mat_entry_index];
        // Each thread stores its entry in s_Amat
        s_Amat[tid] = e_out;
        compute_block_inverse_row_major<IndexType, ValueTypeA, blockrows_per_cta, bsize, bsize_sq>
        ( s_Amat, cta_blockrow_id * bsize_sq, offset, i_ind, j_ind, Dinv );
        blockrow_id += gridDim.x * blockrows_per_cta;
    }
}

template<typename IndexType, typename ValueTypeA, int threads_per_block, int halfwarps_per_block, int bsize, int log_bsize, int bsize_sq, int log_bsize_sq>
__global__
void setupBlockGSSmooth4by4BlockDiaCsrKernel_V2(const IndexType *row_offsets, const IndexType *column_indices, const ValueTypeA *values, const IndexType *dia_indices, ValueTypeA *Dinv, const int num_block_rows)

{
    const int tid = blockDim.x * blockIdx.x + threadIdx.x;
    int halfwarp_id = tid >> log_bsize_sq;
    const int block_halfwarp_id = threadIdx.x >> log_bsize_sq;
    const int mat_entry_index = threadIdx.x & (bsize_sq - 1);
    const int i_ind = mat_entry_index >> log_bsize;
    const int j_ind = mat_entry_index & (bsize - 1);
    __shared__ ValueTypeA s_Amat[bsize_sq * halfwarps_per_block ];
    ValueTypeA e_out;

    while (halfwarp_id < num_block_rows)
    {
        int offset = halfwarp_id * bsize_sq + mat_entry_index;
        // Store the diagonal
        e_out = values[bsize_sq * dia_indices[halfwarp_id] + mat_entry_index];
        // Each thread stores its entry in s_Amat
        s_Amat[threadIdx.x] = e_out;
        compute_block_inverse_row_major<IndexType, ValueTypeA, halfwarps_per_block, bsize, bsize_sq>
        ( s_Amat, block_halfwarp_id * bsize_sq, offset, i_ind, j_ind, Dinv );
        halfwarp_id += gridDim.x * blockDim.x >> log_bsize_sq;
    }
}

template<typename IndexType, typename ValueTypeA, int threads_per_block, int halfwarps_per_block>
__global__
void setupBlockGSSmoothbBybBlockDiaCsrKernel(const IndexType *row_offsets, const IndexType *column_indices, const ValueTypeA *values, const IndexType *dia_indices, ValueTypeA *Dinv, const int num_block_rows, int bsize, int bsize_sq, ValueTypeA *temp1)

{
    const int tid = blockDim.x * blockIdx.x + threadIdx.x;
    int halfwarp_id = tid >> 4;
    const int block_halfwarp_id = threadIdx.x >> 4;
    const int mat_entry_index = threadIdx.x & (16 - 1);
    const int i_ind = mat_entry_index >> 2;
    const int j_ind = mat_entry_index & 3;
    extern __shared__ char sharedc[];
    ValueTypeA *s_Amat;
    s_Amat = (ValueTypeA *)&sharedc[0];
    int tile_num = (bsize - 1) / 4 + 1;
    ValueTypeA *e_out = &temp1[(blockIdx.x * blockDim.x + threadIdx.x) * tile_num * tile_num];

    while (halfwarp_id < num_block_rows)
    {
        int offset = halfwarp_id * bsize_sq + i_ind * bsize + j_ind;
        int s_offset = block_halfwarp_id * bsize_sq;

        // Store the diagonal
        for (int t1 = 0; t1 < tile_num; t1++)
            for (int t2 = 0; t2 < tile_num; t2++)
                if ((t1 * 4 + i_ind) < bsize && (t2 * 4 + j_ind) < bsize)
                {
                    e_out[t1 * tile_num + t2] = values[bsize_sq * dia_indices[halfwarp_id] + (t1 * 4 + i_ind) * bsize + t2 * 4 + j_ind];
                }

        // Each thread stores its entry in s_Amat
        for (int t1 = 0; t1 < tile_num; t1++)
            for (int t2 = 0; t2 < tile_num; t2++)
                if ((t1 * 4 + i_ind) < bsize && (t2 * 4 + j_ind) < bsize)
                {
                    s_Amat[s_offset + (t1 * 4 + i_ind) * bsize + t2 * 4 + j_ind] = e_out[t1 * tile_num + t2];
                }


        compute_block_inverse2<IndexType, ValueTypeA, halfwarps_per_block>
        ( s_Amat, s_offset, offset, i_ind, j_ind, Dinv, tile_num, bsize, bsize_sq );
        halfwarp_id += gridDim.x * halfwarps_per_block;
    }
}

template<typename IndexType, typename ValueTypeA,  int blockrows_per_cta, int bsize, int bsize_sq>
__global__
void setupBlockGSSmoothBlockDiaCsrKernel_V2(const IndexType *row_offsets, const IndexType *column_indices, const ValueTypeA *values, const IndexType *dia_indices,
        ValueTypeA *Dinv, const int num_block_rows)
{
    int cta_blockrow_id = threadIdx.x / bsize_sq;
    int blockrow_id = blockIdx.x * blockrows_per_cta + cta_blockrow_id;
    const int mat_entry_index = threadIdx.x - cta_blockrow_id * bsize_sq;
    const int i_ind = mat_entry_index / bsize;
    const int j_ind = mat_entry_index - i_ind * bsize;
    __shared__ ValueTypeA s_Amat[bsize_sq * blockrows_per_cta];
    int offset, s_offset;
    ValueTypeA e_out;

    while (blockrow_id < num_block_rows && cta_blockrow_id < blockrows_per_cta)
    {
        // Store the diagonal
        offset = blockrow_id * bsize_sq + mat_entry_index;
        e_out = values[bsize_sq * dia_indices[blockrow_id] + mat_entry_index];
        // Each thread stores its entry in s_Amat
        s_Amat[threadIdx.x] = e_out;

        s_offset = cta_blockrow_id * bsize_sq;
// THIS IS DEFINED IN BLOCK COMMON
//#define s_A(ROW,COL)   s_Amat[s_offset+ROW*bsize+COL]
        ValueTypeA diag;

        for (int row = 0; row < bsize; row++)
        {
            diag =   isNotCloseToZero(s_A(row, row)) ? ValueTypeA(1) / s_A(row, row) : ValueTypeA(1) / epsilon(s_A(row, row));
            //This condition is a subset of the condition below for i_ind==0
            //if ((i_ind == 0) && !(j_ind == row))
            //  s_A(row,j_ind) = s_A(row,j_ind)*diag;

            if ((i_ind != row) && !(j_ind == row))
            {
                s_A(i_ind, j_ind) = -(s_A(i_ind, row) * s_A(row, j_ind)) + s_A(i_ind, j_ind);
            }

            if (i_ind == 0)
            {
                s_A(j_ind, row) = (j_ind == row) ? diag : -(s_A(j_ind, row) * diag);
            }
        }

        Dinv[offset] = s_A(i_ind, j_ind);
        blockrow_id += (gridDim.x * blockDim.x) / bsize_sq;
    }
}

// Kernel to smooth, NAIVE implementation with texture
template<typename IndexType, typename ValueTypeA, typename ValueTypeB, int eighthwarps_per_block, int bsize, int log_bsize, int half_bsize>
__global__
void multicolorGSSmooth4by4BlockDiaCsrKernel_NAIVE_tex_readDinv2(const IndexType *row_offsets, const IndexType *column_indices, const IndexType *dia_indices, const ValueTypeA *nonzero_values, const ValueTypeA *Dinv,
        const ValueTypeB *b, const ValueTypeB *x, ValueTypeB weight, const int *sorted_rows_by_color, const int num_rows_per_color, const int num_block_rows, ValueTypeB *xout)
{
    const int vec_entry_index = threadIdx.x & (bsize - 1);
    const int block_eighthwarp_id = threadIdx.x >> log_bsize;
    const int tid = blockDim.x * blockIdx.x + threadIdx.x;
    int eighthwarp_id = tid >> log_bsize;
    volatile __shared__ ValueTypeB s_xtemp[ bsize * eighthwarps_per_block ];
    ValueTypeB bmAx, xin;
    ValueTypeB temp[bsize];
    int offset, i, s_offset;

    while (eighthwarp_id < num_rows_per_color)
    {
        i = sorted_rows_by_color[eighthwarp_id];
        offset = i * bsize + vec_entry_index;
        bmAx = b[offset];
        // Contribution from diagonal
        xin = x[offset];
        s_xtemp[threadIdx.x] = xin;
        // Load dia_values and do matrix multiply
        s_offset = block_eighthwarp_id * bsize;
        loadAsVector<bsize>(nonzero_values + dia_indices[i]*bsize * bsize + vec_entry_index * bsize, temp);
#pragma unroll

        for (int m = 0; m < bsize; m++)
        {
            bmAx -= temp[m] * s_xtemp[s_offset + m];
        }

        // Contribution from each nonzero column
        int jmax = row_offsets[i + 1];

        for (int jind = row_offsets[i]; jind < jmax; jind++)
        {
            IndexType jcol = column_indices[jind];

            if (jcol != i)
            {
                offset = jcol * bsize + vec_entry_index;
                s_xtemp[threadIdx.x] = x[offset];
                // Load nonzero_values
                offset = jind * bsize * bsize + vec_entry_index * bsize;
                loadAsVector<bsize>(nonzero_values + offset, temp);
                s_offset = block_eighthwarp_id * bsize;
#pragma unroll

                for (int m = 0; m < bsize; m++)
                {
                    bmAx -= temp[m] * s_xtemp[s_offset + m];
                }
            }
        }

        s_xtemp[threadIdx.x] = bmAx;
        bmAx = 0.;
        // Load Dinv and multiply to RHS
        offset = i * bsize * bsize + vec_entry_index * bsize;
        loadAsVector<bsize>(Dinv + offset, temp);
        s_offset = block_eighthwarp_id * bsize;
#pragma unroll

        for (int m = 0; m < bsize; m++)
        {
            bmAx += temp[m] * s_xtemp[s_offset + m];
        }

        xout[i * bsize + vec_entry_index] = xin + weight * bmAx;
        eighthwarp_id +=  (gridDim.x * blockDim.x >> log_bsize);
    }
}

// Kernel to smooth, NAIVE implementation with texture
template<typename IndexType, typename ValueTypeA, typename ValueTypeB, int blockrows_per_cta, int bsize>
__global__
void multicolorGSSmoothBlockDiaCsrKernel_NAIVE_tex_readDinv2(const IndexType *row_offsets, const IndexType *column_indices, const IndexType *dia_indices, const ValueTypeA *nonzero_values, const ValueTypeA *Dinv,
        const ValueTypeB *b, const ValueTypeB *x, ValueTypeB weight, const int *sorted_rows_by_color, const int num_rows_per_color, const int num_block_rows, ValueTypeB *xout)
{
    //const int vec_entry_index = threadIdx.x & (bsize-1);
    //const int block_eighthwarp_id = threadIdx.x >> log_bsize;
    //const int tid = blockDim.x*blockIdx.x+ threadIdx.x;
    //int eighthwarp_id = tid >> log_bsize;
    //volatile __shared__ ValueType s_xtemp[ bsize*eighthwarps_per_block ];
    int cta_blockrow_id = (threadIdx.x) / bsize;
    int blockrow_id = blockIdx.x * blockrows_per_cta + cta_blockrow_id;
    const int vec_entry_index = threadIdx.x - cta_blockrow_id * bsize;
    volatile __shared__ ValueTypeB s_xtemp[ bsize * blockrows_per_cta];
    ValueTypeB bmAx, xin;
    ValueTypeB temp[bsize];
    int offset, i, s_offset;

    while (blockrow_id < num_rows_per_color && cta_blockrow_id < blockrows_per_cta)
    {
        i = sorted_rows_by_color[blockrow_id];
        offset = i * bsize + vec_entry_index;
        bmAx = b[offset];
        // Contribution from diagonal
        xin = x[offset];
        s_xtemp[threadIdx.x] = xin;
        // Load dia_values and do matrix multiply
        s_offset = cta_blockrow_id * bsize;
        offset = dia_indices[i] * bsize * bsize + vec_entry_index * bsize;
#pragma unroll

        for (int k = 0; k < bsize; k++)
        {
            temp[k] = nonzero_values[offset + k];
        }

        //loadAsVector<bsize>(dia_values+offset,temp);
#pragma unroll

        for (int m = 0; m < bsize; m++)
        {
            bmAx -= temp[m] * s_xtemp[s_offset + m];
        }

        // Contribution from each nonzero column
        int jmax = row_offsets[i + 1];

        for (int jind = row_offsets[i]; jind < jmax; jind++)
        {
            IndexType jcol = column_indices[jind];

            if (jcol != i)
            {
                offset = jcol * bsize + vec_entry_index;
                s_xtemp[threadIdx.x] = x[offset];
                // Load nonzero_values
                offset = jind * bsize * bsize + vec_entry_index * bsize;
#pragma unroll

                for (int k = 0; k < bsize; k++)
                {
                    temp[k] = nonzero_values[offset + k];
                }

                //loadAsVector<bsize>(nonzero_values+offset,temp);
                s_offset = cta_blockrow_id * bsize;
#pragma unroll

                for (int m = 0; m < bsize; m++)
                {
                    bmAx -= temp[m] * s_xtemp[s_offset + m];
                }
            }
        }

        s_xtemp[threadIdx.x] = bmAx;
        bmAx = 0.;
        // Load Dinv and multiply to RHS
        offset = i * bsize * bsize + vec_entry_index * bsize;
#pragma unroll

        for (int k = 0; k < bsize; k++)
        {
            temp[k] = Dinv[offset + k];
        }

        //loadAsVector<bsize>(Dinv+offset,temp);
        s_offset = cta_blockrow_id * bsize;
#pragma unroll

        for (int m = 0; m < bsize; m++)
        {
            bmAx += temp[m] * s_xtemp[s_offset + m];
        }

        xout[i * bsize + vec_entry_index] = xin + weight * bmAx;
        blockrow_id +=  (gridDim.x * blockDim.x) / bsize;
    }
}



// Kernel to smooth, NAIVE implementation with texture
// This is TERRIBLE in terms of memory access pattern
template<typename IndexType, typename ValueTypeA,  typename ValueTypeB>
__global__
void multicolorGSSmoothCsrKernel_NAIVE_tex(const IndexType *row_offsets, const IndexType *column_indices, const IndexType *diag, const ValueTypeA *nonzero_values, const ValueTypeA *Dinv,
        const ValueTypeB *b, const ValueTypeB *x, ValueTypeB weight, const int *sorted_rows_by_color, const int num_rows_per_color, const int num_block_rows, ValueTypeB *xout)
{
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    int i;
    ValueTypeB bmAx, xin;
    ValueTypeB temp, s_xtemp;
    ValueTypeA dia;

    while (tid < num_rows_per_color)
    {
        i = sorted_rows_by_color[tid];
        bmAx = b[i];
        xin = x[i];
        dia = isNotCloseToZero(nonzero_values[diag[i]]) ? nonzero_values[diag[i]] : epsilon( nonzero_values[diag[i]]);
        bmAx -= xin * dia;
        // Contribution from each nonzero column
        int jmax = row_offsets[i + 1];

        for (int jind = row_offsets[i]; jind < jmax; jind++)
        {
            IndexType jcol = column_indices[jind];

            if (jcol != i)
            {
                s_xtemp = x[jcol];
                temp = nonzero_values[jind];
                bmAx -= temp * s_xtemp;
            }
        }

        bmAx /= dia;
        xout[i] = xin + weight * bmAx;
        tid +=  gridDim.x * blockDim.x;
    }
}

// Kernel to smooth, warp per row version
// Each warp processes a row; contributions of threads are reduced and stored to x by lane 0
template<typename IndexType, typename ValueTypeA,  typename ValueTypeB, int CTA_SIZE, int WARP_SIZE>
__global__
void multicolorGSSmoothCsrKernel_WarpPerRow(const IndexType *row_offsets, const IndexType *column_indices, const IndexType *diag, const ValueTypeA *nonzero_values, const ValueTypeA *Dinv,
        const ValueTypeB *b, const ValueTypeB *x, ValueTypeB weight, const int *sorted_rows_by_color, const int num_rows_per_color, const int num_block_rows, ValueTypeB *xout)
{
    int i;
    ValueTypeB bmAx, xin;
    ValueTypeA dia, diatemp;

    const int NUM_WARPS = CTA_SIZE / WARP_SIZE;
    
    const int warp_id = utils::warp_id();
    const int lane_id = utils::lane_id();

    constexpr unsigned full_mask = 0xffffffff;
    
    int row_id = blockIdx.x * NUM_WARPS + warp_id;
    int a_col_tmp;

    while (row_id < num_rows_per_color)
    {
        i = sorted_rows_by_color[row_id];

        if ( lane_id < 2 )
        {
            a_col_tmp = row_offsets[i + lane_id];
        }

        int a_col_it  = utils::shfl( a_col_tmp, 0 );
        int a_col_end = utils::shfl( a_col_tmp, 1 );

        xin = x[i];
        bmAx = amgx::types::util<ValueTypeB>::get_zero(); //utils::Ld<utils::LD_NC>::load(&b[i]);

        for (a_col_it += lane_id; utils::any(a_col_it < a_col_end); a_col_it += WARP_SIZE )
        {
            const bool is_active = a_col_it < a_col_end;        
       
            int x_row_id;
            ValueTypeB a_value;
            ValueTypeB x_value;

            if (is_active)
            {
                x_row_id = column_indices[a_col_it];
                a_value  = nonzero_values[a_col_it];
                x_value = x[x_row_id];

                bmAx -= a_value*x_value;
            }
        }

        #pragma unroll
        for (int offset = 32 >> 1 ; offset > 0; offset = offset >> 1) 
        {
            bmAx  += __shfl_down_sync(full_mask, bmAx, offset);
        }
        
        if (lane_id == 0)
        {
            bmAx += b[i];
            diatemp = nonzero_values[diag[i]];
            dia = isNotCloseToZero(diatemp) ? diatemp : epsilon(diatemp);
            bmAx /= dia;
            xout[i] = xin + weight * bmAx;
        }

        row_id += gridDim.x * NUM_WARPS;
    }
}

// Kernel to smooth, N per row version
// Each row is processed by N threads; contributions of threads are reduced and stored to x by first thread processing each row
template<typename IndexType, typename ValueTypeA,  typename ValueTypeB, int CTA_SIZE, int WARP_SIZE, int n_per_row>
__global__
void multicolorGSSmoothCsrKernel_nPerRow(const IndexType *row_offsets, const IndexType *column_indices, const IndexType *diag, const ValueTypeA *nonzero_values, const ValueTypeA *Dinv,
        const ValueTypeB *b, const ValueTypeB *x, ValueTypeB weight, const int *sorted_rows_by_color, const int num_rows_per_color, const int num_block_rows, ValueTypeB *xout)
{
    constexpr unsigned full_mask = 0xffffffff;

    const int warp_id = utils::warp_id();
    const int lane_id = utils::lane_id();

    const int me_in_row = lane_id % n_per_row;
    const int batch_row = lane_id / n_per_row;

    int row_id = blockIdx.x * CTA_SIZE/n_per_row + (warp_id*WARP_SIZE)/n_per_row + batch_row;
    const int row_id_increment = gridDim.x * blockDim.x/n_per_row;

    int i;
    ValueTypeB bmAx, temp, s_xtemp, diatemp;
    ValueTypeA dia;
    IndexType jcol;
    int jit, jmax;

    while (row_id < num_rows_per_color)
    {
        i = sorted_rows_by_color[row_id];
        bmAx = amgx::types::util<ValueTypeB>::get_zero();
        jit  = row_offsets[i];
        jmax = row_offsets[i + 1];     

        for (jit = jit + me_in_row; jit < jmax; jit += n_per_row)
        {
            jcol = column_indices[jit];
            temp = nonzero_values[jit];            
            s_xtemp = x[jcol];

            bmAx -= temp * s_xtemp;
        }

        #pragma unroll
        for (int offset = n_per_row >> 1 ; offset > 0; offset = offset >> 1) 
        {
            bmAx  += __shfl_down_sync(full_mask, bmAx, offset);
        }

        if (me_in_row == 0)
        {
            diatemp = nonzero_values[diag[i]];
            dia = isNotCloseToZero(diatemp) ? diatemp : epsilon(diatemp);
            bmAx += b[i];
            bmAx /= dia;
           
            xout[i] = x[i] + weight * bmAx;
        }

        row_id += row_id_increment;
   }
}

// Kernel to smooth, NAIVE implementation with texture
// Batch version
template<typename IndexType, typename ValueTypeA,  typename ValueTypeB>
__global__
void multicolorGSSmoothCsrKernel_NAIVE_tex_batched(const IndexType *row_offsets, const IndexType *column_indices, const IndexType *diag, const ValueTypeA *nonzero_values, const ValueTypeA *Dinv, const int batch_sz,
        const ValueTypeB *b, const ValueTypeB *x, ValueTypeB weight, const int *sorted_rows_by_color, const int num_rows_per_color, const int num_block_rows, ValueTypeB *xout)
{
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    int bid = tid / batch_sz;
    int vid = tid % batch_sz;
    int i;
    ValueTypeB bmAx, xin;
    ValueTypeB temp, s_xtemp;
    ValueTypeA dia;

    while (bid < num_rows_per_color)
    {
        i = sorted_rows_by_color[bid];
        bmAx = b[batch_sz * i + vid];
        xin = x[batch_sz * i + vid];
        dia = isNotCloseToZero(nonzero_values[diag[i]]) ? nonzero_values[diag[i]] : epsilon( nonzero_values[diag[i]]);
        bmAx -= xin * dia;
        // Contribution from each nonzero column
        int jmax = row_offsets[i + 1];

        for (int jind = row_offsets[i]; jind < jmax; jind++)
        {
            IndexType jcol = column_indices[jind];

            if (jcol != i)
            {
                s_xtemp = x[batch_sz * jcol + vid];
                temp = nonzero_values[jind];
                bmAx -= temp * s_xtemp;
            }
        }

        bmAx /= dia;
        xout[batch_sz * i + vid] = xin + weight * bmAx;
        tid +=  gridDim.x * blockDim.x;
        bid = tid / batch_sz;
    }
}

// -------------------
// Methods
// -------------------

// Constructor
template<class T_Config>
MulticolorGaussSeidelSolver_Base<T_Config>::MulticolorGaussSeidelSolver_Base( AMG_Config &cfg, const std::string &cfg_scope) : Solver<T_Config>( cfg, cfg_scope)
{
    this->weight = cfg.AMG_Config::getParameter<double>("relaxation_factor", cfg_scope);
    this->symFlag = cfg.AMG_Config::getParameter<int>("symmetric_GS", cfg_scope);
    this->m_reorder_cols_by_color_desired = (cfg.AMG_Config::getParameter<int>("reorder_cols_by_color", cfg_scope) != 0);
    this->m_insert_diagonal_desired = (cfg.AMG_Config::getParameter<int>("insert_diag_while_reordering", cfg_scope) != 0);

    if (cfg.AMG_Config::getParameter<int>("use_bsrxmv", cfg_scope))
    {
        this->use_bsrxmv = 1;
    }
    else
    {
        this->use_bsrxmv = 0;
    }

    if (this->weight == 0)
    {
        this->weight = 1.;
        amgx_printf("Warning, setting weight to 1 instead of estimating largest_eigen_value in Multicolor GaussSeidel smoother\n");
    }

    this->gs_method = KernelMethod::DEFAULT;

    cudaEventCreateWithFlags(&m_start, cudaEventDisableTiming);
    cudaCheckError();
    cudaEventCreateWithFlags(&m_end, cudaEventDisableTiming);
    cudaCheckError();
}

// Destructor
template<class T_Config>
MulticolorGaussSeidelSolver_Base<T_Config>::~MulticolorGaussSeidelSolver_Base()
{
    this->Dinv.resize(0);
}

template<class T_Config>
void MulticolorGaussSeidelSolver_Base<T_Config>::computeDinv(Matrix<T_Config> &A)
{
    ViewType oldView = A.currentView();
    A.setViewExterior();

    if (A.get_block_dimx() == 4 && A.get_block_dimy() == 4)
    {
        computeDinv_4x4(A);
    }
    else if (A.get_block_dimx() == 3 && A.get_block_dimy() == 3)
    {
        computeDinv_3x3(A);
    }
    else if (A.get_block_dimx() == 1 && A.get_block_dimy() == 1)
    {
        computeDinv_1x1(A);
    }
    else if (A.get_block_dimx() == 2 && A.get_block_dimy() == 2)
    {
        computeDinv_2x2(A);
    }
    else if (A.get_block_dimx() == 5 && A.get_block_dimy() == 5)
    {
        computeDinv_5x5(A);
    }
    else if (A.get_block_dimx() == A.get_block_dimy())
    {
        computeDinv_bxb(A, A.get_block_dimx());
    }
    else
    {
        FatalError("Unsupported block size for MulticolorGaussSeidelSolver computeEinv", AMGX_ERR_NOT_SUPPORTED_BLOCKSIZE);
    }

    A.setView(oldView);
}

// Method to compute the diagonal matrix E in DILU smoother
template <AMGX_VecPrecision t_vecPrec, AMGX_MatPrecision t_matPrec, AMGX_IndPrecision t_indPrec>
void MulticolorGaussSeidelSolver<TemplateConfig<AMGX_host, t_vecPrec, t_matPrec, t_indPrec> >::computeDinv_3x3(const Matrix_h &A)
{
    FatalError("Multicolor GS smoother not implemented with host format, exiting", AMGX_ERR_NOT_SUPPORTED_BLOCKSIZE);
}
template <AMGX_VecPrecision t_vecPrec, AMGX_MatPrecision t_matPrec, AMGX_IndPrecision t_indPrec>
void MulticolorGaussSeidelSolver<TemplateConfig<AMGX_host, t_vecPrec, t_matPrec, t_indPrec> >::computeDinv_4x4(const Matrix_h &A)
{
    FatalError("Multicolor GS smoother not implemented with host format, exiting", AMGX_ERR_NOT_SUPPORTED_BLOCKSIZE);
}

template <AMGX_VecPrecision t_vecPrec, AMGX_MatPrecision t_matPrec, AMGX_IndPrecision t_indPrec>
void MulticolorGaussSeidelSolver<TemplateConfig<AMGX_host, t_vecPrec, t_matPrec, t_indPrec> >::computeDinv_2x2(const Matrix_h &A)
{
    FatalError("Multicolor GS smoother not implemented with host format, exiting", AMGX_ERR_NOT_SUPPORTED_BLOCKSIZE);
}

template <AMGX_VecPrecision t_vecPrec, AMGX_MatPrecision t_matPrec, AMGX_IndPrecision t_indPrec>
void MulticolorGaussSeidelSolver<TemplateConfig<AMGX_host, t_vecPrec, t_matPrec, t_indPrec> >::computeDinv_5x5(const Matrix_h &A)
{
    FatalError("Multicolor GS smoother not implemented with host format, exiting", AMGX_ERR_NOT_SUPPORTED_BLOCKSIZE);
}

template <AMGX_VecPrecision t_vecPrec, AMGX_MatPrecision t_matPrec, AMGX_IndPrecision t_indPrec>
void MulticolorGaussSeidelSolver<TemplateConfig<AMGX_host, t_vecPrec, t_matPrec, t_indPrec> >::computeDinv_1x1(const Matrix_h &A)
{
    FatalError("Multicolor GS smoother not implemented with host format, exiting", AMGX_ERR_NOT_SUPPORTED_BLOCKSIZE);
}

template <AMGX_VecPrecision t_vecPrec, AMGX_MatPrecision t_matPrec, AMGX_IndPrecision t_indPrec>
void MulticolorGaussSeidelSolver<TemplateConfig<AMGX_host, t_vecPrec, t_matPrec, t_indPrec> >::computeDinv_bxb(const Matrix_h &A, const int bsize)
{
    FatalError("Multicolor GS smoother not implemented with host format, exiting", AMGX_ERR_NOT_SUPPORTED_BLOCKSIZE);
}


template <AMGX_VecPrecision t_vecPrec, AMGX_MatPrecision t_matPrec, AMGX_IndPrecision t_indPrec>
void MulticolorGaussSeidelSolver<TemplateConfig<AMGX_device, t_vecPrec, t_matPrec, t_indPrec> >::computeDinv_4x4(const Matrix_d &A)
{
//both DIAG supported
    this->Dinv.resize(A.get_num_cols()*A.get_block_dimx()*A.get_block_dimy(), 0.0);
    const IndexType *A_row_offsets_ptr = A.row_offsets.raw();
    const IndexType *A_column_indices_ptr = A.col_indices.raw();
    const IndexType *A_dia_idx_ptr = A.diag.raw();
    ValueTypeA *Dinv_ptr = this->Dinv.raw();
    const ValueTypeA *A_nonzero_values_ptr = A.values.raw();
    // MUST BE MULTIPLE OF 16
    const int threads_per_block = 512;
    const int halfwarps_per_block = threads_per_block / 16;
    const int num_blocks = min(AMGX_GRID_MAX_SIZE, (int) (A.get_num_rows() - 1) / halfwarps_per_block + 1);
    cudaFuncSetCacheConfig(setupBlockGSSmooth4by4BlockDiaCsrKernel_V2<IndexType, ValueTypeA, threads_per_block, halfwarps_per_block, 4, 2, 16, 4>, cudaFuncCachePreferL1);
    setupBlockGSSmooth4by4BlockDiaCsrKernel_V2<IndexType, ValueTypeA, threads_per_block, halfwarps_per_block, 4, 2, 16, 4> <<< num_blocks, threads_per_block>>>
    (A_row_offsets_ptr, A_column_indices_ptr, A_nonzero_values_ptr, A_dia_idx_ptr, Dinv_ptr, A.get_num_rows());
    cudaCheckError();
}

template <AMGX_VecPrecision t_vecPrec, AMGX_MatPrecision t_matPrec, AMGX_IndPrecision t_indPrec>
void MulticolorGaussSeidelSolver<TemplateConfig<AMGX_device, t_vecPrec, t_matPrec, t_indPrec> >::computeDinv_bxb(const Matrix_d &A, const int bsize)
{
//both DIAG supported
    //TODO: check correctness with blocksizes > 5
    this->Dinv.resize(A.get_num_cols()*A.get_block_dimx()*A.get_block_dimy(), 0.0);
    const IndexType *A_row_offsets_ptr = A.row_offsets.raw();
    const IndexType *A_column_indices_ptr = A.col_indices.raw();
    const IndexType *A_dia_idx_ptr = A.diag.raw();
    ValueTypeA *Dinv_ptr= this->Dinv.raw();
    const ValueTypeA *A_nonzero_values_ptr = A.values.raw();

    MVector temp(AMGX_GRID_MAX_SIZE * ((bsize-1)/4+1)*((bsize-1)/4+1));
    ValueTypeA *temp_ptr= temp.raw();

    // MUST BE MULTIPLE OF 16
    const int threads_per_block = 512;
    const int halfwarps_per_block = threads_per_block/16;
    const int num_blocks = min(AMGX_GRID_MAX_SIZE, (int) (A.get_num_rows()-1)/halfwarps_per_block + 1);

    cudaFuncSetCacheConfig(setupBlockGSSmoothbBybBlockDiaCsrKernel<IndexType,ValueTypeA,threads_per_block,halfwarps_per_block>,cudaFuncCachePreferL1);
    setupBlockGSSmoothbBybBlockDiaCsrKernel<IndexType,ValueTypeA,threads_per_block,halfwarps_per_block> <<<num_blocks,threads_per_block, sizeof(ValueTypeB)*bsize*bsize*halfwarps_per_block>>>
    (A_row_offsets_ptr, A_column_indices_ptr, A_nonzero_values_ptr, A_dia_idx_ptr, Dinv_ptr, A.get_num_rows(), bsize, bsize*bsize, temp_ptr);
    cudaCheckError();
}

template <AMGX_VecPrecision t_vecPrec, AMGX_MatPrecision t_matPrec, AMGX_IndPrecision t_indPrec>
void MulticolorGaussSeidelSolver<TemplateConfig<AMGX_device, t_vecPrec, t_matPrec, t_indPrec> >::computeDinv_3x3(const Matrix_d &A)
{
//both DIAG supported
    this->Dinv.resize(A.get_num_cols()*A.get_block_dimx()*A.get_block_dimy(), 0.0);
    const IndexType *A_row_offsets_ptr = A.row_offsets.raw();
    const IndexType *A_column_indices_ptr = A.col_indices.raw();
    const IndexType *A_dia_idx_ptr = A.diag.raw();
    ValueTypeA *Dinv_ptr = this->Dinv.raw();
    const ValueTypeA *A_nonzero_values_ptr = A.values.raw();
    // MUST BE MULTIPLE OF 16
    const int bsize = 3;
    const int bsize_sq = bsize * bsize;
    const int WARP_SIZE = 32;
    const int threads_per_block = 256;
    const int blockrows_per_warp = WARP_SIZE / bsize_sq;
    const int blockrows_per_cta = (threads_per_block / WARP_SIZE) * blockrows_per_warp;
    const int num_blocks = min(AMGX_GRID_MAX_SIZE, (int) (A.get_num_rows() - 1) / blockrows_per_cta + 1);
    cudaFuncSetCacheConfig(setupBlockGSSmooth3by3BlockDiaCsrKernel<IndexType, ValueTypeA, blockrows_per_cta, blockrows_per_warp, bsize, bsize_sq>, cudaFuncCachePreferL1);
    setupBlockGSSmooth3by3BlockDiaCsrKernel<IndexType, ValueTypeA, blockrows_per_cta, blockrows_per_warp, bsize, bsize_sq> <<< num_blocks, threads_per_block>>>
    (A_row_offsets_ptr, A_column_indices_ptr, A_nonzero_values_ptr, A_dia_idx_ptr, Dinv_ptr, A.get_num_rows());
    cudaCheckError();
}

template <AMGX_VecPrecision t_vecPrec, AMGX_MatPrecision t_matPrec, AMGX_IndPrecision t_indPrec>
void MulticolorGaussSeidelSolver<TemplateConfig<AMGX_device, t_vecPrec, t_matPrec, t_indPrec> >::computeDinv_2x2(const Matrix_d &A)
{
//both DIAG supported
    this->Dinv.resize(A.get_num_cols()*A.get_block_dimx()*A.get_block_dimy(), 0.0);
    const IndexType *A_row_offsets_ptr = A.row_offsets.raw();
    const IndexType *A_column_indices_ptr = A.col_indices.raw();
    const IndexType *A_dia_idx_ptr = A.diag.raw();
    ValueTypeA *Dinv_ptr = this->Dinv.raw();
    const ValueTypeA *A_nonzero_values_ptr = A.values.raw();
    // MUST BE MULTIPLE OF 16
    const int bsize = 2;
    const int bsize_sq = bsize * bsize;
    const int WARP_SIZE = 32;
    const int threads_per_block = 256;
    const int blockrows_per_warp = WARP_SIZE / bsize_sq;
    const int blockrows_per_cta = (threads_per_block / WARP_SIZE) * blockrows_per_warp;
    const int num_blocks = min(AMGX_GRID_MAX_SIZE, (int) (A.get_num_rows() - 1) / blockrows_per_cta + 1);
    cudaFuncSetCacheConfig(setupBlockGSSmooth3by3BlockDiaCsrKernel<IndexType, ValueTypeA, blockrows_per_cta, blockrows_per_warp, bsize, bsize_sq>, cudaFuncCachePreferL1);
    setupBlockGSSmooth3by3BlockDiaCsrKernel<IndexType, ValueTypeA, blockrows_per_cta, blockrows_per_warp, bsize, bsize_sq> <<< num_blocks, threads_per_block>>>
    (A_row_offsets_ptr, A_column_indices_ptr, A_nonzero_values_ptr, A_dia_idx_ptr, Dinv_ptr, A.get_num_rows());
    cudaCheckError();
}

template <AMGX_VecPrecision t_vecPrec, AMGX_MatPrecision t_matPrec, AMGX_IndPrecision t_indPrec>
void MulticolorGaussSeidelSolver<TemplateConfig<AMGX_device, t_vecPrec, t_matPrec, t_indPrec> >::computeDinv_5x5(const Matrix_d &A)
{
//both DIAG supported
    this->Dinv.resize(A.get_num_cols()*A.get_block_dimx()*A.get_block_dimy(), 0.0);
    const IndexType *A_row_offsets_ptr = A.row_offsets.raw();
    const IndexType *A_column_indices_ptr = A.col_indices.raw();
    const IndexType *A_dia_idx_ptr = A.diag.raw();
    ValueTypeA *Dinv_ptr = this->Dinv.raw();
    const ValueTypeA *A_nonzero_values_ptr = A.values.raw();
    // MUST BE MULTIPLE OF 16
    const int bsize = 5;
    const int bsize_sq = bsize * bsize;
    const int WARP_SIZE = 32;
    const int threads_per_block = 256;
    const int blockrows_per_warp = WARP_SIZE / bsize_sq;
    const int blockrows_per_cta = (threads_per_block / WARP_SIZE) * blockrows_per_warp;
    const int num_blocks = min(AMGX_GRID_MAX_SIZE, (int) (A.get_num_rows() - 1) / blockrows_per_cta + 1);
    cudaFuncSetCacheConfig(setupBlockGSSmooth3by3BlockDiaCsrKernel<IndexType, ValueTypeA, blockrows_per_cta, blockrows_per_warp, bsize, bsize_sq>, cudaFuncCachePreferL1);
    setupBlockGSSmooth3by3BlockDiaCsrKernel<IndexType, ValueTypeA, blockrows_per_cta, blockrows_per_warp, bsize, bsize_sq> <<< num_blocks, threads_per_block>>>
    (A_row_offsets_ptr, A_column_indices_ptr, A_nonzero_values_ptr, A_dia_idx_ptr, Dinv_ptr, A.get_num_rows());
    cudaCheckError();
}

template <AMGX_VecPrecision t_vecPrec, AMGX_MatPrecision t_matPrec, AMGX_IndPrecision t_indPrec>
void MulticolorGaussSeidelSolver<TemplateConfig<AMGX_device, t_vecPrec, t_matPrec, t_indPrec> >::computeDinv_1x1(const Matrix_d &A)
{
    if (this->use_bsrxmv && A.get_block_dimx() == A.get_block_dimy())
    {
        this->Dinv.resize(A.get_num_cols()*A.get_block_dimx()*A.get_block_dimy(), 0.0);
        const int threads_per_block = 256;
        const int num_blocks = min(AMGX_GRID_MAX_SIZE, (int) (A.get_num_rows() + threads_per_block - 1) / threads_per_block);
        cudaFuncSetCacheConfig(setupBlockGSSmooth1x1<IndexType, ValueTypeA>, cudaFuncCachePreferL1);
        setupBlockGSSmooth1x1<IndexType, ValueTypeA> <<< num_blocks, threads_per_block>>>(A.row_offsets.raw(), A.col_indices.raw(), A.values.raw(), A.diag.raw(), this->Dinv.raw(), A.get_num_rows());
        cudaCheckError();
    }
}

template<class T_Config>
void
MulticolorGaussSeidelSolver_Base<T_Config>::printSolverParameters() const
{
    std::cout << "relaxation_factor= " << this->weight << std::endl;
    std::cout << "use_bsrxmv = " << this->use_bsrxmv << std::endl;
    std::cout << "symmetric_GS = " << this->symFlag << std::endl;
}

// Solver setup
template<class T_Config>
void
MulticolorGaussSeidelSolver_Base<T_Config>::solver_setup(bool reuse_matrix_structure)
{
    this->m_explicit_A = dynamic_cast<Matrix<T_Config>*>(this->m_A);

    if (!this->m_explicit_A)
    {
        FatalError("MulticolorGaussSeidelSolver only works with explicit matrices", AMGX_ERR_INTERNAL);
    }

    if ( this->m_explicit_A->getColoringLevel() < 1)
    {
        FatalError("Matrix must be colored to use multicolor gauss-seidel solver. Try setting: coloring_level=1 in the configuration file", AMGX_ERR_NOT_IMPLEMENTED);
    }

    if ( this->m_explicit_A->getBlockFormat() != ROW_MAJOR )
    {
        FatalError("Multicolor DILU solver only supports row major format for the blocks", AMGX_ERR_CONFIGURATION);
    }

    computeDinv( *this->m_explicit_A );
}

//
template<class T_Config>
void
MulticolorGaussSeidelSolver_Base<T_Config>::solve_init( VVector &b, VVector &x, bool xIsZero )
{
}

template <AMGX_VecPrecision t_vecPrec, AMGX_MatPrecision t_matPrec, AMGX_IndPrecision t_indPrec>
void MulticolorGaussSeidelSolver<TemplateConfig<AMGX_host, t_vecPrec, t_matPrec, t_indPrec> >::smooth_1x1(const Matrix_h &A, const VVector &b, VVector &x, ViewType separation_flag)
{
    FatalError("Haven't implemented Block Multicolor Gauss-Seidel smoother for host format", AMGX_ERR_NOT_SUPPORTED_TARGET);
}

template <AMGX_VecPrecision t_vecPrec, AMGX_MatPrecision t_matPrec, AMGX_IndPrecision t_indPrec>
void MulticolorGaussSeidelSolver<TemplateConfig<AMGX_host, t_vecPrec, t_matPrec, t_indPrec> >::smooth_1x1_naive(const Matrix_h &A, const VVector &b, VVector &x, ViewType separation_flag)
{
    FatalError("Haven't implemented Block Multicolor Gauss-Seidel smoother for host format", AMGX_ERR_NOT_SUPPORTED_TARGET);
}

template <AMGX_VecPrecision t_vecPrec, AMGX_MatPrecision t_matPrec, AMGX_IndPrecision t_indPrec>
void MulticolorGaussSeidelSolver<TemplateConfig<AMGX_host, t_vecPrec, t_matPrec, t_indPrec> >::smooth_3x3(const Matrix_h &A, const VVector &b, VVector &x, ViewType separation_flag)
{
    FatalError("Haven't implemented Block Multicolor Gauss-Seidel smoother for host format", AMGX_ERR_NOT_SUPPORTED_TARGET);
}

template <AMGX_VecPrecision t_vecPrec, AMGX_MatPrecision t_matPrec, AMGX_IndPrecision t_indPrec>
void MulticolorGaussSeidelSolver<TemplateConfig<AMGX_host, t_vecPrec, t_matPrec, t_indPrec> >::smooth_4x4(const Matrix_h &A, const VVector &b, VVector &x, ViewType separation_flag)
{
    FatalError("Haven't implemented Block Multicolor Gauss-Seidel smoother for host format", AMGX_ERR_NOT_SUPPORTED_TARGET);
}

template <AMGX_VecPrecision t_vecPrec, AMGX_MatPrecision t_matPrec, AMGX_IndPrecision t_indPrec>
void MulticolorGaussSeidelSolver<TemplateConfig<AMGX_host, t_vecPrec, t_matPrec, t_indPrec> >::smooth_BxB(Matrix_h &A, VVector &b, VVector &x, ViewType separation_flag)
{
    FatalError("Haven't implemented Block Multicolor Gauss-Seidel smoother for host format", AMGX_ERR_NOT_SUPPORTED_TARGET);
}

template <AMGX_VecPrecision t_vecPrec, AMGX_MatPrecision t_matPrec, AMGX_IndPrecision t_indPrec>
void MulticolorGaussSeidelSolver<TemplateConfig<AMGX_device, t_vecPrec, t_matPrec, t_indPrec> >::smooth_1x1_naive(const Matrix_d &A, const VVector &b, VVector &x, ViewType separation_flag)
{
    const IndexType *A_row_offsets_ptr = A.row_offsets.raw();
    const IndexType *A_column_indices_ptr = A.col_indices.raw();
    const ValueTypeA *A_nonzero_values_ptr = A.values.raw();
    const ValueTypeB *b_ptr = b.raw();
    const IndexType *A_sorted_rows_by_color_ptr = A.getMatrixColoring().getSortedRowsByColor().raw();
    const IndexType *A_diag_ptr = A.diag.raw();
    const ValueTypeA *Dinv_ptr = this->Dinv.raw();
    ValueTypeB *x_ptr = x.raw();
    const int threads_per_block = 64;
    const int num_colors = this->m_explicit_A->getMatrixColoring().getNumColors();

    for (int i = 0; i < num_colors; i++)
    {
        const IndexType color_offset = ((separation_flag & INTERIOR) == 0) ? A.getMatrixColoring().getSeparationOffsetsRowsPerColor()[i] : A.getMatrixColoring().getOffsetsRowsPerColor()[i];
        const IndexType num_rows_per_color = ((separation_flag == this->m_explicit_A->getViewInterior()) ? A.getMatrixColoring().getSeparationOffsetsRowsPerColor()[i] : A.getMatrixColoring().getOffsetsRowsPerColor()[i + 1]) - color_offset;

        if (num_rows_per_color == 0) { continue; }

        const int threads_per_block = 512;
        //const int eightwarps_per_block = threads_per_block/4;
        const int num_blocks = min( AMGX_GRID_MAX_SIZE, (int)(num_rows_per_color - 1) / threads_per_block + 1 );
        cudaFuncSetCacheConfig(multicolorGSSmoothCsrKernel_NAIVE_tex<IndexType, ValueTypeA, ValueTypeB>, cudaFuncCachePreferL1);
        multicolorGSSmoothCsrKernel_NAIVE_tex<IndexType, ValueTypeA, ValueTypeB> <<< num_blocks, threads_per_block>>>
        (A_row_offsets_ptr, A_column_indices_ptr, A_diag_ptr, A_nonzero_values_ptr, Dinv_ptr,
         b_ptr, x_ptr, this->weight, A_sorted_rows_by_color_ptr + color_offset, num_rows_per_color, A.get_num_rows(), x_ptr);
        cudaCheckError();
    }

    if (this->symFlag == 1)
    {
        for (int i = num_colors - 1; i >= 0; i--)
        {
            const IndexType color_offset = ((separation_flag & INTERIOR) == 0) ? A.getMatrixColoring().getSeparationOffsetsRowsPerColor()[i] : A.getMatrixColoring().getOffsetsRowsPerColor()[i];
            const IndexType num_rows_per_color = ((separation_flag == this->m_explicit_A->getViewInterior()) ? A.getMatrixColoring().getSeparationOffsetsRowsPerColor()[i] : A.getMatrixColoring().getOffsetsRowsPerColor()[i + 1]) - color_offset;

            if (num_rows_per_color == 0) { continue; }

            const int num_blocks = min( AMGX_GRID_MAX_SIZE, (int)(num_rows_per_color - 1) / threads_per_block + 1 );
            cudaFuncSetCacheConfig(multicolorGSSmoothCsrKernel_NAIVE_tex<IndexType, ValueTypeA, ValueTypeB>, cudaFuncCachePreferL1);
            multicolorGSSmoothCsrKernel_NAIVE_tex<IndexType, ValueTypeA, ValueTypeB> <<< num_blocks, threads_per_block>>>
            (A_row_offsets_ptr, A_column_indices_ptr, A_diag_ptr, A_nonzero_values_ptr, Dinv_ptr,
             b_ptr, x_ptr, this->weight, A_sorted_rows_by_color_ptr + color_offset, num_rows_per_color, A.get_num_rows(), x_ptr);
            cudaCheckError();
        }
    }

    cudaCheckError();
}

template <AMGX_VecPrecision t_vecPrec, AMGX_MatPrecision t_matPrec, AMGX_IndPrecision t_indPrec>
void MulticolorGaussSeidelSolver<TemplateConfig<AMGX_device, t_vecPrec, t_matPrec, t_indPrec> >::smooth_1x1(const Matrix_d &A, const VVector &b, VVector &x, ViewType separation_flag)
{
    const IndexType *A_row_offsets_ptr = A.row_offsets.raw();
    const IndexType *A_column_indices_ptr = A.col_indices.raw();
    const ValueTypeA *A_nonzero_values_ptr = A.values.raw();
    const ValueTypeB *b_ptr = b.raw();
    const IndexType *A_sorted_rows_by_color_ptr = A.getMatrixColoring().getSortedRowsByColor().raw();
    const IndexType *A_diag_ptr = A.diag.raw();
    const ValueTypeA *Dinv_ptr = this->Dinv.raw();
    ValueTypeB *x_ptr = x.raw();
    const int num_colors = this->m_explicit_A->getMatrixColoring().getNumColors();

    cudaStream_t stream = 0; // Stream where previous work is done. Currently - default stream
    cudaStream_t work_stream = stream;
    int use_l2_hint = 0;
    int use_aux_stream = 0; // use separate async stream to do smoothing
    cudaDeviceProp props = getDeviceProperties();
    int arch = 10 * props.major + props.minor;

    const int WARP_SIZE = 32;
    const int GRID_SIZE = 1024;
    const int CTA_SIZE  = 672;
    
    KernelMethod selected_method = this->gs_method;
    if (this->gs_method == KernelMethod::DEFAULT)
    {
        // Use 4 threads per row
        // If >500K rows, use L2 hint to keep ‘x’ cached (Ampere and newer)
        // If <500K NNZ per color, use T32_PER_ROW (WARP_PER_ROW might work as well or better)
        // If >20 NNZ per row, use T32_PER_ROW (WARP_PER_ROW might work as well or better)
        selected_method = KernelMethod::T4_PER_ROW;
        if (A.get_num_rows() > 500000) use_l2_hint = 1;
        if (A.get_num_nz() / A.get_num_rows() > 20) selected_method = KernelMethod::T32_PER_ROW;
        if (A.get_num_nz() / this->m_explicit_A->getMatrixColoring().getNumColors() < 500000) selected_method = KernelMethod::T32_PER_ROW;
    }

    if (use_l2_hint) use_aux_stream = 1; // l2 hint can be enabled only on async stream
    if (use_aux_stream) work_stream = this->get_aux_stream();

    if (use_aux_stream)
    {
        cudaEventRecord(this->m_start, stream);
        cudaStreamWaitEvent(work_stream, this->m_start, 0); // work_stream to wait for stream 
    }

    // try to keep 'x' in L2 cache, if at least Ampere & CUDA 11
#if CUDART_VERSION >= 11000
    cudaStreamAttrValue stream_attribute;
    if (arch >= 80 && use_l2_hint == 1)
    {
        cudaDeviceProp prop = getDeviceProperties();
        size_t x_size = min( A.get_num_rows()*8 , prop.persistingL2CacheMaxSize );      // set-aside length of 'x' (number of rows in A)*8 bytes
        cudaDeviceSetLimit( cudaLimitPersistingL2CacheSize, x_size);                    //            for persisting accesses or the max allowed
        stream_attribute.accessPolicyWindow.base_ptr  = reinterpret_cast<void*>(x_ptr); 
        stream_attribute.accessPolicyWindow.num_bytes = x_size;
        stream_attribute.accessPolicyWindow.hitRatio  = 1.0;                            // Hint for cache hit ratio
        stream_attribute.accessPolicyWindow.hitProp   = cudaAccessPropertyPersisting;   // Type of access property on cache hit
        stream_attribute.accessPolicyWindow.missProp  = cudaAccessPropertyStreaming;    // Type of access property on cache miss.
        cudaStreamSetAttribute(work_stream, cudaStreamAttributeAccessPolicyWindow, &stream_attribute);   
    }
#endif

    for (int i = 0; i < num_colors; i++)
    {
        const IndexType color_offset = ((separation_flag & INTERIOR) == 0) ? A.getMatrixColoring().getSeparationOffsetsRowsPerColor()[i] : A.getMatrixColoring().getOffsetsRowsPerColor()[i];
        const IndexType num_rows_per_color = ((separation_flag == this->m_explicit_A->getViewInterior()) ? A.getMatrixColoring().getSeparationOffsetsRowsPerColor()[i] : A.getMatrixColoring().getOffsetsRowsPerColor()[i + 1]) - color_offset;

        if (num_rows_per_color == 0) { continue; }

        if (selected_method == KernelMethod::NAIVE)
        {
            const int num_blocks = min( AMGX_GRID_MAX_SIZE, (int)(num_rows_per_color - 1) / CTA_SIZE + 1 );
            cudaFuncSetCacheConfig(multicolorGSSmoothCsrKernel_NAIVE_tex<IndexType, ValueTypeA, ValueTypeB>, cudaFuncCachePreferL1);
            multicolorGSSmoothCsrKernel_NAIVE_tex<IndexType, ValueTypeA, ValueTypeB> <<< num_blocks, CTA_SIZE, 0, work_stream>>>
            (A_row_offsets_ptr, A_column_indices_ptr, A_diag_ptr, A_nonzero_values_ptr, Dinv_ptr,
            b_ptr, x_ptr, this->weight, A_sorted_rows_by_color_ptr + color_offset, num_rows_per_color, A.get_num_rows(), x_ptr);
        }
        else if (selected_method == KernelMethod::WARP_PER_ROW)
        {
            cudaFuncSetCacheConfig(multicolorGSSmoothCsrKernel_WarpPerRow<IndexType, ValueTypeA, ValueTypeB, CTA_SIZE, WARP_SIZE>, cudaFuncCachePreferL1);
            multicolorGSSmoothCsrKernel_WarpPerRow<IndexType, ValueTypeA, ValueTypeB, CTA_SIZE, WARP_SIZE> <<< GRID_SIZE, CTA_SIZE, 0, work_stream>>>
            (A_row_offsets_ptr, A_column_indices_ptr, A_diag_ptr, A_nonzero_values_ptr, Dinv_ptr,
             b_ptr, x_ptr, this->weight, A_sorted_rows_by_color_ptr + color_offset, num_rows_per_color, A.get_num_rows(), x_ptr);
        }
        else if (selected_method == KernelMethod::T32_PER_ROW)
        {
            const int N_PER_ROW = 32;
            cudaFuncSetCacheConfig(multicolorGSSmoothCsrKernel_nPerRow<IndexType, ValueTypeA, ValueTypeB, CTA_SIZE, WARP_SIZE, N_PER_ROW>, cudaFuncCachePreferL1);
            multicolorGSSmoothCsrKernel_nPerRow<IndexType, ValueTypeA, ValueTypeB, CTA_SIZE, WARP_SIZE, N_PER_ROW> <<< GRID_SIZE, CTA_SIZE, 0, work_stream>>>
            (A_row_offsets_ptr, A_column_indices_ptr, A_diag_ptr, A_nonzero_values_ptr, Dinv_ptr,
             b_ptr, x_ptr, this->weight, A_sorted_rows_by_color_ptr + color_offset, num_rows_per_color, A.get_num_rows(), x_ptr);  
        }
        else if (selected_method == KernelMethod::T4_PER_ROW)
        {
            const int N_PER_ROW = 4;
            cudaFuncSetCacheConfig(multicolorGSSmoothCsrKernel_nPerRow<IndexType, ValueTypeA, ValueTypeB, CTA_SIZE, WARP_SIZE, N_PER_ROW>, cudaFuncCachePreferL1);
            multicolorGSSmoothCsrKernel_nPerRow<IndexType, ValueTypeA, ValueTypeB, CTA_SIZE, WARP_SIZE, N_PER_ROW> <<< GRID_SIZE, CTA_SIZE, 0, work_stream>>>
            (A_row_offsets_ptr, A_column_indices_ptr, A_diag_ptr, A_nonzero_values_ptr, Dinv_ptr,
             b_ptr, x_ptr, this->weight, A_sorted_rows_by_color_ptr + color_offset, num_rows_per_color, A.get_num_rows(), x_ptr);  
        }
        else
        {
            FatalError("Incorrect GS configuration (gs_method value?)", AMGX_ERR_CONFIGURATION);
        }
        cudaCheckError();
    }

    if (this->symFlag == 1)
    {
        for (int i = num_colors - 1; i >= 0; i--)
        {
            const IndexType color_offset = ((separation_flag & INTERIOR) == 0) ? A.getMatrixColoring().getSeparationOffsetsRowsPerColor()[i] : A.getMatrixColoring().getOffsetsRowsPerColor()[i];
            const IndexType num_rows_per_color = ((separation_flag == this->m_explicit_A->getViewInterior()) ? A.getMatrixColoring().getSeparationOffsetsRowsPerColor()[i] : A.getMatrixColoring().getOffsetsRowsPerColor()[i + 1]) - color_offset;

            if (num_rows_per_color == 0) { continue; }

            if (selected_method == KernelMethod::NAIVE)
            {
                const int num_blocks = min( AMGX_GRID_MAX_SIZE, (int)(num_rows_per_color - 1) / CTA_SIZE + 1 );
                cudaFuncSetCacheConfig(multicolorGSSmoothCsrKernel_NAIVE_tex<IndexType, ValueTypeA, ValueTypeB>, cudaFuncCachePreferL1);
                multicolorGSSmoothCsrKernel_NAIVE_tex<IndexType, ValueTypeA, ValueTypeB> <<< num_blocks, CTA_SIZE, 0, work_stream>>>
                (A_row_offsets_ptr, A_column_indices_ptr, A_diag_ptr, A_nonzero_values_ptr, Dinv_ptr,
                b_ptr, x_ptr, this->weight, A_sorted_rows_by_color_ptr + color_offset, num_rows_per_color, A.get_num_rows(), x_ptr);
            }
            else if (selected_method == KernelMethod::WARP_PER_ROW)
            {
                cudaFuncSetCacheConfig(multicolorGSSmoothCsrKernel_WarpPerRow<IndexType, ValueTypeA, ValueTypeB, CTA_SIZE, WARP_SIZE>, cudaFuncCachePreferL1);
                multicolorGSSmoothCsrKernel_WarpPerRow<IndexType, ValueTypeA, ValueTypeB, CTA_SIZE, WARP_SIZE> <<< GRID_SIZE, CTA_SIZE, 0, work_stream>>>
                (A_row_offsets_ptr, A_column_indices_ptr, A_diag_ptr, A_nonzero_values_ptr, Dinv_ptr,
                 b_ptr, x_ptr, this->weight, A_sorted_rows_by_color_ptr + color_offset, num_rows_per_color, A.get_num_rows(), x_ptr);
            }
            else if (selected_method == KernelMethod::T32_PER_ROW)
            {
                const int N_PER_ROW = 32;
                cudaFuncSetCacheConfig(multicolorGSSmoothCsrKernel_nPerRow<IndexType, ValueTypeA, ValueTypeB, CTA_SIZE, WARP_SIZE, N_PER_ROW>, cudaFuncCachePreferL1);
                multicolorGSSmoothCsrKernel_nPerRow<IndexType, ValueTypeA, ValueTypeB, CTA_SIZE, WARP_SIZE, N_PER_ROW> <<< GRID_SIZE, CTA_SIZE, 0, work_stream>>>
                (A_row_offsets_ptr, A_column_indices_ptr, A_diag_ptr, A_nonzero_values_ptr, Dinv_ptr,
                 b_ptr, x_ptr, this->weight, A_sorted_rows_by_color_ptr + color_offset, num_rows_per_color, A.get_num_rows(), x_ptr);  
            }
            else if (selected_method == KernelMethod::T4_PER_ROW)
            {
                const int N_PER_ROW = 4;
                cudaFuncSetCacheConfig(multicolorGSSmoothCsrKernel_nPerRow<IndexType, ValueTypeA, ValueTypeB, CTA_SIZE, WARP_SIZE, N_PER_ROW>, cudaFuncCachePreferL1);
                multicolorGSSmoothCsrKernel_nPerRow<IndexType, ValueTypeA, ValueTypeB, CTA_SIZE, WARP_SIZE, N_PER_ROW> <<< GRID_SIZE, CTA_SIZE, 0, work_stream>>>
                (A_row_offsets_ptr, A_column_indices_ptr, A_diag_ptr, A_nonzero_values_ptr, Dinv_ptr,
                 b_ptr, x_ptr, this->weight, A_sorted_rows_by_color_ptr + color_offset, num_rows_per_color, A.get_num_rows(), x_ptr);  
            }
            else
            {
                FatalError("Incorrect GS configuration (gs_method value?)", AMGX_ERR_CONFIGURATION);
            }
            cudaCheckError();  
        }
    }

    // reset persisting L2 cache
#if CUDART_VERSION >= 11000
    if (arch >= 80 && use_l2_hint == 1)
    {        
        stream_attribute.accessPolicyWindow.num_bytes = 0;                                          // Setting the window size to 0 disable it
        cudaStreamSetAttribute(work_stream, cudaStreamAttributeAccessPolicyWindow, &stream_attribute);   // Overwrite the access policy attribute to a CUDA Stream
        cudaCtxResetPersistingL2Cache();                                                            // Remove any persistent lines in L2 
    }
#endif

    if (use_aux_stream)
    {
        cudaEventRecord(this->m_start, work_stream);
        cudaStreamWaitEvent(stream, this->m_start, 0); // stream to wait for work_stream
    }
    cudaCheckError();
}

template <AMGX_VecPrecision t_vecPrec, AMGX_MatPrecision t_matPrec, AMGX_IndPrecision t_indPrec>
void MulticolorGaussSeidelSolver<TemplateConfig<AMGX_device, t_vecPrec, t_matPrec, t_indPrec> >::batch_smooth_1x1(const Matrix_d &A, int batch_sz, const VVector &b, VVector &x)
{
    const IndexType *A_row_offsets_ptr = A.row_offsets.raw();
    const IndexType *A_column_indices_ptr = A.col_indices.raw();
    const ValueTypeA *A_nonzero_values_ptr = A.values.raw();
    const IndexType *A_sorted_rows_by_color_ptr = A.getMatrixColoring().getSortedRowsByColor().raw();
    const IndexType *A_diag_ptr = A.diag.raw();
    const ValueTypeA *Dinv_ptr = this->Dinv.raw();
    const int threads_per_block = 64;
    const int num_colors = this->m_explicit_A->getMatrixColoring().getNumColors();

    for (int ivec = 0; ivec < batch_sz; ivec++)
    {
        const ValueTypeB *b_ptr = b.raw() + ivec * A.get_num_rows();
        ValueTypeB *x_ptr = x.raw() + ivec * A.get_num_rows();

        for (int i = 0; i < num_colors; i++)
        {
            const IndexType color_offset =  A.getMatrixColoring().getOffsetsRowsPerColor()[i];
            const IndexType num_rows_per_color = A.getMatrixColoring().getOffsetsRowsPerColor()[i + 1] - color_offset;

            if (num_rows_per_color == 0) { continue; }

            const int threads_per_block = 512;
            //const int eightwarps_per_block = threads_per_block/4;
            const int num_blocks = min( AMGX_GRID_MAX_SIZE, (int)(num_rows_per_color - 1) / threads_per_block + 1 );
            cudaFuncSetCacheConfig(multicolorGSSmoothCsrKernel_NAIVE_tex<IndexType, ValueTypeA, ValueTypeB>, cudaFuncCachePreferL1);
            multicolorGSSmoothCsrKernel_NAIVE_tex<IndexType, ValueTypeA, ValueTypeB> <<< num_blocks, threads_per_block>>>
            (A_row_offsets_ptr, A_column_indices_ptr, A_diag_ptr, A_nonzero_values_ptr, Dinv_ptr,
             b_ptr, x_ptr, this->weight, A_sorted_rows_by_color_ptr + color_offset, num_rows_per_color, A.get_num_rows(), x_ptr);
            cudaCheckError();
        }

        if (this->symFlag == 1)
        {
            for (int i = num_colors - 1; i >= 0; i--)
            {
                const IndexType color_offset =  A.getMatrixColoring().getOffsetsRowsPerColor()[i];
                const IndexType num_rows_per_color = A.getMatrixColoring().getOffsetsRowsPerColor()[i + 1] - color_offset;

                if (num_rows_per_color == 0) { continue; }

                const int num_blocks = min( AMGX_GRID_MAX_SIZE, (int)(num_rows_per_color - 1) / threads_per_block + 1 );
                cudaFuncSetCacheConfig(multicolorGSSmoothCsrKernel_NAIVE_tex<IndexType, ValueTypeA, ValueTypeB>, cudaFuncCachePreferL1);
                multicolorGSSmoothCsrKernel_NAIVE_tex<IndexType, ValueTypeA, ValueTypeB> <<< num_blocks, threads_per_block>>>
                (A_row_offsets_ptr, A_column_indices_ptr, A_diag_ptr, A_nonzero_values_ptr, Dinv_ptr,
                 b_ptr, x_ptr, this->weight, A_sorted_rows_by_color_ptr + color_offset, num_rows_per_color, A.get_num_rows(), x_ptr);
                cudaCheckError();
            }
        }
    }

    cudaCheckError();
}

template <AMGX_VecPrecision t_vecPrec, AMGX_MatPrecision t_matPrec, AMGX_IndPrecision t_indPrec>
void MulticolorGaussSeidelSolver<TemplateConfig<AMGX_device, t_vecPrec, t_matPrec, t_indPrec> >::batch_smooth_1x1_fast(const Matrix_d &A, int batch_sz, const VVector &b, VVector &x)
{
    const IndexType *A_row_offsets_ptr = A.row_offsets.raw();
    const IndexType *A_column_indices_ptr = A.col_indices.raw();
    const ValueTypeA *A_nonzero_values_ptr = A.values.raw();
    const IndexType *A_sorted_rows_by_color_ptr = A.getMatrixColoring().getSortedRowsByColor().raw();
    const IndexType *A_diag_ptr = A.diag.raw();
    const ValueTypeA *Dinv_ptr = this->Dinv.raw();
    const int threads_per_block = 64;
    const int num_colors = this->m_explicit_A->getMatrixColoring().getNumColors();
    const ValueTypeB *b_ptr = b.raw();
    ValueTypeB *x_ptr = x.raw() ;

    for (int i = 0; i < num_colors; i++)
    {
        const IndexType color_offset =  A.getMatrixColoring().getOffsetsRowsPerColor()[i];
        const IndexType num_rows_per_color = A.getMatrixColoring().getOffsetsRowsPerColor()[i + 1] - color_offset;

        if (num_rows_per_color == 0) { continue; }

        const int threads_per_block = 512;
        //const int eightwarps_per_block = threads_per_block/4;
        const int num_blocks = min( AMGX_GRID_MAX_SIZE, (int)(num_rows_per_color - 1) / threads_per_block + 1 );
        cudaFuncSetCacheConfig(multicolorGSSmoothCsrKernel_NAIVE_tex_batched<IndexType, ValueTypeA, ValueTypeB>, cudaFuncCachePreferL1);
        multicolorGSSmoothCsrKernel_NAIVE_tex_batched<IndexType, ValueTypeA, ValueTypeB> <<< num_blocks, threads_per_block>>>
        (A_row_offsets_ptr, A_column_indices_ptr, A_diag_ptr, A_nonzero_values_ptr, Dinv_ptr, batch_sz,
         b_ptr, x_ptr, this->weight, A_sorted_rows_by_color_ptr + color_offset, num_rows_per_color, A.get_num_rows(), x_ptr);
        cudaCheckError();
    }

    if (this->symFlag == 1)
    {
        for (int i = num_colors - 1; i >= 0; i--)
        {
            const IndexType color_offset =  A.getMatrixColoring().getOffsetsRowsPerColor()[i];
            const IndexType num_rows_per_color = A.getMatrixColoring().getOffsetsRowsPerColor()[i + 1] - color_offset;

            if (num_rows_per_color == 0) { continue; }

            const int num_blocks = min( AMGX_GRID_MAX_SIZE, (int)(num_rows_per_color - 1) / threads_per_block + 1 );
            cudaFuncSetCacheConfig(multicolorGSSmoothCsrKernel_NAIVE_tex_batched<IndexType, ValueTypeA, ValueTypeB>, cudaFuncCachePreferL1);
            multicolorGSSmoothCsrKernel_NAIVE_tex_batched<IndexType, ValueTypeA, ValueTypeB> <<< num_blocks, threads_per_block>>>
            (A_row_offsets_ptr, A_column_indices_ptr, A_diag_ptr, A_nonzero_values_ptr, Dinv_ptr, batch_sz,
             b_ptr, x_ptr, this->weight, A_sorted_rows_by_color_ptr + color_offset, num_rows_per_color, A.get_num_rows(), x_ptr);
            cudaCheckError();
        }
    }

    cudaCheckError();
}

template <AMGX_VecPrecision t_vecPrec, AMGX_MatPrecision t_matPrec, AMGX_IndPrecision t_indPrec>
void MulticolorGaussSeidelSolver<TemplateConfig<AMGX_device, t_vecPrec, t_matPrec, t_indPrec> >::smooth_4x4(const Matrix_d &A, const VVector &b, VVector &x, ViewType separation_flag)
{
    const IndexType *A_row_offsets_ptr = A.row_offsets.raw();
    const IndexType *A_column_indices_ptr = A.col_indices.raw();
    const IndexType *A_dia_idx_ptr = A.diag.raw();
    const ValueTypeA *A_nonzero_values_ptr = A.values.raw();
    const ValueTypeB *b_ptr = b.raw();
    const IndexType *A_sorted_rows_by_color_ptr = A.getMatrixColoring().getSortedRowsByColor().raw();
    const ValueTypeA *Dinv_ptr = this->Dinv.raw();
    ValueTypeB *x_ptr = x.raw();
    const int num_colors = this->m_explicit_A->getMatrixColoring().getNumColors();

    for (int i = 0; i < num_colors; i++)
    {
        const IndexType color_offset = ((separation_flag & INTERIOR) == 0) ? A.getMatrixColoring().getSeparationOffsetsRowsPerColor()[i] : A.getMatrixColoring().getOffsetsRowsPerColor()[i];
        const IndexType num_rows_per_color = ((separation_flag == this->m_explicit_A->getViewInterior()) ? A.getMatrixColoring().getSeparationOffsetsRowsPerColor()[i] : A.getMatrixColoring().getOffsetsRowsPerColor()[i + 1]) - color_offset;

        if (num_rows_per_color == 0) { continue; }

        const int threads_per_block = 512;
        const int eightwarps_per_block = threads_per_block / 4;
        const int num_blocks = min( AMGX_GRID_MAX_SIZE, (int)(num_rows_per_color - 1) / eightwarps_per_block + 1 );
        cudaFuncSetCacheConfig(multicolorGSSmooth4by4BlockDiaCsrKernel_NAIVE_tex_readDinv2<IndexType, ValueTypeA, ValueTypeB, eightwarps_per_block, 4, 2, 2>, cudaFuncCachePreferL1);
        multicolorGSSmooth4by4BlockDiaCsrKernel_NAIVE_tex_readDinv2<IndexType, ValueTypeA, ValueTypeB, eightwarps_per_block, 4, 2, 2> <<< num_blocks, threads_per_block>>>
        (A_row_offsets_ptr, A_column_indices_ptr, A_dia_idx_ptr, A_nonzero_values_ptr, Dinv_ptr,
         b_ptr, x_ptr, this->weight, A_sorted_rows_by_color_ptr + color_offset, num_rows_per_color, A.get_num_rows(), x_ptr);
        cudaCheckError();
    }

    if (this->symFlag == 1)
    {
        for (int i = num_colors - 1; i >= 0; i--)
        {
            const IndexType color_offset = ((separation_flag & INTERIOR) == 0) ? A.getMatrixColoring().getSeparationOffsetsRowsPerColor()[i] : A.getMatrixColoring().getOffsetsRowsPerColor()[i];
            const IndexType num_rows_per_color = ((separation_flag == this->m_explicit_A->getViewInterior()) ? A.getMatrixColoring().getSeparationOffsetsRowsPerColor()[i] : A.getMatrixColoring().getOffsetsRowsPerColor()[i + 1]) - color_offset;

            if (num_rows_per_color == 0) { continue; }

            const int threads_per_block = 512;
            const int eightwarps_per_block = threads_per_block / 4;
            const int num_blocks = min( AMGX_GRID_MAX_SIZE, (int)(num_rows_per_color - 1) / eightwarps_per_block + 1 );
            cudaFuncSetCacheConfig(multicolorGSSmooth4by4BlockDiaCsrKernel_NAIVE_tex_readDinv2<IndexType, ValueTypeA, ValueTypeB, eightwarps_per_block, 4, 2, 2>, cudaFuncCachePreferL1);
            multicolorGSSmooth4by4BlockDiaCsrKernel_NAIVE_tex_readDinv2<IndexType, ValueTypeA, ValueTypeB, eightwarps_per_block, 4, 2, 2> <<< num_blocks, threads_per_block>>>
            (A_row_offsets_ptr, A_column_indices_ptr, A_dia_idx_ptr, A_nonzero_values_ptr, Dinv_ptr,
             b_ptr, x_ptr, this->weight, A_sorted_rows_by_color_ptr + color_offset, num_rows_per_color, A.get_num_rows(), x_ptr);
            cudaCheckError();
        } // End of loop over colors
    } // End of if symFlag

    //cudaCheckError();
}

template <AMGX_VecPrecision t_vecPrec, AMGX_MatPrecision t_matPrec, AMGX_IndPrecision t_indPrec>
void MulticolorGaussSeidelSolver<TemplateConfig<AMGX_device, t_vecPrec, t_matPrec, t_indPrec> >::smooth_3x3(const Matrix_d &A, const VVector &b, VVector &x, ViewType separation_flag)
{
    const IndexType *A_row_offsets_ptr = A.row_offsets.raw();
    const IndexType *A_column_indices_ptr = A.col_indices.raw();
    const IndexType *A_dia_idx_ptr = A.diag.raw();
    const ValueTypeA *A_nonzero_values_ptr = A.values.raw();
    const ValueTypeB *b_ptr = b.raw();
    const IndexType *A_sorted_rows_by_color_ptr = A.getMatrixColoring().getSortedRowsByColor().raw();
    const ValueTypeA *Dinv_ptr = this->Dinv.raw();
    ValueTypeB *x_ptr = x.raw();
    const int threads_per_block = 64;
    const int num_colors = this->m_explicit_A->getMatrixColoring().getNumColors();

    for (int i = 0; i < num_colors; i++)
    {
        const IndexType color_offset = ((separation_flag & INTERIOR) == 0) ? A.getMatrixColoring().getSeparationOffsetsRowsPerColor()[i] : A.getMatrixColoring().getOffsetsRowsPerColor()[i];
        const IndexType num_rows_per_color = ((separation_flag == this->m_explicit_A->getViewInterior()) ? A.getMatrixColoring().getSeparationOffsetsRowsPerColor()[i] : A.getMatrixColoring().getOffsetsRowsPerColor()[i + 1]) - color_offset;

        if (num_rows_per_color == 0) { continue; }

        const int threads_per_block = 512;
        const int blockrows_per_cta = threads_per_block / 3;
        const int num_blocks = min( AMGX_GRID_MAX_SIZE, (int) (num_rows_per_color - 1) / blockrows_per_cta + 1);
        cudaFuncSetCacheConfig(multicolorGSSmoothBlockDiaCsrKernel_NAIVE_tex_readDinv2<IndexType, ValueTypeA, ValueTypeB, blockrows_per_cta, 3>, cudaFuncCachePreferL1);
        multicolorGSSmoothBlockDiaCsrKernel_NAIVE_tex_readDinv2<IndexType, ValueTypeA, ValueTypeB, blockrows_per_cta, 3> <<< num_blocks, threads_per_block>>>
        (A_row_offsets_ptr, A_column_indices_ptr, A_dia_idx_ptr, A_nonzero_values_ptr, Dinv_ptr,
         b_ptr, x_ptr, this->weight, A_sorted_rows_by_color_ptr + color_offset, num_rows_per_color, A.get_num_rows(), x_ptr);
        cudaCheckError();
    }

    if (this->symFlag == 1)
    {
        for (int i = num_colors - 1; i >= 0; i--)
        {
            const IndexType color_offset = ((separation_flag & INTERIOR) == 0) ? A.getMatrixColoring().getSeparationOffsetsRowsPerColor()[i] : A.getMatrixColoring().getOffsetsRowsPerColor()[i];
            const IndexType num_rows_per_color = ((separation_flag == this->m_explicit_A->getViewInterior()) ? A.getMatrixColoring().getSeparationOffsetsRowsPerColor()[i] : A.getMatrixColoring().getOffsetsRowsPerColor()[i + 1]) - color_offset;

            if (num_rows_per_color == 0) { continue; }

            const int blockrows_per_cta = threads_per_block / 3;
            const int num_blocks = min( AMGX_GRID_MAX_SIZE, (int) (num_rows_per_color - 1) / blockrows_per_cta + 1);
            cudaFuncSetCacheConfig(multicolorGSSmoothBlockDiaCsrKernel_NAIVE_tex_readDinv2<IndexType, ValueTypeA, ValueTypeB, blockrows_per_cta, 3>, cudaFuncCachePreferL1);
            multicolorGSSmoothBlockDiaCsrKernel_NAIVE_tex_readDinv2<IndexType, ValueTypeA, ValueTypeB, blockrows_per_cta, 3> <<< num_blocks, threads_per_block>>>
            (A_row_offsets_ptr, A_column_indices_ptr, A_dia_idx_ptr, A_nonzero_values_ptr, Dinv_ptr,
             b_ptr, x_ptr, this->weight, A_sorted_rows_by_color_ptr + color_offset, num_rows_per_color, A.get_num_rows(), x_ptr);
            cudaCheckError();
        } // End of loop over colors
    } // End of if symFlag
}

template <AMGX_VecPrecision t_vecPrec, AMGX_MatPrecision t_matPrec, AMGX_IndPrecision t_indPrec>
void MulticolorGaussSeidelSolver<TemplateConfig<AMGX_device, t_vecPrec, t_matPrec, t_indPrec> >::smooth_BxB(Matrix_d &A, VVector &b, VVector &x, ViewType separation_flag)
{
    const int num_colors = this->m_explicit_A->getMatrixColoring().getNumColors();
    int x_dirty_status = x.dirtybit;
    int x_delayed = x.delayed_send;
    int b_delayed = b.delayed_send;
    VVector y(b.size());
    y.set_block_dimx(b.get_block_dimx());
    y.set_block_dimy(b.get_block_dimy());
    y.tag = this->tag * 100 + 3;
    thrust::copy(b.begin(), b.end(), y.begin());
    cudaCheckError();
    y.dirtybit = 0;
    y.delayed_send = 1;
    x.delayed_send = 1;
    b.delayed_send = 1;

    for (int color = 0; color < num_colors; color++)
    {
        Cusparse::bsrmv(Cusparse::ALL_COLORS, color, (ValueTypeA) - 1.0f, A, x, (ValueTypeA)1.0f, y, separation_flag); // y= -A.x + y      for current color

        if (separation_flag == A.getViewExterior()) { y.dirtybit = 0; }

        Cusparse::bsrmv(color, (ValueTypeA)this->weight, A, this->Dinv, y, (ValueTypeA)1.0f, x, separation_flag);                // x= w*Dinv.y + x  for current color

        if (separation_flag == A.getViewExterior()) { x.dirtybit = 0; }
    }

    if (this->symFlag)
    {
        thrust::copy(b.begin(), b.end(), y.begin());

        for (int color = num_colors - 1; color >= 0; color--)
        {
            Cusparse::bsrmv(Cusparse::ALL_COLORS, color, (ValueTypeA) - 1.0f, A, x, (ValueTypeA)1.0f, y, separation_flag); // y= -A.x + y      for current color

            if (separation_flag == A.getViewExterior()) { y.dirtybit = 0; }

            Cusparse::bsrmv(color, (ValueTypeA)this->weight, A, this->Dinv, y, (ValueTypeA)1.0f, x, separation_flag);                // x= w*Dinv.y + x  for current color

            if (separation_flag == A.getViewExterior()) { x.dirtybit = 0; }
        }

        cudaCheckError();
    }

    x.delayed_send = x_delayed;
    b.delayed_send = b_delayed;
    x.dirtybit = x_dirty_status;
}

// Solve one iteration
template<class T_Config>
bool
MulticolorGaussSeidelSolver_Base<T_Config>::solve_iteration( VVector &b, VVector &x, bool xIsZero )
{
    if (xIsZero) { x.dirtybit = 0; }

    if (!this->m_explicit_A->is_matrix_singleGPU())
    {
        this->m_explicit_A->manager->exchange_halo_async(x, x.tag);
        this->m_explicit_A->manager->exchange_halo_async(b, b.tag);
    }

    if (this->m_explicit_A->getViewExterior() == this->m_explicit_A->getViewInterior())
    {
        if (!this->m_explicit_A->is_matrix_singleGPU())
        {
            this->m_explicit_A->manager->exchange_halo_wait(x, x.tag);
            this->m_explicit_A->manager->exchange_halo_wait(b, b.tag);
        }
    }

    ViewType oldView = this->m_explicit_A->currentView();
    bool latencyHiding = true;
    ViewType flags;

    if (this->m_explicit_A->is_matrix_singleGPU() || (x.dirtybit == 0 && b.dirtybit == 0))
    {
        latencyHiding = false;
        this->m_explicit_A->setViewExterior();
        flags = this->m_explicit_A->getViewExterior();
    }
    else
    {
        this->m_explicit_A->setViewInterior();
        flags = this->m_explicit_A->getViewInterior();
    }

    if (xIsZero)
    {
        thrust::fill(x.begin(), x.end(), 0.);
        cudaCheckError();
    }

    if ( this->m_explicit_A->get_block_dimx() == 1 && this->m_explicit_A->get_block_dimy() == 1 )
    {
        if (this->use_bsrxmv)
        {
            smooth_BxB(*this->m_explicit_A, b, x, flags);
        }
        else
        {
            smooth_1x1(*this->m_explicit_A, b, x, flags);
        }
    }
    else if ( this->m_explicit_A->get_block_dimx() == 3 && this->m_explicit_A->get_block_dimy() == 3 )
    {
        //if (this->use_bsrxmv)
        smooth_BxB(*this->m_explicit_A, b, x, flags);
        //else
        //  smooth_3x3(*this->m_explicit_A, b, x, flags);
    }
    else if ( this->m_explicit_A->get_block_dimx() == 4 && this->m_explicit_A->get_block_dimy() == 4 )
    {
        if (this->use_bsrxmv)
        {
            smooth_BxB(*this->m_explicit_A, b, x, flags);
        }
        else
        {
            smooth_4x4(*this->m_explicit_A, b, x, flags);
        }
    }
    else if (this->m_explicit_A->get_block_dimx() == this->m_explicit_A->get_block_dimy())
    {
        smooth_BxB(*this->m_explicit_A, b, x, flags);
    }
    else
    {
        FatalError("Unsupported block size for MulticolorGaussSeidelSolver smooth", AMGX_ERR_NOT_SUPPORTED_BLOCKSIZE);
    }

    if (latencyHiding)
    {
        if (!this->m_explicit_A->is_matrix_singleGPU())
        {
            this->m_explicit_A->manager->exchange_halo_wait(x, x.tag);
            this->m_explicit_A->manager->exchange_halo_wait(b, b.tag);
        }

        this->m_explicit_A->setViewExterior();
        flags = (ViewType)(~(this->m_explicit_A->getViewInterior()) & this->m_explicit_A->getViewExterior());

        if (flags != 0)
        {
            if ( this->m_explicit_A->get_block_dimx() == 1 && this->m_explicit_A->get_block_dimy() == 1 )
            {
                if (this->use_bsrxmv)
                {
                    smooth_BxB(*this->m_explicit_A, b, x, flags);
                }
                else
                {
                    smooth_1x1(*this->m_explicit_A, b, x, flags);
                }
            }
            else if ( this->m_explicit_A->get_block_dimx() == 3 && this->m_explicit_A->get_block_dimy() == 3 )
            {
                smooth_BxB(*this->m_explicit_A, b, x, flags);
            }
            else if ( this->m_explicit_A->get_block_dimx() == 4 && this->m_explicit_A->get_block_dimy() == 4 )
            {
                if (this->use_bsrxmv)
                {
                    smooth_BxB(*this->m_explicit_A, b, x, flags);
                }
                else
                {
                    smooth_4x4(*this->m_explicit_A, b, x, flags);
                }
            }
            else if (this->m_explicit_A->get_block_dimx() == this->m_explicit_A->get_block_dimy())
            {
                smooth_BxB(*this->m_explicit_A, b, x, flags);
            }
            else
            {
                FatalError("Unsupported block size for MulticolorGaussSeidelSolver smooth", AMGX_ERR_NOT_SUPPORTED_BLOCKSIZE);
            }
        }
    }

    x.dirtybit = 1;
    //if (!this->m_explicit_A->is_matrix_singleGPU() && x.delayed_send==0)
    //  this->m_explicit_A->manager->exchange_halo_async(x, x.tag);
    this->m_explicit_A->setView(oldView);
    return this->converged( b, x );
}

template<class T_Config>
void
MulticolorGaussSeidelSolver_Base<T_Config>::solve_finalize( VVector &b, VVector &x )
{}

template<class T_Config>
cudaStream_t
MulticolorGaussSeidelSolver_Base<T_Config>::get_aux_stream()
{
    static std::shared_ptr<Stream> aux_stream;
    if (!aux_stream)
    {
        aux_stream = std::make_shared<Stream>();
    }

    return aux_stream->get();
}


/****************************************
 * Explict instantiations
 ***************************************/
#define AMGX_CASE_LINE(CASE) template class MulticolorGaussSeidelSolver_Base<TemplateMode<CASE>::Type>;
AMGX_FORALL_BUILDS(AMGX_CASE_LINE)
#undef AMGX_CASE_LINE

#define AMGX_CASE_LINE(CASE) template class MulticolorGaussSeidelSolver<TemplateMode<CASE>::Type>;
AMGX_FORALL_BUILDS(AMGX_CASE_LINE)
#undef AMGX_CASE_LINE

}
} // namespace amgx
