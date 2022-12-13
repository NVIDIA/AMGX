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
#include <solvers/fixcolor_gauss_seidel_solver.h>
#include <solvers/block_common_solver.h>
#include <gaussian_elimination.h>
#include <basic_types.h>
#include <util.h>
#include <texture.h>

namespace amgx
{
namespace fixcolor_gauss_seidel_solver
{

// -------------------------
//  Kernels
// -------------------------

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
    volatile __shared__ ValueTypeA s_Amat[bsize_sq * blockrows_per_cta ];
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
    volatile __shared__ ValueTypeA s_Amat[bsize_sq * halfwarps_per_block ];
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

template<typename IndexType, typename ValueTypeA,  int threads_per_block, int halfwarps_per_block>
__global__
void setupBlockGSSmoothbBybBlockDiaCsrKernel(const IndexType *row_offsets, const IndexType *column_indices, const ValueTypeA *values, const IndexType *dia_indices, ValueTypeA *Dinv, const int num_block_rows, int bsize, int bsize_sq, ValueTypeA *temp1)

{
    const int tid = blockDim.x * blockIdx.x + threadIdx.x;
    int halfwarp_id = tid >> 4;
    const int block_halfwarp_id = threadIdx.x >> 4;
    const int mat_entry_index = threadIdx.x & (16 - 1);
    const int i_ind = mat_entry_index >> 2;
    const int j_ind = mat_entry_index & 3;
    extern __shared__ volatile char sharedc[];
    volatile ValueTypeA *s_Amat;
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

template<typename IndexType, typename ValueTypeA, int blockrows_per_cta, int bsize, int bsize_sq>
__global__
void setupBlockGSSmoothBlockDiaCsrKernel_V2(const IndexType *row_offsets, const IndexType *column_indices, const ValueTypeA *values, const IndexType *dia_indices,
        ValueTypeA *Dinv, const int num_block_rows)
{
    int cta_blockrow_id = threadIdx.x / bsize_sq;
    int blockrow_id = blockIdx.x * blockrows_per_cta + cta_blockrow_id;
    const int mat_entry_index = threadIdx.x - cta_blockrow_id * bsize_sq;
    const int i_ind = mat_entry_index / bsize;
    const int j_ind = mat_entry_index - i_ind * bsize;
    volatile __shared__ ValueTypeA s_Amat[bsize_sq * blockrows_per_cta];
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
#define s_A(ROW,COL)   s_Amat[s_offset+ROW*bsize+COL]
        ValueTypeA diag;

        for (int row = 0; row < bsize; row++)
        {
            diag = 1.0 / s_A(row, row);

            if ((i_ind == 0) && !(j_ind == row))
            {
                s_A(row, j_ind) = s_A(row, j_ind) * diag;
            }

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
        const ValueTypeB *b, const ValueTypeB *x, ValueTypeB weight, const int num_rows_per_line, const int color_num, const int num_block_rows, ValueTypeB *xout)
{
    const int vec_entry_index = threadIdx.x & (bsize - 1);
    const int block_eighthwarp_id = threadIdx.x >> log_bsize;
    const int tid = blockDim.x * blockIdx.x + threadIdx.x;
    int eighthwarp_id = tid >> log_bsize;
    volatile __shared__ ValueTypeB s_xtemp[ bsize * eighthwarps_per_block ];
    ValueTypeB bmAx, xin;
    ValueTypeB temp[bsize];
    int offset, i, s_offset;
    int n1d2 =  num_rows_per_line * num_rows_per_line;
    int pln, tid2, row, col;
    int idx_1 = color_num & 1;
    int idx_2 = (color_num & 2) >> 1;
    int idx_3 = (color_num & 4) >> 2;

    while (true)
    {
        pln = eighthwarp_id / n1d2;
        tid2 = eighthwarp_id % n1d2;
        row = tid2 / num_rows_per_line;
        col = tid2 % num_rows_per_line;
        i = 4 * n1d2 * (2 * pln + idx_1) + 2 * num_rows_per_line * (2 * row + idx_2) + 2 * col + idx_3;

        //if (pln >= num_rows_per_line) return;
        //if (row >= num_rows_per_line) return;
        if (i > num_block_rows) { return; }

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
void fixcolorGSSmoothCsrKernel_NAIVE_tex(const IndexType *row_offsets, const IndexType *column_indices, const IndexType *diag, const ValueTypeA *nonzero_values, const ValueTypeA *Dinv,
        const ValueTypeB *b, const ValueTypeB *x, ValueTypeB weight, const int num_rows_per_line, const int color_num, const int num_block_rows, ValueTypeB *xout)
{
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    int n1d2 =  num_rows_per_line * num_rows_per_line;
    int pln, tid2, row, col;
    int idx_1 = color_num & 1;
    int idx_2 = (color_num & 2) >> 1;
    int idx_3 = (color_num & 4) >> 2;
    int i;
    ValueTypeB bmAx, xin;
    ValueTypeB temp, s_xtemp;
    ValueTypeB dia;

    while (true)
    {
        pln = tid / n1d2;
        tid2 = tid % n1d2;
        row = tid2 / num_rows_per_line;
        col = tid2 % num_rows_per_line;
        i = 4 * n1d2 * (2 * pln + idx_1) + 2 * num_rows_per_line * (2 * row + idx_2) + 2 * col + idx_3;

        //if (pln >= num_rows_per_line) return;
        //if (row >= num_rows_per_line) return;
        if (i > num_block_rows) { return; }

        bmAx = b[i];
        xin = x[i];
        dia = nonzero_values[diag[i]];
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


// -------------------
// Methods
// -------------------


// Constructor
template<class T_Config>
FixcolorGaussSeidelSolver_Base<T_Config>::FixcolorGaussSeidelSolver_Base( AMG_Config &cfg, const std::string &cfg_scope) : Solver<T_Config>( cfg, cfg_scope)
{
    this->weight = cfg.AMG_Config::getParameter<double>("relaxation_factor", cfg_scope);
    symFlag = cfg.AMG_Config::getParameter<int>("symmetric_GS", cfg_scope);

    if (cfg.AMG_Config::getParameter<int>("use_bsrxmv", cfg_scope))
    {
        this->use_bsrxmv = 1;
    }
    else
    {
        this->use_bsrxmv = 0;
    }

    if (weight == 0)
    {
        weight = 1.;
        amgx_printf("Warning, setting weight to 1 instead of estimating largest_eigen_value in Multicolor GaussSeidel smoother\n");
    }

    num_colors = 8;
}

// Destructor
template<class T_Config>
FixcolorGaussSeidelSolver_Base<T_Config>::~FixcolorGaussSeidelSolver_Base()
{
}

template<class T_Config>
void
FixcolorGaussSeidelSolver_Base<T_Config>::printSolverParameters() const
{
    std::cout << "relaxation_factor= " << this->weight << std::endl;
    std::cout << "use_bsrxmv = " << this->use_bsrxmv << std::endl;
    std::cout << "symmetric_GS = " << this->symFlag << std::endl;
}


template<class T_Config>
void FixcolorGaussSeidelSolver_Base<T_Config>::computeDinv(Matrix<T_Config> &A)
{
    Matrix<T_Config> *A_as_matrix = (Matrix<T_Config> *) this->m_A;
    ViewType oldView = A.currentView();
    A.setView(A_as_matrix->getViewExterior());

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
        FatalError("Unsupported block size for FixcolorGaussSeidelSolver computeEinv", AMGX_ERR_NOT_SUPPORTED_BLOCKSIZE);
    }

    A.setView(oldView);
}

// Method to compute the diagonal matrix E in DILU smoother
template <AMGX_VecPrecision t_vecPrec, AMGX_MatPrecision t_matPrec, AMGX_IndPrecision t_indPrec>
void FixcolorGaussSeidelSolver<TemplateConfig<AMGX_host, t_vecPrec, t_matPrec, t_indPrec> >::computeDinv_3x3(const Matrix_h &A)
{
    FatalError("Multicolor GS smoother not implemented with host format, exiting", AMGX_ERR_NOT_SUPPORTED_BLOCKSIZE);
}
template <AMGX_VecPrecision t_vecPrec, AMGX_MatPrecision t_matPrec, AMGX_IndPrecision t_indPrec>
void FixcolorGaussSeidelSolver<TemplateConfig<AMGX_host, t_vecPrec, t_matPrec, t_indPrec> >::computeDinv_4x4(const Matrix_h &A)
{
    FatalError("Multicolor GS smoother not implemented with host format, exiting", AMGX_ERR_NOT_SUPPORTED_BLOCKSIZE);
}

template <AMGX_VecPrecision t_vecPrec, AMGX_MatPrecision t_matPrec, AMGX_IndPrecision t_indPrec>
void FixcolorGaussSeidelSolver<TemplateConfig<AMGX_host, t_vecPrec, t_matPrec, t_indPrec> >::computeDinv_2x2(const Matrix_h &A)
{
    FatalError("Multicolor GS smoother not implemented with host format, exiting", AMGX_ERR_NOT_SUPPORTED_BLOCKSIZE);
}

template <AMGX_VecPrecision t_vecPrec, AMGX_MatPrecision t_matPrec, AMGX_IndPrecision t_indPrec>
void FixcolorGaussSeidelSolver<TemplateConfig<AMGX_host, t_vecPrec, t_matPrec, t_indPrec> >::computeDinv_5x5(const Matrix_h &A)
{
    FatalError("Multicolor GS smoother not implemented with host format, exiting", AMGX_ERR_NOT_SUPPORTED_BLOCKSIZE);
}

template <AMGX_VecPrecision t_vecPrec, AMGX_MatPrecision t_matPrec, AMGX_IndPrecision t_indPrec>
void FixcolorGaussSeidelSolver<TemplateConfig<AMGX_host, t_vecPrec, t_matPrec, t_indPrec> >::computeDinv_1x1(const Matrix_h &A)
{
    FatalError("Multicolor GS smoother not implemented with host format, exiting", AMGX_ERR_NOT_SUPPORTED_BLOCKSIZE);
}

template <AMGX_VecPrecision t_vecPrec, AMGX_MatPrecision t_matPrec, AMGX_IndPrecision t_indPrec>
void FixcolorGaussSeidelSolver<TemplateConfig<AMGX_host, t_vecPrec, t_matPrec, t_indPrec> >::computeDinv_bxb(const Matrix_h &A, const int bsize)
{
    FatalError("Multicolor GS smoother not implemented with host format, exiting", AMGX_ERR_NOT_SUPPORTED_BLOCKSIZE);
}


template <AMGX_VecPrecision t_vecPrec, AMGX_MatPrecision t_matPrec, AMGX_IndPrecision t_indPrec>
void FixcolorGaussSeidelSolver<TemplateConfig<AMGX_device, t_vecPrec, t_matPrec, t_indPrec> >::computeDinv_4x4(const Matrix_d &A)
{
//both DIAG supported
    this->Dinv.resize(A.get_num_rows()*A.get_block_dimx()*A.get_block_dimy(), 0.0);
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
void FixcolorGaussSeidelSolver<TemplateConfig<AMGX_device, t_vecPrec, t_matPrec, t_indPrec> >::computeDinv_bxb(const Matrix_d &A, const int bsize)
{
//both DIAG supported
    this->Dinv.resize(A.get_num_rows()*A.get_block_dimx()*A.get_block_dimy(), 0.0);
    const IndexType *A_row_offsets_ptr = A.row_offsets.raw();
    const IndexType *A_column_indices_ptr = A.col_indices.raw();
    const IndexType *A_dia_idx_ptr = A.diag.raw();
    ValueTypeA *Dinv_ptr = this->Dinv.raw();
    const ValueTypeA *A_nonzero_values_ptr = A.values.raw();
    MVector temp(AMGX_GRID_MAX_SIZE * ((bsize - 1) / 4 + 1) * ((bsize - 1) / 4 + 1));
    ValueTypeA *temp_ptr = temp.raw();
    // MUST BE MULTIPLE OF 16
    const int threads_per_block = 512;
    const int halfwarps_per_block = threads_per_block / 16;
    const int num_blocks = min(AMGX_GRID_MAX_SIZE, (int) (A.get_num_rows() - 1) / halfwarps_per_block + 1);
    cudaFuncSetCacheConfig(setupBlockGSSmoothbBybBlockDiaCsrKernel<IndexType, ValueTypeA, threads_per_block, halfwarps_per_block>, cudaFuncCachePreferL1);
    setupBlockGSSmoothbBybBlockDiaCsrKernel<IndexType, ValueTypeA, threads_per_block, halfwarps_per_block> <<< num_blocks, threads_per_block, sizeof(ValueTypeA)*bsize *bsize *halfwarps_per_block>>>
    (A_row_offsets_ptr, A_column_indices_ptr, A_nonzero_values_ptr, A_dia_idx_ptr, Dinv_ptr, A.get_num_rows(), bsize, bsize * bsize, temp_ptr);
    cudaCheckError();
}

template <AMGX_VecPrecision t_vecPrec, AMGX_MatPrecision t_matPrec, AMGX_IndPrecision t_indPrec>
void FixcolorGaussSeidelSolver<TemplateConfig<AMGX_device, t_vecPrec, t_matPrec, t_indPrec> >::computeDinv_3x3(const Matrix_d &A)
{
//both DIAG supported
    this->Dinv.resize(A.get_num_rows()*A.get_block_dimx()*A.get_block_dimy(), 0.0);
    const IndexType *A_row_offsets_ptr = A.row_offsets.raw();
    const IndexType *A_column_indices_ptr = A.col_indices.raw();
    const IndexType *A_dia_idx_ptr = A.diag.raw();
    ValueTypeA *Dinv_ptr = this->Dinv.raw();
    const ValueTypeA *A_nonzero_values_ptr = A.values.raw();
    // MUST BE MULTIPLE OF 16
    const int threads_per_block = 256;
    const int blockrows_per_cta = threads_per_block / 9;
    const int num_blocks = min(AMGX_GRID_MAX_SIZE, (int) (A.get_num_rows() - 1) / blockrows_per_cta + 1);
    cudaFuncSetCacheConfig(setupBlockGSSmooth3by3BlockDiaCsrKernel < IndexType, ValueTypeA, blockrows_per_cta, 32 / 9, 3, 9 >, cudaFuncCachePreferL1);
    setupBlockGSSmooth3by3BlockDiaCsrKernel < IndexType, ValueTypeA, blockrows_per_cta, 32 / 9, 3, 9 > <<< num_blocks, threads_per_block>>>
    (A_row_offsets_ptr, A_column_indices_ptr, A_nonzero_values_ptr, A_dia_idx_ptr, Dinv_ptr, A.get_num_rows());
    cudaCheckError();
}

template <AMGX_VecPrecision t_vecPrec, AMGX_MatPrecision t_matPrec, AMGX_IndPrecision t_indPrec>
void FixcolorGaussSeidelSolver<TemplateConfig<AMGX_device, t_vecPrec, t_matPrec, t_indPrec> >::computeDinv_2x2(const Matrix_d &A)
{
//both DIAG supported
    this->Dinv.resize(A.get_num_rows()*A.get_block_dimx()*A.get_block_dimy(), 0.0);
    const IndexType *A_row_offsets_ptr = A.row_offsets.raw();
    const IndexType *A_column_indices_ptr = A.col_indices.raw();
    const IndexType *A_dia_idx_ptr = A.diag.raw();
    ValueTypeA *Dinv_ptr = this->Dinv.raw();
    const ValueTypeA *A_nonzero_values_ptr = A.values.raw();
    // MUST BE MULTIPLE OF 16
    const int threads_per_block = 256;
    const int blockrows_per_cta = threads_per_block / 4;
    const int num_blocks = min(AMGX_GRID_MAX_SIZE, (int) (A.get_num_rows() - 1) / blockrows_per_cta + 1);
    cudaFuncSetCacheConfig(setupBlockGSSmooth3by3BlockDiaCsrKernel < IndexType, ValueTypeA, blockrows_per_cta, 32 / 4, 2, 4 >, cudaFuncCachePreferL1);
    setupBlockGSSmooth3by3BlockDiaCsrKernel < IndexType, ValueTypeA, blockrows_per_cta, 32 / 4, 2, 4 > <<< num_blocks, threads_per_block>>>
    (A_row_offsets_ptr, A_column_indices_ptr, A_nonzero_values_ptr, A_dia_idx_ptr, Dinv_ptr, A.get_num_rows());
    cudaCheckError();
}

template <AMGX_VecPrecision t_vecPrec, AMGX_MatPrecision t_matPrec, AMGX_IndPrecision t_indPrec>
void FixcolorGaussSeidelSolver<TemplateConfig<AMGX_device, t_vecPrec, t_matPrec, t_indPrec> >::computeDinv_5x5(const Matrix_d &A)
{
//both DIAG supported
    this->Dinv.resize(A.get_num_rows()*A.get_block_dimx()*A.get_block_dimy(), 0.0);
    const IndexType *A_row_offsets_ptr = A.row_offsets.raw();
    const IndexType *A_column_indices_ptr = A.col_indices.raw();
    const IndexType *A_dia_idx_ptr = A.diag.raw();
    ValueTypeA *Dinv_ptr = this->Dinv.raw();
    const ValueTypeA *A_nonzero_values_ptr = A.values.raw();
    // MUST BE MULTIPLE OF 16
    const int threads_per_block = 256;
    const int blockrows_per_cta = threads_per_block / 25;
    const int num_blocks = min(AMGX_GRID_MAX_SIZE, (int) (A.get_num_rows() - 1) / blockrows_per_cta + 1);
    cudaFuncSetCacheConfig(setupBlockGSSmooth3by3BlockDiaCsrKernel < IndexType, ValueTypeA, blockrows_per_cta, 32 / 25, 5, 25 >, cudaFuncCachePreferL1);
    setupBlockGSSmooth3by3BlockDiaCsrKernel < IndexType, ValueTypeA, blockrows_per_cta, 32 / 25, 5, 25 > <<< num_blocks, threads_per_block>>>
    (A_row_offsets_ptr, A_column_indices_ptr, A_nonzero_values_ptr, A_dia_idx_ptr, Dinv_ptr, A.get_num_rows());
    cudaCheckError();
}

template <AMGX_VecPrecision t_vecPrec, AMGX_MatPrecision t_matPrec, AMGX_IndPrecision t_indPrec>
void FixcolorGaussSeidelSolver<TemplateConfig<AMGX_device, t_vecPrec, t_matPrec, t_indPrec> >::computeDinv_1x1(const Matrix_d &A)
{
}

// Solver setup
template<class T_Config>
void
FixcolorGaussSeidelSolver_Base<T_Config>::solver_setup(bool reuse_matrix_structure)
{
    Matrix<T_Config> *A_as_matrix = dynamic_cast<Matrix<T_Config>*>(Base::m_A);

    if (!A_as_matrix)
    {
        FatalError("FixcolorGaussSeidelSolver only works with explicit matrices", AMGX_ERR_INTERNAL);
    }

    if (!this->use_bsrxmv)
    {
        FatalError("Use bsrxmv implementation, old implementation is buggy for diagonal matrix", AMGX_ERR_NOT_IMPLEMENTED);
    }
    computeDinv(*A_as_matrix);
}

//
template<class T_Config>
void
FixcolorGaussSeidelSolver_Base<T_Config>::solve_init( VVector &b, VVector &x, bool xIsZero )
{
}

template <AMGX_VecPrecision t_vecPrec, AMGX_MatPrecision t_matPrec, AMGX_IndPrecision t_indPrec>
void FixcolorGaussSeidelSolver<TemplateConfig<AMGX_host, t_vecPrec, t_matPrec, t_indPrec> >::smooth_1x1(const Matrix_h &A, const VVector &b, VVector &x, ViewType separation_flag)
{
    FatalError("Haven't implemented Block Multicolor Gauss-Seidel smoother for host format", AMGX_ERR_NOT_SUPPORTED_TARGET);
}

template <AMGX_VecPrecision t_vecPrec, AMGX_MatPrecision t_matPrec, AMGX_IndPrecision t_indPrec>
void FixcolorGaussSeidelSolver<TemplateConfig<AMGX_host, t_vecPrec, t_matPrec, t_indPrec> >::smooth_3x3(const Matrix_h &A, const VVector &b, VVector &x, ViewType separation_flag)
{
    FatalError("Haven't implemented Block Multicolor Gauss-Seidel smoother for host format", AMGX_ERR_NOT_SUPPORTED_TARGET);
}

template <AMGX_VecPrecision t_vecPrec, AMGX_MatPrecision t_matPrec, AMGX_IndPrecision t_indPrec>
void FixcolorGaussSeidelSolver<TemplateConfig<AMGX_host, t_vecPrec, t_matPrec, t_indPrec> >::smooth_4x4(const Matrix_h &A, const VVector &b, VVector &x, ViewType separation_flag)
{
    FatalError("Haven't implemented Block Multicolor Gauss-Seidel smoother for host format", AMGX_ERR_NOT_SUPPORTED_TARGET);
}

template <AMGX_VecPrecision t_vecPrec, AMGX_MatPrecision t_matPrec, AMGX_IndPrecision t_indPrec>
void FixcolorGaussSeidelSolver<TemplateConfig<AMGX_host, t_vecPrec, t_matPrec, t_indPrec> >::smooth_BxB(Matrix_h &A, VVector &b, VVector &x, ViewType separation_flag)
{
    FatalError("Haven't implemented Block Multicolor Gauss-Seidel smoother for host format", AMGX_ERR_NOT_SUPPORTED_TARGET);
}

template <AMGX_VecPrecision t_vecPrec, AMGX_MatPrecision t_matPrec, AMGX_IndPrecision t_indPrec>
void FixcolorGaussSeidelSolver<TemplateConfig<AMGX_device, t_vecPrec, t_matPrec, t_indPrec> >::smooth_1x1(const Matrix_d &A, const VVector &b, VVector &x, ViewType separation_flag)
{
    const IndexType *A_row_offsets_ptr = A.row_offsets.raw();
    const IndexType *A_column_indices_ptr = A.col_indices.raw();
    const ValueTypeA *A_nonzero_values_ptr = A.values.raw();
    const ValueTypeB *b_ptr = b.raw();
    const IndexType *A_diag_ptr = A.diag.raw();
    const ValueTypeA *Dinv_ptr = this->Dinv.raw();
    ValueTypeB *x_ptr = x.raw();
    int n = A.get_num_rows();
    int num_rows_per_line;
    num_rows_per_line = (int) ceil(cbrt((double)n) / 2);

    for (int i = 0; i < this->num_colors; i++)
    {
        int num_rows_per_color = n / this->num_colors + 1;
        const int threads_per_block = 512;
        //const int eightwarps_per_block = threads_per_block/4;
        const int num_blocks = min( AMGX_GRID_MAX_SIZE, (int)(num_rows_per_color - 1) / threads_per_block + 1 );
        cudaFuncSetCacheConfig(fixcolorGSSmoothCsrKernel_NAIVE_tex<IndexType, ValueTypeA, ValueTypeB>, cudaFuncCachePreferL1);
        fixcolorGSSmoothCsrKernel_NAIVE_tex<IndexType, ValueTypeA, ValueTypeB> <<< num_blocks, threads_per_block>>>
        (A_row_offsets_ptr, A_column_indices_ptr, A_diag_ptr, A_nonzero_values_ptr, Dinv_ptr,
         b_ptr, x_ptr, this->weight, num_rows_per_line, i, n, x_ptr);
        cudaCheckError();
    }

    if (this->symFlag == 1)
    {
        int num_rows_per_color = n / this->num_colors + 1;

        for (int i = this->num_colors - 1; i >= 0; i--)
        {
            const int threads_per_block = 512;
            const int num_blocks = min( AMGX_GRID_MAX_SIZE, (int)(num_rows_per_color - 1) / threads_per_block + 1 );
            cudaFuncSetCacheConfig(fixcolorGSSmoothCsrKernel_NAIVE_tex<IndexType, ValueTypeA, ValueTypeB>, cudaFuncCachePreferL1);
            fixcolorGSSmoothCsrKernel_NAIVE_tex<IndexType, ValueTypeA, ValueTypeB> <<< num_blocks, threads_per_block>>>
            (A_row_offsets_ptr, A_column_indices_ptr, A_diag_ptr, A_nonzero_values_ptr, Dinv_ptr,
             b_ptr, x_ptr, this->weight, num_rows_per_line, i, n, x_ptr);
        }
    }

    cudaCheckError();
}

template <AMGX_VecPrecision t_vecPrec, AMGX_MatPrecision t_matPrec, AMGX_IndPrecision t_indPrec>
void FixcolorGaussSeidelSolver<TemplateConfig<AMGX_device, t_vecPrec, t_matPrec, t_indPrec> >::smooth_4x4(const Matrix_d &A, const VVector &b, VVector &x, ViewType separation_flag)
{
    const IndexType *A_row_offsets_ptr = A.row_offsets.raw();
    const IndexType *A_column_indices_ptr = A.col_indices.raw();
    const IndexType *A_dia_idx_ptr = A.diag.raw();
    const ValueTypeA *A_nonzero_values_ptr = A.values.raw();
    const ValueTypeB *b_ptr = b.raw();
    //const IndexType *A_sorted_rows_by_color_ptr = A.getMatrixColoring().getSortedRowsByColor().raw();
    const ValueTypeA *Dinv_ptr = this->Dinv.raw();
    ValueTypeB *x_ptr = x.raw();
    int n = A.get_num_rows();
    int num_rows_per_line;
    num_rows_per_line = (int) ceil(cbrt((double)n) / 2);

    for (int i = 0; i < this->num_colors; i++)
    {
        const IndexType num_rows_per_color = n / this->num_colors + 1;
        const int threads_per_block = 512;
        const int eightwarps_per_block = threads_per_block / 4;
        const int num_blocks = min( AMGX_GRID_MAX_SIZE, (int)(num_rows_per_color - 1) / eightwarps_per_block + 1 );
        cudaFuncSetCacheConfig(multicolorGSSmooth4by4BlockDiaCsrKernel_NAIVE_tex_readDinv2<IndexType, ValueTypeA, ValueTypeB, eightwarps_per_block, 4, 2, 2>, cudaFuncCachePreferL1);
        multicolorGSSmooth4by4BlockDiaCsrKernel_NAIVE_tex_readDinv2<IndexType, ValueTypeA, ValueTypeB, eightwarps_per_block, 4, 2, 2> <<< num_blocks, threads_per_block>>>
        (A_row_offsets_ptr, A_column_indices_ptr, A_dia_idx_ptr, A_nonzero_values_ptr, Dinv_ptr,
         b_ptr, x_ptr, this->weight, num_rows_per_line, i, n, x_ptr);
        cudaCheckError();
    }

    if (this->symFlag == 1)
    {
        for (int i = this->num_colors - 1; i >= 0; i--)
        {
            const IndexType num_rows_per_color = n / this->num_colors + 1;
            const int threads_per_block = 512;
            const int eightwarps_per_block = threads_per_block / 4;
            const int num_blocks = min( AMGX_GRID_MAX_SIZE, (int)(num_rows_per_color - 1) / eightwarps_per_block + 1 );
            cudaFuncSetCacheConfig(multicolorGSSmooth4by4BlockDiaCsrKernel_NAIVE_tex_readDinv2<IndexType, ValueTypeA, ValueTypeB, eightwarps_per_block, 4, 2, 2>, cudaFuncCachePreferL1);
            multicolorGSSmooth4by4BlockDiaCsrKernel_NAIVE_tex_readDinv2<IndexType, ValueTypeA, ValueTypeB, eightwarps_per_block, 4, 2, 2> <<< num_blocks, threads_per_block>>>
            (A_row_offsets_ptr, A_column_indices_ptr, A_dia_idx_ptr, A_nonzero_values_ptr, Dinv_ptr,
             b_ptr, x_ptr, this->weight, num_rows_per_line, i, n, x_ptr);
            cudaCheckError();
        } // End of loop over colors
    } // End of if symFlag
}

template <AMGX_VecPrecision t_vecPrec, AMGX_MatPrecision t_matPrec, AMGX_IndPrecision t_indPrec>
void FixcolorGaussSeidelSolver<TemplateConfig<AMGX_device, t_vecPrec, t_matPrec, t_indPrec> >::smooth_3x3(const Matrix_d &A, const VVector &b, VVector &x, ViewType separation_flag)
{
    Matrix_d *A_as_matrix = (Matrix_d *) this->m_A;

    if (!A.hasProps(COLORING)) { FatalError("Matrix is not colored, exiting", AMGX_ERR_BAD_PARAMETERS); }

    const IndexType *A_row_offsets_ptr = A.row_offsets.raw();
    const IndexType *A_column_indices_ptr = A.col_indices.raw();
    const IndexType *A_dia_idx_ptr = A.diag.raw();
    const ValueTypeA *A_nonzero_values_ptr = A.values.raw();
    const ValueTypeB *b_ptr = b.raw();
    //TODO: generate required indices on the fly since coloring is not requested for this solver
    const IndexType *A_sorted_rows_by_color_ptr = A.getMatrixColoring().getSortedRowsByColor().raw();
    const ValueTypeA *Dinv_ptr = this->Dinv.raw();
    ValueTypeB *x_ptr = x.raw();
    const int threads_per_block = 64;
    int num_colors = this->num_colors;

    for (int i = 0; i < num_colors; i++)
    {
        const IndexType color_offset = ((separation_flag & INTERIOR) == 0) ? A.getMatrixColoring().getSeparationOffsetsRowsPerColor()[i] : A.getMatrixColoring().getOffsetsRowsPerColor()[i];
        const IndexType num_rows_per_color = ((separation_flag == A_as_matrix->getViewInterior()) ? A.getMatrixColoring().getSeparationOffsetsRowsPerColor()[i] : A.getMatrixColoring().getOffsetsRowsPerColor()[i + 1]) - color_offset;

        if (num_rows_per_color <= 0) { continue; }

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
            const IndexType num_rows_per_color = ((separation_flag == A_as_matrix->getViewInterior()) ? A.getMatrixColoring().getSeparationOffsetsRowsPerColor()[i] : A.getMatrixColoring().getOffsetsRowsPerColor()[i + 1]) - color_offset;

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

template <>
void FixcolorGaussSeidelSolver<TemplateConfig<AMGX_device, AMGX_vecDouble, AMGX_matFloat, AMGX_indInt> >::smooth_BxB(Matrix<TemplateConfig<AMGX_device, AMGX_vecDouble, AMGX_matFloat, AMGX_indInt> > &A, Vector<TemplateConfig<AMGX_device, AMGX_vecDouble, AMGX_matFloat, AMGX_indInt> > &b, Vector<TemplateConfig<AMGX_device, AMGX_vecDouble, AMGX_matFloat, AMGX_indInt> > &x, ViewType separation_flag)
{
    FatalError("Mixed precision is not supported", AMGX_ERR_NOT_IMPLEMENTED);
};

template <AMGX_VecPrecision t_vecPrec, AMGX_MatPrecision t_matPrec, AMGX_IndPrecision t_indPrec>
void FixcolorGaussSeidelSolver<TemplateConfig<AMGX_device, t_vecPrec, t_matPrec, t_indPrec> >::smooth_BxB(Matrix_d &A, VVector &b, VVector &x, ViewType separation_flag)
{
    if (!A.hasProps(COLORING)) { FatalError("Matrix is not colored, exiting", AMGX_ERR_BAD_PARAMETERS); }

    int num_colors = this->num_colors;
    VVector y(b.size());
    amgx::thrust::copy(b.begin(), b.end(), y.begin());                                                                          // y= b             for all colors
    cudaCheckError();

    for (int color = 0; color < num_colors; color++)
    {
        Cusparse::bsrmv(Cusparse::ALL_COLORS, color, (ValueTypeB) - 1.0f, A, x, (ValueTypeB)1.0f, y, separation_flag); // y= -A.x + y      for current color
        Cusparse::bsrmv(color, (ValueTypeB)this->weight, A, this->Dinv, y, (ValueTypeB)1.0f, x, separation_flag);                // x= w*Dinv.y + x  for current color
    }

    cudaCheckError();

    if (this->symFlag)
    {
        y = b;                                                                                        // y= b             for all colors

        for (int color = num_colors - 1; color >= 0; color--)
        {
            Cusparse::bsrmv(Cusparse::ALL_COLORS, color, (ValueTypeB) - 1.0f, A, x, (ValueTypeB)1.0f, y, separation_flag); // y= -A.x + y      for current color
            Cusparse::bsrmv(color, (ValueTypeB)this->weight, A, this->Dinv, y, (ValueTypeB)1.0f, x, separation_flag);                // x= w*Dinv.y + x  for current color
        }
    }

    cudaCheckError();
}

// Solve one iteration
template<class T_Config>
bool
FixcolorGaussSeidelSolver_Base<T_Config>::solve_iteration( VVector &b, VVector &x, bool xIsZero)
{
    Matrix<T_Config> *A_as_matrix = (Matrix<T_Config> *) this->m_A;

    if (xIsZero) { x.dirtybit = 0; }

    if (!A_as_matrix->is_matrix_singleGPU())
    {
        A_as_matrix->manager->exchange_halo_async(x, x.tag);
        A_as_matrix->manager->exchange_halo_async(b, b.tag);
    }

    if (A_as_matrix->getViewExterior() == A_as_matrix->getViewInterior())
    {
        if (!A_as_matrix->is_matrix_singleGPU())
        {
            A_as_matrix->manager->exchange_halo_wait(x, x.tag);
            A_as_matrix->manager->exchange_halo_wait(b, b.tag);
        }
    }

    if ((A_as_matrix->get_block_dimx() == 4 && A_as_matrix->get_block_dimy() == 4) || (A_as_matrix->get_block_dimx() == 1 && A_as_matrix->get_block_dimy() == 1))
    {
        if (!A_as_matrix->is_matrix_singleGPU())
        {
            A_as_matrix->manager->exchange_halo_wait(x, x.tag);
            A_as_matrix->manager->exchange_halo_wait(b, b.tag);
        }
    }

    ViewType oldView = A_as_matrix->currentView();
    bool latencyHiding = true;
    ViewType flags;

    if (A_as_matrix->is_matrix_singleGPU() || (x.dirtybit == 0 && b.dirtybit == 0))
    {
        latencyHiding = false;
        A_as_matrix->setViewExterior();
        flags = A_as_matrix->getViewExterior();
    }
    else
    {
        flags = A_as_matrix->getViewInterior();
        A_as_matrix->setViewInterior();
    }

    if (xIsZero)
    {
        thrust_wrapper::fill<T_Config::memSpace>(x.begin(), x.end(), 0.);
        cudaCheckError();
    }

    if (this->use_bsrxmv && A_as_matrix->get_block_dimx() == A_as_matrix->get_block_dimy())
    {
        smooth_BxB(*A_as_matrix, b, x, flags);
    }
    else if ( A_as_matrix->get_block_dimx() == 4 && A_as_matrix->get_block_dimy() == 4 )
    {
        smooth_4x4(*A_as_matrix, b, x, flags);
    }
    else if ( A_as_matrix->get_block_dimx() == 3 && A_as_matrix->get_block_dimy() == 3 )
    {
        smooth_3x3(*A_as_matrix, b, x, flags);
    }
    else if ( A_as_matrix->get_block_dimx() == 1 && A_as_matrix->get_block_dimy() == 1 )
    {
        smooth_1x1(*A_as_matrix, b, x, flags);
    }
    else
    {
        FatalError("Unsupported block size for MulticolorGaussSeidelSolver smooth", AMGX_ERR_NOT_SUPPORTED_BLOCKSIZE);
    }

    if (latencyHiding)
    {
        if (!A_as_matrix->is_matrix_singleGPU())
        {
            A_as_matrix->manager->exchange_halo_wait(x, x.tag);
            A_as_matrix->manager->exchange_halo_wait(b, b.tag);
        }

        A_as_matrix->setViewExterior();
        flags = (ViewType)(~(A_as_matrix->getViewInterior()) & A_as_matrix->getViewExterior());

        if (flags != 0)
        {
            if (this->use_bsrxmv && A_as_matrix->get_block_dimx() == A_as_matrix->get_block_dimy())
            {
                smooth_BxB(*A_as_matrix, b, x, flags);
            }
            else if ( A_as_matrix->get_block_dimx() == 4 && A_as_matrix->get_block_dimy() == 4 )
            {
                smooth_4x4(*A_as_matrix, b, x, flags);
            }
            else if ( A_as_matrix->get_block_dimx() == 3 && A_as_matrix->get_block_dimy() == 3 )
            {
                smooth_3x3(*A_as_matrix, b, x, flags);
            }
            else if ( A_as_matrix->get_block_dimx() == 1 && A_as_matrix->get_block_dimy() == 1 )
            {
                smooth_1x1(*A_as_matrix, b, x, flags);
            }
        }
    }

    x.dirtybit = 1;
    A_as_matrix->setView(oldView);
    return this->converged( b, x );
}

template<class T_Config>
void
FixcolorGaussSeidelSolver_Base<T_Config>::solve_finalize(VVector &b, VVector &x)
{
}



/****************************************
 * Explict instantiations
 ***************************************/
#define AMGX_CASE_LINE(CASE) template class FixcolorGaussSeidelSolver_Base<TemplateMode<CASE>::Type>;
AMGX_FORALL_BUILDS(AMGX_CASE_LINE)
#undef AMGX_CASE_LINE

#define AMGX_CASE_LINE(CASE) template class FixcolorGaussSeidelSolver<TemplateMode<CASE>::Type>;
AMGX_FORALL_BUILDS(AMGX_CASE_LINE)
#undef AMGX_CASE_LINE

}
} // namespace amgx
