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

#pragma once

template <typename index_type, typename mat_value_type, bool reorder_rows, bool reorder_cols>
void reorder_partition_host(index_type n, index_type nnz, index_type *Ap, index_type *Ac, mat_value_type *Av,
                            index_type *Bp, index_type *Bc, mat_value_type *Bv, index_type l, index_type *p,
			                      index_type block_dimx, index_type block_dimy)
{
    //applies reordering P from left adn right on matrix A, so that B=P'*A*P.
    //notice that matrix A may be rectangular (does not need to be square).
    //WARNING: columns within the rows are not sorted in the output matrix B.
    using namespace amgx;
    index_type row, col, i, j, k;
    index_type block_size = block_dimx * block_dimy;
    //(i)  reorder rows
    //compute number of elements per row (in new locations)
    Bp[0] = 0;

    for (row = 0; row < n; row++)
    {
        i = reorder_rows ? p[row] : row;
        Bp[i + 1] = Ap[row + 1] - Ap[row];
    }

    //prefix scan to create row pointers in CSR format
    for (i = 0; i < n; i++)
    {
        Bp[i + 1] += Bp[i];
    }

    //(ii) reorder columns and values (accounting for earlier row permutation)
    for (row = 0; row < n; row++)
    {
        i = reorder_rows ? p[row] : row;
        for (k = Bp[i], j = Ap[row]; j < Ap[row + 1]; j++, k++)
        {
            col  = Ac[j];
            Bc[k] = reorder_cols ? p[col] : col;
            for (int kx = 0; kx < block_dimx; kx++) 
            {
                for (int ky = 0; ky < block_dimy; ky++)
                {
                    Bv[k * block_size + kx * block_dimy + ky] = Av[j * block_size + kx * block_dimy + ky];
                }
            }
        }
    }
}

template <typename index_type, typename mat_value_type, bool reorder_rows, bool reorder_cols>
void reorder_partition(index_type n, index_type nnz, index_type *Ap, index_type *Ac, mat_value_type *Av,
                       index_type *Bp, index_type *Bc, mat_value_type *Bv, index_type l, index_type *p,
		       index_type block_dimx, index_type block_dimy)
{
    using namespace amgx;
    cudaError_t st1, st2, st3, st4;
    index_type block_size = block_dimx * block_dimy;
    //applies reordering P from left adn right on matrix A, so that B=P'*A*P.
    //notice that matrix A may be rectangular (does not need to be square).
    //for now implement device by copying matrices to the host and calling host.
    index_type     *p_h = NULL;
    index_type     *Ap_h = NULL;
    index_type     *Ac_h = NULL;
    mat_value_type *Av_h = NULL;
    index_type     *Bp_h = NULL;
    index_type     *Bc_h = NULL;
    mat_value_type *Bv_h = NULL;
    p_h  = (index_type *)malloc(    l * sizeof( p_h[0]));
    Ap_h = (index_type *)malloc((n + 1) * sizeof(Ap_h[0]));
    Ac_h = (index_type *)malloc(  nnz * sizeof(Ac_h[0]));
    Av_h = (mat_value_type *)malloc(  nnz * block_size * sizeof(Av_h[0]));
    Bp_h = (index_type *)malloc((n + 1) * sizeof(Bp_h[0]));
    Bc_h = (index_type *)malloc(  nnz * sizeof(Bc_h[0]));
    Bv_h = (mat_value_type *)malloc(  nnz * block_size * sizeof(Bv_h[0]));

    if (!p_h || !Ap_h || !Ac_h || !Av_h || !Bp_h || !Bc_h || !Bv_h)
    {
        FatalError("reorder_partition (one of the (host) mallocs failed", AMGX_ERR_CORE);
    }

    st1 = cudaMemcpy(p_h,  p,    l * sizeof( p[0]), cudaMemcpyDeviceToHost);
    st2 = cudaMemcpy(Ap_h, Ap, (n + 1) * sizeof(Ap[0]), cudaMemcpyDeviceToHost);
    st3 = cudaMemcpy(Ac_h, Ac,  nnz * sizeof(Ac[0]), cudaMemcpyDeviceToHost);
    st4 = cudaMemcpy(Av_h, Av,  nnz * block_size * sizeof(Av[0]), cudaMemcpyDeviceToHost);

    if ((st1 != cudaSuccess) || (st2 != cudaSuccess) || (st3 != cudaSuccess) || (st4 != cudaSuccess))
    {
        FatalError("reorder_partition (one of the cudaMemcpy back to host failed", AMGX_ERR_CORE);
    }

    reorder_partition_host<index_type, mat_value_type, reorder_rows, reorder_cols>
    (n, nnz, Ap_h, Ac_h, Av_h, Bp_h, Bc_h, Bv_h, l, p_h, block_dimx, block_dimy);
    st1 = cudaMemcpy(Bp, Bp_h, (n + 1) * sizeof(Bp[0]), cudaMemcpyHostToDevice);
    st2 = cudaMemcpy(Bc, Bc_h,  nnz * sizeof(Bc[0]), cudaMemcpyHostToDevice);
    st3 = cudaMemcpy(Bv, Bv_h,  nnz * block_size * sizeof(Bv[0]), cudaMemcpyHostToDevice);

    if ((st1 != cudaSuccess) || (st2 != cudaSuccess) || (st3 != cudaSuccess))
    {
        FatalError("reorder_partition (one of the cudaMemcpy back to device failed", AMGX_ERR_CORE);
    }

    if (p_h) { free(p_h); }

    if (Ap_h) { free(Ap_h); }

    if (Ac_h) { free(Ac_h); }

    if (Av_h) { free(Av_h); }

    if (Bp_h) { free(Bp_h); }

    if (Bc_h) { free(Bc_h); }

    if (Bv_h) { free(Bv_h); }
}
