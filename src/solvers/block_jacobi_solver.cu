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

#include <solvers/block_jacobi_solver.h>
#include <solvers/block_common_solver.h>
#include <gaussian_elimination.h>
#include <basic_types.h>
#include <cutil.h>
#include <util.h>
#include <string>
#include <miscmath.h>
#include <texture.h>
#include <amgx_cusparse.h>
#include <ostream>

#include <amgx_types/util.h>

using namespace std;
namespace amgx
{
namespace block_jacobi_solver
{

template <typename ValueTypeA, typename ValueTypeB>
struct jacobi_presmooth_functor
{
    double omega;
    jacobi_presmooth_functor( double omega ) : omega( omega ) {}
    __host__ __device__ ValueTypeB operator()( const ValueTypeB &b, const ValueTypeA &d ) const { return isNotCloseToZero(d) ? b * omega / d : b * omega / epsilon(d); }
};

template <typename ValueTypeA, typename ValueTypeB>
struct jacobi_postsmooth_functor
{
    double omega;
    jacobi_postsmooth_functor( double omega ) : omega( omega ) {}
    template<typename Tuple> __host__ __device__  ValueTypeB operator( )( const Tuple &t ) const
    {
        ValueTypeB x = amgx::thrust::get<0>(t);
        ValueTypeA d = amgx::thrust::get<1>(t);
        ValueTypeB b = amgx::thrust::get<2>(t);
        ValueTypeB y = amgx::thrust::get<3>(t);
        // return x + omega * (b - y) / d.
        d = isNotCloseToZero(d) ? d :  epsilon(d);
        d = types::util<ValueTypeA>::get_one() / d;
        b = b - y;
        b = b * omega;
        return b * d + x;
    }
};

template <typename ValueTypeB>
struct add_functor
{
    __host__ __device__  ValueTypeB operator()( const ValueTypeB &x, const ValueTypeB &y )const { return x + y; }
};

template<typename T>
__device__ __forceinline__ T fmnaOp (T a, T b, T c)
{
    return -(a * b) + c;
}
template<typename T>
__device__ __forceinline__ T mulOp (T a, T b)
{
    return a * b;
}
template<typename T>
__device__ __forceinline__ T rcpOp (T a)
{
    return 1.0 / (isNotCloseToZero(a) ? a : epsilon(a));
}
template<typename T>
__device__ __forceinline__ T absOp (T a)
{
    return fabs(a);
}

// -----------------------------------
//  KERNELS
// -----------------------------------

template<typename T1, typename T2, int N>
__global__ void matinv_matrix_per_thread_pivot (const T1 *A, T2 *Ainv, int batch)
{
#define A(row,col)     A[(col)*N+(row)]
#define AA(row,col)    AA[(col)*N+(row)]
#define Ainv(row,col)  Ainv[(col)*N+(row)]
    const int blkNum = blockIdx.x * blockDim.x + threadIdx.x;
    int perm0, perm1, perm2, perm3;
    int icol0, icol1, icol2, icol3;
    T2 AA00, AA01, AA02, AA03, AA10, AA11, AA12, AA13;
    T2 AA20, AA21, AA22, AA23, AA30, AA31, AA32, AA33;
    T2 p, t;
    int i, pvt;
    A    += blkNum * N * N;
    Ainv += blkNum * N * N;

    if (blkNum < batch)
    {
        AA00 = A[0];
        AA10 = A[1];
        AA20 = A[2];
        AA30 = A[3];
        AA01 = A[4];
        AA11 = A[5];
        AA21 = A[6];
        AA31 = A[7];
        AA02 = A[8];
        AA12 = A[9];
        AA22 = A[10];
        AA32 = A[11];
        AA03 = A[12];
        AA13 = A[13];
        AA23 = A[14];
        AA33 = A[15];
        perm0 = 0;
        perm1 = 1;
        perm2 = 2;
        perm3 = 3;
        /****************** iteration 0 ***********/
        /* search pivot row */
        p = absOp (AA00);
        pvt = 0;
        t = absOp (AA10);

        if (t > p)
        {
            p = t;
            pvt = 1;
        }

        t = absOp (AA20);

        if (t > p)
        {
            p = t;
            pvt = 2;
        }

        t = absOp (AA30);

        if (t > p)
        {
            p = t;
            pvt = 3;
        }

        /* swap pivot row with row 0 */
        if (pvt == 1)
        {
            t = AA00;
            AA00 = AA10;
            AA10 = t;
            t = AA01;
            AA01 = AA11;
            AA11 = t;
            t = AA02;
            AA02 = AA12;
            AA12 = t;
            t = AA03;
            AA03 = AA13;
            AA13 = t;
            /* update permutation vector based on row swap */
            i = perm0;
            perm0 = perm1;
            perm1 = i;
        }

        if (pvt == 2)
        {
            t = AA00;
            AA00 = AA20;
            AA20 = t;
            t = AA01;
            AA01 = AA21;
            AA21 = t;
            t = AA02;
            AA02 = AA22;
            AA22 = t;
            t = AA03;
            AA03 = AA23;
            AA23 = t;
            /* update permutation vector based on row swap */
            i = perm0;
            perm0 = perm2;
            perm2 = i;
        }

        if (pvt == 3)
        {
            t = AA00;
            AA00 = AA30;
            AA30 = t;
            t = AA01;
            AA01 = AA31;
            AA31 = t;
            t = AA02;
            AA02 = AA32;
            AA32 = t;
            t = AA03;
            AA03 = AA33;
            AA33 = t;
            /* update permutation vector based on row swap */
            i = perm0;
            perm0 = perm3;
            perm3 = i;
        }

        /* scale current row */
        t = rcpOp (AA00);
        icol0 = perm0;
        AA00 = t;
        AA01 = mulOp (t, AA01);
        AA02 = mulOp (t, AA02);
        AA03 = mulOp (t, AA03);
        /* eliminate above and below current row */
        t = AA10;
        AA10 = mulOp (-t, AA00);
        AA11 = fmnaOp (t, AA01, AA11);
        AA12 = fmnaOp (t, AA02, AA12);
        AA13 = fmnaOp (t, AA03, AA13);
        t = AA20;
        AA20 = mulOp (-t, AA00);
        AA21 = fmnaOp (t, AA01, AA21);
        AA22 = fmnaOp (t, AA02, AA22);
        AA23 = fmnaOp (t, AA03, AA23);
        t = AA30;
        AA30 = mulOp (-t, AA00);
        AA31 = fmnaOp (t, AA01, AA31);
        AA32 = fmnaOp (t, AA02, AA32);
        AA33 = fmnaOp (t, AA03, AA33);
        /****************** iteration 1 ***********/
        /* search pivot row */
        p = absOp (AA11);
        pvt = 1;
        t = absOp (AA21);

        if (t > p)
        {
            p = t;
            pvt = 2;
        }

        t = absOp (AA31);

        if (t > p)
        {
            p = t;
            pvt = 3;
        }

        /* swap pivot row with row 1 */
        if (pvt == 2)
        {
            t = AA10;
            AA10 = AA20;
            AA20 = t;
            t = AA11;
            AA11 = AA21;
            AA21 = t;
            t = AA12;
            AA12 = AA22;
            AA22 = t;
            t = AA13;
            AA13 = AA23;
            AA23 = t;
            /* update permutation vector based on row swap */
            i = perm1;
            perm1 = perm2;
            perm2 = i;
        }
        else if (pvt == 3)
        {
            t = AA10;
            AA10 = AA30;
            AA30 = t;
            t = AA11;
            AA11 = AA31;
            AA31 = t;
            t = AA12;
            AA12 = AA32;
            AA32 = t;
            t = AA13;
            AA13 = AA33;
            AA33 = t;
            /* update permutation vector based on row swap */
            i = perm1;
            perm1 = perm3;
            perm3 = i;
        }

        /* scale current row */
        t = rcpOp (AA11);
        icol1 = perm1;
        AA10 = mulOp (t, AA10);
        AA11 = t;
        AA12 = mulOp (t, AA12);
        AA13 = mulOp (t, AA13);
        /* eliminate above and below current row */
        t = AA01;
        AA00 = fmnaOp (t, AA10, AA00);
        AA01 = mulOp (-t, AA11);
        AA02 = fmnaOp (t, AA12, AA02);
        AA03 = fmnaOp (t, AA13, AA03);
        t = AA21;
        AA20 = fmnaOp (t, AA10, AA20);
        AA21 = mulOp (-t, AA11);
        AA22 = fmnaOp (t, AA12, AA22);
        AA23 = fmnaOp (t, AA13, AA23);
        t = AA31;
        AA30 = fmnaOp (t, AA10, AA30);
        AA31 = mulOp (-t, AA11);
        AA32 = fmnaOp (t, AA12, AA32);
        AA33 = fmnaOp (t, AA13, AA33);
        /****************** iteration 2 ****************/
        /* search pivot row */
        p = absOp (AA22);
        pvt = 2;
        t = absOp (AA32);

        if (t > p)
        {
            p = t;
            pvt = 3;
        }

        /* swap pivot row with row 2 */
        if (pvt == 3)
        {
            t = AA20;
            AA20 = AA30;
            AA30 = t;
            t = AA21;
            AA21 = AA31;
            AA31 = t;
            t = AA22;
            AA22 = AA32;
            AA32 = t;
            t = AA23;
            AA23 = AA33;
            AA33 = t;
            /* update permutation vector based on row swap */
            i = perm2;
            perm2 = perm3;
            perm3 = i;
        }

        /* scale current row */
        t = rcpOp (AA22);
        icol2 = perm2;
        AA20 = mulOp (t, AA20);
        AA21 = mulOp (t, AA21);
        AA22 = t;
        AA23 = mulOp (t, AA23);
        /* eliminate above and below current row */
        t = AA02;
        AA00 = fmnaOp (t, AA20, AA00);
        AA01 = fmnaOp (t, AA21, AA01);
        AA02 = mulOp (-t, AA22);
        AA03 = fmnaOp (t, AA23, AA03);
        t = AA12;
        AA10 = fmnaOp (t, AA20, AA10);
        AA11 = fmnaOp (t, AA21, AA11);
        AA12 = mulOp (-t, AA22);
        AA13 = fmnaOp (t, AA23, AA13);
        t = AA32;
        AA30 = fmnaOp (t, AA20, AA30);
        AA31 = fmnaOp (t, AA21, AA31);
        AA32 = mulOp (-t, AA22);
        AA33 = fmnaOp (t, AA23, AA33);
        /****************** iteration 3 ****************/
        /* scale current row */
        t = rcpOp (AA33);
        icol3 = perm3;
        AA30 = mulOp (t, AA30);
        AA31 = mulOp (t, AA31);
        AA32 = mulOp (t, AA32);
        AA33 = t;
        /* eliminate above and below current row */
        t = AA03;
        AA00 = fmnaOp (t, AA30, AA00);
        AA01 = fmnaOp (t, AA31, AA01);
        AA02 = fmnaOp (t, AA32, AA02);
        AA03 = mulOp (-t, AA33);
        t = AA13;
        AA10 = fmnaOp (t, AA30, AA10);
        AA11 = fmnaOp (t, AA31, AA11);
        AA12 = fmnaOp (t, AA32, AA12);
        AA13 = mulOp (-t, AA33);
        t = AA23;
        AA20 = fmnaOp (t, AA30, AA20);
        AA21 = fmnaOp (t, AA31, AA21);
        AA22 = fmnaOp (t, AA32, AA22);
        AA23 = mulOp (-t, AA33);
        /* sort columns into the correct order */
        Ainv(0, icol0) = AA00;
        Ainv(1, icol0) = AA10;
        Ainv(2, icol0) = AA20;
        Ainv(3, icol0) = AA30;
        Ainv(0, icol1) = AA01;
        Ainv(1, icol1) = AA11;
        Ainv(2, icol1) = AA21;
        Ainv(3, icol1) = AA31;
        Ainv(0, icol2) = AA02;
        Ainv(1, icol2) = AA12;
        Ainv(2, icol2) = AA22;
        Ainv(3, icol2) = AA32;
        Ainv(0, icol3) = AA03;
        Ainv(1, icol3) = AA13;
        Ainv(2, icol3) = AA23;
        Ainv(3, icol3) = AA33;
    }
}


template<typename T, int N>
__global__ void matinv_matrix_per_thread_no_pivot (const T *A, T *Ainv, int batch)
{
#define A(row,col)     A[(col)*N+(row)]
#define AA(row,col)    AA[(col)*N+(row)]
#define Ainv(row,col)  Ainv[(col)*N+(row)]
    const int blkNum = blockIdx.x * blockDim.x + threadIdx.x;
    T AA00, AA01, AA02, AA03, AA10, AA11, AA12, AA13;
    T AA20, AA21, AA22, AA23, AA30, AA31, AA32, AA33;
    T t;
    A    += blkNum * N * N;
    Ainv += blkNum * N * N;

    if (blkNum < batch)
    {
        AA00 = A[0];
        AA10 = A[1];
        AA20 = A[2];
        AA30 = A[3];
        AA01 = A[4];
        AA11 = A[5];
        AA21 = A[6];
        AA31 = A[7];
        AA02 = A[8];
        AA12 = A[9];
        AA22 = A[10];
        AA32 = A[11];
        AA03 = A[12];
        AA13 = A[13];
        AA23 = A[14];
        AA33 = A[15];
        /****************** iteration 0 ***********/
        /* search pivot row */
        t = 1.0 / (AA00);
        AA00 = t;
        AA01 = t * AA01;
        AA02 = t * AA02;
        AA03 = t * AA03;
        /* eliminate above and below current row */
        t = AA10;
        AA10 = -t * AA00;
        AA11 = fmnaOp (t, AA01, AA11);
        AA12 = fmnaOp (t, AA02, AA12);
        AA13 = fmnaOp (t, AA03, AA13);
        t = AA20;
        AA20 = -t * AA00;
        AA21 = fmnaOp (t, AA01, AA21);
        AA22 = fmnaOp (t, AA02, AA22);
        AA23 = fmnaOp (t, AA03, AA23);
        t = AA30;
        AA30 = -t * AA00;
        AA31 = fmnaOp (t, AA01, AA31);
        AA32 = fmnaOp (t, AA02, AA32);
        AA33 = fmnaOp (t, AA03, AA33);
        /****************** iteration 1 ***********/
        /* scale current row */
        t = 1.0 / (AA11);
        AA10 = t * AA10;
        AA11 = t;
        AA12 = t * AA12;
        AA13 = t * AA13;
        /* eliminate above and below current row */
        t = AA01;
        AA00 = fmnaOp (t, AA10, AA00);
        AA01 = -t * AA11;
        AA02 = fmnaOp (t, AA12, AA02);
        AA03 = fmnaOp (t, AA13, AA03);
        t = AA21;
        AA20 = fmnaOp (t, AA10, AA20);
        AA21 = -t * AA11;
        AA22 = fmnaOp (t, AA12, AA22);
        AA23 = fmnaOp (t, AA13, AA23);
        t = AA31;
        AA30 = fmnaOp (t, AA10, AA30);
        AA31 = -t * AA11;
        AA32 = fmnaOp (t, AA12, AA32);
        AA33 = fmnaOp (t, AA13, AA33);
        /****************** iteration 2 ****************/
        /* scale current row */
        t = 1.0 / (AA22);
        AA20 = t * AA20;
        AA21 = t * AA21;
        AA22 = t;
        AA23 = t * AA23;
        /* eliminate above and below current row */
        t = AA02;
        AA00 = fmnaOp (t, AA20, AA00);
        AA01 = fmnaOp (t, AA21, AA01);
        AA02 = -t * AA22;
        AA03 = fmnaOp (t, AA23, AA03);
        t = AA12;
        AA10 = fmnaOp (t, AA20, AA10);
        AA11 = fmnaOp (t, AA21, AA11);
        AA12 = -t * AA22;
        AA13 = fmnaOp (t, AA23, AA13);
        t = AA32;
        AA30 = fmnaOp (t, AA20, AA30);
        AA31 = fmnaOp (t, AA21, AA31);
        AA32 = -t * AA22;
        AA33 = fmnaOp (t, AA23, AA33);
        /****************** iteration 3 ****************/
        /* scale current row */
        t = 1.0 / (AA33);
        AA30 =  t * AA30;
        AA31 =  t * AA31;
        AA32 =  t * AA32;
        AA33 = t;
        /* eliminate above and below current row */
        t = AA03;
        AA00 = fmnaOp (t, AA30, AA00);
        AA01 = fmnaOp (t, AA31, AA01);
        AA02 = fmnaOp (t, AA32, AA02);
        AA03 = -t * AA33;
        t = AA13;
        AA10 = fmnaOp (t, AA30, AA10);
        AA11 = fmnaOp (t, AA31, AA11);
        AA12 = fmnaOp (t, AA32, AA12);
        AA13 = -t * AA33;
        t = AA23;
        AA20 = fmnaOp (t, AA30, AA20);
        AA21 = fmnaOp (t, AA31, AA21);
        AA22 = fmnaOp (t, AA32, AA22);
        AA23 = -t * AA33;
        /* sort columns into the correct order */
        Ainv(0, 0) = AA00;
        Ainv(1, 0) = AA10;
        Ainv(2, 0) = AA20;
        Ainv(3, 0) = AA30;
        Ainv(0, 1) = AA01;
        Ainv(1, 1) = AA11;
        Ainv(2, 1) = AA21;
        Ainv(3, 1) = AA31;
        Ainv(0, 2) = AA02;
        Ainv(1, 2) = AA12;
        Ainv(2, 2) = AA22;
        Ainv(3, 2) = AA32;
        Ainv(0, 3) = AA03;
        Ainv(1, 3) = AA13;
        Ainv(2, 3) = AA23;
        Ainv(3, 3) = AA33;
    }
}

template<typename IndexType, typename ValueTypeA, int threads_per_block, int halfwarps_per_block>
__global__
void setupBlockJacobiSmoothbBigBlockDiaCsrKernel(const IndexType *row_offsets, const IndexType *column_indices, const ValueTypeA *values, const IndexType *dia_indices, ValueTypeA *Dinv, const int num_block_rows, int bsize, int bsize_sq, ValueTypeA *temp1)
{
    const int tid = blockDim.x * blockIdx.x + threadIdx.x;
    int halfwarp_id = tid >> 4;
    const int block_halfwarp_id = threadIdx.x >> 4;
    const int mat_entry_index = threadIdx.x & (16 - 1);
    const int i_ind = mat_entry_index >> 2;
    const int j_ind = mat_entry_index & 3;
    extern __shared__ volatile char schar[];
    volatile ValueTypeA *s_Amat;
    s_Amat = (ValueTypeA *)&schar[0];
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
                    types::util<ValueTypeA>::volcast( e_out[t1 * tile_num + t2], s_Amat + (s_offset + (t1 * 4 + i_ind) * bsize + t2 * 4 + j_ind) );
                }

        compute_block_inverse2<IndexType, ValueTypeA, halfwarps_per_block>
        ( s_Amat, s_offset, offset, i_ind, j_ind, Dinv, tile_num, bsize, bsize_sq );
        halfwarp_id += gridDim.x * halfwarps_per_block;
    }
}

template<typename IndexType, typename ValueTypeA, int blockrows_per_cta, int blockrows_per_warp, int bsize, int bsize_sq>
__global__
void setupBlockJacobiSmoothBbyBBlockDiaCsrKernel(const IndexType *row_offsets, const IndexType *column_indices, const ValueTypeA *values, const IndexType *dia_indices, ValueTypeA *Dinv, const int num_block_rows)

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
        types::util<ValueTypeA>::volcast(e_out, s_Amat + tid);
        compute_block_inverse_row_major<IndexType, ValueTypeA, blockrows_per_cta, bsize, bsize_sq>
        ( s_Amat, cta_blockrow_id * bsize_sq, offset, i_ind, j_ind, Dinv );
        blockrow_id += gridDim.x * blockrows_per_cta;
    }
}

template<typename ValueTypeA, typename ValueTypeB, typename IndexType, int threads_per_block, int halfwarps_per_block, int bsize, int log_bsize, int bsize_sq, int log_bsize_sq>
__global__
void setupBlockJacobiSmooth4by4BlockDiaCsrKernel_V2(const IndexType *dia_indices, const ValueTypeA *A_values, ValueTypeA *Dinv, const int num_block_rows)
{
    const int tid = blockDim.x * blockIdx.x + threadIdx.x;
    int halfwarp_id = tid >> log_bsize_sq;
    const int block_halfwarp_id = threadIdx.x >> log_bsize_sq;
    const int mat_entry_index = threadIdx.x & (bsize_sq - 1);
    const int i_ind = mat_entry_index >> log_bsize;
    const int j_ind = mat_entry_index & (bsize - 1);
    volatile __shared__ ValueTypeA s_Amat[bsize_sq * halfwarps_per_block ];
    int offset;
    ValueTypeA e_out;

    while (halfwarp_id < num_block_rows)
    {
        // Store the diagonal
        offset = halfwarp_id * bsize_sq;
        e_out = A_values[bsize_sq * dia_indices[halfwarp_id] + mat_entry_index];
        // Each thread stores its entry in s_Amat
        types::util<ValueTypeA>::volcast(e_out, s_Amat + threadIdx.x);
        compute_block_inverse_row_major<int, ValueTypeA, halfwarps_per_block, bsize, bsize_sq>
        ( s_Amat, block_halfwarp_id * bsize_sq, offset + mat_entry_index, i_ind, j_ind, Dinv );
        halfwarp_id += gridDim.x * blockDim.x >> log_bsize_sq;
    }
}

// Kernel to smooth with Jacobi smoother, Dinv assumed to be computed
template<typename IndexType, typename ValueTypeA, typename ValueTypeB, int eighthwarps_per_block, int bsize, int log_bsize, int half_bsize, int bsize_sq>
__global__
void jacobiSmooth4by4BlockDiaCsrKernel_NAIVE_tex_readDinv2(const IndexType *row_offsets, const IndexType *column_indices, const IndexType *dia_indices, const ValueTypeA *nonzero_values, const ValueTypeA *Dinv,
        const ValueTypeB *b, const ValueTypeB *x, double weight, const int num_block_rows, ValueTypeB *xout, const int row_offset)
{
    const int tid = blockDim.x * blockIdx.x + threadIdx.x;
    int eighthwarp_id = row_offset + (tid >> log_bsize);
    const int block_eighthwarp_id = threadIdx.x >> log_bsize;
    const int vec_entry_index = threadIdx.x & (bsize - 1);
    volatile __shared__ ValueTypeB s_xtemp[ bsize * eighthwarps_per_block ];
    ValueTypeB bmAx, xin;
    ValueTypeB temp[bsize];
    int offset, i, s_offset;

    while (eighthwarp_id < num_block_rows)
    {
        i = eighthwarp_id;
        offset = i * bsize + vec_entry_index;
        // 1. COMPUTING b-Ax
        bmAx = b[offset];
        // Contribution from diagonal
        xin = x[offset];
        types::util<ValueTypeB>::volcast(xin, s_xtemp + threadIdx.x);
        // Load dia_values and do matrix multiply
        loadAsVector<bsize>(nonzero_values + bsize_sq * dia_indices[i] + vec_entry_index * bsize, temp);
        s_offset = block_eighthwarp_id * bsize;
#pragma unroll

        for (int m = 0; m < bsize; m++)
        {
            bmAx = bmAx - temp[m] * types::util<ValueTypeB>::volcast(s_xtemp[s_offset + m]);
        }

        // Contribution from each nonzero column
        int jmax = row_offsets[i + 1];

        for (int jind = row_offsets[i]; jind < jmax; jind++)
        {
            IndexType jcol = __cachingLoad(&column_indices[jind]);

            if (jcol != i)
            {
                offset = jcol * bsize + vec_entry_index;
                types::util<ValueTypeB>::volcast(x[offset], s_xtemp + threadIdx.x);
                // Load nonzero_values
                s_offset = block_eighthwarp_id * bsize;
                offset = jind * bsize * bsize + vec_entry_index * bsize;
                loadAsVector<bsize>(nonzero_values + offset, temp);
#pragma unroll

                for (int m = 0; m < bsize; m++)
                {
                    bmAx = bmAx - temp[m] * types::util<ValueTypeB>::volcast(s_xtemp[s_offset + m]);
                }
            }
        }

        types::util<ValueTypeB>::volcast(bmAx, s_xtemp + threadIdx.x);
        bmAx = types::util<ValueTypeB>::get_zero();
        // 2. Multiply by Dinv
        // Load Dinv and multiply to RHS
        offset = i * bsize * bsize + vec_entry_index * bsize;
        loadAsVector<bsize>(Dinv + offset, temp);
        s_offset = block_eighthwarp_id * bsize;
#pragma unroll

        for (int m = 0; m < bsize; m++)
        {
            bmAx = bmAx + temp[m] * types::util<ValueTypeB>::volcast(s_xtemp[s_offset + m]);
        }

        xout[i * bsize + vec_entry_index] = xin + bmAx * weight;
        eighthwarp_id += gridDim.x * blockDim.x >> log_bsize;
    }
}

// Kernel to smooth with jacobi smoother, zero initial guess
template<typename IndexType, typename ValueTypeA, typename ValueTypeB, int eighthwarps_per_block, int bsize, int log_bsize, int half_bsize>
__global__
void jacobiSmooth4by4ZeroBlockDiaCsrKernel_NAIVE_tex_readDinv2(const ValueTypeA *Dinv, const ValueTypeB *b, double weight, const int num_block_rows, ValueTypeB *xout, const int row_offset)
{
    const int tid = blockDim.x * blockIdx.x + threadIdx.x;
    const int block_eighthwarp_id = threadIdx.x >> log_bsize;
    const int vec_entry_index = threadIdx.x & (bsize - 1);
    volatile __shared__ ValueTypeB s_xtemp[ bsize * eighthwarps_per_block ];
    ValueTypeB bmAx;
    ValueTypeB temp[bsize];
    int offset, i, s_offset;

    for (int eighthwarp_id = row_offset + (tid >> log_bsize); eighthwarp_id < num_block_rows; eighthwarp_id += (gridDim.x * blockDim.x >> log_bsize))
    {
        i = eighthwarp_id;
        offset = i * bsize + vec_entry_index;
        types::util<ValueTypeB>::volcast(b[offset], s_xtemp + threadIdx.x);
        bmAx = types::util<ValueTypeB>::get_zero();
        // Load Dinv and multiply to RHS
        offset = i * bsize * bsize + vec_entry_index * bsize;
        loadAsVector<bsize>(Dinv + offset, temp);
        s_offset = block_eighthwarp_id * bsize;
#pragma unroll

        for (int m = 0; m < bsize; m++)
        {
            bmAx = bmAx + temp[m] * types::util<ValueTypeB>::volcast(s_xtemp[s_offset + m]);
        }

        xout[i * bsize + vec_entry_index] = bmAx * weight;
    }
}

//--------------------------------
// Methods
//--------------------------------

// Constructor
template<class T_Config>
BlockJacobiSolver_Base<T_Config>::BlockJacobiSolver_Base( AMG_Config &cfg, const std::string &cfg_scope) : Solver<T_Config>( cfg, cfg_scope)
{
    weight = cfg.AMG_Config::getParameter<double>("relaxation_factor", cfg_scope);

    if (weight == 0)
    {
        weight = 1.;
        amgx_printf("Warning, setting weight to 1 instead of estimating largest_eigen_value in Block Jacobi smoother\n");
    }
}

// Destructor
template<class T_Config>
BlockJacobiSolver_Base<T_Config>::~BlockJacobiSolver_Base()
{
    this->Dinv.resize(0);
}

template<class T_Config>
void
BlockJacobiSolver_Base<T_Config>::printSolverParameters() const
{
    std::cout << "relaxation_factor= " << this->weight << std::endl;
}

// Solver setup
template<class T_Config>
void
BlockJacobiSolver_Base<T_Config>::solver_setup(bool reuse_matrix_structure)
{
    Matrix<T_Config> *A_as_matrix = dynamic_cast<Matrix<T_Config>*>(this->m_A);

    if (!A_as_matrix)
    {
        FatalError("BlockJacobiSolver only works with explicit matrices", AMGX_ERR_INTERNAL);
    }

    computeDinv( *A_as_matrix );

    if ( A_as_matrix->getBlockFormat() != ROW_MAJOR )
    {
        FatalError("Block Jacobi solver only supports row major format", AMGX_ERR_CONFIGURATION);
    }
}

//
template<class T_Config>
void
BlockJacobiSolver_Base<T_Config>::solve_init( VVector &b, VVector &x, bool xIsZero )
{
}

// Solve one iteration
template<class T_Config>
bool
BlockJacobiSolver_Base<T_Config>::solve_iteration( VVector &b, VVector &x, bool xIsZero )
{
    //bool done = false;
    Matrix<T_Config> *A_as_matrix = (Matrix<T_Config> *) this->m_A;

    if (xIsZero) { x.dirtybit = 0; }

    ViewType oldView = A_as_matrix->currentView();

    A_as_matrix->setViewExterior();
    ViewType flags = (ViewType)(A_as_matrix->getViewInterior() | A_as_matrix->getViewExterior());

    if (A_as_matrix->get_block_dimx() == 1 && A_as_matrix->get_block_dimy() == 1)
    {
        if (xIsZero)
        {
            smooth_with_0_initial_guess_1x1(*A_as_matrix, b, x, flags);
        }
        else
        {
            smooth_1x1(*A_as_matrix, b, x, flags);
        }
    }
    else if (A_as_matrix->get_block_dimx() == 4 && A_as_matrix->get_block_dimy() == 4)
    {
        if (xIsZero)
        {
            smooth_with_0_initial_guess_4x4(*A_as_matrix, b, x, flags);
        }
        else
        {
            smooth_4x4(*A_as_matrix, b, x, flags);
        }
    }
    else if (A_as_matrix->get_block_dimx() == A_as_matrix->get_block_dimy())
    {
        if (xIsZero)
        {
            amgx::thrust::fill(x.begin(), x.end(), types::util<ValueTypeB>::get_zero());
            cudaCheckError();
        }

        smooth_BxB(*A_as_matrix, b, x, true, flags);
    }
    else
    {
        FatalError("Unsupported block size for BlockJacobi_Solver", AMGX_ERR_NOT_SUPPORTED_BLOCKSIZE);
    }

    if (A_as_matrix->get_block_dimx() == 4 && A_as_matrix->get_block_dimy() == 4)
    {
        if (!xIsZero) // we write to t_res vector to avoid race condition in the kernel
        {
            x.swap(this->t_res);
        }
    }

    x.dirtybit = 1;
    A_as_matrix->setView(oldView);
    return this->converged( b, x );
}

template<class T_Config>
void
BlockJacobiSolver_Base<T_Config>::solve_finalize( VVector &b, VVector &x )
{}

template<class T_Config>
void BlockJacobiSolver_Base<T_Config>::computeDinv( Matrix<T_Config> &A)
{
    Matrix<T_Config> *A_as_matrix = (Matrix<T_Config> *) this->m_A;
    ViewType oldView = A.currentView();
    A.setView(A_as_matrix->getViewExterior());

    if (A.get_block_dimx() == 1 && A.get_block_dimy() == 1)
    {
        this->computeDinv_1x1(A);
    }
    else if (A.get_block_dimx() == 2 && A.get_block_dimy() == 2)
    {
        this->computeDinv_bxb<2>(A);
    }
    else if (A.get_block_dimx() == 3 && A.get_block_dimy() == 3)
    {
        this->computeDinv_3x3(A);
    }
    else if (A.get_block_dimx() == 4 && A.get_block_dimy() == 4)
    {
        this->computeDinv_4x4(A);
    }
    else if (A.get_block_dimx() == 5 && A.get_block_dimy() == 5)
    {
        this->computeDinv_bxb<5>(A);
    }
    else if (A.get_block_dimx() == A.get_block_dimy() && A.get_block_dimy() > 5)
    {
        this->computeDinv_Big(A, A.get_block_dimx());
    }

    A.setView(oldView);
}


// Method to compute the inverse of the diagonal blocks
template <AMGX_VecPrecision t_vecPrec, AMGX_MatPrecision t_matPrec, AMGX_IndPrecision t_indPrec>
void BlockJacobiSolver<TemplateConfig<AMGX_device, t_vecPrec, t_matPrec, t_indPrec> >::computeDinv_1x1(const Matrix_d &A)
{
    Matrix_d *A_as_matrix = (Matrix_d *) this->m_A;
// supports both diag
    this->Dinv.resize(A.get_num_rows()*A.get_block_dimx()*A.get_block_dimy(), types::util<ValueTypeA>::get_zero());

    if ( A_as_matrix->hasProps(DIAG) )
    {
        const int num_values = A_as_matrix->diagOffset() * A_as_matrix->get_block_size();
        amgx::thrust::copy( A_as_matrix->values.begin() + num_values, A_as_matrix->values.begin() + num_values + A_as_matrix->get_num_rows()*A_as_matrix->get_block_size(), this->Dinv.begin() );
        cudaCheckError();
    }
    else
    {
        find_diag( *A_as_matrix );
    }
}


// Method to compute the inverse of the diagonal blocks
template <AMGX_VecPrecision t_vecPrec, AMGX_MatPrecision t_matPrec, AMGX_IndPrecision t_indPrec>
void BlockJacobiSolver<TemplateConfig<AMGX_host, t_vecPrec, t_matPrec, t_indPrec> >::computeDinv_1x1(const Matrix_h &A)
{
    // Do nothing
}



// Method to compute the inverse of the diagonal blocks
template <AMGX_VecPrecision t_vecPrec, AMGX_MatPrecision t_matPrec, AMGX_IndPrecision t_indPrec>
void BlockJacobiSolver<TemplateConfig<AMGX_host, t_vecPrec, t_matPrec, t_indPrec> >::computeDinv_4x4(const Matrix_h &A)
{
    //FatalError("Block Jacobi smoother not implemented with this format, exiting");
    //std::cout << "Warning, CPU version doesn't store the inverse of the blocks, like the GPU version" << std::endl;
}

template <AMGX_VecPrecision t_vecPrec, AMGX_MatPrecision t_matPrec, AMGX_IndPrecision t_indPrec>
void BlockJacobiSolver<TemplateConfig<AMGX_host, t_vecPrec, t_matPrec, t_indPrec> >::computeDinv_3x3(const Matrix_h &A)
{
    FatalError("3*3 Block Jacobi smoother not implemented with host format, exiting", AMGX_ERR_NOT_IMPLEMENTED);
    //std::cout << "Warning, CPU version doesn't store the inverse of the blocks, like the GPU version" << std::endl;
}

template <AMGX_VecPrecision t_vecPrec, AMGX_MatPrecision t_matPrec, AMGX_IndPrecision t_indPrec>
void BlockJacobiSolver<TemplateConfig<AMGX_host, t_vecPrec, t_matPrec, t_indPrec> >::computeDinv_Big(const Matrix_h &A, const int bsize)
{
    FatalError("Big Block Jacobi smoother not implemented with host format, exiting", AMGX_ERR_NOT_IMPLEMENTED);
    //std::cout << "Warning, CPU version doesn't store the inverse of the blocks, like the GPU version" << std::endl;
}


// Finding diag on device, CSR format
template <AMGX_VecPrecision t_vecPrec, AMGX_MatPrecision t_matPrec, AMGX_IndPrecision t_indPrec>
void BlockJacobiSolver<TemplateConfig<AMGX_host, t_vecPrec, t_matPrec, t_indPrec> >::find_diag( const Matrix_h &A )
{
    //for each row
    for (int i = 0; i < A.get_num_rows(); i++)
    {
        //for each column
        for (int j = A.row_offsets[i]; j < A.row_offsets[i + 1]; j++)
        {
            if (A.col_indices[j] == i)
            {
                this->Dinv[i] = A.values[j];
                break;
            }

            if (j == A.row_offsets[i + 1] - 1)
            {
                FatalError("Could not find a diagonal value", AMGX_ERR_BAD_PARAMETERS);
            }
        }
    }
}


// Finding diag on device, CSR format
template <AMGX_VecPrecision t_vecPrec, AMGX_MatPrecision t_matPrec, AMGX_IndPrecision t_indPrec>
void BlockJacobiSolver<TemplateConfig<AMGX_device, t_vecPrec, t_matPrec, t_indPrec> >::find_diag( const Matrix_d &A )
{
    AMGX_CPU_PROFILER( "JacobiSolver::find_diag " );
    const size_t THREADS_PER_BLOCK  = 128;
    const size_t NUM_BLOCKS = min(AMGX_GRID_MAX_SIZE, (int)ceil((float)(A.get_num_rows()) / (float)(THREADS_PER_BLOCK)));
    find_diag_kernel_indexed_dia <<< (unsigned int)NUM_BLOCKS, (unsigned int)THREADS_PER_BLOCK>>>(
        A.get_num_rows(),
        A.diag.raw(),
        A.values.raw(),
        this->Dinv.raw());
    cudaCheckError();
}


// Method to compute the inverse of the diagonal blocks
template <AMGX_VecPrecision t_vecPrec, AMGX_MatPrecision t_matPrec, AMGX_IndPrecision t_indPrec>
void BlockJacobiSolver<TemplateConfig<AMGX_device, t_vecPrec, t_matPrec, t_indPrec> >::computeDinv_4x4(const Matrix_d &A)
{
// supports both diag
    this->Dinv.resize(A.get_num_rows()*A.get_block_dimx()*A.get_block_dimy(), types::util<ValueTypeA>::get_zero());
    const IndexType *A_row_offsets_ptr = A.row_offsets.raw();
    const IndexType *A_column_indices_ptr = A.col_indices.raw();
    const IndexType *A_dia_indices_ptr = A.diag.raw();
    const ValueTypeA *A_values = A.values.raw();
    ValueTypeA *Dinv_ptr = this->Dinv.raw();
#if 1
    const int threads_per_block = 512;
    const int halfwarps_per_block = threads_per_block / 16;
    const int num_blocks = min(AMGX_GRID_MAX_SIZE, (int) (A.get_num_rows() - 1) / halfwarps_per_block + 1);
    cudaFuncSetCacheConfig(setupBlockJacobiSmooth4by4BlockDiaCsrKernel_V2<ValueTypeA, ValueTypeB, IndexType, threads_per_block, halfwarps_per_block, 4, 2, 16, 4>, cudaFuncCachePreferL1);
    setupBlockJacobiSmooth4by4BlockDiaCsrKernel_V2<ValueTypeA, ValueTypeB, IndexType, threads_per_block, halfwarps_per_block, 4, 2, 16, 4> <<< num_blocks, threads_per_block>>>
    (A_dia_indices_ptr, A_values, Dinv_ptr, A.get_num_rows());
    cudaCheckError();
#else
    cudaFuncSetCacheConfig(matinv_matrix_per_thread_pivot<ValueTypeA, ValueTypeB, 4>, cudaFuncCachePreferL1);
    matinv_matrix_per_thread_pivot<ValueTypeA, ValueTypeB, 4> <<< (A.num_block_rows + 127) / 128, 128 >>> (A_dia_values_ptr, Dinv_ptr, A.num_block_rows);
    cudaCheckError();
#endif
}

template <AMGX_VecPrecision t_vecPrec, AMGX_MatPrecision t_matPrec, AMGX_IndPrecision t_indPrec>
void BlockJacobiSolver<TemplateConfig<AMGX_device, t_vecPrec, t_matPrec, t_indPrec> >::computeDinv_3x3(const Matrix_d &A)
{
    const int bsize = 3;
// supports both diag
    this->Dinv.resize(A.get_num_rows()*A.get_block_dimx()*A.get_block_dimy(), types::util<ValueTypeA>::get_zero());
    const IndexType *A_row_offsets_ptr = A.row_offsets.raw();
    const IndexType *A_column_indices_ptr = A.col_indices.raw();
    const IndexType *A_dia_idx_ptr = A.diag.raw();
    ValueTypeA *Dinv_ptr = this->Dinv.raw();
    const ValueTypeA *A_nonzero_values_ptr = A.values.raw();
    // MUST BE MULTIPLE OF 16
    const int threads_per_block = 256;
    const int blockrows_per_warp = 32 / (bsize * bsize);
    // blockrows per cta = blockrows_per_warp * number_of_warps_per_cta
    const int blockrows_per_cta = (threads_per_block / 32) * blockrows_per_warp ;
    const int num_blocks = min(AMGX_GRID_MAX_SIZE, (int) (A.get_num_rows() - 1) / blockrows_per_cta + 1);
    cudaFuncSetCacheConfig(setupBlockJacobiSmoothBbyBBlockDiaCsrKernel<IndexType, ValueTypeA, blockrows_per_cta, blockrows_per_warp, bsize, bsize *bsize>, cudaFuncCachePreferL1);
    setupBlockJacobiSmoothBbyBBlockDiaCsrKernel<IndexType, ValueTypeA, blockrows_per_cta, blockrows_per_warp, bsize, bsize *bsize> <<< num_blocks, threads_per_block>>>
    (A_row_offsets_ptr, A_column_indices_ptr, A_nonzero_values_ptr, A_dia_idx_ptr, Dinv_ptr, A.get_num_rows());
    cudaCheckError();
}

template <AMGX_VecPrecision t_vecPrec, AMGX_MatPrecision t_matPrec, AMGX_IndPrecision t_indPrec>
void BlockJacobiSolver<TemplateConfig<AMGX_device, t_vecPrec, t_matPrec, t_indPrec> >::computeDinv_Big(const Matrix_d &A, const int bsize)
{
//both DIAG supported
    this->Dinv.resize(A.get_num_rows()*A.get_block_dimx()*A.get_block_dimy(), types::util<ValueTypeA>::get_zero());
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
    cudaFuncSetCacheConfig(setupBlockJacobiSmoothbBigBlockDiaCsrKernel<IndexType, ValueTypeA, threads_per_block, halfwarps_per_block>, cudaFuncCachePreferL1);
    setupBlockJacobiSmoothbBigBlockDiaCsrKernel<IndexType, ValueTypeA, threads_per_block, halfwarps_per_block> <<< num_blocks, threads_per_block, sizeof(ValueTypeA)*bsize *bsize *halfwarps_per_block>>>
    (A_row_offsets_ptr, A_column_indices_ptr, A_nonzero_values_ptr, A_dia_idx_ptr, Dinv_ptr, A.get_num_rows(), bsize, bsize * bsize, temp_ptr);
    cudaCheckError();
}


template<class T_Config>
template<int bsize>
void BlockJacobiSolver_Base<T_Config>::computeDinv_bxb(const Matrix<T_Config> &A)
{
    if (TConfig::memSpace == AMGX_host)
    {
        FatalError("BlockJacobiSmooth Not implemented for host", AMGX_ERR_NOT_IMPLEMENTED);
    }
    else
    {
        // supports both diag
        this->Dinv.resize(A.get_num_rows()*A.get_block_dimx()*A.get_block_dimy(), types::util<ValueTypeA>::get_zero());
        const IndexType *A_row_offsets_ptr = A.row_offsets.raw();
        const IndexType *A_column_indices_ptr = A.col_indices.raw();
        const IndexType *A_dia_idx_ptr = A.diag.raw();
        ValueTypeA *Dinv_ptr = this->Dinv.raw();
        const ValueTypeA *A_nonzero_values_ptr = A.values.raw();
        // MUST BE MULTIPLE OF 16
        const int threads_per_block = 256;
        const int blockrows_per_warp = 32 / (bsize * bsize);
        const int blockrows_per_cta = (threads_per_block / 32) * blockrows_per_warp;
        const int num_blocks = min(AMGX_GRID_MAX_SIZE, (int) (A.get_num_rows() - 1) / blockrows_per_cta + 1);
        cudaFuncSetCacheConfig(setupBlockJacobiSmoothBbyBBlockDiaCsrKernel<IndexType, ValueTypeA, blockrows_per_cta, blockrows_per_warp, bsize, bsize *bsize>, cudaFuncCachePreferL1);
        setupBlockJacobiSmoothBbyBBlockDiaCsrKernel<IndexType, ValueTypeA, blockrows_per_cta, blockrows_per_warp, bsize, bsize *bsize> <<< num_blocks, threads_per_block>>>
        (A_row_offsets_ptr, A_column_indices_ptr, A_nonzero_values_ptr, A_dia_idx_ptr, Dinv_ptr, A.get_num_rows());
        cudaCheckError();
    }
}

template <AMGX_VecPrecision t_vecPrec, AMGX_MatPrecision t_matPrec, AMGX_IndPrecision t_indPrec>
void BlockJacobiSolver<TemplateConfig<AMGX_host, t_vecPrec, t_matPrec, t_indPrec> >::smooth_BxB(Matrix_h &A, VVector &b, VVector &x, bool firstStep, ViewType separation_flags)
{
    FatalError("M*M Block Jacobi smoother not implemented with host format, exiting", AMGX_ERR_NOT_IMPLEMENTED);
}


template <AMGX_VecPrecision t_vecPrec, AMGX_MatPrecision t_matPrec, AMGX_IndPrecision t_indPrec>
void BlockJacobiSolver<TemplateConfig<AMGX_host, t_vecPrec, t_matPrec, t_indPrec> >::smooth_4x4(const Matrix_h &A, const VVector &b, VVector &x, ViewType separation_flags)
{
    VVector newx(x.size());
    int bsize = A.get_block_dimx();
    // Allocate space for block_matrix
    ValueTypeA **E = new ValueTypeA* [bsize];

    for ( int i = 0; i < bsize; i++)
    {
        E[i] = new ValueTypeA[bsize];
    }

    ValueTypeB *bmAx = new ValueTypeB[bsize];
    ValueTypeB *temp = new ValueTypeB[bsize];

    //for each block row
    for (int i = 0; i < A.get_num_rows(); i++)
    {
        // Compute b - sum_j A_j x_j (denoted by bmAx) for block_row i
        // Load diagonal
        for (int m = 0; m < bsize; m++)
        {
            for (int n = 0; n < bsize; n++)
            {
                E[m][n] = A.values[A.diag[i * bsize * bsize + bsize * m + n]];
            }

            bmAx[m] = types::util<ValueTypeB>::get_zero();
            temp[m] = types::util<ValueTypeB>::get_zero();
        }

        // Contribution from diagonal
        for (int m = 0; m < bsize; m++)
            for (int n = 0; n < bsize; n++)
            {
                bmAx[m] = bmAx[m] - E[m][n] * x[i * bsize + n];
            }

        // Contribution from each nonzero column
        for (int j = A.row_offsets[i]; j < A.row_offsets[i + 1]; j++)
        {
            IndexType jcol = A.col_indices[j];

            for (int m = 0; m < bsize; m++)
                for (int n = 0; n < bsize; n++)
                {
                    bmAx[m] = bmAx[m]  - A.values[j * bsize * bsize + bsize * m + n] * x[jcol * bsize + n];
                }
        }

        // Add b
        for (int m = 0; m < bsize; m++)
        {
            bmAx[m] = bmAx[m] + b[i * bsize + m];
        }

        gaussianEliminationRowMajor(E, temp, bmAx, bsize);

        // Compute new value of x
        for (int m = 0; m < bsize; m++)
        {
            newx[i * bsize + m] = x[i * bsize + m] + temp[m] * this->weight;
        }
    }

    x.swap(newx);
}



template <AMGX_VecPrecision t_vecPrec, AMGX_MatPrecision t_matPrec, AMGX_IndPrecision t_indPrec>
void BlockJacobiSolver<TemplateConfig<AMGX_host, t_vecPrec, t_matPrec, t_indPrec> >::smooth_with_0_initial_guess_4x4(const Matrix_h &A, const VVector &b, VVector &x, ViewType separation_flags)
{
    IndexType bsize = A.get_block_dimy();
    ValueTypeA **E = new ValueTypeA* [bsize];

    for ( int i = 0; i < bsize; i++)
    {
        E[i] = new ValueTypeA[bsize];
    }

    ValueTypeB *rhs = new ValueTypeB[bsize];
    ValueTypeB *temp = new ValueTypeB[bsize];

    //for each block row
    for (int i = 0; i < A.get_num_rows(); i++)
    {
        // Load diagonal
        for (int m = 0; m < bsize; m++)
        {
            for (int n = 0; n < bsize; n++)
            {
                E[m][n] = A.values[A.diag[i * bsize * bsize + bsize * m + n]];
            }

            rhs[m] = types::util<ValueTypeB>::get_zero();
        }

        //rhs
        for (int m = 0; m < bsize; m++)
        {
            rhs[m] = rhs[m] + b[i * bsize + m];
        }

        // Solve for temp
        gaussianEliminationRowMajor(E, temp, rhs, bsize);

        for (int m = 0; m < bsize; m++)
        {
            x[i * bsize + m] = temp[m] * this->weight;
        }
    }
}

template <AMGX_VecPrecision t_vecPrec, AMGX_MatPrecision t_matPrec, AMGX_IndPrecision t_indPrec>
void BlockJacobiSolver<TemplateConfig<AMGX_host, t_vecPrec, t_matPrec, t_indPrec> >::smooth_1x1_const(const Matrix_h &A, const VVector &b, VVector &x, ViewType separation_flags)
{
    VVector newx(x.size());

    //for each row
    for (int i = 0; i < A.get_num_rows(); i++)
    {
        ValueTypeB Axi = types::util<ValueTypeB>::get_zero();
        ValueTypeB d = types::util<ValueTypeB>::get_one() * A.values[A.diag[i]];
        ValueTypeB mydiaginv = types::util<ValueTypeB>::get_one() * this->weight / (isNotCloseToZero(d) ? d : epsilon(d) );

        //for each column
        for (int j = A.row_offsets[i]; j < A.row_offsets[i + 1]; j++)
        {
            Axi = Axi + A.values[j] * x[A.col_indices[j]];
        }

        newx[i] = x[i] +  (b[i] - Axi) * mydiaginv ;
    }

    x.swap(newx);
}

template <AMGX_VecPrecision t_vecPrec, AMGX_MatPrecision t_matPrec, AMGX_IndPrecision t_indPrec>
void BlockJacobiSolver<TemplateConfig<AMGX_host, t_vecPrec, t_matPrec, t_indPrec> >::smooth_1x1(Matrix_h &A, VVector &b, VVector &x, ViewType separation_flags)
{
    this->smooth_1x1_const(A, b, x, separation_flags);
}

template <AMGX_VecPrecision t_vecPrec, AMGX_MatPrecision t_matPrec, AMGX_IndPrecision t_indPrec>
void BlockJacobiSolver<TemplateConfig<AMGX_device, t_vecPrec, t_matPrec, t_indPrec> >::smooth_1x1(Matrix_d &A, VVector &b, VVector &x, ViewType separation_flags)
{
    AMGX_CPU_PROFILER( "JacobiSolver::smooth_1x1 " );

    if (this->t_res.size() != x.size())
    {
        this->t_res.resize(x.size());
    }

    if (this->y.size() != b.size())
    {
        this->y.resize(b.size());
        this->y.tag = this->tag * 100 + 3;
        this->y.set_block_dimx(b.get_block_dimx());
        this->y.set_block_dimy(b.get_block_dimy());
    }

    int num_rows = A.get_num_rows();
    int offset = 0;
    A.getOffsetAndSizeForView(separation_flags, &offset, &num_rows);
    this->y.dirtybit = 0;
    multiply( A, x, this->y, separation_flags );
    amgx::thrust::transform( amgx::thrust::make_zip_iterator(amgx::thrust::make_tuple( x.begin() + offset, this->Dinv.begin() + offset, b.begin() + offset, this->y.begin() + offset)),
                       amgx::thrust::make_zip_iterator(amgx::thrust::make_tuple( x.begin() + A.get_num_rows(),   this->Dinv.begin() + A.get_num_rows(),   b.begin() + A.get_num_rows(),   this->y.begin() + A.get_num_rows())),
                       x.begin() + offset,
                       jacobi_postsmooth_functor<ValueTypeA, ValueTypeB>( this->weight ));
    cudaCheckError();
}


template <AMGX_VecPrecision t_vecPrec, AMGX_MatPrecision t_matPrec, AMGX_IndPrecision t_indPrec>
void BlockJacobiSolver<TemplateConfig<AMGX_host, t_vecPrec, t_matPrec, t_indPrec> >::smooth_with_0_initial_guess_1x1(const Matrix_h &A, const VVector &b, VVector &x, ViewType separation_flags)
{
    //for each row
    for (int i = 0; i < A.get_num_rows(); i++)
    {
        ValueTypeB d = types::util<ValueTypeB>::get_one() * A.values[A.diag[i]];
        ValueTypeB mydiag = types::util<ValueTypeB>::get_one() * this->weight / (isNotCloseToZero(d) ? d : epsilon(d));
        x[i] =  b[i] * mydiag;
    }
}

template <AMGX_VecPrecision t_vecPrec, AMGX_MatPrecision t_matPrec, AMGX_IndPrecision t_indPrec>
void BlockJacobiSolver<TemplateConfig<AMGX_device, t_vecPrec, t_matPrec, t_indPrec> >::smooth_with_0_initial_guess_1x1(const Matrix_d &A, const VVector &b, VVector &x, ViewType separation_flags)
{
    AMGX_CPU_PROFILER( "JacobiSolver::smooth_with_0_initial_guess_1x1 " );
    int num_rows = A.get_num_rows();
    int offset = 0;
    A.getOffsetAndSizeForView(separation_flags, &offset, &num_rows);
    amgx::thrust::transform( b.begin( ) + offset,
                       b.begin( ) + A.get_num_rows(),
                       this->Dinv.begin( ) + offset,
                       x.begin( ) + offset,
                       jacobi_presmooth_functor<ValueTypeA, ValueTypeB>( this->weight ));
    cudaCheckError();
}

template <AMGX_VecPrecision t_vecPrec, AMGX_MatPrecision t_matPrec, AMGX_IndPrecision t_indPrec>
void BlockJacobiSolver<TemplateConfig<AMGX_device, t_vecPrec, t_matPrec, t_indPrec> >::smooth_BxB(Matrix_d &A, VVector &b, VVector &x, bool firstStep, ViewType separation_flags)
{
    IndexType num_rows;
    IndexType offset;
    A.getOffsetAndSizeForView(separation_flags, &offset, &num_rows);

    // aux vector initialization
    if (this->y.size() != b.size())
    {
        this->y.resize(b.size());
        this->y.tag = this->tag * 100 + 3;
        this->y.set_block_dimx(b.get_block_dimx());
        this->y.set_block_dimy(b.get_block_dimy());
    }

    amgx::thrust::copy(b.begin(), b.end(), this->y.begin()); // copy of vector b
    cudaCheckError();
    Cusparse::bsrmv(types::util<ValueTypeB>::get_minus_one(), A, x, types::util<ValueTypeB>::get_one(), this->y, separation_flags);         // y= -1.0f*(A.x) + y
    cudaCheckError();

    Cusparse::bsrmv(types::util<ValueTypeB>::get_one() * this->weight, A, this->Dinv, this->y, types::util<ValueTypeB>::get_one(), x, separation_flags); // t_res = t_res + w*(Dinv.y) @ view
    cudaCheckError();
}

template <AMGX_VecPrecision t_vecPrec, AMGX_MatPrecision t_matPrec, AMGX_IndPrecision t_indPrec>
void BlockJacobiSolver<TemplateConfig<AMGX_device, t_vecPrec, t_matPrec, t_indPrec> >::smooth_4x4(const Matrix_d &A, const VVector &b, VVector &x, ViewType separation_flags)
{
    if (this->t_res.size() != x.size())
    {
        this->t_res.resize(x.size());
    }

    const IndexType *A_row_offsets_ptr = A.row_offsets.raw();
    const IndexType *A_column_indices_ptr = A.col_indices.raw();
    const IndexType *A_dia_idx_ptr = A.diag.raw();
    const ValueTypeA *Dinv_ptr = this->Dinv.raw();
    const ValueTypeA *A_nonzero_values_ptr = A.values.raw();
    const ValueTypeB *b_ptr = b.raw();
    ValueTypeB *x_ptr = x.raw();
    ValueTypeB *xout_ptr = this->t_res.raw(); // always store original x
    IndexType num_rows = A.get_num_rows();
    IndexType offset = 0;
    A.getOffsetAndSizeForView(separation_flags, &offset, &num_rows);
    const int threads_per_block = 512;
    const int eightwarps_per_block = threads_per_block / 4;
    const int num_blocks = min( AMGX_GRID_MAX_SIZE, (int) (num_rows - 1) / eightwarps_per_block + 1);
    cudaFuncSetCacheConfig(jacobiSmooth4by4BlockDiaCsrKernel_NAIVE_tex_readDinv2<IndexType, ValueTypeA, ValueTypeB, eightwarps_per_block, 4, 2, 2, 16>, cudaFuncCachePreferL1);
    jacobiSmooth4by4BlockDiaCsrKernel_NAIVE_tex_readDinv2<IndexType, ValueTypeA, ValueTypeB, eightwarps_per_block, 4, 2, 2, 16> <<< num_blocks, threads_per_block>>>
    (A_row_offsets_ptr, A_column_indices_ptr, A_dia_idx_ptr, A_nonzero_values_ptr, Dinv_ptr,
     b_ptr, x_ptr, this->weight, offset + num_rows, xout_ptr, offset);
    cudaCheckError();
}


template <AMGX_VecPrecision t_vecPrec, AMGX_MatPrecision t_matPrec, AMGX_IndPrecision t_indPrec>
void BlockJacobiSolver<TemplateConfig<AMGX_device, t_vecPrec, t_matPrec, t_indPrec> >::smooth_with_0_initial_guess_4x4(const Matrix_d &A, const VVector &b, VVector &x, ViewType separation_flags)
{
    cudaCheckError();
    const ValueTypeA *A_values_ptr = A.values.raw();
    const ValueTypeB *b_ptr = b.raw();
    const ValueTypeA *Dinv_ptr = this->Dinv.raw();
    ValueTypeB *x_ptr = x.raw();
    IndexType num_rows = A.get_num_rows();
    IndexType offset = 0;
    A.getOffsetAndSizeForView(separation_flags, &offset, &num_rows);
    const int threads_per_block = 512;
    const int eightwarps_per_block = threads_per_block / 4;
    const int num_blocks = min( AMGX_GRID_MAX_SIZE, (int) (num_rows - 1) / eightwarps_per_block + 1);
    cudaFuncSetCacheConfig(jacobiSmooth4by4ZeroBlockDiaCsrKernel_NAIVE_tex_readDinv2<IndexType, ValueTypeA, ValueTypeB, eightwarps_per_block, 4, 2, 2>, cudaFuncCachePreferL1);
    jacobiSmooth4by4ZeroBlockDiaCsrKernel_NAIVE_tex_readDinv2<IndexType, ValueTypeA, ValueTypeB, eightwarps_per_block, 4, 2, 2> <<< num_blocks, threads_per_block>>>
    (Dinv_ptr, b_ptr, this->weight, offset + num_rows, x_ptr, offset);
    cudaCheckError();
}

/****************************************
 * Explict instantiations
 ***************************************/
#define AMGX_CASE_LINE(CASE) template class BlockJacobiSolver_Base<TemplateMode<CASE>::Type>;
AMGX_FORALL_BUILDS(AMGX_CASE_LINE)
AMGX_FORCOMPLEX_BUILDS(AMGX_CASE_LINE)
#undef AMGX_CASE_LINE

#define AMGX_CASE_LINE(CASE) template class BlockJacobiSolver<TemplateMode<CASE>::Type>;
AMGX_FORALL_BUILDS(AMGX_CASE_LINE)
AMGX_FORCOMPLEX_BUILDS(AMGX_CASE_LINE)
#undef AMGX_CASE_LINE

} // namespace block_jacobi
} // namespace amgx
