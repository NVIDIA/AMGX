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

#pragma once
#include <energymin/selectors/em_selector.h>
#include <energymin/interpolators/em_interpolator.h>

#include <amgx_cusolverDn.h>

namespace amgx
{
namespace energymin
{

/*
 * hold general routines
 */

struct is_coarse
{
    __host__ __device__
    int operator()(const int &x)
    {
        return (int) (x == COARSE);
    }
};

struct square
{
    __host__ __device__
    int operator()(const int &x)
    {
        return x * x;
    }
};


template< typename DataType, typename IndexType >
void allocMem(DataType *&ptr,
              IndexType numEntry,
              bool initToZero)
{
    if ( ptr != NULL ) { amgx::memory::cudaFreeAsync(ptr); }

    cudaCheckError();
    size_t sz = numEntry * sizeof(DataType);
    if(sz == 0) { return; }
    amgx::memory::cudaMalloc((void **)&ptr, sz);
    cudaCheckError();

    if (initToZero)
    {
        cudaMemset(ptr, 0x0, sz);
        cudaCheckError();
    }
}


/* LU factorization */
/*
static inline
cudsStatus_t cudsXgetrf(cudsHandle_t handle,
                        int m, int n, float *A, int lda,
                        int *devIpiv, int *devInfo)
{
  return cudsSgetrf(handle, m, n, A, lda, devIpiv, devInfo);
}

static inline
cudsStatus_t cudsXgetrf(cudsHandle_t handle,
                        int m, int n, double *A, int lda,
                        int *devIpiv, int *devInfo)
{
  return cudsDgetrf(handle, m, n, A, lda, devIpiv, devInfo);
}



static inline
cudsStatus_t cudsXgetrs(cudsHandle_t handle,
                        int m, int n, const float *A, int lda,
                        const int *devIpiv, float *B, int ldb, int *info)
{
  return cudsSgetrs(handle, CUDS_OP_N, m, n, A, lda, devIpiv, B, ldb, info);
}

static inline
cudsStatus_t cudsXgetrs(cudsHandle_t handle,
                        int m, int n, const double *A, int lda,
                        const int *devIpiv, double *B, int ldb, int *info)
{
  return cudsDgetrs(handle, CUDS_OP_N, m, n, A, lda, devIpiv, B, ldb, info);
}
*/


/*
 * Kernels used in em.cu
 */
/*
__global__
void  count_Ma_nnz_per_row_kernel(const int *PcolOffsets, const int *ProwInd, int PnumCols,
                                  int *Ma_nnzPerRow, int AnumRows);

__global__
void  get_unsorted_MaColInd_kernel( const int *PcolOffsets, const int *ProwInd, int PnumCols,
                                    const int *MaRowOffsets, int *MaColInd, int AnumRows);
*/

/*
__global__
void  count_Ma_nnz_per_row_kernel(const int *PcolOffsets, const int *ProwInd, const int PnumCols,
                                  int *Ma_nnzPerRow, const int AnumRows);

__global__
void  get_unsorted_MaColInd_kernel( const int *PcolOffsets, const int *ProwInd, const int PnumCols,
                                    const int *MaRowOffsets, int *MaColInd, const int AnumRows);
*/

} // namespace energymin
} // namespace amgx
