// SPDX-FileCopyrightText: 2011 - 2024 NVIDIA CORPORATION. All Rights Reserved.
//
// SPDX-License-Identifier: BSD-3-Clause

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
    amgx::memory::cudaMallocAsync((void **)&ptr, sz);
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
