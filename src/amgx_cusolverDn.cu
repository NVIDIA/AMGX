// SPDX-FileCopyrightText: 2011 - 2024 NVIDIA CORPORATION. All Rights Reserved.
//
// SPDX-License-Identifier: BSD-3-Clause

#include <amgx_cusolverDn.h>

namespace amgx
{

//
// LU factorization
//
cusolverStatus_t cusolverDnXgetrf_bufferSize (cusolverDnHandle_t handle,
        int m,
        int n,
        float *A,
        int lda,
        int *Lwork )
{
    return cusolverDnSgetrf_bufferSize(handle, m, n, A, lda, Lwork);
}

cusolverStatus_t cusolverDnXgetrf_bufferSize (cusolverDnHandle_t handle,
        int m,
        int n,
        double *A,
        int lda,
        int *Lwork )
{
    return cusolverDnDgetrf_bufferSize(handle, m, n, A, lda, Lwork);
}

cusolverStatus_t cusolverDnXgetrf_bufferSize (cusolverDnHandle_t handle,
        int m,
        int n,
        cuComplex *A,
        int lda,
        int *Lwork )
{
    return cusolverDnCgetrf_bufferSize(handle, m, n, A, lda, Lwork);
}

cusolverStatus_t cusolverDnXgetrf_bufferSize (cusolverDnHandle_t handle,
        int m,
        int n,
        cuDoubleComplex *A,
        int lda,
        int *Lwork )
{
    return cusolverDnZgetrf_bufferSize(handle, m, n, A, lda, Lwork);
}

cusolverStatus_t cusolverDnXgetrf (cusolverDnHandle_t handle,
                                   int m,
                                   int n,
                                   float *A,
                                   int lda,
                                   float *wspace,
                                   int *devIpiv,
                                   int *info)
{
    return cusolverDnSgetrf(handle, m, n, A, lda, wspace, devIpiv, info);
}

cusolverStatus_t cusolverDnXgetrf (cusolverDnHandle_t handle,
                                   int m,
                                   int n,
                                   double *A,
                                   int lda,
                                   double *wspace,
                                   int *devIpiv,
                                   int *info)
{
    return cusolverDnDgetrf(handle, m, n, A, lda, wspace, devIpiv, info);
}

cusolverStatus_t cusolverDnXgetrf (cusolverDnHandle_t handle,
                                   int m,
                                   int n,
                                   cuComplex *A,
                                   int lda,
                                   cuComplex *wspace,
                                   int *devIpiv,
                                   int *info)
{
    return cusolverDnCgetrf(handle, m, n, A, lda, wspace, devIpiv, info);
}

cusolverStatus_t cusolverDnXgetrf (cusolverDnHandle_t handle,
                                   int m,
                                   int n,
                                   cuDoubleComplex *A,
                                   int lda,
                                   cuDoubleComplex *wspace,
                                   int *devIpiv,
                                   int *info)
{
    return cusolverDnZgetrf(handle, m, n, A, lda, wspace, devIpiv, info);
}

//
// solve
//
cusolverStatus_t cusolverDnXgetrs(cusolverDnHandle_t handle,
                                  cublasOperation_t trans,
                                  int n,
                                  int nrhs,
                                  const float *A,
                                  int lda,
                                  const int *devIpiv,
                                  float *B,
                                  int ldb,
                                  int *devInfo )
{
    return cusolverDnSgetrs(handle, trans, n, 1, A, lda, devIpiv, B, ldb, devInfo );
}

cusolverStatus_t cusolverDnXgetrs(cusolverDnHandle_t handle,
                                  cublasOperation_t trans,
                                  int n,
                                  int nrhs,
                                  const double *A,
                                  int lda,
                                  const int *devIpiv,
                                  double *B,
                                  int ldb,
                                  int *devInfo )
{
    return cusolverDnDgetrs(handle, trans, n, 1, A, lda, devIpiv, B, ldb, devInfo );
}

cusolverStatus_t cusolverDnXgetrs(cusolverDnHandle_t handle,
                                  cublasOperation_t trans,
                                  int n,
                                  int nrhs,
                                  const cuComplex *A,
                                  int lda,
                                  const int *devIpiv,
                                  cuComplex *B,
                                  int ldb,
                                  int *devInfo )
{
    return cusolverDnCgetrs(handle, trans, n, 1, A, lda, devIpiv, B, ldb, devInfo );
}

cusolverStatus_t cusolverDnXgetrs(cusolverDnHandle_t handle,
                                  cublasOperation_t trans,
                                  int n,
                                  int nrhs,
                                  const cuDoubleComplex *A,
                                  int lda,
                                  const int *devIpiv,
                                  cuDoubleComplex *B,
                                  int ldb,
                                  int *devInfo )
{
    return cusolverDnZgetrs(handle, trans, n, 1, A, lda, devIpiv, B, ldb, devInfo );
}

} // namespace amgx