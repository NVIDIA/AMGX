// SPDX-FileCopyrightText: 2011 - 2025 NVIDIA CORPORATION. All Rights Reserved.
//
// SPDX-License-Identifier: BSD-3-Clause

//
// LU factorization
//

#pragma once

#include "cusolverDn.h"

namespace amgx
{

cusolverStatus_t
cusolverDnXgetrf_bufferSize (cusolverDnHandle_t handle,
                             int m,
                             int n,
                             float *A,
                             int lda,
                             int *Lwork );
cusolverStatus_t
cusolverDnXgetrf_bufferSize (cusolverDnHandle_t handle,
                             int m,
                             int n,
                             double *A,
                             int lda,
                             int *Lwork );

cusolverStatus_t
cusolverDnXgetrf_bufferSize (cusolverDnHandle_t handle,
                             int m,
                             int n,
                             cuComplex *A,
                             int lda,
                             int *Lwork );

cusolverStatus_t
cusolverDnXgetrf_bufferSize (cusolverDnHandle_t handle,
                             int m,
                             int n,
                             cuDoubleComplex *A,
                             int lda,
                             int *Lwork );

cusolverStatus_t
cusolverDnXgetrf (cusolverDnHandle_t handle,
                  int m,
                  int n,
                  float *A,
                  int lda,
                  float *wspace,
                  int *devIpiv,
                  int *info);

cusolverStatus_t
cusolverDnXgetrf (cusolverDnHandle_t handle,
                  int m,
                  int n,
                  double *A,
                  int lda,
                  double *wspace,
                  int *devIpiv,
                  int *info);
cusolverStatus_t
cusolverDnXgetrf (cusolverDnHandle_t handle,
                  int m,
                  int n,
                  cuComplex *A,
                  int lda,
                  cuComplex *wspace,
                  int *devIpiv,
                  int *info);

cusolverStatus_t
cusolverDnXgetrf (cusolverDnHandle_t handle,
                  int m,
                  int n,
                  cuDoubleComplex *A,
                  int lda,
                  cuDoubleComplex *wspace,
                  int *devIpiv,
                  int *info);

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
                                  int *devInfo );

cusolverStatus_t cusolverDnXgetrs(cusolverDnHandle_t handle,
                                  cublasOperation_t trans,
                                  int n,
                                  int nrhs,
                                  const double *A,
                                  int lda,
                                  const int *devIpiv,
                                  double *B,
                                  int ldb,
                                  int *devInfo );

cusolverStatus_t cusolverDnXgetrs(cusolverDnHandle_t handle,
                                  cublasOperation_t trans,
                                  int n,
                                  int nrhs,
                                  const cuComplex *A,
                                  int lda,
                                  const int *devIpiv,
                                  cuComplex *B,
                                  int ldb,
                                  int *devInfo );

cusolverStatus_t cusolverDnXgetrs(cusolverDnHandle_t handle,
                                  cublasOperation_t trans,
                                  int n,
                                  int nrhs,
                                  const cuDoubleComplex *A,
                                  int lda,
                                  const int *devIpiv,
                                  cuDoubleComplex *B,
                                  int ldb,
                                  int *devInfo );

} // namespace amgx