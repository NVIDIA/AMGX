// SPDX-FileCopyrightText: 2013 - 2025 NVIDIA CORPORATION. All Rights Reserved.
//
// SPDX-License-Identifier: BSD-3-Clause

#include <amgx_cublas.h>
#ifdef AMGX_USE_LAPACK
#include "mkl.h"
#endif
namespace amgx
{

cublasHandle_t Cublas::m_handle = 0;

namespace
{
// real valued calls
cublasStatus_t cublas_axpy(cublasHandle_t handle, int n,
                           const float *alpha,
                           const float *x, int incx,
                           float *y, int incy)
{
    return cublasSaxpy(handle, n, alpha, x, incx, y, incy);
}

cublasStatus_t cublas_axpy(cublasHandle_t handle, int n,
                           const double *alpha,
                           const double *x, int incx,
                           double *y, int incy)
{
    return cublasDaxpy(handle, n, alpha, x, incx, y, incy);
}

cublasStatus_t cublas_copy(cublasHandle_t handle, int n,
                           const float *x, int incx,
                           float *y, int incy)
{
    return cublasScopy(handle, n, x, incx, y, incy);
}

cublasStatus_t cublas_copy(cublasHandle_t handle, int n,
                           const double *x, int incx,
                           double *y, int incy)
{
    return cublasDcopy(handle, n, x, incx, y, incy);
}

cublasStatus_t cublas_dot(cublasHandle_t handle, int n,
                          const float *x, int incx, const float *y, int incy,
                          float *result)
{
    return cublasSdot(handle, n, x, incx, y, incy, result);
}

cublasStatus_t cublas_dot(cublasHandle_t handle, int n,
                          const double *x, int incx, const double *y, int incy,
                          double *result)
{
    return cublasDdot(handle, n, x, incx, y, incy, result);
}

cublasStatus_t cublas_dotc(cublasHandle_t handle, int n,
                           const float *x, int incx, const float *y, int incy,
                           float *result)
{
    return cublasSdot(handle, n, x, incx, y, incy, result);
}

cublasStatus_t cublas_dotc(cublasHandle_t handle, int n,
                           const double *x, int incx, const double *y, int incy,
                           double *result)
{
    return cublasDdot(handle, n, x, incx, y, incy, result);
}


cublasStatus_t cublas_trsv_v2(cublasHandle_t handle,
                              cublasFillMode_t uplo,
                              cublasOperation_t trans,
                              cublasDiagType_t diag,
                              int n,
                              const float *A,
                              int lda,
                              float *x,
                              int incx)
{
    return cublasStrsv (handle, uplo, trans, diag, n, A, lda, x, incx);
}
cublasStatus_t cublas_trsv_v2(cublasHandle_t handle,
                              cublasFillMode_t uplo,
                              cublasOperation_t trans,
                              cublasDiagType_t diag,
                              int n,
                              const double *A,
                              int lda,
                              double *x,
                              int incx)
{
    return cublasDtrsv (handle, uplo, trans, diag, n, A, lda, x, incx);
}

cublasStatus_t cublas_gemm(cublasHandle_t handle,
                           cublasOperation_t transa, cublasOperation_t transb,
                           int m, int n, int k,
                           const float           *alpha,
                           const float           *A, int lda,
                           const float           *B, int ldb,
                           const float           *beta,
                           float           *C, int ldc)
{
    return cublasSgemm(handle, transa, transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
}

cublasStatus_t cublas_gemm(cublasHandle_t handle,
                           cublasOperation_t transa, cublasOperation_t transb,
                           int m, int n, int k,
                           const double          *alpha,
                           const double          *A, int lda,
                           const double          *B, int ldb,
                           const double          *beta,
                           double          *C, int ldc)
{
    return cublasDgemm(handle, transa, transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
}

cublasStatus_t cublas_gemv(cublasHandle_t handle, cublasOperation_t trans, int m, int n,
                           const float *alpha, const float *A, int lda,
                           const float *x, int incx,
                           const float *beta, float *y, int incy)
{
    return cublasSgemv(handle, trans, m, n, alpha, A, lda, x, incx, beta, y, incy);
}

cublasStatus_t cublas_gemv(cublasHandle_t handle, cublasOperation_t trans, int m, int n,
                           const double *alpha, const double *A, int lda,
                           const double *x, int incx,
                           const double *beta, double *y, int incy)
{
    return cublasDgemv(handle, trans, m, n, alpha, A, lda, x, incx, beta, y, incy);
}

cublasStatus_t cublas_ger(cublasHandle_t handle, int m, int n,
                          const float *alpha,
                          const float *x, int incx,
                          const float *y, int incy,
                          float *A, int lda)
{
    return cublasSger(handle, m, n, alpha, x, incx, y, incy, A, lda);
}

cublasStatus_t cublas_ger(cublasHandle_t handle, int m, int n,
                          const double *alpha,
                          const double *x, int incx,
                          const double *y, int incy,
                          double *A, int lda)
{
    return cublasDger(handle, m, n, alpha, x, incx, y, incy, A, lda);
}
cublasStatus_t cublas_gerc(cublasHandle_t handle, int m, int n,
                           const float *alpha,
                           const float *x, int incx,
                           const float *y, int incy,
                           float *A, int lda)
{
    return cublasSger(handle, m, n, alpha, x, incx, y, incy, A, lda);
}

cublasStatus_t cublas_gerc(cublasHandle_t handle, int m, int n,
                           const double *alpha,
                           const double *x, int incx,
                           const double *y, int incy,
                           double *A, int lda)
{
    return cublasDger(handle, m, n, alpha, x, incx, y, incy, A, lda);
}

cublasStatus_t cublas_nrm2(cublasHandle_t handle, int n,
                           const float *x, int incx, float *result)
{
    return cublasSnrm2(handle, n, x, incx, result);
}

cublasStatus_t cublas_nrm2(cublasHandle_t handle, int n,
                           const double *x, int incx, double *result)
{
    return cublasDnrm2(handle, n, x, incx, result);
}

cublasStatus_t cublas_scal(cublasHandle_t handle, int n,
                           const float *alpha,
                           float *x, int incx)
{
    return cublasSscal(handle, n, alpha, x, incx);
}

cublasStatus_t cublas_scal(cublasHandle_t handle, int n,
                           const double *alpha,
                           double *x, int incx)
{
    return cublasDscal(handle, n, alpha, x, incx);
}


// complex valued calls
cublasStatus_t cublas_axpy(cublasHandle_t handle, int n,
                           const cuComplex *alpha,
                           const cuComplex *x, int incx,
                           cuComplex *y, int incy)
{
    return cublasCaxpy(handle, n, alpha, x, incx, y, incy);
}

cublasStatus_t cublas_axpy(cublasHandle_t handle, int n,
                           const cuDoubleComplex *alpha,
                           const cuDoubleComplex *x, int incx,
                           cuDoubleComplex *y, int incy)
{
    return cublasZaxpy(handle, n, alpha, x, incx, y, incy);
}

cublasStatus_t cublas_copy(cublasHandle_t handle, int n,
                           const cuComplex *x, int incx,
                           cuComplex *y, int incy)
{
    return cublasCcopy(handle, n, x, incx, y, incy);
}

cublasStatus_t cublas_copy(cublasHandle_t handle, int n,
                           const cuDoubleComplex *x, int incx,
                           cuDoubleComplex *y, int incy)
{
    return cublasZcopy(handle, n, x, incx, y, incy);
}

cublasStatus_t cublas_dot(cublasHandle_t handle, int n,
                          const cuComplex *x, int incx, const cuComplex *y, int incy,
                          cuComplex *result)
{
    return cublasCdotu(handle, n, x, incx, y, incy, result);
}

cublasStatus_t cublas_dot(cublasHandle_t handle, int n,
                          const cuDoubleComplex *x, int incx, const cuDoubleComplex *y, int incy,
                          cuDoubleComplex *result)
{
    return cublasZdotu(handle, n, x, incx, y, incy, result);
}

cublasStatus_t cublas_dotc(cublasHandle_t handle, int n,
                           const cuComplex *x, int incx, const cuComplex *y, int incy,
                           cuComplex *result)
{
    return cublasCdotc(handle, n, x, incx, y, incy, result);
}

cublasStatus_t cublas_dotc(cublasHandle_t handle, int n,
                           const cuDoubleComplex *x, int incx, const cuDoubleComplex *y, int incy,
                           cuDoubleComplex *result)
{
    return cublasZdotc(handle, n, x, incx, y, incy, result);
}


cublasStatus_t cublas_trsv_v2(cublasHandle_t handle,
                              cublasFillMode_t uplo,
                              cublasOperation_t trans,
                              cublasDiagType_t diag,
                              int n,
                              const cuComplex *A,
                              int lda,
                              cuComplex *x,
                              int incx)
{
    return cublasCtrsv (handle, uplo, trans, diag, n, A, lda, x, incx);
}
cublasStatus_t cublas_trsv_v2(cublasHandle_t handle,
                              cublasFillMode_t uplo,
                              cublasOperation_t trans,
                              cublasDiagType_t diag,
                              int n,
                              const cuDoubleComplex *A,
                              int lda,
                              cuDoubleComplex *x,
                              int incx)
{
    return cublasZtrsv (handle, uplo, trans, diag, n, A, lda, x, incx);
}

cublasStatus_t cublas_gemm(cublasHandle_t handle,
                           cublasOperation_t transa, cublasOperation_t transb,
                           int m, int n, int k,
                           const cuComplex           *alpha,
                           const cuComplex           *A, int lda,
                           const cuComplex           *B, int ldb,
                           const cuComplex           *beta,
                           cuComplex           *C, int ldc)
{
    return cublasCgemm(handle, transa, transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
}

cublasStatus_t cublas_gemm(cublasHandle_t handle,
                           cublasOperation_t transa, cublasOperation_t transb,
                           int m, int n, int k,
                           const cuDoubleComplex          *alpha,
                           const cuDoubleComplex          *A, int lda,
                           const cuDoubleComplex          *B, int ldb,
                           const cuDoubleComplex          *beta,
                           cuDoubleComplex          *C, int ldc)
{
    return cublasZgemm(handle, transa, transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
}

cublasStatus_t cublas_gemv(cublasHandle_t handle, cublasOperation_t trans, int m, int n,
                           const cuComplex *alpha, const cuComplex *A, int lda,
                           const cuComplex *x, int incx,
                           const cuComplex *beta, cuComplex *y, int incy)
{
    return cublasCgemv(handle, trans, m, n, alpha, A, lda, x, incx, beta, y, incy);
}

cublasStatus_t cublas_gemv(cublasHandle_t handle, cublasOperation_t trans, int m, int n,
                           const cuDoubleComplex *alpha, const cuDoubleComplex *A, int lda,
                           const cuDoubleComplex *x, int incx,
                           const cuDoubleComplex *beta, cuDoubleComplex *y, int incy)
{
    return cublasZgemv(handle, trans, m, n, alpha, A, lda, x, incx, beta, y, incy);
}

cublasStatus_t cublas_ger(cublasHandle_t handle, int m, int n,
                          const cuComplex *alpha,
                          const cuComplex *x, int incx,
                          const cuComplex *y, int incy,
                          cuComplex *A, int lda)
{
    return cublasCgeru(handle, m, n, alpha, x, incx, y, incy, A, lda);
}

cublasStatus_t cublas_ger(cublasHandle_t handle, int m, int n,
                          const cuDoubleComplex *alpha,
                          const cuDoubleComplex *x, int incx,
                          const cuDoubleComplex *y, int incy,
                          cuDoubleComplex *A, int lda)
{
    return cublasZgeru(handle, m, n, alpha, x, incx, y, incy, A, lda);
}
cublasStatus_t cublas_gerc(cublasHandle_t handle, int m, int n,
                           const cuComplex *alpha,
                           const cuComplex *x, int incx,
                           const cuComplex *y, int incy,
                           cuComplex *A, int lda)
{
    return cublasCgerc(handle, m, n, alpha, x, incx, y, incy, A, lda);
}

cublasStatus_t cublas_gerc(cublasHandle_t handle, int m, int n,
                           const cuDoubleComplex *alpha,
                           const cuDoubleComplex *x, int incx,
                           const cuDoubleComplex *y, int incy,
                           cuDoubleComplex *A, int lda)
{
    return cublasZgerc(handle, m, n, alpha, x, incx, y, incy, A, lda);
}

cublasStatus_t cublas_nrm2(cublasHandle_t handle, int n,
                           const cuComplex *x, int incx, float *result)
{
    return cublasScnrm2(handle, n, x, incx, result);
}

cublasStatus_t cublas_nrm2(cublasHandle_t handle, int n,
                           const cuDoubleComplex *x, int incx, double *result)
{
    return cublasDznrm2(handle, n, x, incx, result);
}

cublasStatus_t cublas_scal(cublasHandle_t handle, int n,
                           const cuComplex *alpha,
                           cuComplex *x, int incx)
{
    return cublasCscal(handle, n, alpha, x, incx);
}

cublasStatus_t cublas_scal(cublasHandle_t handle, int n,
                           const cuDoubleComplex *alpha,
                           cuDoubleComplex *x, int incx)
{
    return cublasZscal(handle, n, alpha, x, incx);
}

cublasStatus_t cublas_scal(cublasHandle_t handle, int n,
                           const float *alpha,
                           cuComplex *x, int incx)
{
    return cublasCsscal(handle, n, alpha, x, incx);
}

cublasStatus_t cublas_scal(cublasHandle_t handle, int n,
                           const double *alpha,
                           cuDoubleComplex *x, int incx)
{
    return cublasZdscal(handle, n, alpha, x, incx);
}

} // anonymous namespace.

void Cublas::set_pointer_mode_device()
{
    cublasHandle_t handle = Cublas::get_handle();
    cublasSetPointerMode(handle, CUBLAS_POINTER_MODE_DEVICE);
}

void Cublas::set_pointer_mode_host()
{
    cublasHandle_t handle = Cublas::get_handle();
    cublasSetPointerMode(handle, CUBLAS_POINTER_MODE_HOST);
}

template <class TConfig>
void Cublas::gemm(typename TConfig::VecPrec alpha,
                  const Vector<TConfig> &A, const Vector<TConfig> &B,
                  typename TConfig::VecPrec beta, Vector<TConfig> &C,
                  bool A_transposed, bool B_transposed)
{
    cublasOperation_t trans_A = A_transposed ? CUBLAS_OP_T : CUBLAS_OP_N;
    cublasOperation_t trans_B = B_transposed ? CUBLAS_OP_T : CUBLAS_OP_N;
    int m = A_transposed ? A.get_num_cols() : A.get_num_rows();
    int n = B_transposed ? B.get_num_rows() : B.get_num_cols();
    int k = A_transposed ? A.get_num_rows() : A.get_num_cols();
    cublasHandle_t handle = Cublas::get_handle();
    cublasCheckError(cublas_gemm(handle, trans_A, trans_B,
                                 m, n, k,
                                 &alpha, A.raw(), A.get_lda(),
                                 B.raw(), B.get_lda(),
                                 &beta, C.raw(), C.get_lda()));
    C.dirtybit = 1;
}

template <typename T>
void Cublas::axpy(int n, T alpha,
                  const T *x, int incx,
                  T *y, int incy)
{
    cublasHandle_t handle = Cublas::get_handle();
    cublasCheckError(cublas_axpy(handle, n, &alpha, x, incx, y, incy));
}

template <typename T>
void Cublas::copy(int n, const T *x, int incx,
                  T *y, int incy)
{
    cublasHandle_t handle = Cublas::get_handle();
    cublasCheckError(cublas_copy(handle, n, x, incx, y, incy));
}

template <typename T>
void Cublas::dot(int n, const T *x, int incx,
                 const T *y, int incy,
                 T *result)
{
    cublasHandle_t handle = Cublas::get_handle();
    cublasCheckError(cublas_dot(handle, n, x, incx, y, incy, result));
}

template <typename T>
void Cublas::dotc(int n, const T *x, int incx,
                  const T *y, int incy,
                  T *result)
{
    cublasHandle_t handle = Cublas::get_handle();
    cublasCheckError(cublas_dotc(handle, n, x, incx, y, incy, result));
}

template <typename T, typename V>
V Cublas::nrm2(int n, const T *x, int incx)
{
    cublasHandle_t handle = Cublas::get_handle();
    V result;
    Cublas::nrm2(n, x, incx, &result);
    return result;
}

template <typename T, typename V>
void Cublas::nrm2(int n, const T *x, int incx, V *result)
{
    cublasHandle_t handle = Cublas::get_handle();
    cublasCheckError(cublas_nrm2(handle, n, x, incx, result));
}

template <typename T, typename V>
void Cublas::scal(int n, T alpha, V *x, int incx)
{
    Cublas::scal(n, &alpha, x, incx);
}

template <typename T, typename V>
void Cublas::scal(int n, T *alpha, V *x, int incx)
{
    cublasHandle_t handle = Cublas::get_handle();
    cublasCheckError(cublas_scal(handle, n, alpha, x, incx));
}

template <typename T>
void Cublas::gemv(bool transposed, int m, int n,
                  const T *alpha, const T *A, int lda,
                  const T *x, int incx,
                  const T *beta, T *y, int incy)
{
    cublasHandle_t handle = Cublas::get_handle();
    cublasOperation_t trans = transposed ? CUBLAS_OP_T : CUBLAS_OP_N;
    cublasCheckError(cublas_gemv(handle, trans, m, n, alpha, A, lda,
                                 x, incx, beta, y, incy));
}

template <typename T>
void Cublas::gemv_ext(bool transposed, const int m, const int n,
                      const T *alpha, const T *A, const int lda,
                      const T *x, const int incx,
                      const T *beta, T *y, const int incy, const int offsetx, const int offsety, const int offseta)
{
    cublasHandle_t handle = Cublas::get_handle();
    cublasOperation_t trans = transposed ? CUBLAS_OP_T : CUBLAS_OP_N;
    cublasCheckError(cublas_gemv(handle, trans, m, n, alpha, A + offseta, lda,
                                 x + offsetx, incx, beta, y + offsety, incy));
}

template <typename T>
void Cublas::trsv_v2( cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag, int n,
                      const T *A, int lda, T *x, int incx, int offseta)
{
    cublasHandle_t handle = Cublas::get_handle();
    cublasCheckError( cublas_trsv_v2(handle, uplo, trans, diag, n, A + offseta, lda, x, incx));
}


template <typename T>
void Cublas::ger(int m, int n, const T *alpha,
                 const T *x, int incx,
                 const T *y, int incy,
                 T *A, int lda)
{
    cublasHandle_t handle = Cublas::get_handle();
    cublasCheckError(cublas_ger(handle, m, n, alpha, x, incx, y, incy, A, lda));
}

template <typename T>
void Cublas::gerc(int m, int n, const T *alpha,
                  const T *x, int incx,
                  const T *y, int incy,
                  T *A, int lda)
{
    cublasHandle_t handle = Cublas::get_handle();
    cublasCheckError(cublas_gerc(handle, m, n, alpha, x, incx, y, incy, A, lda));
}

#define AMGX_CASE_LINE(CASE) \
    template void Cublas::gemm(typename TemplateMode<CASE>::Type::VecPrec, const Vector<TemplateMode<CASE>::Type>&, const Vector<TemplateMode<CASE>::Type>&, typename TemplateMode<CASE>::Type::VecPrec, Vector<TemplateMode<CASE>::Type>&, bool, bool);
AMGX_FORALL_BUILDS(AMGX_CASE_LINE)
AMGX_FORCOMPLEX_BUILDS(AMGX_CASE_LINE)
#undef AMGX_CASE_LINE

// real valued instantiaions
template void Cublas::axpy(int n, float alpha,
                           const float *x, int incx,
                           float *y, int incy);
template void Cublas::axpy(int n, double alpha,
                           const double *x, int incx,
                           double *y, int incy);

template void Cublas::copy(int n, const float *x, int incx, float *y, int incy);
template void Cublas::copy(int n, const double *x, int incx, double *y, int incy);

template void Cublas::dot(int n, const float *x, int incx,
                          const float *y, int incy,
                          float *result);
template void Cublas::dot(int n, const double *x, int incx,
                          const double *y, int incy,
                          double *result);
template void Cublas::dotc(int n, const float *x, int incx,
                           const float *y, int incy,
                           float *result);
template void Cublas::dotc(int n, const double *x, int incx,
                           const double *y, int incy,
                           double *result);

template void Cublas::gemv(bool transposed, int m, int n,
                           const float *alpha, const float *A, int lda,
                           const float *x, int incx,
                           const float *beta, float *y, int incy);
template void Cublas::gemv(bool transposed, int m, int n,
                           const double *alpha, const double *A, int lda,
                           const double *x, int incx,
                           const double *beta, double *y, int incy);

template void Cublas::ger(int m, int n, const float *alpha,
                          const float *x, int incx,
                          const float *y, int incy,
                          float *A, int lda);
template void Cublas::ger(int m, int n, const double *alpha,
                          const double *x, int incx,
                          const double *y, int incy,
                          double *A, int lda);
template void Cublas::gerc(int m, int n, const float *alpha,
                           const float *x, int incx,
                           const float *y, int incy,
                           float *A, int lda);
template void Cublas::gerc(int m, int n, const double *alpha,
                           const double *x, int incx,
                           const double *y, int incy,
                           double *A, int lda);


template void Cublas::gemv_ext(bool transposed, const int m, const int n,
                               const float *alpha, const float *A, const int lda,
                               const float *x, const int incx,
                               const float *beta, float *y, const int incy, const int offsetx, const int offsety, const int offseta);
template void Cublas::gemv_ext(bool transposed, const int m, const int n,
                               const double *alpha, const double *A, const int lda,
                               const double *x, const int incx,
                               const double *beta, double *y, const int incy, const int offsetx, const int offsety, const int offseta);


template void Cublas::trsv_v2( cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag, int n,
                               const float *A, int lda, float *x, int incx, int offseta);
template void Cublas::trsv_v2( cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag, int n,
                               const double *A, int lda, double *x, int incx, int offseta);

template double Cublas::nrm2(int n, const double *x, int incx);
template float Cublas::nrm2(int n, const float *x, int incx);

template void Cublas::scal(int n, float alpha, float *x, int incx);
template void Cublas::scal(int n, double alpha, double *x, int incx);

// complex valued instantiaions
template void Cublas::axpy(int n, cuComplex alpha,
                           const cuComplex *x, int incx,
                           cuComplex *y, int incy);
template void Cublas::axpy(int n, cuDoubleComplex alpha,
                           const cuDoubleComplex *x, int incx,
                           cuDoubleComplex *y, int incy);

template void Cublas::copy(int n, const cuComplex *x, int incx, cuComplex *y, int incy);
template void Cublas::copy(int n, const cuDoubleComplex *x, int incx, cuDoubleComplex *y, int incy);

template void Cublas::dot(int n, const cuComplex *x, int incx,
                          const cuComplex *y, int incy,
                          cuComplex *result);
template void Cublas::dot(int n, const cuDoubleComplex *x, int incx,
                          const cuDoubleComplex *y, int incy,
                          cuDoubleComplex *result);
template void Cublas::dotc(int n, const cuComplex *x, int incx,
                           const cuComplex *y, int incy,
                           cuComplex *result);
template void Cublas::dotc(int n, const cuDoubleComplex *x, int incx,
                           const cuDoubleComplex *y, int incy,
                           cuDoubleComplex *result);

template void Cublas::gemv(bool transposed, int m, int n,
                           const cuComplex *alpha, const cuComplex *A, int lda,
                           const cuComplex *x, int incx,
                           const cuComplex *beta, cuComplex *y, int incy);
template void Cublas::gemv(bool transposed, int m, int n,
                           const cuDoubleComplex *alpha, const cuDoubleComplex *A, int lda,
                           const cuDoubleComplex *x, int incx,
                           const cuDoubleComplex *beta, cuDoubleComplex *y, int incy);

template void Cublas::ger(int m, int n, const cuComplex *alpha,
                          const cuComplex *x, int incx,
                          const cuComplex *y, int incy,
                          cuComplex *A, int lda);
template void Cublas::ger(int m, int n, const cuDoubleComplex *alpha,
                          const cuDoubleComplex *x, int incx,
                          const cuDoubleComplex *y, int incy,
                          cuDoubleComplex *A, int lda);
template void Cublas::gerc(int m, int n, const cuComplex *alpha,
                           const cuComplex *x, int incx,
                           const cuComplex *y, int incy,
                           cuComplex *A, int lda);
template void Cublas::gerc(int m, int n, const cuDoubleComplex *alpha,
                           const cuDoubleComplex *x, int incx,
                           const cuDoubleComplex *y, int incy,
                           cuDoubleComplex *A, int lda);


template void Cublas::gemv_ext(bool transposed, const int m, const int n,
                               const cuComplex *alpha, const cuComplex *A, const int lda,
                               const cuComplex *x, const int incx,
                               const cuComplex *beta, cuComplex *y, const int incy, const int offsetx, const int offsety, const int offseta);
template void Cublas::gemv_ext(bool transposed, const int m, const int n,
                               const cuDoubleComplex *alpha, const cuDoubleComplex *A, const int lda,
                               const cuDoubleComplex *x, const int incx,
                               const cuDoubleComplex *beta, cuDoubleComplex *y, const int incy, const int offsetx, const int offsety, const int offseta);


template void Cublas::trsv_v2( cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag, int n,
                               const cuComplex *A, int lda, cuComplex *x, int incx, int offseta);
template void Cublas::trsv_v2( cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag, int n,
                               const cuDoubleComplex *A, int lda, cuDoubleComplex *x, int incx, int offseta);

template double Cublas::nrm2(int n, const cuDoubleComplex *x, int incx);
template float Cublas::nrm2(int n, const cuComplex *x, int incx);

template void Cublas::scal(int n, cuComplex alpha, cuComplex *x, int incx);
template void Cublas::scal(int n, cuDoubleComplex alpha, cuDoubleComplex *x, int incx);
template void Cublas::scal(int n, float alpha, cuComplex *x, int incx);
template void Cublas::scal(int n, double alpha, cuDoubleComplex *x, int incx);

} // namespace amgx

