// SPDX-FileCopyrightText: 2013 - 2024 NVIDIA CORPORATION. All Rights Reserved.
//
// SPDX-License-Identifier: BSD-3-Clause

#pragma once


#include <cublas_v2.h>
#include <error.h>
#include <matrix.h>
#include <vector.h>
#include <types.h>

namespace amgx
{

class Cublas;

#define cublasCheckError(status) {\
    switch(status) {\
    case CUBLAS_STATUS_SUCCESS:                   break;\
    case CUBLAS_STATUS_NOT_INITIALIZED:           FatalError("CUBLAS_STATUS_NOT_INITIALIZED", AMGX_ERR_CUDA_FAILURE);\
    case CUBLAS_STATUS_ALLOC_FAILED:              FatalError("CUBLAS_STATUS_ALLOC_FAILED", AMGX_ERR_CUDA_FAILURE);\
    case CUBLAS_STATUS_INVALID_VALUE:             FatalError("CUBLAS_STATUS_INVALID_VALUE", AMGX_ERR_CUDA_FAILURE);\
    case CUBLAS_STATUS_ARCH_MISMATCH:             FatalError("CUBLAS_STATUS_ARCH_MISMATCH", AMGX_ERR_CUDA_FAILURE);\
    case CUBLAS_STATUS_MAPPING_ERROR:             FatalError("CUBLAS_STATUS_MAPPING_ERROR", AMGX_ERR_CUDA_FAILURE);\
    case CUBLAS_STATUS_EXECUTION_FAILED:          FatalError("CUBLAS_STATUS_EXECUTION_FAILED", AMGX_ERR_CUDA_FAILURE);\
    case CUBLAS_STATUS_INTERNAL_ERROR:            FatalError("CUBLAS_STATUS_INTERNAL_ERROR", AMGX_ERR_CUDA_FAILURE);\
    default:                                        FatalError("unknown CUBLAS error", AMGX_ERR_CUDA_FAILURE);\
    }\
}

class Cublas
{
    private:
        static cublasHandle_t m_handle;
        // Private ctor to prevent instantiation.
        Cublas();
        ~Cublas();
    public:

        // Get the handle.
        static cublasHandle_t get_handle()
        {
            if (m_handle == 0)
            {
                cublasCheckError(cublasCreate(&m_handle));
            }

            return m_handle;
        }

        static void destroy_handle()
        {
            if (m_handle != 0)
            {
                cublasCheckError(cublasDestroy(m_handle));
            }

            m_handle = 0;
        }

        static void set_pointer_mode_device();
        static void set_pointer_mode_host();

        template <typename T>
        static void axpy(int n, T alpha,
                         const T *x, int incx,
                         T *y, int incy);

        template <typename T>
        static void copy(int n, const T *x, int incx,
                         T *y, int incy);

        template <typename T>
        static void dot(int n, const T *x, int incx,
                        const T *y, int incy,
                        T *result);

        template <typename T>
        static void dotc(int n, const T *x, int incx,
                         const T *y, int incy,
                         T *result);

        // Only gemm uses Vector objects as arguments since other
        // operations can be used on subvector using offset and size variables
        // given by the view.
        template <typename TConfig>
        static void gemm(typename TConfig::VecPrec alpha,
                         const Vector<TConfig> &A, const Vector<TConfig> &B,
                         typename TConfig::VecPrec beta, Vector<TConfig> &C,
                         bool A_transposed = false, bool B_transposed = false);

        template <typename T>
        static void gemv(bool transposed, int m, int n,
                         const T *alpha, const T *A, int lda,
                         const T *x, int incx,
                         const T *beta, T *y, int incy);
        template <typename T>
        static void gemv_ext(bool transposed, const int m, const int n,
                             const T *alpha, const T *A, const int lda,
                             const T *x, const int incx,
                             const T *beta, T *y, const int incy, const int offsetx, const int offsety, const int offseta);

        template <typename T>
        static void trsv_v2( cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag, int n,
                             const T *A, int lda, T *x, int incx, int offseta);
        template <typename T>
        static void ger(int m, int n, const T *alpha,
                        const T *x, int incx,
                        const T *y, int incy,
                        T *A, int lda);
        template <typename T>
        static void gerc(int m, int n, const T *alpha,
                         const T *x, int incx,
                         const T *y, int incy,
                         T *A, int lda);

        template <typename T, typename V>
        static V nrm2(int n, const T *x, int incx);
        template <typename T, typename V>
        static void nrm2(int n, const T *x, int incx, V *result);

        template <typename T, typename V>
        static void scal(int n, T alpha, V *x, int incx);
        template <typename T, typename V>
        static void scal(int n, T *alpha, V *x, int incx);

};

} // namespace amgx
