// SPDX-FileCopyrightText: 2011 - 2024 NVIDIA CORPORATION. All Rights Reserved.
//
// SPDX-License-Identifier: BSD-3-Clause

#pragma once

namespace amgx
{

//-------------------------------------------------------
// global CUSPARSE object
//-------------------------------------------------------

class Cusparse;
//extern Cusparse cusparse;

//-------------------------------------------------------
// CUSPARSE errors
//-------------------------------------------------------

#define cusparseCheckError(status) {\
    switch(status) {\
    case CUSPARSE_STATUS_SUCCESS:                   break;\
    case CUSPARSE_STATUS_NOT_INITIALIZED:           FatalError("CUSPARSE_STATUS_NOT_INITIALIZED", AMGX_ERR_CUDA_FAILURE);\
    case CUSPARSE_STATUS_ALLOC_FAILED:              FatalError("CUSPARSE_STATUS_ALLOC_FAILED", AMGX_ERR_CUDA_FAILURE);\
    case CUSPARSE_STATUS_INVALID_VALUE:             FatalError("CUSPARSE_STATUS_INVALID_VALUE", AMGX_ERR_CUDA_FAILURE);\
    case CUSPARSE_STATUS_ARCH_MISMATCH:             FatalError("CUSPARSE_STATUS_ARCH_MISMATCH", AMGX_ERR_CUDA_FAILURE);\
    case CUSPARSE_STATUS_MAPPING_ERROR:             FatalError("CUSPARSE_STATUS_MAPPING_ERROR", AMGX_ERR_CUDA_FAILURE);\
    case CUSPARSE_STATUS_EXECUTION_FAILED:          FatalError("CUSPARSE_STATUS_EXECUTION_FAILED", AMGX_ERR_CUDA_FAILURE);\
    case CUSPARSE_STATUS_INTERNAL_ERROR:            FatalError("CUSPARSE_STATUS_INTERNAL_ERROR", AMGX_ERR_CUDA_FAILURE);\
    case CUSPARSE_STATUS_MATRIX_TYPE_NOT_SUPPORTED: FatalError("CUSPARSE_STATUS_MATRIX_TYPE_NOT_SUPPORTED", AMGX_ERR_NOT_IMPLEMENTED);\
    default:                                        FatalError("unknown CUSPARSE error", AMGX_ERR_CUDA_FAILURE);\
    }\
}

} // namespace amgx


#include <cusparse_v2.h>
#include <error.h>
#include <matrix.h>
#include <vector.h>
#include <types.h>
#include <thrust/sequence.h>

#include "amgx_types/util.h"

namespace amgx
{

//-------------------------------------------------------
// C++ CUSPARSE API for NVAMG
//-------------------------------------------------------

// The generic cuSPARSE routines must be used to achieve the same functionality
class Cusparse
{
    private:
        // global CUSPARSE handle for nvAMG
        cusparseHandle_t m_handle;

    public:
        // Which columns should be used in A.x?
        // ALL_COLORS works always
        // !!! All other choices require color-sorted column indices in A !!!
        enum ColumnColorSelector
        {
            ALL_COLORS,        // use all columns
            SMALLER_COLORS,    // use only columns with colColor < rowColor
            LARGER_COLORS,     // use only columns with colColor > rowColor
            DIAG_COL           // use only the diagonal
        };

        // Constructor.
        Cusparse();

        // Destructor.
        ~Cusparse();

        // Get the singleton instance of the class.
        static Cusparse &get_instance();

        // Destroy handle
        void destroy_handle()
        {
            if (m_handle != 0)
            {
                cusparseCheckError( cusparseDestroy(m_handle) );
            }

            m_handle = 0;
        }

        void create_handle()
        {
            if (m_handle == 0)
            {
                cusparseCheckError( cusparseCreate(&m_handle) );
            }
        }

        // Get the handle.
        inline cusparseHandle_t get_handle() { return m_handle; }

        // TODO : here we declare alpha and beta consts as lower precision (VecPrec),
        // however when mixed precision for bsrmv will be supported those constants will
        // likely be of higher precision and we should switch them to VecPrec.

        // operate on all rows and columns y= alpha*A.x + beta*y
        template <class TConfig>
        static void bsrmv( const typename TConfig::VecPrec alphaConst,
                           Matrix<TConfig> &A,
                           Vector<TConfig> &x,
                           const typename TConfig::VecPrec betaConst,
                           Vector<TConfig> &y,
                           ViewType view = OWNED );

        template <class TConfig>
        static void bsrmv_with_mask( const typename TConfig::VecPrec alphaConst,
                                     Matrix<TConfig> &A,
                                     Vector<TConfig> &x,
                                     const typename TConfig::VecPrec betaConst,
                                     Vector<TConfig> &y );

        template <class TConfig>
        static void bsrmv_with_mask_restriction( const typename TConfig::VecPrec alphaConst,
                Matrix<TConfig> &A,
                Vector<TConfig> &x,
                const typename TConfig::VecPrec betaConst,
                Vector<TConfig> &y,
                Matrix<TConfig> &P);


        template <class TConfig>
        static void bsrxmv( const typename TConfig::VecPrec alphaConst,
                            Matrix<TConfig> &A,
                            Vector<TConfig> &x,
                            const typename TConfig::VecPrec betaConst,
                            Vector<TConfig> &y,
                            typename Matrix<TConfig>::IVector &mask,
                            ViewType view = OWNED );

        // E is a vector that represents a diagonal matrix
        // operate on all rows and columns
        // y= alpha*E.x + beta*y
        template <class TConfig>
        static void bsrmv( const typename TConfig::VecPrec alphaConst,
                           Matrix<TConfig> &A,
                           const typename Matrix<TConfig>::MVector &E,
                           Vector<TConfig> &x,
                           const typename TConfig::VecPrec betaConst,
                           Vector<TConfig> &y,
                           ViewType view = OWNED );

        // operate only on columns specified by columnColorSelector, see enum ColumnColorSelector above
        // operate only on rows of specified color, given by A.offsets_rows_per_color, A.sorted_rows_by_color
        // y= alpha*A.x + beta*y
        template <class TConfig>
        static void bsrmv( ColumnColorSelector columnColorSelector,
                           const int color,
                           const typename TConfig::VecPrec alphaConst,
                           Matrix<TConfig> &A,
                           Vector<TConfig> &x,
                           const typename TConfig::VecPrec betaConst,
                           Vector<TConfig> &y,
                           ViewType view = OWNED );

        // E is a vector that represents a diagonal matrix
        // operate only on rows of specified color, given by A.offsets_rows_per_color, A.sorted_rows_by_color
        // y= alpha*E.x + beta*y
        template <class TConfig>
        static void bsrmv( const int color,
                           typename TConfig::VecPrec alphaConst,
                           Matrix<TConfig> &A,
                           const typename Matrix<TConfig>::MVector &E,
                           Vector<TConfig> &x,
                           typename TConfig::VecPrec betaConst,
                           Vector<TConfig> &y,
                           ViewType view = OWNED );

        template <class TConfig>
        static void csrmm(typename TConfig::VecPrec alpha,
                          Matrix<TConfig> &A,
                          Vector<TConfig> &V,
                          typename TConfig::VecPrec beta,
                          Vector<TConfig> &Res);


        template <class TConfig>
        static void transpose(const Matrix<TConfig> &A, Matrix<TConfig> &B);

        template <class TConfig>
        static void transpose(const Matrix<TConfig> &A, Matrix<TConfig> &B, const int nRows, const int nNz);

    private:

        template <class TConfig>
        static void bsrmv_internal( typename TConfig::VecPrec alphaConst,
                                    const Matrix<TConfig> &A,
                                    const Vector<TConfig> &x,
                                    const typename TConfig::VecPrec betaConst,
                                    Vector<TConfig> &y,
                                    ViewType view,
                                    const cudaStream_t &stream);


        template <class TConfig>
        static void bsrmv_internal_with_mask( typename TConfig::VecPrec alphaConst,
                                              const Matrix<TConfig> &A,
                                              const Vector<TConfig> &x,
                                              const typename TConfig::VecPrec betaConst,
                                              Vector<TConfig> &y,
                                              ViewType view,
                                              const cudaStream_t &stream);

        template <class TConfig>
        static void bsrmv_internal_with_mask_restriction( typename TConfig::VecPrec alphaConst,
                const Matrix<TConfig> &A,
                const Vector<TConfig> &x,
                const typename TConfig::VecPrec betaConst,
                Vector<TConfig> &y,
                ViewType view,
                const cudaStream_t &stream,
                Matrix<TConfig> &P);



        template <class TConfig>
        static void bsrmv_internal( const typename TConfig::VecPrec alphaConst,
                                    const Matrix<TConfig> &A,
                                    const typename Matrix<TConfig>::MVector &E,
                                    const Vector<TConfig> &x,
                                    const typename TConfig::VecPrec betaConst,
                                    Vector<TConfig> &y,
                                    ViewType view,
                                    const cudaStream_t &stream);

        template <class TConfig>
        static void bsrmv_internal( ColumnColorSelector columnColorSelector,
                                    const int color,
                                    const typename TConfig::VecPrec alphaConst,
                                    const Matrix<TConfig> &A,
                                    const Vector<TConfig> &x,
                                    const typename TConfig::VecPrec betaConst,
                                    Vector<TConfig> &y,
                                    ViewType view,
                                    const cudaStream_t &stream);

        template <class TConfig>
        static void bsrmv_internal( const int color,
                                    const typename TConfig::VecPrec alphaConst,
                                    const Matrix<TConfig> &A,
                                    const typename Matrix<TConfig>::MVector &E,
                                    const Vector<TConfig> &x,
                                    const typename TConfig::VecPrec betaConst,
                                    Vector<TConfig> &y,
                                    ViewType view,
                                    const cudaStream_t &stream);
        // real-valued declarations
        // overloaded C++ wrappers for cusparse?bsrmv
        static inline void bsrmv( cusparseHandle_t handle, cusparseDirection_t dir, cusparseOperation_t trans,
                                  int mb, int nb, int nnzb,
                                  const float *alpha,
                                  const cusparseMatDescr_t descr,
                                  const float *bsrVal,
                                  const int *bsrMaskPtr,
                                  const int *bsrRowPtr,
                                  const int *bsrColInd,
                                  int rowOff,
                                  int blockDim,
                                  const float *x,
                                  const float *beta,
                                  float *y,
                                  const cudaStream_t& stream);

        static inline void bsrmv( cusparseHandle_t handle, cusparseDirection_t dir, cusparseOperation_t trans,
                                  int mb, int nb, int nnzb,
                                  const double *alpha,
                                  const cusparseMatDescr_t descr,
                                  const double *bsrVal,
                                  const int *bsrMaskPtr,
                                  const int *bsrRowPtr,
                                  const int *bsrColInd,
                                  int rowOff,
                                  int blockDim,
                                  const double *x,
                                  const double *beta,
                                  double *y,
                                  const cudaStream_t& stream);

        static inline void bsrmv( cusparseHandle_t handle, cusparseDirection_t dir, cusparseOperation_t trans,
                                  int mb, int nb, int nnzb,
                                  const double *alpha,
                                  const cusparseMatDescr_t descr,
                                  const float *bsrVal,
                                  const int *bsrMaskPtr,
                                  const int *bsrRowPtr,
                                  const int *bsrColInd,
                                  int rowOff,
                                  int blockDim,
                                  const double *x,
                                  const double *beta,
                                  double *y,
                                  const cudaStream_t& stream);

        // overloaded C++ wrappers for cusparse?bsrxmv
        // bsrxmv
        // matrix - float
        // vector - float
        static inline void bsrxmv_internal( cusparseHandle_t handle, cusparseDirection_t dir, cusparseOperation_t trans, int sizeOfMask,
                                            int mb, int nb, int nnzb,
                                            const float *alpha,
                                            const cusparseMatDescr_t descr,
                                            const float *bsrVal,
                                            const int *bsrMaskPtr,
                                            const int *bsrRowPtr,
                                            const int *bsrEndPtr,
                                            const int *bsrColInd,
                                            int blockDim,
                                            const float *x,
                                            const float *beta,
                                            float *y);

        // bsrxmv
        // matrix - float
        // vector - double
        static inline void bsrxmv_internal( cusparseHandle_t handle, cusparseDirection_t dir, cusparseOperation_t trans, int sizeOfMask,
                                            int mb, int nb, int nnzb,
                                            const double *alpha,
                                            const cusparseMatDescr_t descr,
                                            const float *bsrVal,
                                            const int *bsrMaskPtr,
                                            const int *bsrRowPtr,
                                            const int *bsrEndPtr,
                                            const int *bsrColInd,
                                            int blockDim,
                                            const double *x,
                                            const double *beta,
                                            double *y);
        // bsrxmv
        // matrix - double
        // vector - double
        static inline void bsrxmv_internal( cusparseHandle_t handle, cusparseDirection_t dir, cusparseOperation_t trans, int sizeOfMask,
                                            int mb, int nb, int nnzb,
                                            const double *alpha,
                                            const cusparseMatDescr_t descr,
                                            const double *bsrVal,
                                            const int *bsrMaskPtr,
                                            const int *bsrRowPtr,
                                            const int *bsrEndPtr,
                                            const int *bsrColInd,
                                            int blockDim,
                                            const double *x,
                                            const double *beta,
                                            double *y);

        // complex-valued declarations
        // overloaded C++ wrappers for cusparse?bsrmv
        static inline void bsrmv( cusparseHandle_t handle, cusparseDirection_t dir, cusparseOperation_t trans,
                                  int mb, int nb, int nnzb,
                                  const cuComplex *alpha,
                                  const cusparseMatDescr_t descr,
                                  const cuComplex *bsrVal,
                                  const int *bsrMaskPtr,
                                  const int *bsrRowPtr,
                                  const int *bsrColInd,
                                  int rowOff,
                                  int blockDim,
                                  const cuComplex *x,
                                  const cuComplex *beta,
                                  cuComplex *y,
                                  const cudaStream_t& stream);

        static inline void bsrmv( cusparseHandle_t handle, cusparseDirection_t dir, cusparseOperation_t trans,
                                  int mb, int nb, int nnzb,
                                  const cuDoubleComplex *alpha,
                                  const cusparseMatDescr_t descr,
                                  const cuDoubleComplex *bsrVal,
                                  const int *bsrMaskPtr,
                                  const int *bsrRowPtr,
                                  const int *bsrColInd,
                                  int rowOff,
                                  int blockDim,
                                  const cuDoubleComplex *x,
                                  const cuDoubleComplex *beta,
                                  cuDoubleComplex *y,
                                  const cudaStream_t& stream);

        static inline void bsrmv( cusparseHandle_t handle, cusparseDirection_t dir, cusparseOperation_t trans,
                                  int mb, int nb, int nnzb,
                                  const cuDoubleComplex *alpha,
                                  const cusparseMatDescr_t descr,
                                  const cuComplex *bsrVal,
                                  const int *bsrMaskPtr,
                                  const int *bsrRowPtr,
                                  const int *bsrColInd,
                                  int rowOff,
                                  int blockDim,
                                  const cuDoubleComplex *x,
                                  const cuDoubleComplex *beta,
                                  cuDoubleComplex *y,
                                  const cudaStream_t& stream);

        // overloaded C++ wrappers for cusparse?bsrxmv
        // bsrxmv
        // matrix - cuComplex
        // vector - cuComplex
        static inline void bsrxmv_internal( cusparseHandle_t handle, cusparseDirection_t dir, cusparseOperation_t trans, int sizeOfMask,
                                            int mb, int nb, int nnzb,
                                            const cuComplex *alpha,
                                            const cusparseMatDescr_t descr,
                                            const cuComplex *bsrVal,
                                            const int *bsrMaskPtr,
                                            const int *bsrRowPtr,
                                            const int *bsrEndPtr,
                                            const int *bsrColInd,
                                            int blockDim,
                                            const cuComplex *x,
                                            const cuComplex *beta,
                                            cuComplex *y);

        // bsrxmv
        // matrix - cuComplex
        // vector - cuDoubleComplex
        static inline void bsrxmv_internal( cusparseHandle_t handle, cusparseDirection_t dir, cusparseOperation_t trans, int sizeOfMask,
                                            int mb, int nb, int nnzb,
                                            const cuDoubleComplex *alpha,
                                            const cusparseMatDescr_t descr,
                                            const cuComplex *bsrVal,
                                            const int *bsrMaskPtr,
                                            const int *bsrRowPtr,
                                            const int *bsrEndPtr,
                                            const int *bsrColInd,
                                            int blockDim,
                                            const cuDoubleComplex *x,
                                            const cuDoubleComplex *beta,
                                            cuDoubleComplex *y);
        // bsrxmv
        // matrix - cuDoubleComplex
        // vector - cuDoubleComplex
        static inline void bsrxmv_internal( cusparseHandle_t handle, cusparseDirection_t dir, cusparseOperation_t trans, int sizeOfMask,
                                            int mb, int nb, int nnzb,
                                            const cuDoubleComplex *alpha,
                                            const cusparseMatDescr_t descr,
                                            const cuDoubleComplex *bsrVal,
                                            const int *bsrMaskPtr,
                                            const int *bsrRowPtr,
                                            const int *bsrEndPtr,
                                            const int *bsrColInd,
                                            int blockDim,
                                            const cuDoubleComplex *x,
                                            const cuDoubleComplex *beta,
                                            cuDoubleComplex *y);
};

} // namespace amgx
