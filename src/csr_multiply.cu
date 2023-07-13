/* Copyright (c) 2013-2022, NVIDIA CORPORATION. All rights reserved.
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

#include <csr_multiply.h>
#include <csr_multiply_detail.h>
#include <util.h>
#include <device_properties.h>
#include <amgx_cusparse.h>

namespace amgx
{

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template< AMGX_VecPrecision V, AMGX_MatPrecision M, AMGX_IndPrecision I >
void *CSR_Multiply<TemplateConfig<AMGX_device, V, M, I> >::csr_workspace_create()
{
    cudaDeviceProp props = getDeviceProperties();
    int arch = 10 * props.major + props.minor;

    if ( arch >= 70 )
    {
        return new CSR_Multiply_Detail<TConfig_d>();
    }

    FatalError( "CSR_Multiply: Unsupported architecture. It requires a Volta GPU or newer!!!", AMGX_ERR_NOT_SUPPORTED_BLOCKSIZE );
}

// ====================================================================================================================

template< AMGX_VecPrecision V, AMGX_MatPrecision M, AMGX_IndPrecision I >
void *CSR_Multiply<TemplateConfig<AMGX_device, V, M, I> >::csr_workspace_create( AMG_Config &cfg, const std::string &cfg_scope )
{
    int max_attempts = cfg.getParameter<int>("spmm_max_attempts", cfg_scope);
    int use_opt_kernels = cfg.getParameter<int>("use_opt_kernels", "default");
    int use_cusparse_kernels = cfg.getParameter<int>("use_cusparse_kernels", "default");

    cudaDeviceProp props = getDeviceProperties();
    int arch = 10 * props.major + props.minor;

    if ( arch >= 70 )
    {
        CSR_Multiply_Detail<TConfig_d> *wk = new CSR_Multiply_Detail<TConfig_d>();
        wk->set_max_attempts(max_attempts);
        wk->set_opt_multiply(use_opt_kernels);
        wk->set_use_cusparse_kernels(use_cusparse_kernels);
        return wk;
    }

    FatalError( "CSR_Multiply: Unsupported architecture. It requires a Volta GPU or newer!!!", AMGX_ERR_NOT_SUPPORTED_BLOCKSIZE );
}

// ====================================================================================================================

template <AMGX_VecPrecision V, AMGX_MatPrecision M, AMGX_IndPrecision I>
void CSR_Multiply<TemplateConfig<AMGX_device, V, M, I> >::csr_workspace_delete( void *workspace )
{
    CSR_Multiply_Impl<TConfig_d> *impl = static_cast<CSR_Multiply_Impl<TConfig_d> *>(workspace);
    delete impl;
}

// ====================================================================================================================

template< AMGX_VecPrecision V, AMGX_MatPrecision M, AMGX_IndPrecision I >
void CSR_Multiply<TemplateConfig<AMGX_device, V, M, I> >::csr_multiply( const Matrix_d &A, const Matrix_d &B, Matrix_d &C, void *wk )
{
    if ( A.get_block_size() != 1 || B.get_block_size() != 1 )
    {
        FatalError( "csr_multiply: Unsupported block size", AMGX_ERR_NOT_SUPPORTED_BLOCKSIZE );
    }

    if (A.hasProps(DIAG) || ( A.hasProps(DIAG) != B.hasProps(DIAG) ) )
    {
        FatalError( "csr_multiply does not support external diagonal and the two matrices have to use the same storage for the diagonal", AMGX_ERR_NOT_SUPPORTED_BLOCKSIZE );
    }

    CSR_Multiply_Impl<TConfig_d> *impl = NULL;

    if ( wk == NULL )
    {
        printf("csr_multiply: wk is NULL\n");
        impl = static_cast<CSR_Multiply_Impl<TConfig_d> *>( csr_workspace_create() );
    }
    else
    {
        impl = static_cast<CSR_Multiply_Impl<TConfig_d> *>( wk );
    }

    assert( impl != NULL );
    impl->multiply( A, B, C, NULL, NULL, NULL, NULL );

    if ( wk != NULL )
    {
        return;
    }

    delete impl;
}

// ====================================================================================================================

template< AMGX_VecPrecision V, AMGX_MatPrecision M, AMGX_IndPrecision I >
void CSR_Multiply<TemplateConfig<AMGX_device, V, M, I> >::csr_sparsity( const Matrix_d &A, Matrix_d &B, void *wk )
{
    if ( A.get_block_size() != 1 || B.get_block_size() != 1 )
    {
        FatalError( "csr_sparsity: Unsupported block size", AMGX_ERR_NOT_SUPPORTED_BLOCKSIZE );
    }

    if (A.hasProps(DIAG) || ( A.hasProps(DIAG) != B.hasProps(DIAG) ) )
    {
        FatalError( "csr_sparsity does not support external diagonal and the two matrices have to use the same storage for the diagonal", AMGX_ERR_NOT_SUPPORTED_BLOCKSIZE );
    }

    CSR_Multiply_Impl<TConfig_d> *impl = NULL;

    if ( wk == NULL )
    {
        printf("csr_sparsity: wk is NULL\n");
        impl = static_cast<CSR_Multiply_Impl<TConfig_d> *>( csr_workspace_create() );
    }
    else
    {
        impl = static_cast<CSR_Multiply_Impl<TConfig_d> *>( wk );
    }

    assert( impl != NULL );
    impl->sparsity( A, B );

    if ( wk != NULL )
    {
        return;
    }

    delete impl;
}

// ====================================================================================================================

template< AMGX_VecPrecision V, AMGX_MatPrecision M, AMGX_IndPrecision I >
void CSR_Multiply<TemplateConfig<AMGX_device, V, M, I> >::csr_sparsity( const Matrix_d &A, const Matrix_d &B, Matrix_d &C, void *wk )
{
    if ( A.get_block_size() != 1 || B.get_block_size() != 1 )
    {
        FatalError( "csr_sparsity: Unsupported block size", AMGX_ERR_NOT_SUPPORTED_BLOCKSIZE );
    }

    if (A.hasProps(DIAG) || ( A.hasProps(DIAG) != B.hasProps(DIAG) ) )
    {
        FatalError( "csr_sparsity does not support external diagonal and the two matrices have to use the same storage for the diagonal", AMGX_ERR_NOT_SUPPORTED_BLOCKSIZE );
    }

    CSR_Multiply_Impl<TConfig_d> *impl = NULL;

    if ( wk == NULL )
    {
        printf("csr_sparsity 2: wk is NULL\n");
        impl = static_cast<CSR_Multiply_Impl<TConfig_d> *>( csr_workspace_create() );
    }
    else
    {
        impl = static_cast<CSR_Multiply_Impl<TConfig_d> *>( wk );
    }

    assert( impl != NULL );
    impl->sparsity( A, B, C );

    if ( wk != NULL )
    {
        return;
    }

    delete impl;
}

// ====================================================================================================================

template< AMGX_VecPrecision V, AMGX_MatPrecision M, AMGX_IndPrecision I >
void CSR_Multiply<TemplateConfig<AMGX_device, V, M, I> >::csr_sparsity_ilu1( const Matrix_d &A, Matrix_d &B, void *wk )
{
    CSR_Multiply_Impl<TConfig_d> *impl = NULL;

    if ( wk == NULL )
    {
        printf("csr_sparsity_ilu1: wk is NULL\n");
        impl = static_cast<CSR_Multiply_Impl<TConfig_d> *>( csr_workspace_create() );
    }
    else
    {
        impl = static_cast<CSR_Multiply_Impl<TConfig_d> *>( wk );
    }

    assert( impl != NULL );
    impl->sparsity_ilu1( A, B );

    if ( wk != NULL )
    {
        return;
    }

    delete impl;
}

// ====================================================================================================================

template< AMGX_VecPrecision V, AMGX_MatPrecision M, AMGX_IndPrecision I >
void
CSR_Multiply<TemplateConfig<AMGX_device, V, M, I> >::csr_galerkin_product( const Matrix_d &R, const Matrix_d &A, const Matrix_d &P, Matrix_d &RAP, IVector *Rq1, IVector *Aq1, IVector *Pq1, IVector *Rq2, IVector *Aq2, IVector *Pq2, void *wk)
{
    if ( R.get_block_size( ) != 1 || A.get_block_size( ) != 1 || P.get_block_size( ) != 1 )
    {
        FatalError( "csr_galerkin_product: Unsupported block size", AMGX_ERR_NOT_SUPPORTED_BLOCKSIZE );
    }

    if ( A.hasProps(DIAG) || R.hasProps( DIAG ) != A.hasProps( DIAG ) || P.hasProps( DIAG ) != A.hasProps( DIAG ) )
    {
        FatalError( "csr_galerkin_product: The three matrices have to use the same storage for the diagonal, and cannot support external diagonal", AMGX_ERR_NOT_SUPPORTED_BLOCKSIZE );
    }

    if ( R.get_num_rows( ) == 0 || A.get_num_rows( ) == 0 || P.get_num_rows( ) == 0 )
    {
        return;
    }

    CSR_Multiply_Impl<TConfig_d> *impl = NULL;

    if ( wk == NULL )
    {
        printf("csr_galerkin_product: wk is NULL\n");
        impl = static_cast<CSR_Multiply_Impl<TConfig_d> *>( csr_workspace_create() );
    }
    else
    {
        impl = static_cast<CSR_Multiply_Impl<TConfig_d> *>( wk );
    }

    assert( impl != NULL );
    impl->galerkin_product( R, A, P, RAP, Rq1, Aq1, Pq1, Rq2, Aq2, Pq2 );

    if ( wk != NULL )
    {
        return;
    }

    delete impl;
}

template< AMGX_VecPrecision V, AMGX_MatPrecision M, AMGX_IndPrecision I >
void
CSR_Multiply<TemplateConfig<AMGX_device, V, M, I> >::csr_RAP_sparse_add( Matrix_d &RAP, const Matrix_d &RAP_int, std::vector<IVector> &RAP_ext_row_offsets, std::vector<IVector> &RAP_ext_col_indices, std::vector<MVector> &RAP_ext_values, std::vector<IVector> &RAP_ext_row_ids, void *wk )
{
    if ( RAP_int.get_block_size( ) != 1 )
    {
        FatalError( "csr_RAP_sparse_add: Unsupported block size", AMGX_ERR_NOT_SUPPORTED_BLOCKSIZE );
    }

    if ( RAP_int.hasProps(DIAG) )
    {
        FatalError( "csr_RAP_sparse_add: Does not support external diagonal", AMGX_ERR_NOT_SUPPORTED_BLOCKSIZE );
    }

    CSR_Multiply_Impl<TConfig_d> *impl = NULL;

    if ( wk == NULL )
    {
        impl = static_cast<CSR_Multiply_Impl<TConfig_d> *>( csr_workspace_create() );
    }
    else
    {
        impl = static_cast<CSR_Multiply_Impl<TConfig_d> *>( wk );
    }

    assert( impl != NULL );
    impl->RAP_sparse_add( RAP, RAP_int, RAP_ext_row_offsets, RAP_ext_col_indices, RAP_ext_values, RAP_ext_row_ids);

    if ( wk != NULL )
    {
        return;
    }

    delete impl;
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template< AMGX_VecPrecision V, AMGX_MatPrecision M, AMGX_IndPrecision I >
CSR_Multiply_Impl<TemplateConfig<AMGX_device, V, M, I> >::CSR_Multiply_Impl( bool allocate_vals, int grid_size, int max_warp_count, int gmem_size )
    : Base( allocate_vals, grid_size, max_warp_count, gmem_size )
    , m_max_attempts(10)
{
}

// ====================================================================================================================
#ifndef CUSPARSE_USE_GENERIC_SPGEMM
#define CUSPARSE_CSRGEMM(type, func) \
cusparseStatus_t cusparseCsrgemm2(cusparseHandle_t handle,             \
                                 int m,                               \
                                 int n,                               \
                                 int k,                               \
                                 const type* alpha,                   \
                                 const cusparseMatDescr_t descrA,     \
                                 int nnzA,                            \
                                 const type *csrValA,                 \
                                 const int *csrRowPtrA,               \
                                 const int *csrColIndA,               \
                                 const cusparseMatDescr_t descrB,     \
                                 int nnzB,                            \
                                 const type *csrValB,                 \
                                 const int *csrRowPtrB,               \
                                 const int *csrColIndB,               \
                                 const type* beta,                    \
                                 const cusparseMatDescr_t descrD,     \
                                 int nnzD,                            \
                                 const type *csrValD,                 \
                                 const int *csrRowPtrD,               \
                                 const int *csrColIndD,               \
                                 const cusparseMatDescr_t descrC,     \
                                 type *csrValC,                       \
                                 const int *csrRowPtrC,               \
                                 int *csrColIndC,                     \
                                 const csrgemm2Info_t info,           \
                                 void* pBuffer                        \
                                 )                                    \
{                                                                     \
  return func(handle, m, n, k, alpha, \
              descrA, nnzA, csrValA, csrRowPtrA, csrColIndA, \
              descrB, nnzB, csrValB, csrRowPtrB, csrColIndB, \
              beta, \
              descrD, nnzD, csrValD, csrRowPtrD, csrColIndD, \
              descrC, csrValC, csrRowPtrC, csrColIndC, info, pBuffer);  \
}

CUSPARSE_CSRGEMM(float,           cusparseScsrgemm2)
CUSPARSE_CSRGEMM(double,          cusparseDcsrgemm2)
CUSPARSE_CSRGEMM(cuComplex,       cusparseCcsrgemm2)
CUSPARSE_CSRGEMM(cuDoubleComplex, cusparseZcsrgemm2)
#endif

#ifndef CUSPARSE_USE_GENERIC_SPGEMM
#define CUSPARSE_CSRGEMMBUFSZ(type, func) \
cusparseStatus_t cusparseCsrgemmBufferSize(cusparseHandle_t handle,             \
                                 int m,                               \
                                 int n,                               \
                                 int k,                               \
                                 const type* alpha,                   \
                                 const cusparseMatDescr_t descrA,     \
                                 int nnzA,                            \
                                 const int *csrRowPtrA,               \
                                 const int *csrColIndA,               \
                                 const cusparseMatDescr_t descrB,     \
                                 int nnzB,                            \
                                 const int *csrRowPtrB,               \
                                 const int *csrColIndB,               \
                                 const type* beta,                    \
                                 const cusparseMatDescr_t descrD,     \
                                 int nnzD,                            \
                                 const int *csrRowPtrD,               \
                                 const int *csrColIndD,               \
                                 const csrgemm2Info_t info,           \
                                 size_t* pBufferSizeInBytes           \
                                 )                                    \
{                                                                     \
  return func(handle, m, n, k, alpha, descrA, nnzA,         \
              csrRowPtrA, csrColIndA, descrB, nnzB,         \
              csrRowPtrB, csrColIndB, beta, descrD, nnzD,   \
              csrRowPtrD, csrColIndD, info, pBufferSizeInBytes);       \
}

CUSPARSE_CSRGEMMBUFSZ(float,           cusparseScsrgemm2_bufferSizeExt)
CUSPARSE_CSRGEMMBUFSZ(double,          cusparseDcsrgemm2_bufferSizeExt)
CUSPARSE_CSRGEMMBUFSZ(cuComplex,       cusparseCcsrgemm2_bufferSizeExt)
CUSPARSE_CSRGEMMBUFSZ(cuDoubleComplex, cusparseZcsrgemm2_bufferSizeExt)
#endif

// ====================================================================================================================

#ifdef CUSPARSE_USE_GENERIC_SPGEMM
template< AMGX_VecPrecision V, AMGX_MatPrecision M, AMGX_IndPrecision I > void CSR_Multiply_Impl<TemplateConfig<AMGX_device, V, M, I> >::cusparse_multiply( const Matrix_d &A, const Matrix_d &B, Matrix_d &C, IVector *Aq1, IVector *Bq1, IVector *Aq2, IVector *Bq2 ) {
   // CUSPARSE APIs
    cusparseHandle_t handle = Cusparse::get_instance().get_handle();
    cusparseSpMatDescr_t matA, matB, matC;
    void*  dBuffer1    = NULL, *dBuffer2   = NULL;
    size_t bufferSize1 = 0,    bufferSize2 = 0;
    cudaDataType matType;
    if (M == AMGX_matDouble) matType = CUDA_R_64F;
    else if (M == AMGX_matFloat) matType = CUDA_R_32F;
    else if (M == AMGX_matDoubleComplex) matType = CUDA_C_64F;
    else if (M == AMGX_matComplex) matType = CUDA_C_32F;
    else FatalError("multiply::cusparse_multiply unknown matrix format", AMGX_ERR_INTERNAL);

    cusparseIndexType_t indType;
    if (I == AMGX_indInt) indType = CUSPARSE_INDEX_32I;
    //if (I == AMGX_indInt64) indType = CUSPARSE_INDEX_64I; // As of CUDA 11.0, cusparseSpGEMM supports only 32-bit indices CUSPARSE_INDEX_32I
    else FatalError("multiply::cusparse_multiply unknown index format", AMGX_ERR_INTERNAL);

    cusparseCheckError( cusparseCreateCsr(&matA, A.get_num_rows(), A.get_num_cols(), A.get_num_nz(),
                                      (void*)A.row_offsets.raw(), (void*)A.col_indices.raw(), (void*)A.values.raw(),
                                      indType, indType,
                                      CUSPARSE_INDEX_BASE_ZERO, matType) );
    cusparseCheckError( cusparseCreateCsr(&matB, B.get_num_rows(), B.get_num_cols(), B.get_num_nz(),
                                      (void*)B.row_offsets.raw(), (void*)B.col_indices.raw(), (void*)B.values.raw(),
                                      indType, indType,
                                      CUSPARSE_INDEX_BASE_ZERO, matType) );
    cusparseCheckError( cusparseCreateCsr(&matC, A.get_num_rows(), B.get_num_cols(), 0,
                                      NULL, NULL, NULL,
                                      indType, indType,
                                      CUSPARSE_INDEX_BASE_ZERO, matType) );

    typename Matrix_d::value_type alpha = types::util<typename Matrix_d::value_type>::get_one();
    typename Matrix_d::value_type beta  = types::util<typename Matrix_d::value_type>::get_zero();
    cusparseOperation_t opA         = CUSPARSE_OPERATION_NON_TRANSPOSE;
    cusparseOperation_t opB         = CUSPARSE_OPERATION_NON_TRANSPOSE;
    cudaDataType        computeType = matType;

    // SpGEMM Computation
    cusparseSpGEMMDescr_t spgemmDesc;
    cusparseCheckError( cusparseSpGEMM_createDescr(&spgemmDesc) );

    // ask bufferSize1 bytes for external memory
    cusparseSpGEMM_workEstimation(handle, opA, opB,
                                  &alpha, matA, matB, &beta, matC,
                                  computeType, CUSPARSE_SPGEMM_DEFAULT,
                                  spgemmDesc, &bufferSize1, NULL);
    if(bufferSize1 > 0) {
    amgx::memory::cudaMalloc(&dBuffer1, bufferSize1);
    }
    // inspect the matrices A and B to understand the memory requiremnent for
    // the next step
    cusparseSpGEMM_workEstimation(handle, opA, opB,
                                  &alpha, matA, matB, &beta, matC,
                                  computeType, CUSPARSE_SPGEMM_DEFAULT,
                                  spgemmDesc, &bufferSize1, dBuffer1);

    // ask bufferSize2 bytes for external memory
    cusparseSpGEMM_compute(handle, opA, opB,
                           &alpha, matA, matB, &beta, matC,
                           computeType, CUSPARSE_SPGEMM_DEFAULT,
                           spgemmDesc, &bufferSize2, NULL);
    if(bufferSize2 > 0) {
    amgx::memory::cudaMalloc(&dBuffer2, bufferSize2);
    }

    // compute the intermediate product of A * B
    cusparseSpGEMM_compute(handle, opA, opB,
                           &alpha, matA, matB, &beta, matC,
                           computeType, CUSPARSE_SPGEMM_DEFAULT,
                           spgemmDesc, &bufferSize2, dBuffer2);
    // get matrix C non-zero entries C_num_nnz1
    int64_t C_num_rows1, C_num_cols1, C_num_nnz1;
    cusparseSpMatGetSize(matC, &C_num_rows1, &C_num_cols1, &C_num_nnz1);
    // Setup C metadata
    C.set_initialized(0);
    C.row_offsets.resize( A.get_num_rows() + 1 );
    C.col_indices.resize( C_num_nnz1 );
    C.m_seq_offsets.resize( A.get_num_rows() + 1 );
    thrust::sequence(C.m_seq_offsets.begin(), C.m_seq_offsets.end());
    C.set_num_rows( A.get_num_rows() );
    C.set_num_cols( B.get_num_cols() );
    C.diag.resize(C.get_num_rows());
    C.set_block_dimx(A.get_block_dimx());
    C.set_block_dimy(B.get_block_dimy());
    C.setColsReorderedByColor(false);
    C.set_num_nz( C_num_nnz1 );
    C.values.resize( C_num_nnz1 );

    cusparseCsrSetPointers(matC, C.row_offsets.raw(), C.col_indices.raw(), C.values.raw());

    // copy the final products to the matrix C
    cusparseSpGEMM_copy(handle, opA, opB,
                        &alpha, matA, matB, &beta, matC,
                        computeType, CUSPARSE_SPGEMM_DEFAULT, spgemmDesc);

    C.set_initialized(1);

    // destroy matrix/vector descriptors
    cusparseCheckError( cusparseSpGEMM_destroyDescr(spgemmDesc) );
    cusparseCheckError( cusparseDestroySpMat(matA) );
    cusparseCheckError( cusparseDestroySpMat(matB) );
    cusparseCheckError( cusparseDestroySpMat(matC) );
    amgx::memory::cudaFreeAsync(dBuffer1);
    amgx::memory::cudaFreeAsync(dBuffer2);
}
#endif

#ifndef CUSPARSE_USE_GENERIC_SPGEMM
template< AMGX_VecPrecision V, AMGX_MatPrecision M, AMGX_IndPrecision I > void CSR_Multiply_Impl<TemplateConfig<AMGX_device, V, M, I> >::cusparse_multiply( const Matrix_d &A, const Matrix_d &B, Matrix_d &C, IVector *Aq1, IVector *Bq1, IVector *Aq2, IVector *Bq2 ) {
    size_t pBufferSizeInBytes = 0;
    void *pBuffer = NULL;

    cusparseHandle_t handle = Cusparse::get_instance().get_handle();
    cusparsePointerMode_t old_pointer_mode;
    cusparseCheckError(cusparseGetPointerMode(handle, &old_pointer_mode));
    cusparseSetPointerMode(Cusparse::get_instance().get_handle(), CUSPARSE_POINTER_MODE_HOST);

    // CUSPARSE does not work if the matrix is not sorted. The column indices are not necessarily in order...
    const_cast<Matrix_d &>(A).sortByRowAndColumn();
    const_cast<Matrix_d &>(B).sortByRowAndColumn();

    // Note: If we are re-setup this step then most of this could have been cached...

    // Setup the info structure
    csrgemm2Info_t info = NULL;
    cusparseCheckError(
        cusparseCreateCsrgemm2Info(&info));

    typename Matrix_d::value_type alpha = types::util<typename Matrix_d::value_type>::get_one();

    // Determine the buffer size
    cusparseCheckError(
        cusparseCsrgemmBufferSize(
            handle, A.get_num_rows(), B.get_num_cols(), A.get_num_cols(), &alpha,
            A.cuMatDescr, A.get_num_nz(), A.row_offsets.raw(), A.col_indices.raw(),
            B.cuMatDescr, B.get_num_nz(), B.row_offsets.raw(), B.col_indices.raw(),
            NULL, A.cuMatDescr, 0, A.row_offsets.raw(), A.col_indices.raw(),
            info, &pBufferSizeInBytes));

    // Allocate the intermediary buffer
    amgx::memory::cudaMalloc(&pBuffer, pBufferSizeInBytes);

    int nnzC;
    int *nnzTotalDevHostPtr = &nnzC;

    // Setup C metadata
    C.set_initialized(0);
    C.row_offsets.resize( A.get_num_rows() + 1 );
    C.m_seq_offsets.resize( A.get_num_rows() + 1 );
    thrust::sequence(C.m_seq_offsets.begin(), C.m_seq_offsets.end());
    C.set_num_rows( A.get_num_rows() );
    C.set_num_cols( B.get_num_cols() );
    C.diag.resize(C.get_num_rows());
    C.set_block_dimx(A.get_block_dimx());
    C.set_block_dimy(B.get_block_dimy());
    C.setColsReorderedByColor(false);

    // Compute the row offsets for C
    cusparseCheckError(
        cusparseXcsrgemm2Nnz(
            handle, A.get_num_rows(), B.get_num_cols(), A.get_num_cols(),
            A.cuMatDescr, A.get_num_nz(), A.row_offsets.raw(), A.col_indices.raw(),
            B.cuMatDescr, B.get_num_nz(), B.row_offsets.raw(), B.col_indices.raw(),
            A.cuMatDescr, 0, A.row_offsets.raw(), A.col_indices.raw(),
            C.cuMatDescr, C.row_offsets.raw(), nnzTotalDevHostPtr,
            info, pBuffer));

    // Note the number of non-zeros in C
    int baseC;
    cudaMemcpy(&baseC, C.row_offsets.raw(), sizeof(int), cudaMemcpyDefault);
    cudaMemcpy(&nnzC, C.row_offsets.raw()+A.get_num_rows(), sizeof(int), cudaMemcpyDefault);
    nnzC -= baseC;

    C.col_indices.resize(nnzC);
    C.values.resize(nnzC);
    C.set_num_nz(nnzC);

    // Call the generic cuSPARSE CSR GEMM routine
    cusparseCheckError(
        cusparseCsrgemm2(
            handle, A.get_num_rows(), B.get_num_cols(), A.get_num_cols(),
            &alpha,
            A.cuMatDescr, A.get_num_nz(), A.values.raw(), A.row_offsets.raw(), A.col_indices.raw(),
            B.cuMatDescr, B.get_num_nz(), B.values.raw(), B.row_offsets.raw(), B.col_indices.raw(),
            NULL,
            A.cuMatDescr, 0, A.values.raw(), A.row_offsets.raw(), A.col_indices.raw(),
            C.cuMatDescr, C.values.raw(), C.row_offsets.raw(), C.col_indices.raw(),
            info, pBuffer));

    // Finalise
    C.set_initialized(1);
    cusparseCheckError(
        cusparseSetPointerMode(handle, old_pointer_mode));
    cusparseCheckError(
        cusparseDestroyCsrgemm2Info(info));
    amgx::memory::cudaFreeAsync(pBuffer);
}
#endif



template< AMGX_VecPrecision V, AMGX_MatPrecision M, AMGX_IndPrecision I >
void CSR_Multiply_Impl<TemplateConfig<AMGX_device, V, M, I> >::multiply_opt( 
    const Matrix_d &A, const Matrix_d &B, Matrix_d &C )
{
    // Make C "mutable".
    C.set_initialized(0);
    // Compute row offsets C.
    C.set_num_rows( A.get_num_rows() );
    C.set_num_cols( B.get_num_cols() );
    C.row_offsets.resize( A.get_num_rows() + 1 );
    C.m_seq_offsets.resize( A.get_num_rows() + 1 );
    thrust::sequence(C.m_seq_offsets.begin(), C.m_seq_offsets.end());
    cudaCheckError();

    this->count_non_zeroes_opt(A, B, C, 32);

    // Compute row offsets.
    this->compute_offsets( C );

    // Allocate memory to store columns/values.
    int num_vals = C.row_offsets[C.get_num_rows()];

    C.col_indices.resize(num_vals);
    C.values.resize(num_vals);
    C.set_num_nz(num_vals);
    C.diag.resize( C.get_num_rows() );
    C.set_block_dimx(A.get_block_dimx());
    C.set_block_dimy(B.get_block_dimy());
    C.setColsReorderedByColor(false);

    this->compute_values_opt(A, B, C, 32);
    
    // Finalize the initialization of the matrix.
    C.set_initialized(1);
}

template< AMGX_VecPrecision V, AMGX_MatPrecision M, AMGX_IndPrecision I >
void CSR_Multiply_Impl<TemplateConfig<AMGX_device, V, M, I> >::multiply( const Matrix_d &A, const Matrix_d &B, Matrix_d &C, IVector *Aq1, IVector *Bq1, IVector *Aq2, IVector *Bq2 )
{
    // Make C "mutable".
    C.set_initialized(0);
    // Compute row offsets C.
    C.set_num_rows( A.get_num_rows() );
    C.set_num_cols( B.get_num_cols() );
    C.row_offsets.resize( A.get_num_rows() + 1 );
    C.m_seq_offsets.resize( A.get_num_rows() + 1 );
    thrust::sequence(C.m_seq_offsets.begin(), C.m_seq_offsets.end());
    cudaCheckError();
    bool done = false;

    try
    {
        for ( int attempt = 0 ; !done && attempt < get_max_attempts() ; ++attempt )
        {
            // Double the amount of GMEM (if needed).
            if ( attempt > 0 )
            {
                this->m_gmem_size *= 2;
                this->allocate_workspace();
            }

            // Reset the status.
            int status = 0;
            cudaMemcpy( this->m_status, &status, sizeof(int), cudaMemcpyHostToDevice );
            // Count the number of non-zeroes. The function count_non_zeroes assumes status has been
            // properly set but it is responsible for setting the work queue.
            this->count_non_zeroes( A, B, C, Aq1, Bq1, Aq2, Bq2 );
            // Read the result from count_non_zeroes.
            cudaMemcpy( &status, this->m_status, sizeof(int), cudaMemcpyDeviceToHost );
            done = status == 0;
        }
    }
    catch (std::bad_alloc &e) // We are running out of memory. Try the fallback instead.
    {
        if ( done ) // Just in case but it should never happen.
        {
            throw e;
        }
    }

    // We have to fallback to the CUSPARSE path.
    if ( !done )
    {
        this->cusparse_multiply(A,B,C,Aq1,Bq1,Aq2,Bq2);
    }

    if ( done )
    {
       // Compute row offsets.
       this->compute_offsets( C );
       // Allocate memory to store columns/values.
       int num_vals = C.row_offsets[C.get_num_rows()];
       C.col_indices.resize(num_vals);
       C.values.resize(num_vals);
       C.set_num_nz(num_vals);
       C.diag.resize( C.get_num_rows() );
       C.set_block_dimx(A.get_block_dimx());
       C.set_block_dimy(B.get_block_dimy());
       C.setColsReorderedByColor(false);
       // Like count_non_zeroes, compute_values is responsible for setting its work queue (if it dares :)).
       done = false;

       if ( this->m_num_threads_per_row_count != this->m_num_threads_per_row_compute )
       {
           // Reset the status.
           int status = 0;
           cudaMemcpy( this->m_status, &status, sizeof(int), cudaMemcpyHostToDevice );
           // Count the number of non-zeroes. The function count_non_zeroes assumes status has been
           // properly set but it is responsible for setting the work queue.
           this->compute_values( A, B, C, this->m_num_threads_per_row_compute, Aq1, Bq1, Aq2, Bq2 );
           // Read the result from count_non_zeroes.
           cudaMemcpy( &status, this->m_status, sizeof(int), cudaMemcpyDeviceToHost );
           done = status == 0;
       }

       // Re-run if needed.
       if ( !done )
       {
           this->compute_values( A, B, C, this->m_num_threads_per_row_count, Aq1, Bq1, Aq2, Bq2 );
       }
    }
    // Finalize the initialization of the matrix.
    C.set_initialized(1);
}

template< AMGX_VecPrecision V, AMGX_MatPrecision M, AMGX_IndPrecision I >
void CSR_Multiply_Impl<TemplateConfig<AMGX_device, V, M, I> >::sparse_add( Matrix_d &RAP, const Matrix_d &RAP_int, std::vector<IVector> &RAP_ext_row_offsets, std::vector<IVector> &RAP_ext_col_indices, std::vector<MVector> &RAP_ext_values, std::vector<IVector> &RAP_ext_row_ids)
{
    // Make C "mutable".
    RAP.set_initialized(0);
    RAP.m_seq_offsets.resize( RAP.get_num_rows() + 1 );
    thrust::sequence(RAP.m_seq_offsets.begin(), RAP.m_seq_offsets.end());
    cudaCheckError();
    int attempt = 0;

    for ( bool done = false ; !done && attempt < 10 ; ++attempt )
    {
        // Double the amount of GMEM (if needed).
        if ( attempt > 0 )
        {
            this->m_gmem_size *= 2;
            this->allocate_workspace();
        }

        // Reset the status.
        int status = 0;
        cudaMemcpy( this->m_status, &status, sizeof(int), cudaMemcpyHostToDevice );
        // Count the number of non-zeroes. The function count_non_zeroes assumes status has been
        // properly set but it is responsible for setting the work queue.
        this->count_non_zeroes_RAP_sparse_add( RAP, RAP_int, RAP_ext_row_offsets, RAP_ext_col_indices, RAP_ext_values, RAP_ext_row_ids );
        // Read the result from count_non_zeroes.
        cudaMemcpy( &status, this->m_status, sizeof(int), cudaMemcpyDeviceToHost );
        done = status == 0;
    }

    // Compute row offsets.
    this->compute_offsets( RAP );
    // Allocate memory to store columns/values.
    int num_vals = RAP.row_offsets[RAP.get_num_rows()];
    RAP.col_indices.resize(num_vals);
    RAP.values.resize(num_vals);
    RAP.set_num_nz(num_vals);
    RAP.diag.resize( RAP.get_num_rows() );
    RAP.set_block_dimx(RAP_int.get_block_dimx());
    RAP.set_block_dimy(RAP_int.get_block_dimy());
    RAP.setColsReorderedByColor(false);
    // Like count_non_zeroes, compute_values is responsible for setting its work queue (if it dares :)).
    bool done = false;

    if ( this->m_num_threads_per_row_count != this->m_num_threads_per_row_compute )
    {
        // Reset the status.
        int status = 0;
        cudaMemcpy( this->m_status, &status, sizeof(int), cudaMemcpyHostToDevice );
        // Count the number of non-zeroes. The function count_non_zeroes assumes status has been
        // properly set but it is responsible for setting the work queue.
        this->compute_values_RAP_sparse_add( RAP, RAP_int, RAP_ext_row_offsets, RAP_ext_col_indices, RAP_ext_values, RAP_ext_row_ids,  this->m_num_threads_per_row_compute );
        // Read the result from count_non_zeroes.
        cudaMemcpy( &status, this->m_status, sizeof(int), cudaMemcpyDeviceToHost );
        done = status == 0;
    }

    // Re-run if needed.
    if ( !done )
    {
        this->compute_values_RAP_sparse_add( RAP, RAP_int, RAP_ext_row_offsets, RAP_ext_col_indices, RAP_ext_values, RAP_ext_row_ids, this->m_num_threads_per_row_count );
    }
}



// ====================================================================================================================

template< AMGX_VecPrecision V, AMGX_MatPrecision M, AMGX_IndPrecision I >
void CSR_Multiply_Impl<TemplateConfig<AMGX_device, V, M, I> >::galerkin_product( const Matrix_d &R, const Matrix_d &A, const Matrix_d &P, Matrix_d &RAP, IVector *Rq1, IVector *Aq1, IVector *Pq1, IVector *Rq2, IVector *Aq2, IVector *Pq2)
{
    nvtxRangePush("galerkin");

    Matrix_d AP;
    AP.set_initialized(0);
    int avg_nz_per_row = P.get_num_nz() / P.get_num_rows();

    if ( avg_nz_per_row < 2 )
    {
        this->set_num_threads_per_row_count(2);
        this->set_num_threads_per_row_compute(2);
    }
    else
    {
        this->set_num_threads_per_row_count(4);
        this->set_num_threads_per_row_compute(4);
    }

    nvtxRangePush("AP");
    if(this->m_use_opt_kernels)
    {
        this->multiply_opt( A, P, AP );
    }
    else if(this->m_use_cusparse_kernels)
    {
        this->cusparse_multiply(A, P, AP, NULL, NULL, NULL, NULL);
    }
    else
    {
        this->multiply( A, P, AP, NULL, NULL, NULL, NULL );
    }
    nvtxRangePop();

    AP.set_initialized(1);
    avg_nz_per_row = AP.get_num_nz() / AP.get_num_rows();
    this->set_num_threads_per_row_count(avg_nz_per_row <= 16.0 ? 8 : 32);
    this->set_num_threads_per_row_compute(32);
    RAP.set_initialized(0);

    nvtxRangePush("RAP");
    if(this->m_use_opt_kernels)
    {
        this->multiply_opt( R, AP, RAP );
    }
    else if(this->m_use_cusparse_kernels)
    {
        this->cusparse_multiply(R, AP, RAP, NULL, NULL, NULL, NULL);
    }
    else
    {
        this->multiply( R, AP, RAP, NULL, NULL, NULL, NULL );
    }
    nvtxRangePop();

    RAP.computeDiagonal();
    RAP.set_initialized(1);
    nvtxRangePop();
}

template< AMGX_VecPrecision V, AMGX_MatPrecision M, AMGX_IndPrecision I >
void CSR_Multiply_Impl<TemplateConfig<AMGX_device, V, M, I> >::RAP_sparse_add( Matrix_d &RAP, const Matrix_d &RAP_int, std::vector<IVector> &RAP_ext_row_offsets, std::vector<IVector> &RAP_ext_col_indices, std::vector<MVector> &RAP_ext_values, std::vector<IVector> &RAP_ext_row_ids)
{
    if (RAP_int.get_num_rows() <= 0)
    {
        return;
    }

    int avg_nz_per_row = RAP_int.get_num_nz() / RAP_int.get_num_rows();
    this->set_num_threads_per_row_count(avg_nz_per_row <= 16.0 ? 8 : 32);
    this->set_num_threads_per_row_compute(32);
    RAP.set_initialized(0);
    this->sparse_add( RAP, RAP_int, RAP_ext_row_offsets, RAP_ext_col_indices, RAP_ext_values, RAP_ext_row_ids );
    RAP.computeDiagonal();
    RAP.set_initialized(1);
}


// ====================================================================================================================

template< AMGX_VecPrecision V, AMGX_MatPrecision M, AMGX_IndPrecision I >
void CSR_Multiply_Impl<TemplateConfig<AMGX_device, V, M, I> >::sparsity( const Matrix_d &A, const Matrix_d &B, Matrix_d &C )
{
    // Make C "mutable".
    C.set_initialized(0);
    // Compute row offsets C.
    C.set_num_rows( A.get_num_rows() );
    C.set_num_cols( B.get_num_cols() );
    C.row_offsets.resize( A.get_num_rows() + 1 );
    int attempt = 0;

    for ( bool done = false ; !done && attempt < 10 ; ++attempt )
    {
        // Double the amount of GMEM (if needed).
        if ( attempt > 0 )
        {
            this->m_gmem_size *= 2;
            this->allocate_workspace();
        }

        // Reset the status.
        int status = 0;
        cudaMemcpy( this->m_status, &status, sizeof(int), cudaMemcpyHostToDevice );
        // Count the number of non-zeroes. The function count_non_zeroes assumes status has been
        // properly set but it is responsible for setting the work queue.
        this->count_non_zeroes( A, B, C, NULL, NULL, NULL, NULL );
        // Read the result from count_non_zeroes.
        cudaMemcpy( &status, this->m_status, sizeof(int), cudaMemcpyDeviceToHost );
        done = status == 0;
    }

    // Compute row offsets.
    this->compute_offsets( C );
    // Allocate memory to store columns/values.
    int num_vals = C.row_offsets[C.get_num_rows()];
    C.col_indices.resize(num_vals);
    C.values.resize(num_vals);
    C.set_num_nz(num_vals);
    C.diag.resize( C.get_num_rows( ) );
    C.setColsReorderedByColor(false);
    // Like count_non_zeroes, compute_values is responsible for setting its work queue (if it dares :)).
    this->compute_sparsity( A, B, C );
    // Finalize the initialization of the matrix.
    C.set_initialized(1);
}

// ====================================================================================================================

template< AMGX_VecPrecision V, AMGX_MatPrecision M, AMGX_IndPrecision I >
void CSR_Multiply_Impl<TemplateConfig<AMGX_device, V, M, I> >::sparsity( const Matrix_d &A, Matrix_d &B )
{
    this->sparsity( A, A, B );
}

// ====================================================================================================================

template< AMGX_VecPrecision V, AMGX_MatPrecision M, AMGX_IndPrecision I >
void CSR_Multiply_Impl<TemplateConfig<AMGX_device, V, M, I> >::sparsity_ilu1( const Matrix_d &A, Matrix_d &B )
{
    // Make C "mutable".
    B.set_initialized(0);
    // Compute row offsets C.
    B.set_num_rows( A.get_num_rows() );
    B.set_num_cols( A.get_num_cols() );
    B.row_offsets.resize( A.get_num_rows() + 1 );
    int attempt = 0;

    for ( bool done = false ; !done && attempt < 10 ; ++attempt )
    {
        // Double the amount of GMEM (if needed).
        if ( attempt > 0 )
        {
            this->m_gmem_size *= 2;
            this->allocate_workspace();
        }

        // Reset the status.
        int status = 0;
        CUDA_SAFE_CALL( cudaMemcpy( this->m_status, &status, sizeof(int), cudaMemcpyHostToDevice ) );
        // Count the number of non-zeroes. The function count_non_zeroes assumes status has been
        // properly set but it is responsible for setting the work queue.
        this->count_non_zeroes_ilu1( A, B );
        // Read the result from count_non_zeroes.
        CUDA_SAFE_CALL( cudaMemcpy( &status, this->m_status, sizeof(int), cudaMemcpyDeviceToHost ) );
        done = status == 0;
    }

    // Compute row offsets.
    this->compute_offsets(B);
    // Allocate memory to store columns/values.
    int num_vals = B.row_offsets[B.get_num_rows()];
    B.col_indices.resize(num_vals);
    B.values.resize((num_vals + 1)*A.get_block_size());
    B.set_num_nz(num_vals);
    B.diag.resize( B.get_num_rows() );
    B.set_block_dimx(A.get_block_dimx());
    B.set_block_dimy(A.get_block_dimy());
    // Like count_non_zeroes, compute_values is responsible for setting its work queue (if it dares :)).
    this->compute_sparsity_ilu1( A, B );
    // Finalize the initialization of the matrix.
    B.setColsReorderedByColor(false);
    B.set_initialized(1);
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#define AMGX_CASE_LINE(CASE) template class CSR_Multiply<TemplateMode<CASE>::Type>;
AMGX_FORALL_BUILDS(AMGX_CASE_LINE)
AMGX_FORCOMPLEX_BUILDS(AMGX_CASE_LINE)
#undef AMGX_CASE_LINE

#define AMGX_CASE_LINE(CASE) template class CSR_Multiply_Impl<TemplateMode<CASE>::Type>;
AMGX_FORALL_BUILDS(AMGX_CASE_LINE)
AMGX_FORCOMPLEX_BUILDS(AMGX_CASE_LINE)
#undef AMGX_CASE_LINE

} // namespace amgx
