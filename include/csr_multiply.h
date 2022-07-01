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

#include <amg.h>
#include <basic_types.h>
#include <hash_workspace.h>
#include <cusp/detail/device/utils.h> // CUDA_SAFE_CALL

namespace amgx
{

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template< class T_Config >
struct CSR_Multiply
{
    typedef Matrix<T_Config> Matrix_hd;
    typedef typename Matrix_hd::IVector IVector_hd;
    typedef typename Matrix_hd::IVector IVector;
    typedef typename Matrix_hd::MVector MVector;

    // Run a simple sparse matrix-matrix multiplication.
    static void
    csr_multiply( const Matrix_hd &A, const Matrix_hd &B, Matrix_hd &C, void *wk )
    {
        FatalError( "csr_multiply: Not implemented on host", AMGX_ERR_NOT_IMPLEMENTED );
    };

    // Compute the Galerkin product RAP.
    static void
    csr_galerkin_product( const Matrix_hd &R, const Matrix_hd &A, const Matrix_hd &P, Matrix_hd &RAP, IVector *Rq1, IVector *Aq1, IVector *Pq1, IVector *Rq2, IVector *Aq2, IVector *Pq2,  void *wk)
    {
        FatalError( "csr_galerkin_product: Not implemented on host", AMGX_ERR_NOT_IMPLEMENTED );
    }

    // Compute the sparse addition of RAP and RAP_ext
    static void
    csr_RAP_sparse_add( Matrix_hd &RAP, const Matrix_hd &RAP_int, std::vector<IVector> &RAP_ext_row_offsets, std::vector<IVector> &RAP_ext_col_indices, std::vector<MVector> &RAP_ext_values, std::vector<IVector> &RAP_ext_row_ids, void *wk )
    {
        FatalError( "csr_RAP_sparse_add: Not implemented on host", AMGX_ERR_NOT_IMPLEMENTED );
    }

    // Create a new workspace.
    static void
    csr_workspace_delete( void *workspace )
    {
        FatalError( "csr_workspace_delete: Not implemented on host", AMGX_ERR_NOT_IMPLEMENTED );
    }
};

template< class T_Config >
class CSR_Multiply_Impl
{};

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Device specialization
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template< AMGX_VecPrecision V, AMGX_MatPrecision M, AMGX_IndPrecision I >
struct CSR_Multiply<TemplateConfig<AMGX_device, V, M, I> >
{
    typedef TemplateConfig<AMGX_device, V, M, I> TConfig_d;

    static const AMGX_VecPrecision vecPrec = TConfig_d::vecPrec;
    static const AMGX_MatPrecision matPrec = TConfig_d::matPrec;
    static const AMGX_IndPrecision indPrec = TConfig_d::indPrec;

    typedef Matrix<TConfig_d> Matrix_d;
    typedef typename Matrix_d::IVector IVector_d;
    typedef typename Matrix_d::IVector IVector;
    typedef typename Matrix_d::MVector MVector;

    // Run a simple sparse matrix-matrix multiplication.
    static void
    csr_multiply( const Matrix_d &A, const Matrix_d &B, Matrix_d &C, void *wk = NULL );

    // Compute the sparsity pattern of B = A*A.
    static void
    csr_sparsity( const Matrix_d &A, Matrix_d &B, void *wk = NULL );
    // Compute the sparsity pattern of C = A*B.
    static void
    csr_sparsity( const Matrix_d &A, const Matrix_d &B, Matrix_d &C, void *wk = NULL );
    // Compute the sparsity pattern of ILU1 of A.
    static void
    csr_sparsity_ilu1( const Matrix_d &A, Matrix_d &B, void *wk = NULL );

    // Compute the sparse addition of RAP and RAP_ext
    static void
    csr_RAP_sparse_add( Matrix_d &RAP, const Matrix_d &RAP_int, std::vector<IVector> &RAP_ext_row_offsets, std::vector<IVector> &RAP_ext_col_indices, std::vector<MVector> &RAP_ext_values, std::vector<IVector> &RAP_ext_row_ids, void *wk = NULL );

    // Compute the Galerkin product RAP.
    static void
    csr_galerkin_product( const Matrix_d &R, const Matrix_d &A, const Matrix_d &P, Matrix_d &RAP, IVector *Rq1, IVector *Aq1, IVector *Pq1, IVector *Rq2, IVector *Aq2, IVector *Pq2, void *wk = NULL);

    // Create a new workspace.
    static void *
    csr_workspace_create();
    // Create a new workspace.
    static void *
    csr_workspace_create( AMG_Config &cfg, const std::string &cfg_scope );
    // Delete an existing workspace.
    static void
    csr_workspace_delete( void *workspace );
};

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Base class for architecture-dependent implementation of the routines.
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template< AMGX_VecPrecision V, AMGX_MatPrecision M, AMGX_IndPrecision I >
class CSR_Multiply_Impl<TemplateConfig<AMGX_device, V, M, I> > : public Hash_Workspace<TemplateConfig<AMGX_device, V, M, I>, int >
{
    public:
        typedef TemplateConfig<AMGX_device, V, M, I> TConfig_d;
        typedef Hash_Workspace<TConfig_d, int> Base;

        typedef Matrix<TConfig_d> Matrix_d;
        typedef typename Matrix_d::IVector IVector_d;
        typedef typename MatPrecisionMap<M>::Type Value_type;
        typedef typename Matrix_d::IVector IVector;
        typedef typename Matrix_d::MVector MVector;

    protected:
        // The number of attempts before the fallback to CUSPARSE.
        int m_max_attempts;

        int m_use_opt_kernels = 0;

    public:
        // Create a workspace to run the product.
        CSR_Multiply_Impl( bool allocate_vals = true, int grid_size = 128, int max_warp_count = 8, int gmem_size = 2048 );

        // Compute the product between two CSR matrices.
        void multiply( const Matrix_d &A, const Matrix_d &B, Matrix_d &C, IVector *Aq1, IVector *Bq1, IVector *Aq2, IVector *Bq2 );

        void multiply_opt( const Matrix_d &A, const Matrix_d &B, Matrix_d &C );

        void sparse_add( Matrix_d &RAP, const Matrix_d &RAP_int, std::vector<IVector> &RAP_ext_row_offsets, std::vector<IVector> &RAP_ext_col_indices, std::vector<MVector> &RAP_ext_values, std::vector<IVector> &RAP_ext_row_ids);

        // Compute the sparsity pattern of a product.
        void sparsity( const Matrix_d &A, Matrix_d &B );
        // Compute the sparsity pattern of a product.
        void sparsity( const Matrix_d &A, const Matrix_d &B, Matrix_d &C );
        // Compute the sparsity pattern of a product.
        void sparsity_ilu1( const Matrix_d &A, Matrix_d &B );
        // Compute the Galerkin product of three matrices.
        void galerkin_product( const Matrix_d &R, const Matrix_d &A, const Matrix_d &P, Matrix_d &RAP, IVector *Rq1, IVector *Aq1, IVector *Pq1, IVector *Rq2, IVector *Aq2, IVector *Pq2 );
        void RAP_sparse_add( Matrix_d &RAP, const Matrix_d &RAP_int, std::vector<IVector> &RAP_ext_row_offsets, std::vector<IVector> &RAP_ext_col_indices, std::vector<MVector> &RAP_ext_values, std::vector<IVector> &RAP_ext_row_ids);

        // The max number of attempts before the fallback to CUSPARSE.
        inline int get_max_attempts() const { return m_max_attempts; }
        // Set the max number of attempts before the fallback to CUSPARSE.
        inline void set_max_attempts(int max_attempts) { m_max_attempts = max_attempts; }
        inline void set_opt_multiply(bool use_opt_kernels) { m_use_opt_kernels = use_opt_kernels; }

    protected:
        // Count the number of non-zero elements. The callee is responsible for setting the work queue value.
        virtual void count_non_zeroes( const Matrix_d &A, const Matrix_d &B, Matrix_d &C, IVector *Aq1, IVector *Bq1, IVector *Aq2, IVector *Bq2 ) = 0;
        virtual void count_non_zeroes_opt( const Matrix_d &A, const Matrix_d &B, Matrix_d &C, int num_threads ) = 0;

        // Compute the sparsity of RAP_int + RAP_ext
        virtual void count_non_zeroes_RAP_sparse_add( Matrix_d &RAP, const Matrix_d &RAP_int, std::vector<IVector> &RAP_ext_row_offsets, std::vector<IVector> &RAP_ext_col_indices, std::vector<MVector> &RAP_ext_values, std::vector<IVector> &RAP_ext_row_ids) = 0;

        // Compute the ILU1 sparsity of A.
        virtual void count_non_zeroes_ilu1( const Matrix_d &A, Matrix_d &B ) = 0;
        // Compute offsets.
        virtual void compute_offsets( Matrix_d &C ) = 0;
        // Compute the sparsity of the product AxB.
        virtual void compute_sparsity( const Matrix_d &A, const Matrix_d &B, Matrix_d &C ) = 0;
        // Compute the ILU1 sparsity of A.
        virtual void compute_sparsity_ilu1( const Matrix_d &A, Matrix_d &B ) = 0;
        // Compute values.
        virtual void compute_values( const Matrix_d &A, const Matrix_d &B, Matrix_d &C, int num_threads, IVector *Aq1, IVector *Bq1, IVector *Aq2, IVector *Bq2 ) = 0;
        virtual void compute_values_opt( const Matrix_d &A, const Matrix_d &B, Matrix_d &C, int num_threads) = 0;
        virtual void compute_values_RAP_sparse_add( Matrix_d &RAP, const Matrix_d &RAP_int, std::vector<IVector> &RAP_ext_row_offsets, std::vector<IVector> &RAP_ext_col_indices, std::vector<MVector> &RAP_ext_values, std::vector<IVector> &RAP_ext_row_ids, int num_threads) = 0;

    private:
        void cusparse_multiply( const Matrix_d &A, const Matrix_d &B, Matrix_d &C, IVector *Aq1, IVector *Bq1, IVector *Aq2, IVector *Bq2 );

};

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace amgx

