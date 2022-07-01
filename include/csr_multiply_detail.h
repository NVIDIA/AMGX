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

//#pragma once

namespace amgx
{

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Implementation of the CSR_Multiply routines.
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template< typename T_Config >
class CSR_Multiply_Detail
{};

template< AMGX_VecPrecision V, AMGX_MatPrecision M, AMGX_IndPrecision I >
class CSR_Multiply_Detail<TemplateConfig<AMGX_device, V, M, I> > : public CSR_Multiply_Impl<TemplateConfig<AMGX_device, V, M, I> >
{
        typedef CSR_Multiply_Impl<TemplateConfig<AMGX_device, V, M, I> > Base;
    public:
        typedef typename Base::TConfig_d TConfig_d;
        typedef typename Base::Matrix_d Matrix_d;
        typedef typename Matrix_d::IVector IVector_d;
        typedef typename Base::Value_type Value_type;
        typedef typename Matrix_d::IVector IVector;
        typedef typename Matrix_d::MVector MVector;

    public:
        // Create a workspace to run the product.
        CSR_Multiply_Detail( bool allocate_values = true, int grid_size = 1024, int max_warp_count = 8, int gmem_size = 512 );

    protected:
        // Count the number of non-zero elements. The callee is responsible for setting the work queue value.
        void count_non_zeroes( const Matrix_d &A, const Matrix_d &B, Matrix_d &C, IVector *Aq1, IVector *Bq1, IVector *Aq2, IVector *Bq2 );
        void count_non_zeroes_opt( const Matrix_d &A, const Matrix_d &B, Matrix_d &C, int num_threads);
        // Compute the sparsity of RAP_int + RAP_ext
        void count_non_zeroes_RAP_sparse_add( Matrix_d &RAP, const Matrix_d &RAP_int, std::vector<IVector> &RAP_ext_row_offsets, std::vector<IVector> &RAP_ext_col_indices, std::vector<MVector> &RAP_ext_values, std::vector<IVector> &RAP_ext_row_ids);

        // Compute the ILU1 sparsity of A.
        void count_non_zeroes_ilu1( const Matrix_d &A, Matrix_d &B );
        // Compute offsets.
        void compute_offsets( Matrix_d &C );
        // Compute the sparsity of the product AxB.
        void compute_sparsity( const Matrix_d &A, const Matrix_d &B, Matrix_d &C );
        // Compute the ILU1 sparsity of A.
        void compute_sparsity_ilu1( const Matrix_d &A, Matrix_d &B );
        // Compute values.
        void compute_values( const Matrix_d &A, const Matrix_d &B, Matrix_d &C, int num_threads, IVector *Aq1, IVector *Bq1, IVector *Aq2, IVector *Bq2 );
        void compute_values_opt( const Matrix_d &A, const Matrix_d &B, Matrix_d &C, int num_threads);
        void compute_values_RAP_sparse_add( Matrix_d &RAP, const Matrix_d &RAP_int, std::vector<IVector> &RAP_ext_row_offsets, std::vector<IVector> &RAP_ext_col_indices, std::vector<MVector> &RAP_ext_values, std::vector<IVector> &RAP_ext_row_ids, int num_threads);
};

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace amgx

