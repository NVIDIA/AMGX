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


#include <cusparse_v2.h>
#include <error.h>
#include <matrix.h>
#include <vector.h>
#include <types.h>
#include <thrust/sequence.h>
#include <util.h>
#include <sm_utils.inl>

#include <amgx_cusparse.h>

#include <amgx_types/util.h>

#if CUDART_VERSION < 11000
#define CUSPARSE_SPMM_ALG_DEFAULT CUSPARSE_MM_ALG_DEFAULT
#endif

namespace amgx
{

Cusparse::Cusparse() : m_handle(0)
{
    cusparseCheckError( cusparseCreate(&m_handle) );
}

Cusparse::~Cusparse()
{
    destroy_handle();
}

Cusparse &Cusparse::get_instance()
{
    static Cusparse s_instance;
    s_instance.create_handle();
    return s_instance;
}

#ifndef DISABLE_MIXED_PRECISION
template <class T_Config>
cusparseStatus_t
CusparseMatPrec<T_Config>::set(cusparseMatDescr_t &cuMatDescr)
{
    return cusparseSetMatFullPrecision(cuMatDescr, true);
}

template <AMGX_MemorySpace t_memSpace, AMGX_IndPrecision t_indPrec>
cusparseStatus_t CusparseMatPrec< TemplateConfig<t_memSpace, AMGX_vecDouble, AMGX_matFloat, t_indPrec> >::set(cusparseMatDescr_t &cuMatDescr)
{
    return cusparseSetMatFullPrecision(cuMatDescr, false);
}

template <AMGX_MemorySpace t_memSpace, AMGX_IndPrecision t_indPrec>
cusparseStatus_t CusparseMatPrec< TemplateConfig<t_memSpace, AMGX_vecDoubleComplex, AMGX_matComplex, t_indPrec> >::set(cusparseMatDescr_t &cuMatDescr)
{
    return cusparseSetMatFullPrecision(cuMatDescr, false);
}
#endif

template< class TConfig >
void Cusparse::bsrmv(
    const typename TConfig::VecPrec alphaConst,
    Matrix<TConfig> &A,
    Vector<TConfig> &x,
    const typename TConfig::VecPrec betaConst,
    Vector<TConfig> &y,
    ViewType view )
{
    cudaStream_t null_stream = 0;

    // If only COO, add CSR since bsrmv doesn't support COO
    if (A.hasProps(COO) && !A.hasProps(CSR))
    {
        A.set_initialized(0);
        A.addProps(CSR);
        A.computeDiagonal();
        A.set_initialized(1);
    }

    // Handle cases where the view is set by the calling routine
    if(view != A.getViewExterior())
    {
        bsrmv_internal(alphaConst, A, x, betaConst, y, view, null_stream);
        return;
    }

    bool latencyHiding = (A.getViewInterior() != A.getViewExterior() && !A.is_matrix_singleGPU() && x.dirtybit != 0);

    if (latencyHiding)
    {
        A.manager->exchange_halo_split_gather(x, x.tag);

        // Multiply interior rows
        bsrmv_internal(alphaConst, A, x, betaConst, y, A.getViewInterior(), null_stream);

        // Finish halo exchange
        A.manager->exchange_halo_split_finish(x, x.tag);

        // Multiply rows with halo dependencies
        ViewType bnd_view = (ViewType)(~(A.getViewInterior()) & A.getViewExterior());
        bsrmv_internal(alphaConst, A, x, betaConst, y, bnd_view, null_stream);
    }
    else
    {
        if (!A.is_matrix_singleGPU() && x.dirtybit != 0)
        {
            A.manager->exchange_halo_v2(x, x.tag);
        }

        bsrmv_internal(alphaConst, A, x, betaConst, y, A.getViewExterior(), null_stream);
    }

    y.dirtybit = 1;
}

template< class TConfig >
void Cusparse::bsrmv_with_mask(
    const typename TConfig::VecPrec alphaConst,
    Matrix<TConfig> &A,
    Vector<TConfig> &x,
    const typename TConfig::VecPrec betaConst,
    Vector<TConfig> &y)
{
    cudaStream_t null_stream = 0;

    // If only COO, add CSR since bsrmv doesn't support COO
    if (A.hasProps(COO) && !A.hasProps(CSR))
    {
        A.set_initialized(0);
        A.addProps(CSR);
        A.computeDiagonal();
        A.set_initialized(1);
    }

    bool latencyHiding = (A.getViewInterior() != A.getViewExterior() && !A.is_matrix_singleGPU() && x.dirtybit != 0);

    if (latencyHiding)
    {
        A.manager->exchange_halo_split_gather(x, x.tag);

        // Multiply interior
        bsrmv_internal_with_mask(alphaConst, A, x, betaConst, y, INTERIOR, null_stream);

        A.manager->exchange_halo_split_finish(x, x.tag);

        // Multiply exterior
        bsrmv_internal_with_mask(alphaConst, A, x, betaConst, y, BOUNDARY, null_stream);
    }
    else
    {
        if (!A.is_matrix_singleGPU() && x.dirtybit != 0)
        {
            A.manager->exchange_halo_v2(x, x.tag);
        }

        bsrmv_internal(alphaConst, A, x, betaConst, y, OWNED, null_stream);
    }

    y.dirtybit = 1;
}

template< class TConfig >
void Cusparse::bsrmv_with_mask_restriction(
    const typename TConfig::VecPrec alphaConst,
    Matrix<TConfig> &R,
    Vector<TConfig> &x,
    const typename TConfig::VecPrec betaConst,
    Vector<TConfig> &y,
    Matrix<TConfig> &P)
{
    // If only COO, add CSR since bsrmv doesn't support COO
    //if (A.hasProps(COO) && !A.hasProps(CSR))
    //{
    //  A.set_initialized(0);
    //  A.addProps(CSR);
    //  A.computeDiagonal();
    //  A.set_initialized(1);
    //}

    bool latencyHiding = (R.getViewInterior() != R.getViewExterior() && !P.is_matrix_singleGPU() && x.dirtybit != 0);

    if (latencyHiding)
    {
        cudaStream_t null_stream = 0;
        bsrmv_internal_with_mask_restriction(alphaConst, R, x, betaConst, y, HALO1, null_stream, P);
        P.manager->add_from_halo_split_gather(y, y.tag);
        cudaEventRecord(P.manager->get_comm_event());
        bsrmv_internal_with_mask_restriction(alphaConst, R, x, betaConst, y, OWNED, null_stream, P);

        if (P.manager->neighbors.size() != 0)
        {
            cudaEventSynchronize(P.manager->get_comm_event());
            P.manager->add_from_halo_split_finish(y, y.tag, P.manager->get_bdy_stream());
            cudaStreamSynchronize(P.manager->get_bdy_stream());
        }
    }
    else
    {
        bsrmv_internal(alphaConst, R, x, betaConst, y, OWNED, 0);

        // Add contribution from neighbors
        P.manager->add_from_halo_v2(y, y.tag);
    }

    y.dirtybit = 1;
}

template< class TConfig >
void Cusparse::bsrxmv(
    const typename TConfig::VecPrec alphaConst,
    Matrix<TConfig> &A,
    Vector<TConfig> &x,
    const typename TConfig::VecPrec betaConst,
    Vector<TConfig> &y,
    typename Matrix<TConfig>::IVector &mask,
    ViewType view )
{
    // If only COO, add CSR since bsrmv doesn't support COO
    if (A.hasProps(COO) && !A.hasProps(CSR))
    {
        A.set_initialized(0);
        A.addProps(CSR);
        A.computeDiagonal();
        A.set_initialized(1);
    }

    const int *start_offsets, *end_offsets;
    start_offsets = A.row_offsets.raw();
    end_offsets = A.row_offsets.raw() + 1;
    cusparseDirection_t direction = CUSPARSE_DIRECTION_COLUMN;

    if ( A.getBlockFormat() == ROW_MAJOR )
    {
        direction = CUSPARSE_DIRECTION_ROW;
    }

    if (view == OWNED)  //This is already a view, thus do not even attempt to do latency hiding
    {
        bsrxmv_internal(Cusparse::get_instance().m_handle, direction, CUSPARSE_OPERATION_NON_TRANSPOSE,
                        mask.size(),
                        A.get_num_rows(), A.get_num_cols(), A.get_num_nz(), &alphaConst,
                        A.cuMatDescr,
                        A.values.raw(),
                        mask.raw(),
                        start_offsets, end_offsets,
                        A.col_indices.raw(),
                        A.get_block_dimx(),
                        x.raw(),
                        &betaConst,
                        y.raw());
    }
    else //Try and do latency hiding
    {
        // latency hiding?
        FatalError("Trying to do latency hiding in the bsrxmv", AMGX_ERR_NOT_IMPLEMENTED);
    }
}

// E is a vector that represents a diagonal matrix
// operate on all rows and columns
// y= alpha*E.x + beta*y
template< class TConfig >
void Cusparse::bsrmv( const typename TConfig::VecPrec alphaConst,
                      Matrix<TConfig> &A,
                      const typename Matrix<TConfig>::MVector &E,
                      Vector<TConfig> &x,
                      const typename TConfig::VecPrec betaConst,
                      Vector<TConfig> &y,
                      ViewType view )
{
    cudaStream_t null_stream = 0;

    // If only COO, add CSR since bsrmv doesn't support COO
    if (A.hasProps(COO) && !A.hasProps(CSR))
    {
        A.addProps(CSR);
    }

    if (view != A.getViewExterior()) //This is already a view, thus do not even attempt to do latency hiding
    {
        bsrmv_internal(alphaConst, A, E, x, betaConst, y, view, null_stream);
    }
    else  //Try and do latency hiding
    {
        ViewType oldView = A.currentView();

        if (!A.is_matrix_singleGPU())
        {
            A.manager->exchange_halo_async(x, x.tag);
        }

        if (A.getViewExterior() == A.getViewInterior())
        {
            if (!A.is_matrix_singleGPU())
            {
                A.manager->exchange_halo_wait(x, x.tag);
            }
        }

        ViewType flags;
        bool latencyHiding = true;

        //if (A.manager->num_neighbors() == 0 || (x.dirtybit == 0)) {
        if (A.is_matrix_singleGPU() || (x.dirtybit == 0))
        {
            latencyHiding = false;
            A.setViewExterior();
            flags = (ViewType)(A.getViewExterior());
        }
        else
        {
            flags = (ViewType)(A.getViewInterior());
            A.setViewInterior();
        }

        if (latencyHiding)
        {
            bsrmv_internal(alphaConst, A, E, x, betaConst, y, flags, null_stream);

            if (!A.is_matrix_singleGPU())
            {
                A.manager->exchange_halo_wait(x, x.tag);
            }

            A.setViewExterior();
            flags = (ViewType)(~(A.getViewInterior()) & A.getViewExterior());

            if (flags != 0)
            {
                bsrmv_internal(alphaConst, A, E, x, betaConst, y, flags, null_stream);
            }
        }
        else
        {
            bsrmv_internal(alphaConst, A, E, x, betaConst, y, flags, null_stream);
        }

        y.dirtybit = 1;
        //if (!A.is_matrix_singleGPU())
        //  if (y.size() == x.size() && y.delayed_send==0) A.manager->exchange_halo_async(y, y.tag);
        A.setView(oldView);
    }
}

// operate only on columns specified by columnColorSelector, see enum ColumnColorSelector above
// operate only on rows of specified color, given by A.offsets_rows_per_color, A.sorted_rows_by_color
// y= alpha*A.x + beta*y
template< class TConfig >
void Cusparse::bsrmv( ColumnColorSelector columnColorSelector,
                      const int color,
                      const typename TConfig::VecPrec alphaConst,
                      Matrix<TConfig> &A,
                      Vector<TConfig> &x,
                      const typename TConfig::VecPrec betaConst,
                      Vector<TConfig> &y,
                      ViewType view )
{
    cudaStream_t null_stream = 0;

    // If only COO, add CSR since bsrmv doesn't support COO
    if (A.hasProps(COO) && !A.hasProps(CSR))
    {
        A.addProps(CSR);
    }

    if (view != A.getViewExterior())  //This is already a view, thus do not even attempt to do latency hiding
    {
        // std::cerr << "exterior view with selector" << std::endl;
        bsrmv_internal(columnColorSelector, color, alphaConst, A, x, betaConst, y, view, null_stream);
    }
    else
    {
        //Try and do latency hiding
        ViewType oldView = A.currentView();

        if (!A.is_matrix_singleGPU())
        {
            A.manager->exchange_halo_async(x, x.tag);
        }

        if (A.getViewExterior() == A.getViewInterior())
        {
            if (!A.is_matrix_singleGPU())
            {
                A.manager->exchange_halo_wait(x, x.tag);
            }
        }

        ViewType flags;
        bool latencyHiding = true;

        if (A.is_matrix_singleGPU() || (x.dirtybit == 0))
        {
            latencyHiding = false;
            A.setViewExterior();
            flags = (ViewType)(A.getViewExterior());
        }
        else
        {
            flags = (ViewType)(A.getViewInterior());
            A.setViewInterior();
        }

        if (latencyHiding)
        {
            bsrmv_internal(columnColorSelector, color, alphaConst, A, x, betaConst, y, flags, null_stream);

            if (!A.is_matrix_singleGPU())
            {
                A.manager->exchange_halo_wait(x, x.tag);
            }

            A.setViewExterior();
            flags = (ViewType)(~(A.getViewInterior()) & A.getViewExterior());

            if (flags != 0)
            {
                bsrmv_internal(columnColorSelector, color, alphaConst, A, x, betaConst, y, flags, null_stream);
            }
        }
        else
        {
            bsrmv_internal(columnColorSelector, color, alphaConst, A, x, betaConst, y, flags, null_stream);
        }

        y.dirtybit = 1;
        //if (!A.is_matrix_singleGPU() && y.size() == x.size() && y.delayed_send==0)
        //  A.manager->exchange_halo_async(y, y.tag);
        A.setView(oldView);
    }
}

// E is a vector that represents a diagonal matrix
// operate only on rows of specified color, given by A.offsets_rows_per_color, A.sorted_rows_by_color
// y= alpha*E.x + beta*y
template< class TConfig >
void Cusparse::bsrmv( const int color,
                      const typename TConfig::VecPrec alphaConst,
                      Matrix<TConfig> &A,
                      const typename Matrix<TConfig>::MVector &E,
                      Vector<TConfig> &x,
                      const typename TConfig::VecPrec betaConst,
                      Vector<TConfig> &y,
                      ViewType view)
{
    cudaStream_t null_stream = 0;

    // If only COO, add CSR since bsrmv doesn't support COO
    if (A.hasProps(COO) && !A.hasProps(CSR))
    {
        A.addProps(CSR);
    }

    if (view != A.getViewExterior())  //This is already a view, thus do not even attempt to do latency hiding
    {
        bsrmv_internal(color, alphaConst, A, E, x, betaConst, y, view, null_stream);
    }
    else  //Try and do latency hiding
    {
        //std::ccooor << "de" << std::endl;
        //std::cerr << "not an exterior view" << std::endl;
        ViewType oldView = A.currentView();

        if (!A.is_matrix_singleGPU())
        {
            A.manager->exchange_halo_async(x, x.tag);
        }

        if (A.getViewExterior() == A.getViewInterior())
        {
            //std::cerr << "exchange_halo_wait" << std::endl;
            if (!A.is_matrix_singleGPU())
            {
                A.manager->exchange_halo_wait(x, x.tag);
            }
        }

        //std::cerr << "xxxeded" << std::endl;
        ViewType flags;
        bool latencyHiding = true;

        if (A.is_matrix_singleGPU() || (x.dirtybit == 0))
        {
            latencyHiding = false;
            A.setViewExterior();
            flags = (ViewType)(A.getViewExterior());
        }
        else
        {
            flags = (ViewType)(A.getViewInterior());
            A.setViewInterior();
        }

        if (latencyHiding)
        {
            bsrmv_internal(color, alphaConst, A, E, x, betaConst, y, flags, null_stream);

            if (!A.is_matrix_singleGPU())
            {
                A.manager->exchange_halo_wait(x, x.tag);
            }

            A.setViewExterior();
            flags = (ViewType)(~(A.getViewInterior()) & A.getViewExterior());

            if (flags != 0)
            {
                bsrmv_internal(color, alphaConst, A, E, x, betaConst, y, flags, null_stream);
            }
        }
        else
        {
            bsrmv_internal(color, alphaConst, A, E, x, betaConst, y, flags, null_stream);
        }

        y.dirtybit = 1;
        //if (!A.is_matrix_singleGPU() && y.size() == x.size() && y.delayed_send==0)
        //  A.manager->exchange_halo_async(y, y.tag);
        A.setView(oldView);
    }
}

__global__ void offset_by_col_off(int nrows, int* rows, const int* bsrRowPtr)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if(i >= nrows+1)
    {
        return;
    }

    rows[i] = bsrRowPtr[i] - bsrRowPtr[0];
}

template< class TConfig >
void Cusparse::bsrmv_internal( const typename TConfig::VecPrec alphaConst,
                               const Matrix<TConfig> &A,
                               const Vector<TConfig> &x,
                               const typename TConfig::VecPrec betaConst,
                               Vector<TConfig> &y,
                               ViewType view,
                               const cudaStream_t &stream)
{
    typedef typename TConfig::VecPrec ValueTypeB;
    int row_off, nrows, nnz;
    A.getOffsetAndSizeForView(view, &row_off, &nrows);
    A.getNnzForView(view, &nnz);

    cusparseDirection_t direction = CUSPARSE_DIRECTION_COLUMN;

    if ( A.getBlockFormat() == ROW_MAJOR )
    {
        direction = CUSPARSE_DIRECTION_ROW;
    }

    bool has_offdiag = nnz != 0;

    if (has_offdiag )
    {
        bsrmv( Cusparse::get_instance().m_handle,  direction, CUSPARSE_OPERATION_NON_TRANSPOSE,
               nrows, A.get_num_cols(), nnz, &alphaConst,
               A.cuMatDescr,
               A.values.raw(),
               A.m_seq_offsets.raw() + row_off,
               A.row_offsets.raw() + row_off, A.col_indices.raw(),
               A.get_block_dimx(),
               x.raw(), &betaConst,
               y.raw() + row_off * A.get_block_dimx(),
               stream);
    }

    if (A.hasProps(DIAG))
    {
        ValueTypeB beta;

        if (!has_offdiag)
        {
            beta = betaConst;
        }
        else
        {
            beta = types::util<ValueTypeB>::get_one();
        }

        bsrmv( Cusparse::get_instance().m_handle,  direction, CUSPARSE_OPERATION_NON_TRANSPOSE,
               nrows, A.get_num_cols(), A.get_num_rows(), &alphaConst,
               A.cuMatDescr,
               A.values.raw() + A.diagOffset()*A.get_block_size(),
               A.m_seq_offsets.raw(),
               A.m_seq_offsets.raw() + row_off, A.m_seq_offsets.raw(),
               A.get_block_dimx(),
               x.raw(), &beta,
               y.raw() + row_off * A.get_block_dimx(),
               stream);
    }
}

template< class TConfig >
void Cusparse::bsrmv_internal_with_mask( const typename TConfig::VecPrec alphaConst,
        const Matrix<TConfig> &A,
        const Vector<TConfig> &x,
        const typename TConfig::VecPrec betaConst,
        Vector<TConfig> &y,
        ViewType view,
        const cudaStream_t &stream)
{
    if (A.is_matrix_singleGPU())
    {
        FatalError("Should not be here in bsrmv_internal_with_mask", AMGX_ERR_NOT_IMPLEMENTED);
    }

    if(view != INTERIOR && view != BOUNDARY)
    {
        FatalError("Only INTERIOR and BOUNDARY views supported for bsrmv_internal_with_mask", AMGX_ERR_NOT_IMPLEMENTED);
    }

    typedef typename TConfig::VecPrec ValueType;
    cusparseDirection_t direction = CUSPARSE_DIRECTION_COLUMN;

    if ( A.getBlockFormat() == ROW_MAJOR )
    {
        direction = CUSPARSE_DIRECTION_ROW;
    }

    const int *start_offsets, *end_offsets;
    start_offsets = A.row_offsets.raw();
    end_offsets = A.row_offsets.raw() + 1;
    typedef typename Matrix<TConfig>::index_type index_type;

    int offset, nrows, nnz;
    A.getFixedSizesForView(view, &offset, &nrows, &nnz);

    if (nrows <= 0)
    {
        return;    // nothing to do, early exit
    }

    bool has_offdiag = nnz != 0;

    if (has_offdiag)
    {
        cusparseSetStream(Cusparse::get_instance().m_handle, stream);
        bsrxmv_internal( Cusparse::get_instance().m_handle, direction, CUSPARSE_OPERATION_NON_TRANSPOSE, nrows,
                         nrows, A.get_num_cols(), nnz, &alphaConst,
                         A.cuMatDescr,
                         A.values.raw(),
                         A.manager->getRowsListForView(view).raw(),
                         start_offsets, end_offsets, A.col_indices.raw(),
                         A.get_block_dimx(),
                         x.raw(), &betaConst,
                         y.raw() );

        // Reset to default stream
        cusparseSetStream(Cusparse::get_instance().m_handle, 0);
    }

    if (A.hasProps(DIAG))
    {
        FatalError("Diag not supported in multiply with mask\n", AMGX_ERR_NOT_IMPLEMENTED);
    }
}


template< class TConfig >
void Cusparse::bsrmv_internal_with_mask_restriction( const typename TConfig::VecPrec alphaConst,
        const Matrix<TConfig> &R,
        const Vector<TConfig> &x,
        const typename TConfig::VecPrec betaConst,
        Vector<TConfig> &y,
        ViewType view,
        const cudaStream_t &stream,
        Matrix<TConfig> &P)
{
    if (P.is_matrix_singleGPU())
    {
        FatalError("Should not be here in bsrmv_internal_with_mask_with_restriction", AMGX_ERR_NOT_IMPLEMENTED);
    }

    if(view != OWNED && view != HALO1)
    {
        FatalError("View not supported in restriction operation", AMGX_ERR_NOT_IMPLEMENTED);
    }

    typedef typename TConfig::VecPrec ValueType;
    cusparseDirection_t direction = CUSPARSE_DIRECTION_COLUMN;

    if ( R.getBlockFormat() == ROW_MAJOR )
    {
        direction = CUSPARSE_DIRECTION_ROW;
    }

    int row_off, nrows, nnz;
    R.getFixedSizesForView(view, &row_off, &nrows, &nnz);

    bool has_offdiag = nnz != 0;
    typedef typename Matrix<TConfig>::index_type index_type;

    if (nrows <= 0)
    {
        return;    // nothing to do, early exit
    }

    if (has_offdiag)
    {
        bsrmv( Cusparse::get_instance().m_handle,  direction, CUSPARSE_OPERATION_NON_TRANSPOSE,
               nrows, R.get_num_cols(), nnz, &alphaConst,
               R.cuMatDescr,
               R.values.raw(),
               R.m_seq_offsets.raw() + row_off,
               R.row_offsets.raw() + row_off, R.col_indices.raw(),
               R.get_block_dimx(),
               x.raw(), &betaConst,
               y.raw() + row_off * R.get_block_dimx(),
               stream);
    }

    if (R.hasProps(DIAG))
    {
        FatalError("Diag not supported in multiply with mask\n", AMGX_ERR_NOT_IMPLEMENTED);
    }
}



template< class TConfig >
void Cusparse::bsrmv_internal( const typename TConfig::VecPrec alphaConst,
                               const Matrix<TConfig> &A,
                               const typename Matrix<TConfig>::MVector &E,
                               const Vector<TConfig> &x,
                               const typename TConfig::VecPrec betaConst,
                               Vector<TConfig> &y,
                               ViewType view,
                               const cudaStream_t &stream)
{
    typedef typename TConfig::VecPrec ValueType;
    int row_off, nrows, nnz;
    A.getFixedSizesForView(view, &row_off, &nrows, &nnz);
    cusparseDirection_t direction = A.getBlockFormat() == ROW_MAJOR ? CUSPARSE_DIRECTION_ROW : CUSPARSE_DIRECTION_COLUMN;

    bsrmv( Cusparse::get_instance().m_handle, direction, CUSPARSE_OPERATION_NON_TRANSPOSE,
           nrows, A.get_num_cols(), nnz, &alphaConst,
           A.cuMatDescr,
           E.raw(),
           A.m_seq_offsets.raw(),
           A.m_seq_offsets.raw() + row_off, A.m_seq_offsets.raw(),
           A.get_block_dimx(),
           x.raw(), &betaConst,
           y.raw() + row_off * A.get_block_dimx(),
           stream);
}


template< class TConfig >
void Cusparse::bsrmv_internal( ColumnColorSelector columnColorSelector,
                               const int color,
                               const typename TConfig::VecPrec alphaConst,
                               const Matrix<TConfig> &A,
                               const Vector<TConfig> &x,
                               const typename TConfig::VecPrec betaConst,
                               Vector<TConfig> &y,
                               ViewType view,
                               const cudaStream_t &stream)
{
    typedef typename TConfig::VecPrec ValueType;

    if (!A.hasProps(COLORING))
    {
        FatalError("Matrix is not colored, exiting", AMGX_ERR_BAD_PARAMETERS);
    }

    if (color < 0 || color >= A.getMatrixColoring().getNumColors())
    {
        FatalError("Unknown color", AMGX_ERR_BAD_PARAMETERS);
    }

    typedef typename Matrix<TConfig>::index_type index_type;
    index_type colorStart = 0;

    if ( !(view & INTERIOR) )
    {
        colorStart = A.getMatrixColoring().getSeparationOffsetsRowsPerColor()[color];
    }
    else
    {
        colorStart = A.getMatrixColoring().getOffsetsRowsPerColor()[color];
    }

    index_type colorNum = 0;

    if ( view == A.getViewInterior() )
    {
        colorNum = A.getMatrixColoring().getSeparationOffsetsRowsPerColor()[color];
    }
    else
    {
        colorNum = A.getMatrixColoring().getOffsetsRowsPerColor()[color + 1];
    }

    colorNum -= colorStart;

    if (colorNum <= 0)
    {
        return;    // nothing to do, early exit
    }

    if (columnColorSelector == DIAG_COL)
    {
        FatalError("DIAG_COL has been disabled to avoid allocating diag_offsets", AMGX_ERR_NOT_IMPLEMENTED);
    }

    const int *start_offsets, *end_offsets;

    switch (columnColorSelector)
    {
        case ALL_COLORS:
            start_offsets = A.row_offsets.raw();
            end_offsets = A.row_offsets.raw() + 1;
            break;

        case SMALLER_COLORS:
            start_offsets = A.row_offsets.raw();
            end_offsets = A.m_smaller_color_offsets.raw();
            break;

        case LARGER_COLORS:
            start_offsets = A.m_larger_color_offsets.raw();
            end_offsets = A.row_offsets.raw() + 1;
            break;

        case DIAG_COL:
            start_offsets = A.diag.raw();
            end_offsets = A.m_diag_end_offsets.raw();
            break;

        default:
            FatalError("Unknown color selector", AMGX_ERR_CORE);
    }

    cusparseDirection_t direction = CUSPARSE_DIRECTION_COLUMN;

    if ( A.getBlockFormat() == ROW_MAJOR )
    {
        direction = CUSPARSE_DIRECTION_ROW;
    }

    bool has_offdiag = A.get_num_nz() != 0;
    cusparseSetStream(Cusparse::get_instance().m_handle, stream);

    if (has_offdiag)
    {
        bsrxmv_internal( Cusparse::get_instance().m_handle, direction, CUSPARSE_OPERATION_NON_TRANSPOSE, colorNum,
                         A.get_num_rows(), A.get_num_cols(), A.get_num_nz(), &alphaConst,
                         A.cuMatDescr,
                         A.values.raw(),
                         A.getMatrixColoring().getSortedRowsByColor().raw() + colorStart,
                         start_offsets, end_offsets, A.col_indices.raw(),
                         A.get_block_dimx(),
                         x.raw(), &betaConst,
                         y.raw() );
    }

    if (A.hasProps(DIAG) && columnColorSelector == ALL_COLORS)
    {
        typename TConfig::VecPrec beta = betaConst;

        if (has_offdiag)
        {
            beta = types::util<typename TConfig::VecPrec>::get_one();
        }

        bsrxmv_internal( Cusparse::get_instance().m_handle, direction, CUSPARSE_OPERATION_NON_TRANSPOSE, colorNum,
                         A.get_num_rows(), A.get_num_cols(), A.get_num_rows(), &alphaConst,
                         A.cuMatDescr,
                         A.values.raw() + A.diagOffset()*A.get_block_size(),
                         A.getMatrixColoring().getSortedRowsByColor().raw() + colorStart,
                         A.m_seq_offsets.raw(),
                         NULL,
                         A.m_seq_offsets.raw(),
                         A.get_block_dimx(),
                         x.raw(), &beta,
                         y.raw() );
    }

    // Reset to default stream
    cusparseSetStream(Cusparse::get_instance().m_handle, 0);
}

template< class TConfig >
void Cusparse::bsrmv_internal( const int color,
                               const typename TConfig::VecPrec alphaConst,
                               const Matrix<TConfig> &A,
                               const typename Matrix<TConfig>::MVector &E,
                               const Vector<TConfig> &x,
                               const typename TConfig::VecPrec betaConst,
                               Vector<TConfig> &y,
                               ViewType view,
                               const cudaStream_t &stream)
{
    typedef typename TConfig::VecPrec ValueType;

    if ( !A.hasProps(COLORING) )
    {
        FatalError("Matrix is not colored, exiting", AMGX_ERR_BAD_PARAMETERS);
    }

    if ( color < 0 || color >= A.getMatrixColoring().getNumColors() )
    {
        FatalError("Unknown color", AMGX_ERR_BAD_PARAMETERS);
    }

    typedef typename Matrix<TConfig>::index_type index_type;
    //const index_type colorStart = ((view & INTERIOR) == 0) ? A.getMatrixColoring().getSeparationOffsetsRowsPerColor()[color] : A.getMatrixColoring().getOffsetsRowsPerColor()[color];
    //const index_type colorNum = ((view == A.getViewInterior()) ? A.getMatrixColoring().getSeparationOffsetsRowsPerColor()[color] : A.getMatrixColoring().getOffsetsRowsPerColor()[color+1]) - colorStart;
    //const index_type colorStart= A.getMatrixColoring().getOffsetsRowsPerColor()[color];
    //const index_type colorNum= A.getMatrixColoring().getOffsetsRowsPerColor()[color+1] - colorStart;
    index_type colorStart = 0;

    if ( !(view & INTERIOR) )
    {
        colorStart = A.getMatrixColoring().getSeparationOffsetsRowsPerColor()[color];
    }
    else
    {
        colorStart = A.getMatrixColoring().getOffsetsRowsPerColor()[color];
    }

    index_type colorNum = 0;

    if ( view == A.getViewInterior() )
    {
        colorNum = A.getMatrixColoring().getSeparationOffsetsRowsPerColor()[color];
    }
    else
    {
        colorNum = A.getMatrixColoring().getOffsetsRowsPerColor()[color + 1];
    }

    colorNum -= colorStart;

    if (colorNum <= 0)
    {
        return;    // nothing to do, early exit
    }

    cusparseDirection_t direction = CUSPARSE_DIRECTION_COLUMN;

    if ( A.getBlockFormat() == ROW_MAJOR )
    {
        direction = CUSPARSE_DIRECTION_ROW;
    }

    cusparseSetStream(Cusparse::get_instance().m_handle, stream);

    bsrxmv_internal( Cusparse::get_instance().m_handle, direction, CUSPARSE_OPERATION_NON_TRANSPOSE, colorNum,
                     A.get_num_rows(), A.get_num_cols(), A.get_num_nz(), &alphaConst,
                     A.cuMatDescr,
                     E.raw(),
                     A.getMatrixColoring().getSortedRowsByColor().raw() + colorStart,
                     A.m_seq_offsets.raw(),
                     NULL,
                     A.m_seq_offsets.raw(),
                     A.get_block_dimx(),
                     x.raw(), &betaConst,
                     y.raw() );
    // Reset to default stream
    cusparseSetStream(Cusparse::get_instance().m_handle, 0);
}

#ifdef CUSPARSE_GENERIC_INTERFACES
template<class MatType, class VecType, class IndType>
inline void generic_SpMV(cusparseHandle_t handle, cusparseOperation_t trans,
                             int mb, int nb, int nnzb,
                             const MatType *alpha,
                             const MatType *val,
                             const IndType *rowPtr,
                             const IndType *colInd,
                             const VecType *x,
                             const VecType *beta,
                             VecType *y,
                             cudaDataType matType,
                             cudaDataType vecType,
                             const cudaStream_t& stream)
{
    int col_off;
    cudaMemcpyAsync(&col_off, &rowPtr[0], sizeof(int), cudaMemcpyDefault, stream);
    cudaStreamSynchronize(stream);

    IndType* rows = const_cast<IndType*>(rowPtr);
    IndType* cols = const_cast<IndType*>(colInd) + col_off;
    MatType* vals = const_cast<MatType*>(val) + col_off;

    if(col_off > 0)
    {
        amgx::memory::cudaMalloc((void**)&rows, sizeof(IndType)*(mb+1));

        constexpr int nthreads = 128;
        const int nblocks = (mb + 1) / nthreads + 1;
        offset_by_col_off<<<nblocks, nthreads, 0, stream>>>(mb, rows, rowPtr);
    }

    cusparseSpMatDescr_t matA_descr;
    cusparseDnVecDescr_t vecX_descr;
    cusparseDnVecDescr_t vecY_descr;
    cusparseCheckError(cusparseCreateDnVec(&vecX_descr, nb, const_cast<VecType*>(x), vecType));
    cusparseCheckError(cusparseCreateDnVec(&vecY_descr, mb, const_cast<VecType*>(y), vecType));
    cusparseCheckError(
            cusparseCreateCsr(&matA_descr, mb, nb, nnzb, const_cast<IndType*>(rows), const_cast<IndType*>(cols),
                          const_cast<MatType*>(vals), CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO, matType));

    size_t bufferSize = 0;
    cusparseCheckError(cusparseSpMV_bufferSize(handle, trans, alpha, matA_descr, vecX_descr, beta, vecY_descr, matType, CUSPARSE_CSRMV_ALG2, &bufferSize));

    void* dBuffer = NULL;
    if(bufferSize > 0)
    {
        amgx::memory::cudaMalloc(&dBuffer, bufferSize);
    }

    cusparseCheckError(cusparseSpMV(handle, trans, alpha, matA_descr, vecX_descr, beta, vecY_descr, matType, CUSPARSE_CSRMV_ALG2, dBuffer) );

    cusparseCheckError(cusparseDestroySpMat(matA_descr));
    cusparseCheckError(cusparseDestroyDnVec(vecX_descr));
    cusparseCheckError(cusparseDestroyDnVec(vecY_descr));

    if(bufferSize > 0)
    {
        amgx::memory::cudaFreeAsync(dBuffer);
    }

    if(col_off > 0)
    {
        amgx::memory::cudaFreeAsync(rows);
    }
}
#endif

inline void Cusparse::bsrmv( cusparseHandle_t handle, cusparseDirection_t dir, cusparseOperation_t trans,
                             int mb, int nb, int nnzb,
                             const float *alpha,
                             const cusparseMatDescr_t descr,
                             const float *bsrVal,
                             const int *bsrMaskPtr,
                             const int *bsrRowPtr,
                             const int *bsrColInd,
                             int blockDim,
                             const float *x,
                             const float *beta,
                             float *y,
                             const cudaStream_t& stream)
{
    // Run cuSparse on selected stream
    cusparseSetStream(handle, stream);

    if (blockDim == 1)
    {
        #ifdef CUSPARSE_GENERIC_INTERFACES
            generic_SpMV(handle, trans, mb, nb, nnzb, alpha, bsrVal, bsrRowPtr, bsrColInd, x, beta, y, CUDA_R_32F, CUDA_R_32F, stream);
        #else
            cusparseCheckError(cusparseScsrmv(handle, trans, mb, nb, nnzb, alpha, descr, bsrVal, bsrRowPtr, bsrColInd, x, beta, y));
        #endif
    }
    else
    {
        cusparseCheckError(cusparseSbsrmv(handle, dir, trans, mb, nb, nnzb, alpha, descr, bsrVal, bsrRowPtr, bsrColInd, blockDim, x, beta, y));
    }

    // Reset cuSparse to default stream
    cusparseSetStream(handle, 0);
}

inline void Cusparse::bsrmv( cusparseHandle_t handle, cusparseDirection_t dir, cusparseOperation_t trans,
                             int mb, int nb, int nnzb,
                             const double *alpha,
                             const cusparseMatDescr_t descr,
                             const double *bsrVal,
                             const int *bsrMaskPtr,
                             const int *bsrRowPtr,
                             const int *bsrColInd,
                             int blockDim,
                             const double *x,
                             const double *beta,
                             double *y,
                             const cudaStream_t& stream)
{
    // Run cuSparse on selected stream
    cusparseSetStream(handle, stream);

    if (blockDim == 1)
    {
        #ifdef CUSPARSE_GENERIC_INTERFACES
            generic_SpMV(handle, trans, mb, nb, nnzb, alpha, bsrVal, bsrRowPtr, bsrColInd, x, beta, y, CUDA_R_64F, CUDA_R_64F, stream);

        #else
            cusparseCheckError(cusparseDcsrmv(handle, trans, mb, nb, nnzb, alpha, descr, bsrVal, bsrRowPtr, bsrColInd, x, beta, y));
        #endif
    }
    else
    {
        cusparseCheckError(cusparseDbsrmv(handle, dir, trans, mb, nb, nnzb, alpha, descr, bsrVal, bsrRowPtr, bsrColInd, blockDim, x, beta, y));
    }

    // Reset cuSparse to default stream
    cusparseSetStream(handle, 0);
}

inline void Cusparse::bsrmv( cusparseHandle_t handle, cusparseDirection_t dir, cusparseOperation_t trans,
                             int mb, int nb, int nnzb,
                             const double *alpha,
                             const cusparseMatDescr_t descr,
                             const float *bsrVal,
                             const int *bsrMaskPtr,
                             const int *bsrRowPtr,
                             const int *bsrColInd,
                             int blockDim,
                             const double *x,
                             const double *beta,
                             double *y,
                             const cudaStream_t& stream)
{
    // Run cuSparse on selected stream
    cusparseSetStream(handle, stream);

    #ifndef DISABLE_MIXED_PRECISION
        const double *d_bsrVal = reinterpret_cast<const double *>(const_cast<float *>(bsrVal)); // this works due to private API call in the matrix initialization which sets cusparse matrix description in the half precision mode
        cusparseCheckError(cusparseDbsrxmv(handle, dir, trans, mb, mb, nb, nnzb, alpha, descr, d_bsrVal, bsrMaskPtr, bsrRowPtr, bsrRowPtr + 1, bsrColInd, blockDim, x, beta, y));
    #else
        FatalError("Mixed precision modes not currently supported for CUDA 10.1 or later.", AMGX_ERR_NOT_IMPLEMENTED);
    #endif

    // Reset cuSparse to default stream
    cusparseSetStream(handle, 0);
}

// Custom implementation of matrix-vector product to replace the original bsrxmv,
// but with block size of 1.
template<unsigned UNROLL, class T>
__global__ void csrxmv(
    int sizeOfMask,
    const T alpha,
    const T* __restrict__ csrVal,
    const int* __restrict__ csrMask,
    const int* __restrict__ csrRow,
    const int* __restrict__ csrCol,
    const T* __restrict__ x,
    const T beta,
    T* __restrict__ y)
{
    for(int i = threadIdx.x + blockIdx.x*blockDim.x; i < sizeOfMask; i += blockDim.x*gridDim.x)
    {
        int row = csrMask[i];
        T y_tmp = amgx::types::util<T>::get_zero();

        int row_b = csrRow[row];
        int row_e = csrRow[row+1];

        // Unrolling is important for performance here.
        // Possible to squeeze more performance out of the key kernels if we
        // measure the sparsity and use it to inform unrolling.
        for (int col = row_b; col < row_e; col += UNROLL)
        {
#pragma unroll UNROLL
            for(int off = 0; off < UNROLL; ++off)
            {
                int c = col + off;
                if(c < row_e) y_tmp = alpha * csrVal[c] * x[csrCol[c]] + y_tmp;
            }
        }

        // Don't read y unnecessarily
        if(amgx::types::util<T>::is_zero(beta))
        {
            y[row] = y_tmp;
        }
        else
        {
            y[row] = beta*y[row] + y_tmp;
        }
    }
}

// Replaces the functionality of cusparse?bsrxmv for blockDim == 1
template<class T>
inline void Xcsrxmv( cusparseHandle_t handle, cusparseDirection_t dir, cusparseOperation_t trans, int sizeOfMask,
                                       int mb, int nb, int nnzb,
                                       const T *alpha,
                                       const cusparseMatDescr_t descr,
                                       const T *bsrVal,
                                       const int *bsrMaskPtr,
                                       const int *bsrRowPtr,
                                       const int *bsrEndPtr,
                                       const int *bsrColInd,
                                       int blockDim,
                                       const T *x,
                                       const T *beta,
                                       T *y)
{
    if(blockDim != 1)
    {
        FatalError("Xcsrxmv only to be called with scalar matrices.", AMGX_ERR_INTERNAL);
    }
    if (trans != CUSPARSE_OPERATION_NON_TRANSPOSE)
    {
        FatalError("Cannot currently latency hide a transposed matrix.", AMGX_ERR_NOT_IMPLEMENTED);
    }
    if (dir != CUSPARSE_DIRECTION_ROW)
    {
        FatalError("Cannot currently latency hide if matrix is not row major.", AMGX_ERR_NOT_IMPLEMENTED);
    }

    constexpr int nthreads = 128;
    constexpr int unroll_factor = 16;
    int nblocks = sizeOfMask / nthreads + 1;
    csrxmv<unroll_factor><<<nblocks, nthreads>>>(sizeOfMask, *alpha, bsrVal, bsrMaskPtr, bsrRowPtr, bsrColInd, x, *beta, y);
}

// overloaded C++ wrappers for cusparse?bsrxmv
// bsrxmv
// matrix - float
// vector - float
inline void Cusparse::bsrxmv_internal(cusparseHandle_t handle, cusparseDirection_t dir, cusparseOperation_t trans, int sizeOfMask,
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
                                      float *y)
{
    if (bsrEndPtr == NULL && bsrMaskPtr == NULL)
    {
        cusparseCheckError(CUSPARSE_STATUS_MATRIX_TYPE_NOT_SUPPORTED);
        //      bsrmv(handle, dir, trans, mb, nb, nnzb, alpha, descr, bsrVal, bsrRowPtr, bsrColInd, blockDim, x, beta, y);
    }
    else
    {
        if (bsrEndPtr == NULL)
        {
            bsrEndPtr = bsrRowPtr + 1;
        }

        if (blockDim == 1)
        {
            Xcsrxmv(handle, dir, trans, sizeOfMask, mb, nb, nnzb, alpha, descr, bsrVal, bsrMaskPtr, bsrRowPtr, bsrEndPtr, bsrColInd, blockDim, x, beta, y);
        }
        else
        {
            cusparseCheckError(cusparseSbsrxmv(handle, dir, trans, sizeOfMask, mb, nb, nnzb, alpha, descr, bsrVal, bsrMaskPtr, bsrRowPtr, bsrEndPtr, bsrColInd, blockDim, x, beta, y));
        }
    }
}

// bsrxmv
// matrix - float
// vector - double
inline void Cusparse::bsrxmv_internal( cusparseHandle_t handle, cusparseDirection_t dir, cusparseOperation_t trans, int sizeOfMask,
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
                                       double *y)
{
    cusparseCheckError(CUSPARSE_STATUS_MATRIX_TYPE_NOT_SUPPORTED);
}

// bsrxmv
// matrix - double
// vector - double
inline void Cusparse::bsrxmv_internal( cusparseHandle_t handle, cusparseDirection_t dir, cusparseOperation_t trans, int sizeOfMask,
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
                                       double *y)
{
    if (bsrEndPtr == NULL && bsrMaskPtr == NULL)
    {
        cusparseCheckError(CUSPARSE_STATUS_MATRIX_TYPE_NOT_SUPPORTED);
    }
    else
    {
        if (bsrEndPtr == NULL)
        {
            bsrEndPtr = bsrRowPtr + 1;
        }

        if (blockDim == 1)
        {
            Xcsrxmv(handle, dir, trans, sizeOfMask, mb, nb, nnzb, alpha, descr, bsrVal, bsrMaskPtr, bsrRowPtr, bsrEndPtr, bsrColInd, blockDim, x, beta, y);
        }
        else
        {
            cusparseCheckError(cusparseDbsrxmv(handle, dir, trans, sizeOfMask, mb, nb, nnzb, alpha, descr, bsrVal, bsrMaskPtr, bsrRowPtr, bsrEndPtr, bsrColInd, blockDim, x, beta, y));
        }
    }
}



inline void Cusparse::bsrmv( cusparseHandle_t handle, cusparseDirection_t dir, cusparseOperation_t trans,
                             int mb, int nb, int nnzb,
                             const cuComplex *alpha,
                             const cusparseMatDescr_t descr,
                             const cuComplex *bsrVal,
                             const int *bsrMaskPtr,
                             const int *bsrRowPtr,
                             const int *bsrColInd,
                             int blockDim,
                             const cuComplex *x,
                             const cuComplex *beta,
                             cuComplex *y,
                             const cudaStream_t& stream)
{
    // Run cuSparse on selected stream
    cusparseSetStream(handle, stream);

    if (blockDim == 1)
    {
        #ifdef CUSPARSE_GENERIC_INTERFACES
            generic_SpMV(handle, trans, mb, nb, nnzb, alpha, bsrVal, bsrRowPtr, bsrColInd, x, beta, y, CUDA_C_32F, CUDA_C_32F, stream);
        #else
            cusparseCheckError(cusparseCcsrmv(handle, trans, mb, nb, nnzb, alpha, descr, bsrVal, bsrRowPtr, bsrColInd, x, beta, y));
        #endif
    }
    else
    {
        cusparseCheckError(cusparseCbsrmv(handle, dir, trans, mb, nb, nnzb, alpha, descr, bsrVal, bsrRowPtr, bsrColInd, blockDim, x, beta, y));
    }

    // Reset cuSparse to default stream
    cusparseSetStream(handle, 0);
}

inline void Cusparse::bsrmv( cusparseHandle_t handle, cusparseDirection_t dir, cusparseOperation_t trans,
                             int mb, int nb, int nnzb,
                             const cuDoubleComplex *alpha,
                             const cusparseMatDescr_t descr,
                             const cuDoubleComplex *bsrVal,
                             const int *bsrMaskPtr,
                             const int *bsrRowPtr,
                             const int *bsrColInd,
                             int blockDim,
                             const cuDoubleComplex *x,
                             const cuDoubleComplex *beta,
                             cuDoubleComplex *y,
                             const cudaStream_t& stream)
{
    // Run cuSparse on selected stream
    cusparseSetStream(handle, stream);

    if (blockDim == 1)
    {
        #ifdef CUSPARSE_GENERIC_INTERFACES
            generic_SpMV(handle, trans, mb, nb, nnzb, alpha, bsrVal, bsrRowPtr, bsrColInd, x, beta, y, CUDA_C_64F, CUDA_C_64F, stream);
        #else
            cusparseCheckError(cusparseZcsrmv(handle, trans, mb, nb, nnzb, alpha, descr, bsrVal, bsrRowPtr, bsrColInd, x, beta, y));
        #endif
    }
    else
    {
        cusparseCheckError(cusparseZbsrmv(handle, dir, trans, mb, nb, nnzb, alpha, descr, bsrVal, bsrRowPtr, bsrColInd, blockDim, x, beta, y));
    }

    // Reset cuSparse to default stream
    cusparseSetStream(handle, 0);
}

inline void Cusparse::bsrmv( cusparseHandle_t handle, cusparseDirection_t dir, cusparseOperation_t trans,
                             int mb, int nb, int nnzb,
                             const cuDoubleComplex *alpha,
                             const cusparseMatDescr_t descr,
                             const cuComplex *bsrVal,
                             const int *bsrMaskPtr,
                             const int *bsrRowPtr,
                             const int *bsrColInd,
                             int blockDim,
                             const cuDoubleComplex *x,
                             const cuDoubleComplex *beta,
                             cuDoubleComplex *y,
                             const cudaStream_t& stream)
{
    // Run cuSparse on selected stream
    cusparseSetStream(handle, stream);

    #ifndef DISABLE_MIXED_PRECISION
        const cuDoubleComplex *d_bsrVal = reinterpret_cast<cuDoubleComplex *>(const_cast<cuComplex *>(bsrVal));
        cusparseCheckError(cusparseZbsrxmv(handle, dir, trans, mb, mb, nb, nnzb, alpha, descr, d_bsrVal, bsrMaskPtr, bsrRowPtr, bsrRowPtr + 1, bsrColInd, blockDim, x, beta, y));
    #else
        FatalError("Mixed precision modes not currently supported for CUDA 10.1 or later.", AMGX_ERR_NOT_IMPLEMENTED);
    #endif

    // Reset cuSparse to default stream
    cusparseSetStream(handle, 0);
}


// overloaded C++ wrappers for cusparse?bsrxmv
// bsrxmv
// matrix - cuComplex
// vector - cuComplex
inline void Cusparse::bsrxmv_internal( cusparseHandle_t handle, cusparseDirection_t dir, cusparseOperation_t trans, int sizeOfMask,
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
                                       cuComplex *y)
{
    if (bsrEndPtr == NULL && bsrMaskPtr == NULL)
    {
        cusparseCheckError(CUSPARSE_STATUS_MATRIX_TYPE_NOT_SUPPORTED);
        //      bsrmv(handle, dir, trans, mb, nb, nnzb, alpha, descr, bsrVal, bsrRowPtr, bsrColInd, blockDim, x, beta, y);
    }
    else
    {
        if (bsrEndPtr == NULL)
        {
            bsrEndPtr = bsrRowPtr + 1;
        }

        if (blockDim == 1)
        {
            Xcsrxmv(handle, dir, trans, sizeOfMask, mb, nb, nnzb, alpha, descr, bsrVal, bsrMaskPtr, bsrRowPtr, bsrEndPtr, bsrColInd, blockDim, x, beta, y);
        }
        else
        {
            cusparseCheckError(cusparseCbsrxmv(handle, dir, trans, sizeOfMask, mb, nb, nnzb, alpha, descr, bsrVal, bsrMaskPtr, bsrRowPtr, bsrEndPtr, bsrColInd, blockDim, x, beta, y));
        }
    }
}

// bsrxmv
// matrix - cuComplex
// vector - cuDoubleComplex
inline void Cusparse::bsrxmv_internal( cusparseHandle_t handle, cusparseDirection_t dir, cusparseOperation_t trans, int sizeOfMask,
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
                                       cuDoubleComplex *y)
{
    cusparseCheckError(CUSPARSE_STATUS_MATRIX_TYPE_NOT_SUPPORTED);
}

// bsrxmv
// matrix - cuDoubleComplex
// vector - cuDoubleComplex
inline void Cusparse::bsrxmv_internal( cusparseHandle_t handle, cusparseDirection_t dir, cusparseOperation_t trans, int sizeOfMask,
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
                                       cuDoubleComplex *y)
{
    if (bsrEndPtr == NULL && bsrMaskPtr == NULL)
    {
        cusparseCheckError(CUSPARSE_STATUS_MATRIX_TYPE_NOT_SUPPORTED);
    }
    else
    {
        if (bsrEndPtr == NULL)
        {
            bsrEndPtr = bsrRowPtr + 1;
        }

        if (blockDim == 1)
        {
            Xcsrxmv(handle, dir, trans, sizeOfMask, mb, nb, nnzb, alpha, descr, bsrVal, bsrMaskPtr, bsrRowPtr, bsrEndPtr, bsrColInd, blockDim, x, beta, y);
        }
        else
        {
            cusparseCheckError(cusparseZbsrxmv(handle, dir, trans, sizeOfMask, mb, nb, nnzb, alpha, descr, bsrVal, bsrMaskPtr, bsrRowPtr, bsrEndPtr, bsrColInd, blockDim, x, beta, y));
        }
    }
}


namespace
{
#ifdef CUSPARSE_GENERIC_INTERFACES
template<class MatType, class IndType>
inline void
generic_SpMM(cusparseHandle_t handle, cusparseOperation_t transA,
             int m, int n, int k, int nnz,
             int ldb, int ldc,
             const MatType *alpha,
             const MatType *Avals,
             const MatType *Bvals,
             MatType *Cvals,
             const IndType *rowPtr,
             const IndType *colInd,
             const MatType *beta,
             cudaDataType matType)
{
    // Create the matrix descriptors
    cusparseSpMatDescr_t matA_descr;
    cusparseDnMatDescr_t matB_descr;
    cusparseDnMatDescr_t matC_descr;
    cusparseCheckError(
        cusparseCreateCsr(&matA_descr, m, k, nnz, const_cast<IndType*>(rowPtr), const_cast<IndType*>(colInd),
                          const_cast<MatType*>(Avals), CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO, matType));
    cusparseCheckError(
        cusparseCreateDnMat(&matB_descr, k, n, ldb, const_cast<MatType*>(Bvals), matType, CUSPARSE_ORDER_COL));
    cusparseCheckError(
        cusparseCreateDnMat(&matC_descr, m, n, ldc, const_cast<MatType*>(Cvals), matType, CUSPARSE_ORDER_COL));

    // Check if a buffer is required, and if so allocate it using caching allocator
    size_t bufferSize = 0;
    cusparseCheckError(
        cusparseSpMM_bufferSize(handle, transA, CUSPARSE_OPERATION_NON_TRANSPOSE, alpha, matA_descr, matB_descr,
                                beta, matC_descr, matType, CUSPARSE_SPMM_ALG_DEFAULT, &bufferSize));

    void* dBuffer = NULL;
    if(bufferSize > 0)
    {
        amgx::memory::cudaMalloc(&dBuffer, bufferSize);
    }

    // Compute the sparse matrix - dense matrix product
    cusparseCheckError(
        cusparseSpMM(handle, transA, CUSPARSE_OPERATION_NON_TRANSPOSE, alpha, matA_descr, matB_descr, beta,
                     matC_descr, matType, CUSPARSE_SPMM_ALG_DEFAULT, dBuffer));

    // Clean up
    cusparseCheckError(cusparseDestroySpMat(matA_descr));
    cusparseCheckError(cusparseDestroyDnMat(matB_descr));
    cusparseCheckError(cusparseDestroyDnMat(matC_descr));

    if(bufferSize > 0)
    {
        amgx::memory::cudaFreeAsync(dBuffer);
    }
}
#endif

void
cusparse_csrmm(cusparseHandle_t handle, cusparseOperation_t transA,
               int m, int n, int k, int nnz,
               const float           *alpha,
               const cusparseMatDescr_t descrA,
               const float            *csrValA,
               const int *csrRowPtrA, const int *csrColIndA,
               const float            *B, int ldb,
               const float            *beta, float          *C, int ldc)
{
    #ifdef CUSPARSE_GENERIC_INTERFACES
        generic_SpMM(handle, transA, m, n, k, nnz, ldb, ldc, alpha, csrValA, B, C, csrRowPtrA, csrColIndA, beta, CUDA_R_32F);
    #else
        cusparseCheckError(cusparseScsrmm(handle, transA, m, n, k, nnz, alpha, descrA, csrValA, csrRowPtrA, csrColIndA, B, ldb, beta, C, ldc));
    #endif
}

void
cusparse_csrmm(cusparseHandle_t handle, cusparseOperation_t transA,
               int m, int n, int k, int nnz,
               const double            *alpha,
               const cusparseMatDescr_t descrA,
               const float            *csrValA,
               const int *csrRowPtrA, const int *csrColIndA,
               const double            *B, int ldb,
               const double           *beta, double          *C, int ldc)
{
    cusparseCheckError(CUSPARSE_STATUS_MATRIX_TYPE_NOT_SUPPORTED);
}

void
cusparse_csrmm(cusparseHandle_t handle, cusparseOperation_t transA,
               int m, int n, int k, int nnz,
               const double          *alpha,
               const cusparseMatDescr_t descrA,
               const double          *csrValA,
               const int *csrRowPtrA, const int *csrColIndA,
               const double           *B, int ldb,
               const double           *beta, double         *C, int ldc)
{
    #ifdef CUSPARSE_GENERIC_INTERFACES
        generic_SpMM(handle, transA, m, n, k, nnz, ldb, ldc, alpha, csrValA, B, C, csrRowPtrA, csrColIndA, beta, CUDA_R_64F);
    #else
        cusparseCheckError(cusparseDcsrmm(handle, transA, m, n, k, nnz, alpha, descrA, csrValA, csrRowPtrA, csrColIndA, B, ldb, beta, C, ldc));
    #endif
}

void
cusparse_csrmm(cusparseHandle_t handle, cusparseOperation_t transA,
               int m, int n, int k, int nnz,
               const cuComplex           *alpha,
               const cusparseMatDescr_t descrA,
               const cuComplex            *csrValA,
               const int *csrRowPtrA, const int *csrColIndA,
               const cuComplex            *B, int ldb,
               const cuComplex            *beta, cuComplex          *C, int ldc)
{
    #ifdef CUSPARSE_GENERIC_INTERFACES
        generic_SpMM(handle, transA, m, n, k, nnz, ldb, ldc, alpha, csrValA, B, C, csrRowPtrA, csrColIndA, beta, CUDA_C_32F);
    #else
        cusparseCheckError(cusparseCcsrmm(handle, transA, m, n, k, nnz, alpha, descrA, csrValA, csrRowPtrA, csrColIndA, B, ldb, beta, C, ldc));
    #endif
}

void
cusparse_csrmm(cusparseHandle_t handle, cusparseOperation_t transA,
               int m, int n, int k, int nnz,
               const cuDoubleComplex            *alpha,
               const cusparseMatDescr_t descrA,
               const cuComplex            *csrValA,
               const int *csrRowPtrA, const int *csrColIndA,
               const cuDoubleComplex            *B, int ldb,
               const cuDoubleComplex           *beta, cuDoubleComplex          *C, int ldc)
{
    cusparseCheckError(CUSPARSE_STATUS_MATRIX_TYPE_NOT_SUPPORTED);
}

void
cusparse_csrmm(cusparseHandle_t handle, cusparseOperation_t transA,
               int m, int n, int k, int nnz,
               const cuDoubleComplex          *alpha,
               const cusparseMatDescr_t descrA,
               const cuDoubleComplex          *csrValA,
               const int *csrRowPtrA, const int *csrColIndA,
               const cuDoubleComplex           *B, int ldb,
               const cuDoubleComplex           *beta, cuDoubleComplex         *C, int ldc)
{
    #ifdef CUSPARSE_GENERIC_INTERFACES
        generic_SpMM(handle, transA, m, n, k, nnz, ldb, ldc, alpha, csrValA, B, C, csrRowPtrA, csrColIndA, beta, CUDA_C_64F);
    #else
        cusparseCheckError(cusparseZcsrmm(handle, transA, m, n, k, nnz, alpha, descrA, csrValA, csrRowPtrA, csrColIndA, B, ldb, beta, C, ldc));
    #endif
}
}


template <class TConfig>
void Cusparse::csrmm(typename TConfig::VecPrec alpha,
                     Matrix<TConfig> &A,
                     Vector<TConfig> &V,
                     typename TConfig::VecPrec beta,
                     Vector<TConfig> &Res)
{
    if (!A.is_matrix_singleGPU())
    {
        A.manager->exchange_halo(V, V.tag);
    }

    if (Res.get_num_rows() != A.get_num_rows() || Res.get_num_cols() != V.get_num_cols())
    {
        FatalError("Cusparse::csrmm error, dimensions of result matrix do not match input matrices.", AMGX_ERR_INTERNAL);
    }

    cusparseHandle_t handle = Cusparse::get_instance().m_handle;
    cusparse_csrmm(handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                   A.get_num_rows(), V.get_num_cols(), A.get_num_cols(),
                   A.values.size(), &alpha, A.cuMatDescr,
                   A.values.raw(), A.row_offsets.raw(), A.col_indices.raw(),
                   V.raw(), V.get_lda(),
                   &beta, Res.raw(), Res.get_lda());
    Res.dirtybit = 1;
}

template <class T>
void transpose_internal(cusparseHandle_t handle, int nRows, int nCols, int nNz, const T* Avals, const int* Arows, const int* Acols, T* Bvals, int* Brows, int* Bcols, cudaDataType valType)
{
    size_t bufferSize;
    cusparseCheckError(cusparseCsr2cscEx2_bufferSize(
        handle, nRows, nCols, nNz, Avals, Arows, Acols, Bvals, Brows, Bcols, valType,
        CUSPARSE_ACTION_NUMERIC, CUSPARSE_INDEX_BASE_ZERO, CUSPARSE_CSR2CSC_ALG2, &bufferSize));

    void *buffer = nullptr;
    if (bufferSize > 0)
    {
        amgx::memory::cudaMalloc(&buffer, bufferSize);
    }

    cusparseCheckError(cusparseCsr2cscEx2(
        handle, nRows, nCols, nNz, Avals, Arows, Acols, Bvals, Brows, Bcols, valType,
        CUSPARSE_ACTION_NUMERIC, CUSPARSE_INDEX_BASE_ZERO, CUSPARSE_CSR2CSC_ALG2, buffer));

    if(bufferSize > 0)
    {
        amgx::memory::cudaFreeAsync(buffer);
    }
}

void transpose_internal(cusparseHandle_t handle, int nRows, int nCols, int nNz, const float* Avals, const int* Arows, const int* Acols, float* Bvals, int* Brows, int* Bcols)
{
    transpose_internal(handle, nRows, nCols, nNz, Avals, Arows, Acols, Bvals, Brows, Bcols, CUDA_R_32F);
}
void transpose_internal(cusparseHandle_t handle, int nRows, int nCols, int nNz, const double* Avals, const int* Arows, const int* Acols, double* Bvals, int* Brows, int* Bcols)
{
    transpose_internal(handle, nRows, nCols, nNz, Avals, Arows, Acols, Bvals, Brows, Bcols, CUDA_R_64F);
}
void transpose_internal(cusparseHandle_t handle, int nRows, int nCols, int nNz, const cuComplex* Avals, const int* Arows, const int* Acols, cuComplex* Bvals, int* Brows, int* Bcols)
{
    transpose_internal(handle, nRows, nCols, nNz, Avals, Arows, Acols, Bvals, Brows, Bcols, CUDA_C_32F);
}
void transpose_internal(cusparseHandle_t handle, int nRows, int nCols, int nNz, const cuDoubleComplex* Avals, const int* Arows, const int* Acols, cuDoubleComplex* Bvals, int* Brows, int* Bcols)
{
    transpose_internal(handle, nRows, nCols, nNz, Avals, Arows, Acols, Bvals, Brows, Bcols, CUDA_C_64F);
}

template <class TConfig>
void Cusparse::transpose(const Matrix<TConfig>& A, Matrix<TConfig>& B, const int nRows, const int nNz)
{
    cusparseHandle_t handle = Cusparse::get_instance().m_handle;
    transpose_internal(handle, nRows, A.get_num_cols(), nNz,
        A.values.raw(), A.row_offsets.raw(), A.col_indices.raw(),
        B.values.raw(), B.row_offsets.raw(), B.col_indices.raw());
}

template <class TConfig>
void Cusparse::transpose(const Matrix<TConfig>& A, Matrix<TConfig>& B)
{
    cusparseHandle_t handle = Cusparse::get_instance().m_handle;
    transpose_internal(handle, A.get_num_rows(), A.get_num_cols(), A.get_num_nz(),
        A.values.raw(), A.row_offsets.raw(), A.col_indices.raw(),
        B.values.raw(), B.row_offsets.raw(), B.col_indices.raw());
}

//#define AMGX_CASE_LINE(CASE) template class Cusparse<TemplateMode<CASE>::Type>;
//    AMGX_FORALL_BUILDS(AMGX_CASE_LINE)
//#undef AMGX_CASE_LINE

#define AMGX_CASE_LINE(CASE) \
  template void Cusparse::bsrmv(const typename TemplateMode<CASE>::Type::VecPrec ,  Matrix<TemplateMode<CASE>::Type>&, Vector<TemplateMode<CASE>::Type>&, const typename TemplateMode<CASE>::Type::VecPrec, Vector<TemplateMode<CASE>::Type>  &, ViewType);
AMGX_FORALL_BUILDS(AMGX_CASE_LINE)
AMGX_FORCOMPLEX_BUILDS(AMGX_CASE_LINE)
#undef AMGX_CASE_LINE

#define AMGX_CASE_LINE(CASE) \
  template void Cusparse::bsrxmv(const typename TemplateMode<CASE>::Type::VecPrec ,  Matrix<TemplateMode<CASE>::Type>&, Vector<TemplateMode<CASE>::Type>&, const typename TemplateMode<CASE>::Type::VecPrec, Vector<TemplateMode<CASE>::Type>&, typename Matrix<TemplateMode<CASE>::Type>::IVector&, ViewType);
AMGX_FORALL_BUILDS(AMGX_CASE_LINE)
AMGX_FORCOMPLEX_BUILDS(AMGX_CASE_LINE)
#undef AMGX_CASE_LINE

#define AMGX_CASE_LINE(CASE) \
  template void Cusparse::bsrmv_with_mask(const typename TemplateMode<CASE>::Type::VecPrec ,  Matrix<TemplateMode<CASE>::Type>&, Vector<TemplateMode<CASE>::Type>&, const typename TemplateMode<CASE>::Type::VecPrec, Vector<TemplateMode<CASE>::Type>  &);
AMGX_FORALL_BUILDS(AMGX_CASE_LINE)
AMGX_FORCOMPLEX_BUILDS(AMGX_CASE_LINE)
#undef AMGX_CASE_LINE

#define AMGX_CASE_LINE(CASE) \
  template void Cusparse::bsrmv_with_mask_restriction(const typename TemplateMode<CASE>::Type::VecPrec ,  Matrix<TemplateMode<CASE>::Type>&, Vector<TemplateMode<CASE>::Type>&, const typename TemplateMode<CASE>::Type::VecPrec, Vector<TemplateMode<CASE>::Type>  &, Matrix<TemplateMode<CASE>::Type>& );
AMGX_FORALL_BUILDS(AMGX_CASE_LINE)
AMGX_FORCOMPLEX_BUILDS(AMGX_CASE_LINE)
#undef AMGX_CASE_LINE

#define AMGX_CASE_LINE(CASE) \
  template void Cusparse::bsrmv(const typename TemplateMode<CASE>::Type::VecPrec ,  Matrix<TemplateMode<CASE>::Type>&, const typename Matrix<TemplateMode<CASE>::Type>::MVector&, Vector<TemplateMode<CASE>::Type>&, const typename TemplateMode<CASE>::Type::VecPrec, Vector<TemplateMode<CASE>::Type>  &, ViewType);
AMGX_FORALL_BUILDS(AMGX_CASE_LINE)
AMGX_FORCOMPLEX_BUILDS(AMGX_CASE_LINE)
#undef AMGX_CASE_LINE

#define AMGX_CASE_LINE(CASE) \
  template void Cusparse::bsrmv(ColumnColorSelector, const int, const typename TemplateMode<CASE>::Type::VecPrec ,  Matrix<TemplateMode<CASE>::Type>&, Vector<TemplateMode<CASE>::Type>&, const typename TemplateMode<CASE>::Type::VecPrec, Vector<TemplateMode<CASE>::Type>  &, ViewType);
AMGX_FORALL_BUILDS(AMGX_CASE_LINE)
AMGX_FORCOMPLEX_BUILDS(AMGX_CASE_LINE)
#undef AMGX_CASE_LINE


#define AMGX_CASE_LINE(CASE) \
  template void Cusparse::bsrmv(const int, const typename TemplateMode<CASE>::Type::VecPrec ,  Matrix<TemplateMode<CASE>::Type>&, const typename Matrix<TemplateMode<CASE>::Type>::MVector &, Vector<TemplateMode<CASE>::Type>&, const typename TemplateMode<CASE>::Type::VecPrec, Vector<TemplateMode<CASE>::Type>  &, ViewType);
AMGX_FORALL_BUILDS(AMGX_CASE_LINE)
AMGX_FORCOMPLEX_BUILDS(AMGX_CASE_LINE)
#undef AMGX_CASE_LINE

#define AMGX_CASE_LINE(CASE) \
  template void Cusparse::csrmm(typename TemplateMode<CASE>::Type::VecPrec, Matrix<TemplateMode<CASE>::Type>&, Vector<TemplateMode<CASE>::Type>&, typename TemplateMode<CASE>::Type::VecPrec, Vector<TemplateMode<CASE>::Type>&);
AMGX_FORALL_BUILDS(AMGX_CASE_LINE)
AMGX_FORCOMPLEX_BUILDS(AMGX_CASE_LINE)
#undef AMGX_CASE_LINE

#define AMGX_CASE_LINE(CASE) \
  template void Cusparse::transpose(const Matrix<TemplateMode<CASE>::Type>& A, Matrix<TemplateMode<CASE>::Type>& B); \
  template void Cusparse::transpose(const Matrix<TemplateMode<CASE>::Type>& A, Matrix<TemplateMode<CASE>::Type>& B, const int nRows, const int nNz);
AMGX_FORALL_BUILDS(AMGX_CASE_LINE)
AMGX_FORCOMPLEX_BUILDS(AMGX_CASE_LINE)
#undef AMGX_CASE_LINE

#ifndef DISABLE_MIXED_PRECISION
#define AMGX_CASE_LINE(CASE) template struct CusparseMatPrec<TemplateMode<CASE>::Type>;
AMGX_FORALL_BUILDS(AMGX_CASE_LINE)
AMGX_FORCOMPLEX_BUILDS(AMGX_CASE_LINE)
#undef AMGX_CASE_LINE
#endif

} // namespace amgx
