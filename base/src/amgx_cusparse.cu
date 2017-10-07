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

#include <amgx_cusparse.h>

#include <amgx_types/util.h>

namespace amgx
{

// global CUSPARSE handle for AMGX
// Cusparse cusparse;

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

    if (view != A.getViewExterior())  //This is already a view, thus do not even attempt to do latency hiding
    {
        bsrmv_internal(alphaConst, A, x, betaConst, y, view, null_stream);
    }
    else //Try and do latency hiding
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
            // Launch interior in interior stream
            bsrmv_internal(alphaConst, A, x, betaConst, y, flags, A.manager->get_int_stream());

            if (!A.is_matrix_singleGPU())
            {
                A.manager->exchange_halo_wait(x, x.tag);
            }

            A.setViewExterior();
            flags = (ViewType)(~(A.getViewInterior()) & A.getViewExterior());

            if (flags != 0)
            {
                bsrmv_internal(alphaConst, A, x, betaConst, y, flags, A.manager->get_bdy_stream());
            }
        }
        else
        {
            bsrmv_internal(alphaConst, A, x, betaConst, y, flags, null_stream);
        }

        y.dirtybit = 1;
        //if (!A.is_matrix_singleGPU() && y.size() == x.size() && y.delayed_send==0)
        //    A.manager->exchange_halo_async(y, y.tag);
        A.setView(oldView);
    }
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

    bool do_latency_hiding = true;

    if (A.getViewInterior() == A.getViewExterior())
    {
        do_latency_hiding = false;
    }

    if (do_latency_hiding)
    {
        // First gather the data into the device buffers
        if (!A.is_matrix_singleGPU())
        {
            A.manager->gather_b2l(x, x.tag);
        }

        // launch the interior
        ViewType int_view = INTERIOR;
        bsrmv_internal_with_mask(alphaConst, A, x, betaConst, y, int_view, A.manager->get_int_stream());

        // Copy data to host, call mpi send, mpi recv, copy to device
        if (!A.is_matrix_singleGPU())
        {
            A.manager->send_receive_wait(x, x.tag, A.manager->get_bdy_stream());
        }

        // Then process boundary
        ViewType bdy_view = BOUNDARY;
        bsrmv_internal_with_mask(alphaConst, A, x, betaConst, y, bdy_view, A.manager->get_bdy_stream());
//    }
    }
    else
    {
        //std::cout << "skiping latency hiding bsrmv, size = " << A.get_num_rows()  << std::endl;
        if (!A.is_matrix_singleGPU())
        {
            A.manager->exchange_halo(x, x.tag);
        }

        ViewType view = OWNED;
        bsrmv_internal(alphaConst, A, x, betaConst, y, view, null_stream);
    }

    y.dirtybit = 1;
}

template< class TConfig >
void Cusparse::bsrmv_with_mask_restriction(
    const typename TConfig::VecPrec alphaConst,
    Matrix<TConfig> &A,
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
    bool do_latency_hiding = true;

    if (A.getViewInterior() == A.getViewExterior())
    {
        do_latency_hiding = false;
    }

    cudaStream_t null_stream = 0;

    if (do_latency_hiding)
    {
        if (!P.is_matrix_singleGPU() && P.manager->neighbors.size() != 0)
        {
            // First compute the halo rows in default stream
            ViewType halo_view = HALO1;
            bsrmv_internal_with_mask_restriction(alphaConst, A, x, betaConst, y, halo_view, null_stream, P);
        }

        if (!P.is_matrix_singleGPU() && P.manager->neighbors.size() != 0)
        {
            // On GPU, gather data to a linear buffer
            P.manager->gather_l2h(y, y.tag);
        }

        // Then launch the owned rows
        ViewType owned_view = OWNED;
        bsrmv_internal_with_mask_restriction(alphaConst, A, x,  betaConst, y, owned_view, P.manager->get_int_stream(), P);

        if (!P.is_matrix_singleGPU() && P.manager->neighbors.size() != 0)
        {
            // While interior rows are processed, send, receive wait
            P.manager->add_from_halo_only(y, y.tag, P.manager->get_bdy_stream());
            // In default stream, add contribution from neighbors to vector
            //P.manager->scatter_b2l_v2(y, y.tag);
            P.manager->scatter_b2l(y, y.tag);
        }
    }
    else
    {
        //std::cout << "skiping latency hiding restriction, size = " << A.get_num_rows()  << std::endl;
        // Multiply
        ViewType view = OWNED;
        bsrmv_internal(alphaConst, A, x, betaConst, y, view, null_stream);
        // Add contribution from neighbors
        y.dirtybit = 1;
        P.manager->add_from_halo(y, y.tag);
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
            bsrmv_internal(alphaConst, A, E, x, betaConst, y, flags, A.manager->get_int_stream());

            if (!A.is_matrix_singleGPU())
            {
                A.manager->exchange_halo_wait(x, x.tag);
            }

            A.setViewExterior();
            flags = (ViewType)(~(A.getViewInterior()) & A.getViewExterior());

            if (flags != 0)
            {
                bsrmv_internal(alphaConst, A, E, x, betaConst, y, flags, A.manager->get_bdy_stream());
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
            bsrmv_internal(columnColorSelector, color, alphaConst, A, x, betaConst, y, flags, A.manager->get_int_stream());

            if (!A.is_matrix_singleGPU())
            {
                A.manager->exchange_halo_wait(x, x.tag);
            }

            A.setViewExterior();
            flags = (ViewType)(~(A.getViewInterior()) & A.getViewExterior());

            if (flags != 0)
            {
                bsrmv_internal(columnColorSelector, color, alphaConst, A, x, betaConst, y, flags, A.manager->get_bdy_stream());
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
            bsrmv_internal(color, alphaConst, A, E, x, betaConst, y, flags, A.manager->get_int_stream());

            if (!A.is_matrix_singleGPU())
            {
                A.manager->exchange_halo_wait(x, x.tag);
            }

            A.setViewExterior();
            flags = (ViewType)(~(A.getViewInterior()) & A.getViewExterior());

            if (flags != 0)
            {
                bsrmv_internal(color, alphaConst, A, E, x, betaConst, y, flags, A.manager->get_bdy_stream());
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
    int offset, size;
    A.getOffsetAndSizeForView(view, &offset, &size);
    cusparseDirection_t direction = CUSPARSE_DIRECTION_COLUMN;

    if ( A.getBlockFormat() == ROW_MAJOR )
    {
        direction = CUSPARSE_DIRECTION_ROW;
    }

    bool has_offdiag = A.get_num_nz() != 0;

    if (has_offdiag )
    {
        cusparseSetStream(Cusparse::get_instance().m_handle, stream);
        bsrmv( Cusparse::get_instance().m_handle,  direction, CUSPARSE_OPERATION_NON_TRANSPOSE,
               size, A.get_num_cols(), A.get_num_nz(), &alphaConst,
               A.cuMatDescr,
               A.values.raw(),
               A.m_seq_offsets.raw() + offset,
               A.row_offsets.raw() + offset, A.col_indices.raw(),
               A.get_block_dimx(),
               x.raw(), &betaConst,
               y.raw() + offset * A.get_block_dimx() );
        // Reset to default stream
        cusparseSetStream(Cusparse::get_instance().m_handle, 0);
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

        cusparseSetStream(Cusparse::get_instance().m_handle, stream);
        bsrmv( Cusparse::get_instance().m_handle,  direction, CUSPARSE_OPERATION_NON_TRANSPOSE,
               size, A.get_num_cols(), A.get_num_rows(), &alphaConst,
               A.cuMatDescr,
               A.values.raw() + A.diagOffset()*A.get_block_size(),
               A.m_seq_offsets.raw(),
               A.m_seq_offsets.raw() + offset, A.m_seq_offsets.raw(),
               A.get_block_dimx(),
               x.raw(), &beta,
               y.raw() + offset * A.get_block_dimx() );
        // Reset to default stream
        cusparseSetStream(Cusparse::get_instance().m_handle, 0);
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

    typedef typename TConfig::VecPrec ValueType;
    //int offset, size;
    //A.getOffsetAndSizeForView(view, &offset, &size);
    cusparseDirection_t direction = CUSPARSE_DIRECTION_COLUMN;

    if ( A.getBlockFormat() == ROW_MAJOR )
    {
        direction = CUSPARSE_DIRECTION_ROW;
    }

    bool has_offdiag = A.get_num_nz() != 0;
    const int *start_offsets, *end_offsets;
    start_offsets = A.row_offsets.raw();
    end_offsets = A.row_offsets.raw() + 1;
    typedef typename Matrix<TConfig>::index_type index_type;
    // num rows to
    index_type NumRows = A.manager->getRowsListForView(view).size();

    if (NumRows <= 0)
    {
        return;    // nothing to do, early exit
    }

    if (has_offdiag )
    {
        cusparseSetStream(Cusparse::get_instance().m_handle, stream);
        bsrxmv_internal( Cusparse::get_instance().m_handle, direction, CUSPARSE_OPERATION_NON_TRANSPOSE, NumRows,
                         A.get_num_rows(), A.get_num_cols(), A.get_num_nz(), &alphaConst,
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
        const Matrix<TConfig> &A,
        const Vector<TConfig> &x,
        const typename TConfig::VecPrec betaConst,
        Vector<TConfig> &y,
        ViewType view,
        const cudaStream_t &stream,
        Matrix<TConfig> &P)
{
    if (P.is_matrix_singleGPU())
    {
        FatalError("Should not be here in bsrmv_internal_with_mask", AMGX_ERR_NOT_IMPLEMENTED);
    }

    typedef typename TConfig::VecPrec ValueType;
    //int offset, size;
    //A.getOffsetAndSizeForView(view, &offset, &size);
    cusparseDirection_t direction = CUSPARSE_DIRECTION_COLUMN;

    if ( A.getBlockFormat() == ROW_MAJOR )
    {
        direction = CUSPARSE_DIRECTION_ROW;
    }

    bool has_offdiag = A.get_num_nz() != 0;
    typedef typename Matrix<TConfig>::index_type index_type;
    // num rows to
    int offset, NumRows;

    if (view == OWNED)
    {
        NumRows = P.manager->halo_offsets[0];
        offset = 0;
    }
    else if (view == HALO1)
    {
        offset = P.manager->halo_offsets[0];
        NumRows = P.manager->halo_offsets[P.manager->neighbors.size()] - offset;
    }
    else
    {
        FatalError("View not supported in restriction operation", AMGX_ERR_NOT_IMPLEMENTED);
    }

    if (NumRows <= 0)
    {
        return;    // nothing to do, early exit
    }

    if (has_offdiag )
    {
        cusparseSetStream(Cusparse::get_instance().m_handle, stream);
        bsrmv( Cusparse::get_instance().m_handle,  direction, CUSPARSE_OPERATION_NON_TRANSPOSE,
               NumRows, A.get_num_cols(), A.get_num_nz(), &alphaConst,
               A.cuMatDescr,
               A.values.raw(),
               A.m_seq_offsets.raw() + offset,
               A.row_offsets.raw() + offset, A.col_indices.raw(),
               A.get_block_dimx(),
               x.raw(), &betaConst,
               y.raw() + offset * A.get_block_dimx() );
        // Reset to default stream
        cusparseSetStream(Cusparse::get_instance().m_handle, 0);
    }

    if (A.hasProps(DIAG))
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
    int offset, size;
    A.getOffsetAndSizeForView(view, &offset, &size);
    cusparseDirection_t direction = A.getBlockFormat() == ROW_MAJOR ? CUSPARSE_DIRECTION_ROW : CUSPARSE_DIRECTION_COLUMN;
    cusparseSetStream(Cusparse::get_instance().m_handle, stream);
    bsrmv( Cusparse::get_instance().m_handle, direction, CUSPARSE_OPERATION_NON_TRANSPOSE,
           size, A.get_num_cols(), A.get_num_nz(), &alphaConst,
           A.cuMatDescr,
           E.raw(),
           A.m_seq_offsets.raw(),
           A.m_seq_offsets.raw() + offset, A.m_seq_offsets.raw(),
           A.get_block_dimx(),
           x.raw(), &betaConst,
           y.raw() + offset * A.get_block_dimx() );
    // Reset to default stream
    cusparseSetStream(Cusparse::get_instance().m_handle, 0);
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
                             float *y)
{
    if (blockDim == 1)
    {
        cusparseCheckError(cusparseScsrmv(handle, trans, mb, nb, nnzb, alpha, descr, bsrVal, bsrRowPtr, bsrColInd, x, beta, y));
    }
    else
    {
        cusparseCheckError(cusparseSbsrmv(handle, dir, trans, mb, nb, nnzb, alpha, descr, bsrVal, bsrRowPtr, bsrColInd, blockDim, x, beta, y));
    }
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
                             double *y)
{
    if (blockDim == 1)
    {
        cusparseCheckError(cusparseDcsrmv(handle, trans, mb, nb, nnzb, alpha, descr, bsrVal, bsrRowPtr, bsrColInd, x, beta, y));
    }
    else
    {
        cusparseCheckError(cusparseDbsrmv(handle, dir, trans, mb, nb, nnzb, alpha, descr, bsrVal, bsrRowPtr, bsrColInd, blockDim, x, beta, y));
    }
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
                             double *y)
{
    const double *d_bsrVal = reinterpret_cast<const double *>(const_cast<float *>(bsrVal)); // this works due to private API call in the matrix initialization which sets cusparse matrix description in the half precision mode
    cusparseCheckError(cusparseDbsrxmv(handle, dir, trans, mb, mb, nb, nnzb, alpha, descr, d_bsrVal, bsrMaskPtr, bsrRowPtr, bsrRowPtr + 1, bsrColInd, blockDim, x, beta, y));
}

// overloaded C++ wrappers for cusparse?bsrxmv
// bsrxmv
// matrix - float
// vector - float
inline void Cusparse::bsrxmv_internal( cusparseHandle_t handle, cusparseDirection_t dir, cusparseOperation_t trans, int sizeOfMask,
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
        if (bsrEndPtr == NULL) { bsrEndPtr = bsrRowPtr + 1; }

        cusparseCheckError(cusparseSbsrxmv(handle, dir, trans, sizeOfMask, mb, nb, nnzb, alpha, descr, bsrVal, bsrMaskPtr, bsrRowPtr, bsrEndPtr, bsrColInd, blockDim, x, beta, y));
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

        cusparseCheckError(cusparseDbsrxmv(handle, dir, trans, sizeOfMask, mb, nb, nnzb, alpha, descr, bsrVal, bsrMaskPtr, bsrRowPtr, bsrEndPtr, bsrColInd, blockDim, x, beta, y));
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
                             cuComplex *y)
{
    if (blockDim == 1)
    {
        cusparseCheckError(cusparseCcsrmv(handle, trans, mb, nb, nnzb, alpha, descr, bsrVal, bsrRowPtr, bsrColInd, x, beta, y));
    }
    else
    {
        cusparseCheckError(cusparseCbsrmv(handle, dir, trans, mb, nb, nnzb, alpha, descr, bsrVal, bsrRowPtr, bsrColInd, blockDim, x, beta, y));
    }
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
                             cuDoubleComplex *y)
{
    if (blockDim == 1)
    {
        cusparseCheckError(cusparseZcsrmv(handle, trans, mb, nb, nnzb, alpha, descr, bsrVal, bsrRowPtr, bsrColInd, x, beta, y));
    }
    else
    {
        cusparseCheckError(cusparseZbsrmv(handle, dir, trans, mb, nb, nnzb, alpha, descr, bsrVal, bsrRowPtr, bsrColInd, blockDim, x, beta, y));
    }
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
                             cuDoubleComplex *y)
{
    const cuDoubleComplex *d_bsrVal = reinterpret_cast<cuDoubleComplex *>(const_cast<cuComplex *>(bsrVal));
    cusparseCheckError(cusparseZbsrxmv(handle, dir, trans, mb, mb, nb, nnzb, alpha, descr, d_bsrVal, bsrMaskPtr, bsrRowPtr, bsrRowPtr + 1, bsrColInd, blockDim, x, beta, y));
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
        if (bsrEndPtr == NULL) { bsrEndPtr = bsrRowPtr + 1; }

        cusparseCheckError(cusparseCbsrxmv(handle, dir, trans, sizeOfMask, mb, nb, nnzb, alpha, descr, bsrVal, bsrMaskPtr, bsrRowPtr, bsrEndPtr, bsrColInd, blockDim, x, beta, y));
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

        cusparseCheckError(cusparseZbsrxmv(handle, dir, trans, sizeOfMask, mb, nb, nnzb, alpha, descr, bsrVal, bsrMaskPtr, bsrRowPtr, bsrEndPtr, bsrColInd, blockDim, x, beta, y));
    }
}


namespace
{
cusparseStatus_t
cusparse_csrmm(cusparseHandle_t handle, cusparseOperation_t transA,
               int m, int n, int k, int nnz,
               const float           *alpha,
               const cusparseMatDescr_t descrA,
               const float            *csrValA,
               const int *csrRowPtrA, const int *csrColIndA,
               const float            *B, int ldb,
               const float            *beta, float          *C, int ldc)
{
    return cusparseScsrmm(handle, transA, m, n, k, nnz, alpha, descrA, csrValA, csrRowPtrA, csrColIndA, B, ldb, beta, C, ldc);
}

cusparseStatus_t
cusparse_csrmm(cusparseHandle_t handle, cusparseOperation_t transA,
               int m, int n, int k, int nnz,
               const double            *alpha,
               const cusparseMatDescr_t descrA,
               const float            *csrValA,
               const int *csrRowPtrA, const int *csrColIndA,
               const double            *B, int ldb,
               const double           *beta, double          *C, int ldc)
{
    return CUSPARSE_STATUS_MATRIX_TYPE_NOT_SUPPORTED;
}

cusparseStatus_t
cusparse_csrmm(cusparseHandle_t handle, cusparseOperation_t transA,
               int m, int n, int k, int nnz,
               const double          *alpha,
               const cusparseMatDescr_t descrA,
               const double          *csrValA,
               const int *csrRowPtrA, const int *csrColIndA,
               const double           *B, int ldb,
               const double           *beta, double         *C, int ldc)
{
    return cusparseDcsrmm(handle, transA, m, n, k, nnz, alpha, descrA, csrValA, csrRowPtrA, csrColIndA, B, ldb, beta, C, ldc);
}

cusparseStatus_t
cusparse_csrmm(cusparseHandle_t handle, cusparseOperation_t transA,
               int m, int n, int k, int nnz,
               const cuComplex           *alpha,
               const cusparseMatDescr_t descrA,
               const cuComplex            *csrValA,
               const int *csrRowPtrA, const int *csrColIndA,
               const cuComplex            *B, int ldb,
               const cuComplex            *beta, cuComplex          *C, int ldc)
{
    return cusparseCcsrmm(handle, transA, m, n, k, nnz, alpha, descrA, csrValA, csrRowPtrA, csrColIndA, B, ldb, beta, C, ldc);
}

cusparseStatus_t
cusparse_csrmm(cusparseHandle_t handle, cusparseOperation_t transA,
               int m, int n, int k, int nnz,
               const cuDoubleComplex            *alpha,
               const cusparseMatDescr_t descrA,
               const cuComplex            *csrValA,
               const int *csrRowPtrA, const int *csrColIndA,
               const cuDoubleComplex            *B, int ldb,
               const cuDoubleComplex           *beta, cuDoubleComplex          *C, int ldc)
{
    return CUSPARSE_STATUS_MATRIX_TYPE_NOT_SUPPORTED;
}

cusparseStatus_t
cusparse_csrmm(cusparseHandle_t handle, cusparseOperation_t transA,
               int m, int n, int k, int nnz,
               const cuDoubleComplex          *alpha,
               const cusparseMatDescr_t descrA,
               const cuDoubleComplex          *csrValA,
               const int *csrRowPtrA, const int *csrColIndA,
               const cuDoubleComplex           *B, int ldb,
               const cuDoubleComplex           *beta, cuDoubleComplex         *C, int ldc)
{
    return cusparseZcsrmm(handle, transA, m, n, k, nnz, alpha, descrA, csrValA, csrRowPtrA, csrColIndA, B, ldb, beta, C, ldc);
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
    cusparseCheckError(cusparse_csrmm(handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                                      A.get_num_rows(), V.get_num_cols(), A.get_num_cols(),
                                      A.values.size(), &alpha, A.cuMatDescr,
                                      A.values.raw(), A.row_offsets.raw(), A.col_indices.raw(),
                                      V.raw(), V.get_lda(),
                                      &beta, Res.raw(), Res.get_lda()));
    Res.dirtybit = 1;
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

#define AMGX_CASE_LINE(CASE) template struct CusparseMatPrec<TemplateMode<CASE>::Type>;
AMGX_FORALL_BUILDS(AMGX_CASE_LINE)
AMGX_FORCOMPLEX_BUILDS(AMGX_CASE_LINE)
#undef AMGX_CASE_LINE

} // namespace amgx
