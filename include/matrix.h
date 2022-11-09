/* Copyright (c) 2011-2017, NVIDIA CORPORATION. All rights reserved.
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

#define AMGX_WARP_SIZE 32
#define AMGX_GRID_MAX_SIZE 4096

namespace amgx
{
template <class T_Config> class Matrix;
template <class T_Config> class MatrixBase;
template <class T_Config> class MatrixView;

/* The matrix properties (props) below are managed using the following routines:
   bool hasProps()
   void setProps(p)
   void addProps(p)
   void delProps(p) */
enum MatrixProps
{
    NONE = 0,   /* matrix has no properties (default) */
    COO = 1,    /* matrix is stored in COO format (row_indices, col_indices and values have been set) */
    CSR = 2,    /* matrix is stored in CSR format (row_offsets, col_indices and values have been set) */
    DIAG = 4,   /* matrix has external diagonal (diag) seprately stored */
    COLORING = 8, /* matrix coloring (m_matrix_coloring) has been computed */
    COMPLEX = 16 /* enum for defining complex property. is used in binary matrix read/write */
};
}
//enum BLOCK_FORMAT{ ROW_MAJOR=0, COLUMN_MAJOR=1};
#include <cstdio>
#include <map>

#include <operators/operator.h>
#include <amgx_cusparse.h>
#include <thrust/sequence.h>
#include <thrust/scan.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/copy.h>
#include <vector.h>
#include <error.h>
#include <permute.h>
#include <matrix_coloring/matrix_coloring.h>
#include <distributed/distributed_manager.h>
#include <resources.h>
#include <thrust_wrapper.h>

namespace amgx
{

#ifndef DISABLE_MIXED_PRECISION
template <class T_Config> struct CusparseMatPrec;
#endif

template <class TConfig> class Solver;
template <class TConfig> class Matrix;
template <class TConfig> class Operator;

typedef typename IndPrecisionMap<AMGX_indInt>::Type INDEX_TYPE;

extern void computeRowOffsetsDevice(int num_blocks, INDEX_TYPE num_rows, INDEX_TYPE num_nz, const INDEX_TYPE *row_indices, INDEX_TYPE *row_offsets, INDEX_TYPE block_size  );
extern void computeRowIndicesDevice(int num_blocks, INDEX_TYPE num_rows, const INDEX_TYPE *row_offsets, INDEX_TYPE *row_indices, INDEX_TYPE block_size );
extern void computeColorOffsetsDeviceCSR(int num_blocks, INDEX_TYPE num_rows, INDEX_TYPE *row_offsets, INDEX_TYPE *col_indices, const INDEX_TYPE *row_colors, INDEX_TYPE *smaller_color_offsets, INDEX_TYPE *larger_color_offsets, INDEX_TYPE block_size, INDEX_TYPE *diag );
template <typename T>
void reorderElementsDeviceCSR(INDEX_TYPE num_rows, INDEX_TYPE *row_offsets, INDEX_TYPE *permutation, INDEX_TYPE *col_indices, T *values, INDEX_TYPE block_size);

template <class T_Config>
class MatrixBase : public AuxData, public Operator<T_Config>
{
    public:
        typedef T_Config TConfig;
        typedef typename TConfig::MemSpace memory_space;
        typedef typename TConfig::MatPrec  value_type;
        typedef typename TConfig::IndPrec  index_type;
        DEFINE_VECTOR_TYPES

    protected:
        index_type block_dimy;
        index_type block_dimx;
        index_type block_size;
        unsigned int props;
        index_type num_rows, num_cols, num_nz;
        bool allow_recompute_diag;
        BlockFormat block_format;
        ViewType current_view;  //distributed: current view of the matrix (returned get_num_rows()/get_num_nz() are different)
        bool allow_boundary_separation; // serves distributed read system. Switches off reordering if false

        MatrixColoring<T_Config> *m_matrix_coloring;

        ViewType m_separation_exterior;
        ViewType m_separation_interior;

        int m_cols_reordered_by_color;
        bool m_is_matrix_setup;
        bool m_is_permutation_inplace;
        bool m_is_read_partitioned; //distributed: need this to tell the partitioner the matrix is coming from distributed reader

    private:
        inline void setProps( unsigned int new_props ) { props = new_props; }

    public:

        inline index_type get_block_dimx() const { return block_dimx; }
        inline index_type get_block_dimy() const { return block_dimy; }
        inline index_type get_block_size() const { return block_size; }
        inline index_type get_num_rows()   const { return num_rows; }
        inline index_type get_num_cols()   const { return num_cols; }
        inline index_type get_num_nz()     const { return num_nz; }

        inline void set_block_dimx(index_type new_dimx)
        {
            if (this->is_initialized())
            {
                FatalError("Trying to modify already initialized matrix\n", AMGX_ERR_BAD_PARAMETERS);
            }

            block_dimx = new_dimx;
            block_size = block_dimx * block_dimy;
            values.set_block_dimx(block_dimx);
        }
        inline void set_block_dimy(index_type new_dimy)
        {
            if (this->is_initialized())
            {
                FatalError("Trying to modify already initialized matrix\n", AMGX_ERR_BAD_PARAMETERS);
            }

            block_dimy = new_dimy;
            block_size = block_dimx * block_dimy;
            values.set_block_dimy(block_dimy);
        }
        inline void set_num_nz(index_type new_num_nz)
        {
            if (this->is_initialized())
            {
                FatalError("Trying to modify already initialized matrix\n", AMGX_ERR_BAD_PARAMETERS);
            }

            num_nz = new_num_nz;
        }
        inline void set_num_cols(index_type new_num_cols)
        {
            if (this->is_initialized())
            {
                FatalError("Trying to modify already initialized matrix\n", AMGX_ERR_BAD_PARAMETERS);
            }

            num_cols = new_num_cols;
        }
        inline void set_num_rows(index_type new_num_rows)
        {
            if (this->is_initialized())
            {
                FatalError("Trying to modify already initialized matrix\n", AMGX_ERR_BAD_PARAMETERS);
            }

            num_rows = new_num_rows;
        }
        inline void set_allow_recompute_diag(bool new_val)
        {
            if (this->is_initialized())
            {
                FatalError("Trying to modify already initialized matrix\n", AMGX_ERR_BAD_PARAMETERS);
            }

            allow_recompute_diag = new_val;
        }
        inline bool get_allow_boundary_separation() const {return allow_boundary_separation;}
        inline void set_allow_boundary_separation(int new_val)
        {
            if (new_val == 0) { allow_boundary_separation = false; }
            else { allow_boundary_separation = true; }
        }
        inline BlockFormat getBlockFormat()  { return block_format; }
        inline BlockFormat getBlockFormat() const  { return block_format; }

        inline void setBlockFormat(BlockFormat &in_block_format)
        {
            block_format = in_block_format;
        }

        // End of old block matrices support
        DistributedManager<T_Config> *manager;

        //integer index of the AMG level to which the matrix belongs (fine level has index 0, the next coarse level has index 1, and so on)
        int amg_level_index;

        void setDefaultParameters()
        {
            this->setParameter("level", (int)(0));
        }

        MatrixBase() :  m_is_read_partitioned(false), manager(NULL), amg_level_index(0), manager_internal(true), props(NONE), num_rows(0), num_cols(0), num_nz(0), block_dimy(1), block_dimx(1), block_size(1), m_initialized(0), row_offsets(0), col_indices(0), values(0), row_indices(0), diag(0), current_view(ALL), m_matrix_coloring(NULL), m_cols_reordered_by_color(0), m_separation_interior(INTERIOR), m_separation_exterior(OWNED), m_is_matrix_setup(false), m_is_permutation_inplace(false), m_values_permutation_vector(0), m_larger_color_offsets(0), m_smaller_color_offsets(0), m_seq_offsets(0), m_diag_end_offsets(0), allow_recompute_diag(true), block_format(ROW_MAJOR), m_resources(NULL), allow_boundary_separation(true)
        {
            setDefaultParameters();
            resize(0, 0, 0, 1);
            cusparseCheckError(cusparseCreateMatDescr(&cuMatDescr));

            #ifndef DISABLE_MIXED_PRECISION
                cusparseCheckError(CusparseMatPrec<T_Config>::set(cuMatDescr));
            #endif
        }

        inline MatrixBase(index_type num_rows, index_type num_cols, index_type num_nz, unsigned int props ) : m_is_read_partitioned(false), manager(NULL), amg_level_index(0), manager_internal(true), block_dimy(1), block_dimx(1), block_size(1), m_initialized(0), row_offsets(0), col_indices(0), values(0), row_indices(0), diag(0), m_matrix_coloring(NULL), m_cols_reordered_by_color(0), m_is_matrix_setup(false), m_is_permutation_inplace(false), m_separation_interior(INTERIOR), m_separation_exterior(OWNED), m_larger_color_offsets(0), m_smaller_color_offsets(0), m_seq_offsets(0), m_diag_end_offsets(0), allow_recompute_diag(true), current_view(ALL), block_format(ROW_MAJOR), m_resources(NULL), allow_boundary_separation(true)
        {
            setDefaultParameters();
            this->props = props;
            resize(num_rows, num_cols, num_nz, 1);
            cusparseCheckError(cusparseCreateMatDescr(&cuMatDescr));

            #ifndef DISABLE_MIXED_PRECISION
                cusparseCheckError(CusparseMatPrec<T_Config>::set(cuMatDescr));
            #endif
        }
        inline MatrixBase(index_type num_rows, index_type num_cols, index_type num_nz, index_type block_dimy, index_type block_dimx, unsigned int props): m_is_read_partitioned(false), manager(NULL), amg_level_index(0), manager_internal(true), m_initialized(0), row_offsets(0), col_indices(0), values(0), row_indices(0), diag(0), m_matrix_coloring(NULL), m_cols_reordered_by_color(0), m_is_matrix_setup(false), m_is_permutation_inplace(false), m_separation_interior(INTERIOR), m_separation_exterior(OWNED), m_larger_color_offsets(0), m_smaller_color_offsets(0), m_seq_offsets(0), m_diag_end_offsets(0), allow_recompute_diag(true), current_view(ALL), block_format(ROW_MAJOR), m_resources(NULL), allow_boundary_separation(true)
        {
            setDefaultParameters();
            this->props = props;
            resize(num_rows, num_cols, num_nz, block_dimy, block_dimx, 1);
            cusparseCheckError(cusparseCreateMatDescr(&cuMatDescr));

            #ifndef DISABLE_MIXED_PRECISION
                cusparseCheckError(CusparseMatPrec<T_Config>::set(cuMatDescr));
            #endif
        }

        virtual ~MatrixBase()
        {
            if (manager_internal && manager != NULL)
            {
                delete manager;
            }

            if (cuMatDescr != NULL)
            {
                cusparseDestroyMatDescr(cuMatDescr);
                cuMatDescr = NULL;
            }

            m_values_permutation_vector.clear();
            m_values_permutation_vector.shrink_to_fit();
            m_larger_color_offsets.clear();
            m_larger_color_offsets.shrink_to_fit();
            m_smaller_color_offsets.clear();
            m_smaller_color_offsets.shrink_to_fit();
            m_seq_offsets.clear();
            m_seq_offsets.shrink_to_fit();
            m_diag_end_offsets.clear();
            m_diag_end_offsets.shrink_to_fit();

            if (m_matrix_coloring != NULL && m_matrix_coloring->release())
            {
                delete m_matrix_coloring;
            }
        }

        inline void copyCusparseMatDescr(cusparseMatDescr_t d_cuMatDescr, const cusparseMatDescr_t s_cuMatDescr)
        {
            cusparseSetMatType(d_cuMatDescr, cusparseGetMatType(s_cuMatDescr));
            cusparseSetMatFillMode(d_cuMatDescr, cusparseGetMatFillMode(s_cuMatDescr));
            cusparseSetMatDiagType(d_cuMatDescr, cusparseGetMatDiagType(s_cuMatDescr));
            cusparseSetMatIndexBase(d_cuMatDescr, cusparseGetMatIndexBase(s_cuMatDescr));
        }

        template<class MatrixType>
        inline void copy(const MatrixType &a)
        {
            this->set_initialized(0);
            copyAuxData(&a);
            block_dimy = a.get_block_dimy();
            block_dimx = a.get_block_dimx();
            block_size = a.get_block_size();
            this->setProps(a.getProps());
            num_rows = a.get_num_rows();
            num_cols = a.get_num_cols();
            num_nz = a.get_num_nz();
            values = a.values;
            row_offsets = a.row_offsets;
            row_indices = a.row_indices;
            col_indices = a.col_indices;
            diag = a.diag;
            m_resources = a.getResources();
            /*
            WARNING: ideally we would copy the entire matrix descriptor, including
            any hidden fields, but the matrix descriptor is an opaque structure,
            so there is no way to do it right now. Notice that
            you don't want to copy the pointer because then you will not know when
            and which matrix object should free it, at least not without careful
            reference counting.
            */
            copyCusparseMatDescr(cuMatDescr, a.cuMatDescr);

            if (a.hasProps(COLORING))
            {
                copyMatrixColoring(a.getMatrixColoring());
            }

            //manager->copy(*(a.manager));
            amg_level_index = a.amg_level_index;
            m_cols_reordered_by_color = a.getColsReorderedByColor();
            m_is_matrix_setup = a.is_matrix_setup();
            m_is_permutation_inplace = a.is_permutation_inplace();
            m_larger_color_offsets = a.m_larger_color_offsets;
            m_smaller_color_offsets = a.m_smaller_color_offsets;
            m_values_permutation_vector = a.m_values_permutation_vector;
            m_diag_end_offsets = a.m_diag_end_offsets;
            m_seq_offsets = a.m_seq_offsets;
            //end_offsets=a.end_offsets;
            this->set_initialized(a.is_initialized() ? 1 : 0);
        }

        template<class MatrixType>
        inline void copy_structure(const MatrixType &a)
        {
            this->set_initialized(0);
            copyAuxData(&a);
            block_dimy = a.get_block_dimy();
            block_dimx = a.get_block_dimx();
            block_size = a.get_block_size();
            values.set_block_dimx(block_dimx);
            values.set_block_dimy(block_dimy);
            this->setProps(a.getProps());
            num_rows = a.get_num_rows();
            num_cols = a.get_num_cols();
            num_nz = a.get_num_nz();
            row_offsets = a.row_offsets;
            row_indices = a.row_indices;
            col_indices = a.col_indices;
            diag = a.diag;
            m_resources = a.getResources();
            /*
            WARNING: ideally we would copy the entire matrix descriptor, including
            any hidden fields, but the matrix descriptor is an opaque structure,
            so there is no way to do it right now. Notice that
            you don't want to copy the pointer because then you will not know when
            and which matrix object should free it, at least not without careful
            reference counting.
            */
            copyCusparseMatDescr(cuMatDescr, a.cuMatDescr);

            if (a.hasProps(COLORING))
            {
                copyMatrixColoring(a.getMatrixColoring());
            }

            amg_level_index = a.amg_level_index;
            // If matrix is reordered
            m_cols_reordered_by_color = a.getColsReorderedByColor();
            m_is_matrix_setup = a.is_matrix_setup();
            m_is_permutation_inplace = a.is_permutation_inplace();
            m_larger_color_offsets = a.m_larger_color_offsets;
            m_smaller_color_offsets = a.m_smaller_color_offsets;
            m_values_permutation_vector = a.m_values_permutation_vector;
            m_diag_end_offsets = a.m_diag_end_offsets;
            m_seq_offsets = a.m_seq_offsets;
            this->set_initialized(a.is_initialized() ? 1 : 0);
        }

        template<class MatrixType>
        inline void copy_async(const MatrixType &a, cudaStream_t stream = 0)
        {
            this->set_initialized(0);
            copyAuxData(&a);
            block_dimy = a.get_block_dimy();
            block_dimx = a.get_block_dimx();
            block_size = a.get_block_size();
            this->setProps(a.getProps());
            num_rows = a.get_num_rows();
            num_cols = a.get_num_cols();
            num_nz = a.get_num_nz();
            values.copy_async(a.values, stream);
            row_offsets.copy_async(a.row_offsets, stream);
            row_indices.copy_async(a.row_indices, stream);
            col_indices.copy_async(a.col_indices, stream);
            diag.copy_async(a.diag, stream);
            m_resources = a.getResources();
            /*
            WARNING: ideally we would copy the entire matrix descriptor, including
            any hidden fields, but the matrix descriptor is an opaque structure,
            so there is no way to do it right now. Notice that
            you don't want to copy the pointer because then you will not know when
            and which matrix object should free it, at least not without careful
            reference counting.
            */
            copyCusparseMatDescr(cuMatDescr, a.cuMatDescr);
            //if (this->getColoringLevel())
            //  m_matrix_coloring = new MatrixColoring<T_Config>(a.getMatrixColoring());
            // If matrix is reordered
            m_cols_reordered_by_color = a.getColsReorderedByColor();
            m_is_matrix_setup = a.is_matrix_setup();
            m_is_permutation_inplace = a.is_permutation_inplace();
            m_larger_color_offsets.copy_async(a.m_larger_color_offsets, stream);
            m_smaller_color_offsets.copy_async(a.m_smaller_color_offsets, stream);
            m_values_permutation_vector.copy_async(a.m_values_permutation_vector, stream);
            m_diag_end_offsets.copy_async(a.m_diag_end_offsets, stream);
            m_seq_offsets.copy_async(a.m_seq_offsets, stream);
            //end_offsets.copy_async(a.end_offsets,stream);
            //manager->copy(*(a.manager));
            amg_level_index = a.amg_level_index;
            this->set_initialized(a.is_initialized() ? 1 : 0);
        }

        template<class MatrixType>
        inline void copy_structure_async(const MatrixType &a, cudaStream_t stream = 0)
        {
            this->set_initialized(0);
            copyAuxData(&a);
            block_dimy = a.get_block_dimy();
            block_dimx = a.get_block_dimx();
            block_size = a.get_block_size();
            values.set_block_dimx(block_dimx);
            values.set_block_dimy(block_dimy);
            this->setProps(a.getProps());
            num_rows = a.get_num_rows();
            num_cols = a.get_num_cols();
            num_nz = a.get_num_nz();
            row_offsets.copy_async(a.row_offsets, stream);
            row_indices.copy_async(a.row_indices, stream);
            col_indices.copy_async(a.col_indices, stream);
            diag.copy_async(a.diag, stream);
            m_resources = a.getResources();
            /*
            WARNING: ideally we would copy the entire matrix descriptor, including
            any hidden fields, but the matrix descriptor is an opaque structure,
            so there is no way to do it right now. Notice that
            you don't want to copy the pointer because then you will not know when
            and which matrix object should free it, at least not without careful
            reference counting.
            */
            copyCusparseMatDescr(cuMatDescr, a.cuMatDescr);
            //if (this->getColoringLevel())
            //  m_matrix_coloring = new MatrixColoring<T_Config>(a.getMatrixColoring());
            // If matrix is reordered
            m_cols_reordered_by_color = a.getColsReorderedByColor();
            m_is_matrix_setup = a.is_matrix_setup();
            m_is_permutation_inplace = a.is_permutation_inplace();
            m_larger_color_offsets.copy_async(a.m_larger_color_offsets, stream);
            m_smaller_color_offsets.copy_async(a.m_smaller_color_offsets, stream);
            m_values_permutation_vector.copy_async(a.m_values_permutation_vector, stream);
            m_diag_end_offsets.copy_async(a.m_diag_end_offsets, stream);
            m_seq_offsets.copy_async(a.m_seq_offsets, stream);
            amg_level_index = a.amg_level_index;
            this->set_initialized(a.is_initialized() ? 1 : 0);
        }

        inline void sync()
        {
            //only need to sync the last copy
            m_seq_offsets.sync();
        }

        template<class MatrixType>
        inline void swap(MatrixType &a)
        {
            int was_init = this->is_initialized();
            int was_init2 = a.is_initialized();
            this->set_initialized(0);
            a.set_initialized(0);
            index_type temp;
            temp =  block_dimy;
            block_dimy = a.get_block_dimy();
            values.set_block_dimy(block_dimy);
            a.set_block_dimy(temp);
            a.values.set_block_dimy(temp);
            temp = block_dimx;
            block_dimx = a.get_block_dimx();
            values.set_block_dimx(block_dimx);
            a.set_block_dimx(temp);
            a.values.set_block_dimx(temp);
            temp = num_rows;
            num_rows = a.get_num_rows();
            a.set_num_rows(temp);
            temp = num_cols;
            num_cols = a.get_num_cols();
            a.set_num_cols(temp);
            temp = num_nz;
            num_nz = a.get_num_nz();
            a.set_num_nz(temp);
            unsigned int temp2 = this->getProps();
            this->setProps(a.getProps());
            a.setProps(temp2);
            values.swap(a.values);
            row_offsets.swap(a.row_offsets);
            row_indices.swap(a.row_indices);
            col_indices.swap(a.col_indices);
            diag.swap(a.diag);
            m_seq_offsets.swap(a.m_seq_offsets);
            m_larger_color_offsets.swap(a.m_larger_color_offsets);
            m_smaller_color_offsets.swap(a.m_smaller_color_offsets);
            m_values_permutation_vector.swap(a.m_values_permutation_vector);
            m_diag_end_offsets.swap(a.m_diag_end_offsets);
            Resources *temp_rsrc = m_resources;
            m_resources = a.getResources();
            a.setResources(temp_rsrc);
            /*
            WARNING: ideally we would swap the entire matrix descriptor, including
            any hidden fields, but the matrix descriptor is an opaque structure,
            so there is no way to do it right now. Notice that
            you don't want to copy the pointer because then you will not know when
            and which matrix object should free it, at least not without careful
            reference counting.
            */
            cusparseMatDescr_t t_cuMatDescr;
            cusparseCreateMatDescr(&t_cuMatDescr);
            copyCusparseMatDescr(t_cuMatDescr, cuMatDescr);
            copyCusparseMatDescr(cuMatDescr,  a.cuMatDescr);
            copyCusparseMatDescr(a.cuMatDescr, t_cuMatDescr);
            cusparseDestroyMatDescr(t_cuMatDescr);
            //manager->swap(*(a.manager));
            amg_level_index = a.amg_level_index;
            this->set_initialized(was_init2);
            a.set_initialized(was_init);
        }

        // We don't specialize the above template to hide CUSP dependency here.
        template< class CuspCsrMatrix >
        inline void copyFromCuspCsr( const CuspCsrMatrix &A )
        {
            this->set_initialized(0);
            this->setProps(CSR);
            num_rows    = A.get_num_rows();
            num_cols    = A.get_num_cols();
            num_nz      = A.num_entries;
            row_offsets = A.row_offsets;
            col_indices = A.column_indices;
            values      = A.values;
            m_seq_offsets = A.m_seq_offsets;
            //end_offsets = A.end_offsets;
            block_size  = index_type( 1 );
            block_dimx  = index_type( 1 );
            values.set_block_dimx(block_dimx);
            block_dimy  = index_type( 1 );
            values.set_block_dimy(block_dimy);
            row_indices.clear( );
            computeDiagonal( );
            this->set_initialized(A.is_initialized() ? 1 : 0);
        }

        // We don't specialize the above template to hide CUSP dependency here.
        template< class CuspCooMatrix >
        inline void copyFromCuspCoo( const CuspCooMatrix &A )
        {
            this->set_initialized(0);
            this->setProps(COO);
            num_rows    = A.get_num_rows();
            num_cols    = A.get_num_cols();
            num_nz      = A.num_entries;
            row_indices = A.row_indices;
            col_indices = A.col_indices;
            values      = A.values;
            m_seq_offsets = A.m_seq_offsets;
            //end_offsets = A.end_offsets;
            block_size  = index_type( 1 );
            block_dimx  = index_type( 1 );
            values.set_block_dimx(block_dimx);
            block_dimy  = index_type( 1 );
            values.set_block_dimy(block_dimy);
            row_offsets.clear( );
            computeDiagonal( );
            this->set_initialized(A.is_initialized() ? 1 : 0);
        }

        template<class MatrixType>
        inline MatrixBase<TConfig> &operator=(const MatrixType &a)
        {
            this->copy(a);
            return *this;
        }

        /* WARNING: without this additional implementation of the operator=
           the compiler will use its own (shallow) copy operator= which results
           in cuMatDescr pointer being copied from one object to the next and
           ultimately causes double free. This probably happens because compiler
           does not instantiate MatrixType correctly for all the types.  */
        inline MatrixBase<TConfig> &operator=(const MatrixBase<TConfig> &a)
        {
            this->copy(a);
            return *this;
        }

        AMGX_ERROR resize(index_type num_rows, index_type num_cols, index_type num_nz, int skipDiaCompute = 0);

        //Allocate more space for nonzeros. Must resize col_indices, row_indices and values at the point when the number of values is known
        inline AMGX_ERROR resize_spare(index_type num_rows, index_type num_cols, index_type num_nz, index_type block_dimy, index_type block_dimx, double spare)
        {
            int new_nz = (int)(((double) num_nz) * spare);
            this->block_dimy = block_dimy;
            this->block_dimx = block_dimx;
            this->block_size = block_dimy * block_dimx;
            AMGX_ERROR err = this->resize(num_rows, num_cols, new_nz, 1);
            this->num_nz = num_nz;
            return err;
        }

        inline AMGX_ERROR resize(index_type num_rows, index_type num_cols, index_type num_nz, index_type block_dimy, index_type block_dimx, int skipDiaCompute = 0)
        {
            this->block_dimy = block_dimy;
            this->block_dimx = block_dimx;
            this->block_size = block_dimy * block_dimx;
            this->values.set_block_dimx(block_dimx);
            this->values.set_block_dimy(block_dimy);
            return this->resize(num_rows, num_cols, num_nz, skipDiaCompute);
        }

        static inline bool hasProps( unsigned int query, unsigned int props) { return (query | props) == props; }
        inline bool hasProps(unsigned int props) const { return hasProps(props, this->props); }

        inline void addProps(unsigned int new_props)
        {
            if (is_initialized())
            {
                FatalError("Trying to modify already initialized matrix\n", AMGX_ERR_BAD_PARAMETERS);
            }

            //combine properties
            new_props |= props;

            if (props == new_props)
            {
                return;
            }

            bool addCOO = hasProps(COO, new_props) && !hasProps(COO, props);

            if ( addCOO )
            {
                row_indices.resize(num_nz);

                //update row_indices from row_offsets
                if (num_nz > 0 && num_rows > 0)
                {
                    computeRowIndices();
                }
            }

            bool addCSR = hasProps(CSR, new_props) && !hasProps(CSR, props);

            if ( addCSR )
            {
                row_offsets.resize(num_rows + 1);

                //update row_offsets from row_indices
                if (num_nz > 0 && num_rows > 0)
                {
                    computeRowOffsets();
                }
            }

            bool addDiag = hasProps(DIAG, new_props) && !hasProps(DIAG, props);
            props = new_props;
        }

        inline void delProps(unsigned int rem_props)
        {
            if (is_initialized())
            {
                FatalError("Trying to modify already initialized matrix\n", AMGX_ERR_BAD_PARAMETERS);
            }

            //compute the new properties (subtract common properties between rem_props and props from the current props)
            unsigned int new_props = props - (props & rem_props);

            if (props == new_props)
            {
                return;
            }

            bool delCOO = hasProps(COO, props) && !hasProps(COO, new_props);

            if ( delCOO )
            {
                row_indices.resize(0);
            }

            bool delCSR = hasProps(CSR, props) && !hasProps(CSR, new_props);

            if ( delCSR )
            {
                row_offsets.resize(0);
            }

            bool delDIAG = hasProps(DIAG, props) && !hasProps(DIAG, new_props);
//      if( delDIAG )
//        diag.resize(0);
            props = new_props;
        }

        /********************************************************/
        inline unsigned int getProps() const { return props; };

        inline size_t bytes(bool device_only = false)
        {
            return row_offsets.bytes(device_only) +
                   col_indices.bytes(device_only) +
                   values.bytes(device_only) +
                   row_indices.bytes(device_only) +
                   diag.bytes(device_only) +
                   m_seq_offsets.bytes(device_only) +
                   m_larger_color_offsets.bytes(device_only) +
                   m_smaller_color_offsets.bytes(device_only) +
                   m_diag_end_offsets.bytes(device_only) +
                   m_values_permutation_vector.bytes(device_only) ;
        }
        /********************************************************/

        /* Color related */
        //size_t num_colors;
        //// Offsets of the sorted_rows_by_color
        //IVector_h offsets_rows_per_color;
        //// Storage for the indices of a certain color.
        //IVector sorted_rows_by_color;
        //// Storage for the color of each row
        //IVector row_colors;

        /* Coloring related */

        inline int getColoringLevel()
        {
            if (!hasProps(COLORING))
            {
                return 0;
            }
            else
            {
                return this->getMatrixColoring().getColoringLevel();
            }
        }

        inline int getColoringLevel() const
        {
            if (!hasProps(COLORING))
            {
                return 0;
            }
            else
            {
                return this->getMatrixColoring().getColoringLevel();
            }
        }

        inline MatrixColoring<T_Config> &getMatrixColoring()
        {
            if (hasProps(COLORING))
            {
                return *m_matrix_coloring;
            }
            else
            {
                FatalError("Matrix Coloring does not exists. colorMatrix must be called first", AMGX_ERR_CORE);
            }
        }
        inline const MatrixColoring<T_Config> &getMatrixColoring() const
        {
            if (hasProps(COLORING))
            {
                return *m_matrix_coloring;
            }
            else
            {
                FatalError("Matrix Coloring does not exists. colorMatrix must be called first", AMGX_ERR_CORE);
            }
        }


        inline void setMatrixColoring(MatrixColoring<T_Config> *m_coloring)
        {
            this->set_initialized(0);

            if (m_matrix_coloring != NULL && m_matrix_coloring->release())
            {
                delete m_matrix_coloring;
            }

            m_matrix_coloring = m_coloring;
            m_matrix_coloring->retain();
            this->addProps(COLORING);
            this->set_initialized(1);
        }

        // Copy Matrix Coloring
        template <class MatrixColoringType>
        inline void copyMatrixColoring(MatrixColoringType &m_coloring)
        {
            if (m_matrix_coloring != NULL && m_matrix_coloring->release())
            {
                delete m_matrix_coloring;
            }

            m_matrix_coloring = new MatrixColoring<T_Config>(m_coloring);
            addProps(COLORING);
        }

        virtual void colorMatrix(AMG_Config &cfg, const std::string &cfg_scope) = 0;
        virtual void colorMatrixUsingAggregates( AMG_Config &cfg, const std::string &cfg_scope, IVector &R_row_offsets, IVector &R_col_indices, IVector &aggregates ) {if ( !this->hasProps(COLORING)) colorMatrix( cfg, cfg_scope ); }

        virtual void setupMatrix(Solver<TConfig> *solver, AMG_Config &cfg, bool reuse_matrix_structure);

        inline void setColsReorderedByColor(bool reordered_by_colors) {m_cols_reordered_by_color = reordered_by_colors;}
        inline bool getColsReorderedByColor() const {return m_cols_reordered_by_color;}

        inline bool is_matrix_setup() const {return m_is_matrix_setup;}

        inline void set_is_matrix_setup(bool is_matrix_setup) {m_is_matrix_setup = is_matrix_setup;}

        inline bool is_permutation_inplace() const {return m_is_permutation_inplace;}

        inline bool is_matrix_distributed() const { return (this->manager != NULL); }

        inline bool is_matrix_singleGPU() const
        {
            if (this->manager == NULL) { return 1; }

            return (this->manager->num_neighbors() == 0);
        }
        // else return 0;}

        inline bool is_manager_external() const {return !manager_internal;}

        inline bool is_matrix_read_partitioned() const {return m_is_read_partitioned;}
        void set_is_matrix_read_partitioned(bool is_read_partitioned) {m_is_read_partitioned = is_read_partitioned;}

        inline Resources *getResources() const { return m_resources; }
        inline void setResources(Resources *resources) { m_resources = resources; }

        bool isLatencyHidingEnabled(AMG_Config& cfg);

        IVector m_larger_color_offsets; //size: num_rows
        IVector m_smaller_color_offsets; //size: num_rows,
        IVector m_values_permutation_vector;

        /* CSR Members */
        IVector row_offsets; //size: num_rows+1

        /* CSR/COO Members */
        IVector col_indices; //size: num_nz
        MVector values;      //size: num_nz*block_size

        /* COO Members */
        IVector row_indices; //size: num_nz

        /* DIAG */
        IVector diag;        //size: num_rows*block_size
#ifdef DEBUG
        IVector diag_copy;        //size: num_rows*block_size
#endif
        /* cusparse */
        cusparseMatDescr_t cuMatDescr;

        Resources *m_resources;
        /* sequence */
        IVector m_seq_offsets; //size: num_rows+1, contents= {0, ... , num_rows}

        IVector m_diag_end_offsets; //size: num_rows+1, contents= {diag[0]+1,diag[1]+1, ... , diag[num_rows]+1}
        /********************************************************/

        //initializes the diag vector from either COO or CSR format
        virtual void computeDiagonal() = 0;

        inline int is_initialized() const {return m_initialized;};
        inline void set_initialized(int new_value)
        {
            m_initialized = new_value;

            if (new_value > 0)
            {
                if (!is_matrix_singleGPU())
                {
                    manager->set_initialized(row_offsets);
                }
            }
        };

        void printConfig()
        {
            printf("Configuration: %s, %s, %s, %s\n",
                   TConfig::MemSpaceInfo::getName(),
                   TConfig::VecPrecInfo::getName(),
                   TConfig::MatPrecInfo::getName(),
                   TConfig::IndPrecInfo::getName());
        }

        //initializes the row_offsets vector from row_indices
        //assumes row_indices are sorted
        virtual void computeRowOffsets() = 0;

        void reorderColumnsByColor(bool insert_diagonal);

        void sortByRowAndColumn();

        //void reorderRowsAndColumnsByColor(bool permute_values_flag);

        virtual void permuteValues() = 0;


        inline ViewType currentView() const
        {
            return current_view;
        }

        //should be used in a distributed setting to get the offset into this->values() where the diagonal values are stored.
        inline index_type diagOffset() const
        {
            if (is_matrix_singleGPU()) { return get_num_nz(); }
            else if (m_initialized) { return manager->num_nz_all(); }
            else if (hasProps(CSR)) { return row_offsets[row_offsets.size() - 1]; }
            else { return get_num_nz(); }
        }

        //set num_rows/nz returned by get_num_rows()/get_num_nz()
        inline void setView(ViewType type)
        {
            if (is_matrix_singleGPU()) { return; }

            current_view = type;
            manager->getView(type, num_rows, num_nz);
        }

        inline void setViewInterior() { setView(m_separation_interior); }
        inline void setViewExterior() { setView(m_separation_exterior); }
        inline void setInteriorView(ViewType view) { m_separation_interior = view; }
        inline void setExteriorView(ViewType view) { m_separation_exterior = view; }
        inline ViewType getViewInterior() const { return m_separation_interior; }
        inline ViewType getViewExterior() const { return m_separation_exterior; }
        inline ViewType getViewIntExt() const { return (ViewType)(this->getViewInterior() | this->getViewExterior()); }

        //return row offset into matrix and number of rows for a given view
        inline void getOffsetAndSizeForView(ViewType type, int *offset, int *size) const
        {
            if (is_matrix_singleGPU())
            {
                *offset = 0;
                *size = num_rows;

                if ((type & INTERIOR) == 0) { *size = 0; }

                return;
            }

            manager->getOffsetAndSizeForView(type, offset, size);
        }

        // return number of nnz for a view
        inline void getNnzForView(ViewType type, int *nnz) const
        {
            if (is_matrix_singleGPU())
            {
                *nnz = num_nz;

                if ((type & INTERIOR) == 0)
                {
                    *nnz = 0;
                }
                return;
            }

            manager->getNnzForView(type, nnz);
        }

        // Get the offset, nrows, and nnz for this view
        inline void getFixedSizesForView(ViewType type, int *offset, int *nrows, int* nnz) const
        {
            if(!is_matrix_distributed() || !manager->isViewSizeFixed())
            {
                FatalError("getFixedSizesForView should not be called by a non-distributed matrix", AMGX_ERR_INTERNAL);
            }

            manager->getOffsetAndSizeForView(type, offset, nrows);
            manager->getNnzForView(type, nnz);
        }

        inline void setManager(DistributedManager<T_Config> &manager_)
        {
            if (manager_internal && manager != NULL)
            {
                delete manager;
            }

            manager = &manager_;
            manager_internal = false;
        }

        DistributedManager<TConfig> *getManager() const
        {
            return manager;
        }

        inline void setManagerExternal()
        {
            manager_internal = false;
        }

    protected:
        //initializes the row_indices vector from row_offsets
        virtual void computeRowIndices() = 0;

        virtual void computeColorOffsets() = 0;

        bool manager_internal;
        int m_initialized;
};

// specialization for host
template <AMGX_VecPrecision t_vecPrec, AMGX_MatPrecision t_matPrec, AMGX_IndPrecision t_indPrec>
class Matrix< TemplateConfig<AMGX_host, t_vecPrec, t_matPrec, t_indPrec> > : public MatrixBase< TemplateConfig<AMGX_host, t_vecPrec, t_matPrec, t_indPrec> >
{
    public:
        typedef TemplateConfig<AMGX_host, t_vecPrec, t_matPrec, t_indPrec> TConfig;
        typedef typename TConfig::MemSpace memory_space;
        typedef typename TConfig::MatPrec  value_type;
        typedef typename TConfig::IndPrec  index_type;
        DEFINE_VECTOR_TYPES

        template< AMGX_MemorySpace MemorySpace > struct Rebind { typedef Matrix<TemplateConfig<MemorySpace, t_vecPrec, t_matPrec, t_indPrec> > Type; };

        //constructors
        inline Matrix() : MatrixBase<TConfig>() { }
        inline Matrix(index_type num_rows, index_type num_cols, index_type num_nz, unsigned int props ) : MatrixBase<TConfig>(num_rows, num_cols, num_nz, props) { }
        inline Matrix(index_type num_rows, index_type num_cols, index_type num_nz, index_type block_dimy, index_type block_dimx, unsigned int props ) : MatrixBase<TConfig>(num_rows, num_cols, num_nz, block_dimy, block_dimx, props) { }
        inline Matrix(const Matrix<TConfig>   &a) : MatrixBase<TConfig>() { this->copy(a); }
        inline Matrix(const Matrix<TConfig_d> &a) : MatrixBase<TConfig>() { this->copy(a); }
        //destructor
        virtual ~Matrix() {}

        template<class MatrixType>
        inline Matrix<TConfig> &operator=(const MatrixType &a)
        {
            this->copy(a);
            return *this;
        }
        /* WARNING: without this additional implementation of the operator=
           the compiler will use its own (shallow) copy operator= which results
           in cuMatDescr pointer being copied from one object to the next and
           ultimately causes double free. This probably happens because compiler
           does not instantiate MatrixType correctly for all the types. */
        Matrix<TConfig> &operator=(const Matrix<TConfig> &a)
        {
            this->copy(a);
            return *this;
        }

        void print(char *f, char *s, int srows, int erows, int trank);

        void printToFile(char *f, char *s, int srows, int erows);

        void apply(const Vector<TConfig> &v, Vector<TConfig> &res, ViewType view = OWNED);

        void colorMatrix(AMG_Config &cfg, const std::string &cfg_scope)
        {
            //locally downwind needs aggregates to perform coloring
            if (this->m_matrix_coloring != NULL && this->m_matrix_coloring->release())
            {
                delete (this->m_matrix_coloring);
            }

            this->m_matrix_coloring = MatrixColoringFactory<TConfig>::allocate(cfg, cfg_scope);

            if (this->hasParameter("coloring") && this->template getParameter<int>("coloring_size") == this->get_num_rows())
            {
                IVector_h *row_colors = this->template getParameterPtr< IVector_h >("coloring");
                this->m_matrix_coloring->setRowColors(*row_colors);
                int colors_num = this->template getParameter< int >("colors_num");
                this->m_matrix_coloring->setNumColors(colors_num);
            }
            else
            {
                this->m_matrix_coloring->colorMatrix(*this);
            }

            this->m_matrix_coloring->createColorArrays(*this);

            if (this->m_seq_offsets.raw() == NULL)
            {
                this->m_seq_offsets.resize(this->row_offsets.size());
                thrust_wrapper::sequence<AMGX_host>(this->m_seq_offsets.begin(), this->m_seq_offsets.end());
                cudaCheckError();
            }

            this->addProps(COLORING);
        }

        // general conversion routine
        void convert( const Matrix<TConfig> &mat, unsigned int new_props, int block_dimy, int block_dimx ) ;

        void computeColorOffsets()
        {
            FatalError("Compute color end offsets not support for host matrices", AMGX_ERR_NOT_IMPLEMENTED);
        }

        void reorderValuesInPlace()
        {
            FatalError("Compute color end offsets not support for host matrices", AMGX_ERR_NOT_IMPLEMENTED);
        }

        void permuteValues()
        {
            FatalError("Permute Values not support for host matrices", AMGX_ERR_NOT_IMPLEMENTED);
        }


        /*
        void reorderColumnsByColor(const bool permute_values_flag)
        {
          FatalError("Reorder not supported on host matrices", AMGX_ERR_NOT_IMPLEMENTED);
        }
        void reorderColumnsByColor(IVector &color_permutation_vector, const bool permute_values_flag)
        {
          FatalError("Reorder not supported on host matrices", AMGX_ERR_NOT_IMPLEMENTED);
        }

        void reorderRowsAndColumnsByColor(const bool permute_values_flag)
        {
          FatalError("Reorder not supported on host matrices", AMGX_ERR_NOT_IMPLEMENTED);
        }
        */

        //void reorderMatrix(IVector permutation) {
        //  FatalError("Reorder not supported on host matrices", AMGX_ERR_NOT_IMPLEMENTED);
        //}

        void computeDiagonal();

        //assumes row_indices is sorted
        void computeRowOffsets()
        {
            IVector &row_offsets = this->row_offsets;
            IVector &row_indices = this->row_indices;
            row_offsets[0] = 0;
            int nz = 0;

            for (int row = 1; row < this->get_num_rows(); row++)
            {
                row_offsets[row] = row_offsets[row - 1];

                while ( nz < this->get_num_nz() && row_indices[nz] == row - 1)
                {
                    row_offsets[row]++;
                    nz++;
                }
            }

            row_offsets[this->get_num_rows()] = this->get_num_nz();
        }

    protected:
        void computeRowIndices()
        {
            int j = 0;
            IVector &row_offsets = this->row_offsets;
            IVector &row_indices = this->row_indices;

            for (int i = 1; i <= this->num_rows; i++)
            {
                while (j < row_offsets[i])
                {
                    row_indices[j] = i - 1;
                    j++;
                }
            }
        }


};

// specialization for device
template <AMGX_VecPrecision t_vecPrec, AMGX_MatPrecision t_matPrec, AMGX_IndPrecision t_indPrec>
class Matrix< TemplateConfig<AMGX_device, t_vecPrec, t_matPrec, t_indPrec> > : public MatrixBase< TemplateConfig<AMGX_device, t_vecPrec, t_matPrec, t_indPrec> >
{
    public:
        typedef TemplateConfig<AMGX_device, t_vecPrec, t_matPrec, t_indPrec> TConfig;
        typedef typename TConfig::IndPrec  index_type;
        typedef typename TConfig::VecPrec  vec_value_type;
        typedef typename TConfig::MatPrec  value_type;
        DEFINE_VECTOR_TYPES

        template< AMGX_MemorySpace MemorySpace > struct Rebind { typedef Matrix<TemplateConfig<MemorySpace, t_vecPrec, t_matPrec, t_indPrec> > Type; };

        //constructors
        inline Matrix() : MatrixBase<TConfig>() {}
        inline Matrix(index_type num_rows, index_type num_cols, index_type num_nz, unsigned int props ) : MatrixBase<TConfig>(num_rows, num_cols, num_nz, props) { }
        inline Matrix(index_type num_rows, index_type num_cols, index_type num_nz, index_type block_dimy, index_type block_dimx, unsigned int props ) : MatrixBase<TConfig>(num_rows, num_cols, num_nz, block_dimy, block_dimx, props) { }
        inline Matrix(const Matrix<TConfig_h> &a) : MatrixBase<TConfig>() { this->copy(a); }
        inline Matrix(const Matrix<TConfig_d> &a) : MatrixBase<TConfig>() { this->copy(a); }
        //destructor
        virtual ~Matrix() {}

        template<class MatrixType>
        inline Matrix<TConfig> &operator=(const MatrixType &a)
        {
            this->copy(a);
            return *this;
        }
        /* WARNING: without this additional implementation of the operator=
           the compiler will use its own (shallow) copy operator= which results
           in cuMatDescr pointer being copied from one object to the next and
           ultimately causes double free. This probably happens because compiler
           does not instantiate MatrixType correctly for all the types.  */
        Matrix<TConfig> &operator=(const Matrix<TConfig> &a)
        {
            this->copy(a);
            return *this;
        }

        void print(char *f, char *s, int srows, int erows, int trank);

        void printToFile(char *f, char *s, int srows, int erows);

        void apply(const Vector<TConfig> &v, Vector<TConfig> &res, ViewType view = OWNED);

        void colorMatrix(AMG_Config &cfg, const std::string &cfg_scope)
        {
            //locally downwind needs the aggregates to perform the coloring
            std::string coloring_algorithm = cfg.AMG_Config::getParameter<std::string>("matrix_coloring_scheme", cfg_scope );

            if ( coloring_algorithm.compare( "LOCALLY_DOWNWIND" ) == 0 )
            {
                return;
            }

            if (this->m_matrix_coloring != NULL && this->m_matrix_coloring->release())
            {
                delete (this->m_matrix_coloring);
            }

            this->m_matrix_coloring = MatrixColoringFactory<TConfig>::allocate(cfg, cfg_scope);

            // Copy the colors if provided by user
            if (this->hasParameter("coloring") && this->template getParameter<int>("coloring_size") == this->get_num_rows())
            {
                IVector_h *row_colors = this->template getParameterPtr< IVector_h >("coloring");
                this->m_matrix_coloring->setRowColors(*row_colors);
                int colors_num = this->template getParameter< int >("colors_num");
                this->m_matrix_coloring->setNumColors(colors_num);
            }
            else
            {
                this->m_matrix_coloring->colorMatrix(*this);;
            }

            this->m_matrix_coloring->createColorArrays(*this);;

            if (this->m_seq_offsets.raw() == NULL)
            {
                this->m_seq_offsets.resize(this->row_offsets.size());
                thrust_wrapper::sequence<AMGX_device>(this->m_seq_offsets.begin(), this->m_seq_offsets.end());
                cudaCheckError();
            }

            this->addProps(COLORING);
        }

        void colorMatrixUsingAggregates( AMG_Config &cfg, const std::string &cfg_scope, IVector &R_row_offsets, IVector &R_col_indices, IVector &aggregates )
        {
            if ( this->hasProps(COLORING) )
            {
                if ( cfg.AMG_Config::getParameter<int>( "print_coloring_info", cfg_scope ) == 1 )
                {
                    this->m_matrix_coloring->assertColoring( *this, aggregates );
                }

                return;
            }

            if (this->m_matrix_coloring != NULL && this->m_matrix_coloring->release())
            {
                delete (this->m_matrix_coloring);
            }

            this->m_matrix_coloring = MatrixColoringFactory<TConfig>::allocate(cfg, cfg_scope);

            if (this->hasParameter("coloring") && this->template getParameter<int>("coloring_size") == this->get_num_rows())
            {
                IVector_h *row_colors = this->template getParameterPtr< IVector_h >("coloring");
                this->m_matrix_coloring->setRowColors(*row_colors);
                int colors_num = this->template getParameter< int >("colors_num");
                this->m_matrix_coloring->setNumColors(colors_num);
            }
            else
            {
                this->m_matrix_coloring->colorMatrixUsingAggregates(*this, R_row_offsets, R_col_indices, aggregates);
            }

            if ( cfg.AMG_Config::getParameter<int>( "print_coloring_info", cfg_scope ) == 1 )
            {
                this->m_matrix_coloring->assertColoring( *this, aggregates );
            }

            this->m_matrix_coloring->createColorArrays(*this);

            if (this->m_seq_offsets.raw() == NULL)
            {
                this->m_seq_offsets.resize(this->row_offsets.size());
                thrust_wrapper::sequence<AMGX_device>(this->m_seq_offsets.begin(), this->m_seq_offsets.end());
                cudaCheckError();
            }

            this->addProps(COLORING);
        }


        void computeColorOffsets()
        {
            if (!(this->get_num_rows() > 0)) { return; }

            int num_blocks = min(4096, (this->get_num_rows() + 511) / 512);
            computeColorOffsetsDeviceCSR(num_blocks, this->get_num_rows(), this->row_offsets.raw(), this->col_indices.raw(), this->m_matrix_coloring->getRowColors().raw(), this->m_smaller_color_offsets.raw(), this->m_larger_color_offsets.raw(), this->get_block_size(), this->diag.raw());
        }

        void reorderValuesInPlace()
        {
            reorderElementsDeviceCSR(this->get_num_rows(), this->row_offsets.raw(), this->m_values_permutation_vector.raw(), this->col_indices.raw(), this->values.raw(), this->get_block_size());
        }

        void permuteValues();

        void computeDiagonal();

        //assumes row_indices is sorted
        void computeRowOffsets()
        {
            int num_blocks = min(4096, (this->get_num_nz() + 511) / 512);
            computeRowOffsetsDevice(num_blocks, this->get_num_rows(), this->get_num_nz(), this->row_indices.raw(), this->row_offsets.raw(), this->get_block_size());
        }

        template <typename IndexType, typename metricType, typename matrixType>
        struct copy_pred
        {
            double _trunc_factor;
            metricType *_metric_arr;

            copy_pred() : _trunc_factor(0), _metric_arr(NULL) {};
            copy_pred(const double trunc_factor, metricType *metric_arr) : _trunc_factor(trunc_factor),
                _metric_arr(metric_arr) {};

            __host__ __device__
            bool operator()(const amgx::thrust::tuple<IndexType, IndexType, matrixType> &a)
            {
                metricType metric = _metric_arr[amgx::thrust::get<0>(a)];

                if (fabs(amgx::thrust::get<2>(a)) >= _trunc_factor * metric) { return true; }

                return false;
            }

            copy_pred<IndexType, metricType, matrixType> &operator=(const copy_pred<IndexType, metricType, matrixType> &a)
            {
                this->_trunc_factor = a._trunc_factor;
                this->_metric_arr = a._metric_arr;
                return *this;
            }
        };

        template <typename IndexType, typename VectorType, typename MatrixType>
        struct scale_op
        {
            const VectorType *scale_vec;

            scale_op(const VectorType *s) : scale_vec(s) {};

            __host__ __device__
            amgx::thrust::tuple<IndexType, MatrixType> operator()(const amgx::thrust::tuple<IndexType, MatrixType> &a)
            {
                const IndexType row = amgx::thrust::get<0>(a);
                return amgx::thrust::tuple<IndexType, MatrixType>(row, amgx::thrust::get<1>(a) / scale_vec[row]);
            }
        };

    protected:
        void computeRowIndices()
        {
            int num_blocks = min(4096, (this->get_num_rows() + 511) / 512);
            computeRowIndicesDevice(num_blocks, this->get_num_rows(), this->row_offsets.raw(), this->row_indices.raw(), this->get_block_size());
        }
};

} //end namespace amgx

