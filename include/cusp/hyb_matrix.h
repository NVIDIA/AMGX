/*
 *  Copyright 2008-2009 NVIDIA Corporation
 *
 *  Licensed under the Apache License, Version 2.0 (the "License");
 *  you may not use this file except in compliance with the License.
 *  You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 *  Unless required by applicable law or agreed to in writing, software
 *  distributed under the License is distributed on an "AS IS" BASIS,
 *  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  See the License for the specific language governing permissions and
 *  limitations under the License.
 */

/*! \file hyb_matrix.h
 *  \brief Hybrid ELL/COO matrix format
 */

#pragma once

#include <cusp/detail/config.h>

#include <cusp/array1d.h>
#include <cusp/format.h>
#include <cusp/detail/matrix_base.h>

namespace cusp
{
    // Forward definitions
    template <typename IndexType, typename ValueType, class MemorySpace> class ell_matrix;
    template <typename IndexType, typename ValueType, class MemorySpace> class coo_matrix;
    template <typename Matrix1, typename Matrix2, typename IndexType, typename ValueType, class MemorySpace> class hyb_matrix_view;

/*! \addtogroup sparse_matrices Sparse Matrices
 */

/*! \addtogroup sparse_matrix_containers Sparse Matrix Containers
 *  \ingroup sparse_matrices
 *  \{
 */

/*! \p hyb_matrix : Hybrid ELL/COO matrix container
 *
 * The \p hyb_matrix is a combination of the \p ell_matrix and
 * \p coo_matrix formats.  Specifically, the \p hyb_matrix format
 * splits a matrix into two portions, one stored in ELL format 
 * and one stored in COO format.
 *
 * While the ELL format is well-suited to vector and SIMD
 * architectures, its efficiency rapidly degrades when the number of
 * nonzeros per matrix row varies.  In contrast, the storage efficiency of
 * the COO format is invariant to the distribution of nonzeros per row, and
 * the use of segmented reduction makes its performance largely invariant
 * as well.  To obtain the advantages of both, we combine these
 * into a hybrid ELL/COO format.
 *
 * The purpose of the HYB format is to store the typical number of
 * nonzeros per row in the ELL data structure and the remaining entries of
 * exceptional rows in the COO format.  
 *
 * \tparam IndexType Type used for matrix indices (e.g. \c int).
 * \tparam ValueType Type used for matrix values (e.g. \c float).
 * \tparam MemorySpace A memory space (e.g. \c cusp::host_memory or cusp::device_memory)
 *
 * \note The \p ell_matrix entries must be sorted by column index.
 * \note The \p ell_matrix entries within each row should be shifted to the left.
 * \note The \p coo_matrix entries must be sorted by row index.
 * \note The matrix should not contain duplicate entries.
 *
 *  The following code snippet demonstrates how to create a \p hyb_matrix.
 *  In practice we usually do not construct the HYB format directly and
 *  instead convert from a simpler format such as (COO, CSR) into HYB.
 *
 *  \code
 *  #include <cusp/hyb_matrix.h>
 *  ...
 *
 *  // allocate storage for (4,3) matrix with 8 nonzeros
 *  //     ELL portion has 5 nonzeros and storage for 2 nonzeros per row
 *  //     COO portion has 3 nonzeros
 *
 *  cusp::hyb_matrix<int, float, cusp::host_memory> A(3, 4, 5, 3, 2);
 *  
 *  // Initialize A to represent the following matrix
 *  // [10  20  30  40]
 *  // [ 0  50   0   0]
 *  // [60   0  70  80]
 *  
 *  // A is split into ELL and COO parts as follows
 *  // [10  20  30  40]    [10  20   0   0]     [ 0   0  30  40] 
 *  // [ 0  50   0   0]  = [ 0  50   0   0]  +  [ 0   0   0   0]
 *  // [60   0  70  80]    [60   0  70   0]     [ 0   0   0  80]
 *  
 *  
 *  // Initialize ELL part
 *
 *  // X is used to fill unused entries in the ELL portion of the matrix 
 *  const int X = cusp::ell_matrix<int,float,cusp::host_memory>::invalid_index;
 *
 *  // first row
 *  A.ell.column_indices(0,0) = 0; A.ell.values(0,0) = 10;
 *  A.ell.column_indices(0,1) = 1; A.ell.values(0,1) = 20;
 *
 *  // second row
 *  A.ell.column_indices(1,0) = 1; A.ell.values(1,0) = 50;  // shifted to leftmost position 
 *  A.ell.column_indices(1,1) = X; A.ell.values(1,1) =  0;  // padding
 *
 *  // third row
 *  A.ell.column_indices(2,0) = 0; A.ell.values(2,0) = 60;
 *  A.ell.column_indices(2,1) = 2; A.ell.values(2,1) = 70;  // shifted to leftmost position 
 *
 *
 *  // Initialize COO part
 *  A.coo.row_indices[0] = 0;  A.coo.column_indices[0] = 2;  A.coo.values[0] = 30;
 *  A.coo.row_indices[1] = 0;  A.coo.column_indices[1] = 3;  A.coo.values[1] = 40;
 *  A.coo.row_indices[2] = 2;  A.coo.column_indices[2] = 3;  A.coo.values[2] = 80;
 *
 *  \endcode
 *
 *  \see \p ell_matrix
 *  \see \p coo_matrix
 */
template <typename IndexType, typename ValueType, class MemorySpace>
class hyb_matrix : public detail::matrix_base<IndexType,ValueType,MemorySpace,cusp::hyb_format>
{
  typedef cusp::detail::matrix_base<IndexType,ValueType,MemorySpace,cusp::hyb_format> Parent;
    public:
    /*! rebind matrix to a different MemorySpace
     */
    template<typename MemorySpace2>
    struct rebind { typedef cusp::hyb_matrix<IndexType, ValueType, MemorySpace2> type; };
    
    /*! equivalent container type
     */
    typedef typename cusp::hyb_matrix<IndexType, ValueType, MemorySpace> container;

    /*! equivalent view type
     */
    typedef typename cusp::hyb_matrix_view<typename cusp::ell_matrix<IndexType,ValueType,MemorySpace>::view,
                                           typename cusp::coo_matrix<IndexType,ValueType,MemorySpace>::view,
                                           IndexType, ValueType, MemorySpace> view;
    
    /*! equivalent const_view type
     */
    typedef typename cusp::hyb_matrix_view<typename cusp::ell_matrix<IndexType,ValueType,MemorySpace>::const_view,
                                           typename cusp::coo_matrix<IndexType,ValueType,MemorySpace>::const_view,
                                           IndexType, ValueType, MemorySpace> const_view;
    
    /*! type of \p ELL portion of the HYB structure 
     */
    typedef cusp::ell_matrix<IndexType,ValueType,MemorySpace> ell_matrix_type;
    
    /*! type of \p COO portion of the HYB structure 
     */
    typedef cusp::coo_matrix<IndexType,ValueType,MemorySpace> coo_matrix_type;
    
    /*! Storage for the \p ell_matrix portion.
     */
    ell_matrix_type ell;
    
    /*! Storage for the \p ell_matrix portion.
     */
    coo_matrix_type coo;

    /*! Construct an empty \p hyb_matrix.
     */
    hyb_matrix() {}

    /*! Construct a \p hyb_matrix with a specific shape and separation into ELL and COO portions.
     *
     *  \param num_rows Number of rows.
     *  \param num_cols Number of columns.
     *  \param num_ell_entries Number of nonzero matrix entries in the ELL portion.
     *  \param num_coo_entries Number of nonzero matrix entries in the ELL portion.
     *  \param num_entries_per_row Maximum number of nonzeros per row in the ELL portion.
     *  \param alignment Amount of padding used to align the ELL data structure (default 32).
     */
    hyb_matrix(IndexType num_rows, IndexType num_cols,
               IndexType num_ell_entries, IndexType num_coo_entries,
               IndexType num_entries_per_row, IndexType alignment = 32)
    : Parent(num_rows, num_cols, num_ell_entries + num_coo_entries),
      ell(num_rows, num_cols, num_ell_entries, num_entries_per_row, alignment),
      coo(num_rows, num_cols, num_coo_entries) {}

    // TODO remove default alignment of 32

    /*! Construct a \p hyb_matrix from another matrix.
     *
     *  \param matrix Another sparse or dense matrix.
     */
    template <typename MatrixType>
    hyb_matrix(const MatrixType& matrix);
    
    /*! Resize matrix dimensions and underlying storage
     */
    void resize(IndexType num_rows, IndexType num_cols,
                IndexType num_ell_entries, IndexType num_coo_entries,
                IndexType num_entries_per_row, IndexType alignment = 32)
    {
      Parent::resize(num_rows, num_cols, num_ell_entries + num_coo_entries);
      ell.resize(num_rows, num_cols, num_ell_entries, num_entries_per_row, alignment);
      coo.resize(num_rows, num_cols, num_coo_entries);
    }

    /*! Swap the contents of two \p hyb_matrix objects.
     *
     *  \param matrix Another \p hyb_matrix with the same IndexType and ValueType.
     */
    void swap(hyb_matrix& matrix)
    {
      Parent::swap(matrix);
      ell.swap(matrix.ell);
      coo.swap(matrix.coo);
    }
    
    /*! Assignment from another matrix.
     *
     *  \param matrix Another sparse or dense matrix.
     */
    template <typename MatrixType>
    hyb_matrix& operator=(const MatrixType& matrix);
}; // class hyb_matrix
/*! \}
 */


/*! \addtogroup sparse_matrix_views Sparse Matrix Views
 *  \ingroup sparse_matrices
 *  \{
 */

/*! \p hyb_matrix_view : Hybrid ELL/COO matrix view
 *
 * \tparam Matrix1 Type of \c ell
 * \tparam Matrix2 Type of \c coo
 * \tparam IndexType Type used for matrix indices (e.g. \c int).
 * \tparam ValueType Type used for matrix values (e.g. \c float).
 * \tparam MemorySpace A memory space (e.g. \c cusp::host_memory or cusp::device_memory)
 *
 */
template <typename Matrix1,
          typename Matrix2,
          typename IndexType   = typename Matrix1::index_type,
          typename ValueType   = typename Matrix1::value_type,
          typename MemorySpace = typename cusp::minimum_space<typename Matrix1::memory_space, typename Matrix2::memory_space>::type >
class hyb_matrix_view : public detail::matrix_base<IndexType,ValueType,MemorySpace,cusp::hyb_format>
{
  typedef cusp::detail::matrix_base<IndexType,ValueType,MemorySpace,cusp::hyb_format> Parent;
  public:
    /*! type of \p ELL portion of the HYB structure
     */
    typedef Matrix1 ell_matrix_type;
    
    /*! type of \p COO portion of the HYB structure 
     */
    typedef Matrix2 coo_matrix_type;

    /*! equivalent container type
     */
    typedef typename cusp::hyb_matrix<IndexType, ValueType, MemorySpace> container;

    /*! equivalent view type
     */
    typedef typename cusp::hyb_matrix_view<Matrix1, Matrix2, IndexType, ValueType, MemorySpace> view;

    /*! View to the \p ELL portion of the HYB structure.
     */
    ell_matrix_type ell;
    
    /*! View to the \p COO portion of the HYB structure.
     */
    coo_matrix_type coo;

    /*! Construct an empty \p hyb_matrix_view.
     */
    hyb_matrix_view() {}

    template <typename OtherMatrix1, typename OtherMatrix2>
    hyb_matrix_view(OtherMatrix1& ell, OtherMatrix2& coo)
    : Parent(ell.num_rows, ell.num_cols, ell.num_entries + coo.num_entries), ell(ell), coo(coo) {}
    
    template <typename OtherMatrix1, typename OtherMatrix2>
    hyb_matrix_view(const OtherMatrix1& ell, const OtherMatrix2& coo)
    : Parent(ell.num_rows, ell.num_cols, ell.num_entries + coo.num_entries), ell(ell), coo(coo) {}

    template <typename Matrix>
    hyb_matrix_view(Matrix& A)
    : Parent(A), ell(A.ell), coo(A.coo) {}
    
    template <typename Matrix>
    hyb_matrix_view(const Matrix& A)
    : Parent(A), ell(A.ell), coo(A.coo) {}

    /*! Resize matrix dimensions and underlying storage
     */
    void resize(size_t num_rows, size_t num_cols,
                size_t num_ell_entries, size_t num_coo_entries,
                size_t num_entries_per_row, size_t alignment = 32)
    {
      Parent::resize(num_rows, num_cols, num_ell_entries + num_coo_entries);
      ell.resize(num_rows, num_cols, num_ell_entries, num_entries_per_row, alignment);
      coo.resize(num_rows, num_cols, num_coo_entries);
    }
};
/*! \} // end Views
 */

} // end namespace cusp

#include <cusp/detail/hyb_matrix.inl>

