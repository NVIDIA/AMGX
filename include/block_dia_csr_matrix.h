// SPDX-FileCopyrightText: 2011 - 2024 NVIDIA CORPORATION. All Rights Reserved.
//
// SPDX-License-Identifier: BSD-3-Clause

#pragma once

namespace amgx
{
struct block_dia_csr_format {};

template <typename IndexType, typename ValueType, class MemorySpace> class block_dia_csr_matrix;
}

#include <cusp/detail/config.h>

#include <cusp/format.h>
#include <cusp/detail/matrix_base.h>
#include <iostream>
#include <iomanip>
#include <vector.h>

namespace amgx
{

template <typename IndexType, typename ValueType, class MemorySpace>
class block_dia_csr_matrix : public cusp::detail::matrix_base<IndexType, ValueType, MemorySpace, amgx::block_dia_csr_format>
{
        typedef cusp::detail::matrix_base<IndexType, ValueType, MemorySpace, amgx::block_dia_csr_format> Parent;
    public:
        /*! rebind matrix to a different MemorySpace
         */
        template<typename MemorySpace2>
        struct rebind { typedef amgx::block_dia_csr_matrix<IndexType, ValueType, MemorySpace2> type; };

        /*! type of row offsets indices array
         */
        typedef typename amgx::Vector<IndexType, MemorySpace> row_offsets_array_type;

        /*! type of column indices array
         */
        typedef typename amgx::Vector<IndexType, MemorySpace> column_indices_array_type;

        /*! type of nonzero values array
         */
        typedef typename amgx::Vector<ValueType, MemorySpace> nonzero_values_array_type;

        /*! type of diagonal values array
         */
        typedef typename amgx::Vector<ValueType, MemorySpace> dia_values_array_type;

        /*! equivalent container type
         */
        typedef typename amgx::block_dia_csr_matrix<IndexType, ValueType, MemorySpace> container;

        /*! Storage for the row offsets of the CSR data structure.  Also called the "row pointer" array.
         */
        row_offsets_array_type row_offsets;

        /*! Storage for the column indices of the CSR data structure.
         */
        column_indices_array_type column_indices;

        /*! Storage for the nonzero entries of the CSR data structure.
         */
        nonzero_values_array_type nonzero_values;

        /*! Storage for the nonzero entries of the CSR data structure.
         */
        dia_values_array_type dia_values;

        bool isColored;

        size_t num_colors;

        // Offsets of the sorted_rows_by_color
        Vector<IndexType, host_memory> offsets_rows_per_color;

        // Storage for the indices of a certain color.
        Vector<IndexType, MemorySpace> sorted_rows_by_color;

        // Storage for the color of each row
        Vector<IndexType, MemorySpace> row_colors;

        /*! Construct an empty \p block_dia_csr_matrix.
         */
        block_dia_csr_matrix() {}

        size_t num_block_rows;
        size_t num_block_cols;
        size_t num_nonzero_blocks;
        size_t block_size;

        /*! Construct a \p block_dia_csr_matrix with a specific shape and number of nonzero entries.
         *
         *  \param num_rows Number of rows.
         *  \param num_cols Number of columns.
         *  \param num_nonzeros Number of nonzero matrix entries (not including diagonal entries)
         */
        block_dia_csr_matrix(size_t block_rows, size_t block_cols, size_t nonzero_blocks, size_t bsize)
            : Parent(block_rows * bsize, block_cols * bsize, (nonzero_blocks + block_rows) * bsize * bsize),
              row_offsets(block_rows + 1),
              column_indices(nonzero_blocks),
              nonzero_values(nonzero_blocks * bsize * bsize),
              dia_values(block_rows * bsize * bsize),
              num_block_rows(block_rows),
              num_block_cols(block_cols),
              num_nonzero_blocks(nonzero_blocks),
              block_size(bsize) {}

        /*! Construct a \p block_dia_csr_matrix from another matrix.
         *
         *  \param matrix Another sparse or dense matrix.
         */
        template <typename MatrixType>
        block_dia_csr_matrix(const MatrixType &matrix);

        /*! Resize matrix dimensions and underlying storage
         */
        void resize(size_t block_rows, size_t block_cols, size_t nonzero_blocks, size_t bsize)
        {
            Parent::resize(block_rows * bsize, block_cols * bsize, (nonzero_blocks + block_rows)*bsize * bsize);
            row_offsets.resize(block_rows + 1);
            column_indices.resize(nonzero_blocks);
            nonzero_values.resize(nonzero_blocks * bsize * bsize);
            dia_values.resize(block_rows * bsize * bsize);
            num_block_rows = block_rows;
            num_block_cols = block_cols;
            num_nonzero_blocks = nonzero_blocks;
            block_size = bsize;
        }

        // Function to print itself
        void print(const char *label) const;

        /*! Swap the contents of two \p block_dia_csr_matrix objects.
         *
         *  \param matrix Another \p block_dia_csr_matrix with the same IndexType and ValueType.
         */
        void swap(block_dia_csr_matrix &matrix)
        {
            Parent::swap(matrix);
            row_offsets.swap(matrix.row_offsets);
            column_indices.swap(matrix.column_indices);
            nonzero_values.swap(matrix.nonzero_values);
            dia_values.swap(matrix.dia_values);
        }

        /*! Assignment from another matrix.
         *
         *  \param matrix Another sparse or dense matrix.
         */
        template <typename MatrixType>
        block_dia_csr_matrix &operator=(const MatrixType &matrix);

    private:

        // Entry point for convertToBlockDiaCSR
        template <typename SourceType, typename DestinationType>
        void convertToBlockDiaCsr(const SourceType &src, DestinationType &dst);

        // COPY
        template <typename SourceType, typename DestinationType,
                  typename MemorySpace1, typename MemorySpace2>
        void convertToBlockDiaCsr(const SourceType &src, DestinationType &dst,
                                  amgx::block_dia_csr_format, amgx::block_dia_csr_format,
                                  const MemorySpace1 &mem1, const MemorySpace2 &mem2);

        // Default
        template <typename SourceType, typename DestinationType,
                  typename T1, typename T2,
                  typename MemorySpace1, typename MemorySpace2>
        void convertToBlockDiaCsr(const SourceType &src, DestinationType &dst,
                                  const T1 &t1, T2 &t2,
                                  const MemorySpace1 &mem1, const MemorySpace2 &mem2);

        // HOST CSR to blockCSR
        template <typename Matrix1, typename Matrix2>
        void convertToBlockDiaCsr(const Matrix1 &src, Matrix2 &dst,
                                  cusp::csr_format, amgx::block_dia_csr_format,
                                  host_memory, host_memory);


        // DEVICE CSR to blockCSR
        template <typename Matrix1, typename Matrix2>
        void convertToBlockDiaCsr(const Matrix1 &src, Matrix2 &dst,
                                  cusp::csr_format, amgx::block_dia_csr_format,
                                  device_memory, device_memory);

}; // class block_dia_csr_matrix

} // end namespace amgx


#include <block_dia_csr_matrix.inl>


