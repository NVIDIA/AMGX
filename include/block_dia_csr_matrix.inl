// SPDX-FileCopyrightText: 2011 - 2025 NVIDIA CORPORATION. All Rights Reserved.
//
// SPDX-License-Identifier: BSD-3-Clause

namespace amgx
{

template <typename T1, typename T2>
void copy_matrix_dimensions(const T1 &src, T2 &dst)
{
    dst.num_block_rows    = src.num_block_rows;
    dst.num_block_cols    = src.num_block_cols;
    dst.num_nonzero_blocks = src.num_nonzero_blocks;
    dst.block_size = src.block_size;
    dst.num_rows = src.num_rows;
    dst.num_cols = src.num_cols;
    dst.num_entries = src.num_entries;
}

//////////////////
// Constructors //
//////////////////

// construct from a different matrix
template <typename IndexType, typename ValueType, class MemorySpace>
template <typename MatrixType>
block_dia_csr_matrix<IndexType, ValueType, MemorySpace>
::block_dia_csr_matrix(const MatrixType &matrix)
{
    convertToBlockDiaCsr(matrix, *this);
}

//////////////////////
// Member Functions //
//////////////////////


// assignment from another matrix
template <typename IndexType, typename ValueType, class MemorySpace>
template <typename MatrixType>
block_dia_csr_matrix<IndexType, ValueType, MemorySpace> &
block_dia_csr_matrix<IndexType, ValueType, MemorySpace>
::operator=(const MatrixType &matrix)
{
    convertToBlockDiaCsr(matrix, *this);
    return *this;
}


// Entry point for convert
template <typename IndexType, typename ValueType, class MemorySpace>
template <typename Matrix1, typename Matrix2>
void block_dia_csr_matrix<IndexType, ValueType, MemorySpace>
::convertToBlockDiaCsr(const Matrix1 &src, Matrix2 &dst)
{
    convertToBlockDiaCsr(src, dst,
                         typename Matrix1::format(),
                         typename Matrix2::format(),
                         typename Matrix1::memory_space(),
                         typename Matrix2::memory_space());
}


// COPY
template <typename IndexType, typename ValueType, class MemorySpace>
template <typename SourceType, typename DestinationType,
          typename MemorySpace1, typename MemorySpace2>
void block_dia_csr_matrix<IndexType, ValueType, MemorySpace>
::convertToBlockDiaCsr(const SourceType &src, DestinationType &dst,
                       amgx::block_dia_csr_format, amgx::block_dia_csr_format,
                       const MemorySpace1 &mem1, const MemorySpace2 &mem2)
{
    amgx::copy_matrix_dimensions(src, dst);
    dst.row_offsets = src.row_offsets;
    dst.column_indices = src.column_indices;
    dst.dia_values = src.dia_values;
    dst.nonzero_values = src.nonzero_values;
}

// HOST CSR to blockCSR
template <typename IndexType, typename ValueType, class MemorySpace>
template <typename Matrix1, typename Matrix2>
void block_dia_csr_matrix<IndexType, ValueType, MemorySpace>
::convertToBlockDiaCsr(const Matrix1 &src, Matrix2 &dst,
                       cusp::csr_format, amgx::block_dia_csr_format,
                       host_memory, host_memory)
{
    std::cout << "Haven't implemented method to convert from serial CSR to block CSR on host, exiting" << std::endl;
    exit(EXIT_FAILURE);
}

// DEVICE CSR to blockCSR
template <typename IndexType, typename ValueType, class MemorySpace>
template <typename Matrix1, typename Matrix2>
void block_dia_csr_matrix<IndexType, ValueType, MemorySpace>
::convertToBlockDiaCsr(const Matrix1 &src, Matrix2 &dst,
                       cusp::csr_format, amgx::block_dia_csr_format,
                       device_memory, device_memory)
{
    std::cout << "Haven't implemented method to convert from serial CSR to block CSR on device, exiting" << std::endl;
    exit(EXIT_FAILURE);
}

// DEFAULT
template <typename IndexType, typename ValueType, class MemorySpace>
template <typename SourceType, typename DestinationType,
          typename T1, typename T2,
          typename MemorySpace1, typename MemorySpace2>
void block_dia_csr_matrix<IndexType, ValueType, MemorySpace>
::convertToBlockDiaCsr(const SourceType &src, DestinationType &dst,
                       const T1 &t1, T2 &t2,
                       const MemorySpace1 &mem1, const MemorySpace2 &mem2)
{
    std::cout << "Input format must be CSR, exiting(1)" << std::endl;
    exit(EXIT_FAILURE);
}


// Print
template <typename IndexType, typename ValueType, class MemorySpace>
void block_dia_csr_matrix<IndexType, ValueType, MemorySpace>::print(const char *label) const
{
    std::cout << label << ": " << std::endl;

    for (int i = 0; i < num_block_rows; i++)
    {
        std::cout << i << " " << i << std::endl;
        // Print diagonal block
        int icount = 0;

        for (int m = 0; m < block_size; m++)
        {
            for (int n = 0; n < block_size; n++)
            {
                std::cout << std::scientific << std::setw(10) << std::setprecision(3) << dia_values[i * block_size * block_size + icount] << " ";
                icount++;
            }

            std::cout << std::endl;
        }

        // Print off-diagonal blocks
        for (int r = row_offsets[i]; r < row_offsets[i + 1]; r++)
        {
            int j = column_indices[r];
            std::cout << i << " " << j << std::endl;
            int icount = 0;

            for (int m = 0; m < block_size; m++)
            {
                for (int n = 0; n < block_size; n++)
                {
                    std::cout << std::scientific << std::setw(10) << std::setprecision(3) << nonzero_values[r * block_size * block_size + icount] << " ";
                    icount++;
                }

                std::cout << std::endl;
            }
        }
    }
}



} // End of namespace amgx
