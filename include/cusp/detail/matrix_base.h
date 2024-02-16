// SPDX-FileCopyrightText: 2008 - 2024 NVIDIA CORPORATION. All Rights Reserved.
//
// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#include <cusp/detail/config.h>

#include <thrust/swap.h>

namespace cusp
{
namespace detail
{

    template<typename IndexType, typename ValueType, typename MemorySpace, typename Format>
    class matrix_base
    {
        public:
            typedef IndexType   index_type;
            typedef ValueType   value_type;
            typedef Format      format;
            typedef MemorySpace memory_space;

            size_t num_rows;
            size_t num_cols;
            size_t num_entries;
            
            matrix_base()
                : num_rows(0), num_cols(0), num_entries(0) {}
            
            template <typename Matrix>
            matrix_base(const Matrix& m)
                : num_rows(m.num_rows), num_cols(m.num_cols), num_entries(m.num_entries) {}

            matrix_base(size_t rows, size_t cols)
                : num_rows(rows), num_cols(cols), num_entries(0) {}

            matrix_base(size_t rows, size_t cols, size_t entries)
                : num_rows(rows), num_cols(cols), num_entries(entries) {}

            void resize(size_t rows, size_t cols, size_t entries)
            {
                num_rows = rows;
                num_cols = cols;
                num_entries = entries;
            }

            void swap(matrix_base& base)
            {
                amgx::thrust::swap(num_rows,    base.num_rows);
                amgx::thrust::swap(num_cols,    base.num_cols);
                amgx::thrust::swap(num_entries, base.num_entries);
            }
    };

} // end namespace detail
} // end namespace cusp

