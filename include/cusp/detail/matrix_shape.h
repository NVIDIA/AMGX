// SPDX-FileCopyrightText: 2008 - 2025 NVIDIA CORPORATION. All Rights Reserved.
//
// SPDX-License-Identifier: BSD-3-Clause

#pragma once

namespace cusp
{

    template<typename IndexType>
    class matrix_shape
    {
        public:
            typedef IndexType index_type;

            index_type num_rows;
            index_type num_cols;
            
            matrix_shape()
                : num_rows(0), num_cols(0) {}

            matrix_shape(IndexType rows, IndexType cols)
                : num_rows(rows), num_cols(cols) {}

            void swap(matrix_shape& shape)
            {
                amgx::thrust::swap(num_rows, shape.num_rows);
                amgx::thrust::swap(num_cols, shape.num_cols);
            }
    };

} // end namespace cusp
