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
                thrust::swap(num_rows, shape.num_rows);
                thrust::swap(num_cols, shape.num_cols);
            }
    };

} // end namespace cusp
