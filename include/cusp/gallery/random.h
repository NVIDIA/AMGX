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

/*! \file random.h
 *  \brief Random matrix generators
 */

#pragma once

#include <cusp/detail/config.h>

#include <cusp/coo_matrix.h>
#include <thrust/unique.h>
#include <thrust/sort.h>

#include <stdlib.h> // XXX remove when we switch RNGs

namespace cusp
{
namespace gallery
{
/*! \addtogroup gallery Matrix Gallery
 *  \ingroup gallery
 *  \{
 */

// TODO use thrust RNGs, add seed parameter defaulting to num_rows ^ num_cols ^ num_samples
// TODO document
template <class MatrixType>
void random(size_t num_rows, size_t num_cols, size_t num_samples, MatrixType& output)
{
    typedef typename MatrixType::index_type IndexType;
    typedef typename MatrixType::value_type ValueType;

    cusp::coo_matrix<IndexType,ValueType,cusp::host_memory> coo(num_rows, num_cols, num_samples);

    srand(num_rows ^ num_cols ^ num_samples);

    for(size_t n = 0; n < num_samples; n++)
    {
        coo.row_indices[n]    = rand() % num_rows;
        coo.column_indices[n] = rand() % num_cols;
        coo.values[n]         = ValueType(1);
    }

    // sort indices by (row,column)
    coo.sort_by_row_and_column();

    size_t num_entries = thrust::unique(thrust::make_zip_iterator(thrust::make_tuple(coo.row_indices.begin(), coo.column_indices.begin())),
                                        thrust::make_zip_iterator(thrust::make_tuple(coo.row_indices.end(),   coo.column_indices.end())))
                         - thrust::make_zip_iterator(thrust::make_tuple(coo.row_indices.begin(), coo.column_indices.begin()));

    coo.resize(num_rows, num_cols, num_entries);
    
    output = coo;
}
/*! \}
 */

} // end namespace gallery
} // end namespace cusp

