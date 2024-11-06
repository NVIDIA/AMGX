// SPDX-FileCopyrightText: 2008 - 2024 NVIDIA CORPORATION. All Rights Reserved.
//
// SPDX-License-Identifier: BSD-3-Clause

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

    size_t num_entries = amgx::thrust::unique(amgx::thrust::make_zip_iterator(amgx::thrust::make_tuple(coo.row_indices.begin(), coo.column_indices.begin())),
                                        amgx::thrust::make_zip_iterator(amgx::thrust::make_tuple(coo.row_indices.end(),   coo.column_indices.end())))
                         - amgx::thrust::make_zip_iterator(amgx::thrust::make_tuple(coo.row_indices.begin(), coo.column_indices.begin()));

    coo.resize(num_rows, num_cols, num_entries);
    
    output = coo;
}
/*! \}
 */

} // end namespace gallery
} // end namespace cusp

