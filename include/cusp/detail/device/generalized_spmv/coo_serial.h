// SPDX-FileCopyrightText: 2008 - 2024 NVIDIA CORPORATION. All Rights Reserved.
//
// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#include <cusp/coo_matrix.h>

namespace cusp
{
namespace detail
{
namespace device
{

// COO format SpMV kernel that uses only one thread
// This is incredibly slow, so it is only useful for testing purposes,
// *extremely* small matrices, or a few elements at the end of a 
// larger matrix

template <typename IndexType, typename ValueType>
__global__ void
spmv_coo_serial_kernel(const IndexType num_nonzeros,
                       const IndexType * I, 
                       const IndexType * J, 
                       const ValueType * V, 
                       const ValueType * x, 
                             ValueType * y)
{
    for(IndexType n = 0; n < num_nonzeros; n++)
    {
        y[I[n]] += V[n] * x[J[n]];
    }
}


template <typename IndexType, typename ValueType>
void spmv_coo_serial_device(const coo_matrix<IndexType,ValueType,cusp::device_memory>& coo, 
                            const ValueType * d_x, 
                                  ValueType * d_y)
{
    const IndexType * I = amgx::thrust::raw_pointer_cast(&coo.row_indices[0]);
    const IndexType * J = amgx::thrust::raw_pointer_cast(&coo.column_indices[0]);
    const ValueType * V = amgx::thrust::raw_pointer_cast(&coo.values[0]);

    spmv_coo_serial_kernel<IndexType,ValueType> <<<1,1>>>
        (coo.num_nonzeros, coo.I, coo.J, coo.V, d_x, d_y);
}

} // end namespace device
} // end namespace detail
} // end namespace cusp

