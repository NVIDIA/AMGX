// SPDX-FileCopyrightText: 2008 - 2024 NVIDIA CORPORATION. All Rights Reserved.
//
// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#include <cusp/detail/device/spmv/csr_scalar.h>
#include <cusp/detail/device/spmv/csr_vector.h>

namespace cusp
{
namespace detail
{
namespace device
{

template <typename IndexType, typename ValueType>
void spmv(const csr_matrix<IndexType,ValueType,cusp::device_memory>& csr, 
          const ValueType * x, 
                ValueType * y)
{ 
    spmv_csr_vector(csr, x, y);
}

template <typename IndexType, typename ValueType>
void spmv_tex(const csr_matrix<IndexType,ValueType,cusp::device_memory>& csr, 
              const ValueType * x, 
                    ValueType * y)
{ 
    spmv_csr_vector_tex(csr, x, y);
}

} // end namespace device
} // end namespace detail
} // end namespace cusp

