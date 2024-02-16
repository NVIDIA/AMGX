// SPDX-FileCopyrightText: 2008 - 2024 NVIDIA CORPORATION. All Rights Reserved.
//
// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#include <cusp/detail/device/spmv/coo_flat.h>

namespace cusp
{
namespace detail
{
namespace device
{

template <typename IndexType, typename ValueType>
void spmv(const coo_matrix<IndexType,ValueType,cusp::device_memory>& coo, 
          const ValueType * x, 
                ValueType * y)
{ 
    spmv_coo_flat(coo, x, y);
}

template <typename IndexType, typename ValueType>
void spmv_tex(const coo_matrix<IndexType,ValueType,cusp::device_memory>& coo, 
              const ValueType * x, 
                    ValueType * y)
{ 
    spmv_coo_flat_tex(coo, x, y);
}

} // end namespace device
} // end namespace detail
} // end namespace cusp

