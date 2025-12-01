// SPDX-FileCopyrightText: 2008 - 2025 NVIDIA CORPORATION. All Rights Reserved.
//
// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#include <cusp/hyb_matrix.h>

#include <cusp/detail/device/spmv/ell.h>
#include <cusp/detail/device/spmv/coo.h>

namespace cusp
{
namespace detail
{
namespace device
{

// SpMV kernels for the hybrid ELL/COO matrix format.
template <typename IndexType, typename ValueType>
void spmv_hyb(const cusp::hyb_matrix<IndexType, ValueType, cusp::device_memory>& hyb, 
              const ValueType * x, 
                    ValueType * y)
{
    cusp::detail::device::spmv(hyb.ell, x, y);
    cusp::detail::device::spmv(hyb.coo, x, y);
}

template <typename IndexType, typename ValueType>
void spmv_hyb_tex(const cusp::hyb_matrix<IndexType, ValueType, cusp::device_memory>& hyb, 
                  const ValueType * x, 
                        ValueType * y)
{
    cusp::detail::device::spmv_tex(hyb.ell, x, y);
    cusp::detail::device::spmv_tex(hyb.coo, x, y);
}

    
template <typename IndexType, typename ValueType>
void spmv(const cusp::hyb_matrix<IndexType, ValueType, cusp::device_memory>& hyb, 
          const ValueType * x, 
                ValueType * y)
{
    spmv_hyb(hyb, x, y);
}

template <typename IndexType, typename ValueType>
void spmv_tex(const cusp::hyb_matrix<IndexType, ValueType, cusp::device_memory>& hyb, 
              const ValueType * x, 
                    ValueType * y)
{
    spmv_hyb_tex(hyb, x, y);
}

} // end namespace device
} // end namespace detail
} // end namespace cusp

