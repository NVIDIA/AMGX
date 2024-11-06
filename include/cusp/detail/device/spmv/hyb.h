// SPDX-FileCopyrightText: 2008 - 2024 NVIDIA CORPORATION. All Rights Reserved.
//
// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#include <cusp/detail/device/spmv/ell.h>
#include <cusp/detail/device/spmv/coo_flat.h>

namespace cusp
{
namespace detail
{
namespace device
{

template <typename Matrix,
          typename ValueType>
void spmv_hyb(const Matrix&    A, 
              const ValueType* x, 
                    ValueType* y)
{
    spmv_ell(A.ell, x, y);
    __spmv_coo_flat<false, false>(A.coo, x, y);
}

template <typename Matrix,
          typename ValueType>
void spmv_hyb_tex(const Matrix&    A,
                  const ValueType* x, 
                        ValueType* y)
{
    spmv_ell_tex(A.ell, x, y);
    __spmv_coo_flat<true, false>(A.coo, x, y);
}

} // end namespace device
} // end namespace detail
} // end namespace cusp

