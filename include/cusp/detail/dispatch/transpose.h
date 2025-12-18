// SPDX-FileCopyrightText: 2008 - 2025 NVIDIA CORPORATION. All Rights Reserved.
//
// SPDX-License-Identifier: BSD-3-Clause

#include <cusp/detail/host/transpose.h>
#include <cusp/detail/device/transpose.h>

namespace cusp
{
namespace detail
{
namespace dispatch
{

////////////////
// Host Paths //
////////////////
template <typename MatrixType1,   typename MatrixType2>
void transpose(const MatrixType1& A, MatrixType2& At,
               cusp::host_memory)
{
    cusp::detail::host::transpose(A, At,
                            	  typename MatrixType1::format(),
                            	  typename MatrixType2::format());
}

//////////////////
// Device Paths //
//////////////////
template <typename MatrixType1,   typename MatrixType2>
void transpose(const MatrixType1& A, MatrixType2& At,
               cusp::device_memory)
{
    cusp::detail::device::transpose(A, At,
                            	    typename MatrixType1::format(),
                            	    typename MatrixType2::format());
}

} // end namespace dispatch
} // end namespace detail
} // end namespace cusp

