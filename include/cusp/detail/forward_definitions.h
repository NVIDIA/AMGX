// SPDX-FileCopyrightText: 2008 - 2025 NVIDIA CORPORATION. All Rights Reserved.
//
// SPDX-License-Identifier: BSD-3-Clause

#pragma once

namespace cusp
{

template <typename T, typename MemorySpace>                             class array1d;
template <typename T, typename MemorySpace, typename Orientation>       class array2d;
template <typename IndexType, typename ValueType, typename MemorySpace> class coo_matrix;
template <typename IndexType, typename ValueType, typename MemorySpace> class csr_matrix;
template <typename IndexType, typename ValueType, typename MemorySpace> class dia_matrix;
template <typename IndexType, typename ValueType, typename MemorySpace> class ell_matrix;
template <typename IndexType, typename ValueType, typename MemorySpace> class hyb_matrix;

} // end namespace cusp

