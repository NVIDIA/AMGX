// SPDX-FileCopyrightText: 2008 - 2025 NVIDIA CORPORATION. All Rights Reserved.
//
// SPDX-License-Identifier: BSD-3-Clause

#include <cusp/convert.h>
#include <cusp/ell_matrix.h>
#include <cusp/coo_matrix.h>

namespace cusp
{

//////////////////
// Constructors //
//////////////////
        
// construct from another matrix
template <typename IndexType, typename ValueType, class MemorySpace>
template <typename MatrixType>
hyb_matrix<IndexType,ValueType,MemorySpace>
    ::hyb_matrix(const MatrixType& matrix)
    {
        cusp::convert(matrix, *this);
    }

//////////////////////
// Member Functions //
//////////////////////
        
template <typename IndexType, typename ValueType, class MemorySpace>
template <typename MatrixType>
    hyb_matrix<IndexType,ValueType,MemorySpace>&
    hyb_matrix<IndexType,ValueType,MemorySpace>
    ::operator=(const MatrixType& matrix)
    {
        cusp::convert(matrix, *this);
        
        return *this;
    }

} // end namespace cusp

