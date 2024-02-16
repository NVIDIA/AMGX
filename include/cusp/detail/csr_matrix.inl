// SPDX-FileCopyrightText: 2008 - 2024 NVIDIA CORPORATION. All Rights Reserved.
//
// SPDX-License-Identifier: BSD-3-Clause

#include <cusp/convert.h>

namespace cusp
{

//////////////////
// Constructors //
//////////////////
        
// construct from a different matrix
template <typename IndexType, typename ValueType, class MemorySpace>
template <typename MatrixType>
csr_matrix<IndexType,ValueType,MemorySpace>
    ::csr_matrix(const MatrixType& matrix)
    {
        cusp::convert(matrix, *this);
    }

//////////////////////
// Member Functions //
//////////////////////

// assignment from another matrix
template <typename IndexType, typename ValueType, class MemorySpace>
template <typename MatrixType>
    csr_matrix<IndexType,ValueType,MemorySpace>&
    csr_matrix<IndexType,ValueType,MemorySpace>
    ::operator=(const MatrixType& matrix)
    {
        cusp::convert(matrix, *this);
        
        return *this;
    }

} // end namespace cusp

