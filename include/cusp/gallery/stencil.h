// SPDX-FileCopyrightText: 2008 - 2024 NVIDIA CORPORATION. All Rights Reserved.
//
// SPDX-License-Identifier: BSD-3-Clause

/*! \file stencil.h
 *  \brief Generate sparse matrix from grid stencil
 */

#pragma once

#include <cusp/detail/config.h>

#include <cusp/array1d.h>

namespace cusp
{
namespace gallery
{

template <typename MatrixType,
          typename StencilPoint,
          typename GridDimension>
void generate_matrix_from_stencil(      MatrixType& matrix,
                                  const cusp::array1d<StencilPoint,cusp::host_memory>& stencil,
                                  const GridDimension& grid);
                            
} // end namespace gallery
} // end namespace cusp

#include <cusp/gallery/stencil.inl>

