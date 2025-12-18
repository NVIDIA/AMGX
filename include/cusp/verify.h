// SPDX-FileCopyrightText: 2008 - 2025 NVIDIA CORPORATION. All Rights Reserved.
//
// SPDX-License-Identifier: BSD-3-Clause

/*! \file verify.h
 *  \brief Validate matrix format
 */

#pragma once

#include <cusp/detail/config.h>

namespace cusp
{

template <typename MatrixType>
bool is_valid_matrix(const MatrixType& A);

template <typename MatrixType, typename OutputStream>
bool is_valid_matrix(const MatrixType& A, OutputStream& ostream);

template <typename MatrixType>
void assert_is_valid_matrix(const MatrixType& A);

} // end namespace cusp

#include <cusp/detail/verify.inl>

