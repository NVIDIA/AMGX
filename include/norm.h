// SPDX-FileCopyrightText: 2011 - 2025 NVIDIA CORPORATION. All Rights Reserved.
//
// SPDX-License-Identifier: BSD-3-Clause

#pragma once
#include <getvalue.h>
#include <error.h>
#include <types.h>
#include <basic_types.h>
#include <vector.h>

#include <amgx_types/util.h>

namespace amgx
{

/**********************************************************
 * Returns the norm of a vector
 *********************************************************/
template<class VectorType, class MatrixType>
typename types::PODTypes<typename VectorType::value_type>::type get_norm(const MatrixType &A, const VectorType &r, const NormType norm_type, typename types::PODTypes<typename VectorType::value_type>::type norm_factor = 1.0);

template <class VectorType, class MatrixType, class PlainVectorType>
void get_norm(const MatrixType &A, const VectorType &r, const int block_size, const NormType norm_type, PlainVectorType &block_nrm, typename types::PODTypes<typename VectorType::value_type>::type norm_factor = 1.0);

template <class VectorType, class MatrixType>
void compute_norm_factor(MatrixType &A, VectorType &b, VectorType &x, const NormType normType, typename types::PODTypes<typename VectorType::value_type>::type &normFactor);

} // namespace amgx

