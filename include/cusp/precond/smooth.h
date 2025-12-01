// SPDX-FileCopyrightText: 2008 - 2025 NVIDIA CORPORATION. All Rights Reserved.
//
// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#include <cusp/detail/config.h>

#include <cusp/csr_matrix.h>
#include <cusp/coo_matrix.h>

namespace cusp
{
namespace precond
{

	//   Smoothed (final) prolongator defined by P = (I - omega/rho(K) K) * T
	//   where K = diag(S)^-1 * S and rho(K) is an approximation to the 
	//   spectral radius of K.
	template <typename IndexType, typename ValueType, typename MemorySpace>
	void smooth_prolongator(const cusp::coo_matrix<IndexType,ValueType,MemorySpace>& S,
				const cusp::coo_matrix<IndexType,ValueType,MemorySpace>& T,
				      cusp::coo_matrix<IndexType,ValueType,MemorySpace>& P,
				const ValueType omega = 4.0/3.0,
				const ValueType rho_Dinv_S = 0.0);

	template <typename IndexType, typename ValueType>
	void smooth_prolongator(const cusp::csr_matrix<IndexType,ValueType,cusp::host_memory>& S,
                        	const cusp::csr_matrix<IndexType,ValueType,cusp::host_memory>& T,
                              	      cusp::csr_matrix<IndexType,ValueType,cusp::host_memory>& P,
                        	const ValueType omega = 4.0/3.0,
                        	const ValueType rho_Dinv_S = 0.0);

} // end namespace precond
} // end namespace cusp

#include <cusp/precond/detail/smooth.inl>
