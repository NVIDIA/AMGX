/*
 *  Copyright 2008-2009 NVIDIA Corporation
 *
 *  Licensed under the Apache License, Version 2.0 (the "License");
 *  you may not use this file except in compliance with the License.
 *  You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 *  Unless required by applicable law or agreed to in writing, software
 *  distributed under the License is distributed on an "AS IS" BASIS,
 *  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  See the License for the specific language governing permissions and
 *  limitations under the License.
 */

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
