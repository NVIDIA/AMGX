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

#include <thrust/count.h>

namespace cusp
{
namespace precond
{
	template <typename IndexType, typename ValueType, typename MemorySpace,
		  typename ArrayType>
	void mis_to_aggregates(const cusp::coo_matrix<IndexType,ValueType,MemorySpace>& C,
			       const ArrayType& mis,
				     ArrayType& aggregates);

	template <typename IndexType, typename ValueType,
		  typename ArrayType>
	void standard_aggregation(const cusp::coo_matrix<IndexType,ValueType,cusp::device_memory>& C,
					ArrayType& aggregates);

	template <typename IndexType, typename ValueType,
		  typename ArrayType>
	void standard_aggregation(const cusp::csr_matrix<IndexType,ValueType,cusp::host_memory>& C,
					ArrayType& aggregates);

} // end namespace precond
} // end namespace cusp

#include <cusp/precond/detail/aggregate.inl>
