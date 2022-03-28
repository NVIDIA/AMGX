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

#include <cusp/detail/device/spmv/csr_scalar.h>
#include <cusp/detail/device/spmv/csr_vector.h>

namespace cusp
{
namespace detail
{
namespace device
{

template <typename IndexType, typename ValueType>
void spmv(const csr_matrix<IndexType,ValueType,cusp::device_memory>& csr, 
          const ValueType * x, 
                ValueType * y)
{ 
    spmv_csr_vector(csr, x, y);
}

template <typename IndexType, typename ValueType>
void spmv_tex(const csr_matrix<IndexType,ValueType,cusp::device_memory>& csr, 
              const ValueType * x, 
                    ValueType * y)
{ 
    spmv_csr_vector_tex(csr, x, y);
}

} // end namespace device
} // end namespace detail
} // end namespace cusp

