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

#include <cusp/hyb_matrix.h>

#include <cusp/detail/device/spmv/ell.h>
#include <cusp/detail/device/spmv/coo.h>

namespace cusp
{
namespace detail
{
namespace device
{

// SpMV kernels for the hybrid ELL/COO matrix format.
template <typename IndexType, typename ValueType>
void spmv_hyb(const cusp::hyb_matrix<IndexType, ValueType, cusp::device_memory>& hyb, 
              const ValueType * x, 
                    ValueType * y)
{
    cusp::detail::device::spmv(hyb.ell, x, y);
    cusp::detail::device::spmv(hyb.coo, x, y);
}

template <typename IndexType, typename ValueType>
void spmv_hyb_tex(const cusp::hyb_matrix<IndexType, ValueType, cusp::device_memory>& hyb, 
                  const ValueType * x, 
                        ValueType * y)
{
    cusp::detail::device::spmv_tex(hyb.ell, x, y);
    cusp::detail::device::spmv_tex(hyb.coo, x, y);
}

    
template <typename IndexType, typename ValueType>
void spmv(const cusp::hyb_matrix<IndexType, ValueType, cusp::device_memory>& hyb, 
          const ValueType * x, 
                ValueType * y)
{
    spmv_hyb(hyb, x, y);
}

template <typename IndexType, typename ValueType>
void spmv_tex(const cusp::hyb_matrix<IndexType, ValueType, cusp::device_memory>& hyb, 
              const ValueType * x, 
                    ValueType * y)
{
    spmv_hyb_tex(hyb, x, y);
}

} // end namespace device
} // end namespace detail
} // end namespace cusp

