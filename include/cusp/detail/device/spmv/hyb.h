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

#include <cusp/detail/device/spmv/ell.h>
#include <cusp/detail/device/spmv/coo_flat.h>

namespace cusp
{
namespace detail
{
namespace device
{

template <typename Matrix,
          typename ValueType>
void spmv_hyb(const Matrix&    A, 
              const ValueType* x, 
                    ValueType* y)
{
    spmv_ell(A.ell, x, y);
    __spmv_coo_flat<false, false>(A.coo, x, y);
}

template <typename Matrix,
          typename ValueType>
void spmv_hyb_tex(const Matrix&    A,
                  const ValueType* x, 
                        ValueType* y)
{
    spmv_ell_tex(A.ell, x, y);
    __spmv_coo_flat<true, false>(A.coo, x, y);
}

} // end namespace device
} // end namespace detail
} // end namespace cusp

