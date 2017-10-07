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
    
#include <cusp/detail/host/transpose.h>
#include <cusp/detail/device/transpose.h>

namespace cusp
{
namespace detail
{
namespace dispatch
{

////////////////
// Host Paths //
////////////////
template <typename MatrixType1,   typename MatrixType2>
void transpose(const MatrixType1& A, MatrixType2& At,
               cusp::host_memory)
{
    cusp::detail::host::transpose(A, At,
                            	  typename MatrixType1::format(),
                            	  typename MatrixType2::format());
}

//////////////////
// Device Paths //
//////////////////
template <typename MatrixType1,   typename MatrixType2>
void transpose(const MatrixType1& A, MatrixType2& At,
               cusp::device_memory)
{
    cusp::detail::device::transpose(A, At,
                            	    typename MatrixType1::format(),
                            	    typename MatrixType2::format());
}

} // end namespace dispatch
} // end namespace detail
} // end namespace cusp

