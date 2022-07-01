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

namespace cusp
{

template <typename T, typename MemorySpace>                             class array1d;
template <typename T, typename MemorySpace, typename Orientation>       class array2d;
template <typename IndexType, typename ValueType, typename MemorySpace> class coo_matrix;
template <typename IndexType, typename ValueType, typename MemorySpace> class csr_matrix;
template <typename IndexType, typename ValueType, typename MemorySpace> class dia_matrix;
template <typename IndexType, typename ValueType, typename MemorySpace> class ell_matrix;
template <typename IndexType, typename ValueType, typename MemorySpace> class hyb_matrix;

} // end namespace cusp

