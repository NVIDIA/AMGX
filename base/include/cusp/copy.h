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

/*! \file copy.h
 *  \brief Performs (deep) copy operations between containers and views.
 */
    
#pragma once

#include <cusp/detail/config.h>

namespace cusp
{

/*! \addtogroup algorithms Algorithms
 *  \ingroup algorithms
 *  \{
 */

/*! \p copy : Copy one array or matrix to another
 *
 * \note T1 and T2 must have the same format type
 * \note T2 will be resized as necessary
 *
 * \see \p convert
 */
template <typename T1, typename T2>
void copy(const T1& src, T2& dst);

/*! \}
 */

} // end namespace cusp

#include <cusp/detail/copy.inl>

