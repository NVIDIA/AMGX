// SPDX-FileCopyrightText: 2008 - 2025 NVIDIA CORPORATION. All Rights Reserved.
//
// SPDX-License-Identifier: BSD-3-Clause

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

