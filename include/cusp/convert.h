// SPDX-FileCopyrightText: 2008 - 2024 NVIDIA CORPORATION. All Rights Reserved.
//
// SPDX-License-Identifier: BSD-3-Clause

/*! \file convert.h
 *  \brief Matrix format conversion
 */

#pragma once

#include <cusp/detail/config.h>

namespace cusp
{

/*! \addtogroup algorithms Algorithms
 *  \ingroup algorithms
 *  \{
 */

/*! \p copy : Convert between matrix formats
 *
 * \note DestinationType will be resized as necessary
 *
 * \see \p cusp::copy
 */
template <typename SourceType, typename DestinationType>
void convert(const SourceType& src, DestinationType& dst);

/*! \}
 */

} // end namespace cusp

#include <cusp/detail/convert.inl>

