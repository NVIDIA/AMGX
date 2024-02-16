// SPDX-FileCopyrightText: 2008 - 2024 NVIDIA CORPORATION. All Rights Reserved.
//
// SPDX-License-Identifier: BSD-3-Clause

/*! \file format.h
 *  \brief Format types
 */

#pragma once

#include <cusp/detail/config.h>

namespace cusp
{

struct known_format {};
struct unknown_format {};

struct dense_format : public known_format {};
struct array1d_format : public dense_format {};
struct array2d_format : public dense_format {};

struct sparse_format : public known_format {};
struct coo_format : public sparse_format {};
struct csr_format : public sparse_format {};
struct dia_format : public sparse_format {};
struct ell_format : public sparse_format {};
struct hyb_format : public sparse_format {};

} // end namespace cusp

