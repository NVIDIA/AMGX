// SPDX-FileCopyrightText: 2008 - 2024 NVIDIA CORPORATION. All Rights Reserved.
//
// SPDX-License-Identifier: BSD-3-Clause

#pragma once

namespace cusp
{
namespace detail
{

template <typename IntegralType>
IntegralType round_up(IntegralType n, IntegralType k)
{
    return k * ((n + k - 1) / k);
}

} // end namespace detail
} // end namespace cusp

