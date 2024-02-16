// SPDX-FileCopyrightText: 2013 - 2024 NVIDIA CORPORATION. All Rights Reserved.
//
// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#include <error.h>

namespace amgx
{
namespace eigensolvers
{
AMGX_ERROR initialize();
void finalize();
} //namespace eigensolvers
} // namespace amgx
