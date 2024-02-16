// SPDX-FileCopyrightText: 2011 - 2024 NVIDIA CORPORATION. All Rights Reserved.
//
// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#include <error.h>

namespace amgx
{

void allocate_resources(size_t pool_size,
                        size_t max_alloc_size,
                        size_t scaling_factor,
                        size_t scaling_threshold,
                        size_t max_size);
void free_resources();

AMGX_ERROR initialize();
void finalize();

} // namespace amgx
