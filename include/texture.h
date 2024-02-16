// SPDX-FileCopyrightText: 2011 - 2024 NVIDIA CORPORATION. All Rights Reserved.
//
// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#include <error.h>
#include <cutil.h>

#include "cuda_runtime.h"

namespace amgx
{


template <typename T_ELEM> __inline__ __device__ T_ELEM __cachingLoad(const T_ELEM *addr)
{
    return __ldg(addr);
}

}
