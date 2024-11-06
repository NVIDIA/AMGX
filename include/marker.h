// SPDX-FileCopyrightText: 2013 - 2024 NVIDIA CORPORATION. All Rights Reserved.
//
// SPDX-License-Identifier: BSD-3-Clause

template<int i> __global__ void marker_kernel() {}

template<int i> void marker()
{
    marker_kernel<i> <<< 1, 1>>>();
}

