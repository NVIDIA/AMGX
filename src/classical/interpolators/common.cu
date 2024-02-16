// SPDX-FileCopyrightText: 2011 - 2024 NVIDIA CORPORATION. All Rights Reserved.
//
// SPDX-License-Identifier: BSD-3-Clause

#include <types.h>
#include <classical/interpolators/common.h>

namespace amgx
{

/*
 * Count the # of non-zeros per row
 */
__global__
void nonZerosPerRowKernel(const int num_rows, const int *cf_map, const int *C_hat_start,
                          const int *C_hat_end, int *nonZerosPerRow)
{
    for (int tIdx = threadIdx.x + blockIdx.x * blockDim.x; tIdx < num_rows; tIdx += gridDim.x * blockDim.x)
    {
        int nonZeros = 0;

        if (cf_map[tIdx] >= 0)
        {
            nonZeros = 1;
        }
        else if (cf_map[tIdx] == STRONG_FINE)
        {
            nonZeros = 0;
        }
        else
        {
            nonZeros = C_hat_end[tIdx] - C_hat_start[tIdx];
        }

        nonZerosPerRow[tIdx] = nonZeros;
    }
}

__global__
void nonZerosPerRowSizeKernel(const int num_rows, const int *cf_map,
                              const int *C_hat_size, int *nonZerosPerRow)
{
    for (int tIdx = threadIdx.x + blockIdx.x * blockDim.x; tIdx < num_rows; tIdx += gridDim.x * blockDim.x)
    {
        int nonZeros = 0;

        if (cf_map[tIdx] >= 0)
        {
            nonZeros = 1;
        }
        else if (cf_map[tIdx] == STRONG_FINE)
        {
            nonZeros = 0;
        }
        else
        {
            nonZeros = C_hat_size[tIdx];
        }

        nonZerosPerRow[tIdx] = nonZeros;
    }
}



} // namespace amgx
