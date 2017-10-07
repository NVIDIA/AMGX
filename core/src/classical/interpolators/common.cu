/* Copyright (c) 2011-2017, NVIDIA CORPORATION. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *  * Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *  * Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *  * Neither the name of NVIDIA CORPORATION nor the names of its
 *    contributors may be used to endorse or promote products derived
 *    from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
 * EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 * PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
 * CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 * EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 * PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
 * OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

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
