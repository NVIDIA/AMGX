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

#pragma once
#include <classical/selectors/selector.h>
#include <classical/interpolators/interpolator.h>

namespace amgx
{

/*
 * hold general routines common to both diatance1 and distance2
 */

struct is_non_neg
{
    __host__ __device__
    bool operator()(const int &x)
    {
        return x >= 0;
    }
};

template< typename T >
__device__ __forceinline__
bool
sign( T x ) 
{ 
	return x >= T(0); 
}


__global__
void coarseMarkKernel(int *cf_map, int *mark, const int numEntries);

__global__
void modifyCoarseMapKernel(int *cf_map, int *mark, const int numEntries);

__global__
void nonZerosPerRowKernel(const int num_rows, const int *cf_map, const int *C_hat_start,
                          const int *C_hat_end, int *nonZerosPerRow);

__global__
void nonZerosPerRowSizeKernel(const int num_rows, const int *cf_map, const int *C_hat_size,
                              int *nonZerosPerRow);

} // namespace amgx
