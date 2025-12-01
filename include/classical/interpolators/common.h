// SPDX-FileCopyrightText: 2011 - 2025 NVIDIA CORPORATION. All Rights Reserved.
//
// SPDX-License-Identifier: BSD-3-Clause

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
