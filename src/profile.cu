// SPDX-FileCopyrightText: 2011 - 2024 NVIDIA CORPORATION. All Rights Reserved.
//
// SPDX-License-Identifier: BSD-3-Clause

#include <error.h>
#include <global_thread_handle.h>

__global__ void profileLevelUp_kernel() {}
__global__ void profileLevelDown_kernel() {}
__global__ void profileLevelZero_kernel() {}
__global__ void profilePhaseSetup_kernel() {}
__global__ void profilePhaseSolve_kernel() {}
__global__ void profilePhaseNone_kernel() {}
__global__ void profileSubphaseMatrixColoring_kernel() {}
__global__ void profileSubphaseSmootherSetup_kernel() {}
__global__ void profileSubphaseFindAggregates_kernel() {}
__global__ void profileSubphaseComputeRestriction_kernel() {}
__global__ void profileSubphaseComputeCoarseA_kernel() {}
__global__ void profileSubphaseNone_kernel() {}
__global__ void profileSubphaseTruncateP_kernel() {}

namespace amgx
{

void profileLevelUp()
{
    profileLevelUp_kernel <<< 1, 1, 0, thrust::global_thread_handle::get_stream()>>>();
    cudaCheckError();
}
void profileLevelDown()
{
    profileLevelDown_kernel <<< 1, 1, 0, thrust::global_thread_handle::get_stream()>>>();
    cudaCheckError();
}
void profileLevelZero()
{
    profileLevelZero_kernel <<< 1, 1, 0, thrust::global_thread_handle::get_stream()>>>();
    cudaCheckError();
}
void profilePhaseSetup()
{
    profilePhaseSetup_kernel <<< 1, 1, 0, thrust::global_thread_handle::get_stream()>>>();
    cudaCheckError();
}
void profilePhaseSolve()
{
    profilePhaseSolve_kernel <<< 1, 1, 0, thrust::global_thread_handle::get_stream()>>>();
    cudaCheckError();
}
void profilePhaseNone()
{
    profilePhaseNone_kernel <<< 1, 1, 0, thrust::global_thread_handle::get_stream()>>>();
    cudaCheckError();
}

void profileSubphaseMatrixColoring()
{
    profileSubphaseMatrixColoring_kernel <<< 1, 1, 0, thrust::global_thread_handle::get_stream()>>>();
    cudaCheckError();
}

void profileSubphaseSmootherSetup()
{
    profileSubphaseSmootherSetup_kernel <<< 1, 1, 0, thrust::global_thread_handle::get_stream()>>>();
    cudaCheckError();
}

void profileSubphaseFindAggregates()
{
    profileSubphaseFindAggregates_kernel <<< 1, 1, 0, thrust::global_thread_handle::get_stream()>>>();
    cudaCheckError();
}

void profileSubphaseComputeRestriction()
{
    profileSubphaseComputeRestriction_kernel <<< 1, 1, 0, thrust::global_thread_handle::get_stream()>>>();
    cudaCheckError();
}

void profileSubphaseComputeCoarseA()
{
    profileSubphaseComputeCoarseA_kernel <<< 1, 1, 0, thrust::global_thread_handle::get_stream()>>>();
    cudaCheckError();
}

void profileSubphaseNone()
{
    profileSubphaseNone_kernel <<< 1, 1, 0, thrust::global_thread_handle::get_stream()>>>();
    cudaCheckError();
}

void profileSubphaseTruncateP()
{
    profileSubphaseTruncateP_kernel <<< 1, 1, 0, thrust::global_thread_handle::get_stream()>>>();
    cudaCheckError();
}

} // namespace amgx

