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

void profileLevelUp()
{
    profileLevelUp_kernel <<< 1, 1, 0, thrust::global_thread_handle::get_stream()>>>();
}
void profileLevelDown()
{
    profileLevelDown_kernel <<< 1, 1, 0, thrust::global_thread_handle::get_stream()>>>();
}
void profileLevelZero()
{
    profileLevelZero_kernel <<< 1, 1, 0, thrust::global_thread_handle::get_stream()>>>();
}
void profilePhaseSetup()
{
    profilePhaseSetup_kernel <<< 1, 1, 0, thrust::global_thread_handle::get_stream()>>>();
}
void profilePhaseSolve()
{
    profilePhaseSolve_kernel <<< 1, 1, 0, thrust::global_thread_handle::get_stream()>>>();
}
void profilePhaseNone()
{
    profilePhaseNone_kernel <<< 1, 1, 0, thrust::global_thread_handle::get_stream()>>>();
}

void profileSubphaseMatrixColoring()
{
    profileSubphaseMatrixColoring_kernel <<< 1, 1, 0, thrust::global_thread_handle::get_stream()>>>();
}

void profileSubphaseSmootherSetup()
{
    profileSubphaseSmootherSetup_kernel <<< 1, 1, 0, thrust::global_thread_handle::get_stream()>>>();
}

void profileSubphaseFindAggregates()
{
    profileSubphaseFindAggregates_kernel <<< 1, 1, 0, thrust::global_thread_handle::get_stream()>>>();
}

void profileSubphaseComputeRestriction()
{
    profileSubphaseComputeRestriction_kernel <<< 1, 1, 0, thrust::global_thread_handle::get_stream()>>>();
}

void profileSubphaseComputeCoarseA()
{
    profileSubphaseComputeCoarseA_kernel <<< 1, 1, 0, thrust::global_thread_handle::get_stream()>>>();
}

void profileSubphaseNone()
{
    profileSubphaseNone_kernel <<< 1, 1, 0, thrust::global_thread_handle::get_stream()>>>();
}

void profileSubphaseTruncateP()
{
    profileSubphaseTruncateP_kernel <<< 1, 1, 0, thrust::global_thread_handle::get_stream()>>>();
}



