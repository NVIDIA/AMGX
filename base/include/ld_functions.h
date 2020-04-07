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

namespace amgx
{

#if defined(_WIN64) || defined(__LP64__)
// 64-bit pointer operand constraint for inlined asm
#define _ASM_PTR_ "l"
#else
// 32-bit pointer operand constraint for inlined asm
#define _ASM_PTR_ "r"
#endif


__device__   __inline__ double ld_cg(const double *address)
{
    double reg;
    asm("ld.global.cg.f64 %0, [%1];" : "=d"(reg) : _ASM_PTR_(address));
    return reg;
}

__device__   __inline__ float ld_cg(const float *address)
{
    float reg;
    asm("ld.global.cg.f32 %0, [%1];" : "=f"(reg) : _ASM_PTR_(address));
    return reg;
}

__device__  __inline__ int ld_cg(const int *address)
{
    int reg;
    asm("ld.global.cg.s32 %0, [%1];" : "=r"(reg) : _ASM_PTR_(address));
    return reg;
}

__device__   __inline__ double ld_ca(const double *address)
{
    double reg;
    asm("ld.global.ca.f64 %0, [%1];" : "=d"(reg) : _ASM_PTR_(address));
    return reg;
}

__device__   __inline__ float ld_ca(const float *address)
{
    float reg;
    asm("ld.global.ca.f32 %0, [%1];" : "=f"(reg) : _ASM_PTR_(address));
    return reg;
}

__device__  __inline__ int ld_ca(const int *address)
{
    int reg;
    asm("ld.global.ca.s32 %0, [%1];" : "=r"(reg) : _ASM_PTR_(address));
    return reg;
}

__device__   __inline__ double ld_cs(const double *address)
{
    double reg;
    asm("ld.global.cs.f64 %0, [%1];" : "=d"(reg) : _ASM_PTR_(address));
    return reg;
}

__device__   __inline__ float ld_cs(const float *address)
{
    float reg;
    asm("ld.global.cs.f32 %0, [%1];" : "=f"(reg) : _ASM_PTR_(address));
    return reg;
}

__device__  __inline__ int ld_cs(const int *address)
{
    int reg;
    asm("ld.global.cs.s32 %0, [%1];" : "=r"(reg) : _ASM_PTR_(address));
    return reg;
}

#if defined(__CUDA_ARCH__) & (__CUDA_ARCH__ < 350)
template <class T>
__device__ __inline T ldg(const T *address)
{
    return ld_cg(address);
}
#else
template <class T>
__device__ __inline T ldg(const T *address)
{
    return __ldg(address);
}
#endif

} //end namespace amgx
