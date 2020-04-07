/* Copyright (c) 2013-2017, NVIDIA CORPORATION. All rights reserved.
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
#if (defined(_MSC_VER) && defined(_WIN64)) || defined(__LP64__)
#define __LDG_PTR   "l"
#else
#define __LDG_PTR   "r"
#endif

static __device__ __inline__ int __load_all(const int *ptr) { int ret; asm volatile ("ld.global.ca.s32 %0, [%1];"  : "=r"(ret) : __LDG_PTR (ptr)); return ret; }
static __device__ __inline__ int __load_global(const int *ptr) { int ret; asm volatile ("ld.global.cg.s32 %0, [%1];"  : "=r"(ret) : __LDG_PTR (ptr)); return ret; }
static __device__ __inline__ int __load_streaming(const int *ptr) { int ret; asm volatile ("ld.global.cs.s32 %0, [%1];"  : "=r"(ret) : __LDG_PTR (ptr)); return ret; }
static __device__ __inline__ int __load_lastuse(const int *ptr) { int ret; asm volatile ("ld.global.lu.s32 %0, [%1];"  : "=r"(ret) : __LDG_PTR (ptr)); return ret; }
#if __CUDA_ARCH__ >= 350
static __device__ __inline__ int __load_nc(const int *ptr) { int ret; asm volatile ("ld.global.nc.s32 %0, [%1];"  : "=r"(ret) : __LDG_PTR (ptr)); return ret; }
#else
static __device__ __inline__ int __load_nc(const int *ptr) { int ret; asm volatile ("ld.global.ca.s32 %0, [%1];"  : "=r"(ret) : __LDG_PTR (ptr)); return ret; }
#endif
static __device__ __inline__ int __load_uniform(const int *ptr) { int ret; asm volatile ("ld.global.s32.uniform %0, [%1];"  : "=r"(ret) : __LDG_PTR (ptr)); return ret; }


static __device__ __inline__ int2 __load_all(const int2 *ptr) { int2 ret; asm volatile ("ld.global.ca.v2.s32 {%0,%1}, [%2];"  : "=r"(ret.x), "=r"(ret.y) : __LDG_PTR (ptr)); return ret; }
static __device__ __inline__ int2 __load_global(const int2 *ptr) { int2 ret; asm volatile ("ld.global.cg.v2.s32 {%0,%1}, [%2];"  : "=r"(ret.x), "=r"(ret.y) : __LDG_PTR (ptr)); return ret; }
static __device__ __inline__ int2 __load_streaming(const int2 *ptr) { int2 ret; asm volatile ("ld.global.cs.v2.s32 {%0,%1}, [%2];"  : "=r"(ret.x), "=r"(ret.y) : __LDG_PTR (ptr)); return ret; }
static __device__ __inline__ int2 __load_lastuse(const int2 *ptr) { int2 ret; asm volatile ("ld.global.lu.v2.s32 {%0,%1}, [%2];"  : "=r"(ret.x), "=r"(ret.y) : __LDG_PTR (ptr)); return ret; }
#if __CUDA_ARCH__ >= 350
static __device__ __inline__ int2 __load_nc(const int2 *ptr) { int2 ret; asm volatile ("ld.global.nc.v2.s32 {%0,%1}, [%2];"  : "=r"(ret.x), "=r"(ret.y) : __LDG_PTR (ptr)); return ret; }
#else
static __device__ __inline__ int2 __load_nc(const int2 *ptr) { int2 ret; asm volatile ("ld.global.ca.v2.s32 {%0,%1}, [%2];"  : "=r"(ret.x), "=r"(ret.y) : __LDG_PTR (ptr)); return ret; }
#endif
static __device__ __inline__ int2 __load_uniform(const int2 *ptr) { int2 ret; asm volatile ("ld.global.s32.v2.uniform {%0,%1}, [%2];"  : "=r"(ret.x), "=r"(ret.y) : __LDG_PTR (ptr)); return ret; }


static __device__ __inline__ int4 __load_all(const int4 *ptr) { int4 ret; asm volatile ("ld.global.ca.v4.s32 {%0,%1,%2,%3}, [%4];"  : "=r"(ret.x), "=r"(ret.y), "=r"(ret.z), "=r"(ret.w) : __LDG_PTR (ptr)); return ret; }
static __device__ __inline__ int4 __load_global(const int4 *ptr) { int4 ret; asm volatile ("ld.global.cg.v4.s32 {%0,%1,%2,%3}, [%4];"  : "=r"(ret.x), "=r"(ret.y), "=r"(ret.z), "=r"(ret.w) : __LDG_PTR (ptr)); return ret; }
static __device__ __inline__ int4 __load_streaming(const int4 *ptr) { int4 ret; asm volatile ("ld.global.cs.v4.s32 {%0,%1,%2,%3}, [%4];"  : "=r"(ret.x), "=r"(ret.y), "=r"(ret.z), "=r"(ret.w) : __LDG_PTR (ptr)); return ret; }
static __device__ __inline__ int4 __load_lastuse(const int4 *ptr) { int4 ret; asm volatile ("ld.global.lu.v4.s32 {%0,%1,%2,%3}, [%4];"  : "=r"(ret.x), "=r"(ret.y), "=r"(ret.z), "=r"(ret.w) : __LDG_PTR (ptr)); return ret; }
#if __CUDA_ARCH__ >= 350
static __device__ __inline__ int4 __load_nc(const int4 *ptr) { int4 ret; asm volatile ("ld.global.nc.v4.s32 {%0,%1,%2,%3}, [%4];"  : "=r"(ret.x), "=r"(ret.y), "=r"(ret.z), "=r"(ret.w) : __LDG_PTR (ptr)); return ret; }
#else
static __device__ __inline__ int4 __load_nc(const int4 *ptr) { int4 ret; asm volatile ("ld.global.ca.v4.s32 {%0,%1,%2,%3}, [%4];"  : "=r"(ret.x), "=r"(ret.y), "=r"(ret.z), "=r"(ret.w) : __LDG_PTR (ptr)); return ret; }
#endif
static __device__ __inline__ int4 __load_uniform(const int4 *ptr) { int4 ret; asm volatile ("ld.global.v4.s32.uniform {%0,%1,%2,%3}, [%4];"  : "=r"(ret.x), "=r"(ret.y), "=r"(ret.z), "=r"(ret.w) : __LDG_PTR (ptr)); return ret; }


static __device__ __inline__ float __load_all(const float *ptr) { float ret; asm volatile ("ld.global.ca.f32 %0, [%1];"  : "=f"(ret) : __LDG_PTR (ptr)); return ret; }
static __device__ __inline__ float __load_global(const float *ptr) { float ret; asm volatile ("ld.global.cg.f32 %0, [%1];"  : "=f"(ret) : __LDG_PTR (ptr)); return ret; }
static __device__ __inline__ float __load_streaming(const float *ptr) { float ret; asm volatile ("ld.global.cs.f32 %0, [%1];"  : "=f"(ret) : __LDG_PTR (ptr)); return ret; }
static __device__ __inline__ float __load_lastuse(const float *ptr) { float ret; asm volatile ("ld.global.lu.f32 %0, [%1];"  : "=f"(ret) : __LDG_PTR (ptr)); return ret; }
#if __CUDA_ARCH__ >= 350
static __device__ __inline__ float __load_nc(const float *ptr) { float ret; asm volatile ("ld.global.nc.f32 %0, [%1];"  : "=f"(ret) : __LDG_PTR (ptr)); return ret; }
#else
static __device__ __inline__ float __load_nc(const float *ptr) { float ret; asm volatile ("ld.global.nc.f32 %0, [%1];"  : "=f"(ret) : __LDG_PTR (ptr)); return ret; }
#endif
static __device__ __inline__ float __load_uniform(const float *ptr) { float ret; asm volatile ("ld.global.f32.uniform %0, [%1];"  : "=f"(ret) : __LDG_PTR (ptr)); return ret; }


static __device__ __inline__ float2 __load_all(const float2 *ptr) { float2 ret; asm volatile ("ld.global.ca.v2.f32 {%0,%1}, [%2];"  : "=f"(ret.x), "=f"(ret.y) : __LDG_PTR (ptr)); return ret; }
static __device__ __inline__ float2 __load_global(const float2 *ptr) { float2 ret; asm volatile ("ld.global.cg.v2.f32 {%0,%1}, [%2];"  : "=f"(ret.x), "=f"(ret.y) : __LDG_PTR (ptr)); return ret; }
static __device__ __inline__ float2 __load_streaming(const float2 *ptr) { float2 ret; asm volatile ("ld.global.cs.v2.f32 {%0,%1}, [%2];"  : "=f"(ret.x), "=f"(ret.y) : __LDG_PTR (ptr)); return ret; }
static __device__ __inline__ float2 __load_lastuse(const float2 *ptr) { float2 ret; asm volatile ("ld.global.lu.v2.f32 {%0,%1}, [%2];"  : "=f"(ret.x), "=f"(ret.y) : __LDG_PTR (ptr)); return ret; }
#if __CUDA_ARCH__ >= 350
static __device__ __inline__ float2 __load_nc(const float2 *ptr) { float2 ret; asm volatile ("ld.global.nc.v2.f32 {%0,%1}, [%2];"  : "=f"(ret.x), "=f"(ret.y) : __LDG_PTR (ptr)); return ret; }
#else
static __device__ __inline__ float2 __load_nc(const float2 *ptr) { float2 ret; asm volatile ("ld.global.ca.v2.f32 {%0,%1}, [%2];"  : "=f"(ret.x), "=f"(ret.y) : __LDG_PTR (ptr)); return ret; }
#endif
static __device__ __inline__ float2 __load_uniform(const float2 *ptr) { float2 ret; asm volatile ("ld.global.v2.f32.uniform {%0,%1}, [%2];"  : "=f"(ret.x), "=f"(ret.y) : __LDG_PTR (ptr)); return ret; }


static __device__ __inline__ float4 __load_all(const float4 *ptr) { float4 ret; asm volatile ("ld.global.ca.v4.f32 {%0,%1,%2,%3}, [%4];"  : "=f"(ret.x), "=f"(ret.y), "=f"(ret.z), "=f"(ret.w) : __LDG_PTR (ptr)); return ret; }
static __device__ __inline__ float4 __load_global(const float4 *ptr) { float4 ret; asm volatile ("ld.global.cg.v4.f32 {%0,%1,%2,%3}, [%4];"  : "=f"(ret.x), "=f"(ret.y), "=f"(ret.z), "=f"(ret.w) : __LDG_PTR (ptr)); return ret; }
static __device__ __inline__ float4 __load_streaming(const float4 *ptr) { float4 ret; asm volatile ("ld.global.cs.v4.f32 {%0,%1,%2,%3}, [%4];"  : "=f"(ret.x), "=f"(ret.y), "=f"(ret.z), "=f"(ret.w) : __LDG_PTR (ptr)); return ret; }
static __device__ __inline__ float4 __load_lastuse(const float4 *ptr) { float4 ret; asm volatile ("ld.global.lu.v4.f32 {%0,%1,%2,%3}, [%4];"  : "=f"(ret.x), "=f"(ret.y), "=f"(ret.z), "=f"(ret.w) : __LDG_PTR (ptr)); return ret; }
#if __CUDA_ARCH__ >= 350
static __device__ __inline__ float4 __load_nc(const float4 *ptr) { float4 ret; asm volatile ("ld.global.nc.v4.f32 {%0,%1,%2,%3}, [%4];"  : "=f"(ret.x), "=f"(ret.y), "=f"(ret.z), "=f"(ret.w) : __LDG_PTR (ptr)); return ret; }
#else
static __device__ __inline__ float4 __load_nc(const float4 *ptr) { float4 ret; asm volatile ("ld.global.ca.v4.f32 {%0,%1,%2,%3}, [%4];"  : "=f"(ret.x), "=f"(ret.y), "=f"(ret.z), "=f"(ret.w) : __LDG_PTR (ptr)); return ret; }
#endif
static __device__ __inline__ float4 __load_uniform(const float4 *ptr) { float4 ret; asm volatile ("ld.global.v4.f32.uniform {%0,%1,%2,%3}, [%4];"  : "=f"(ret.x), "=f"(ret.y), "=f"(ret.z), "=f"(ret.w) : __LDG_PTR (ptr)); return ret; }


static __device__ __inline__ double __load_all(const double *ptr) { double ret; asm volatile ("ld.global.ca.f64 %0, [%1];"  : "=d"(ret) : __LDG_PTR (ptr)); return ret; }
static __device__ __inline__ double __load_global(const double *ptr) { double ret; asm volatile ("ld.global.cg.f64 %0, [%1];"  : "=d"(ret) : __LDG_PTR (ptr)); return ret; }
static __device__ __inline__ double __load_streaming(const double *ptr) { double ret; asm volatile ("ld.global.cs.f64 %0, [%1];"  : "=d"(ret) : __LDG_PTR (ptr)); return ret; }
static __device__ __inline__ double __load_lastuse(const double *ptr) { double ret; asm volatile ("ld.global.lu.f64 %0, [%1];"  : "=d"(ret) : __LDG_PTR (ptr)); return ret; }
#if __CUDA_ARCH__ >= 350
static __device__ __inline__ double __load_nc(const double *ptr) { double ret; asm volatile ("ld.global.nc.f64 %0, [%1];"  : "=d"(ret) : __LDG_PTR (ptr)); return ret; }
#else
static __device__ __inline__ double __load_nc(const double *ptr) { double ret; asm volatile ("ld.global.ca.f64 %0, [%1];"  : "=d"(ret) : __LDG_PTR (ptr)); return ret; }
#endif
static __device__ __inline__ double __load_uniform(const double *ptr) { double ret; asm volatile ("ld.global.f64.uniform %0, [%1];"  : "=d"(ret) : __LDG_PTR (ptr)); return ret; }


static __device__ __inline__ double2 __load_all(const double2 *ptr) { double2 ret; asm volatile ("ld.global.ca.v2.f64 {%0,%1}, [%2];"  : "=d"(ret.x), "=d"(ret.y) : __LDG_PTR (ptr)); return ret; }
static __device__ __inline__ double2 __load_global(const double2 *ptr) { double2 ret; asm volatile ("ld.global.cg.v2.f64 {%0,%1}, [%2];"  : "=d"(ret.x), "=d"(ret.y) : __LDG_PTR (ptr)); return ret; }
static __device__ __inline__ double2 __load_streaming(const double2 *ptr) { double2 ret; asm volatile ("ld.global.cs.v2.f64 {%0,%1}, [%2];"  : "=d"(ret.x), "=d"(ret.y) : __LDG_PTR (ptr)); return ret; }
static __device__ __inline__ double2 __load_lastuse(const double2 *ptr) { double2 ret; asm volatile ("ld.global.lu.v2.f64 {%0,%1}, [%2];"  : "=d"(ret.x), "=d"(ret.y) : __LDG_PTR (ptr)); return ret; }
#if __CUDA_ARCH__ >= 350
static __device__ __inline__ double2 __load_nc(const double2 *ptr) { double2 ret; asm volatile ("ld.global.nc.v2.f64 {%0,%1}, [%2];"  : "=d"(ret.x), "=d"(ret.y) : __LDG_PTR (ptr)); return ret; }
#else
static __device__ __inline__ double2 __load_nc(const double2 *ptr) { double2 ret; asm volatile ("ld.global.ca.v2.f64 {%0,%1}, [%2];"  : "=d"(ret.x), "=d"(ret.y) : __LDG_PTR (ptr)); return ret; }
#endif
static __device__ __inline__ double2 __load_uniform(const double2 *ptr) { double2 ret; asm volatile ("ld.global.v2.f64.uniform {%0,%1}, [%2];"  : "=d"(ret.x), "=d"(ret.y) : __LDG_PTR (ptr)); return ret; }
