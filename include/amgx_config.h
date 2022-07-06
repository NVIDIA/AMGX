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

#ifndef __AMGX_CONFIG_H_INCLUDE__
#define __AMGX_CONFIG_H_INCLUDE__

/* WARNING: Always use enums, do NOT use knowledge about the formulas
            for AMGX_Mode because it might change in the future.   */

#if defined(__cplusplus)
extern "C" {
#endif

/* Memory */
typedef enum
{
    AMGX_host,
    AMGX_device,
    AMGX_memorySpaceNum = 16
} AMGX_MemorySpace;

/* Data */
typedef enum
{
    AMGX_double,
    AMGX_float,
    AMGX_int,
    AMGX_doublecomplex,
    AMGX_complex,
    AMGX_usint,
    AMGX_uint,
    AMGX_uint64,
    AMGX_int64,
    AMGX_bool,
    AMGX_scalarPrecisionNum = 16
} AMGX_ScalarPrecision;

/* Vector */
typedef enum
{
    AMGX_vecDouble =          AMGX_double,
    AMGX_vecFloat =           AMGX_float,
    AMGX_vecDoubleComplex =   AMGX_doublecomplex,
    AMGX_vecComplex =         AMGX_complex,
    AMGX_vecInt =             AMGX_int,
    AMGX_vecUSInt =           AMGX_usint,
    AMGX_vecUInt =            AMGX_uint,
    AMGX_vecUInt64 =          AMGX_uint64,
    AMGX_vecInt64 =           AMGX_int64,
    AMGX_vecBool =            AMGX_bool,
    AMGX_vecPrecisionNum =    AMGX_scalarPrecisionNum,
    AMGX_VecPrecisionInst
} AMGX_VecPrecision;

/* Matrix */
typedef enum
{
    AMGX_matDouble =         AMGX_double,
    AMGX_matFloat =          AMGX_float,
    AMGX_matDoubleComplex =  AMGX_doublecomplex,
    AMGX_matComplex =        AMGX_complex,
    AMGX_matInt =            AMGX_int,
    AMGX_matPrecisionNum = AMGX_scalarPrecisionNum,
    AMGX_MatPrecisionInst
} AMGX_MatPrecision;

/* Index */
typedef enum
{
    AMGX_indInt =          AMGX_int,
    AMGX_indInt64 =        AMGX_int64,
    AMGX_indPrecisionNum = AMGX_scalarPrecisionNum,
    AMGX_IndPrecisionInst
} AMGX_IndPrecision;


// User-defined types
constexpr AMGX_MemorySpace AMGX_MemSpace = AMGX_device;
constexpr AMGX_MatPrecision AMGX_MatPrec = AMGX_matDouble;
constexpr AMGX_VecPrecision AMGX_VecPrec = AMGX_vecDouble;
constexpr AMGX_IndPrecision AMGX_IndPrec = AMGX_indInt;

#if defined(__cplusplus)
}
#endif

#endif
