// SPDX-FileCopyrightText: 2011 - 2025 NVIDIA CORPORATION. All Rights Reserved.
//
// SPDX-License-Identifier: BSD-3-Clause

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


/* Mode */
typedef enum
{
    AMGX_MemorySpaceBase =  1,
    AMGX_MemorySpaceSize =  AMGX_memorySpaceNum,
    AMGX_VecPrecisionBase = AMGX_MemorySpaceBase * AMGX_MemorySpaceSize,
    AMGX_VecPrecisionSize = AMGX_vecPrecisionNum,
    AMGX_MatPrecisionBase = AMGX_VecPrecisionBase * AMGX_VecPrecisionSize,
    AMGX_MatPrecisionSize = AMGX_matPrecisionNum,
    AMGX_IndPrecisionBase = AMGX_MatPrecisionBase * AMGX_MatPrecisionSize,
    AMGX_IndPrecisionSize = AMGX_indPrecisionNum
} AMGX_ModeNums;

#define AMGX_ASSEMBLE_MODE(memSpace, vecPrec, matPrec, indPrec)\
(\
  memSpace  * AMGX_MemorySpaceBase  \
  + vecPrec * AMGX_VecPrecisionBase \
  + matPrec * AMGX_MatPrecisionBase \
  + indPrec * AMGX_IndPrecisionBase \
)

#define AMGX_GET_MODE_VAL(type, mode) ((type)( ((mode)/type##Base)%type##Size ))
#define AMGX_SET_MODE_VAL(type, mode, value) ((AMGX_Mode)( (mode) + ((value)-AMGX_GET_MODE_VAL(type, mode))*type##Base ))

typedef enum
{
    AMGX_unset = -1,
    AMGX_modeRange = AMGX_memorySpaceNum * AMGX_vecPrecisionNum * AMGX_matPrecisionNum * AMGX_indPrecisionNum,
    AMGX_mode_hDDI = AMGX_ASSEMBLE_MODE(AMGX_host,   AMGX_vecDouble, AMGX_matDouble, AMGX_indInt), // mode == 8192
    AMGX_mode_hDFI = AMGX_ASSEMBLE_MODE(AMGX_host,   AMGX_vecDouble, AMGX_matFloat,  AMGX_indInt), // mode == 8448
    AMGX_mode_hFFI = AMGX_ASSEMBLE_MODE(AMGX_host,   AMGX_vecFloat,  AMGX_matFloat,  AMGX_indInt), // mode == 8464
    AMGX_mode_dDDI = AMGX_ASSEMBLE_MODE(AMGX_device, AMGX_vecDouble, AMGX_matDouble, AMGX_indInt), // mode == 8193
    AMGX_mode_dDFI = AMGX_ASSEMBLE_MODE(AMGX_device, AMGX_vecDouble, AMGX_matFloat,  AMGX_indInt), // mode == 8449
    AMGX_mode_dFFI = AMGX_ASSEMBLE_MODE(AMGX_device, AMGX_vecFloat,  AMGX_matFloat,  AMGX_indInt), // mode == 8465
    AMGX_mode_hIDI = AMGX_ASSEMBLE_MODE(AMGX_host, AMGX_vecInt,  AMGX_matDouble,  AMGX_indInt),
    AMGX_mode_hIFI = AMGX_ASSEMBLE_MODE(AMGX_host, AMGX_vecInt,  AMGX_matFloat,  AMGX_indInt),
    AMGX_mode_dIDI = AMGX_ASSEMBLE_MODE(AMGX_device, AMGX_vecInt,  AMGX_matDouble,  AMGX_indInt),
    AMGX_mode_dIFI = AMGX_ASSEMBLE_MODE(AMGX_device, AMGX_vecInt,  AMGX_matFloat,  AMGX_indInt),
    AMGX_mode_hZZI = AMGX_ASSEMBLE_MODE(AMGX_host,   AMGX_vecDoubleComplex,  AMGX_matDoubleComplex, AMGX_indInt), // mode == 8192
    AMGX_mode_hZCI = AMGX_ASSEMBLE_MODE(AMGX_host,   AMGX_vecDoubleComplex,  AMGX_matComplex,  AMGX_indInt), // mode == 8448
    AMGX_mode_hCCI = AMGX_ASSEMBLE_MODE(AMGX_host,   AMGX_vecComplex,        AMGX_matComplex,  AMGX_indInt), // mode == 8464
    AMGX_mode_dZZI = AMGX_ASSEMBLE_MODE(AMGX_device, AMGX_vecDoubleComplex,  AMGX_matDoubleComplex, AMGX_indInt), // mode == 8193
    AMGX_mode_dZCI = AMGX_ASSEMBLE_MODE(AMGX_device, AMGX_vecDoubleComplex,  AMGX_matComplex,  AMGX_indInt), // mode == 8449
    AMGX_mode_dCCI = AMGX_ASSEMBLE_MODE(AMGX_device, AMGX_vecComplex,        AMGX_matComplex,  AMGX_indInt), // mode == 8465
    AMGX_modeNum = 10,
    AMGX_ModeInst
} AMGX_Mode;

/* Builds for host */
#ifdef AMGX_build_host
#define AMGX_FORALL_BUILDS_HOST(codeLineMacro)\
  codeLineMacro(AMGX_build_host)
#else
#define AMGX_FORALL_BUILDS_HOST(codeLineMacro)\
  codeLineMacro(AMGX_mode_hDDI)\
  codeLineMacro(AMGX_mode_hDFI)\
  codeLineMacro(AMGX_mode_hFFI)
#endif

#define AMGX_FORINTVEC_BUILDS_HOST(codeLineMacro)\
  codeLineMacro(AMGX_mode_hIDI)\
  codeLineMacro(AMGX_mode_hIFI)

/* Builds for device */
#ifdef AMGX_build_device
#define AMGX_FORALL_BUILDS_DEVICE(codeLineMacro)\
  codeLineMacro(AMGX_build_device)
#else
#define AMGX_FORALL_BUILDS_DEVICE(codeLineMacro)\
  codeLineMacro(AMGX_mode_dDDI)\
  codeLineMacro(AMGX_mode_dDFI)\
  codeLineMacro(AMGX_mode_dFFI)
#endif

#define AMGX_FORINTVEC_BUILDS_DEVICE(codeLineMacro)\
  codeLineMacro(AMGX_mode_dIDI)\
  codeLineMacro(AMGX_mode_dIFI)

#define AMGX_FORCOMPLEX_BUILDS_DEVICE(codeLineMacro)\
  codeLineMacro(AMGX_mode_dZZI)\
  codeLineMacro(AMGX_mode_dZCI)\
  codeLineMacro(AMGX_mode_dCCI)

#define AMGX_FORCOMPLEX_BUILDS_HOST(codeLineMacro)\
  codeLineMacro(AMGX_mode_hZZI)\
  codeLineMacro(AMGX_mode_hZCI)\
  codeLineMacro(AMGX_mode_hCCI)

/* Builds for device and host */
#define AMGX_FORALL_BUILDS(codeLineMacro)\
  AMGX_FORALL_BUILDS_HOST(codeLineMacro)\
  AMGX_FORALL_BUILDS_DEVICE(codeLineMacro)

#define AMGX_FORCOMPLEX_BUILDS(codeLineMacro)\
  AMGX_FORCOMPLEX_BUILDS_DEVICE(codeLineMacro)\
  AMGX_FORCOMPLEX_BUILDS_HOST(codeLineMacro)

#define AMGX_FORINTVEC_BUILDS(codeLineMacro)\
  AMGX_FORINTVEC_BUILDS_HOST(codeLineMacro)\
  AMGX_FORINTVEC_BUILDS_DEVICE(codeLineMacro)


#if defined(__cplusplus)
}
#endif

#endif
