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

#ifdef _WIN32
#pragma warning (push)
#pragma warning (disable : 4244 4267 4521)
#endif
#ifdef _WIN32
#pragma warning (pop)
#endif

#include <cstdio>
#include <amgx_config.h>
#include <cusp/memory.h>
#include <cuComplex.h>

#if ( defined( _MSC_VER ) && ( _MSC_VER < 1600 ) )

typedef __int32  int32_t;
typedef unsigned __int32 uint32_t;
typedef __int64 int64_t;
typedef unsigned __int64 uint64_t;

#else
#include <stdint.h>
#endif

//#include <matrix.h>

// define this here for demonstration
//namespace cusp { struct device_memory {}; struct host_memory {}; };

namespace amgx
{
enum AMGX_STATUS
{
    /*********************************************************
     * Flags for status reporting
     *********************************************************/
    AMGX_ST_CONVERGED = 0,
    AMGX_ST_ERROR = 1,
    AMGX_ST_DIVERGED = 2,
    AMGX_ST_NOT_CONVERGED = 3
};
}

namespace amgx
{

template <AMGX_MemorySpace> struct MemorySpaceMap;
template <> struct MemorySpaceMap<AMGX_host>      { typedef cusp::host_memory Type;   static const AMGX_MemorySpace id = AMGX_host;   static const char *getName() { return "host"; }};
template <> struct MemorySpaceMap<AMGX_device>    { typedef cusp::device_memory Type; static const AMGX_MemorySpace id = AMGX_device; static const char *getName() { return "device"; }};
template <AMGX_VecPrecision> struct VecPrecisionMap;
template <> struct VecPrecisionMap<AMGX_vecDouble>  { typedef double Type;    static const AMGX_VecPrecision id = AMGX_vecDouble; static const char *getName() { return "double"; }};
template <> struct VecPrecisionMap<AMGX_vecFloat>   { typedef float Type;     static const AMGX_VecPrecision id = AMGX_vecFloat;  static const char *getName() { return "float"; }};
template <> struct VecPrecisionMap<AMGX_vecDoubleComplex>   { typedef cuDoubleComplex Type;   static const AMGX_VecPrecision id = AMGX_vecDoubleComplex; static const char *getName() { return "doublecomplex"; }};
template <> struct VecPrecisionMap<AMGX_vecComplex>         { typedef cuComplex Type;         static const AMGX_VecPrecision id = AMGX_vecComplex;       static const char *getName() { return "complex"; }};
template <> struct VecPrecisionMap<AMGX_vecInt>     { typedef int Type;       static const AMGX_VecPrecision id = AMGX_vecInt;    static const char *getName() { return "int"; }};
template <> struct VecPrecisionMap<AMGX_vecUSInt>     { typedef unsigned short int Type;       static const AMGX_VecPrecision id = AMGX_vecUSInt;    static const char *getName() { return "unsigned short"; }};
template <> struct VecPrecisionMap<AMGX_vecUInt>     { typedef unsigned int Type;       static const AMGX_VecPrecision id = AMGX_vecUInt;    static const char *getName() { return "unsigned"; }};
template <> struct VecPrecisionMap<AMGX_vecUInt64>     { typedef uint64_t Type;       static const AMGX_VecPrecision id = AMGX_vecUInt64;    static const char *getName() { return "uint64_t"; }};
template <> struct VecPrecisionMap<AMGX_vecInt64>     { typedef int64_t Type;       static const AMGX_VecPrecision id = AMGX_vecInt64;    static const char *getName() { return "int64_t"; }};
template <> struct VecPrecisionMap<AMGX_vecBool>    { typedef bool Type;      static const AMGX_VecPrecision id = AMGX_vecBool;   static const char *getName() { return "bool"; }};
template <AMGX_MatPrecision> struct MatPrecisionMap;
template <> struct MatPrecisionMap<AMGX_matDouble>  { typedef double Type;    static const AMGX_MatPrecision id = AMGX_matDouble; static const char *getName() { return "double"; }};
template <> struct MatPrecisionMap<AMGX_matFloat>   { typedef float Type;     static const AMGX_MatPrecision id = AMGX_matFloat;  static const char *getName() { return "float"; }};
template <> struct MatPrecisionMap<AMGX_matDoubleComplex>   { typedef cuDoubleComplex Type;   static const AMGX_MatPrecision id = AMGX_matDoubleComplex; static const char *getName() { return "doublecomplex"; }};
template <> struct MatPrecisionMap<AMGX_matComplex>         { typedef cuComplex Type;         static const AMGX_MatPrecision id = AMGX_matComplex;       static const char *getName() { return "complex"; }};
template <AMGX_IndPrecision> struct IndPrecisionMap;
template <> struct IndPrecisionMap<AMGX_indInt>     { typedef int Type;       static const AMGX_IndPrecision id = AMGX_indInt;    static const char *getName() { return "int"; }};
template <> struct IndPrecisionMap<AMGX_indInt64>     { typedef int64_t Type;       static const AMGX_IndPrecision id = AMGX_indInt64;    static const char *getName() { return "int64_t"; }};


template <AMGX_Mode> struct ModeString;
template <> struct ModeString<AMGX_mode_hDDI>  { static const char *getName() { return "hDDI"; }};
template <> struct ModeString<AMGX_mode_hDFI>  { static const char *getName() { return "hDFI"; }};
template <> struct ModeString<AMGX_mode_hFFI>  { static const char *getName() { return "hFFI"; }};
template <> struct ModeString<AMGX_mode_dDDI>  { static const char *getName() { return "dDDI"; }};
template <> struct ModeString<AMGX_mode_dDFI>  { static const char *getName() { return "dDFI"; }};
template <> struct ModeString<AMGX_mode_dFFI>  { static const char *getName() { return "dFFI"; }};
template <> struct ModeString<AMGX_mode_hZZI>  { static const char *getName() { return "hZZI"; }};
template <> struct ModeString<AMGX_mode_hZCI>  { static const char *getName() { return "hZCI"; }};
template <> struct ModeString<AMGX_mode_hCCI>  { static const char *getName() { return "hCCI"; }};
template <> struct ModeString<AMGX_mode_dZZI>  { static const char *getName() { return "dZZI"; }};
template <> struct ModeString<AMGX_mode_dZCI>  { static const char *getName() { return "dZCI"; }};
template <> struct ModeString<AMGX_mode_dCCI>  { static const char *getName() { return "dCCI"; }};


template <AMGX_MemorySpace t_memSpace, AMGX_VecPrecision t_vecPrec, AMGX_MatPrecision t_matPrec, AMGX_IndPrecision t_indPrec>
struct TemplateConfig
{
    static const AMGX_MemorySpace memSpace = t_memSpace;
    static const AMGX_VecPrecision vecPrec = t_vecPrec;
    static const AMGX_MatPrecision matPrec = t_matPrec;
    static const AMGX_IndPrecision indPrec = t_indPrec;

    static const AMGX_Mode mode = (AMGX_Mode)AMGX_ASSEMBLE_MODE(t_memSpace, t_vecPrec, t_matPrec, t_indPrec);

    typedef MemorySpaceMap<memSpace> MemSpaceInfo;
    typedef VecPrecisionMap<vecPrec> VecPrecInfo;
    typedef MatPrecisionMap<matPrec> MatPrecInfo;
    typedef IndPrecisionMap<indPrec> IndPrecInfo;

    // When one of these typedefs reports: incomplete type is not allowed
    // it means that at least one of the value memSpace, vecPrec, matPrec or indPrec is wrong
    typedef typename MemSpaceInfo::Type MemSpace;
    typedef typename VecPrecInfo::Type  VecPrec;
    typedef typename MatPrecInfo::Type  MatPrec;
    typedef typename IndPrecInfo::Type  IndPrec;

    template<AMGX_MemorySpace t_newMemSpace> struct setMemSpace { typedef TemplateConfig<t_newMemSpace, t_vecPrec, t_matPrec, t_indPrec> Type; };
    template<AMGX_VecPrecision t_newVecPrec> struct setVecPrec  { typedef TemplateConfig<t_memSpace, t_newVecPrec, t_matPrec, t_indPrec> Type; };
    template<AMGX_MatPrecision t_newMatPrec> struct setMatPrec  { typedef TemplateConfig<t_memSpace, t_vecPrec, t_newMatPrec, t_indPrec> Type; };
    template<AMGX_IndPrecision t_newIndPrec> struct setIndPrec  { typedef TemplateConfig<t_memSpace, t_vecPrec, t_matPrec, t_newIndPrec> Type; };

    static bool is_built()
    {
        bool built = false;

        switch ((AMGX_Mode)mode)
        {
#define AMGX_CASE_LINE(CASE) case CASE: built= true; break;
                AMGX_FORALL_BUILDS(AMGX_CASE_LINE)
#undef AMGX_CASE_LINE
        }

        return built;
    }

    static void print()
    {
        printf("TemplateConfig: mode %8d = 0x%-8X, built %3s, MemSpace %8s, VecPrec %8s, MatPrec %8s, IndPrec %8s\n",
               mode, mode, is_built() ? "Yes" : "No",
               MemSpaceInfo::getName(),
               VecPrecInfo::getName(),
               MatPrecInfo::getName(),
               IndPrecInfo::getName());
    }

    static void sprint(char *buf)
    {
        sprintf(buf, "mode %8d = 0x%-8X, built %3s, MemSpace %8s, VecPrec %8s, MatPrec %8s, IndPrec %8s",
                mode, mode, is_built() ? "Yes" : "No",
                MemSpaceInfo::getName(),
                VecPrecInfo::getName(),
                MatPrecInfo::getName(),
                IndPrecInfo::getName());
    }
};

template <int t_mode>
struct TemplateMode
{
    typedef TemplateConfig <
    AMGX_GET_MODE_VAL(AMGX_MemorySpace,  t_mode),
                      AMGX_GET_MODE_VAL(AMGX_VecPrecision, t_mode),
                      AMGX_GET_MODE_VAL(AMGX_MatPrecision, t_mode),
                      AMGX_GET_MODE_VAL(AMGX_IndPrecision, t_mode)
                      > Type;
};

}//namespace amgx

