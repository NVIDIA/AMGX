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
#include <types.h>
#include <vector.h>
#include <sstream>

namespace amgx
{

template <class T>
inline T getValue(const char *name);

template <>
inline int getValue<int>(const char *name)
{
    return atoi(name);
}

template <>
inline size_t getValue<size_t>(const char *name)
{
    std::istringstream f(name);
    size_t res;
    f >> res;
    return res;
}

template <>
inline float getValue<float>(const char *name)
{
    return (float)atof(name);
}

template <>
inline double getValue<double>(const char *name)
{
    return atof(name);
}

template <>
inline NormType getValue<NormType>(const char *name)
{
    if (strncmp(name, "L1", 100) == 0)
    {
        return L1;
    }
    else if (strncmp(name, "L1_SCALED", 100) == 0)
    {
        return L1_SCALED;
    }
    else if (strncmp(name, "L2", 100) == 0)
    {
        return L2;
    }
    else if (strncmp(name, "LMAX", 100) == 0)
    {
        return LMAX;
    }

    char error[100];
    sprintf(error, "NormType'%s' is not defined", name);
    FatalError(error, AMGX_ERR_CONFIGURATION);
}

template <>
inline BlockFormat getValue<BlockFormat>(const char *name)
{
    if (strncmp(name, "ROW_MAJOR", 100) == 0)
    {
        return ROW_MAJOR;
    }
    else if (strncmp(name, "COL_MAJOR", 100) == 0)
    {
        return COL_MAJOR;
    }

    char error[100];
    sprintf(error, "BlockFormat'%s' is not defined", name);
    FatalError(error, AMGX_ERR_CONFIGURATION);
}

template <>
inline AlgorithmType getValue<AlgorithmType>(const char *name)
{
    if (strncmp(name, "CLASSICAL", 100) == 0)
    {
        return CLASSICAL;
    }
    else if (strncmp(name, "AGGREGATION", 100) == 0)
    {
        return AGGREGATION;
    }
    else if (strncmp(name, "ENERGYMIN", 100) == 0)
    {
        return ENERGYMIN;
    }

    char error[100];
    sprintf(error, "AlgorithmType'%s' is not defined", name);
    FatalError(error, AMGX_ERR_CONFIGURATION);
}

template <>
inline ViewType getValue<ViewType>(const char *name)
{
    if (strncmp(name, "INTERIOR", 100) == 0)
    {
        return INTERIOR;
    }
    else if (strncmp(name, "OWNED", 100) == 0)
    {
        return OWNED;
    }
    else if (strncmp(name, "FULL", 100) == 0)
    {
        return FULL;
    }
    else if (strncmp(name, "ALL", 100) == 0)
    {
        return ALL;
    }

    char error[100];
    sprintf(error, "ViewType'%s' is not defined", name);
    FatalError(error, AMGX_ERR_CONFIGURATION);
}

template <>
inline ColoringType getValue<ColoringType>(const char *name)
{
    if (strncmp(name, "FIRST", 100) == 0)
    {
        return FIRST;
    }
    else if (strncmp(name, "SYNC_COLORS", 100) == 0)
    {
        return SYNC_COLORS;
    }
    else if (strncmp(name, "LAST", 100) == 0)
    {
        return LAST;
    }

    char error[100];
    sprintf(error, "ColoringType'%s' is not defined", name);
    FatalError(error, AMGX_ERR_CONFIGURATION);
}




} // namespace amgx
