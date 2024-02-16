// SPDX-FileCopyrightText: 2011 - 2024 NVIDIA CORPORATION. All Rights Reserved.
//
// SPDX-License-Identifier: BSD-3-Clause

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
