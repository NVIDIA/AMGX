// SPDX-FileCopyrightText: 2011 - 2025 NVIDIA CORPORATION. All Rights Reserved.
//
// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#include <error.h>
#include <vector.h>

namespace amgx
{

enum ASSIGNMENTS {COARSE = -1, FINE = -2, STRONG_FINE = -3, UNASSIGNED = -4};

// NormType
enum NormType {L1, L1_SCALED, L2, LMAX};
inline const char *getString(NormType p)
{
    switch (p)
    {
        case L1:
            return "L1";

        case L1_SCALED:
            return "L1_SCALED";

        case L2:
            return "L2";

        case LMAX:
            return "LMAX";

        default:
            return "UNKNOWN";
    }
}

// BlockFormat
enum BlockFormat {ROW_MAJOR, COL_MAJOR};
inline const char *getString(BlockFormat p)
{
    switch (p)
    {
        case ROW_MAJOR:
            return "ROW_MAJOR";

        case COL_MAJOR:
            return "COL_MAJOR";

        default:
            return "UNKNOWN";
    }
}

// AlgorithmType

enum AlgorithmType { CLASSICAL, AGGREGATION, ENERGYMIN };

inline const char *getString(AlgorithmType p)
{
    switch (p)
    {
        case CLASSICAL:
            return "CLASSICAL";

        case AGGREGATION:
            return "AGGREGATION";

        case ENERGYMIN:
            return "ENERGYMIN";

        default:
            return "UNKNOWN";
    }
}

inline const char *getString(ViewType p)
{
    switch (p)
    {
        case INTERIOR:
            return "INTERIOR";

        case OWNED:
            return "OWNED";

        case FULL:
            return "FULL";

        case ALL:
            return "ALL";

        default:
            return "UNKNOWN";
    }
}

inline const char *getString(ColoringType p)
{
    switch (p)
    {
        case FIRST:
            return "FIRST";

        case SYNC_COLORS:
            return "SYNC_COLORS";

        case LAST:
            return "LAST";

        default:
            return "UNKNOWN";
    }
}

}

#include<getvalue.h>
