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

#include <error.h>
#include <vector.h>

namespace amgx
{

enum ASSIGNMENTS {COARSE = -1, FINE = -2, STRONG_FINE = -3, UNASSIGNED = -4};

// NormType
enum NormType {L1, L2, LMAX};
inline const char *getString(NormType p)
{
    switch (p)
    {
        case L1:
            return "L1";

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
