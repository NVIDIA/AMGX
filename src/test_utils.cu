// SPDX-FileCopyrightText: 2011 - 2024 NVIDIA CORPORATION. All Rights Reserved.
//
// SPDX-License-Identifier: BSD-3-Clause

#include <test_utils.h>

namespace amgx
{

std::vector<std::string> &split(const std::string &s, char delim, std::vector<std::string> &elems)
{
    std::stringstream ss(s);
    std::string item;

    while (std::getline(ss, item, delim))
    {
        elems.push_back(item);
    }

    return elems;
}


std::vector<std::string> split(const std::string &s, char delim)
{
    std::vector<std::string> elems;
    return split(s, delim, elems);
}

AMGX_Mode getModeFromString(const char *strmode)
{
    if (strcmp(strmode, "dDDI") == 0)
    {
        return AMGX_mode_dDDI;
    }
    else if (strcmp(strmode, "dDFI") == 0)
    {
        return AMGX_mode_dDFI;
    }
    else if (strcmp(strmode, "dFFI") == 0)
    {
        return AMGX_mode_dFFI;
    }
    else if (strcmp(strmode, "hDDI") == 0)
    {
        return AMGX_mode_hDDI;
    }
    else if (strcmp(strmode, "hDFI") == 0)
    {
        return AMGX_mode_hDFI;
    }
    else if (strcmp(strmode, "hFFI") == 0)
    {
        return AMGX_mode_hFFI;
    }
    else if (strcmp(strmode, "dZZI") == 0)
    {
        return AMGX_mode_dZZI;
    }
    else if (strcmp(strmode, "dZCI") == 0)
    {
        return AMGX_mode_dZCI;
    }
    else if (strcmp(strmode, "dCCI") == 0)
    {
        return AMGX_mode_dCCI;
    }
    else if (strcmp(strmode, "hZZI") == 0)
    {
        return AMGX_mode_hZZI;
    }
    else if (strcmp(strmode, "hZCI") == 0)
    {
        return AMGX_mode_hZCI;
    }
    else if (strcmp(strmode, "hCCI") == 0)
    {
        return AMGX_mode_hCCI;
    }
    else
    {
        printf("Unknown mode: \"%s\"\n", strmode);
    }

    return AMGX_mode_dDDI;
}

AMGX_Mode getModeFromString(const std::string &strmode)
{
    return getModeFromString(strmode.c_str());
}

} // namespace amgx
