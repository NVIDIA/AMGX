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
