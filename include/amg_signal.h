// SPDX-FileCopyrightText: 2011 - 2025 NVIDIA CORPORATION. All Rights Reserved.
//
// SPDX-License-Identifier: BSD-3-Clause

#pragma once

namespace amgx
{

class SignalHandler
{
        static bool hooked;
    public:
        static void hook();
        static void unhook();
};

} // namespace amgx
