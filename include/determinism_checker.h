// SPDX-FileCopyrightText: 2013 - 2025 NVIDIA CORPORATION. All Rights Reserved.
//
// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#include <string>

namespace amgx
{
namespace testing_tools
{

struct hash_path_determinism_checker_private;

struct hash_path_determinism_checker
{
    static hash_path_determinism_checker *singleton();
    hash_path_determinism_checker();
    ~hash_path_determinism_checker();

    hash_path_determinism_checker_private *priv;
    void checkpoint(const std::string &name, void *data, long long int size_in_bytes, bool no_permute = true);
    unsigned long long int checksum( void *data, long long int size_in_bytes, bool no_permute = true );
};

}
}
