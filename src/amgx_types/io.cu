// SPDX-FileCopyrightText: 2011 - 2025 NVIDIA CORPORATION. All Rights Reserved.
//
// SPDX-License-Identifier: BSD-3-Clause

#include <amgx_types/util.h>

std::ostream &operator<<(std::ostream &os, const cuComplex &x)
{
    os << amgx::types::get_re(x) << " " << amgx::types::get_im(x);
    return os;
}

std::ostream &operator<<(std::ostream &os, const cuDoubleComplex &x)
{
    os << amgx::types::get_re(x) << " " << amgx::types::get_im(x);
    return os;
}