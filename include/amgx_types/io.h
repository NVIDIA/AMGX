// SPDX-FileCopyrightText: 2011 - 2025 NVIDIA CORPORATION. All Rights Reserved.
//
// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#include <basic_types.h>
#include <ostream>

std::ostream &operator<<(std::ostream &os, const cuComplex &x);
std::ostream &operator<<(std::ostream &os, const cuDoubleComplex &x);