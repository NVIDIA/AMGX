// SPDX-FileCopyrightText: 2011 - 2024 NVIDIA CORPORATION. All Rights Reserved.
//
// SPDX-License-Identifier: BSD-3-Clause

#pragma once

namespace amgx
{

typedef void (*AMGX_output_callback)(const char *msg, int length);
extern AMGX_output_callback amgx_output;
extern AMGX_output_callback error_output;
extern AMGX_output_callback amgx_distributed_output;
int amgx_printf(const char *fmt, ...);

#ifdef NDEBUG
#define amgx_printf_debug(fmt,...)
#define device_printf(fmt,...)
#else
#define amgx_printf_debug(fmt,...) amgx_printf(fmt,##__VA_ARGS__)
#define device_printf(fmt,...) printf(fmt,##__VA_ARGS__)
#endif

} // namespace amgx

