// SPDX-FileCopyrightText: 2011 - 2024 NVIDIA CORPORATION. All Rights Reserved.
//
// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#ifdef AMGX_WITH_OPENMP
#include <omp.h>
#else

static inline int omp_get_num_threads() throw() { return 1; }
static inline int omp_get_thread_num() throw() { return 0; }

#endif


