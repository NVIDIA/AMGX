// SPDX-FileCopyrightText: 2013 - 2025 NVIDIA CORPORATION. All Rights Reserved.
//
// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#include <stdio.h>
#include <stdlib.h>
#include "amgx_config.h"
#include "amgx_c.h"

#if defined(__cplusplus)
extern "C" {
#endif

typedef struct AMGX_eigensolver_handle_struct {char AMGX_eigensolver_handle_dummy;} *AMGX_eigensolver_handle;

AMGX_RC AMGX_API AMGX_eigensolver_create(AMGX_eigensolver_handle *ret, AMGX_resources_handle rsc, AMGX_Mode mode, const AMGX_config_handle config_eigensolver);

AMGX_RC AMGX_API AMGX_eigensolver_setup(AMGX_eigensolver_handle eigensolver, AMGX_matrix_handle mtx);

AMGX_RC AMGX_API AMGX_eigensolver_pagerank_setup(AMGX_eigensolver_handle eigensolver, AMGX_vector_handle a);

AMGX_RC AMGX_API AMGX_eigensolver_solve(AMGX_eigensolver_handle eigensolver, AMGX_vector_handle x);

AMGX_RC AMGX_API AMGX_eigensolver_destroy(AMGX_eigensolver_handle obj);

#if defined(__cplusplus)
}//extern "C"
#endif
