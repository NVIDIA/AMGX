/* Copyright (c) 2013-2017, NVIDIA CORPORATION. All rights reserved.
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
