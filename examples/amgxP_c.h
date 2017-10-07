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

#ifndef __AMGX_P_C_H_INCLUDE__
#define __AMGX_P_C_H_INCLUDE__

#ifdef _WIN32
#ifdef AMGX_API_EXPORTS
#define AMGX_API __declspec( dllexport )
#else
#ifdef AMGX_API_NO_IMPORTS
#define AMGX_API
#else
#define AMGX_API __declspec( dllimport )
#endif
#endif
#else
#define AMGX_API __attribute__((visibility ("default")))
#endif

#if defined(__cplusplus)
extern "C" {
#endif

/* this header file contains experimental routines and routines used for testing */
typedef enum
{
    AMGX_TIMER_CPU = 1,
    AMGX_TIMER_GPU = 2,
    AMGX_TIMER_CREATE_AND_START = 4,
    AMGX_TIMER_ACCUMULATE_AVERAGE = 8,
} AMGX_TIMER_FLAGS;

AMGX_RC AMGX_API AMGX_timer_create
(const char *label,
 unsigned int flags);

AMGX_RC AMGX_API AMGX_timer_start
(const char *label);

//get time upto this moment (do not stop the timer)
AMGX_RC AMGX_API AMGX_timer_elapsed
(const char *label,
 double *sec);

//retrieve accumulated time upto this moment
AMGX_RC AMGX_API AMGX_timer_get_total
(const char *label,
 double *sec);

AMGX_RC AMGX_API AMGX_timer_stop
(const char *label,
 double *sec);

// TODO figure out how to use multiple GPUs with one thread, should there be a resource_activate instead
AMGX_RC AMGX_API AMGX_set_device
(int device);

// TODO might be needed for D1 interpolation, needs to be renamed to be more descriptive of what it does
AMGX_RC AMGX_API AMGX_matrix_sort
(AMGX_matrix_handle obj);

AMGX_RC AMGX_API AMGX_read_geometry
(const char *fname,
 double **geo_x,
 double **geo_y,
 double **geo_z,
 int *dim,
 int *numrows);

AMGX_RC AMGX_API AMGX_read_coloring
(const char *fname,
 int **row_coloring,
 int *colored_rows,
 int *num_colors);

AMGX_RC AMGX_API AMGX_read_system_with_cfg
(AMGX_matrix_handle mtx,
 AMGX_vector_handle rhs,
 AMGX_vector_handle sol,
 const char *filename,
 const AMGX_config_handle cfg_h);


#if defined(__cplusplus)
}//extern "C"
#endif

#endif
