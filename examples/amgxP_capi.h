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

#if defined(__cplusplus)
extern "C" {
#endif

/* WARNING: this is a private header file, it should not be publically exposed.
            it contains experimental and internal routines used for testing */
typedef enum
{
    AMGX_TIMER_CPU = 1,
    AMGX_TIMER_GPU = 2,
    AMGX_TIMER_CREATE_AND_START = 4,
    AMGX_TIMER_ACCUMULATE_AVERAGE = 8,
} AMGX_TIMER_FLAGS;

typedef AMGX_RC (*_AMGX_timer_create)
(const char *label,
 unsigned int flags);

typedef AMGX_RC (*_AMGX_timer_start)
(const char *label);

//get time upto this moment (do not stop the timer)
typedef AMGX_RC (*_AMGX_timer_elapsed)
(const char *label,
 double *sec);

//retrieve accumulated time upto this moment
typedef AMGX_RC (*_AMGX_timer_get_total)
(const char *label,
 double *sec);

typedef AMGX_RC (*_AMGX_timer_stop)
(const char *label,
 double *sec);

// TODO figure out how to use multiple GPUs with one thread, should there be a resource_activate instead
typedef AMGX_RC (*_AMGX_set_device)
(int device);

// TODO might be needed for D1 interpolation, needs to be renamed to be more descriptive of what it does
typedef AMGX_RC (*_AMGX_matrix_sort)
(AMGX_matrix_handle obj);

typedef AMGX_RC (*_AMGX_read_geometry)
(const char *fname,
 double **geo_x,
 double **geo_y,
 double **geo_z,
 int *dim,
 int *numrows);

typedef AMGX_RC (*_AMGX_read_coloring)
(const char *fname,
 int **row_coloring,
 int *colored_rows,
 int *num_colors);

typedef AMGX_RC (*_AMGX_read_system_with_cfg)
(AMGX_matrix_handle mtx,
 AMGX_vector_handle rhs,
 AMGX_vector_handle sol,
 const char *filename,
 const AMGX_config_handle cfg_h);


/*********************************************************
 * Setup dynamic loading of the library
 *********************************************************/
#ifdef _WIN32
#include "windows.h"
#define amgx_libopen(path) (void*)(LoadLibrary(path))
#define amgx_liblink(handle, symbol) GetProcAddress((HMODULE)(handle), symbol)
#define amgx_libclose(handle) FreeLibrary((HMODULE)(handle))
#endif

#ifdef __unix__
#include <dlfcn.h>
#include <unistd.h>
#define amgx_libopen(path) dlopen(path, RTLD_LAZY)
#define amgx_liblink(handle, symbol) dlsym(handle, symbol)
#define amgx_libclose(handle) dlclose(handle)
#endif

/* use types above to define function pointers */
_AMGX_timer_create            AMGX_timer_create;
_AMGX_timer_start             AMGX_timer_start;
_AMGX_timer_elapsed           AMGX_timer_elapsed;
_AMGX_timer_get_total         AMGX_timer_get_total;
_AMGX_timer_stop              AMGX_timer_stop;
_AMGX_matrix_sort             AMGX_matrix_sort;
_AMGX_read_geometry           AMGX_read_geometry;
_AMGX_read_coloring           AMGX_read_coloring;

int amgxP_liblink_all(void *lib_handle)
{
    AMGX_timer_create            = (_AMGX_timer_create)amgx_liblink(lib_handle, "AMGX_timer_create");
    AMGX_timer_start             = (_AMGX_timer_start)amgx_liblink(lib_handle, "AMGX_timer_start");
    AMGX_timer_elapsed           = (_AMGX_timer_elapsed)amgx_liblink(lib_handle, "AMGX_timer_elapsed");
    AMGX_timer_get_total         = (_AMGX_timer_get_total)amgx_liblink(lib_handle, "AMGX_timer_get_total");
    AMGX_timer_stop              = (_AMGX_timer_stop)amgx_liblink(lib_handle, "AMGX_timer_stop");
    AMGX_matrix_sort             = (_AMGX_matrix_sort)amgx_liblink(lib_handle, "AMGX_matrix_sort");
    AMGX_read_geometry           = (_AMGX_read_geometry)amgx_liblink(lib_handle, "AMGX_read_geometry");
    AMGX_read_coloring           = (_AMGX_read_coloring)amgx_liblink(lib_handle, "AMGX_read_coloring");
    AMGX_read_system_with_cfg    = (_AMGX_read_system_with_cfg)amgx_liblink(lib_handle, "_AMGX_read_system_with_cfg");

    if (AMGX_timer_create == NULL ||
            AMGX_timer_start == NULL ||
            AMGX_timer_elapsed == NULL ||
            AMGX_timer_get_total == NULL ||
            AMGX_timer_stop == NULL ||
            AMGX_matrix_sort == NULL ||
            AMGX_read_geometry == NULL ||
            AMGX_read_coloring == NULL ||
            AMGX_read_system_with_cfg == NULL)
    {
        return 0;
    }
    else
    {
        return 1;
    }
}

#if defined(__cplusplus)
}//extern "C"
#endif

#endif
