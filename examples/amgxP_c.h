// SPDX-FileCopyrightText: 2011 - 2024 NVIDIA CORPORATION. All Rights Reserved.
//
// SPDX-License-Identifier: BSD-3-Clause

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
