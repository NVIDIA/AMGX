// SPDX-FileCopyrightText: 2011 - 2024 NVIDIA CORPORATION. All Rights Reserved.
//
// SPDX-License-Identifier: BSD-3-Clause

#ifndef __AMGX_C_H_INCLUDE__
#define __AMGX_C_H_INCLUDE__

#include <stdio.h>
#include <stdlib.h>
#include "amgx_config.h"

#if defined(__cplusplus)
extern "C" {
#endif

/*********************************************************
 ** These flags turn on output and vis data
 **********************************************************/
typedef enum
{
    SOLVE_STATS = 1,
    GRID_STATS = 2,
    CONFIG = 4,
    PROFILE_STATS = 8,
    VISDATA = 16,
    RESIDUAL_HISTORY = 32,
} AMGX_FLAGS;

/*********************************************************
 ** These enums define the return codes
 **********************************************************/
typedef enum
{
    AMGX_RC_OK = 0,
    AMGX_RC_BAD_PARAMETERS = 1,
    AMGX_RC_UNKNOWN = 2,
    AMGX_RC_NOT_SUPPORTED_TARGET = 3,
    AMGX_RC_NOT_SUPPORTED_BLOCKSIZE = 4,
    AMGX_RC_CUDA_FAILURE = 5,
    AMGX_RC_THRUST_FAILURE = 6,
    AMGX_RC_NO_MEMORY = 7,
    AMGX_RC_IO_ERROR = 8,
    AMGX_RC_BAD_MODE = 9,
    AMGX_RC_CORE = 10,
    AMGX_RC_PLUGIN = 11,
    AMGX_RC_BAD_CONFIGURATION = 12,
    AMGX_RC_NOT_IMPLEMENTED = 13,
    AMGX_RC_LICENSE_NOT_FOUND = 14,
    AMGX_RC_INTERNAL = 15
} AMGX_RC ;

/*********************************************************
 * Flags for status reporting
 *********************************************************/
typedef enum
{
    AMGX_SOLVE_SUCCESS = 0,
    AMGX_SOLVE_FAILED = 1,
    AMGX_SOLVE_DIVERGED = 2,
    AMGX_SOLVE_NOT_CONVERGED = 2,
} AMGX_SOLVE_STATUS;

/*********************************************************
 * Flags to determine behavior of distributed matrix partitioning
 *********************************************************/
typedef enum
{
    AMGX_DIST_PARTITION_VECTOR = 0,
    AMGX_DIST_PARTITION_OFFSETS = 1,
} AMGX_DIST_PARTITION_INFO;

/*********************************************************
 * Forward (opaque) handle declaration
 *********************************************************/
typedef void (*AMGX_print_callback)(const char *msg, int length);

typedef struct AMGX_config_handle_struct {char AMGX_config_handle_dummy;}
*AMGX_config_handle;

typedef struct AMGX_resources_handle_struct {char AMGX_resources_handle_dummy;}
*AMGX_resources_handle;

typedef struct AMGX_solver_handle_struct {char AMGX_solver_handle_dummy;}
*AMGX_solver_handle;

typedef struct AMGX_matrix_handle_struct {char AMGX_matrix_handle_dummy;}
*AMGX_matrix_handle;

typedef struct AMGX_vector_handle_struct {char AMGX_vector_handle_dummy;}
*AMGX_vector_handle;

typedef struct AMGX_distribution_handle_struct {char AMGX_distribution_handle_dummy;}
*AMGX_distribution_handle;

/*********************************************************
 * Print C-API error and exit
 *********************************************************/
#define AMGX_SAFE_CALL(rc) \
{ \
  AMGX_RC err;     \
  char msg[4096];   \
  switch(err = (rc)) {    \
  case AMGX_RC_OK: \
    break; \
  default: \
    fprintf(stderr, "AMGX ERROR: file %s line %6d\n", __FILE__, __LINE__); \
    AMGX_get_error_string(err, msg, 4096);\
    fprintf(stderr, "AMGX ERROR: %s\n", msg); \
    AMGX_abort(NULL,1);\
    break; \
  } \
}

/*********************************************************
 * C-API stable
 *********************************************************/
/* Build */
typedef AMGX_RC (*t_AMGX_get_api_version)
(int *major,
 int *minor);

typedef AMGX_RC (*t_AMGX_get_build_info_strings)
(char **version,
 char **date,
 char **time);

typedef AMGX_RC (*t_AMGX_get_error_string)
(AMGX_RC err,
 char *buf,
 int buf_len);

/* Init & Shutdown */
typedef AMGX_RC (*t_AMGX_initialize)();

typedef AMGX_RC (*t_AMGX_finalize)();

typedef void (*t_AMGX_abort)
(AMGX_resources_handle rsc,
 int err);

/* System */
typedef AMGX_RC (*t_AMGX_pin_memory)
(void *ptr,
 unsigned int bytes);

typedef AMGX_RC (*t_AMGX_unpin_memory)
(void *ptr);

typedef AMGX_RC (*t_AMGX_install_signal_handler)();

typedef AMGX_RC (*t_AMGX_reset_signal_handler)();

typedef AMGX_RC (*t_AMGX_register_print_callback)
(AMGX_print_callback func);

/* Config */
typedef AMGX_RC (*t_AMGX_config_create)
(AMGX_config_handle *cfg,
 const char *options);

typedef AMGX_RC (*t_AMGX_config_add_parameters)
(AMGX_config_handle *cfg,
 const char *options);

typedef AMGX_RC (*t_AMGX_config_create_from_file)
(AMGX_config_handle *cfg,
 const char *param_file);

typedef AMGX_RC (*t_AMGX_config_create_from_file_and_string)
(AMGX_config_handle *cfg,
 const char *param_file,
 const char *options);

typedef AMGX_RC (*t_AMGX_config_get_default_number_of_rings)
(AMGX_config_handle cfg,
 int *num_import_rings);

typedef AMGX_RC (*t_AMGX_config_destroy)
(AMGX_config_handle cfg);

/* Resources */
typedef AMGX_RC (*t_AMGX_resources_create)
(AMGX_resources_handle *rsc,
 AMGX_config_handle cfg,
 void *comm,
 int device_num,
 const int *devices);

typedef AMGX_RC (*t_AMGX_resources_create_simple)
(AMGX_resources_handle *rsc,
 AMGX_config_handle cfg);

typedef AMGX_RC (*t_AMGX_resources_destroy)
(AMGX_resources_handle rsc);

/* Distribution */
typedef AMGX_RC (*t_AMGX_distribution_create)
(AMGX_distribution_handle *dist, AMGX_config_handle cfg);

typedef AMGX_RC (*t_AMGX_distribution_destroy)
(AMGX_distribution_handle dist);

typedef AMGX_RC (*t_AMGX_distribution_set_partition_data)
(AMGX_distribution_handle dist, AMGX_DIST_PARTITION_INFO info, const void *partition_data);

typedef AMGX_RC (*t_AMGX_distribution_set_32bit_colindices)
(AMGX_distribution_handle dist, int use32bit);

/* Matrix */
typedef AMGX_RC (*t_AMGX_matrix_create)
(AMGX_matrix_handle *mtx,
 AMGX_resources_handle rsc,
 AMGX_Mode mode);

typedef AMGX_RC (*t_AMGX_matrix_destroy)
(AMGX_matrix_handle mtx);

typedef AMGX_RC (*t_AMGX_matrix_upload_all)
(AMGX_matrix_handle mtx,
 int n,
 int nnz,
 int block_dimx,
 int block_dimy,
 const int *row_ptrs,
 const int *col_indices,
 const void *data,
 const void *diag_data);

typedef AMGX_RC (*t_AMGX_matrix_replace_coefficients)
(AMGX_matrix_handle mtx,
 int n,
 int nnz,
 const void *data,
 const void *diag_data);

typedef AMGX_RC (*t_AMGX_matrix_get_size)
(const AMGX_matrix_handle mtx,
 int *n,
 int *block_dimx,
 int *block_dimy);

typedef AMGX_RC (*t_AMGX_matrix_get_nnz)
(const AMGX_matrix_handle mtx,
 int *nnz);

typedef AMGX_RC (*t_AMGX_matrix_download_all)
(const AMGX_matrix_handle mtx,
 int *row_ptrs,
 int *col_indices,
 void *data,
 void **diag_data);

typedef AMGX_RC (*t_AMGX_matrix_set_boundary_separation)
(AMGX_matrix_handle mtx,
 int boundary_separation);

typedef AMGX_RC (*t_AMGX_matrix_comm_from_maps)
(AMGX_matrix_handle mtx,
 int allocated_halo_depth,
 int num_import_rings,
 int num_neighbors,
 const int *neighbors,
 const int *send_ptrs,
 const int *send_maps,
 const int *recv_ptrs,
 const int *recv_maps);

typedef AMGX_RC (*t_AMGX_matrix_comm_from_maps_one_ring)
(AMGX_matrix_handle mtx,
 int allocated_halo_depth,
 int num_neighbors,
 const int *neighbors,
 const int *send_sizes,
 const int **send_maps,
 const int *recv_sizes,
 const int **recv_maps);

/* Vector */
typedef AMGX_RC (*t_AMGX_vector_create)
(AMGX_vector_handle *vec,
 AMGX_resources_handle rsc,
 AMGX_Mode mode);

typedef AMGX_RC (*t_AMGX_vector_destroy)
(AMGX_vector_handle vec);

typedef AMGX_RC (*t_AMGX_vector_upload)
(AMGX_vector_handle vec,
 int n,
 int block_dim,
 const void *data);

typedef AMGX_RC (*t_AMGX_vector_set_zero)
(AMGX_vector_handle vec,
 int n,
 int block_dim);

typedef AMGX_RC (*t_AMGX_vector_download)
(const AMGX_vector_handle vec,
 void *data);

typedef AMGX_RC (*t_AMGX_vector_get_size)
(const AMGX_vector_handle vec,
 int *n,
 int *block_dim);

typedef AMGX_RC (*t_AMGX_vector_bind)
(AMGX_vector_handle vec,
 const AMGX_matrix_handle mtx);

/* Solver */
typedef AMGX_RC (*t_AMGX_solver_create)
(AMGX_solver_handle *slv,
 AMGX_resources_handle rsc,
 AMGX_Mode mode,
 const AMGX_config_handle cfg);

typedef AMGX_RC (*t_AMGX_solver_destroy)
(AMGX_solver_handle slv);

typedef AMGX_RC (*t_AMGX_solver_setup)
(AMGX_solver_handle slv,
 AMGX_matrix_handle mtx);

typedef AMGX_RC (*t_AMGX_solver_solve)
(AMGX_solver_handle slv,
 AMGX_vector_handle rhs,
 AMGX_vector_handle sol);

typedef AMGX_RC (*t_AMGX_solver_solve_with_0_initial_guess)
(AMGX_solver_handle slv,
 AMGX_vector_handle rhs,
 AMGX_vector_handle sol);

typedef AMGX_RC (*t_AMGX_solver_get_iterations_number)
(AMGX_solver_handle slv,
 int *n);

typedef AMGX_RC (*t_AMGX_solver_get_iteration_residual)
(AMGX_solver_handle slv,
 int it,
 int idx,
 double *res);

typedef AMGX_RC (*t_AMGX_solver_get_status)
(AMGX_solver_handle slv,
 AMGX_SOLVE_STATUS *st);

/* Utilities */
typedef AMGX_RC (*t_AMGX_write_system)
(const AMGX_matrix_handle mtx,
 const AMGX_vector_handle rhs,
 const AMGX_vector_handle sol,
 const char *filename);

typedef AMGX_RC (*t_AMGX_write_system_distributed)
(const AMGX_matrix_handle mtx,
 const AMGX_vector_handle rhs,
 const AMGX_vector_handle sol,
 const char *filename,
 int allocated_halo_depth,
 int num_partitions,
 const int *partition_sizes,
 int partition_vector_size,
 const int *partition_vector);

typedef AMGX_RC (*t_AMGX_read_system)
(AMGX_matrix_handle mtx,
 AMGX_vector_handle rhs,
 AMGX_vector_handle sol,
 const char *filename);

typedef AMGX_RC (*t_AMGX_read_system_distributed)
(AMGX_matrix_handle mtx,
 AMGX_vector_handle rhs,
 AMGX_vector_handle sol,
 const char *filename,
 int allocated_halo_depth,
 int num_partitions,
 const int *partition_sizes,
 int partition_vector_size,
 const int *partition_vector);

typedef AMGX_RC (*t_AMGX_read_system_maps_one_ring)
(int *n,
 int *nnz,
 int *block_dimx,
 int *block_dimy,
 int **row_ptrs,
 int **col_indices,
 void **data,
 void **diag_data,
 void **rhs,
 void **sol,
 int *num_neighbors,
 int **neighbors,
 int **send_sizes,
 int ***send_maps,
 int **recv_sizes,
 int ***recv_maps,
 AMGX_resources_handle rsc,
 AMGX_Mode mode,
 const char *filename,
 int allocated_halo_depth,
 int num_partitions,
 const int *partition_sizes,
 int partition_vector_size,
 const int *partition_vector);

typedef AMGX_RC (*t_AMGX_free_system_maps_one_ring)
(int *row_ptrs,
 int *col_indices,
 void *data,
 void *diag_data,
 void *rhs,
 void *sol,
 int num_neighbors,
 int *neighbors,
 int *send_sizes,
 int **send_maps,
 int *recv_sizes,
 int **recv_maps);

typedef AMGX_RC (*t_AMGX_generate_distributed_poisson_7pt)
(AMGX_matrix_handle mtx,
 AMGX_vector_handle rhs,
 AMGX_vector_handle sol,
 int allocated_halo_depth,
 int num_import_rings,
 int nx,
 int ny,
 int nz,
 int px,
 int py,
 int pz);

/*********************************************************
 * C-API experimental
 *********************************************************/
typedef AMGX_RC (*t_AMGX_matrix_attach_coloring)
(AMGX_matrix_handle mtx,
 int *row_coloring,
 int num_rows,
 int num_colors);

typedef AMGX_RC (*t_AMGX_matrix_attach_geometry)
(AMGX_matrix_handle mtx,
 double *geox,
 double *geoy,
 double *geoz,
 int n);

typedef AMGX_RC (*t_AMGX_read_system_global)
(int *n,
 int *nnz,
 int *block_dimx,
 int *block_dimy,
 int **row_ptrs,
 void **col_indices_global,
 void **data,
 void **diag_data,
 void **rhs,
 void **sol,
 AMGX_resources_handle rsc,
 AMGX_Mode mode,
 const char *filename,
 int allocated_halo_depth,
 int num_partitions,
 const int *partition_sizes,
 int partition_vector_size,
 const int *partition_vector);

typedef AMGX_RC (*t_AMGX_matrix_upload_all_global)
(AMGX_matrix_handle mtx,
 int n_global,
 int n,
 int nnz,
 int block_dimx,
 int block_dimy,
 const int *row_ptrs,
 const void *col_indices_global,
 const void *data,
 const void *diag_data,
 int allocated_halo_depth,
 int num_import_rings,
 const int *partition_vector);

typedef AMGX_RC (*t_AMGX_matrix_upload_all_global_32)
(AMGX_matrix_handle mtx,
 int n_global,
 int n,
 int nnz,
 int block_dimx,
 int block_dimy,
 const int *row_ptrs,
 const void *col_indices_global,
 const void *data,
 const void *diag_data,
 int allocated_halo_depth,
 int num_import_rings,
 const int *partition_vector);

typedef AMGX_RC (*t_AMGX_matrix_upload_distributed)
(AMGX_matrix_handle mtx,
 int n_global,
 int n,
 int nnz,
 int block_dimx,
 int block_dimy,
 const int *row_ptrs,
 const void *col_indices_global,
 const void *data,
 const void *diag_data,
 AMGX_distribution_handle distribution);

/*********************************************************
 * C-API deprecated
 *********************************************************/
typedef AMGX_RC (*t_AMGX_solver_register_print_callback)
(AMGX_print_callback func);

typedef AMGX_RC (*t_AMGX_solver_resetup)
(AMGX_solver_handle slv,
 AMGX_matrix_handle mtx);


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
/* Build */
t_AMGX_get_api_version                    AMGX_get_api_version;
t_AMGX_get_build_info_strings             AMGX_get_build_info_strings;
t_AMGX_get_error_string                   AMGX_get_error_string;
/* Init & Shutdown */
t_AMGX_initialize                         AMGX_initialize;
t_AMGX_finalize                           AMGX_finalize;
t_AMGX_abort                              AMGX_abort;
/* System */
t_AMGX_pin_memory                         AMGX_pin_memory;
t_AMGX_unpin_memory                       AMGX_unpin_memory;
t_AMGX_install_signal_handler             AMGX_install_signal_handler;
t_AMGX_reset_signal_handler               AMGX_reset_signal_handler;
t_AMGX_register_print_callback            AMGX_register_print_callback;
/* Config */
t_AMGX_config_create                      AMGX_config_create;
t_AMGX_config_add_parameters              AMGX_config_add_parameters;
t_AMGX_config_create_from_file            AMGX_config_create_from_file;
t_AMGX_config_create_from_file_and_string AMGX_config_create_from_file_and_string;
t_AMGX_config_get_default_number_of_rings AMGX_config_get_default_number_of_rings;
t_AMGX_config_destroy                     AMGX_config_destroy;
/* Resources */
t_AMGX_resources_create                   AMGX_resources_create;
t_AMGX_resources_create_simple            AMGX_resources_create_simple;
t_AMGX_resources_destroy                  AMGX_resources_destroy;
/* Distribution */
t_AMGX_distribution_create                AMGX_distribution_create;
t_AMGX_distribution_destroy               AMGX_distribution_destroy;
t_AMGX_distribution_set_partition_data    AMGX_distribution_set_partition_data;
t_AMGX_distribution_set_32bit_colindices  AMGX_distribution_set_32bit_colindices;


/* Matrix */
t_AMGX_matrix_create                      AMGX_matrix_create;
t_AMGX_matrix_destroy                     AMGX_matrix_destroy;
t_AMGX_matrix_upload_all                  AMGX_matrix_upload_all;
t_AMGX_matrix_replace_coefficients        AMGX_matrix_replace_coefficients;
t_AMGX_matrix_get_size                    AMGX_matrix_get_size;
t_AMGX_matrix_get_nnz                     AMGX_matrix_get_nnz;
t_AMGX_matrix_download_all                AMGX_matrix_download_all;
t_AMGX_matrix_set_boundary_separation     AMGX_matrix_set_boundary_separation;
t_AMGX_matrix_comm_from_maps              AMGX_matrix_comm_from_maps;
t_AMGX_matrix_comm_from_maps_one_ring     AMGX_matrix_comm_from_maps_one_ring;
/* Vector */
t_AMGX_vector_create                      AMGX_vector_create;
t_AMGX_vector_destroy                     AMGX_vector_destroy;
t_AMGX_vector_upload                      AMGX_vector_upload;
t_AMGX_vector_set_zero                    AMGX_vector_set_zero;
t_AMGX_vector_download                    AMGX_vector_download;
t_AMGX_vector_get_size                    AMGX_vector_get_size;
t_AMGX_vector_bind                        AMGX_vector_bind;
/* Solver */
t_AMGX_solver_create                      AMGX_solver_create;
t_AMGX_solver_destroy                     AMGX_solver_destroy;
t_AMGX_solver_setup                       AMGX_solver_setup;
t_AMGX_solver_solve                       AMGX_solver_solve;
t_AMGX_solver_solve_with_0_initial_guess  AMGX_solver_solve_with_0_initial_guess;
t_AMGX_solver_get_iterations_number       AMGX_solver_get_iterations_number;
t_AMGX_solver_get_iteration_residual      AMGX_solver_get_iteration_residual;
t_AMGX_solver_get_status                  AMGX_solver_get_status;
/* Utilities */
t_AMGX_write_system                       AMGX_write_system;
t_AMGX_write_system_distributed           AMGX_write_system_distributed;
t_AMGX_read_system                        AMGX_read_system;
t_AMGX_read_system_distributed            AMGX_read_system_distributed;
t_AMGX_read_system_maps_one_ring          AMGX_read_system_maps_one_ring;
t_AMGX_free_system_maps_one_ring          AMGX_free_system_maps_one_ring;
t_AMGX_generate_distributed_poisson_7pt   AMGX_generate_distributed_poisson_7pt;
/* Experimental */
t_AMGX_matrix_attach_geometry             AMGX_matrix_attach_geometry;
t_AMGX_matrix_attach_coloring             AMGX_matrix_attach_coloring;
t_AMGX_read_system_global                   AMGX_read_system_global;
t_AMGX_matrix_upload_all_global             AMGX_matrix_upload_all_global;
t_AMGX_matrix_upload_all_global_32       AMGX_matrix_upload_all_global_32;
t_AMGX_matrix_upload_distributed         AMGX_matrix_upload_distributed;

/* dynamically load the library, return: 1 - succeeded, 0 - failed*/
int amgx_liblink_all(void *lib_handle)
{
    /* Build */
    AMGX_get_api_version                     = (t_AMGX_get_api_version)amgx_liblink(lib_handle, "AMGX_get_api_version");
    AMGX_get_build_info_strings              = (t_AMGX_get_build_info_strings)amgx_liblink(lib_handle, "AMGX_get_build_info_strings");
    AMGX_get_error_string                    = (t_AMGX_get_error_string)amgx_liblink(lib_handle, "AMGX_get_error_string");
    /* Init & Shutdown */
    AMGX_initialize                          = (t_AMGX_initialize)amgx_liblink(lib_handle, "AMGX_initialize");
    AMGX_finalize                            = (t_AMGX_finalize)amgx_liblink(lib_handle, "AMGX_finalize");
    AMGX_abort                               = (t_AMGX_abort)amgx_liblink(lib_handle, "AMGX_abort");
    /* System */
    AMGX_pin_memory                          = (t_AMGX_pin_memory)amgx_liblink(lib_handle, "AMGX_pin_memory");
    AMGX_unpin_memory                        = (t_AMGX_unpin_memory)amgx_liblink(lib_handle, "AMGX_unpin_memory");
    AMGX_install_signal_handler              = (t_AMGX_install_signal_handler)amgx_liblink(lib_handle, "AMGX_install_signal_handler");
    AMGX_reset_signal_handler                = (t_AMGX_reset_signal_handler)amgx_liblink(lib_handle, "AMGX_reset_signal_handler");
    AMGX_register_print_callback             = (t_AMGX_register_print_callback)amgx_liblink(lib_handle, "AMGX_register_print_callback");
    /* Config */
    AMGX_config_create                       = (t_AMGX_config_create)amgx_liblink(lib_handle, "AMGX_config_create");
    AMGX_config_add_parameters               = (t_AMGX_config_add_parameters)amgx_liblink(lib_handle, "AMGX_config_add_parameters");
    AMGX_config_create_from_file             = (t_AMGX_config_create_from_file)amgx_liblink(lib_handle, "AMGX_config_create_from_file");
    AMGX_config_create_from_file_and_string  = (t_AMGX_config_create_from_file_and_string)amgx_liblink(lib_handle, "AMGX_config_create_from_file_and_string");
    AMGX_config_get_default_number_of_rings  = (t_AMGX_config_get_default_number_of_rings)amgx_liblink(lib_handle, "AMGX_config_get_default_number_of_rings");
    AMGX_config_destroy                      = (t_AMGX_config_destroy)amgx_liblink(lib_handle, "AMGX_config_destroy");
    /* Resources */
    AMGX_resources_create                    = (t_AMGX_resources_create)amgx_liblink(lib_handle, "AMGX_resources_create");
    AMGX_resources_create_simple             = (t_AMGX_resources_create_simple)amgx_liblink(lib_handle, "AMGX_resources_create_simple");
    AMGX_resources_destroy                   = (t_AMGX_resources_destroy)amgx_liblink(lib_handle, "AMGX_resources_destroy");
    /* Distribution */
    AMGX_distribution_create                 = (t_AMGX_distribution_create)amgx_liblink(lib_handle, "AMGX_distribution_create");
    AMGX_distribution_destroy                = (t_AMGX_distribution_destroy)amgx_liblink(lib_handle, "AMGX_distribution_destroy");
    AMGX_distribution_set_partition_data     = (t_AMGX_distribution_set_partition_data)amgx_liblink(lib_handle, "AMGX_distribution_set_partition_data");
    AMGX_distribution_set_32bit_colindices   = (t_AMGX_distribution_set_32bit_colindices)amgx_liblink(lib_handle, "AMGX_distribution_set_32bit_colindices");
    /* Matrix */
    AMGX_matrix_create                       = (t_AMGX_matrix_create)amgx_liblink(lib_handle, "AMGX_matrix_create");
    AMGX_matrix_destroy                      = (t_AMGX_matrix_destroy)amgx_liblink(lib_handle, "AMGX_matrix_destroy");
    AMGX_matrix_upload_all                   = (t_AMGX_matrix_upload_all)amgx_liblink(lib_handle, "AMGX_matrix_upload_all");
    AMGX_matrix_replace_coefficients         = (t_AMGX_matrix_replace_coefficients)amgx_liblink(lib_handle, "AMGX_matrix_replace_coefficients");
    AMGX_matrix_get_size                     = (t_AMGX_matrix_get_size)amgx_liblink(lib_handle, "AMGX_matrix_get_size");
    AMGX_matrix_get_nnz                      = (t_AMGX_matrix_get_nnz)amgx_liblink(lib_handle, "AMGX_matrix_get_nnz");
    AMGX_matrix_download_all                 = (t_AMGX_matrix_download_all)amgx_liblink(lib_handle, "AMGX_matrix_download_all");
    AMGX_matrix_set_boundary_separation      = (t_AMGX_matrix_set_boundary_separation)amgx_liblink(lib_handle, "AMGX_matrix_set_boundary_separation");
    AMGX_matrix_comm_from_maps               = (t_AMGX_matrix_comm_from_maps)amgx_liblink(lib_handle, "AMGX_matrix_comm_from_maps");
    AMGX_matrix_comm_from_maps_one_ring      = (t_AMGX_matrix_comm_from_maps_one_ring)amgx_liblink(lib_handle, "AMGX_matrix_comm_from_maps_one_ring");
    /* Vector */
    AMGX_vector_create                       = (t_AMGX_vector_create)amgx_liblink(lib_handle, "AMGX_vector_create");
    AMGX_vector_destroy                      = (t_AMGX_vector_destroy)amgx_liblink(lib_handle, "AMGX_vector_destroy");
    AMGX_vector_upload                       = (t_AMGX_vector_upload)amgx_liblink(lib_handle, "AMGX_vector_upload");
    AMGX_vector_set_zero                     = (t_AMGX_vector_set_zero)amgx_liblink(lib_handle, "AMGX_vector_set_zero");
    AMGX_vector_download                     = (t_AMGX_vector_download)amgx_liblink(lib_handle, "AMGX_vector_download");
    AMGX_vector_get_size                     = (t_AMGX_vector_get_size)amgx_liblink(lib_handle, "AMGX_vector_get_size");
    AMGX_vector_bind                         = (t_AMGX_vector_bind)amgx_liblink(lib_handle, "AMGX_vector_bind");
    /* Solver */
    AMGX_solver_create                       = (t_AMGX_solver_create)amgx_liblink(lib_handle, "AMGX_solver_create");
    AMGX_solver_destroy                      = (t_AMGX_solver_destroy)amgx_liblink(lib_handle, "AMGX_solver_destroy");
    AMGX_solver_setup                        = (t_AMGX_solver_setup)amgx_liblink(lib_handle, "AMGX_solver_setup");
    AMGX_solver_solve                        = (t_AMGX_solver_solve)amgx_liblink(lib_handle, "AMGX_solver_solve");
    AMGX_solver_get_iterations_number        = (t_AMGX_solver_get_iterations_number)amgx_liblink(lib_handle, "AMGX_solver_get_iterations_number");
    AMGX_solver_get_iteration_residual       = (t_AMGX_solver_get_iteration_residual)amgx_liblink(lib_handle, "AMGX_solver_get_iteration_residual");
    AMGX_solver_get_status                   = (t_AMGX_solver_get_status)amgx_liblink(lib_handle, "AMGX_solver_get_status");
    /* Utilities */
    AMGX_write_system                        = (t_AMGX_write_system)amgx_liblink(lib_handle, "AMGX_write_system");
    AMGX_write_system_distributed            = (t_AMGX_write_system_distributed)amgx_liblink(lib_handle, "AMGX_write_system_distributed");
    AMGX_read_system                         = (t_AMGX_read_system)amgx_liblink(lib_handle, "AMGX_read_system");
    AMGX_read_system_distributed             = (t_AMGX_read_system_distributed)amgx_liblink(lib_handle, "AMGX_read_system_distributed");
    AMGX_read_system_maps_one_ring           = (t_AMGX_read_system_maps_one_ring)amgx_liblink(lib_handle, "AMGX_read_system_maps_one_ring");
    AMGX_free_system_maps_one_ring           = (t_AMGX_free_system_maps_one_ring)amgx_liblink(lib_handle, "AMGX_free_system_maps_one_ring");
    AMGX_generate_distributed_poisson_7pt    = (t_AMGX_generate_distributed_poisson_7pt)amgx_liblink(lib_handle, "AMGX_generate_distributed_poisson_7pt");
    /* Experimental */
    AMGX_matrix_attach_geometry              = (t_AMGX_matrix_attach_geometry)amgx_liblink(lib_handle, "AMGX_matrix_attach_geometry");
    AMGX_matrix_attach_coloring              = (t_AMGX_matrix_attach_coloring)amgx_liblink(lib_handle, "AMGX_matrix_attach_coloring");
    AMGX_read_system_global                  = (t_AMGX_read_system_global)amgx_liblink(lib_handle, "AMGX_read_system_global");
    AMGX_matrix_upload_all_global            = (t_AMGX_matrix_upload_all_global)amgx_liblink(lib_handle, "AMGX_matrix_upload_all_global");
    AMGX_matrix_upload_all_global_32        = (t_AMGX_matrix_upload_all_global_32)amgx_liblink(lib_handle, "AMGX_matrix_upload_all_global_32");
    AMGX_matrix_upload_distributed          = (t_AMGX_matrix_upload_distributed)amgx_liblink(lib_handle, "AMGX_matrix_upload_distributed");

    if (/* Build */
        AMGX_get_api_version == NULL ||
        AMGX_get_build_info_strings == NULL ||
        AMGX_get_error_string == NULL ||
        /* Init & Shutdown */
        AMGX_initialize == NULL ||
        AMGX_finalize == NULL ||
        AMGX_abort == NULL ||
        /* System */
        AMGX_pin_memory == NULL ||
        AMGX_unpin_memory == NULL ||
        AMGX_install_signal_handler == NULL ||
        AMGX_reset_signal_handler == NULL ||
        AMGX_register_print_callback == NULL ||
        /* Config */
        AMGX_config_create == NULL ||
        AMGX_config_add_parameters == NULL ||
        AMGX_config_create_from_file == NULL ||
        AMGX_config_create_from_file_and_string == NULL ||
        AMGX_config_get_default_number_of_rings == NULL ||
        AMGX_config_destroy == NULL ||
        /* Resources */
        AMGX_resources_create == NULL ||
        AMGX_resources_create_simple == NULL ||
        AMGX_resources_destroy == NULL ||
        /* Distribution */
        AMGX_distribution_create == NULL ||
        AMGX_distribution_destroy == NULL ||
        AMGX_distribution_set_partition_data == NULL ||
        AMGX_distribution_set_32bit_colindices == NULL ||
        /* Matrix */
        AMGX_matrix_create == NULL ||
        AMGX_matrix_destroy == NULL ||
        AMGX_matrix_upload_all == NULL ||
        AMGX_matrix_replace_coefficients == NULL ||
        AMGX_matrix_get_size == NULL ||
        AMGX_matrix_get_nnz  == NULL ||
        AMGX_matrix_download_all  == NULL ||
        AMGX_matrix_set_boundary_separation == NULL ||
        AMGX_matrix_comm_from_maps == NULL ||
        AMGX_matrix_comm_from_maps_one_ring == NULL ||
        /* Vector */
        AMGX_vector_create == NULL ||
        AMGX_vector_destroy == NULL ||
        AMGX_vector_upload == NULL ||
        AMGX_vector_set_zero == NULL ||
        AMGX_vector_download == NULL ||
        AMGX_vector_get_size == NULL ||
        AMGX_vector_bind == NULL ||
        /* Solver */
        AMGX_solver_create == NULL ||
        AMGX_solver_destroy == NULL ||
        AMGX_solver_setup == NULL ||
        AMGX_solver_solve == NULL ||
        AMGX_solver_get_iterations_number == NULL ||
        AMGX_solver_get_iteration_residual == NULL ||
        AMGX_solver_get_status == NULL ||
        /* Utilities */
        AMGX_write_system == NULL ||
        AMGX_write_system_distributed == NULL ||
        AMGX_read_system == NULL ||
        AMGX_read_system_distributed == NULL ||
        AMGX_read_system_maps_one_ring == NULL ||
        AMGX_free_system_maps_one_ring == NULL ||
        AMGX_generate_distributed_poisson_7pt == NULL ||
        /* Experimental */
        AMGX_matrix_attach_geometry == NULL ||
        AMGX_matrix_attach_coloring == NULL ||
        AMGX_read_system_global == NULL ||
        AMGX_matrix_upload_all_global == NULL ||
        AMGX_matrix_upload_all_global_32 == NULL ||
        AMGX_matrix_upload_distributed == NULL)
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
