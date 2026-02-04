// SPDX-FileCopyrightText: 2011 - 2025 NVIDIA CORPORATION. All Rights Reserved.
//
// SPDX-License-Identifier: BSD-3-Clause

#ifndef __AMGX_C_H_INCLUDE__
#define __AMGX_C_H_INCLUDE__

#ifdef SWIG
#define AMGX_API
#else
#ifdef _WIN32
#ifdef AMGX_API_EXPORTS
#define AMGX_API __declspec( dllexport )
#else
#ifdef AMGX_API_NO_IMPORTS
#define AMGX_API
#else
#define AMGX_API __declspec( dllimport )
#endif //AMGX_API_NO_IMPORTS
#endif //AMGX_API_EXPORTS
#else
#define AMGX_API __attribute__((visibility ("default")))
#endif //_WIN32
#endif //SWIG (swig doesn't seem to work with __declspec/__attribute__)


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
    AMGX_SOLVE_NOT_CONVERGED = 3,
} AMGX_SOLVE_STATUS;

/*********************************************************
 * Flags to retrieve parameters description
 *********************************************************/
typedef enum
{
    AMGX_GET_PARAMS_DESC_JSON_TO_FILE = 0,
    AMGX_GET_PARAMS_DESC_JSON_TO_STRING = 1,
    AMGX_GET_PARAMS_DESC_TEXT_TO_FILE = 2,
    AMGX_GET_PARAMS_DESC_TEXT_TO_STRING = 3
} AMGX_GET_PARAMS_DESC_FLAG;

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

typedef struct {char AMGX_config_handle_dummy;} AMGX_config_handle_struct;
typedef AMGX_config_handle_struct *AMGX_config_handle;

typedef struct {char AMGX_resources_handle_dummy;} AMGX_resources_handle_struct;
typedef AMGX_resources_handle_struct *AMGX_resources_handle;

typedef struct {char AMGX_matrix_handle_dummy;} AMGX_matrix_handle_struct;
typedef AMGX_matrix_handle_struct *AMGX_matrix_handle;

typedef struct {char AMGX_vector_handle_dummy;} AMGX_vector_handle_struct;
typedef AMGX_vector_handle_struct *AMGX_vector_handle;

typedef struct {char AMGX_solver_handle_dummy;} AMGX_solver_handle_struct;
typedef AMGX_solver_handle_struct *AMGX_solver_handle;

typedef struct {char AMGX_distribution_handle_dummy;} AMGX_distribution_handle_struct;
/** Stores parameters about global matrix distribution and upload */
typedef AMGX_distribution_handle_struct *AMGX_distribution_handle;


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
AMGX_RC AMGX_API AMGX_get_api_version
(int *major,
 int *minor);

AMGX_RC AMGX_API AMGX_get_build_info_strings
(char **version,
 char **date,
 char **time);

AMGX_RC AMGX_API AMGX_get_error_string
(AMGX_RC err,
 char *buf,
 int buf_len);

/* Init & Shutdown */
AMGX_RC AMGX_API AMGX_initialize();

AMGX_RC AMGX_API AMGX_initialize_plugins();

AMGX_RC AMGX_API AMGX_finalize();

AMGX_RC AMGX_API AMGX_finalize_plugins();

void AMGX_API AMGX_abort
(AMGX_resources_handle rsrc,
 int err);

/* System */
AMGX_RC AMGX_API AMGX_pin_memory
(void *ptr,
 unsigned int bytes);

AMGX_RC AMGX_API AMGX_unpin_memory
(void *ptr);

AMGX_RC AMGX_API AMGX_install_signal_handler();

AMGX_RC AMGX_API AMGX_reset_signal_handler();

AMGX_RC AMGX_API AMGX_register_print_callback
(AMGX_print_callback func);

/* Config */
AMGX_RC AMGX_API AMGX_config_create
(AMGX_config_handle *cfg,
 const char *options);

AMGX_RC AMGX_API AMGX_config_add_parameters
(AMGX_config_handle *cfg,
 const char *options);

AMGX_RC AMGX_API AMGX_config_create_from_file
(AMGX_config_handle *cfg,
 const char *param_file);

AMGX_RC AMGX_API AMGX_config_create_from_file_and_string
(AMGX_config_handle *cfg,
 const char *param_file,
 const char *options);

AMGX_RC AMGX_API AMGX_config_get_default_number_of_rings
(AMGX_config_handle cfg,
 int *num_import_rings);

AMGX_RC AMGX_API AMGX_config_destroy
(AMGX_config_handle cfg);

/* Resources */
AMGX_RC AMGX_API AMGX_resources_create
(AMGX_resources_handle *rsc,
 AMGX_config_handle cfg,
 void *comm,
 int device_num,
 const int *devices);

AMGX_RC AMGX_API AMGX_resources_create_simple
(AMGX_resources_handle *rsc,
 AMGX_config_handle cfg);

AMGX_RC AMGX_API AMGX_resources_destroy
(AMGX_resources_handle rsc);

/* Distribution */
/** Create a distribution handle.
 * `cfg` is used to set parameters from an existing configuration. Can be null. */
AMGX_RC AMGX_API AMGX_distribution_create
(AMGX_distribution_handle *dist, AMGX_config_handle cfg);

AMGX_RC AMGX_API AMGX_distribution_destroy
(AMGX_distribution_handle dist);

/** Set the partitioning scheme used for the matrix.
 * 
 * AMGX_DIST_PARTITION_VECTOR:
 *  Pass in a partition vector of type `int` for `partition_data` with the same format as for AMGX_matrix_upload_all_global().
 * AMGX_DIST_PARTITION_OFFSETS:
 *  For a contiguous partitioning, specifying the offsets allows faster matrix upload.
 *  In this case, `partition_data` must be `int` or `int64_t` array, matching the column index data type.
 * 
 * Use with \see AMGX_matrix_upload_distributed()
*/
AMGX_RC AMGX_API AMGX_distribution_set_partition_data
(AMGX_distribution_handle dist, AMGX_DIST_PARTITION_INFO info, const void *partition_data);

/** Set whether to use 32-bit or 64-bit column indices. Default is 64 bit.
 * 
 * Determines how the `col_indices_global` argument to AMGX_matrix_upload_distributed() is interpreted.
 */
AMGX_RC AMGX_API AMGX_distribution_set_32bit_colindices
(AMGX_distribution_handle dist, int use32bit);

/* Matrix */
AMGX_RC AMGX_API AMGX_matrix_create
(AMGX_matrix_handle *mtx,
 AMGX_resources_handle rsc,
 AMGX_Mode mode);

AMGX_RC AMGX_API AMGX_matrix_destroy
(AMGX_matrix_handle mtx);

AMGX_RC AMGX_API AMGX_matrix_upload_all
(AMGX_matrix_handle mtx,
 int n,
 int nnz,
 int block_dimx,
 int block_dimy,
 const int *row_ptrs,
 const int *col_indices,
 const void *data,
 const void *diag_data);

AMGX_RC AMGX_API AMGX_matrix_replace_coefficients
(AMGX_matrix_handle mtx,
 int n,
 int nnz,
 const void *data,
 const void *diag_data);

AMGX_RC AMGX_API AMGX_matrix_get_size
(const AMGX_matrix_handle mtx,
 int *n,
 int *block_dimx,
 int *block_dimy);

AMGX_RC AMGX_API AMGX_matrix_get_nnz
(const AMGX_matrix_handle mtx,
 int *nnz);

AMGX_RC AMGX_API AMGX_matrix_download_all
(const AMGX_matrix_handle mtx,
 int *row_ptrs,
 int *col_indices,
 void *data,
 void **diag_data);

AMGX_RC AMGX_API AMGX_matrix_vector_multiply
(AMGX_matrix_handle mtx,
 AMGX_vector_handle x,
 AMGX_vector_handle y);

AMGX_RC AMGX_API AMGX_matrix_set_boundary_separation
(AMGX_matrix_handle mtx,
 int boundary_separation);

AMGX_RC AMGX_API AMGX_matrix_comm_from_maps
(AMGX_matrix_handle mtx,
 int allocated_halo_depth,
 int num_import_rings,
 int max_num_neighbors,
 const int *neighbors,
 const int *send_ptrs,
 const int *send_maps,
 const int *recv_ptrs,
 const int *recv_maps);

AMGX_RC AMGX_API AMGX_matrix_comm_from_maps_one_ring
(AMGX_matrix_handle mtx,
 int allocated_halo_depth,
 int num_neighbors,
 const int *neighbors,
 const int *send_sizes,
 const int **send_maps,
 const int *recv_sizes,
 const int **recv_maps);

/* Vector */
AMGX_RC AMGX_API AMGX_vector_create
(AMGX_vector_handle *vec,
 AMGX_resources_handle rsc,
 AMGX_Mode mode);

AMGX_RC AMGX_API AMGX_vector_destroy
(AMGX_vector_handle vec);

AMGX_RC AMGX_API AMGX_vector_upload
(AMGX_vector_handle vec,
 int n,
 int block_dim,
 const void *data);

AMGX_RC AMGX_API AMGX_vector_set_zero
(AMGX_vector_handle vec,
 int n,
 int block_dim);

AMGX_RC AMGX_API AMGX_vector_set_random
(AMGX_vector_handle vec,
 int n);

AMGX_RC AMGX_API AMGX_vector_download
(const AMGX_vector_handle vec,
 void *data);

AMGX_RC AMGX_API AMGX_vector_get_size
(const AMGX_vector_handle vec,
 int *n,
 int *block_dim);

AMGX_RC AMGX_API AMGX_vector_bind
(AMGX_vector_handle vec,
 const AMGX_matrix_handle mtx);

/* Solver */
AMGX_RC AMGX_API AMGX_solver_create
(AMGX_solver_handle *slv,
 AMGX_resources_handle rsc,
 AMGX_Mode mode,
 const AMGX_config_handle cfg_solver);

AMGX_RC AMGX_API AMGX_solver_destroy
(AMGX_solver_handle slv);

AMGX_RC AMGX_API AMGX_solver_setup
(AMGX_solver_handle slv,
 AMGX_matrix_handle mtx);

AMGX_RC AMGX_API AMGX_solver_solve
(AMGX_solver_handle slv,
 AMGX_vector_handle rhs,
 AMGX_vector_handle sol);

AMGX_RC AMGX_API AMGX_solver_solve_with_0_initial_guess
(AMGX_solver_handle slv,
 AMGX_vector_handle rhs,
 AMGX_vector_handle sol);

AMGX_RC AMGX_API AMGX_solver_get_iterations_number
(AMGX_solver_handle slv,
 int *n);

AMGX_RC AMGX_API AMGX_solver_get_iteration_residual
(AMGX_solver_handle slv,
 int it,
 int idx,
 double *res);

AMGX_RC AMGX_API AMGX_solver_get_status
(AMGX_solver_handle slv,
 AMGX_SOLVE_STATUS *st);

AMGX_RC AMGX_API AMGX_solver_calculate_residual_norm
(AMGX_solver_handle solver,
 AMGX_matrix_handle mtx,
 AMGX_vector_handle rhs,
 AMGX_vector_handle x,
 void *norm_vector);

/* Utilities */
AMGX_RC AMGX_API AMGX_write_system
(const AMGX_matrix_handle mtx,
 const AMGX_vector_handle rhs,
 const AMGX_vector_handle sol,
 const char *filename);

AMGX_RC AMGX_API AMGX_write_system_distributed
(const AMGX_matrix_handle mtx,
 const AMGX_vector_handle rhs,
 const AMGX_vector_handle sol,
 const char *filename,
 int allocated_halo_depth,
 int num_partitions,
 const int *partition_sizes,
 int partition_vector_size,
 const int *partition_vector);

AMGX_RC AMGX_API AMGX_read_system
(AMGX_matrix_handle mtx,
 AMGX_vector_handle rhs,
 AMGX_vector_handle sol,
 const char *filename);

AMGX_RC AMGX_API AMGX_read_system_distributed
(AMGX_matrix_handle mtx,
 AMGX_vector_handle rhs,
 AMGX_vector_handle sol,
 const char *filename,
 int allocated_halo_depth,
 int num_partitions,
 const int *partition_sizes,
 int partition_vector_size,
 const int *partition_vector);

AMGX_RC AMGX_API AMGX_read_system_maps_one_ring
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

AMGX_RC AMGX_API AMGX_free_system_maps_one_ring
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

AMGX_RC AMGX_API AMGX_generate_distributed_poisson_7pt
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

AMGX_RC AMGX_API AMGX_write_parameters_description
(char *filename,
 AMGX_GET_PARAMS_DESC_FLAG mode);

/*********************************************************
 * C-API experimental
 *********************************************************/
AMGX_RC AMGX_API AMGX_matrix_attach_coloring
(AMGX_matrix_handle mtx,
 int *row_coloring,
 int num_rows,
 int num_colors);

AMGX_RC AMGX_API AMGX_matrix_attach_geometry
(AMGX_matrix_handle mtx,
 double *geox,
 double *geoy,
 double *geoz,
 int n);

AMGX_RC AMGX_API AMGX_read_system_global
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

AMGX_RC AMGX_API AMGX_matrix_upload_all_global
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
 
AMGX_RC AMGX_API AMGX_matrix_upload_all_global_32
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

 AMGX_RC AMGX_API AMGX_matrix_upload_distributed
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

AMGX_RC AMGX_API AMGX_matrix_check_symmetry
(AMGX_matrix_handle mtx,
 int* structurally_symmetric,
 int* symmetric);

AMGX_RC AMGX_API AMGX_matrix_check_diag_dominant
(const AMGX_matrix_handle mtx, 
 int* diag_dominant);

/*********************************************************
 * C-API deprecated
 *********************************************************/
AMGX_RC AMGX_API AMGX_solver_register_print_callback
(AMGX_print_callback func);

AMGX_RC AMGX_API AMGX_solver_resetup
(AMGX_solver_handle slv,
 AMGX_matrix_handle mtx);

#if defined(__cplusplus)
}//extern "C"
#endif

#endif
