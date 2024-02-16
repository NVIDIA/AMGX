// SPDX-FileCopyrightText: 2011 - 2024 NVIDIA CORPORATION. All Rights Reserved.
//
// SPDX-License-Identifier: BSD-3-Clause

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <mpi.h>
#include <cuda_runtime_api.h>
#include "amgx_c.h"
#include "amgx_eig_c.h"

#define MAX_MSG_SZ 1024

void print_callback(const char *msg, int length)
{
  int rank;
  MPI_Comm_rank( MPI_COMM_WORLD, &rank );
  if (rank == 0) printf("%s", msg);
}

int findParamIndex(char** argv, int argc, const char* parm)
{

  int count = 0;
  int index = -1;
  for (int i = 0; i < argc; i++)
  {
    if (strncmp(argv[i], parm, 100)==0)
    {
      index = i;
      count++;
    }
  }

  if (count == 0 || count == 1)
  {
    return index;
  }
  else
  {
    char msg[MAX_MSG_SZ];
    sprintf(msg,"Error, parameter %s has been specified more than once, exiting\n",parm);
    print_callback(msg, MAX_MSG_SZ);
    exit(1);
  }
  return -1;
}

void errAndExit(const char* err)
{
  printf("%s\n", err);
  fflush(stdout);
  MPI_Abort(MPI_COMM_WORLD, 1);
  //printUsageAndExit();
}

void printUsageAndExit()
{
  char msg[MAX_MSG_SZ] = "Usage: mpirun [-n num_procs] ./eigensolver_mpi [-help] [-mode [dDDI | dDFI | dFFI] [-m mformat mfile] [-c config_file]\n";
  strcat(msg,"     -help display the command options\n");
  strcat(msg,"     -m mfile: reads matrix stored in file mfile\n");
  strcat(msg,"     -mode : which of the solver modes to use\n");
  print_callback(msg, MAX_MSG_SZ);
  MPI_Finalize();
  exit(0);
}

int main(int argc, char **argv)
{

  int partitions = 0;
  int part = 0;
  int local_rank = 0;

  MPI_Init( &argc, &argv );

  // Selecting a group of ranks
  MPI_Comm amgx_mpi_comm = MPI_COMM_WORLD;
  MPI_Comm_size( amgx_mpi_comm, &partitions );
  MPI_Comm_rank( amgx_mpi_comm, &part );

  int gpu_count = 0;
  // calculate local rank assuming equal number of GPUs per node and one GPU per rank
  cudaGetDeviceCount(&gpu_count);
  local_rank = part - (part/gpu_count)*gpu_count;

  printf("Process %d selecting device %d\n", part, local_rank);

  cudaSetDevice(local_rank);

  int pidx = 0;

  if (argc == 1)
  {
    printUsageAndExit();
  }

  if ((pidx = findParamIndex(argv, argc, "--version")) != -1)
  {
    char *ver, *date, *time;

  AMGX_get_build_info_strings(&ver, &date, &time);
  printf("amgx build version: %s\nBuild date and time: %s %s\n", ver, date, time);
  MPI_Finalize();
  exit(0);
  }

  AMGX_SAFE_CALL(AMGX_solver_register_print_callback(&print_callback));

  AMGX_SAFE_CALL(AMGX_initialize());

  AMGX_SAFE_CALL(AMGX_install_signal_handler());

// setup solver
  AMGX_config_handle cfg;
  AMGX_eigensolver_handle solver;

  AMGX_Mode mode;

  // distributed mode is only implemented on device
  if ((pidx = findParamIndex(argv, argc, "-mode")) != -1)
  {
      if (strncmp(argv[pidx + 1], "hDDI", 100) == 0)
      {
          mode = AMGX_mode_hDDI;
      }
      else if (strncmp(argv[pidx + 1], "hDFI", 100) == 0)
      {
          mode = AMGX_mode_hDFI;
      }
      else if (strncmp(argv[pidx + 1], "hFFI", 100) == 0)
      {
          mode = AMGX_mode_hFFI;
      }
      else if (strncmp(argv[pidx + 1], "dDDI", 100) == 0)
      {
          mode = AMGX_mode_dDDI;
      }
      else if (strncmp(argv[pidx + 1], "dDFI", 100) == 0)
      {
          mode = AMGX_mode_dDFI;
      }
      else if (strncmp(argv[pidx + 1], "dFFI", 100) == 0)
      {
          mode = AMGX_mode_dFFI;
      }
      else if (strncmp(argv[pidx + 1], "hCCI", 100) == 0)
      {
          mode = AMGX_mode_hZZI;
      }
      else if (strncmp(argv[pidx + 1], "hZCI", 100) == 0)
      {
          mode = AMGX_mode_hZCI;
      }
      else if (strncmp(argv[pidx + 1], "hZZI", 100) == 0)
      {
          mode = AMGX_mode_hZZI;
      }
      else if (strncmp(argv[pidx + 1], "dCCI", 100) == 0)
      {
          mode = AMGX_mode_dCCI;
      }
      else if (strncmp(argv[pidx + 1], "dZCI", 100) == 0)
      {
          mode = AMGX_mode_dZCI;
      }
      else if (strncmp(argv[pidx + 1], "dZZI", 100) == 0)
      {
          mode = AMGX_mode_dZZI;
      }
      else
      {
          errAndExit("ERROR: invalid mode");
      }
  }
  else
  {
      printf("Warning: No mode specified, using dDDI by default.\n");
      mode = AMGX_mode_dDDI;
  }

  int pidy = findParamIndex(argv,argc,"-c");
  if (pidy != -1)
  {
    AMGX_SAFE_CALL(AMGX_config_create_from_file(&cfg, argv[pidy+1]));
  }
  else
  {
    // default configuration?
    errAndExit("No solver configuration is specified!");
  }

  // Choosing device 0
  AMGX_resources_handle rsrc = NULL;
  AMGX_SAFE_CALL(AMGX_resources_create(&rsrc, cfg, &amgx_mpi_comm, 1, &local_rank));

  AMGX_SAFE_CALL(AMGX_eigensolver_create(&solver, rsrc, mode, cfg));

// import data

  int *partition_vector = NULL;
  int partition_vector_size = 0;
  if ((pidx = findParamIndex(argv, argc, "-partvec")) != -1)
  {
    FILE* fin_rowpart = fopen(argv[pidx + 1], "rb");
    // Find size of partition vector:
    fseek(fin_rowpart, 0L, SEEK_END);
    int size_of_int = sizeof(int);
    partition_vector_size = ftell(fin_rowpart) / size_of_int;
    partition_vector = (int*) malloc (partition_vector_size*size_of_int);
    //reading the partition vector:
    rewind(fin_rowpart);
    int result = fread((void *)partition_vector, size_of_int, partition_vector_size, fin_rowpart);
    if (result != partition_vector_size)
      errAndExit("Error reading partition vector");
    printf("Read partition vector, consisting of %d rows\n", partition_vector_size);
    fclose(fin_rowpart);
  }

  int n, nnz, block_dimx, block_dimy, num_neighbors;

  int *row_ptrs = NULL, *col_indices = NULL, *neighbors = NULL;
  void *values = NULL, *diag = NULL, *b_data = NULL, *x_data = NULL;
  int **btl_maps = NULL;
  int **lth_maps = NULL;
  int *btl_sizes = NULL;
  int *lth_sizes = NULL;
  if ((pidx = findParamIndex(argv, argc, "-m")) != -1)
  {
      int *partition_sizes = NULL;
      AMGX_SAFE_CALL(AMGX_read_system_maps_one_ring(&n, &nnz, &block_dimx, &block_dimy, &row_ptrs, &col_indices, &values, &diag, &b_data, &x_data,
                                                      &num_neighbors, &neighbors, &btl_sizes, &btl_maps, &lth_sizes, &lth_maps,
                                                      rsrc, mode, argv[pidx + 1], 0, 0, partition_sizes, partition_vector_size, partition_vector));
  }
  else if  ((pidx = findParamIndex(argv, argc, "-p")) != -1) {
    errAndExit("Poission matrix not supported by the c api");
  }
  else
  {
    errAndExit("No problem specified for solver!");
  }

  if (partition_vector != NULL) free(partition_vector);

  int sizeof_m_val = ((AMGX_GET_MODE_VAL(AMGX_MatPrecision, mode) == AMGX_matDouble))? sizeof(double): sizeof(float);
  int sizeof_v_val = ((AMGX_GET_MODE_VAL(AMGX_VecPrecision, mode) == AMGX_vecDouble))? sizeof(double): sizeof(float);

  int block_size = block_dimx*block_dimy;

  // ********** Import data and perform calculation **********

  AMGX_matrix_handle A_part;
  AMGX_vector_handle x_dev, b_dev;

  AMGX_SAFE_CALL(AMGX_matrix_create(&A_part, rsrc, mode));

  AMGX_SAFE_CALL( AMGX_matrix_comm_from_maps_one_ring(A_part, 1, num_neighbors, neighbors, btl_sizes, (const int **)btl_maps, lth_sizes, (const int **)lth_maps));

  AMGX_pin_memory(b_data, n*block_dimx*sizeof_v_val);
  AMGX_pin_memory(x_data, n*block_dimx*sizeof_v_val);
  AMGX_pin_memory(col_indices, nnz*sizeof(int));
  AMGX_pin_memory(values, nnz*block_size*sizeof_m_val);
  if (diag != NULL)
    AMGX_pin_memory(diag, n*block_size*sizeof_m_val);

  AMGX_SAFE_CALL(AMGX_matrix_upload_all( A_part, n, nnz, block_dimx, block_dimy, row_ptrs, col_indices, values, diag));

  AMGX_unpin_memory(col_indices);

  AMGX_SAFE_CALL(AMGX_vector_create(&b_dev, rsrc, mode));
  AMGX_SAFE_CALL(AMGX_vector_create(&x_dev, rsrc, mode));

  AMGX_SAFE_CALL(AMGX_vector_bind(b_dev,A_part));
  AMGX_SAFE_CALL(AMGX_vector_bind(x_dev,A_part));

  // setup
  AMGX_SAFE_CALL(AMGX_eigensolver_setup(solver, A_part));
  AMGX_SAFE_CALL(AMGX_vector_upload(b_dev, n, block_dimx, b_data));
  AMGX_SAFE_CALL(AMGX_vector_upload(x_dev, n, block_dimx, x_data));
  // PageRank setup
  AMGX_SAFE_CALL(AMGX_eigensolver_pagerank_setup(solver, b_dev));


  AMGX_SAFE_CALL(AMGX_vector_set_random(x_dev, n));

  // TESTING REPLACE COEFFICIENTS
  //AMGX_matrix_replace_coefficients(A_part, n, nnz, values, diag);
  // END TESTING REPLACE COEFFICIENTS

  // solve
  AMGX_SAFE_CALL(AMGX_eigensolver_solve(solver, x_dev));

  // get the solution
  void* x_soln = malloc(sizeof_v_val*n*block_dimx);
  // when downloaded, b_dev gets reverted. needs to be uploaded/transformed again for solve step
  AMGX_SAFE_CALL(AMGX_vector_download(b_dev, x_soln));

  free(x_soln);

  AMGX_unpin_memory(b_data);
  AMGX_unpin_memory(x_data);
  AMGX_unpin_memory(values);

  if (diag != NULL) {
    AMGX_unpin_memory(diag);
  }

  AMGX_free_system_maps_one_ring(row_ptrs, col_indices, values, diag, x_data, b_data, num_neighbors, neighbors, btl_sizes, btl_maps, lth_sizes, lth_maps);

  AMGX_SAFE_CALL(AMGX_vector_destroy(b_dev));
  AMGX_SAFE_CALL(AMGX_vector_destroy(x_dev));
  AMGX_SAFE_CALL(AMGX_eigensolver_destroy(solver));
  AMGX_SAFE_CALL(AMGX_matrix_destroy(A_part));
  AMGX_SAFE_CALL(AMGX_config_destroy(cfg));

  AMGX_SAFE_CALL(AMGX_resources_destroy(rsrc));

  AMGX_SAFE_CALL(AMGX_finalize());


  MPI_Finalize();
  return 0;
}

