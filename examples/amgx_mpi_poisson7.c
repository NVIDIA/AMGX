// SPDX-FileCopyrightText: 2011 - 2024 NVIDIA CORPORATION. All Rights Reserved.
//
// SPDX-License-Identifier: BSD-3-Clause

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <mpi.h>
#include "cuda_runtime.h"

/* CUDA error macro */
#define CUDA_SAFE_CALL(call) do {                                 \
  cudaError_t err = call;                                         \
  if(cudaSuccess != err) {                                        \
    fprintf(stderr, "Cuda error in file '%s' in line %i : %s.\n", \
            __FILE__, __LINE__, cudaGetErrorString( err) );       \
    exit(EXIT_FAILURE);                                           \
  } } while (0)

//#define AMGX_DYNAMIC_LOADING
//#undef AMGX_DYNAMIC_LOADING
#define MAX_MSG_LEN 4096

/* standard or dynamically load library */
#ifdef AMGX_DYNAMIC_LOADING
#include "amgx_capi.h"
#else
#include "amgx_c.h"
#endif

/* print error message and exit */
void errAndExit(const char *err)
{
    printf("%s\n", err);
    fflush(stdout);
    MPI_Abort(MPI_COMM_WORLD, 1);
    exit(1);
}

/* print callback (could be customized) */
void print_callback(const char *msg, int length)
{
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    if (rank == 0) { printf("%s", msg); }
}

/* print usage and exit */
void printUsageAndExit()
{
    char msg[MAX_MSG_LEN] = "Usage: mpirun [-n nranks] ./amgx_mpi_capi_agg [-mode [dDDI | dDFI | dFFI]] [-p nx ny nz] [-c config_file] [-amg \"variable1=value1 ... variable3=value3\"] [-gpu] [-it k]\n";
    strcat(msg, "     -mode:                select the solver mode\n");
    strcat(msg, "     -p nx ny nz px py pz: select x-, y- and z-dimensions of the 3D (7-points) local discretization of the Poisson operator as well as the x-, y- and z-dimensions splitting of parallel ranks/processes. (the global problem size will be nranks*nx*ny*nz, where nranks=px*py*pz)\n");
    strcat(msg, "     -c:                   set the amg solver options from the config file\n");
    strcat(msg, "     -amg:                 set the amg solver options from the command line\n");
    print_callback(msg, MAX_MSG_LEN);
    MPI_Finalize();
    exit(0);
}

/* parse parameters */
int findParamIndex(char **argv, int argc, const char *parm)
{
    int count = 0;
    int index = -1;

    for (int i = 0; i < argc; i++)
    {
        if (strncmp(argv[i], parm, 100) == 0)
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
        char msg[MAX_MSG_LEN];
        sprintf(msg, "ERROR: parameter %s has been specified more than once, exiting\n", parm);
        print_callback(msg, MAX_MSG_LEN);
        exit(1);
    }

    return -1;
}

int main(int argc, char **argv)
{
    //parameter parsing
    int pidx = 0;
    int pidy = 0;
    //MPI (with CUDA GPUs)
    int rank = 0;
    int lrank = 0;
    int nranks = 0;
    int n;
    int nx, ny, nz;
    int px, py, pz;
    int gpu_count = 0;
    MPI_Comm amgx_mpi_comm = MPI_COMM_WORLD;
    //versions
    int major, minor;
    char *ver, *date, *time;
    //library handles
    AMGX_Mode mode;
    AMGX_config_handle cfg;
    AMGX_resources_handle rsrc;
    AMGX_matrix_handle A;
    AMGX_vector_handle b, x;
    AMGX_solver_handle solver;
    //status handling
    AMGX_SOLVE_STATUS status;
    /* MPI init (with CUDA GPUs) */
    //MPI
    MPI_Init(&argc, &argv);
    MPI_Comm_size(amgx_mpi_comm, &nranks);
    MPI_Comm_rank(amgx_mpi_comm, &rank);
    //CUDA GPUs
    CUDA_SAFE_CALL(cudaGetDeviceCount(&gpu_count));
    lrank = rank % gpu_count;
    CUDA_SAFE_CALL(cudaSetDevice(lrank));
    printf("Process %d selecting device %d\n", rank, lrank);

    /* check arguments */
    if (argc == 1)
    {
        printUsageAndExit();
    }

    /* load the library (if it was dynamically loaded) */
#ifdef AMGX_DYNAMIC_LOADING
    void *lib_handle = NULL;
#ifdef _WIN32
    lib_handle = amgx_libopen("amgxsh.dll");
#else
    lib_handle = amgx_libopen("libamgxsh.so");
#endif

    if (lib_handle == NULL)
    {
        errAndExit("ERROR: can not load the library");
    }

    //load all the routines
    if (amgx_liblink_all(lib_handle) == 0)
    {
        amgx_libclose(lib_handle);
        errAndExit("ERROR: corrupted library loaded\n");
    }

#endif
    /* init */
    AMGX_SAFE_CALL(AMGX_initialize());
    /* system */
    AMGX_SAFE_CALL(AMGX_register_print_callback(&print_callback));
    AMGX_SAFE_CALL(AMGX_install_signal_handler());

    /* get api and build info */
    if ((pidx = findParamIndex(argv, argc, "--version")) != -1)
    {
        AMGX_get_api_version(&major, &minor);
        printf("amgx api version: %d.%d\n", major, minor);
        AMGX_get_build_info_strings(&ver, &date, &time);
        printf("amgx build version: %s\nBuild date and time: %s %s\n", ver, date, time);
        AMGX_SAFE_CALL(AMGX_finalize());
        /* close the library (if it was dynamically loaded) */
#ifdef AMGX_DYNAMIC_LOADING
        amgx_libclose(lib_handle);
#endif
        MPI_Finalize();
        exit(0);
    }

    /* get mode */
    if ((pidx = findParamIndex(argv, argc, "-mode")) != -1)
    {
        if (strncmp(argv[pidx + 1], "dDDI", 100) == 0)
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

    //int sizeof_m_val = ((AMGX_GET_MODE_VAL(AMGX_MatPrecision, mode) == AMGX_matDouble))? sizeof(double): sizeof(float);
    size_t sizeof_v_val = ((AMGX_GET_MODE_VAL(AMGX_VecPrecision, mode) == AMGX_vecDouble)) ? sizeof(double) : sizeof(float);
    /* create config */
    pidx = findParamIndex(argv, argc, "-amg");
    pidy = findParamIndex(argv, argc, "-c");

    if ((pidx != -1) && (pidy != -1))
    {
        printf("%s\n", argv[pidx + 1]);
        AMGX_SAFE_CALL(AMGX_config_create_from_file_and_string(&cfg, argv[pidy + 1], argv[pidx + 1]));
    }
    else if (pidy != -1)
    {
        AMGX_SAFE_CALL(AMGX_config_create_from_file(&cfg, argv[pidy + 1]));
    }
    else if (pidx != -1)
    {
        printf("%s\n", argv[pidx + 1]);
        AMGX_SAFE_CALL(AMGX_config_create(&cfg, argv[pidx + 1]));
    }
    else
    {
        errAndExit("ERROR: no config was specified");
    }

    /* example of how to handle errors */
    //char msg[MAX_MSG_LEN];
    //AMGX_RC err_code = AMGX_resources_create(NULL, cfg, &amgx_mpi_comm, 1, &lrank);
    //AMGX_SAFE_CALL(AMGX_get_error_string(err_code, msg, MAX_MSG_LEN));
    //printf("ERROR: %s\n",msg);
    /* switch on internal error handling (no need to use AMGX_SAFE_CALL after this point) */
    AMGX_SAFE_CALL(AMGX_config_add_parameters(&cfg, "exception_handling=1"));
    /* create resources, matrix, vector and solver */
    AMGX_resources_create(&rsrc, cfg, &amgx_mpi_comm, 1, &lrank);
    AMGX_matrix_create(&A, rsrc, mode);
    AMGX_vector_create(&x, rsrc, mode);
    AMGX_vector_create(&b, rsrc, mode);
    AMGX_solver_create(&solver, rsrc, mode, cfg);
    //generate 3D Poisson matrix, [and rhs & solution]
    //WARNING: use 1 ring for aggregation and 2 rings for classical path
    int nrings; //=1; //=2;
    AMGX_config_get_default_number_of_rings(cfg, &nrings);
    //printf("nrings=%d\n",nrings);

    if  ((pidx = findParamIndex(argv, argc, "-p")) != -1)
    {
        nx = atoi(argv[++pidx]);
        ny = atoi(argv[++pidx]);
        nz = atoi(argv[++pidx]);
        n  = nx * ny * nz;
        px = atoi(argv[++pidx]);
        py = atoi(argv[++pidx]);
        pz = atoi(argv[++pidx]);

        if (nranks != px * py * pz)
        {
            printf("ERROR: nranks [%d] must equal [%d] = px[%d]*py[%d]*pz[%d])\n", nranks, px * py * pz, px, py, pz);
            errAndExit("ERROR: the splitting of ranks in paralllel is incorrect");
        }
    }

    /* generate the matrix (also, set up the connectivity information and upload it).
       In more detail, this routine will create 3D (7 point) discretization of the
       Poisson operator. The discretization is performed on a the 3D domain consisting
       of nx, ny and nz points in x-, y- and z-dimension, respectively. This 3D domain
       will be replicated in px, py and pz times in x-, y- and z-dimension. Therefore,
       creating a large "cube", composed of smaller "sub-cubes" each of which is going
       to be handled on a separate ranks/processor. Finally, the rhs and solution will
       be set to a vector of ones and zeros, respectively. */
    AMGX_generate_distributed_poisson_7pt(A, b, x, nrings, nrings, nx, ny, nz, px, py, pz);

    /* generate the rhs and solution */
    int block_dimx = 1;
    int block_dimy = 1;
    void *x_h = malloc(n * block_dimx * sizeof_v_val);
    void *b_h = malloc(n * block_dimy * sizeof_v_val);
    memset(x_h, 0, n * block_dimx * sizeof_v_val);

    for (int i = 0; i < n; i++)
    {
        if ((AMGX_GET_MODE_VAL(AMGX_VecPrecision, mode) == AMGX_vecFloat))
        {
            ((float *)b_h)[i] = 1.0f;
        }
        else
        {
            ((double *)b_h)[i] = 1.0;
        }
    }

    int nrepeats = 1;
    int tidx = findParamIndex(argv, argc, "-r");
    if(tidx != -1)
    {
      nrepeats = atoi(argv[tidx+1]);
      if (rank == 0) { printf("Running for %d repeats\n", nrepeats); }
    }

    /* set the connectivity information (for the vector) */
    AMGX_vector_bind(x, A);
    AMGX_vector_bind(b, A);
    /* upload the vector (and the connectivity information) */
    AMGX_vector_upload(x, n, 1, x_h);
    AMGX_vector_upload(b, n, 1, b_h);
    for(int r = 0; r < nrepeats; ++r)
    {
      if(r > 0) {
        // Reset the solution for each repeat
        AMGX_vector_upload(x, n, block_dimx, x_h);
      }

      /* solver setup */
      //MPI barrier for stability (should be removed in practice to maximize performance)
      MPI_Barrier(amgx_mpi_comm);
      AMGX_solver_setup(solver, A);
      /* solver solve */
      //MPI barrier for stability (should be removed in practice to maximize performance)
      MPI_Barrier(amgx_mpi_comm);
      AMGX_solver_solve(solver, b, x);
      /* example of how to change parameters between non-linear iterations */
      //AMGX_config_add_parameters(&cfg, "config_version=2, default:tolerance=1e-12");
      //AMGX_solver_solve(solver, b, x);
      /* example of how to replace coefficients between non-linear iterations */
      //AMGX_matrix_replace_coefficients(A, n, nnz, values, diag);
      //AMGX_solver_setup(solver, A);
      //AMGX_solver_solve(solver, b, x);
      MPI_Barrier(MPI_COMM_WORLD);
      AMGX_solver_get_status(solver, &status);
      if(status == AMGX_SOLVE_DIVERGED) {
          print_callback("***Solver Diverged\n", 0);
      }
      else if(status == AMGX_SOLVE_NOT_CONVERGED) {
          print_callback("***Solver Did Not Converge\n", 0);
      }
      else if(status == AMGX_SOLVE_FAILED) {
          print_callback("***Solver Failed\n", 0);
      }
      else if(status == AMGX_SOLVE_SUCCESS) {
          print_callback("***Solver Converged\n", 0);
      }
    }

    /* example of how to get (the local part of) the solution */
    //int sizeof_v_val;
    //sizeof_v_val = ((NVAMG_GET_MODE_VAL(NVAMG_VecPrecision, mode) == NVAMG_vecDouble))? sizeof(double): sizeof(float);
    //void* result_host = malloc(n*block_dimx*sizeof_v_val);
    //AMGX_vector_download(x, result_host);
    //free(result_host);
    /* destroy resources, matrix, vector and solver */
    AMGX_solver_destroy(solver);
    AMGX_vector_destroy(x);
    AMGX_vector_destroy(b);
    AMGX_matrix_destroy(A);
    AMGX_resources_destroy(rsrc);
    /* destroy config (need to use AMGX_SAFE_CALL after this point) */
    AMGX_SAFE_CALL(AMGX_config_destroy(cfg))
      /* shutdown and exit */
      AMGX_SAFE_CALL(AMGX_finalize())
      /* close the library (if it was dynamically loaded) */
#ifdef AMGX_DYNAMIC_LOADING
      amgx_libclose(lib_handle);
#endif
    MPI_Finalize();
    return 0;
}


