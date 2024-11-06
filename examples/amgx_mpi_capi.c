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
    char msg[MAX_MSG_LEN] = "Usage: mpirun [-n nranks] ./amgx_mpi_capi [-mode [dDDI | dDFI | dFFI]] [-m file] [-c config_file] [-amg \"variable1=value1 ... variable3=value3\"]\n";
    strcat(msg, "     -mode:   select the solver mode\n");
    strcat(msg, "     -m file: read matrix stored in the file\n");
    strcat(msg, "     -c:      set the amg solver options from the config file\n");
    strcat(msg, "     -amg:    set the amg solver options from the command line\n");
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
    int gpu_count = 0;
    MPI_Comm amgx_mpi_comm = MPI_COMM_WORLD;
    //versions
    int major, minor;
    char *ver, *date, *time;
    //input matrix and rhs/solution
    int *partition_sizes = NULL;
    int *partition_vector = NULL;
    int partition_vector_size = 0;
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

    /* read and partition the input system: matrix [and rhs & solution]
       Please refer to AMGX_read_system description in the AMGX_Reference.pdf
       manual for details on how to specify the rhs and the solution inside
       the input file. If these are not specified than rhs=[1,...,1]^T and
       (initial guess) sol=[0,...,0]^T. */

    //read partitioning vector
    if ((pidx = findParamIndex(argv, argc, "-partvec")) != -1)
    {
        //open the file
        FILE *fin_rowpart = fopen(argv[pidx + 1], "rb");

        if (fin_rowpart == NULL)
        {
            errAndExit("ERROR: opening the file for the partition vector");
        }

        //find the size of the partition vector
        if (fseek(fin_rowpart, 0L, SEEK_END) != 0)
        {
            errAndExit("ERROR: reading partition vector");
        }

        int size_of_int = sizeof(int);
        partition_vector_size = ftell(fin_rowpart) / size_of_int;

        if (partition_vector_size == -1L)
        {
            errAndExit("ERROR: reading partition vector");
        }

        partition_vector = (int *) malloc (partition_vector_size * size_of_int);
        //reading the partition vector:
        rewind(fin_rowpart);
        int result = fread((void *)partition_vector, size_of_int, partition_vector_size, fin_rowpart);

        if (result != partition_vector_size)
        {
            errAndExit("ERROR: reading partition vector");
        }

        printf("Read partition vector, consisting of %d rows\n", partition_vector_size);
        fclose(fin_rowpart);
    }

    //read the matrix, [and rhs & solution]
    //WARNING: use 1 ring for aggregation and 2 rings for classical path
    int nrings; //=1; //=2;
    AMGX_config_get_default_number_of_rings(cfg, &nrings);
    //printf("nrings=%d\n",nrings);

    if ((pidx = findParamIndex(argv, argc, "-m")) != -1)
    {
        AMGX_read_system_distributed
        (A, b, x,
         argv[pidx + 1], nrings, nranks,
         partition_sizes, partition_vector_size, partition_vector);
    }
    else
    {
        errAndExit("ERROR: no linear system was specified");
    }

    pidx = findParamIndex(argv, argc, "-cd");
    if(pidx != -1) 
    {
        int diag_dominant = 0;

        printf("Checking matrix is diag dominant\n");
        AMGX_matrix_check_diag_dominant(A, &diag_dominant);
    }

    pidx = findParamIndex(argv, argc, "-cs");
    if(pidx != -1) 
    {
        int structurally_symmetric = 0;
        int symmetric = 0;

        printf("Checking matrix symmetry\n");
        AMGX_matrix_check_symmetry(A, &structurally_symmetric, &symmetric);
        if(symmetric) {
            printf("A is symmetric\n");
        }
        else if(structurally_symmetric) {
            printf("A is structurally_symmetric\n");
        }
        else {
            printf("A is assymetric\n");
        }
    }

    pidx = findParamIndex(argv, argc, "-om");
    if(pidx != -1)
    {
        AMGX_write_system
        (A, b, x,
         argv[pidx + 1]);
        return 0;
    }

    // Print out details of matrix as initialised by file and seen by the C API
    int n;
    int block_dimx;
    int block_dimy;
    AMGX_matrix_get_size(A, &n, &block_dimx, &block_dimy);

    if(block_dimx > 1 || block_dimy > 1) {
        if(rank == 0) { printf("Matrix A has %d rows with %d x %d blocks\n", n, block_dimx, block_dimy); };
    }
    else {
        if(rank == 0) { printf("Matrix A is scalar and has %d rows\n", n); };
    }

    size_t sizeof_v_val = ((AMGX_GET_MODE_VAL(AMGX_VecPrecision, mode) == AMGX_vecDouble))? sizeof(double) : sizeof(float);

    // Potential setup dumb repeats for perf analysis
    // Repeats just reset the solution to original state and setup/solve again
    // Typically useful to analyse performance without overheads of a cold start
    void *x_h;
    int nrepeats = 1;
    pidx = findParamIndex(argv, argc, "-r");
    if(pidx != -1)
    {
      nrepeats = atoi(argv[pidx+1]);
      x_h = malloc(n * block_dimx * sizeof_v_val);
      AMGX_vector_download(x, x_h);

      if (rank == 0) { printf("Running for %d repeats\n", nrepeats); }
    }

    //free temporary storage
    if (partition_vector != NULL) { free(partition_vector); }

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
