// SPDX-FileCopyrightText: 2011 - 2024 NVIDIA CORPORATION. All Rights Reserved.
//
// SPDX-License-Identifier: BSD-3-Clause

// compilation: mpicxx amgx_spmv_test.c -I/usr/local/cuda/include -I../include -L/usr/local/cuda/lib64 -lcudart -L../build -lamgxsh -Wl,-rpath=/usr/local/cuda/lib64 -Wl,-rpath=../build -o spmv_test

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <mpi.h>
#include "cuda_runtime.h"
#include "amgx_test_common.h"

#include <stdint.h>

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
    char msg[MAX_MSG_LEN] = "Usage: mpirun [-n nranks] ./amgx_spmv_test -m matrix_file.mtx [-mode [dDDI | dDFI | dFFI]] [-partvec partition_file] [-it k] [-ref]\n";
    strcat(msg, "     -m:                   path to matrix file to use, dDDI by default\n");
    strcat(msg, "     -mode:                select the solver mode\n");
    strcat(msg, "     -partvec:             partition file in binary format that specifies which rows belongs to which process. By default, each rank gets 'num_rows/num_processes' rows\n");
    strcat(msg, "     -it                   number of iterations for timing measurement\n");
    strcat(msg, "     -ref                  check result with CPU code\n");
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
    int pidx = 0;
    int pidy = 0;
    //number of outer (non-linear) iterations
    int i = 0;
    int k = 0;
    int max_it = 1;
    //MPI (with CUDA GPUs)
    int rank = 0;
    int lrank = 0;
    int nranks = 0;
    int gpu_count = 0;
    int check_ref = 0;
    MPI_Comm amgx_mpi_comm = MPI_COMM_WORLD;
    //versions
    int major, minor;
    char *ver, *date, *time;
    //input matrix and rhs/solution
    int n, nnz, block_dimx, block_dimy, block_size, num_neighbors;
    int *row_ptrs = NULL, *neighbors = NULL;
    void *col_indices = NULL;
    void *values = NULL, *diag = NULL, *dh_y = NULL, *dh_x = NULL;
    int *h_row_ptrs = NULL;
    void *h_col_indices = NULL;
    void *h_values = NULL, *h_diag = NULL, *h_y = NULL, *h_x = NULL;
    int *partition_sizes = NULL;
    int *partition_vector = NULL;
    int partition_vector_size = 0;
    //library handles
    AMGX_Mode mode;
    AMGX_config_handle cfg;
    AMGX_resources_handle rsrc;
    AMGX_matrix_handle A;
    AMGX_vector_handle y, x;
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

    if ((pidx = findParamIndex(argv, argc, "-it")) != -1)
    {
        max_it = atoi(argv[pidx + 1]);
    }

    if ((pidx = findParamIndex(argv, argc, "-ref")) != -1)
    {
        check_ref = 1;
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

    int sizeof_m_val = ((AMGX_GET_MODE_VAL(AMGX_MatPrecision, mode) == AMGX_matDouble)) ? sizeof(double) : sizeof(float);
    int sizeof_v_val = ((AMGX_GET_MODE_VAL(AMGX_VecPrecision, mode) == AMGX_vecDouble)) ? sizeof(double) : sizeof(float);
    /* example of how to handle errors */
    //char msg[MAX_MSG_LEN];
    //AMGX_RC err_code = AMGX_resources_create(NULL, cfg, &amgx_mpi_comm, 1, &lrank);
    //AMGX_SAFE_CALL(AMGX_get_error_string(err_code, msg, MAX_MSG_LEN));
    //printf("ERROR: %s\n",msg);
    /* switch on internal error handling (no need to use AMGX_SAFE_CALL after this point) */
    AMGX_SAFE_CALL(AMGX_config_create(&cfg, "algorithm=AGGREGATION")); // aggregation for spmv
    /* create resources, matrix, vector and solver */
    AMGX_resources_create(&rsrc, cfg, &amgx_mpi_comm, 1, &lrank);
    AMGX_matrix_create(&A, rsrc, mode);
    AMGX_vector_create(&x, rsrc, mode);
    AMGX_vector_create(&y, rsrc, mode);
    
    //WARNING: use 1 ring for spmv and 2 rings for spmm
    int nrings = 1;

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

        partition_vector_size = ftell(fin_rowpart) / sizeof(int);

        if (partition_vector_size == -1L)
        {
            errAndExit("ERROR: reading partition vector");
        }

        partition_vector = (int *)malloc(partition_vector_size * sizeof(int));
        //reading the partition vector:
        rewind(fin_rowpart);
        int result = fread((void *)partition_vector, sizeof(int), partition_vector_size, fin_rowpart);

        if (result != partition_vector_size)
        {
            errAndExit("ERROR: reading partition vector");
        }

        printf("Read partition vector, consisting of %d rows\n", partition_vector_size);
        fclose(fin_rowpart);
    }

    //read the matrix, [and rhs & solution]
    if ((pidx = findParamIndex(argv, argc, "-m")) != -1)
    {
        AMGX_read_system_global
        (&n, &nnz, &block_dimx, &block_dimy,
         &h_row_ptrs, &h_col_indices, &h_values, &h_diag, &h_x, &h_y,
         rsrc, mode, argv[pidx + 1], nrings, nranks,
         partition_sizes, partition_vector_size, partition_vector);
    }
    else
    {
        errAndExit("ERROR: no matrix was specified");
    }

    /*
    EXAMPLE
    Say, the initial unpartitioned matrix is:
    CSR row_offsets [0 4 8 13 21 25 32 36 41 46 50 57 61]
    CSR col_indices [0 1 3 8
                     0 1 2 3
                     1 2 3 4 5
                     0 1 2 3 4 5 8 10
                     2 4 5 6
                     2 3 4 5 6 7 10
                     4 5 6 7
                     5 6 7 9 10
                     0 3 8 10 11
                     7 9 10 11
                     3 5 7 8 9 10 11
                     8 9 10 11]
     And we are partitioning it into three pieces with the following partition_vector
     [0 0 0 0 1 1 1 1 2 2 2 2]
     The output of AMGX_read_system_global for partition 0:
     n = 4; nnz = 21;
     row_ptrs = [0 4 8 13 21]
     col_indices = [0 1 3 8
                    0 1 2 3
                    1 2 3 4 5
                    0 1 2 3 4 5 8 10]

     The output of AMGX_read_system_global for partition 1:
     n = 4; nnz = 20
     row_ptrs = [0 4 11 15 20]
     col_indices = [2 4 5 6
                    2 3 4 5 6 7 10
                    4 5 6 7
                    5 6 7 9 10]

     The output of AMGX_read_system_global for partition 2:
     n = 4; nnz = 20;
     row_ptrs = [0 5 9 16 20]
     col_indices = [0 3 8 10 11
                    7 9 10 11
                    3 5 7 8 9 10 11
                    8 9 10 11]
     */
    // upload from GPU memory
    block_size = block_dimx * block_dimy;
    /* pin the memory to improve performance
       WARNING: Even though, internal error handling has been requested,
                AMGX_SAFE_CALL needs to be used on this system call.
                It is an exception to the general rule. */
    AMGX_SAFE_CALL(AMGX_pin_memory(h_y, n * block_dimx * sizeof_v_val));
    AMGX_SAFE_CALL(AMGX_pin_memory(h_x, n * block_dimx * sizeof_v_val));
    AMGX_SAFE_CALL(AMGX_pin_memory(h_col_indices, nnz * sizeof(int64_t)));
    AMGX_SAFE_CALL(AMGX_pin_memory(h_row_ptrs, (n + 1)*sizeof(int)));
    AMGX_SAFE_CALL(AMGX_pin_memory(h_values, nnz * block_size * sizeof_m_val));

    if (h_diag != NULL)
    {
        AMGX_SAFE_CALL(AMGX_pin_memory(h_diag, n * block_size * sizeof_m_val));
    }

    /* set pointers to point to CPU (host) memory */
    row_ptrs = h_row_ptrs;
    col_indices = h_col_indices;
    values = h_values;
    diag = h_diag;
    dh_y = h_y;
    dh_x = h_x;
    /* compute global number of rows */
    int nglobal;
    MPI_Allreduce(&n, &nglobal, 1, MPI_INT, MPI_SUM, amgx_mpi_comm);
    /* upload the matrix with global indices and compute necessary connectivity information */
    printf("Uploading data to the library...\n");
    AMGX_SAFE_CALL(AMGX_matrix_upload_all_global(A, nglobal, n, nnz, block_dimx, block_dimy, row_ptrs, col_indices, values, diag, nrings, nrings, partition_vector));

    /* free temporary storage */
    if (partition_vector != NULL) { free(partition_vector); }

    // initalize x and y with something simple
    for (int i = 0; i < n; i++)
    {
        if ((AMGX_GET_MODE_VAL(AMGX_VecPrecision, mode) == AMGX_vecFloat))
        {
            ((float *)dh_x)[i] = 1.0f * i;
            ((float *)dh_y)[i] = 0.0f;
        }
        else
        {
            ((double *)dh_x)[i] = 1.0 * i;
            ((double *)dh_y)[i] = 0.0;
        }
    }

    /* set the connectivity information (for the vector) so it will be aware of ranks rows mappings*/
    AMGX_SAFE_CALL(AMGX_vector_bind(x, A));
    AMGX_SAFE_CALL(AMGX_vector_bind(y, A));
    /* upload the vector (and the connectivity information) */
    AMGX_SAFE_CALL(AMGX_vector_upload(x, n, 1, dh_x));
    AMGX_SAFE_CALL(AMGX_vector_upload(y, n, 1, dh_y));
    /* solver setup */
    //MPI barrier for stability (should be removed in practice to maximize performance)
    printf("Launch SpMV for %d times...\n", max_it);
    MPI_Barrier(amgx_mpi_comm);
    double cur, total = 0;

    for (int it = 0; it < max_it; it++)
    {
        cur = second();
        AMGX_SAFE_CALL(AMGX_matrix_vector_multiply(A, x, y)); // y = A*x
        total += second() - cur;
    }

    MPI_Barrier(amgx_mpi_comm);
    printf("Average time of SpMV launch for rank# %d is: %10.8gs\n", rank, total / max_it);

    if (check_ref)
    {
        /* example of how to get (the local part of) the result */
        void *result_host = malloc(n * block_dimx * sizeof_v_val);
        void *global_x = malloc(nglobal * block_dimx * sizeof_v_val);
        memset(result_host, 0, n * block_dimy * sizeof_v_val);
        AMGX_SAFE_CALL(AMGX_vector_download(y, result_host));

        if ((AMGX_GET_MODE_VAL(AMGX_VecPrecision, mode) == AMGX_vecFloat))
        {
            for (int row = 0; row < n; row++)
            {
                float sum = 0.f;

                for (int col = row_ptrs[row]; col < row_ptrs[row + 1]; col++)
                {
                    sum += ( (AMGX_GET_MODE_VAL(AMGX_MatPrecision, mode) == AMGX_matFloat) ? ((float *)values)[col] : ((double *)values)[col] ) * ((float*)dh_x)[((int64_t*)col_indices)[col]];
                }

                if (fabs(sum - ((float *)result_host)[row]) > 1e-4)
                {
                    printf ("ERROR in result checking float precision for rank #%d: Difference in row #%d: %10.8g[CPU] vs %10.8g[AMGX]\n", rank, row, sum, ((float *)result_host)[i]);
                    MPI_Finalize();
                    exit(1);
                }
            }
        }
        else
        {
            for (int row = 0; row < n; row++)
            {
                double sum = 0.;

                for (int col = row_ptrs[row]; col < row_ptrs[row + 1]; col++)
                {
                    sum += ((AMGX_GET_MODE_VAL(AMGX_MatPrecision, mode) == AMGX_matFloat) ? ((float *)values)[col] : ((double *)values)[col] ) * ((double*)dh_x)[((int64_t*)col_indices)[col]];
                }

                if (abs(sum - ((double *)result_host)[row]) > 1e-8)
                {
                    printf ("ERROR in result checking double precision for rank #%d: Difference in row #%d: %10.8g[CPU] vs %10.8g[AMGX]\n", rank, row, sum, ((double *)result_host)[i]);
                    MPI_Finalize();
                    exit(1);
                }
            }
        }

        printf("Validation on CPU passed!\n");

        free(result_host);
    }

    /* destroy resources, matrix, vector and solver */
    AMGX_SAFE_CALL(AMGX_vector_destroy(x));
    AMGX_SAFE_CALL(AMGX_vector_destroy(y));
    AMGX_SAFE_CALL(AMGX_matrix_destroy(A));
    AMGX_SAFE_CALL(AMGX_resources_destroy(rsrc));
    /* destroy config (need to use AMGX_SAFE_CALL after this point) */
    AMGX_SAFE_CALL(AMGX_config_destroy(cfg));
    /* shutdown and exit */
    AMGX_SAFE_CALL(AMGX_finalize());
    /* close the library (if it was dynamically loaded) */
#ifdef AMGX_DYNAMIC_LOADING
    amgx_libclose(lib_handle);
#endif
    MPI_Finalize();
    return status;
}


