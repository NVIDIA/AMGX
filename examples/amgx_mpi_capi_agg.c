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
    char msg[MAX_MSG_LEN] = "Usage: mpirun [-n nranks] ./amgx_mpi_capi_agg [-mode [dDDI | dDFI | dFFI]] [-m file] [-c config_file] [-amg \"variable1=value1 ... variable3=value3\"] [-gpu] [-it k]\n";
    strcat(msg, "     -mode:   select the solver mode\n");
    strcat(msg, "     -m file: read matrix stored in the file\n");
    strcat(msg, "     -c:      set the amg solver options from the config file\n");
    strcat(msg, "     -amg:    set the amg solver options from the command line\n");
    strcat(msg, "     -gpu:    load the matrix from the device memory\n");
    strcat(msg, "     -it k:   set the number k of outer (non-linear) iteration\n");
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

void block_element_random_update(int block_dimx, int block_dimy, int i, void *v, AMGX_Mode mode)
{
    int j1, j2, block_size;
    block_size = block_dimy * block_dimx;

    for (j1 = 0; j1 < block_dimy; j1++)
    {
        for (j2 = 0; j2 < block_dimx; j2++)
        {
            if ((AMGX_GET_MODE_VAL(AMGX_MatPrecision, mode) == AMGX_matFloat))
            {
                float *t = (float *)v;
                t[i * block_size + j1 * block_dimx + j2] *= (1.0f + (rand() % 10) * (1e-6f));
            }
            else
            {
                double *t = (double *)v;
                t[i * block_size + j1 * block_dimx + j2] *= (1.0 + (rand() % 10) * (1e-12));
            }
        }
    }
}

int main(int argc, char **argv)
{
    //parameter parsing
    int pidx = 0;
    int pidy = 0;
    //number of outer (non-linear) iterations
    int i = 0;
    int k = 0;
    int max_it = 0;
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
    int n, nnz, block_dimx, block_dimy, block_size, num_neighbors;
    int *row_ptrs = NULL, *col_indices = NULL, *neighbors = NULL;
    void *values = NULL, *diag = NULL, *dh_x = NULL, *dh_b = NULL;
    int *h_row_ptrs = NULL, *h_col_indices = NULL;
    void *h_values = NULL, *h_diag = NULL, *h_x = NULL, *h_b = NULL;
    int *d_row_ptrs = NULL, *d_col_indices = NULL;
    void *d_values = NULL, *d_diag = NULL, *d_x = NULL, *d_b = NULL;
    int sizeof_m_val;
    int sizeof_v_val;
    int *send_sizes = NULL;
    int **send_maps = NULL;
    int *recv_sizes = NULL;
    int **recv_maps = NULL;
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

    sizeof_m_val = ((AMGX_GET_MODE_VAL(AMGX_MatPrecision, mode) == AMGX_matDouble)) ? sizeof(double) : sizeof(float);
    sizeof_v_val = ((AMGX_GET_MODE_VAL(AMGX_VecPrecision, mode) == AMGX_vecDouble)) ? sizeof(double) : sizeof(float);
    /* get max_it number of outer (non-linear) iteration */
    max_it = 1;

    if ((pidx = findParamIndex(argv, argc, "-it")) != -1)
    {
        if ((pidy = findParamIndex(argv, argc, "-gpu")) != -1)
        {
            errAndExit("ERROR: -gpu and -it options are not compatible, you must choose one or the other option");
        }

        max_it = (int)atol(argv[pidx + 1]);
        srand(0);
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
    //WARNING: use 1 ring for aggregation path
    int nrings; //=1;
    AMGX_config_get_default_number_of_rings(cfg, &nrings);
    //printf("nrings=%d\n",nrings);

    if ((pidx = findParamIndex(argv, argc, "-m")) != -1)
    {
        AMGX_read_system_maps_one_ring
        (&n, &nnz, &block_dimx, &block_dimy,
         &h_row_ptrs, &h_col_indices, &h_values, &h_diag, &h_b, &h_x,
         &num_neighbors, &neighbors, &send_sizes, &send_maps, &recv_sizes, &recv_maps,
         rsrc, mode, argv[pidx + 1], nrings, nranks,
         partition_sizes, partition_vector_size, partition_vector);
    }
    else
    {
        errAndExit("ERROR: no linear system was specified");
    }

    //free temporary storage
    if (partition_vector != NULL) { free(partition_vector); }

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
     The output of AMGX_read_system_maps_one_ring for partition 0:
     n = 4; nnz = 21;
     row_ptrs = [0 4 8 13 21]
     col_indices = [0 1 3 6
                    0 1 2 3
                    1 2 3 4 5
                    0 1 2 3 4 5 6 7]
     num_neighbors=2; neighbors = [1 2]
     send_sizes = [0 2 4] send_maps = [2 3| 0 3]
     recv_sizes = [0 2 4] recv_maps = [4 5| 6 7]
     global indices mapping to local indices: 0-0 1-1 2-2 3-3 4-4 5-5 8-6 10-7

     The output of AMGX_read_system_maps_one_ring for partition 1:
     n = 4; nnz = 20
     row_ptrs = [0 4 11 15 20]
     col_indices = [4 0 1 2
                    4 5 0 1 2 3 7
                    0 1 2 3
                    1 2 3 6 7]
     num_neighbors=2; neighbors = [0 2]
     send_sizes = [0 2 4] send_maps = [0 1| 1 3]
     recv_sizes = [0 2 4] recv_maps = [4 5| 6 7]
     global indices mapping to local indices: 4-0 5-1 6-2 7-3 2-4 3-5 9-6 10-7

     The output of AMGX_read_system_maps_one_ring for partition 2:
     n = 4; nnz = 20;
     row_ptrs = [0 5 9 16 20]
     col_indices = [4 5 0 2 3
                    7 1 2 3
                    5 6 7 0 1 2 3
                    0 1 2 3]
     num_neighbors=2; neighbors = [0 1]
     send_sizes = [0 2 4] send_maps = [0 2| 1 2]
     recv_sizes = [0 2 4] recv_maps = [4 5| 6 7]
     global indices mapping to local indices: 8-0 9-1 10-2 11-3 0-4 3-5 5-6 7-7
     */
    block_size = block_dimx * block_dimy;

    if ((pidx = findParamIndex(argv, argc, "-gpu")) != -1)
    {
        /* allocate memory and copy the data to the GPU */
        CUDA_SAFE_CALL(cudaMalloc((void **)&d_x, n * block_dimx * sizeof_v_val));
        CUDA_SAFE_CALL(cudaMalloc((void **)&d_b, n * block_dimy * sizeof_v_val));
        CUDA_SAFE_CALL(cudaMalloc((void **)&d_col_indices, nnz * sizeof(int)));
        CUDA_SAFE_CALL(cudaMalloc((void **)&d_row_ptrs, (n + 1)*sizeof(int)));
        CUDA_SAFE_CALL(cudaMalloc((void **)&d_values, nnz * block_size * sizeof_m_val));
        CUDA_SAFE_CALL(cudaMemcpy(d_x, h_x, n * block_dimx * sizeof_v_val, cudaMemcpyDefault));
        CUDA_SAFE_CALL(cudaMemcpy(d_b, h_b, n * block_dimy * sizeof_v_val, cudaMemcpyDefault));
        CUDA_SAFE_CALL(cudaMemcpy(d_col_indices, h_col_indices, nnz * sizeof(int), cudaMemcpyDefault));
        CUDA_SAFE_CALL(cudaMemcpy(d_row_ptrs, h_row_ptrs, (n + 1)*sizeof(int), cudaMemcpyDefault));
        CUDA_SAFE_CALL(cudaMemcpy(d_values, h_values, nnz * block_size * sizeof_m_val, cudaMemcpyDefault));

        if (h_diag != NULL)
        {
            CUDA_SAFE_CALL(cudaMalloc(&d_diag, n * block_size * sizeof_m_val));
            CUDA_SAFE_CALL(cudaMemcpy(d_diag, h_diag, n * block_size * sizeof_m_val, cudaMemcpyDefault));
        }
        cudaStreamSynchronize(0);

        /* set pointers to point to GPU (device) memory */
        row_ptrs = d_row_ptrs;
        col_indices = d_col_indices;
        values = d_values;
        diag = d_diag;
        dh_x = d_x;
        dh_b = d_b;
    }
    else
    {
        /* pin the memory to improve performance
           WARNING: Even though, internal error handling has been requested,
                    AMGX_SAFE_CALL needs to be used on this system call.
                    It is an exception to the general rule. */
        AMGX_SAFE_CALL(AMGX_pin_memory(h_x, n * block_dimx * sizeof_v_val));
        AMGX_SAFE_CALL(AMGX_pin_memory(h_b, n * block_dimx * sizeof_v_val));
        AMGX_SAFE_CALL(AMGX_pin_memory(h_col_indices, nnz * sizeof(int)));
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
        dh_x = h_x;
        dh_b = h_b;
    }

    /* set the connectivity information (for the matrix) */
    AMGX_matrix_comm_from_maps_one_ring(A, 1, num_neighbors, neighbors, send_sizes, (const int **)send_maps, recv_sizes, (const int **)recv_maps);
    /* set the connectivity information (for the vector) */
    AMGX_vector_bind(x, A);
    AMGX_vector_bind(b, A);
    /* upload the matrix (and the connectivity information) */
    AMGX_matrix_upload_all(A, n, nnz, block_dimx, block_dimy, row_ptrs, col_indices, values, diag);
    /* upload the vector (and the connectivity information) */
    AMGX_vector_upload(x, n, block_dimx, dh_x);
    AMGX_vector_upload(b, n, block_dimx, dh_b);

    /* start outer (non-linear) iterations */
    for (k = 0; k < max_it; k++)
    {
        /* solver setup */
        //MPI barrier for stability (should be removed in practice to maximize performance)
        MPI_Barrier(amgx_mpi_comm);
        AMGX_solver_setup(solver, A);
        /* solver solve */
        //MPI barrier for stability (should be removed in practice to maximize performance)
        MPI_Barrier(amgx_mpi_comm);
        AMGX_solver_solve(solver, b, x);
        /* check the status */
        MPI_Barrier(amgx_mpi_comm);
        AMGX_solver_get_status(solver, &status);

        /* while not the last iteration */
        if (k + 1 < max_it)
        {
            /* example of how to change parameters between non-linear iterations */
            //AMGX_config_add_parameters(&cfg, "config_version=2, default:tolerance=1e-12");
            //AMGX_solver_solve(solver, b, x);

            /* example of how to replace coefficients between non-linear iterations */
            for (i = 0; i < nnz; i++)
            {
                block_element_random_update(block_dimx, block_dimy, i, values, mode);
            }

            if (diag != NULL)
            {
                for (i = 0; i < n; i++)
                {
                    block_element_random_update(block_dimx, block_dimy, i, diag, mode);
                }
            }

            MPI_Barrier(amgx_mpi_comm);
            AMGX_matrix_replace_coefficients(A, n, nnz, values, diag);
            /* upload original vectors (and the connectivity information) */
            AMGX_vector_upload(x, n, block_dimx, dh_x);
            AMGX_vector_upload(b, n, block_dimx, dh_b);
        }
    }

    /* example of how to get (the local part of) the solution */
    //if ((pidx = findParamIndex(argv, argc, "-gpu")) != -1) {
    //    CUDA_SAFE_CALL(cudaMalloc(&d_result, n*block_dimx*sizeof_v_val));
    //    AMGX_vector_download(x, d_result);
    //    CUDA_SAFE_CALL(cudaFree(d_result));
    //}
    //else{
    //    void* h_result = malloc(n*block_dimx*sizeof_v_val);
    //    AMGX_vector_download(x, h_result);
    //    free(h_result);
    //}

    /* example of how to reconstruct the global matrix and write it to a file */
    //AMGX_write_system_distributed(A, b, x, "output_system.mtx", nrings, nranks, partition_sizes, partition_vector_size, partition_vector);

    if ((pidx = findParamIndex(argv, argc, "-gpu")) != -1)
    {
        /* deallocate GPU (device) memory */
        CUDA_SAFE_CALL(cudaFree(d_x));
        CUDA_SAFE_CALL(cudaFree(d_b));
        CUDA_SAFE_CALL(cudaFree(d_row_ptrs));
        CUDA_SAFE_CALL(cudaFree(d_col_indices));
        CUDA_SAFE_CALL(cudaFree(d_values));

        if (d_diag != NULL)
        {
            CUDA_SAFE_CALL(cudaFree(d_diag));
        }
    }
    else
    {
        /* unpin the memory
           WARNING: Even though, internal error handling has been requested,
                    AMGX_SAFE_CALL needs to be used on this system call.
                    It is an exception to the general rule. */
        AMGX_SAFE_CALL(AMGX_unpin_memory(h_x));
        AMGX_SAFE_CALL(AMGX_unpin_memory(h_b));
        AMGX_SAFE_CALL(AMGX_unpin_memory(h_values));
        AMGX_SAFE_CALL(AMGX_unpin_memory(h_row_ptrs));
        AMGX_SAFE_CALL(AMGX_unpin_memory(h_col_indices));

        if (h_diag != NULL)
        {
            AMGX_SAFE_CALL(AMGX_unpin_memory(h_diag));
        }
    }

    /* free buffers allocated during AMGX_read_system_maps_one_ring */
    AMGX_free_system_maps_one_ring(h_row_ptrs, h_col_indices, h_values, h_diag, h_b, h_x, num_neighbors, neighbors, send_sizes, send_maps, recv_sizes, recv_maps);
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
    //return status;
    return 0;
}
