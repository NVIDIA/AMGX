// SPDX-FileCopyrightText: 2024 NVIDIA CORPORATION. All Rights Reserved.
//
// SPDX-License-Identifier: BSD-3-Clause

#include "matrix.h"
#include "cuda_runtime.h"
#include "stdio.h"
#include "multiply.h"
#include "error.h"
#include "thrust/fill.h"
#include "mpi.h"

// example of using AMGX code to get matrix vector product
// to compile
// 1) either link against AMGX static library (make sure it's built with MPI support):
//    nvcc -std=c++11 --gpu-architecture=compute_35 --gpu-code=compute_35,sm_35 -Xcompiler ,\"-DAMGX_WITH_MPI\" -I../../include -I../../../../thrust amgx_spmv_distributed_internal.cu -lamgx -L../../build -lmpi -L/opt/openmpi-1.10.2/lib -lcublas -lcusparse -lcusolver -o amgx_spmv_distributed -Xlinker=-rpath=/usr/local/cuda/lib64
// 2) or use Makefile which will compile and link against selected AMGX files required for SpMV:
//    make example_distributed
// Run: 
// mpirun -n 3 ./amgx_spmv_distributed

// global communicator
MPI_Comm g_comm = MPI_COMM_WORLD;

void registerParameters();

void spmv_example(amgx::Resources& res)
{
    // TemplateConfig parameters:
    // calculate on device
    // double storage for matrix values
    // double storage for vector values
    // integer for indices values
    //
    // see include/basic_types.h for details
    typedef amgx::TemplateConfig<AMGX_device, AMGX_vecDouble, AMGX_matDouble, AMGX_indInt> TConfig; // Type for spmv calculation
    typedef amgx::Vector<amgx::TemplateConfig<AMGX_host, AMGX_vecDouble, AMGX_matDouble, AMGX_indInt>> VVector_h; // vector type to retrieve result

    int nranks, rank;
    MPI_Comm_size(g_comm, &nranks);
    MPI_Comm_rank(g_comm, &rank);

    amgx::DistributedManager<TConfig> d_mgr;// = new amgx::DistributedManager<TConfig>;
    // initalize comms using resources
    d_mgr.initComms(&res);

    // set up distributed matrix
    amgx::Matrix<TConfig> A;// = new amgx::Matrix<TConfig>;
    A.setResources(&res);
    // bind matrix and manager together.
    A.setManager(d_mgr);
    d_mgr.setMatrix(A);

    // set up vectors
    amgx::Vector<TConfig> x;
    x.setResources(&res);
    // bind vector to the same manager as matrix forces same vector row mapping as for distributed matrix
    x.setManager(d_mgr);

    amgx::Vector<TConfig> y;
    y.setResources(&res);
    y.setManager(d_mgr);

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
    where partition_vector[r] = k means that row 'r' belongs to rank 'k'
    */

    int n_global = 12;
    std::vector<int> partition_vector = {0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2};
    std::vector<int> row_offsets_h;
    std::vector<int> column_indices_h;
    std::vector<double> matrix_values_h;
    std::vector<double> x_h;
    int nrows, nnz;

    // Let's initialize each part on respective process
    if (rank == 0)
    {
        // We provide corresponding part data for rank 0
        nrows = 4;
        nnz = 21;
        row_offsets_h = {0, 4, 8, 13, 21};
        column_indices_h = {0, 1, 3, 8,
                            0, 1, 2, 3,
                            1, 2, 3, 4, 5,
                            0, 1, 2, 3, 4, 5, 8, 10};
    }
    else if (rank == 1)
    {
        // Corresponding part data for rank 1
        nrows = 4;
        nnz = 20;
        row_offsets_h = {0, 4, 11, 15, 20};
        column_indices_h = {2, 4, 5, 6,
                            2, 3, 4, 5, 6, 7, 10,
                            4, 5, 6, 7,
                            5, 6, 7, 9, 10};
    }
    else
    {
        // Corresponding part data for rank 2
        nrows = 4;
        nnz = 20;
        row_offsets_h = {0, 5, 9, 16, 20};
        column_indices_h = {0, 3, 8, 10, 11,
                            7, 9, 10, 11,
                            3, 5, 7, 8, 9, 10, 11,
                            8, 9, 10, 11};
    }

    // each partition has it's own corresponding values
    srand(rank);
    matrix_values_h.resize(nnz);
    for (auto &v : matrix_values_h) v = ((double)random())/RAND_MAX;
    // Let distributed manager load data into bound matrix structure
    MatrixDistribution mdist;
    mdist.setPartitionVec(&partition_vector[0]);
    d_mgr.loadDistributedMatrix(nrows, nnz, 1, 1, &row_offsets_h[0], &column_indices_h[0], &matrix_values_h[0], nranks, n_global, NULL, mdist);
    // Matrix should be reordered with 1 halo ring for SpMV 
    d_mgr.renumberMatrixOneRing();
    // Initializing communicator parameters
    d_mgr.getComms()->set_neighbors(A.manager->num_neighbors());
    // Matrix is ready for SpMV
    A.set_initialized(1);

    x.resize(nrows);
    y.resize(nrows);
    amgx::thrust::fill(y.begin(), y.end(), 0.);

    // vector values
    x_h.resize(nrows);
    for (auto &v : x_h) v = ((double)random())/RAND_MAX;

    // mark AMGX vector that halo rows corresponding to neighbour partition rows are not here yet
    x.dirtybit = 1;
    // upload those values to vector structure
    // distributed manager will perform same reordering that is done for matrix data
    d_mgr.transformAndUploadVector(x, &x_h[0], nrows, 1);
    
    // AMGX multiply 
    amgx::multiply(A, x, y);

    // get the result to host
    // distributed manager need to unreorder the vector
    // downloaded only corresponding local part of the result vector
    std::vector<double> y_res_h(nrows);
    d_mgr.revertAndDownloadVector(y, &y_res_h[0], nrows, 1);

    // reference check
    // gather global x vector
    std::vector<double> x_global(n_global), x_global_unmapped(n_global);
    {
        int count = 0;
        std::vector<int> local_sizes(nranks, 0);
        for (auto &v: partition_vector) local_sizes[v]++;
        std::vector<int> local_offsets(nranks, 0);
        for (int i = 0; i < nranks; i++)
        {
            local_offsets[i] = count;
            count += local_sizes[i];
        }

        MPI_Allgather((void*)&x_h[0], nrows, MPI_DOUBLE, (void*)&x_global_unmapped[0], nrows, MPI_DOUBLE, g_comm);

        for (int i = 0; i < n_global; i++)
        {
            x_global[local_offsets[partition_vector[i]]] = x_global_unmapped[i];
            local_offsets[partition_vector[i]]++;
        }
    }

    // reference check for each local partition
    std::vector<double> y_res_ref(nrows, 0.);
    bool err_found = false;
    for (int r = 0; r < nrows; r++)
    {
        double y_res_ref = 0.;
        for (int c = row_offsets_h[r]; c < row_offsets_h[r + 1]; c++)
        {
            y_res_ref += matrix_values_h[c]*x_global[column_indices_h[c]];
        }
        if (std::abs(y_res_ref - y_res_h[r]) > 1e-8)
        {
            printf("Rank: %d: difference in row %d: reference: %f, AMGX: %f\n", rank, r, y_res_ref, y_res_h[r]);
            err_found = true;
        }
    }

    if (!err_found)
        printf("Rank: %d: OK!\n", rank);
}

int main(int argc, char* argv[])
{
    // Initialization
    int nranks, rank, my_gpu, gpu_count;
    MPI_Init(&argc, &argv);
    MPI_Comm_size(g_comm, &nranks);
    MPI_Comm_rank(g_comm, &rank);
    if (nranks != 3) 
    {
        printf("Please, run this example with 3 MPI processes\n");
        MPI_Abort(g_comm, 1);
    }

    // cycling processes through GPUs
    cudaGetDeviceCount(&gpu_count);
    my_gpu = rank % gpu_count;
    cudaSetDevice(my_gpu);
    printf("Process %d selecting device %d\n", rank, my_gpu);

    // register AMGX MPI error handler
    amgx::registerDefaultMPIErrHandler();
    // register required AMGX parameters 
    registerParameters();

    // 1 GPU per process with number = my_gpu
    amgx::AMG_Configuration cfg;
    amgx::Resources* res = new amgx::Resources(&cfg, (void*)&g_comm, 1, &my_gpu);

    // make sure we perform any AMGX functionality within Resources lifetime
    try
    {
        spmv_example(*res);
    }
    catch (amgx::amgx_exception e) 
    {
        std::string err = "Caught amgx exception: " + std::string(e.what()) + " at: "
            + std::string(e.where()) + "\nStack trace:\n" + std::string(e.trace()) + "\n";
        amgx::error_output(err.c_str(), static_cast<int>(err.length()));
    }

    // free resources before MPI_Finalize()
    delete res;
    MPI_Finalize();
    return 0;
}


// Routine to register some of the AMGX parameters manually
// Typically if you want to use AMGX solver you should call core::initialize() which will cover this initialization, 
// however for spmv alone there are only few parameters needed and no need to initialize AMGX core solvers.
void registerParameters()
{
    using namespace amgx;
    std::vector<int> bool_flag_values;
    bool_flag_values.push_back(0);
    bool_flag_values.push_back(1);
    //Register Exception Handling Parameter
    AMG_Config::registerParameter<int>("exception_handling", "a flag that forces internal exception processing instead of returning error codes(1:internal, 0:external)", 0, bool_flag_values);
    //Register System Parameters (memory pools)
    AMG_Config::registerParameter<size_t>("device_mem_pool_size", "size of the device memory pool in bytes", 256 * 1024 * 1024);
    AMG_Config::registerParameter<size_t>("device_consolidation_pool_size", "size of the device memory pool for root partition in bytes", 256 * 1024 * 1024);
    AMG_Config::registerParameter<size_t>("device_mem_pool_max_alloc_size", "maximum size of a single allocation in the device memory pool in bytes", 20 * 1024 * 1024);
    AMG_Config::registerParameter<size_t>("device_alloc_scaling_factor", "over allocation for large buffers (in %% -- a value of X will lead to 100+X%% allocations)", 10);
    AMG_Config::registerParameter<size_t>("device_alloc_scaling_threshold", "buffers smaller than that threshold will NOT be scaled", 16 * 1024);
    AMG_Config::registerParameter<size_t>("device_mem_pool_size_limit", "size of the device memory pool in bytes. 0 - no limit", 0);
    //Register System Parameters (asynchronous framework)
    AMG_Config::registerParameter<int>("num_streams", "number of additional CUDA streams / threads used for async execution", 0);
    AMG_Config::registerParameter<int>("serialize_threads", "flag that enables thread serialization for debugging <0|1>", 0, bool_flag_values);
    AMG_Config::registerParameter<int>("high_priority_stream", "flag that enables high priority CUDA stream <0|1>", 0, bool_flag_values);
    //Register System Parameters (in distributed setting)
    std::vector<std::string> communicator_values;
    communicator_values.push_back("MPI");
    communicator_values.push_back("MPI_DIRECT");
    AMG_Config::registerParameter<std::string>("communicator", "type of communicator <MPI|MPI_DIRECT>", "MPI");
    std::vector<ViewType> viewtype_values;
    viewtype_values.push_back(INTERIOR);
    viewtype_values.push_back(OWNED);
    viewtype_values.push_back(FULL);
    viewtype_values.push_back(ALL);
    AMG_Config::registerParameter<ViewType>("separation_interior", "separation for latency hiding and coloring/smoothing <ViewType>", INTERIOR, viewtype_values);
    AMG_Config::registerParameter<ViewType>("separation_exterior", "limit of calculations for coloring/smoothing <ViewType>", OWNED, viewtype_values);
    AMG_Config::registerParameter<int>("min_rows_latency_hiding", "number of rows at which to disable latency hiding, negative value means latency hiding is completely disabled", -1);
    AMG_Config::registerParameter<int>("matrix_halo_exchange", "0 - No halo exchange on lower levels, 1 - just diagonal values, 2 - full", 0);
    AMG_Config::registerParameter<std::string>("solver", "", "");
    AMG_Config::registerParameter<int>("verbosity_level", "verbosity level for output, 3 - custom print-outs <0|1|2|3>", 3);
    AMG_Config::registerParameter<ColoringType>("boundary_coloring", "handling of boundary coloring for ILU solvers <ColoringType>", SYNC_COLORS);
    AMG_Config::registerParameter<ColoringType>("halo_coloring", "handling of halo coloring for ILU solvers <ColoringType>", LAST);
}
