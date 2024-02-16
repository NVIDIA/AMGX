// SPDX-FileCopyrightText: 2011 - 2024 NVIDIA CORPORATION. All Rights Reserved.
//
// SPDX-License-Identifier: BSD-3-Clause

#include "matrix.h"
#include "cuda_runtime.h"
#include "stdio.h"
#include "multiply.h"
#include "thrust/fill.h"

// example of using AMGX code to get matrix vector product
// to compile
// 1) either link against AMGX static library (might require linking against MPI depending on whether AMGX_WITH_MPI was used for AMGX compilation):
//    nvcc -std=c++11 --gpu-architecture=compute_35 --gpu-code=compute_35,sm_35 -I../include -I../../../thrust amgx_spmv_internal.cu -lamgx -L../build -lmpi -L/opt/openmpi-1.10.2/lib -lcublas -lcusparse -lcusolver -o amgx_spmv -Xlinker=-rpath=/usr/local/cuda/lib64
// 2) or use Makefile which will compile and link against selected AMGX files required for SpMV:
//    make example
// Run: 
// ./amgx_spmv

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

    amgx::Matrix<TConfig> A;
    amgx::Vector<TConfig> x;
    amgx::Vector<TConfig> y;
    A.setResources(&res);
    x.setResources(&res);
    y.setResources(&res);

    int nrows = 5;
    int nnz = 13;
    A.resize(nrows, nrows, nnz, 1);
    x.resize(nrows);
    y.resize(nrows);
    amgx::thrust::fill(y.begin(), y.end(), 0.);
    
    // matrix row offsets
    std::vector<int> row_offsets_h = {0, 2, 5, 7, 10, 13};
    A.row_offsets.assign(row_offsets_h.begin(), row_offsets_h.end());
    
    // matrix colums indices 
    std::vector<int> column_indices_h = {0, 2, 1, 3, 4, 0, 2, 2, 3, 4, 0, 2 ,4};
    A.col_indices.assign(column_indices_h.begin(), column_indices_h.end());
    
    // matrix values
    std::vector<double> matrix_values_h(nnz);
    for (auto &v : matrix_values_h) v = ((double)random())/RAND_MAX;
    A.values.assign(matrix_values_h.begin(), matrix_values_h.end());
    //set matrix "completeness" flag
    A.set_initialized(1);

    // vector values
    std::vector<double> x_h(nrows);
    for (auto &v : x_h) v = ((double)random())/RAND_MAX;
    x.assign(x_h.begin(), x_h.end());

    // AMGX multiply 
    amgx::multiply(A, x, y);

    // get the result to host
    VVector_h y_res_h = y;

    // reference check
    std::vector<double> y_res_ref(nrows, 0.);
    bool err_found = false;
    for (int r = 0; r < nrows; r++)
    {
        double y_res_ref = 0.;
        for (int c = row_offsets_h[r]; c < row_offsets_h[r + 1]; c++)
        {
            y_res_ref += matrix_values_h[c]*x_h[column_indices_h[c]];
        }
        if (std::abs(y_res_ref - y_res_h[r]) > 1e-8)
        {
            printf("Difference in row %d: reference: %f, AMGX: %f\n", r, y_res_ref, y_res_h[r]);
            err_found = true;
        }
    }

    if (!err_found)
        printf("Done!\n");
}

int main(int argc, char* argv[])
{
    // Initialization
    cudaSetDevice(0);
    // register required AMGX parameters 
    registerParameters();
    
    // resources object
    amgx::Resources res;

    // make sure we perform any AMGX functionality within Resources lifetime
    spmv_example(res);
    
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
}
