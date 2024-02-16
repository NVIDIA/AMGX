// SPDX-FileCopyrightText: 2011 - 2024 NVIDIA CORPORATION. All Rights Reserved.
//
// SPDX-License-Identifier: BSD-3-Clause

// This test is intended to check that the distributed matrix upload API
// accepts host, host registered, device, device managed pointers

#include "unit_test.h"
#include "amgx_c.h"
#include "amgxP_c.h"

namespace amgx

{

void preamble(
    AMGX_matrix_handle& A,
    AMGX_distribution_handle& dist,
    AMGX_resources_handle& rsrc,
    int& nrows,
    int& nnz,
    std::vector<int>& rows,
    std::vector<int>& cols)
{
    AMGX_finalize();

    nrows = 10;
    nnz = nrows*nrows;

    rows.resize(nrows+1);
    cols.resize(nnz);

    rows[0] = 0;
    for(int i = 0; i < nrows; ++i)
    {
        rows[i+1] = (i+1)*nrows;
        for(int j = 0; j < nrows; ++j)
        {
            cols[i*nrows + j] = j;
        }
    }

    std::string config_string;
    config_string="config_version=2, ";
    config_string+="solver(slv)=PCG, ";
    config_string+="slv:preconditioner(amg)=NOSOLVER, ";
    config_string+="slv:print_solve_stats=1, ";
    config_string+="slv:obtain_timings=1, ";
    config_string+="slv:max_iters=100, ";
    config_string+="slv:monitor_residual=1, ";
    config_string+="slv:convergence=ABSOLUTE, ";
    config_string+="slv:tolerance=1e-07, ";
    config_string+="slv:norm=L2";

    int dev = 0;
    cudaSetDevice(dev);

    AMGX_initialize();

    AMGX_config_handle cfg;
    AMGX_config_create(&cfg, config_string.c_str());
    AMGX_config_add_parameters(&cfg, "exception_handling=1");

    AMGX_resources_create_simple(&rsrc, cfg);

    AMGX_Mode mode = AMGX_mode_dDDI;
    AMGX_matrix_create(&A, rsrc, mode);

    AMGX_distribution_create(&dist, cfg);
    AMGX_distribution_set_32bit_colindices(dist, true);
    AMGX_distribution_set_partition_data(dist, AMGX_DIST_PARTITION_VECTOR, NULL);
}

void cleanup(AMGX_matrix_handle& A, AMGX_resources_handle& rsrc)
{
    AMGX_matrix_destroy(A);
    AMGX_resources_destroy(rsrc);
    AMGX_finalize();
}

DECLARE_UNITTEST_BEGIN(CAPIUploadCudaMalloc);

void run()
{
    AMGX_matrix_handle A;
    AMGX_distribution_handle dist;
    AMGX_resources_handle rsrc;
    int nrows; 
    int nnz;

    std::vector<int> rows_h;
    std::vector<int> cols_h;
    preamble(A, dist, rsrc, nrows, nnz, rows_h, cols_h);

    int* rows;
    cudaMalloc(&rows, sizeof(int)*(nrows+1));
    cudaMemcpy(rows, rows_h.data(), sizeof(int)*(nrows+1), cudaMemcpyDefault);

    int* cols;
    cudaMalloc(&cols, sizeof(int)*nnz);
    cudaMemcpy(cols, cols_h.data(), sizeof(int)*nnz, cudaMemcpyDefault);

    double* vals;
    cudaMalloc(&vals, sizeof(double)*nnz);

    UNITTEST_ASSERT_EQUAL(AMGX_matrix_upload_distributed(A, nrows, nrows, nnz, 1, 1, rows, cols, vals, NULL, dist), AMGX_RC_OK);

    cudaFree(rows);
    cudaFree(cols);

    cleanup(A, rsrc);
}

DECLARE_UNITTEST_END(CAPIUploadCudaMalloc);

DECLARE_UNITTEST_BEGIN(CAPIUploadCudaMallocManaged);

void run()
{
    AMGX_matrix_handle A;
    AMGX_distribution_handle dist;
    AMGX_resources_handle rsrc;
    int nrows;
    int nnz;

    std::vector<int> rows_h;
    std::vector<int> cols_h;
    preamble(A, dist, rsrc, nrows, nnz, rows_h, cols_h);

    int* rows;
    cudaMallocManaged(&rows, sizeof(int)*(nrows+1));
    cudaMemcpy(rows, rows_h.data(), sizeof(int)*(nrows+1), cudaMemcpyDefault);

    int* cols;
    cudaMallocManaged(&cols, sizeof(int)*nnz);
    cudaMemcpy(cols, cols_h.data(), sizeof(int)*nnz, cudaMemcpyDefault);

    double* vals;
    cudaMallocManaged(&vals, sizeof(double)*nnz);

    UNITTEST_ASSERT_EQUAL(AMGX_matrix_upload_distributed(A, nrows, nrows, nnz, 1, 1, rows, cols, vals, NULL, dist), AMGX_RC_OK);

    cudaFree(rows);
    cudaFree(cols);
    cleanup(A, rsrc);
}

DECLARE_UNITTEST_END(CAPIUploadCudaMallocManaged);

DECLARE_UNITTEST_BEGIN(CAPIUploadNew);

void run()
{
    AMGX_matrix_handle A;
    AMGX_distribution_handle dist;
    AMGX_resources_handle rsrc;
    int nrows;
    int nnz;

    std::vector<int> rows_h;
    std::vector<int> cols_h;
    preamble(A, dist, rsrc, nrows, nnz, rows_h, cols_h);

    double* vals = new double[nnz];

    UNITTEST_ASSERT_EQUAL(AMGX_matrix_upload_distributed(A, nrows, nrows, nnz, 1, 1, rows_h.data(), cols_h.data(), vals, NULL, dist), AMGX_RC_OK);
    cleanup(A, rsrc);
}

DECLARE_UNITTEST_END(CAPIUploadNew);

DECLARE_UNITTEST_BEGIN(CAPIUploadCudaHostRegister);

void run()
{
    AMGX_matrix_handle A;
    AMGX_distribution_handle dist;
    AMGX_resources_handle rsrc;
    int nrows; 
    int nnz;

    std::vector<int> rows_h;
    std::vector<int> cols_h;
    preamble(A, dist, rsrc, nrows, nnz, rows_h, cols_h);

    cudaHostRegister(rows_h.data(), sizeof(int)*(nrows+1), cudaHostRegisterDefault);

    cudaHostRegister(cols_h.data(), sizeof(int)*nnz, cudaHostRegisterDefault);

    double* vals = new double[nnz];
    cudaHostRegister(&vals, sizeof(double)*nnz, cudaHostRegisterDefault);

    UNITTEST_ASSERT_EQUAL(AMGX_matrix_upload_distributed(A, nrows, nrows, nnz, 1, 1, rows_h.data(), cols_h.data(), vals, NULL, dist), AMGX_RC_OK);
    cleanup(A, rsrc);
}

DECLARE_UNITTEST_END(CAPIUploadCudaHostRegister);

DECLARE_UNITTEST_BEGIN(CAPIUploadCudaMallocHost);

void run()
{
    AMGX_matrix_handle A;
    AMGX_distribution_handle dist;
    AMGX_resources_handle rsrc;
    int nrows; 
    int nnz;

    std::vector<int> rows_h;
    std::vector<int> cols_h;
    preamble(A, dist, rsrc, nrows, nnz, rows_h, cols_h);

    int* rows;
    cudaMallocHost(&rows, sizeof(int)*(nrows+1));
    memcpy(rows, rows_h.data(), sizeof(int)*(nrows+1));

    int* cols;
    cudaMallocHost(&cols, sizeof(int)*nnz);
    memcpy(cols, cols_h.data(), sizeof(int)*nnz);

    double* vals;
    cudaMallocHost(&vals, sizeof(double)*nnz);

    UNITTEST_ASSERT_EQUAL(AMGX_matrix_upload_distributed(A, nrows, nrows, nnz, 1, 1, rows, cols, vals, NULL, dist), AMGX_RC_OK);

    cudaFreeHost(rows);
    cudaFreeHost(cols);
    cleanup(A, rsrc);
}

DECLARE_UNITTEST_END(CAPIUploadCudaMallocHost);

// or you can specify several desired configs
CAPIUploadCudaMallocHost <TemplateMode<AMGX_mode_dDDI>::Type>  CAPIUploadCudaMallocHost_dDDI;
CAPIUploadCudaHostRegister <TemplateMode<AMGX_mode_dDDI>::Type>  CAPIUploadCudaHostRegister_dDDI;
CAPIUploadNew <TemplateMode<AMGX_mode_dDDI>::Type>  CAPIUploadNew_dDDI;
CAPIUploadCudaMallocManaged <TemplateMode<AMGX_mode_dDDI>::Type>  CAPIUploadCudaMallocManaged_dDDI;
CAPIUploadCudaMalloc <TemplateMode<AMGX_mode_dDDI>::Type>  CAPIUploadCudaMalloc_dDDI;

} //namespace amgx
