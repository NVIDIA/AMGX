// SPDX-FileCopyrightText: 2011 - 2024 NVIDIA CORPORATION. All Rights Reserved.
//
// SPDX-License-Identifier: BSD-3-Clause

#include "amgx_c.h"
#include <string>

// Tool for conversion MatrixMarket files to binary files (mainly for faster reading or storing on disk)
// compilation: g++ convert.c -o convert -lamgxsh -L../lib -Wl,-rpath=../lib
// run: convert <MMfile>

int main(int argc, char *argv[])
{
    AMGX_config_handle cfg;
    AMGX_matrix_handle A;
    AMGX_vector_handle b, x;
    AMGX_resources_handle rsrc;
    AMGX_Mode mode = AMGX_mode_hDDI;

    if (argc < 1)
    {
        printf("Specify matrix file as first argument");
        exit(2);
    }

    AMGX_SAFE_CALL(AMGX_initialize());
    AMGX_SAFE_CALL(AMGX_config_create(&cfg, "config_version=2, matrix_writer=binary"));
    AMGX_resources_create_simple(&rsrc, cfg);
    AMGX_matrix_create(&A, rsrc, mode);
    AMGX_vector_create(&x, rsrc, mode);
    AMGX_vector_create(&b, rsrc, mode);
    std::string arg = argv[1];
    int n, bsize_x, bsize_y, sol_size, sol_bsize;
    AMGX_read_system(A, b, x, arg.c_str());
    AMGX_matrix_get_size(A, &n, &bsize_x, &bsize_y);
    AMGX_vector_get_size(x, &sol_size, &sol_bsize);

    if (sol_size == 0 || sol_bsize == 0)
    {
        printf("Initializing solution with 0\n");
        AMGX_vector_set_zero(x, n, bsize_x);
    }

    arg = arg + ".bin";
    AMGX_write_system(A, b, x, arg.c_str());
    AMGX_resources_destroy(rsrc);
    AMGX_SAFE_CALL(AMGX_config_destroy(cfg));
    AMGX_SAFE_CALL(AMGX_finalize());
}
