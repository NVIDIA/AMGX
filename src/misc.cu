// SPDX-FileCopyrightText: 2011 - 2025 NVIDIA CORPORATION. All Rights Reserved.
//
// SPDX-License-Identifier: BSD-3-Clause

#include <misc.h>
#include <stdio.h>
#include <stdarg.h>

#ifdef AMGX_WITH_MPI
#include <mpi.h>
#endif

namespace amgx
{

#define PRINT_BUF_SIZE 4096

void amgx_default_output(const char *msg, int length)
{
    printf("%s", msg);
}

void amgx_dist_output(const char *msg, int length)
{
#ifdef AMGX_WITH_MPI
    int rank = 0;
    int mpi_initialized = 0;
    MPI_Initialized(&mpi_initialized); // We want to make sure MPI_Init has been called.

    if (mpi_initialized)
    {
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    }

    if (rank == 0) { amgx_output(msg, length); }

#else
    amgx_output(msg, length);
#endif
}

AMGX_output_callback amgx_output = amgx_default_output;
AMGX_output_callback error_output = amgx_default_output;
AMGX_output_callback amgx_distributed_output = amgx_dist_output;

int amgx_printf(const char *fmt, ...)
{
    int retval = 0;
    char buffer[PRINT_BUF_SIZE];
    va_list ap;
    va_start(ap, fmt);
    retval = vsnprintf(buffer, PRINT_BUF_SIZE, fmt, ap);
    va_end(ap);
    amgx_distributed_output(buffer, strlen(buffer));
    return retval;
}

} // namespace amgx
