/* Copyright (c) 2011-2017, NVIDIA CORPORATION. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *  * Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *  * Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *  * Neither the name of NVIDIA CORPORATION nor the names of its
 *    contributors may be used to endorse or promote products derived
 *    from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
 * EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 * PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
 * CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 * EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 * PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
 * OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

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
    amgx_output(buffer, strlen(buffer));
    return retval;
}

} // namespace amgx
