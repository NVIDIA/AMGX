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

#include <error.h>
#include <map>
#include <string>

#define addError(code, err)  ErrorStrings::errors[code] = std::string(err)

namespace amgx
{

// support MPI error handling
#ifdef AMGX_WITH_MPI
using namespace amgx;

void MPIErrorHandler(MPI_Comm *comm, int *errcode, ...)
{
    char string_name[1000];
    int result_len = 1000;
    MPI_Error_string(*errcode, string_name, &result_len);
    std::stringstream _error;
    _error << "AMGX MPI Error: \"" << string_name << "\"";
    FatalError(_error.str(), AMGX_ERR_CUDA_FAILURE);
}

MPI_Errhandler glbMPIErrorHandler = MPI_ERRHANDLER_NULL;

void registerDefaultMPIErrHandler()
{
    int mpi_initialized = 0;
    MPI_Initialized(&mpi_initialized); // We want to make sure MPI_Init has been called.
    if ( mpi_initialized)
    {
        MPI_Comm_create_errhandler(MPIErrorHandler, &glbMPIErrorHandler);
    }
}

void freeDefaultMPIErrHandler()
{
    int mpi_initialized = 0;
    MPI_Initialized(&mpi_initialized); // We want to make sure MPI_Init has been called.
    
    if ( (glbMPIErrorHandler != MPI_ERRHANDLER_NULL) && mpi_initialized )
    {
        MPI_Errhandler_free(&glbMPIErrorHandler); //sets to MPI_ERRHANDLER_NULL
    }
}

#endif

class ErrorStrings
{
        std::map<AMGX_ERROR, std::string> errors;

    public:

        ErrorStrings()
        {
            addError(AMGX_OK,                              "No error.");
            addError(AMGX_ERR_BAD_PARAMETERS,              "Incorrect parameters for amgx call.");
            addError(AMGX_ERR_UNKNOWN,                     "Unknown error.");
            addError(AMGX_ERR_NOT_SUPPORTED_TARGET,        "Unsupported device/host algorithm.");
            addError(AMGX_ERR_NOT_SUPPORTED_BLOCKSIZE,     "Unsupported block size for the algorithm.");
            addError(AMGX_ERR_CUDA_FAILURE,                "CUDA kernel launch error.");
            addError(AMGX_ERR_THRUST_FAILURE,              "Thrust failure.");
            addError(AMGX_ERR_NO_MEMORY,                   "Insufficient memory.");
            addError(AMGX_ERR_IO,                          "I/O error.");
            addError(AMGX_ERR_BAD_MODE,                    "Incorrect C API mode.");
            addError(AMGX_ERR_CORE,                        "Error initializing amgx core.");
            addError(AMGX_ERR_PLUGIN,                      "Error initializing plugins.");
            addError(AMGX_ERR_CONFIGURATION,               "Incorrect amgx configuration provided.");
            addError(AMGX_ERR_NOT_IMPLEMENTED,             "Configuration feature is not implemented.");
            addError(AMGX_ERR_LICENSE_NOT_FOUND,           "Valid license is not found.");
            addError(AMGX_ERR_INTERNAL,                    "Internal error.");
        };

        const char *GetErrorString(AMGX_ERROR error)
        {
            std::map<AMGX_ERROR, std::string>::iterator iter = errors.find(error);

            if (iter != errors.end())
            {
                return iter->second.c_str();
            }
            else
            {
                return errors[AMGX_ERR_UNKNOWN].c_str();
            }
        };
};

static ErrorStrings errorstrings;

int AMGX_GetErrorString( AMGX_ERROR error, char *buffer, int buf_len)
{
    strncpy(buffer, errorstrings.GetErrorString(error), buf_len);
    buffer[buf_len - 1] = 0;
    return 0;
}

}
