// SPDX-FileCopyrightText: 2011 - 2025 NVIDIA CORPORATION. All Rights Reserved.
//
// SPDX-License-Identifier: BSD-3-Clause

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
