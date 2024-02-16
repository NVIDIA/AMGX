// SPDX-FileCopyrightText: 2011 - 2024 NVIDIA CORPORATION. All Rights Reserved.
//
// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#ifdef _WIN32
#else
#include <execinfo.h>
#include <dlfcn.h>
#include <cxxabi.h>
#include <unistd.h>
#endif

#include <stdio.h>
#include <misc.h>
#include <string>
#include <sstream>

#include <stacktrace.h>

#ifdef AMGX_WITH_MPI
#include <mpi.h>
#endif

namespace amgx
{

// support MPI errors
#ifdef AMGX_WITH_MPI
void MPIErrorHandler(MPI_Comm *comm, int *errcode, ...);
extern MPI_Errhandler glbMPIErrorHandler;
void registerDefaultMPIErrHandler();
void freeDefaultMPIErrHandler();
#endif

enum AMGX_ERROR
{
    /*********************************************************
     * Flags for status reporting
     *********************************************************/
    AMGX_OK = 0,
    AMGX_ERR_BAD_PARAMETERS = 1,
    AMGX_ERR_UNKNOWN = 2,
    AMGX_ERR_NOT_SUPPORTED_TARGET = 3,
    AMGX_ERR_NOT_SUPPORTED_BLOCKSIZE = 4,
    AMGX_ERR_CUDA_FAILURE = 5,
    AMGX_ERR_THRUST_FAILURE = 6,
    AMGX_ERR_NO_MEMORY = 7,
    AMGX_ERR_IO = 8,
    AMGX_ERR_BAD_MODE = 9,
    AMGX_ERR_CORE = 10,
    AMGX_ERR_PLUGIN = 11,
    AMGX_ERR_CONFIGURATION = 12,
    AMGX_ERR_NOT_IMPLEMENTED = 13, // actually this error includes #3 and #4
    AMGX_ERR_LICENSE_NOT_FOUND = 14,
    AMGX_ERR_INTERNAL = 15,
};

// define our own bad_alloc so we can set its .what()
class amgx_exception: public std::exception
{
    public:
        inline amgx_exception(const std::string &w, const std::string &where, const std::string &trace, AMGX_ERROR reason) : m_trace(trace), m_what(w), m_reason(reason), m_where(where)
        {
        }

        inline virtual ~amgx_exception(void) throw () {};

        inline virtual const char *what(void) const throw()
        {
            return m_what.c_str();
        }
        inline virtual const char *where(void) const throw()
        {
            return m_where.c_str();
        }
        inline virtual const char *trace(void) const throw()
        {
            return m_trace.c_str();
        }
        inline virtual AMGX_ERROR reason(void) const throw()
        {
            return m_reason;
        }


    private:
        std::string  m_what;
        std::string  m_where;
        std::string  m_trace;
        AMGX_ERROR m_reason;
}; // end bad_alloc


int AMGX_GetErrorString( AMGX_ERROR error, char *buffer, int buf_len);

/********************************************************
 * Prints the error message, the stack trace, and exits
 * ******************************************************/
#define FatalError(s, reason) {                                                 \
  std::stringstream _where;                                                     \
  _where << __FILE__ << ':' << __LINE__;                                        \
  std::stringstream _trace;                                                     \
  printStackTrace(_trace);                                                      \
  cudaDeviceSynchronize();                                                      \
  throw amgx_exception(std::string(s) + "\n", _where.str(), _trace.str(), reason); \
}

#ifndef NDEBUG
#define cudaCheckError() {                                              \
  cudaDeviceSynchronize();                                              \
  cudaError_t e=cudaGetLastError();                                     \
  if(e!=cudaSuccess) {                                                  \
    std::stringstream _error;                                           \
    _error << "Cuda failure: '" << cudaGetErrorString(e) << "'";        \
    FatalError(_error.str(), AMGX_ERR_CUDA_FAILURE);                   \
  }                                                                     \
}

#else

#define cudaCheckError() {                                              \
  cudaError_t e=cudaGetLastError();                                     \
  if(e!=cudaSuccess) {                                                  \
    std::stringstream _error;                                           \
    _error << "Cuda failure: '" << cudaGetErrorString(e) << "'";        \
    FatalError(_error.str(), AMGX_ERR_CUDA_FAILURE);                   \
  }                                                                     \
}
#endif

//#define DISABLE_EXCEPTION_HANDLING

#ifndef DISABLE_EXCEPTION_HANDLING \

#define AMGX_TRIES() try

#define AMGX_CATCHES(rc) \
  catch (amgx_exception e) {                                                \
    std::string err = "Caught amgx exception: " + std::string(e.what()) + " at: "                   \
        + std::string(e.where()) + "\nStack trace:\n" + std::string(e.trace()) + "\n";               \
    error_output(err.c_str(), static_cast<int>(err.length()));                                       \
    rc = e.reason();                                                                                 \
  } catch (amgx::thrust::system_error &e) {                                                                \
    std::string err = "Thrust failure: " + std::string(e.what())                                     \
        + "\nFile and line number are not available for this exception.\n";                          \
    error_output(err.c_str(), static_cast<int>(err.length()));                                       \
    rc = AMGX_ERR_THRUST_FAILURE;                                                                   \
  } catch (amgx::thrust::system::detail::bad_alloc e) {                                                    \
    std::string err = "Thrust failure: " + std::string(e.what())                                     \
        + "\nFile and line number are not available for this exception.\n";                          \
    error_output(err.c_str(), static_cast<int>(err.length()));                                       \
    rc = AMGX_ERR_THRUST_FAILURE;                                                                   \
  } catch (std::bad_alloc e) {                                                                       \
    std::string err = "Not enough memory: " + std::string(e.what())                                  \
        + "\nFile and line number are not available for this exception.\n";                          \
    error_output(err.c_str(), static_cast<int>(err.length()));                                       \
    rc = AMGX_ERR_NO_MEMORY;                                                                        \
  } catch (std::exception e) {                                                                       \
    std::string err = "Caught unknown exception: " + std::string(e.what())                           \
        + "\nFile and line number are not available for this exception.\n";                          \
    error_output(err.c_str(), static_cast<int>(err.length()));                                       \
    rc = AMGX_ERR_UNKNOWN;                                                                          \
  } catch (...) {                                                                                    \
    std::string err =                                                                                \
        "Caught unknown exception\nFile and line number are not available for this exception.\n";    \
    error_output(err.c_str(), static_cast<int>(err.length()));                                       \
    rc = AMGX_ERR_UNKNOWN;                                                                          \
  }

#else

#define AMGX_TRIES()

#define AMGX_CATCHES(rc)

#endif


} // namespace amgx
