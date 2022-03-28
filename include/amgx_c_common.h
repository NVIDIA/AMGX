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
#pragma once

#include <map>
#include <memory>
#include <tuple>
#include <stdexcept>
#include <sstream>

#include "amgx_c.h"
#include "resources.h"


namespace amgx
{


AMGX_RC getCAPIerror_x(AMGX_ERROR err);
AMGX_ERROR getAMGXerror(AMGX_RC err);
void amgx_error_exit(Resources *rsc = NULL, int err = 1);

#define AMGX_CHECK_API_ERROR(rc, rsc) \
  { \
    int handle_errors = 0; \
    AMGX_RC _check_err = getCAPIerror_x(rc);     \
    if (rsc != NULL)\
      handle_errors = ((Resources*) rsc)->getHandleErrors(); \
    if (handle_errors) { \
      char msg[4096];   \
      switch(_check_err) {    \
      case AMGX_RC_OK: \
        break; \
      default: \
        fprintf(stderr, "AMGX ERROR: file %s line %6d\n", __FILE__, __LINE__); \
        AMGX_GetErrorString(rc, msg, 4096);\
        fprintf(stderr, "AMGX ERROR: %s\n", msg); \
        amgx_error_exit(rsc);\
        break; \
      } \
      } \
      else \
      if (_check_err != AMGX_RC_OK) \
        return _check_err; \
  }

//memory manager for malloc/free arrays
struct MemCArrManager
{
        struct CDeleter
        {
            void operator() (void *ptr)
            {
                ::free(ptr);
            }
        };

        template<typename T>
        T *allocate(size_t nelems)
        {
            T *ptr = malloc(nelems * sizeof(T));
            pool_.insert(std::make_pair(ptr, std::shared_ptr<void>(ptr, CDeleter())));
            return ptr;
        }

        void *allocate(size_t n_all)
        {
            void *ptr = malloc(n_all);
            pool_.insert(std::make_pair(ptr, std::shared_ptr<void>(ptr, CDeleter())));
            return ptr;
        }

        template<typename T>
        T *callocate(size_t nelems)
        {
            T *ptr = (T *)calloc(nelems, sizeof(T));
            pool_.insert(std::make_pair(ptr, std::shared_ptr<void>(ptr, CDeleter())));
            return ptr;
        }

        bool free(void *ptr)
        {
            return (pool_.erase(ptr) > 0);
        }

    private:
        std::map<void *, std::shared_ptr<void>> pool_;
};

MemCArrManager &get_c_arr_mem_manager(void);

// defined in amgx_c.h
AMGX_RC getResourcesFromSolverHandle(AMGX_solver_handle slv, Resources **resources);

AMGX_RC getResourcesFromMatrixHandle(AMGX_matrix_handle mtx, Resources **resources);

AMGX_RC getResourcesFromVectorHandle(AMGX_vector_handle vec, Resources **resources);

}//end namespace amgx