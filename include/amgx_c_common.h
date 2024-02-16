// SPDX-FileCopyrightText: 2011 - 2024 NVIDIA CORPORATION. All Rights Reserved.
//
// SPDX-License-Identifier: BSD-3-Clause

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