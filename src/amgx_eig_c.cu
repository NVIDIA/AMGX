/* Copyright (c) 2013-2017, NVIDIA CORPORATION. All rights reserved.
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

#ifdef _WIN32
#ifndef AMGX_API_EXPORTS
#define AMGX_API_EXPORTS
#endif
#endif

#include "vector.h"

#include <amgx_eig_c.h>
#include <eigensolvers/eigensolver.h>

#include "amg_eigensolver.h"

#include "amgx_c_wrappers.inl"
#include "amgx_c_common.h"


namespace amgx
{

AMGX_RC getResourcesFromEigenSolverHandle(AMGX_eigensolver_handle slv, Resources **resources)
{
    AMGX_ERROR rc = AMGX_OK;

    try
    {
        auto& slv_wrp = *((CWrapHandle<AMGX_eigensolver_handle, AMG_EigenSolver<TConfigGeneric>>*)slv)->wrapped();
        *resources = slv_wrp.getResources();
    }
    AMGX_CATCHES(rc)
    AMGX_CHECK_API_ERROR(rc, NULL)
    return AMGX_RC_OK;
}

inline AMGX_ERROR eigensolve_setup(AMGX_eigensolver_handle slv,
                                   AMGX_matrix_handle mtx,
                                   Resources *resources)
{
    auto& solver = *((CWrapHandle<AMGX_eigensolver_handle, AMG_EigenSolver<TConfigGeneric>>*)slv)->wrapped();
    auto& A_wrp = ((CWrapHandle<AMGX_matrix_handle, Matrix<TConfigGeneric>>*)mtx)->wrapped();

    if (A_wrp->getResources() != solver.getResources())
    {
        FatalError("Error: matrix and solver use different resources object, exiting", AMGX_ERR_BAD_PARAMETERS);
    }

    return solver.setup_capi_no_throw(A_wrp);
}


inline AMGX_ERROR eigensolve_setup_pagerank(AMGX_eigensolver_handle slv,
        AMGX_vector_handle a,
        Resources *resources)
{
    auto& solver = ((CWrapHandle<AMGX_eigensolver_handle, AMG_EigenSolver<TConfigGeneric>>*)slv)->wrapped();
    auto& vec_wrp = ((CWrapHandle<AMGX_vector_handle, Vector<TConfigGeneric>>*)a)->wrapped();

    if (vec_wrp->getResources() != solver->getResources())
    {
        FatalError("Error: matrix and solver use different resources object, exiting", AMGX_ERR_BAD_PARAMETERS);
    }

    return solver->pagerank_setup_no_throw(*vec_wrp);
}



inline AMGX_ERROR eigensolve_solve(AMGX_eigensolver_handle slv,
                                   AMGX_vector_handle sol,
                                   Resources *resources)
{
    auto& slv_wrp = ((CWrapHandle<AMGX_eigensolver_handle, AMG_EigenSolver<TConfigGeneric>>*)slv)->wrapped();
    auto& x_wrp = ((CWrapHandle<AMGX_vector_handle, Vector<TConfigGeneric>>*)sol)->wrapped();

    if (x_wrp->getResources() != slv_wrp->getResources())
    {
        FatalError("Error: Inconsistency between solver and sol resources object, exiting", AMGX_ERR_BAD_PARAMETERS);
    }

    AMGX_STATUS solve_status;
    AMGX_ERROR ret = slv_wrp->solve_no_throw(*x_wrp, solve_status);
    return ret;
}



}

using namespace amgx;
extern "C" {

    typedef CWrapHandle<AMGX_config_handle, AMG_Configuration> ConfigW;
    typedef CWrapHandle<AMGX_resources_handle, Resources> ResourceW;

    AMGX_RC AMGX_eigensolver_create(AMGX_eigensolver_handle *slv, AMGX_resources_handle rsc, const AMGX_config_handle config_eigensolver)
    {
        AMGX_ERROR rc = AMGX_OK;
        Resources *resources = NULL;

        try
        {
            auto* config = ((CWrapHandle<AMGX_config_handle, AMG_Configuration>*)config_eigensolver)->wrapped().get();

            auto& resrcs = ((CWrapHandle<AMGX_resources_handle, Resources>*)rsc)->wrapped();

            if (!resrcs)
            {
                AMGX_CHECK_API_ERROR(AMGX_ERR_BAD_PARAMETERS, NULL);    //return AMGX_RC_BAD_PARAMETERS;
            }

            resources = resrcs.get();
            cudaSetDevice(resources->getDevice(0)); //because solver cnstr allocates resources on the device

            using Solver = AMG_EigenSolver<TConfigGeneric>;
            using SolveHandle = CWrapHandle<AMGX_eigensolver_handle, Solver>;
            auto* solver = get_mem_manager<SolveHandle>().
                template allocate<SolveHandle>(new Solver(resources, config)).get();

            solver->set_last_solve_status(AMGX_ST_ERROR);

            *slv = (AMGX_eigensolver_handle)solver;
        }
        AMGX_CATCHES(rc)
        AMGX_CHECK_API_ERROR(rc, resources);
        return AMGX_RC_OK;
    }

    AMGX_RC AMGX_eigensolver_setup(AMGX_eigensolver_handle eigensolver, AMGX_matrix_handle mtx)
    {
        Resources *resources;
        AMGX_CHECK_API_ERROR(getAMGXerror(getResourcesFromEigenSolverHandle(eigensolver, &resources)), NULL);
        AMGX_ERROR rc = AMGX_OK;

        try
        {
            eigensolve_setup(eigensolver, mtx, resources);
        }
        AMGX_CATCHES(rc)
        return getCAPIerror_x(rc);
    }


    AMGX_RC AMGX_eigensolver_pagerank_setup(AMGX_eigensolver_handle eigensolver, AMGX_vector_handle a)
    {
        Resources *resources;
        AMGX_CHECK_API_ERROR(getAMGXerror(getResourcesFromEigenSolverHandle(eigensolver, &resources)), NULL);
        AMGX_ERROR rc = AMGX_OK;

        try
        {
            eigensolve_setup_pagerank(eigensolver, a, resources);
        }
        AMGX_CATCHES(rc)
        return getCAPIerror_x(rc);
    }


    AMGX_RC AMGX_eigensolver_solve(AMGX_eigensolver_handle eigensolver, AMGX_vector_handle x)
    {
        Resources *resources;
        AMGX_CHECK_API_ERROR(getAMGXerror(getResourcesFromEigenSolverHandle(eigensolver, &resources)), NULL);
        AMGX_ERROR rc = AMGX_OK;

        try
        {
            AMGX_ERROR rcs = eigensolve_solve(eigensolver, x, resources);
            AMGX_CHECK_API_ERROR(rcs, resources);
        }
        AMGX_CATCHES(rc)
        return getCAPIerror_x(rc);
    }

    AMGX_RC AMGX_eigensolver_destroy(AMGX_eigensolver_handle slv)
    {
        Resources *resources;
        AMGX_CHECK_API_ERROR(getAMGXerror(getResourcesFromEigenSolverHandle(slv, &resources)), NULL);
        AMGX_ERROR rc = AMGX_OK;

        try
        {
            using SolverHandle = CWrapHandle<AMGX_eigensolver_handle, AMG_EigenSolver<TConfigGeneric>>;
            auto* solver = (SolverHandle*)slv;
            get_mem_manager<SolverHandle>().template free<SolverHandle>(solver);
        }
        AMGX_CATCHES(rc)
        AMGX_CHECK_API_ERROR(rc, resources)
        return AMGX_RC_OK;
    }

}//extern "C"
