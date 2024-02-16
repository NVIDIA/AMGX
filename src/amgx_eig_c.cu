// SPDX-FileCopyrightText: 2013 - 2024 NVIDIA CORPORATION. All Rights Reserved.
//
// SPDX-License-Identifier: BSD-3-Clause

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

    AMGX_TRIES()
    {
        AMGX_Mode mode = get_mode_from<AMGX_eigensolver_handle>(slv);

        switch (mode)
        {
#define AMGX_CASE_LINE(CASE) case CASE: { \
        *resources = get_mode_object_from<CASE, AMG_EigenSolver, AMGX_eigensolver_handle>(slv)->getResources();\
        } \
        break;
                AMGX_FORALL_BUILDS(AMGX_CASE_LINE)
                AMGX_FORCOMPLEX_BUILDS(AMGX_CASE_LINE)
#undef AMGX_CASE_LINE

            default:
                AMGX_CHECK_API_ERROR(AMGX_ERR_BAD_MODE, NULL);
        }
    }

    AMGX_CATCHES(rc)
    AMGX_CHECK_API_ERROR(rc, NULL)
    return AMGX_RC_OK;
}

template<AMGX_Mode CASE,
         template<typename> class SolverType,
         template<typename> class MatrixType>
inline AMGX_ERROR eigensolve_setup(AMGX_eigensolver_handle slv,
                                   AMGX_matrix_handle mtx,
                                   Resources *resources)
{
    typedef SolverType<typename TemplateMode<CASE>::Type> SolverLetterT;
    typedef CWrapHandle<AMGX_eigensolver_handle, SolverLetterT> SolverW;
    typedef MatrixType<typename TemplateMode<CASE>::Type> MatrixLetterT;
    typedef CWrapHandle<AMGX_matrix_handle, MatrixLetterT> MatrixW;
    MatrixW wrapA(mtx);
    MatrixLetterT &A = *wrapA.wrapped();
    SolverW wrapSolver(slv);
    SolverLetterT &solver = *wrapSolver.wrapped();

    if (wrapA.mode() != wrapSolver.mode() )
    {
        FatalError("Error: mismatch between Matrix mode and Solver Mode.\n", AMGX_ERR_BAD_PARAMETERS);
    }

    if (A.getResources() != solver.getResources())
    {
        FatalError("Error: matrix and solver use different resources object, exiting", AMGX_ERR_BAD_PARAMETERS);
    }

    //cudaSetDevice(solver.getResources()->getDevice(0));
    return solver.setup_capi_no_throw(wrapA.wrapped());
}


template<AMGX_Mode CASE,
         template<typename> class SolverType,
         template<typename> class VectorType>
inline AMGX_ERROR eigensolve_setup_pagerank(AMGX_eigensolver_handle slv,
        AMGX_vector_handle a,
        Resources *resources)
{
    typedef SolverType<typename TemplateMode<CASE>::Type> SolverLetterT;
    typedef CWrapHandle<AMGX_eigensolver_handle, SolverLetterT> SolverW;
    typedef VectorType<typename TemplateMode<CASE>::Type> VectorLetterT;
    typedef CWrapHandle<AMGX_vector_handle, VectorLetterT> VectorW;
    VectorW wrapA(a);
    VectorLetterT &vec = *wrapA.wrapped();
    SolverW wrapSolver(slv);
    SolverLetterT &solver = *wrapSolver.wrapped();

    if (wrapA.mode() != wrapSolver.mode() )
    {
        FatalError("Error: mismatch between Matrix mode and Solver Mode.\n", AMGX_ERR_BAD_PARAMETERS);
    }

    if (vec.getResources() != solver.getResources())
    {
        FatalError("Error: matrix and solver use different resources object, exiting", AMGX_ERR_BAD_PARAMETERS);
    }

    //cudaSetDevice(solver.getResources()->getDevice(0));
    return solver.pagerank_setup_no_throw(vec);
}



template<AMGX_Mode CASE,
         template<typename> class SolverType,
         template<typename> class VectorType>
inline AMGX_ERROR eigensolve_solve(AMGX_eigensolver_handle slv,
                                   AMGX_vector_handle sol,
                                   Resources *resources)
{
    typedef SolverType<typename TemplateMode<CASE>::Type> SolverLetterT;
    typedef CWrapHandle<AMGX_eigensolver_handle, SolverLetterT> SolverW;
    typedef VectorType<typename TemplateMode<CASE>::Type> VectorLetterT;
    typedef CWrapHandle<AMGX_vector_handle, VectorLetterT> VectorW;
    SolverW wrapSolver(slv);
    SolverLetterT &solver = *wrapSolver.wrapped();
    VectorW wrapSol(sol);
    VectorLetterT &x = *wrapSol.wrapped();

    if (wrapSol.mode() != wrapSolver.mode())
    {
        FatalError("Error: mismatch between X mode and Solver Mode.\n", AMGX_ERR_BAD_PARAMETERS);
    }

    if (x.getResources() != solver.getResources())
    {
        FatalError("Error: Inconsistency between solver and sol resources object, exiting", AMGX_ERR_BAD_PARAMETERS);
    }

    AMGX_STATUS solve_status;
    AMGX_ERROR ret = solver.solve_no_throw(x, solve_status);
    return ret;
}



}

using namespace amgx;
extern "C" {

    typedef CWrapHandle<AMGX_config_handle, AMG_Configuration> ConfigW;
    typedef CWrapHandle<AMGX_resources_handle, Resources> ResourceW;

    AMGX_RC AMGX_eigensolver_create(AMGX_eigensolver_handle *ret, AMGX_resources_handle rsc, AMGX_Mode mode, const AMGX_config_handle config_eigensolver)
    {
        AMGX_ERROR rc = AMGX_OK;
        AMGX_RC rc_solver;
        Resources *resources = NULL;

        AMGX_TRIES()
        {
            ///amgx::CWrapper<AMGX_resources_handle> *c_resources= (amgx::CWrapper<AMGX_resources_handle>*)rsc;
            ResourceW c_r(rsc);
            ConfigW cfg(config_eigensolver);

            ///if (!c_resources)
            if (!c_r.wrapped())
            {
                AMGX_CHECK_API_ERROR(AMGX_ERR_BAD_PARAMETERS, NULL);    //return AMGX_RC_BAD_PARAMETERS;
            }

            resources = c_r.wrapped().get();/// (Resources*)(c_resources->hdl);
            cudaSetDevice(resources->getDevice(0));//because solver cnstr allocates resources on the device

            switch (mode)
            {
#define AMGX_CASE_LINE(CASE) case CASE: {       \
             auto* solver = create_managed_mode_object<CASE, AMG_EigenSolver, AMGX_eigensolver_handle>(ret, mode, resources, cfg.wrapped().get()); \
             solver->set_last_solve_status(AMGX_ST_ERROR); \
             rc_solver = solver->is_valid() ? AMGX_RC_OK : AMGX_RC_UNKNOWN; \
           }            \
      break;
                    AMGX_FORALL_BUILDS(AMGX_CASE_LINE)
                    AMGX_FORCOMPLEX_BUILDS(AMGX_CASE_LINE)
#undef AMGX_CASE_LINE

                default:
                    AMGX_CHECK_API_ERROR(AMGX_ERR_BAD_MODE, resources)
            }
        }

        AMGX_CATCHES(rc)
        AMGX_CHECK_API_ERROR(rc, resources);
        return rc_solver;
    }

    AMGX_RC AMGX_eigensolver_setup(AMGX_eigensolver_handle eigensolver, AMGX_matrix_handle mtx)
    {
        Resources *resources;
        AMGX_CHECK_API_ERROR(getAMGXerror(getResourcesFromEigenSolverHandle(eigensolver, &resources)), NULL);
        AMGX_ERROR rc = AMGX_OK;

        AMGX_TRIES()
        {
            AMGX_Mode mode = get_mode_from<AMGX_eigensolver_handle>(eigensolver);

            switch (mode)
            {
#define AMGX_CASE_LINE(CASE) case CASE: { \
      typedef TemplateMode<CASE>::Type TConfig; \
      eigensolve_setup<CASE, AMG_EigenSolver, Matrix>(eigensolver, mtx, resources); \
      break;\
        }
                    AMGX_FORALL_BUILDS(AMGX_CASE_LINE)
                    AMGX_FORCOMPLEX_BUILDS(AMGX_CASE_LINE)
#undef AMGX_CASE_LINE

                default:
                    AMGX_CHECK_API_ERROR(AMGX_ERR_BAD_MODE, resources) \
            }
        }

        AMGX_CATCHES(rc)
        return getCAPIerror_x(rc);
    }


    AMGX_RC AMGX_eigensolver_pagerank_setup(AMGX_eigensolver_handle eigensolver, AMGX_vector_handle a)
    {
        Resources *resources;
        AMGX_CHECK_API_ERROR(getAMGXerror(getResourcesFromEigenSolverHandle(eigensolver, &resources)), NULL);
        AMGX_ERROR rc = AMGX_OK;

        AMGX_TRIES()
        {
            AMGX_Mode mode = get_mode_from<AMGX_eigensolver_handle>(eigensolver);

            switch (mode)
            {
#define AMGX_CASE_LINE(CASE) case CASE: { \
      typedef TemplateMode<CASE>::Type TConfig; \
      eigensolve_setup_pagerank<CASE, AMG_EigenSolver, Vector>(eigensolver, a, resources); \
      break;\
        }
                    AMGX_FORALL_BUILDS(AMGX_CASE_LINE)
                    AMGX_FORCOMPLEX_BUILDS(AMGX_CASE_LINE)
#undef AMGX_CASE_LINE

                default:
                    AMGX_CHECK_API_ERROR(AMGX_ERR_BAD_MODE, resources) \
            }
        }

        AMGX_CATCHES(rc)
        return getCAPIerror_x(rc);
    }


    AMGX_RC AMGX_eigensolver_solve(AMGX_eigensolver_handle eigensolver, AMGX_vector_handle x)
    {
        Resources *resources;
        AMGX_CHECK_API_ERROR(getAMGXerror(getResourcesFromEigenSolverHandle(eigensolver, &resources)), NULL);
        AMGX_ERROR rc = AMGX_OK;

        AMGX_TRIES()
        {
            AMGX_Mode mode = get_mode_from<AMGX_eigensolver_handle>(eigensolver);

            switch (mode)
            {
#define AMGX_CASE_LINE(CASE) case CASE: { \
        AMGX_ERROR rcs = eigensolve_solve<CASE, AMG_EigenSolver, Vector>(eigensolver, x, resources); \
        AMGX_CHECK_API_ERROR(rcs, resources); break;\
      }
                    AMGX_FORALL_BUILDS(AMGX_CASE_LINE)
                    AMGX_FORCOMPLEX_BUILDS(AMGX_CASE_LINE)
#undef AMGX_CASE_LINE

                default:
                    AMGX_CHECK_API_ERROR(AMGX_ERR_BAD_MODE, resources)
            }
        }

        AMGX_CATCHES(rc)
        return getCAPIerror_x(rc);
    }

    AMGX_RC AMGX_eigensolver_destroy(AMGX_eigensolver_handle slv)
    {
        Resources *resources;
        AMGX_CHECK_API_ERROR(getAMGXerror(getResourcesFromEigenSolverHandle(slv, &resources)), NULL);
        AMGX_ERROR rc = AMGX_OK;

        AMGX_TRIES()
        {
            AMGX_Mode mode = get_mode_from<AMGX_eigensolver_handle>(slv);

            switch (mode)
            {
                    //cudaSetDevice(get_mode_object_from<CASE,EigenSolver,AMGX_eigensolver_handle>(slv)->getResources()->getDevice(0));
#define AMGX_CASE_LINE(CASE) case CASE: { \
      \
      remove_managed_object<CASE, AMG_EigenSolver, AMGX_eigensolver_handle>(slv); \
      } \
      break;
                    AMGX_FORALL_BUILDS(AMGX_CASE_LINE)
                    AMGX_FORCOMPLEX_BUILDS(AMGX_CASE_LINE)
#undef AMGX_CASE_LINE

                default:
                    AMGX_CHECK_API_ERROR(AMGX_ERR_BAD_MODE, resources)
            }
        }

        AMGX_CATCHES(rc)
        AMGX_CHECK_API_ERROR(rc, resources)
        return AMGX_RC_OK;
    }

}//extern "C"
