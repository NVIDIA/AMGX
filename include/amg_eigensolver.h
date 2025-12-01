// SPDX-FileCopyrightText: 2011 - 2025 NVIDIA CORPORATION. All Rights Reserved.
//
// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#include <string>
#include <iostream>

#ifdef _WIN32
#pragma warning (push)
#pragma warning (disable : 4244 4267 4521)
#endif
#ifdef _WIN32
#pragma warning (pop)
#endif

#include <fstream>
#include <limits>

#include <vector.h>
#include <matrix.h>
#include <basic_types.h>
#include <types.h>
#include <misc.h>

#include <amg_config.h>
#include <resources.h>
#include <thread_manager.h>

#include <error.h>

#include <memory>

#include <amgx_types/util.h>

namespace amgx
{

template <class T_Config> class AMG_EigenSolver;
template <class T_Config> class EigenSolver;

template <class T_Config>
class AMG_EigenSolver
{
        static const AMGX_VecPrecision vecPrec = T_Config::vecPrec;
        static const AMGX_MatPrecision matPrec = T_Config::matPrec;
        static const AMGX_IndPrecision indPrec = T_Config::indPrec;
        typedef TemplateConfig<AMGX_host, vecPrec, matPrec, indPrec> T_Config_h;
        typedef TemplateConfig<AMGX_device, vecPrec, matPrec, indPrec> T_Config_d;
        typedef Matrix<T_Config_h> Matrix_h;
        typedef Matrix<T_Config_d> Matrix_d;

        typedef Vector<T_Config_h> Vector_h;
        typedef Vector<T_Config_d> Vector_d;

        typedef typename T_Config_h::MatPrec ValueTypeA;
        typedef typename T_Config_h::VecPrec ValueTypeB;

        typedef typename T_Config_h::template setVecPrec< types::PODTypes< ValueTypeB >::vec_prec >::Type PODConfig_h;
        typedef typename T_Config_d::template setVecPrec< types::PODTypes< ValueTypeB >::vec_prec >::Type PODConfig_d;

        typedef Vector<PODConfig_h> PODVector_h;
        typedef Vector<PODConfig_d> PODVector_d;

    public:
        AMG_EigenSolver(Resources *res, AMG_Configuration *cfg = NULL);     // new in API v2, grab configuration by the pointer (if NULL - from resources), saves the pointer
        AMG_EigenSolver(Resources *res, AMG_Configuration &cfg);           // external configuration, saves the copy
        AMG_EigenSolver(const AMG_EigenSolver<T_Config>  &amg_solver);
        AMG_EigenSolver &operator=(const AMG_EigenSolver &amg_solver);
        ~AMG_EigenSolver();

        /****************************************************
        * Sets A as the matrix for the AMG system
        ****************************************************/
        void setup( Matrix<T_Config> &A );
        AMGX_ERROR setup_no_throw( Matrix<T_Config> &A );

        void pagerank_setup(Vector<T_Config> &a);
        AMGX_ERROR pagerank_setup_no_throw(Vector<T_Config> &a);

        /****************************************************
        * Sets A as the matrix for the AMG system
        ****************************************************/

        void setup_capi( std::shared_ptr<Matrix<T_Config>> pA0);
        AMGX_ERROR setup_capi_no_throw( std::shared_ptr<Matrix<T_Config>> pA0);

        /****************************************************
        * Solves the AMG system Ax=b.
        ***************************************************/
        AMGX_ERROR solve_no_throw( Vector<T_Config> &x, AMGX_STATUS &status);

        int get_num_iters();

        EigenSolver<T_Config> *getSolverObject( ) { return solver; }
        const EigenSolver<T_Config> *getSolverObject( ) const { return solver; }

        inline Resources *getResources() const { return m_resources; }
        inline AMG_Config *getConfig() const { return m_cfg; }
        inline void setResources(Resources *resources) { m_resources = resources; }

    private:
        void process_config(AMG_Config &in_cfg, std::string solver_scope);

        void init();

        AMG_Config *m_cfg;
        bool m_cfg_self;
        Resources *m_resources;
        EigenSolver<T_Config> *solver;

        int ref_count;

        // reusing matrix structure
        std::string structure_reuse_levels_scope;

        // Do we include timings.
        bool m_with_timings;
        cudaEvent_t m_setup_start, m_setup_stop;
        cudaEvent_t m_solve_start, m_solve_stop;

        Matrix<T_Config> &get_A(void)
        {
            return *m_ptrA;
        }

        std::shared_ptr<Matrix<T_Config>> m_ptrA;

        void mem_manage(Matrix<T_Config> &A)
        {
            m_ptrA.reset(new Matrix<T_Config>(A));
        }
};

} // namespace amgx
