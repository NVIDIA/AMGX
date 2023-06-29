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

#include <amg_solver.h>
#include <amg_config.h>
#include <amg.h>
#include <basic_types.h>
#include <misc.h>
#include <assert.h>
#include <util.h>

using std::string;

namespace amgx
{

template< class T_Config >
void AMG_Solver<T_Config>::process_config(AMG_Config &in_cfg, std::string solver_scope)
{
    if (in_cfg.getParameter<int>("print_config", solver_scope) == 1)
    {
        in_cfg.printAMGConfig();
    }
}

template< class T_Config >
void AMG_Solver<T_Config>::init()
{
    std::string solver_value, solver_scope;
    m_cfg->template getParameter<std::string>("solver", solver_value, "default", solver_scope);
    process_config(*m_cfg, solver_scope);
    // pass thread manager to solver
    solver = SolverFactory<T_Config>::allocate(*m_cfg, "default", "solver", m_resources->get_tmng());
    structure_reuse_levels_scope = "";

    // Reusing structure
    if (solver_value == "AMG") // AMG is used as main solver
    {
        structure_reuse_levels_scope = solver_scope;
    }
    else
    {
        // AMG might be used as preconditioner of the main solver    std::string preconditioner_value, preconditioner_scope;
        std::string preconditioner_value, preconditioner_scope;
        m_cfg->template getParameter<std::string>("preconditioner", preconditioner_value, solver_scope, preconditioner_scope);

        if (preconditioner_value == "AMG")
        {
            structure_reuse_levels_scope = preconditioner_scope;
        }
        else
        {
            structure_reuse_levels_scope = "";
        }
    }

    if ( m_with_timings )
    {
        cudaEventCreate(&m_setup_start);
        cudaEventCreate(&m_setup_stop);
        cudaEventCreate(&m_solve_start);
        cudaEventCreate(&m_solve_stop);
    }
}

template< class T_Config >
AMG_Solver<T_Config>::AMG_Solver(Resources *res, AMG_Configuration *cfg) : m_with_timings(false), m_resources(res), m_cfg_self(false)
{
    if (cfg)
    {
        m_cfg = cfg->getConfigObject();
    }
    else
    {
        m_cfg = res->getResourcesConfig();
    }

    init();
}

template< class T_Config >
AMG_Solver<T_Config>::AMG_Solver(Resources *res, AMG_Configuration &cfg) : m_with_timings(false), m_resources(res), m_cfg_self(true)
{
    m_cfg = new AMG_Config;
    *m_cfg = *(cfg.getConfigObject());
    init();
}

template< class T_Config >
AMG_Solver<T_Config>::AMG_Solver(const AMG_Solver<T_Config> &amg_solver)
{
    solver = amg_solver.solver;
    m_resources = amg_solver.getResources();
    m_cfg = amg_solver.getConfig();
    m_cfg_self = false;
    solver->incr_ref_count();
    m_ptrA = amg_solver.m_ptrA;
    m_with_timings = amg_solver.m_with_timings;

    if ( m_with_timings )
    {
        cudaEventCreate(&m_setup_start);
        cudaEventCreate(&m_setup_stop);
        cudaEventCreate(&m_solve_start);
        cudaEventCreate(&m_solve_stop);
    }
}

template< class T_Config >
AMG_Solver<T_Config> &AMG_Solver<T_Config>::operator=(const AMG_Solver<T_Config> &amg_solver)
{
    solver = amg_solver.solver;
    m_resources = amg_solver.getResources();
    m_cfg = amg_solver.getConfig();
    m_cfg_self = false;
    solver->incr_ref_count();
    m_ptrA = amg_solver.m_ptrA;
    m_with_timings = amg_solver.m_with_timings;

    if ( m_with_timings )
    {
        cudaEventCreate(&m_setup_start);
        cudaEventCreate(&m_setup_stop);
        cudaEventCreate(&m_solve_start);
        cudaEventCreate(&m_solve_stop);
    }

    return *this;
}

template< class T_Config >
AMG_Solver<T_Config>::~AMG_Solver()
{
    if (m_cfg_self)
    {
        delete m_cfg;
    }

    if ( solver->decr_ref_count() )
    {
        delete solver;

        if ( !m_with_timings )
        {
            return;
        }

        std::cerr << std::endl;
        float elapsed_time = 0.0f;
        cudaEventElapsedTime(&elapsed_time, m_setup_start, m_setup_stop);
        std::cerr << "AMG_Solver::setup time: " << 1.0e-3 * elapsed_time << "s" << std::endl;
        cudaEventElapsedTime(&elapsed_time, m_solve_start, m_solve_stop);
        std::cerr << "AMG_Solver::solve time: " << 1.0e-3 * elapsed_time << "s" << std::endl;
        cudaEventDestroy(m_setup_start);
        cudaEventDestroy(m_setup_stop);
        cudaEventDestroy(m_solve_start);
        cudaEventDestroy(m_solve_stop);
    }
}

template < class T_Config >
const typename AMG_Solver<T_Config>::PODVector_h   &AMG_Solver<T_Config>::get_residual( int res_num ) const
{
    return solver->get_residual(res_num);
}

/****************************************************
* Sets A as the matrix for the AMG system
****************************************************/
template< class T_Config >
AMGX_ERROR AMG_Solver<T_Config>::setup( Matrix<T_Config> &A)//&A0)
{
    bool reuse_fine_matrix = (getStructureReuseLevels() > 0) && A.is_matrix_setup();
    bool reuse_all = (getStructureReuseLevels() == -1) && A.is_matrix_setup();

    if (reuse_all)
    {
        solver->reset_setup_timer();
        return AMGX_OK;
    }

    if ( m_with_timings )
    {
        cudaEventRecord(m_setup_start);
    }

    // postpone free syncs, use device pool
    memory::setAsyncFreeFlag(true);
    AMGX_ERROR e = solver->setup_no_throw(A, reuse_fine_matrix);
    m_resources->get_tmng()->wait_threads();
    amgx::thrust::global_thread_handle::joinDevicePools();
    // reset settings to normal
    memory::setAsyncFreeFlag(false);
    // free postponed objects
    amgx::thrust::global_thread_handle::cudaFreeWait();

    if ( m_with_timings )
    {
        cudaEventRecord(m_setup_stop);
        cudaEventSynchronize(m_setup_stop);
    }

    return e;
}

template< class T_Config >
AMGX_ERROR AMG_Solver<T_Config>::resetup( Matrix<T_Config> &A)//&A0 )
{
    if ( m_with_timings )
    {
        cudaEventRecord(m_setup_start);
    }

    // postpone free syncs, use device pool
    memory::setAsyncFreeFlag(true);
    AMGX_ERROR e = solver->setup_no_throw(A, true);
    m_resources->get_tmng()->wait_threads();
    amgx::thrust::global_thread_handle::joinDevicePools();
    // reset settings to normal
    memory::setAsyncFreeFlag(false);
    // free postponed objects
    amgx::thrust::global_thread_handle::cudaFreeWait();

    if ( m_with_timings )
    {
        cudaEventRecord(m_setup_stop);
        cudaEventSynchronize(m_setup_stop);
    }

    return e;
}

template< class T_Config >
AMGX_ERROR AMG_Solver<T_Config>::setup_capi( std::shared_ptr<Matrix<T_Config>> pA0)
{
    m_ptrA = pA0;
    return setup(*m_ptrA);
}

template< class T_Config >
AMGX_ERROR AMG_Solver<T_Config>::resetup_capi( std::shared_ptr<Matrix<T_Config>> pA0)
{
    m_ptrA = pA0;
    return resetup(*m_ptrA);
}

/****************************************************
* Solves the AMG system Ax=b
***************************************************/
template<class T_Config>
AMGX_ERROR AMG_Solver<T_Config>::solve( Vector<T_Config> &b, Vector<T_Config> &x, AMGX_STATUS &status, bool xIsZero )
{
    if ( m_with_timings )
    {
        cudaEventRecord(m_solve_start);
    }

    AMGX_ERROR e = solver->solve_no_throw( b, x, status, xIsZero );
    amgx::thrust::global_thread_handle::cudaFreeWait();

    if ( m_with_timings )
    {
        cudaEventRecord(m_solve_stop);
        cudaEventSynchronize(m_solve_stop);
    }

    return e;
}

template<class T_Config>
int AMG_Solver<T_Config>::get_num_iters()
{
    return solver->get_num_iters();
}

template< class T_Config >
int AMG_Solver<T_Config>::getStructureReuseLevels()
{
    int lvls = 0;

    if (structure_reuse_levels_scope != "") // AMG is used as main solver
    {
        lvls = m_cfg->template getParameter<int>("structure_reuse_levels", structure_reuse_levels_scope);
    }

    return lvls;
}

/****************************************
 * Explict instantiations
 ***************************************/
#define AMGX_CASE_LINE(CASE) template class AMG_Solver<TemplateMode<CASE>::Type>;
AMGX_FORALL_BUILDS(AMGX_CASE_LINE)
AMGX_FORCOMPLEX_BUILDS(AMGX_CASE_LINE)
#undef AMGX_CASE_LINE

} // namespace amgx
