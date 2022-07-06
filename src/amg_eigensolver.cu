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

#include <amg_eigensolver.h>
#include <amg_config.h>
#include <eigensolvers/eigensolver.h>
#include <basic_types.h>
#include <misc.h>
#include <assert.h>
#include <logger.h>
#include <util.h>
#include <profile.h>

using std::string;

namespace amgx
{

template< class T_Config >
void AMG_EigenSolver<T_Config>::process_config(AMG_Config &in_cfg, std::string solver_scope)
{
    if (in_cfg.getParameter<int>("print_config", solver_scope) == 1)
    {
        in_cfg.printAMGConfig();
    }
}

template< class T_Config >
void AMG_EigenSolver<T_Config>::init()
{
    std::string solver_value, solver_scope;
    m_cfg->template getParameter<std::string>("eig_solver", solver_value, "default", solver_scope);
    process_config(*m_cfg, solver_scope);
    // pass thread manager to solver
    solver = EigenSolverFactory<T_Config>::allocate(*m_cfg, "default", "eig_solver", m_resources->get_tmng());

    if ( m_with_timings )
    {
        cudaEventCreate(&m_setup_start);
        cudaEventCreate(&m_setup_stop);
        cudaEventCreate(&m_solve_start);
        cudaEventCreate(&m_solve_stop);
    }
}

template< class T_Config >
AMG_EigenSolver<T_Config>::AMG_EigenSolver(Resources *res, AMG_Configuration *cfg) : m_with_timings(false), m_resources(res), m_cfg_self(false)
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
AMG_EigenSolver<T_Config>::AMG_EigenSolver(Resources *res, AMG_Configuration &cfg) : m_with_timings(false), m_resources(res), m_cfg_self(true)
{
    m_cfg = new AMG_Config;
    *m_cfg = *(cfg.getConfigObject());
    init();
}

template< class T_Config >
AMG_EigenSolver<T_Config>::AMG_EigenSolver(const AMG_EigenSolver<T_Config> &other)
{
    solver = other.solver;
    m_resources = other.getResources();
    m_cfg = other.getConfig();
    m_cfg_self = false;
    solver->incr_ref_count();
    m_ptrA = other.m_ptrA;
    m_with_timings = other.m_with_timings;

    if ( m_with_timings )
    {
        cudaEventCreate(&m_setup_start);
        cudaEventCreate(&m_setup_stop);
        cudaEventCreate(&m_solve_start);
        cudaEventCreate(&m_solve_stop);
    }
}

template< class T_Config >
AMG_EigenSolver<T_Config> &AMG_EigenSolver<T_Config>::operator=(const AMG_EigenSolver<T_Config> &other)
{
    solver = other.solver;
    m_resources = other.getResources();
    m_cfg = other.getConfig();
    m_cfg_self = false;
    solver->incr_ref_count();
    m_ptrA = other.m_ptrA;
    m_with_timings = other.m_with_timings;

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
AMG_EigenSolver<T_Config>::~AMG_EigenSolver()
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
        std::cerr << "AMG_EigenSolver::setup time: " << 1.0e-3 * elapsed_time << "s" << std::endl;
        cudaEventElapsedTime(&elapsed_time, m_solve_start, m_solve_stop);
        std::cerr << "AMG_EigenSolver::solve time: " << 1.0e-3 * elapsed_time << "s" << std::endl;
        cudaEventDestroy(m_setup_start);
        cudaEventDestroy(m_setup_stop);
        cudaEventDestroy(m_solve_start);
        cudaEventDestroy(m_solve_stop);
    }
}


/****************************************************
* Sets A as the matrix for the AMG system
****************************************************/
template< class T_Config >
void AMG_EigenSolver<T_Config>::setup( Matrix<T_Config> &A)//&A0)
{
    if ( m_with_timings )
    {
        cudaEventRecord(m_setup_start);
    }

    // postpone free syncs, use device pool
    memory::setAsyncFreeFlag(true);
    solver->setup(A);
    m_resources->get_tmng()->wait_threads();
    thrust::global_thread_handle::joinDevicePools();
    // reset settings to normal
    memory::setAsyncFreeFlag(false);
    // free postponed objects
    thrust::global_thread_handle::cudaFreeWait();

    if ( m_with_timings )
    {
        cudaEventRecord(m_setup_stop);
        cudaEventSynchronize(m_setup_stop);
    }
}

template< class T_Config >
AMGX_ERROR AMG_EigenSolver<T_Config>::setup_no_throw( Matrix<T_Config> &A)
{
    AMGX_ERROR rc = AMGX_OK;

    try
    {
        this->setup(A);
    }

    AMGX_CATCHES(rc)
    return rc;
}

template< class T_Config >
void AMG_EigenSolver<T_Config>::pagerank_setup( Vector<T_Config> &vec)//&A0)
{
    if ( m_with_timings )
    {
        cudaEventRecord(m_setup_start);
    }

    // postpone free syncs, use device pool
    memory::setAsyncFreeFlag(true);
    solver->solver_pagerank_setup(vec);
    m_resources->get_tmng()->wait_threads();
    thrust::global_thread_handle::joinDevicePools();
    // reset settings to normal
    memory::setAsyncFreeFlag(false);
    // free postponed objects
    thrust::global_thread_handle::cudaFreeWait();

    if ( m_with_timings )
    {
        cudaEventRecord(m_setup_stop);
        cudaEventSynchronize(m_setup_stop);
    }
}

template< class T_Config >
AMGX_ERROR AMG_EigenSolver<T_Config>::pagerank_setup_no_throw( Vector<T_Config> &vec)
{
    AMGX_ERROR rc = AMGX_OK;

    try
    {
        this->pagerank_setup(vec);
    }

    AMGX_CATCHES(rc)
    return rc;
}

template< class T_Config >
void AMG_EigenSolver<T_Config>::setup_capi( std::shared_ptr<Matrix<T_Config>> pA0)
{
    m_ptrA = pA0;
    setup(*m_ptrA);
}


template< class T_Config >
AMGX_ERROR AMG_EigenSolver<T_Config>::setup_capi_no_throw( std::shared_ptr<Matrix<T_Config>> pA0)
{
    AMGX_ERROR rc = AMGX_OK;

    try
    {
        this->setup_capi(pA0);
    }

    AMGX_CATCHES(rc)
    return rc;
}

/****************************************************
* Solves the AMG system Ax=b
***************************************************/
template<class T_Config>
AMGX_ERROR AMG_EigenSolver<T_Config>::solve_no_throw( Vector<T_Config> &x, AMGX_STATUS &status )
{
    if ( m_with_timings )
    {
        cudaEventRecord(m_solve_start);
    }

    AMGX_ERROR e = solver->solve_no_throw( x, status );
    thrust::global_thread_handle::cudaFreeWait();

    if ( m_with_timings )
    {
        cudaEventRecord(m_solve_stop);
        cudaEventSynchronize(m_solve_stop);
    }

    return e;
}

template<class T_Config>
int AMG_EigenSolver<T_Config>::get_num_iters()
{
    return solver->get_num_iters();
}

/****************************************
 * Explict instantiations
 ***************************************/

template class AMG_EigenSolver<TConfigGeneric_d>;
template class AMG_EigenSolver<TConfigGeneric_h>;

} // namespace amgx
