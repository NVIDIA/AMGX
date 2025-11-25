// SPDX-FileCopyrightText: 2013 - 2024 NVIDIA CORPORATION. All Rights Reserved.
//
// SPDX-License-Identifier: BSD-3-Clause

#include <eigensolvers/eigensolver.h>
#include <eigensolvers/eigenvector_solver.h>
#include <norm.h>
#include <amgx_timer.h>
#include <blas.h>
#include <memory_info.h>

#include "amgx_types/util.h"
#include "amgx_types/rand.h"
#include "amgx_types/io.h"

#include <sstream>
#include <iomanip>

namespace amgx
{

template <class TConfig>
EigenSolver<TConfig>::EigenSolver(AMG_Config &cfg, const std::string &cfg_scope)
    : m_A(0), m_converged(false),
      m_curr_iter(0), m_num_iters(0), m_max_iters(0), m_ref_count(1)
{
    m_want_eigenvectors = cfg.getParameter<int>("eig_eigenvector", cfg_scope);
    m_tolerance = cfg.getParameter<double>("eig_tolerance", cfg_scope);
    m_shift = types::util<ValueTypeVec>::get_one() * cfg.getParameter<double>("eig_shift", cfg_scope);
    m_damping_factor = cfg.getParameter<double>("eig_damping_factor", cfg_scope);
    m_max_iters = cfg.getParameter<int>("eig_max_iters", cfg_scope);
    m_verbosity_level = cfg.getParameter<int>("verbosity_level", cfg_scope);
    m_eigenvector_solver_name = cfg.getParameter<std::string>("eig_eigenvector_solver", cfg_scope);
    m_norm_type = cfg.getParameter<NormType>("norm", cfg_scope);
    std::string which = cfg.getParameter<std::string>("eig_which", cfg_scope);

    if (which == "smallest")
    {
        m_which = EIG_SMALLEST;
    }
    else if (which == "largest")
    {
        m_which = EIG_LARGEST;
    }
    else if (which == "pagerank")
    {
        m_which = EIG_PAGERANK;
    }
    else if (which == "shift")
    {
        m_which = EIG_SHIFT;
    }
    else
    {
        FatalError("EigenSolver: invalid target spectrum.", AMGX_ERR_CONFIGURATION);
    }

    // Allocate events.
    cudaEventCreate(&m_setup_start);
    cudaCheckError();
    cudaEventCreate(&m_setup_stop);
    cudaCheckError();
    cudaEventCreate(&m_solve_start);
    cudaCheckError();
    cudaEventCreate(&m_solve_stop);
    cudaCheckError();
    cudaEventCreate(&m_iter_start);
    cudaCheckError();
    cudaEventCreate(&m_iter_stop);
    cudaCheckError();
    m_setup_time = 0.0f;
    m_solve_time = 0.0f;
}

template <class TConfig>
EigenSolver<TConfig>::~EigenSolver()
{
    m_eigenvalues.clear();
    m_eigenvectors.clear();
    cudaEventDestroy(m_setup_start);
    cudaCheckError();
    cudaEventDestroy(m_setup_stop);
    cudaCheckError();
    cudaEventDestroy(m_solve_start);
    cudaCheckError();
    cudaEventDestroy(m_solve_stop);
    cudaCheckError();
    cudaEventDestroy(m_iter_start);
    cudaCheckError();
    cudaEventDestroy(m_iter_stop);
    cudaCheckError();
}

template <class TConfig>
int EigenSolver<TConfig>::get_num_iters() const
{
    return m_num_iters;
}

template <class TConfig>
void EigenSolver<TConfig>::set_max_iters(int max_iters)
{
    m_max_iters = max_iters;
}

template <class TConfig>
void EigenSolver<TConfig>::set_tolerance(double tol)
{
    m_tolerance = tol;
}

template <class TConfig>
void EigenSolver<TConfig>::set_shift(ValueTypeVec shift)
{
    m_shift = shift;
}


template <class TConfig>
bool EigenSolver<TConfig>::converged() const
{
    return m_converged;
}

template <class TConfig>
void EigenSolver<TConfig>::setup(Operator<TConfig> &A)
{
    m_A = &A;
    m_converged = false;
#ifdef AMGX_WITH_MPI
#ifdef MPI_SOLVE_PROFILE
    MPI_Barrier(MPI_COMM_WORLD);
#endif
#endif
    cudaEventRecord(m_setup_start);
    cudaCheckError();
    solver_setup();
#ifdef AMGX_WITH_MPI
#ifdef MPI_SOLVE_PROFILE
    MPI_Barrier(MPI_COMM_WORLD);
#endif
#endif
    cudaEventRecord(m_setup_stop);
    cudaCheckError();
    cudaEventSynchronize(m_setup_stop);
    cudaCheckError();
    cudaEventElapsedTime(&m_setup_time, m_setup_start, m_setup_stop);
    m_setup_time *= 1e-3f;
}

template<class TConfig>
void EigenSolver<TConfig>::exchangeSolveResultsConsolidation(AMGX_STATUS &status)
{
    std::vector<PODVector_h> m_res_history;
    PODVector_h res(1);

    for (int i = 0; i < m_residuals.size(); i++)
    {
        res[0] = m_residuals[i];
        m_res_history.push_back(res);
    }

    this->m_A->getManager()->exchangeSolveResultsConsolidation(m_num_iters, m_res_history, status, true /*looks like we always store residual history*/);
}

template<class TConfig>
AMGX_ERROR EigenSolver<TConfig>::solve_no_throw(VVector &x, AMGX_STATUS &status)
{
    AMGX_ERROR rc = AMGX_OK;

    AMGX_TRIES()
    {
        // Check if fine level is consolidated and not a root partition
        if ( !(this->m_A->getManager() != NULL && this->m_A->getManager()->isFineLevelConsolidated() && !this->m_A->getManager()->isFineLevelRootPartition() ))
        {
            // If matrix is consolidated on fine level and not a root partition
            if (x.tag == -1)
            {
                x.tag = 4242 * 100 + 1;
            }

            status = this->solve(x);
        }

        // Exchange residual history, number of iterations, solve status if fine level consoildation was used
        if (this->m_A->getManager() != NULL && this->m_A->getManager()->isFineLevelConsolidated())
        {
            this->exchangeSolveResultsConsolidation(status);
        }
    }

    AMGX_CATCHES(rc)
    return rc;
}

template <class TConfig>
AMGX_STATUS EigenSolver<TConfig>::solve(VVector &x)
{
    // initial vector is empty, initialize it with random values.
    if (x.empty())
    {
        Operator<TConfig> &A = *this->m_A;
        ViewType oldView = A.currentView();
        A.setViewExterior();
        int N = A.get_num_cols();
        Vector_h h_x(N);

        for (int i = 0; i < N; ++i)
        {
            h_x[i] = types::get_rand<ValueTypeVec>();
        }

        x = h_x;
        A.setView(oldView);
    }

    // This code is needed for MPI implementation of eigensolvers.
    x.set_block_dimx(1);
    x.set_block_dimy(m_A->get_block_dimx());

    if (x.tag == -1)
    {
        x.tag = 1;
    }

    x.dirtybit = 1;
    x.delayed_send = 1;
    m_eigenvectors.clear();
    m_eigenvalues.clear();
#ifdef AMGX_WITH_MPI
#ifdef MPI_SOLVE_PROFILE
    MPI_Barrier(MPI_COMM_WORLD);
#endif
#endif
    cudaEventRecord(m_solve_start);
    cudaCheckError();
    solve_init(x);
    bool done = false;

    for (m_curr_iter = 0; m_curr_iter < m_max_iters && !done; ++m_curr_iter)
    {
        done = solve_iteration(x);

        // solve_iteration did not update the residuals, add an undefined norm.
        if (m_residuals.size() == m_curr_iter)
        {
            m_residuals.push_back( types::util<PODValueB>::get_minus_one());
        }

        if (m_verbosity_level == 3)
        {
            print_iter_stats();
        }
    }

    print_final_stats();

    if (done)
    {
        m_converged = true;
    }

    m_num_iters = m_curr_iter;
    solve_finalize();

    if (m_want_eigenvectors && m_eigenvectors.empty())
    {
        std::string str = "Eigenvectors requested but not provided by solver.\n";
        amgx_output(str.c_str(), str.length());

        if (m_eigenvector_solver_name.empty())
        {
            FatalError("Eigenvectors requested but no eigenvector solver provided", AMGX_ERR_CONFIGURATION);
        }

        EigenVectorSolver<TConfig> *eigenvector_solver = EigenVectorSolverFactory<TConfig>::create(m_eigenvector_solver_name);
        ValueTypeVec eigenvalue = m_eigenvalues.front();
        eigenvector_solver->setup(*m_A);
        m_eigenvectors.resize(m_eigenvalues.size());
        eigenvector_solver->solve(eigenvalue, m_eigenvectors[0]);
        delete eigenvector_solver;
    }

#ifdef AMGX_WITH_MPI
#ifdef MPI_SOLVE_PROFILE
    MPI_Barrier(MPI_COMM_WORLD);
#endif
#endif
    cudaEventRecord(m_solve_stop);
    cudaCheckError();
    cudaEventSynchronize(m_solve_stop);
    cudaCheckError();
    cudaEventElapsedTime(&m_solve_time, m_solve_start, m_solve_stop);
    m_solve_time *= 1e-3f;

    if (m_verbosity_level == 3)
    {
        std::stringstream ss;

        if (m_converged)
        {
            ss << "Eigensolver converged after " << get_num_iters()
               << " iterations." << std::endl;
            std::vector<ValueTypeVec> eigenvalues = get_eigenvalues();
            ss << "Eigenvalue: ";

            for (int i = 0; i < eigenvalues.size(); ++i)
            {
                ss << eigenvalues[i] << " ";
            }

            ss << std::endl;
        }
        else
        {
            ss << "Eigensolver did not converge after "
               << this->get_num_iters()
               << " iterations." << std::endl;
        }

        amgx_output(ss.str().c_str(), static_cast<int>(ss.str().length()));
        print_timings();
    }

    return m_converged ? AMGX_ST_CONVERGED : AMGX_ST_NOT_CONVERGED;
}

template<class TConfig>
void EigenSolver<TConfig>::postprocess_eigenpairs()
{
    // If the smallest eigenvalues were computed (with A^-1),
    // we need to invert the eigenvalue.
    if (m_which == EIG_SMALLEST)
    {
        for (int i = 0; i < m_eigenvalues.size(); ++i)
        {
            ValueTypeVec eigenvalue = m_eigenvalues[i];
            ValueTypeVec inv_eigenvalue = types::util<ValueTypeVec>::get_one() / eigenvalue;
            m_eigenvalues[i] = inv_eigenvalue + m_shift;
        }
    }
}

template<class TConfig>
void EigenSolver<TConfig>::print_timings()
{
    std::stringstream ss;
    ss << "Total Time: " << m_setup_time + m_solve_time << std::endl;
    ss << "    setup: " << m_setup_time << " s\n";
    ss << "    solve: " << m_solve_time << " s\n";
    ss << "    solve(per iteration): " << ((m_num_iters == 0) ? m_num_iters : m_solve_time / m_num_iters) << " s\n";
    amgx_output(ss.str().c_str(), static_cast<int>(ss.str().length()));
}

template <typename TConfig>
void EigenSolver<TConfig>::print_iter_stats()
{
    if (m_curr_iter == 0)
    {
        std::stringstream ss;
        ss << std::setw(15) << "iter" << std::setw(20) << " Mem Usage (GB)"
           << std::setw(15) << "residual";
        ss << std::setw(15) << "rate";
        ss << std::endl;
        ss
                << "         --------------------------------------------------------------";
        ss << std::endl;
        amgx_output(ss.str().c_str(), static_cast<int>(ss.str().length()));
    }

    std::stringstream ss;
    ss << std::setw(15) << m_curr_iter;
    MemoryInfo::updateMaxMemoryUsage();
    ss << std::setw(20) << MemoryInfo::getMaxMemoryUsage();
    PODValueB iter_residual = m_residuals[m_curr_iter];

    if (iter_residual >= 0)
    {
        ss << std::scientific << std::setprecision(6) << std::setw(15) << iter_residual;

        // Compute convergence rate.
        if (m_curr_iter > 0)
        {
            PODValueB prev_residual = m_residuals[m_curr_iter - 1];

            if (prev_residual > 0)
            {
                ss << std::setw(15);
                ss << std::fixed << std::setprecision(4) << iter_residual / prev_residual;
            }
        }
    }

    ss << std::endl;
    amgx_output(ss.str().c_str(), static_cast<int>(ss.str().length()));
}

template <typename TConfig>
void EigenSolver<TConfig>::print_final_stats()
{
    std::stringstream ss;
    ss << "         --------------------------------------------------------------";
    ss << std::endl;
    amgx_output(ss.str().c_str(), static_cast<int>(ss.str().length()));
}

template<class TConfig>
typename EigenSolverFactory<TConfig>::EigenSolverFactoryMap &
EigenSolverFactory<TConfig>::getFactories()
{
    static EigenSolverFactoryMap factories;
    return factories;
}


template<class TConfig>
void EigenSolverFactory<TConfig>::registerFactory(const std::string &name,
        EigenSolverFactory<TConfig> *f)
{
    EigenSolverFactoryMap &factories = getFactories();
    typename EigenSolverFactoryMap::const_iterator it = factories.find(name);

    if (it != factories.end())
    {
        std::string error = "EigenSolverFactory '" + name + "' has already been registered\n";
        FatalError(error.c_str(), AMGX_ERR_CORE);
    }

    factories[name] = f;
}

template<class TConfig>
void EigenSolverFactory<TConfig>::unregisterFactory(const std::string &name)
{
    EigenSolverFactoryMap &factories = getFactories();
    typename EigenSolverFactoryMap::iterator it = factories.find(name);

    if (it == factories.end())
    {
        std::string error = "EigenSolverFactory '" + name + "' has not been registered\n";
        FatalError(error.c_str(), AMGX_ERR_CORE);
    }

    EigenSolverFactory<TConfig> *factory = it->second;
    assert(factory != NULL);
    delete factory;
    factories.erase(it);
}

template<class TConfig>
void EigenSolverFactory<TConfig>::unregisterFactories()
{
    EigenSolverFactoryMap &factories = getFactories();
    typename EigenSolverFactoryMap::iterator it = factories.begin();

    for (; it != factories.end();)
    {
        EigenSolverFactory<TConfig> *factory = it->second;
        assert(factory != NULL);
        it++;
        delete factory;
    }

    factories.clear();
}

template<class TConfig>
EigenSolver<TConfig> *EigenSolverFactory<TConfig>::allocate(AMG_Config &cfg,
        const std::string &current_scope, const std::string &solverType,
        ThreadManager *tmng)
{
    std::string solverName, new_scope;
    cfg.getParameter<std::string>(solverType, solverName, current_scope,
                                  new_scope);
    EigenSolverFactoryMap &factories = getFactories();
    typename EigenSolverFactoryMap::const_iterator it = factories.find(solverName);

    if (it == factories.end())
    {
        std::string error = "EigenSolverFactory '" + solverName
                            + "' has not been registered\n";
        FatalError(error.c_str( ), AMGX_ERR_CORE);
    }

    EigenSolver<TConfig> *solver = it->second->create(cfg, new_scope, tmng);
    solver->setName(solverName);
    return solver;
}
;

template<class TConfig>
EigenSolver<TConfig> *EigenSolverFactory<TConfig>::allocate(AMG_Config &cfg,
        const std::string &solverType, ThreadManager *tmng)
{
    return EigenSolverFactory<TConfig>::allocate(cfg, "default", solverType, tmng);
}

// Explicit template instantiation.
#define AMGX_CASE_LINE(CASE) template class EigenSolver<TemplateMode<CASE>::Type>;
AMGX_FORALL_BUILDS(AMGX_CASE_LINE)
AMGX_FORCOMPLEX_BUILDS(AMGX_CASE_LINE)
#undef AMGX_CASE_LINE

#define AMGX_CASE_LINE(CASE) template class EigenSolverFactory<TemplateMode<CASE>::Type>;
AMGX_FORALL_BUILDS(AMGX_CASE_LINE)
AMGX_FORCOMPLEX_BUILDS(AMGX_CASE_LINE)
#undef AMGX_CASE_LINE

};
