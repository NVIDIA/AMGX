// SPDX-FileCopyrightText: 2013 - 2025 NVIDIA CORPORATION. All Rights Reserved.
//
// SPDX-License-Identifier: BSD-3-Clause

#include <eigensolvers/eigenvector_solver.h>

namespace amgx
{

template <class TConfig>
EigenVectorSolver<TConfig>::EigenVectorSolver(AMG_Config &cfg, const std::string &cfg_scope)
    : m_cfg(cfg)
{
    m_solver = EigenSolverFactory<TConfig>::allocate(m_cfg, "default", "eig_solver");
}

template <class TConfig>
EigenVectorSolver<TConfig>::~EigenVectorSolver()
{
    delete m_solver;
}

template <class TConfig>
void EigenVectorSolver<TConfig>::setup(Operator<TConfig> &A)
{
    m_A = &A;
}

template <class TConfig>
AMGX_STATUS EigenVectorSolver<TConfig>::solve(ValueTypeVec eigenvalue, VVector &eigenvector)
{
    Operator<TConfig> &A = *m_A;
    m_solver->set_shift(eigenvalue);
    m_solver->setup(A);
    // Pass an empty vector to the solver, it will be initialized automatically.
    VVector x;
    AMGX_STATUS status = m_solver->solve(x);
    const std::vector<VVector> &eigenvectors = m_solver->get_eigenvectors();

    if (eigenvectors.empty())
    {
        FatalError("Eigenvector solver did not succeed in computing eigenvectors", AMGX_ERR_INTERNAL);
    }

    eigenvector = eigenvectors.front();
    return status;
}

template <class TConfig>
EigenVectorSolver<TConfig> *EigenVectorSolverFactory<TConfig>::create(std::string &name)
{
    if (name == "default" || name == "inverse_iteration_fgmres")
    {
        return create_inverse_iteration();
    }
    else
    {
        FatalError("Unknown eigenvector solver requested", AMGX_ERR_CONFIGURATION);
    }
}

template <class TConfig>
EigenVectorSolver<TConfig> *EigenVectorSolverFactory<TConfig>::create_inverse_iteration()
{
    AMG_Config cfg;
    cfg.parseParameterString("config_version=2;"
                             "eig_solver=INVERSE_ITERATION;"
                             "eig_max_iters=100;"
                             "eig_tolerance=1e-4;"
                             "eig_which=smallest;"
                             //gmres accelerator config
                             "solver(main)=FGMRES;"
                             "main:gmres_n_restart=10;"
                             "main:preconditioner(amg)=AMG;"
                             //outer solver setup
                             "main:convergence=RELATIVE_INI;"
                             "main:norm=L2;"
                             "main:use_scalar_norm=1;"
                             "main:max_iters=100;"
                             "main:tolerance=1e-4;"
                             //amg specific
                             "amg:max_iters=1;"
                             "amg:algorithm=AGGREGATION;"
                             "amg:selector=SIZE_2;"
                             "amg:cycle=V;"
                             "amg:smoother=MULTICOLOR_DILU;"
                             "amg:presweeps=0;"
                             "amg:postsweeps=3;"
                             "amg:error_scaling=0;"
                             "amg:max_levels=100;"
                             "amg:coarseAgenerator=LOW_DEG;"
                             "amg:matrix_coloring_scheme=PARALLEL_GREEDY;"
                             "amg:max_uncolored_percentage=0.05;"
                             "amg:relaxation_factor=0.75;"
                             "amg:coarse_solver=DENSE_LU_SOLVER;"
                             "amg:min_coarse_rows=32;"
                             //printing obtions
                             "main:print_solve_stats=1;"
                             "main:monitor_residual=1;"
                             "main:obtain_timings=1;"
                             "amg:print_grid_stats=1;");
    return new EigenVectorSolver<TConfig>(cfg, "default");
}

// Explicit template instantiation.
#define AMGX_CASE_LINE(CASE) template class EigenVectorSolver<TemplateMode<CASE>::Type>;
AMGX_FORALL_BUILDS(AMGX_CASE_LINE)
AMGX_FORCOMPLEX_BUILDS(AMGX_CASE_LINE)
#undef AMGX_CASE_LINE

#define AMGX_CASE_LINE(CASE) template class EigenVectorSolverFactory<TemplateMode<CASE>::Type>;
AMGX_FORALL_BUILDS(AMGX_CASE_LINE)
AMGX_FORCOMPLEX_BUILDS(AMGX_CASE_LINE)
#undef AMGX_CASE_LINE


}
