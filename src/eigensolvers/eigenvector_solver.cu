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
template class EigenVectorSolver<TConfigGeneric_d>;
template class EigenVectorSolver<TConfigGeneric_h>;

template class EigenVectorSolverFactory<TConfigGeneric_d>;
template class EigenVectorSolverFactory<TConfigGeneric_h>;


}
