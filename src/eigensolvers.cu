// SPDX-FileCopyrightText: 2013 - 2024 NVIDIA CORPORATION. All Rights Reserved.
//
// SPDX-License-Identifier: BSD-3-Clause

#include <amg_config.h>
#include <basic_types.h>
#include <eigensolvers/single_iteration_eigensolver.h>
#include <eigensolvers/lanczos_eigensolver.h>
#include <eigensolvers/arnoldi_eigensolver.h>
#include <eigensolvers/subspace_iteration_eigensolver.h>
#include <eigensolvers/lobpcg_eigensolver.h>
#include <eigensolvers/jacobi_davidson_eigensolver.h>


namespace amgx
{
namespace eigensolvers
{

inline void registerParameters()
{
    AMG_Config::registerParameter<std::string>("eig_solver", "the eigensolver algorithm to use <POWER_ITERATION>", "POWER_ITERATION");
    AMG_Config::registerParameter<double>("eig_tolerance", "eigensolver: the residual tolerance for eigenpairs residuals", 1e-4);
    AMG_Config::registerParameter<double>("eig_shift", "eigensolver: the shift lambda, the eigensolver will be applied on A - lambda * I", 0.);
    AMG_Config::registerParameter<double>("eig_damping_factor", "eigensolver: the damping factor for PageRank", 0.8);
    AMG_Config::registerParameter<int>("eig_max_iters", "eigensolver: the maximum number of iterations", 100);
    AMG_Config::registerParameter<std::string>("eig_which", "eigensolver: which eigenpairs to compute", "largest");
    AMG_Config::registerParameter<int>("eig_wanted_count", "eigensolver: the number of sought eigenvalues", 1);
    AMG_Config::registerParameter<int>("eig_subspace_size", "eigensolver: size of the subspace to use for block methods", -1);
    AMG_Config::registerParameter<int>("eig_convergence_check_freq", "eigensolver: frequencies of convergence checks", 1);
    AMG_Config::registerParameter<int>("eig_eigenvector", "eigensolver: flag enabling computation of the eigenvectors (default: 0)", 0);
    AMG_Config::registerParameter<std::string>("eig_eigenvector_solver", "eigensolver: which solver to use to compute eigenvectors.", "");
}

template <typename TConfig>
inline void registerClasses()
{
    EigenSolverFactory<TConfig>::registerFactory("SINGLE_ITERATION", new SingleIteration_EigenSolverFactory<TConfig>);
    // Pagerank is an alias to single iteration.
    EigenSolverFactory<TConfig>::registerFactory("PAGERANK", new SingleIteration_EigenSolverFactory<TConfig>);
    // Power iteration and inverse iteration are aliases to single iteration.
    EigenSolverFactory<TConfig>::registerFactory("POWER_ITERATION", new SingleIteration_EigenSolverFactory<TConfig>);
    EigenSolverFactory<TConfig>::registerFactory("INVERSE_ITERATION", new SingleIteration_EigenSolverFactory<TConfig>);
    EigenSolverFactory<TConfig>::registerFactory("SUBSPACE_ITERATION", new SubspaceIteration_EigenSolverFactory<TConfig>);
    EigenSolverFactory<TConfig>::registerFactory("LANCZOS", new Lanczos_EigenSolverFactory<TConfig>);
    EigenSolverFactory<TConfig>::registerFactory("ARNOLDI", new Arnoldi_EigenSolverFactory<TConfig>);
    EigenSolverFactory<TConfig>::registerFactory("LOBPCG", new LOBPCG_EigenSolverFactory<TConfig>);
    EigenSolverFactory<TConfig>::registerFactory("JACOBI_DAVIDSON", new JacobiDavidson_EigenSolverFactory<TConfig>);
}

template <typename TConfig>
inline void unregisterClasses()
{
    EigenSolverFactory<TConfig>::unregisterFactory("SINGLE_ITERATION");
    EigenSolverFactory<TConfig>::unregisterFactory("PAGERANK");
    EigenSolverFactory<TConfig>::unregisterFactory("POWER_ITERATION");
    EigenSolverFactory<TConfig>::unregisterFactory("INVERSE_ITERATION");
    EigenSolverFactory<TConfig>::unregisterFactory("SUBSPACE_ITERATION");
    EigenSolverFactory<TConfig>::unregisterFactory("LANCZOS");
    EigenSolverFactory<TConfig>::unregisterFactory("ARNOLDI");
    EigenSolverFactory<TConfig>::unregisterFactory("LOBPCG");
    EigenSolverFactory<TConfig>::unregisterFactory("JACOBI_DAVIDSON");
}

AMGX_ERROR initialize()
{
    //Call registration of classes and parameters here
    try
    {
#define AMGX_CASE_LINE(CASE) registerClasses<TemplateMode<CASE>::Type>();
        AMGX_FORALL_BUILDS(AMGX_CASE_LINE)
#undef AMGX_CASE_LINE
        registerParameters();
    }
    catch (amgx_exception e)
    {
        std::string buf = "Error initializing Eigensolver plugin: ";
        amgx_output(buf.c_str(), buf.length());
        amgx_output(e.what(), strlen(e.what()) );
        return AMGX_ERR_PLUGIN;
    }

    return AMGX_OK;
}

void finalize()
{
#define AMGX_CASE_LINE(CASE) unregisterClasses<TemplateMode<CASE>::Type>();
    AMGX_FORALL_BUILDS(AMGX_CASE_LINE)
#undef AMGX_CASE_LINE
    AMG_Config::unregisterParameter("eig_solver");
    AMG_Config::unregisterParameter("eig_tolerance");
    AMG_Config::unregisterParameter("eig_shift");
    AMG_Config::unregisterParameter("eig_damping_factor");
    AMG_Config::unregisterParameter("eig_max_iters");
    AMG_Config::unregisterParameter("eig_which");
    AMG_Config::unregisterParameter("eig_wanted_count");
    AMG_Config::unregisterParameter("eig_subspace_size");
    AMG_Config::unregisterParameter("eig_convergence_check_freq");
    AMG_Config::unregisterParameter("eig_eigenvector");
    AMG_Config::unregisterParameter("eig_eigenvector_solver");
}
} // namespace eigensolvers
} // namespace amgx
