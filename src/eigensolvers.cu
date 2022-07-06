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
        registerClasses<TConfigGeneric>();
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
    unregisterClasses<TConfigGeneric>();

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
