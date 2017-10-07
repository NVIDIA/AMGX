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

#pragma once

#include <auxdata.h>
#include <matrix.h>
#include "operators/operator.h"

#include <amgx_types/util.h>

namespace amgx
{

enum EIG_WHICH
{
    EIG_PAGERANK,
    EIG_SMALLEST,
    EIG_LARGEST,
    EIG_SHIFT
};

template <class TConfig>
class EigenSolver
{
    public:
        typedef Matrix<TConfig> MMatrix;
        typedef Vector<TConfig> VVector;

        typedef typename TConfig::template setMemSpace<AMGX_host  >::Type TConfig_h;
        typedef typename TConfig::template setMemSpace<AMGX_device>::Type TConfig_d;

        typedef Matrix<TConfig_h> Matrix_h;
        typedef Matrix<TConfig_d> Matrix_d;

        typedef Vector<TConfig_h> Vector_h;
        typedef Vector<TConfig_d> Vector_d;

        typedef typename TConfig::MatPrec ValueTypeMat;
        typedef typename TConfig::VecPrec ValueTypeVec;
        typedef typename TConfig::IndPrec IndType;
        typedef typename types::PODTypes<ValueTypeVec>::type PODValueB;

        typedef typename TConfig_h::template setVecPrec< types::PODTypes< ValueTypeVec >::vec_prec >::Type PODConfig_h;
        typedef typename TConfig_d::template setVecPrec< types::PODTypes< ValueTypeVec >::vec_prec >::Type PODConfig_d;

        typedef Vector<PODConfig_h> PODVector_h;
        typedef Vector<PODConfig_d> PODVector_d;

        EigenSolver(AMG_Config &cfg, const std::string &cfg_scope);
        virtual ~EigenSolver();

        // Get the number of iterations done by the solver.
        int get_num_iters() const;
        // Override the maximum number of iterations.
        void set_max_iters(int max_iters);

        bool is_last_iter() const
        {
            return m_curr_iter == m_max_iters - 1;
        }

        // Override the tolerance
        void set_tolerance(double tol);

        void set_shift(ValueTypeVec shift);

        // Sets the solver name
        inline void setName(std::string &solver_name ) { m_solver_name = solver_name; }

        // Returns the name of the solver
        inline std::string getName() const { return m_solver_name; }

        bool converged() const;

        void setup(Operator<TConfig> &A);

        virtual void solver_setup() = 0;
        virtual void solver_pagerank_setup(VVector &a) = 0;
        AMGX_STATUS solve(VVector &x);

        void exchangeSolveResultsConsolidation(AMGX_STATUS &status);

        AMGX_ERROR solve_no_throw(VVector &x, AMGX_STATUS &status);
        // Initialize the solver before running the iterations.
        virtual void solve_init(VVector &x) {}
        // Run a single iteration of the Eigensolver.
        virtual bool solve_iteration(VVector &x) = 0;
        // Finalize the solver after running the iterations.
        virtual void solve_finalize() {}

        void postprocess_eigenpairs();

        const std::vector<ValueTypeVec> &get_eigenvalues()
        {
            return m_eigenvalues;
        }

        const std::vector<VVector> &get_eigenvectors()
        {
            return m_eigenvectors;
        }

        // Increment the reference counter.
        void incr_ref_count() { ++m_ref_count;}
        // Decrement the reference counter.
        bool decr_ref_count() { return --m_ref_count == 0;}

    private:
        void print_timings();
        void print_iter_stats();
        void print_final_stats();

    protected:
        Operator<TConfig> *m_A;

        bool m_want_eigenvectors;
        double m_tolerance;
        ValueTypeVec m_shift;
        double m_damping_factor;
        bool m_converged;
        int m_curr_iter;
        int m_num_iters;
        int m_max_iters;
        NormType m_norm_type;
        EIG_WHICH m_which;
        // Solver name
        std::string m_solver_name;
        std::string m_eigenvector_solver_name;
        // Store the computed eigenvalues.
        std::vector<ValueTypeVec> m_eigenvalues;
        // Store the computed eigenvectors.
        std::vector<VVector> m_eigenvectors;
        std::vector<PODValueB> m_residuals;

        // Events and time informations.
        cudaEvent_t m_setup_start, m_setup_stop;
        cudaEvent_t m_solve_start, m_solve_stop;
        cudaEvent_t m_iter_start,  m_iter_stop;

        float m_setup_time;
        float m_solve_time;

        int m_ref_count;

        int m_verbosity_level;
};

template <class TConfig>
class EigenSolverFactory
{
    public:
        typedef std::map<std::string, EigenSolverFactory<TConfig> *> EigenSolverFactoryMap;
        virtual EigenSolver<TConfig> *create(AMG_Config &cfg, const std::string &cfg_scope, ThreadManager *tmng = NULL) = 0;
        virtual ~EigenSolverFactory() {};

        /********************************************
         * Register a solver class with key "name"
         *******************************************/
        static void registerFactory(const std::string &name, EigenSolverFactory<TConfig> *f);

        /********************************************
         * Unregister a solver class with key "name"
         *******************************************/
        static void unregisterFactory(const std::string &name);

        /********************************************
         * Unregister all the solver classes
         *******************************************/
        static void unregisterFactories( );

        /*********************************************
         * Allocates solvers based on cfg
         *********************************************/
        static EigenSolver<TConfig> *allocate(AMG_Config &cfg, const std::string &cfg_scope, const std::string &solverType, ThreadManager *tmng = NULL);
        static EigenSolver<TConfig> *allocate(AMG_Config &cfg, const std::string &solverType, ThreadManager *tmng = NULL);

    private:
        static EigenSolverFactoryMap &getFactories();

};

}
