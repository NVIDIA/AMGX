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

#pragma once

namespace amgx
{
template<class Matrix> class Solver;
template<class T_config> class Scaler;
}

#include <convergence/convergence.h>
#include <thread_manager.h>

#include <amgx_types/util.h>

namespace amgx
{

template<class TConfig>
class Solver : public AuxData
{
    public:
        typedef Operator<TConfig> TOperator;
        typedef Operator<TConfig> MMatrix;
        typedef Vector<TConfig> VVector;

        typedef typename TConfig::MatPrec ValueTypeA;
        typedef typename TConfig::VecPrec ValueTypeB;

        typedef typename types::PODTypes<ValueTypeB>::type PODValueB;

        typedef typename TConfig::template setMemSpace<AMGX_host  >::Type TConfig_h;
        typedef typename TConfig::template setMemSpace<AMGX_device>::Type TConfig_d;

        typedef Matrix<TConfig_h> Matrix_h;
        typedef Matrix<TConfig_d> Matrix_d;

        typedef Vector<TConfig_h> Vector_h;
        typedef Vector<TConfig_d> Vector_d;

        typedef typename TConfig_h::template setVecPrec< types::PODTypes< ValueTypeB >::vec_prec >::Type PODConfig_h;
        typedef typename TConfig_d::template setVecPrec< types::PODTypes< ValueTypeB >::vec_prec >::Type PODConfig_d;

        typedef Vector<PODConfig_h> PODVector_h;
        typedef Vector<PODConfig_d> PODVector_d;

        // Constructor.
        Solver( AMG_Config &cfg, const std::string &cfg_scope, ThreadManager *tmng = NULL );
        virtual ~Solver();

        inline Operator<TConfig> &get_A()
        {
            if (m_A != NULL)
            {
                return *m_A;
            }

            FatalError("Matrix was not initialized\n", AMGX_ERR_BAD_PARAMETERS);
        }
        inline Vector<TConfig> &get_R() { return *m_r; }


        // Thread manager for multi-threading
        inline ThreadManager *get_thread_manager() { return m_tmng; }

        // Get the number of iterations done by the solver.
        inline int get_num_iters() { return m_num_iters; }
        // Get the residual obtained at iteration i.
        const PODVector_h &get_residual( int i ) const;
        // Override the maximum number of iterations.
        void set_max_iters(int max_iters);

        // Override the tolerance
        void setTolerance(double tol);

        // Is it the last iteration?
        inline bool is_last_iter() const { return m_curr_iter == m_max_iters - 1; }

        // reset timers
        void reset_setup_timer();
        void reset_solve_timer();

        // Compute the residual and set m_r.
        void compute_residual( const VVector &b, VVector &x );
        // Compute the norm m_nrm.
        void compute_norm();
        // Set the norm m_nrm.
        void set_norm( const PODVector_h &nrm );
        // Compute residual and decide convergence.
        bool converged( const VVector &b, VVector &x );
        // Decide convergence based m_nrm.
        bool converged() const;

        // Compute the norm and decide convergence.
        inline bool compute_norm_and_converged()
        {
            compute_norm();
            return converged();
        }

        // Compute a residual r = b - Ax.
        void compute_residual( const VVector &b, VVector &x, VVector &r ) const;
        // Compute the norm of v.
        void compute_norm( const VVector &v, PODVector_h &nrm ) const;
        // Compute the norm of v.
        void compute_residual_norm_external( Operator<TConfig> &mtx, const VVector &b, const VVector &x, typename PODVector_h::value_type *nrm ) const;
        // Decide convergence based on nrm.
        bool converged( PODVector_h &nrm ) const;

        // Compute the norm and decide convergence.
        inline bool compute_norm_and_converged( const VVector &v, PODVector_h &nrm ) const
        {
            compute_norm( v, nrm );
            return converged( nrm );
        }

        void exchangeSolveResultsConsolidation(AMGX_STATUS &status);

        ////////////////////////////////////////////////////////////////////////////////////////
        //// Interface for coarse solvers.
        ////////////////////////////////////////////////////////////////////////////////////////

        AMGX_ERROR try_solve( VVector &b, VVector &x, AMGX_STATUS &status, bool xIsZero = false );

        // Setup.
        void setup( Operator<TConfig> &A, bool reuse_matrix_structure );
        // Setup. It calls setup and  catches exceptions.
        AMGX_ERROR setup_no_throw( Operator<TConfig> &A, bool reuse_matrix_structure );
        // Solver specific setup
        virtual void solver_setup(bool reuse_matrix_structure) = 0;
        // Solve
        AMGX_STATUS solve( Vector<TConfig> &b, Vector<TConfig> &x, bool xIsZero ) ;
        // Solve. It calls solve and catches exceptions.
        AMGX_ERROR solve_no_throw( VVector &b, VVector &x, AMGX_STATUS &status, bool xIsZero = false );

        // What coloring level is needed for the matrix?
        virtual bool isColoringNeeded() const = 0;

        // What reordering level is needed for the matrix?
        virtual bool getReorderColsByColorDesired() const = 0;
        virtual bool getInsertDiagonalDesired() const = 0;

        virtual void getColoringScope( std::string &cfg_scope_for_coloring)  const { cfg_scope_for_coloring = "default"; }

        // Does the solver requires the residual vector storage
        virtual bool is_residual_needed( ) const { return true; }

        // Initialize the solver before running the iterations.
        virtual void solve_init( VVector &b, VVector &x, bool xIsZero ) {}
        // Run a single iteration. Compute the residual and its norm and decide convergence.
        virtual bool solve_iteration( VVector &b, VVector &x, bool xIsZero ) = 0;
        // Finalize the solver after running the iterations.
        virtual void solve_finalize( VVector &b, VVector &x ) {}

        // Solve
        virtual void smooth( Vector<TConfig> &b, Vector<TConfig> &x, bool xIsZero ) { solve(b, x, xIsZero); }

        // Print the timings.
        void print_timings();
        // Print grid stats.
        virtual void print_grid_stats() {}
        virtual void print_grid_stats2() {}
        // Print visualization data.
        virtual void print_vis_data() {}
        // Print the solver settings
        virtual void printSolverParameters() const {}

        // Sets the solver name
        inline void setName(std::string &solver_name ) { m_solver_name = solver_name; }

        // Returns the name of the solver
        inline std::string getName() const { return m_solver_name; }

        // Returns the name m_cfg_scope
        inline std::string getScope() const { return m_cfg_scope; }

        // Increment the reference counter.
        void incr_ref_count() { ++m_ref_count; }
        // Decrement the reference counter.
        bool decr_ref_count() { return --m_ref_count == 0; }

        int tag;

        int level;
    protected:
        // Define the matrix and the residual.
        inline void set_A(Operator<TConfig> &in_A)
        {
            m_A = &in_A;
        }
        inline void set_R(Vector<TConfig> &in_r) { m_r = &in_r; }

        bool getPrintGridStats();
        bool getPrintSolveStats();

        // Equation scaler  NONE | BINORMALIZATION | DIAGONAL_SYMMETRIC
        std::string m_scaling;

        // Print a norm.
        void print_norm( std::stringstream &ss ) const;
        void print_norm2( std::stringstream &ss ) const;

        // Reference counter.
        int m_ref_count;

        // Configuration information.
        AMG_Config *m_cfg;
        const std::string m_cfg_scope;

        // The matrix.
        Operator<TConfig> *m_A;

        // Scaler to re-scale equations
        Scaler<TConfig> *m_Scaler;

        // Iteration counter and limits.
        int m_num_iters, m_curr_iter;
        int m_max_iters;

        // Solver name
        std::string m_solver_name;

        // Residual and norms.
        Vector<TConfig> *m_r;
        std::vector<PODVector_h> m_res_history;
        PODVector_h m_nrm;
        PODVector_h m_nrm_ini;
        bool m_use_scalar_norm;

        // Convergence object. To decide convergence.
        Convergence<TConfig> *m_convergence;
        // The type of norms.
        NormType m_norm_type;

        //
        bool m_verbose;

        // Flags.
        bool m_is_solver_setup;

        // Configuration flags.
        int m_verbosity_level;
        //bool m_print_solve_stats;
        //bool m_print_grid_stats;
        bool m_print_vis_data;
        bool m_monitor_residual;
        bool m_monitor_convergence;
        bool m_store_res_history;
        bool m_obtain_timings;

        // Events and time informations.
        cudaEvent_t m_setup_start, m_setup_stop;
        cudaEvent_t m_solve_start, m_solve_stop;
        cudaEvent_t m_iter_start,  m_iter_stop;

        // Timings.
        float m_setup_time, m_solve_time;

        ThreadManager *m_tmng;
};

template<class TConfig>
class SolverFactory
{
    public:
        virtual Solver<TConfig> *create( AMG_Config &cfg, const std::string &cfg_scope, ThreadManager *tmng = NULL ) = 0;
        virtual ~SolverFactory() {};

        /********************************************
          * Register a solver class with key "name"
          *******************************************/
        static void registerFactory(std::string name, SolverFactory<TConfig> *f);

        /********************************************
          * Unregister a solver class with key "name"
          *******************************************/
        static void unregisterFactory(std::string name);

        /********************************************
          * Unregister all the solver classes
          *******************************************/
        static void unregisterFactories( );

        /*********************************************
        * Allocates solvers based on cfg
        *********************************************/
        static Solver<TConfig> *allocate( AMG_Config &cfg, const std::string &cfg_scope, const std::string &solverType, ThreadManager *tmng = NULL );
        static Solver<TConfig> *allocate( AMG_Config &cfg, const std::string &solverType, ThreadManager *tmng = NULL );

        /*********************************************
        * Allocates solvers based on cfg
        *********************************************/
        typedef typename std::map<std::string, SolverFactory<TConfig>*>::const_iterator Iterator;

        static Iterator getIterator() { return getFactories().begin(); };
        static bool isIteratorLast(const Iterator &iter) { if ( iter == getFactories().end() ) return true; else return false; };

    private:
        static std::map<std::string, SolverFactory<TConfig>*> &getFactories( );
};

} // namespace amgx
