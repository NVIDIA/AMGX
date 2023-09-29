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

#include "unit_test.h"
#include "amg_solver.h"
#include <matrix_io.h>
#include "test_utils.h"
#include "util.h"
#include "time.h"
#include <determinism_checker.h>
#include <solvers/solver.h>
#ifdef AMGX_WITH_MPI
#include <mpi.h>
#endif


namespace amgx
{

template<class TConfig>
class UnittestSolverFactory;

template<class TConfig>
class UnittestSolver: public Solver<TConfig>
{
    public:
        typedef Matrix<TConfig> MMatrix;
        typedef Vector<TConfig> VVector;

        UnittestSolver( AMG_Config &cfg, const std::string &cfg_scope ): is_setup(false),
            is_init(false),
            is_finalized(true),
            Solver<TConfig>( cfg, cfg_scope )
        {
            UnittestSolverFactory<TConfig>::ref_count++;
        }

        virtual ~UnittestSolver()
        {
            UnittestSolverFactory<TConfig>::ref_count--;

            if ( !is_finalized )
            {
                FatalError("UnittestSolver has not been finalized before destruction", AMGX_ERR_UNKNOWN);
            }
        }

        virtual void solver_setup(bool reuse_matrix_structure)
        {
            is_setup = true;
            is_init = false;
        }

        // What coloring level is needed for the matrix?
        virtual bool isColoringNeeded() const { return true; }

        // What reordering level is needed for the matrix?
        virtual bool getReorderColsByColorDesired() const { return true; }
        virtual bool getInsertDiagonalDesired() const { return true; }

        // Does the solver requires the residual vector storage
        virtual bool is_residual_needed( ) const { return false; }

        // Initialize the solver before running the iterations.
        virtual void solve_init( VVector &b, VVector &x, bool xIsZero )
        {
            if ( is_init )
            {
                FatalError("UnittestSolver has already been initialized", AMGX_ERR_UNKNOWN);
            }

            if ( !is_setup )
            {
                FatalError("UnittestSolver has not been set up before calling init_solve", AMGX_ERR_UNKNOWN);
            }

            is_init = true;
            is_finalized = false;
        }
        // Run a single iteration. Compute the residual and its norm and decide convergence.
        virtual AMGX_STATUS solve_iteration( VVector &b, VVector &x, bool xIsZero )
        {
            if ( !is_init )
            {
                FatalError("UnittestSolver has not been initialized before calling solve_iteration", AMGX_ERR_UNKNOWN);
            }

            return AMGX_ST_CONVERGED;
        }
        // Finalize the solver after running the iterations.
        virtual void solve_finalize( VVector &b, VVector &x )
        {
            if ( is_finalized )
            {
                FatalError("UnittestSolver has already been finalized", AMGX_ERR_UNKNOWN);
            }

            if ( !is_init )
            {
                FatalError("UnittestSolver has not been initialized before calling solve_iteration", AMGX_ERR_UNKNOWN);
            }

            is_finalized = true;
            is_init = false;
        }


    private:

        bool is_setup;
        bool is_init;
        bool is_finalized;

};


template<class T_Config>
class UnittestSolverFactory : public SolverFactory<T_Config>
{
    public:
        Solver<T_Config> *create( AMG_Config &cfg, const std::string &cfg_scope, ThreadManager *tmng ) { return new UnittestSolver<T_Config>( cfg, cfg_scope); }
        static int ref_count;
};

//init ref_count
template<class TConfig>
int UnittestSolverFactory<TConfig>::ref_count = 0;

//instantiate template solver
#define AMGX_CASE_LINE(CASE) template class UnittestSolver<TemplateMode<CASE>::Type>;
AMGX_FORALL_BUILDS(AMGX_CASE_LINE)
#undef AMGX_CASE_LINE

// parameter is used as test name
DECLARE_UNITTEST_BEGIN(PreconditionerUsage);

void run()
{
#ifdef AMGX_WITH_MPI
    int mpiFlag;
    MPI_Initialized(&mpiFlag);

    if ( !mpiFlag )
    {
        int argc = 1;
        char **argv = NULL;
        MPI_Init( &argc, &argv);
    }

#endif
    AMG_Config cfg;
    cfg.parseParameterString("config_version=2,\
                              determinism_flag=1,\
                              max_levels=1,\
                              max_iters=1,\
                              use_scalar_norm=1,\
                              reorder_cols_by_color=1,\
                              insert_diag_while_reordering=1,\
                              coloring_level=1,\
                              preconditioner=UNITTEST_SOLVER");
    //read matrices
    std::string filename = UnitTest::get_configuration().data_folder + "Internal/poisson/poisson9x4x4x4.mtx.bin";
    MatrixA A1, A2;
    Vector<T_Config> x1, b1, x2, b2;
    Vector_h x_h, b_h;
    Matrix_h A_h;
    typedef TemplateConfig<AMGX_host,   T_Config::vecPrec, T_Config::matPrec, AMGX_indInt> Config_h;
    A_h.set_initialized(0);
    A_h.addProps(CSR);
    std::string fail_msg = "Cannot open " + filename;
    this->PrintOnFail(fail_msg.c_str());
    UNITTEST_ASSERT_TRUE(MatrixIO<Config_h>::readSystem( filename.c_str(), A_h, b_h, x_h ) == AMGX_OK);
    //copy to device
    A1 = A_h;
    x1 = x_h;
    b1 = b_h;
    cudaDeviceSynchronize();
    cudaCheckError();
    b1.set_block_dimx(1);
    b1.set_block_dimy(A1.get_block_dimy());
    x1.set_block_dimx(1);
    x1.set_block_dimy(A1.get_block_dimx());
    x1.resize( b1.size(), 1.0 );
    b2.set_block_dimx(1);
    b2.set_block_dimy(A2.get_block_dimy());
    x2.set_block_dimx(1);
    x2.set_block_dimy(A2.get_block_dimx());
    x2.resize( b2.size(), 1.0 );
    typename SolverFactory<T_Config>::Iterator iter = SolverFactory<T_Config>::getIterator();
    SolverFactory<T_Config>::registerFactory("UNITTEST_SOLVER", new UnittestSolverFactory<T_Config>);
    MatrixA *A;
    Vector<T_Config> *x, *b;

    while (!SolverFactory<T_Config>::isIteratorLast(iter))
    {
        //skip
        std::string m_name = iter->first.c_str();

        if ( m_name.compare( "BLOCK_DILU" ) == 0 || //not implemented on device
                //m_name.compare( "KACZMARZ" ) == 0 ||   // breaks with coloring
                m_name.compare( "FIXCOLOR_GS" ) == 0 ) //deprecated, unusable
        {
            iter++;
            continue;
        }

        if ( m_name.compare( "MULTICOLOR_ILU" ) == 0 )
        {
            A = &A2;
            x = &x2;
            b = &b2;
        }
        else
        {
            A = &A1;
            x = &x1;
            b = &b1;
        }

        Solver<T_Config> *solver = iter->second->create( cfg, "default" );
        A->setupMatrix( solver, cfg, false );
        solver->setup( *A, false );
        solver->solve( *b, *x, false );
        solver->solve( *b, *x, false );
        delete solver;
        fail_msg = "preconditioner has not been destroyed by " + iter->first;
        PrintOnFail(fail_msg.c_str());
        UNITTEST_ASSERT_TRUE( UnittestSolverFactory<T_Config>::ref_count == 0 );
        iter++;
    }
}

DECLARE_UNITTEST_END(PreconditionerUsage);

// or run for all device configs
#define AMGX_CASE_LINE(CASE) PreconditionerUsage <TemplateMode<CASE>::Type>  PreconditionerUsage_##CASE;
AMGX_FORALL_BUILDS_DEVICE(AMGX_CASE_LINE)
#undef AMGX_CASE_LINE


} //namespace amgx


