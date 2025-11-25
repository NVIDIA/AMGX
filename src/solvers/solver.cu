// SPDX-FileCopyrightText: 2011 - 2024 NVIDIA CORPORATION. All Rights Reserved.
//
// SPDX-License-Identifier: BSD-3-Clause

#include <solvers/solver.h>
#include <scalers/scaler.h>
#include <assert.h>
#include <blas.h>
#include <multiply.h>
#include <util.h>
#include <memory_info.h>
#include <matrix_coloring/matrix_coloring.h>
#include <cusp/blas.h>
#include <numerical_zero.h>
#include <distributed/glue.h>

#include "amgx_types/util.h"

#ifdef AMGX_USE_VAMPIR_TRACE
#include <vt_user.h>
#endif

namespace amgx
{

template<class TConfig>
Solver<TConfig>::Solver(AMG_Config &cfg, const std::string &cfg_scope,
                        ThreadManager *tmng) :
    m_cfg(&cfg), m_cfg_scope(cfg_scope), m_is_solver_setup(false), m_A(NULL), 
    m_r(NULL), m_num_iters(0), m_curr_iter(0), m_ref_count(1), tag(0), 
    m_solver_name("SolverNameNotSet"), m_skip_glued_setup(false), m_tmng(tmng)
{
    m_norm_factor = types::util<PODValueB>::get_one();
    m_verbosity_level = cfg.getParameter<int>("verbosity_level", cfg_scope);
    m_print_vis_data = cfg.getParameter<int>("print_vis_data", cfg_scope) != 0;
    m_monitor_residual = cfg.getParameter<int>("monitor_residual", cfg_scope) != 0;
    m_store_res_history = cfg.getParameter<int>("store_res_history", cfg_scope) != 0;
    m_obtain_timings = cfg.getParameter<int>("obtain_timings", cfg_scope) != 0;
    m_scaling = cfg.getParameter<std::string>("scaling", cfg_scope);

    if ( m_scaling.compare("NONE") != 0 ) //create scaler object
    {
        m_Scaler = ScalerFactory<TConfig>::allocate( cfg, cfg_scope);
    }
    else
    {
        m_Scaler = NULL;    //Mark as no scaler present
    }

    // Today we monitor convergence iff we monitor residual. We could be skipping convergence analysis even if we monitor residual.
    m_monitor_convergence = m_monitor_residual;

    // If scopes haven't been used, reset the print parameters to default value. This is for backward compatibility
    if (cfg_scope == "default")
    {
        cfg.setParameter("obtain_timings", 0, cfg_scope);
        cfg.setParameter("print_solve_stats", 0, cfg_scope);
        cfg.setParameter("print_grid_stats", 0, cfg_scope);
        cfg.setParameter("print_vis_data", 0, cfg_scope);
        cfg.setParameter("store_res_history", 0, cfg_scope);
        cfg.setParameter("monitor_residual", 0, cfg_scope);
    }

    // Make sure parameters are compatible.
    if (getPrintSolveStats() && !m_monitor_residual)
        FatalError(
            "Cannot print solver information if residual is not monitored (i.e. print_solve_stats=1 and monitor_residual=0) ",
            AMGX_ERR_BAD_PARAMETERS);

    if (m_store_res_history && !m_monitor_residual)
        FatalError(
            "Cannot store residual information if residual is not monitored (i.e. store_res_history=1 and monitor_residual=0) ",
            AMGX_ERR_BAD_PARAMETERS);

    // Get the max number of iterations/the type of norm and the convergence object.
    m_max_iters = cfg.getParameter<int>("max_iters", cfg_scope);
    m_norm_type = cfg.getParameter<NormType>("norm", cfg_scope);
    m_use_scalar_norm = cfg.getParameter<int>("use_scalar_norm", cfg_scope);
    m_convergence = ConvergenceFactory<TConfig>::allocate(cfg, cfg_scope);
    m_verbose = (cfg.getParameter<int>("solver_verbose", cfg_scope) == 1);

    // Resize residual history to make sure we have enough space. We store the initial residual at position 0.
    if (m_store_res_history)
    {
        m_res_history.resize(m_max_iters + 1);
    }

    if (!m_obtain_timings)
    {
        return;
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
    // Reset times.
    m_setup_time = 0.0f;
    m_solve_time = 0.0f;
}

template<class TConfig>
Solver<TConfig>::~Solver() noexcept(false)
{
    if (m_obtain_timings)
    {
#ifdef AMGX_WITH_MPI
#ifdef MPI_SOLVE_PROFILE

        if ( m_A && !m_A->is_matrix_singleGPU())
        {
            MPI_Barrier(MPI_COMM_WORLD);
        }

#ifdef AMGX_USE_VAMPIR_TRACE
        int tag = VT_User_marker_def__("Solver_Destructor", VT_MARKER_TYPE_HINT);
        VT_User_marker__(tag, "Solver Destructor");
#endif
#endif
#endif
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

    delete m_r;
    delete m_convergence;

    if ( m_Scaler != NULL) { delete m_Scaler; }
}


template<class TConfig>
void Solver<TConfig>::reset_setup_timer()
{
    m_setup_time = 0.0f;
}

template<class TConfig>
void Solver<TConfig>::reset_solve_timer()
{
    m_solve_time = 0.0f;
}

// Method to print current norm
template<class TConfig>
void Solver<TConfig>::print_norm(std::stringstream &ss) const
{
    for (int i = 0; i < m_nrm.size(); ++i)
    {
        ss << std::scientific << std::setprecision(6) << std::setw(15) << m_nrm[i];
    }
}

// Method to print current norm
template<class TConfig>
void Solver<TConfig>::print_norm2(std::stringstream &ss) const
{
    for (int i = 0; i < m_nrm.size(); ++i)
    {
        ss << std::scientific << std::setprecision(4) << m_nrm[i] << " ";
    }
}

// Method to compute residual
template<class TConfig>
void Solver<TConfig>::compute_residual(const VVector &b, VVector &x)
{
    AMGX_CPU_PROFILER( "Solver::compute_residual_bx " );
    assert(m_A);
    assert(m_r); //r and b/x are not the same size
    int size, offset;
    m_A->getOffsetAndSizeForView(OWNED, &offset, &size);
    // r = b - Ax.
    m_A->apply(x, *m_r);
    axpby(b, *m_r, *m_r, types::util<ValueTypeB>::get_one(), types::util<ValueTypeB>::get_minus_one(), offset, size);
}

template<class TConfig>
void Solver<TConfig>::compute_residual(const VVector &b, VVector &x,
                                       VVector &r) const
{
    AMGX_CPU_PROFILER( "Solver::compute_residual_bxr " );
    int size, offset;
    m_A->getOffsetAndSizeForView(OWNED, &offset, &size);
    m_A->apply(x, r);
    axpby(b, r, r, types::util<ValueTypeB>::get_one(), types::util<ValueTypeB>::get_minus_one(), offset, size);
}

template<class TConfig>
void Solver<TConfig>::compute_norm()
{
    AMGX_CPU_PROFILER( "Solver::compute_norm " );
    get_norm(*m_A, *m_r, (m_use_scalar_norm ? 1 : m_A->get_block_dimy()),
             m_norm_type, m_nrm, m_norm_factor);
}

template<class TConfig>
void Solver<TConfig>::compute_norm(const VVector &v, PODVector_h &nrm) const
{
    AMGX_CPU_PROFILER( "Solver::compute_norm_vh " );
    get_norm(*m_A, v, (m_use_scalar_norm ? 1 : m_A->get_block_dimy()),
             m_norm_type, nrm, m_norm_factor);
}

template<class TConfig>
void Solver<TConfig>::compute_residual_norm_external(Operator<TConfig> &mtx, const VVector &b, const VVector &x, typename PODVector_h::value_type *nrm) const
{
    AMGX_CPU_PROFILER( "Solver::compute_residual_norm_vh " );
    int size, offset;
    VVector r;
    PODVector_h t_norm((m_use_scalar_norm ? 1 : mtx.get_block_dimy()));
    {
        r.resize(mtx.get_num_cols() * mtx.get_block_dimy());
        r.set_block_dimy(mtx.get_block_dimy());
        r.set_block_dimx(1);
    }
    mtx.getOffsetAndSizeForView(OWNED, &offset, &size);
    mtx.apply(x, r);
    axpby(b, r, r, types::util<ValueTypeB>::get_one(), types::util<ValueTypeB>::get_minus_one(), offset, size);
    get_norm(mtx, r, (m_use_scalar_norm ? 1 : mtx.get_block_dimy()), m_norm_type, t_norm);

    for (int i = 0; i < t_norm.size(); i++)
    {
        nrm[i] = t_norm[i];
    }
}

template<class TConfig>
void Solver<TConfig>::set_norm(const PODVector_h &nrm)
{
    if (nrm.size() != 1)
        FatalError("Functionality not tested for bsize != 1",
                   AMGX_ERR_NOT_SUPPORTED_TARGET);

    if (m_nrm_ini[0] == -1)
    {
        m_nrm_ini = nrm;
    }

    m_nrm = nrm;
}

// Method to check convergence
template<class TConfig>
AMGX_STATUS Solver<TConfig>::converged(const VVector &b, VVector &x)
{
    AMGX_CPU_PROFILER( "Solver::converged_bx " );

    if (m_monitor_residual)
    {
        this->compute_residual(b, x);
    }

    AMGX_STATUS converged = AMGX_ST_NOT_CONVERGED;

    if (m_monitor_convergence)
    {
        this->compute_norm();
        converged = this->converged();
    }

    return converged;
}

template<class TConfig>
AMGX_STATUS Solver<TConfig>::converged() const
{
    AMGX_CPU_PROFILER( "Solver::converged " );
    return m_convergence->convergence_update_and_check(m_nrm, m_nrm_ini);
}

template<class TConfig>
AMGX_STATUS Solver<TConfig>::converged(PODVector_h &nrm) const
{
    AMGX_CPU_PROFILER( "Solver::converged_nrm " );
    return m_convergence->convergence_update_and_check(nrm, m_nrm_ini);
}

template<class TConfig>
void Solver<TConfig>::exchangeSolveResultsConsolidation(AMGX_STATUS &status)
{
    this->get_A().getManager()->exchangeSolveResultsConsolidation(m_num_iters, m_res_history, status, m_store_res_history == 1);
}

template<class TConfig>
const typename Solver<TConfig>::PODVector_h &Solver<TConfig>::get_residual(
    int idx) const
{
    if (!m_store_res_history)
    {
        FatalError("Residual history was not recorded", AMGX_ERR_BAD_PARAMETERS);
    }

    if (idx >= m_res_history.size())
        FatalError("Invalid iteration index while retrieving residual",
                   AMGX_ERR_BAD_PARAMETERS);

    return m_res_history[idx];
}

template<class TConfig>
void Solver<TConfig>::set_max_iters(int max_iters)
{
    if (m_store_res_history && max_iters + 1 > m_res_history.size())
    {
        m_res_history.resize(max_iters + 1);
    }

    m_max_iters = max_iters;
}

//Setup the solver
template<class TConfig>
void Solver<TConfig>::setup( Operator<TConfig> &A, bool reuse_matrix_structure)
{
    AMGX_CPU_PROFILER("Solver::setup ");

    if (m_verbose)
    {
        std::cout
                << "----------------------------------------------------------------------------------"
                << std::endl;
        std::cout << "Parameters for solver: " << this->getName()
                  << " with scope name: " << this->getScope() << std::endl << std::endl;
        printSolverParameters();
        std::cout << "max_iters = " << m_max_iters << std::endl;
        std::cout << "scaling = " << m_scaling << std::endl;
        std::cout << "norm = " << getString(m_norm_type) << std::endl;
        std::cout << "convergence = " << m_convergence->getName() << std::endl;
        std::cout << "solver_verbose= " << m_verbose << std::endl;
        std::cout << "use_scalar_norm = " << m_use_scalar_norm << std::endl;
        std::cout << "print_solve_stats = " << getPrintSolveStats() << std::endl;
        std::cout << "print_grid_stats = " << getPrintGridStats() << std::endl;
        std::cout << "print_vis_data = " << m_print_vis_data << std::endl;
        std::cout << "monitor_residual = " << m_monitor_residual << std::endl;
        std::cout << "store_res_history = " << m_store_res_history << std::endl;
        std::cout << "obtain_timings = " << m_obtain_timings << std::endl;
        std::cout
                << "----------------------------------------------------------------------------------"
                << std::endl << std::endl;
    }

    // Start the timer if needed.
    if (m_obtain_timings)
    {
#ifdef AMGX_WITH_MPI
#ifdef MPI_SOLVE_PROFILE

        if ( !A.is_matrix_singleGPU() )
        {
            MPI_Barrier(MPI_COMM_WORLD);
        }

#ifdef AMGX_USE_VAMPIR_TRACE
        int tag = VT_User_marker_def__("Setup_Start", VT_MARKER_TYPE_HINT);
        VT_User_marker__(tag, "Setup Constructor");
#endif
        cudaDeviceSynchronize();
        cudaCheckError();
#endif
#endif
        cudaEventRecord(m_setup_start);
        cudaCheckError();
    }

    Matrix<TConfig> *B_ptr = dynamic_cast<Matrix<TConfig>*>(&A);

    if (B_ptr)
    {
        Matrix<TConfig> &B = *B_ptr;

        if (!B.is_initialized())
            FatalError("Trying to setup from the uninitialized matrix",
                       AMGX_ERR_BAD_PARAMETERS);

        if (reuse_matrix_structure && (&(this->get_A())) != &A)
            FatalError("Cannot call resetup with a different matrix",
                       AMGX_ERR_UNKNOWN);

#ifdef AMGX_WITH_MPI
        // skiping setup for glued matrices
        // this->level was set to -999 in amg_level.cu because it is empty
        // block jacobi fails to find diagonal if the matrix is empty
        if (B.manager != NULL)
        {
            if (this->m_skip_glued_setup)
            {
                this->set_A(A);
                m_is_solver_setup = true;
                return;
            }
        }
#endif

        // Color the matrix, set the block format, reorder columns if necessary
        if (!B.is_matrix_setup())
        {
            B.setupMatrix(this, *m_cfg, reuse_matrix_structure);
        }
        // The matrix has been created without coloring. Color it!!!
        // NOTE: It's a hack added to deal with asynchronous execution. When we have
        // a better execution model, we can remove that hack.
        else if (this->isColoringNeeded() && !B.hasProps(COLORING))
        {
            std::string cfg_scope_for_coloring;
            this->getColoringScope(cfg_scope_for_coloring);
            B.set_initialized(0);
            B.colorMatrix(*this->m_cfg, cfg_scope_for_coloring);
            B.set_initialized(1);
        }
    }

    // Set the pointer to matrix A
    this->set_A(A);
    // Apply scaling to matrix, first check that it is a matrix and not just an operator.
/// ================== WARNING ================= ///
/// Current state of scaling is not finished. This is due to the fact that it was working incorrectly without current workaround
/// Currently it is very slow and works as following:
/// I. setup
/// 1) Scale matrix in the setup, cache the scaling coefficients
/// 2) Setup solver using _scaled matrix_
/// 3) Unscale matrix
/// II. solve
/// 1) Scale matrix, rhs and unscale the initial solution
/// 2) Solve system using solver
/// 3) Scale initial solution, unscale matrix and rhs
/// This is implemented this way because there are a lot of norm residual checks which are
/// not aware of scaling and will trigger convergence criteria when it is not met.
/// Current implementation is very slow due to all of those scales/unscales, but it produces _correct_ convergence checks.
/// The way to get rid of these computations: custom kernels of residual norm calculations that will unscale/scale on the fly using current scaler.
    Matrix<TConfig> *m_A =  dynamic_cast<Matrix<TConfig>*>(this->m_A);
    Matrix<TConfig> &mref_A = *m_A;
    m_A->template setParameter<int>("scaled", 0);

    if ( m_scaling.compare("NONE") != 0 )
    {
        int lvl = m_A->template getParameter<int>("level");
        // We should not scale if it is preconditioner for some outer solver - in this case finest level matrix will be already scaled.
        // However this could be avoided by providing scaling parameter for outer solver and not inner (or vice versa)
        {
            if ( !m_A) //not a matrix!
            {
                FatalError("Matrix scaling only works with explicit matrices, set scaling=NONE for operators.", AMGX_ERR_INTERNAL);
            }

            m_Scaler->setup( mref_A );
            m_Scaler->scaleMatrix( mref_A, amgx::SCALE );
            m_A->template setParameter<int>("scaled", 1);
        }
    }

    // Setup the solver
    solver_setup(reuse_matrix_structure);

    if ( m_scaling.compare("NONE") != 0 )
    {
        m_Scaler->scaleMatrix( mref_A, amgx::UNSCALE );
    }

    // Setup the solver
    // Allocate residual vector if needed
    if (m_monitor_residual || is_residual_needed())
    {
        int needed_size = m_A->get_num_cols() * m_A->get_block_dimy();

        if (m_r == NULL)
        {
            m_r = new VVector(needed_size);
        }
        else
        {
            m_r->resize(needed_size);
        }

        if (m_r->tag == -1)
        {
            m_r->tag = this->tag * 100 + 2;
        }

        assert(m_r != NULL && m_r->size() >= needed_size);
        m_r->set_block_dimy(m_A->get_block_dimy());
        m_r->set_block_dimx(1);
        m_r->delayed_send = 1;
    }

    // Make sure the norm vectors have the right size.
    if (m_monitor_convergence)
    {
        const int bsize = m_use_scalar_norm ? 1 : this->get_A().get_block_dimy();
        m_nrm.resize(bsize, types::util<PODValueB>::get_zero());
        m_nrm_ini.resize(bsize, types::util<PODValueB>::get_zero());
    }

    // Stop the timer if needed.
    if (m_obtain_timings)
    {
        cudaEventRecord(m_setup_stop);
        cudaCheckError();
        cudaEventSynchronize(m_setup_stop);
        cudaCheckError();
        cudaEventElapsedTime(&m_setup_time, m_setup_start, m_setup_stop);
        m_setup_time *= 1e-3f;
#ifdef AMGX_WITH_MPI
#ifdef MPI_SOLVE_PROFILE

        if ( !A.is_matrix_singleGPU() )
        {
            MPI_Barrier(MPI_COMM_WORLD);
        }

#ifdef AMGX_USE_VAMPIR_TRACE
        int tag = VT_User_marker_def__("Setup_End", VT_MARKER_TYPE_HINT);
        VT_User_marker__(tag, "Setup Destructor");
#endif
        cudaDeviceSynchronize();
        cudaCheckError();
#endif
#endif
    }

    // Print information if needed.
    if (m_verbosity_level > 2 && getPrintGridStats())
    {
        print_grid_stats();
    }

    if (m_verbosity_level > 2 && m_print_vis_data)
    {
        print_vis_data();
    }

    // The solver has been setup!!!
    m_is_solver_setup = true;
}
template<class TConfig>
AMGX_ERROR Solver<TConfig>::setup_no_throw(Operator<TConfig> &A,
        bool reuse_matrix_structure)
{
    AMGX_ERROR rc = AMGX_OK;

    AMGX_TRIES()
    {
        if ( (A.getManager() != NULL && A.getManager()->isFineLevelConsolidated() && !A.getManager()->isFineLevelRootPartition() ))
        {
            this->set_A(A);
            // Do nothing else since this partition shouldn't participate
        }
        else
        {
            // Matrix values have changed, so need to repermute values, color the matrix if necessary
            if (Matrix<TConfig> *B = dynamic_cast<Matrix<TConfig>*>(&A))
            {
                B->set_is_matrix_setup(false);
            }

            // Setup the solver
            this->setup(A, reuse_matrix_structure);
        }
    }

    AMGX_CATCHES(rc)
    return rc;
}
;

template<class TConfig>
AMGX_STATUS Solver<TConfig>::solve(Vector<TConfig> &b, Vector<TConfig> &x,
                                   bool xIsZero)
{
    PODValueB eps = (sizeof(PODValueB) == 4) ? AMGX_NUMERICAL_SZERO : AMGX_NUMERICAL_DZERO;
    AMGX_CPU_PROFILER("Solver::solve ");

    if (!m_is_solver_setup)
    {
        FatalError("Error, setup must be called before calling solve",
                   AMGX_ERR_CONFIGURATION);
    }

    if (b.get_block_size() != m_A->get_block_dimy())
    {
        FatalError("Block sizes do not match", AMGX_ERR_BAD_PARAMETERS);
    }

    //  --- Gluing path for vectors ---
#ifdef AMGX_WITH_MPI
    Matrix<TConfig> *nv_mtx_ptr =  dynamic_cast<Matrix<TConfig>*>(m_A);

    if (nv_mtx_ptr)
    {
        if (nv_mtx_ptr->manager != NULL )
        {
            if (nv_mtx_ptr->manager->isGlued() && nv_mtx_ptr->manager->getDestinationPartitions().size() != 0 && nv_mtx_ptr->amg_level_index == 0)
            {
                MPI_Comm comm, temp_com;
                comm = nv_mtx_ptr->manager->getComms()->get_mpi_comm();
                // Compute the temporary splited communicator to glue vectors
                temp_com = compute_glue_matrices_communicator(*nv_mtx_ptr);
                int usz = nv_mtx_ptr->manager->halo_offsets_before_glue[0];
                glue_vector(*nv_mtx_ptr, comm, b, temp_com);
                b.set_unconsolidated_size(usz);
                b.getManager()->setIsFineLevelGlued(true);
                MPI_Barrier(MPI_COMM_WORLD);
                glue_vector(*nv_mtx_ptr, comm, x, temp_com);
                x.set_unconsolidated_size(usz);
                x.getManager()->setIsFineLevelGlued(true);
                MPI_Barrier(MPI_COMM_WORLD);
                //Make sure we will not glue the vectors twice on the finest level
                nv_mtx_ptr->manager->setIsGlued(false);
            }
            else
            {
                nv_mtx_ptr->manager->setIsGlued(false);
            }
        }
    }

#endif
    // -- end of gluing path modifications --

    if (b.tag == -1 || x.tag == -1)
    {
        b.tag = this->tag * 100 + 0;
        x.tag = this->tag * 100 + 1;
    }

    if (m_obtain_timings)
    {
#ifdef AMGX_WITH_MPI
#ifdef MPI_SOLVE_PROFILE

        if ( m_A && !m_A->is_matrix_singleGPU() )
        {
            MPI_Barrier(MPI_COMM_WORLD);
        }

#ifdef AMGX_USE_VAMPIR_TRACE
        int tag = VT_User_marker_def__("Solver_Start", VT_MARKER_TYPE_HINT);
        VT_User_marker__(tag, "Solver Start");
#endif
        cudaDeviceSynchronize();
        cudaCheckError();
#endif
#endif
        cudaEventRecord(m_solve_start);
        cudaCheckError();
    }

    // if scaling, time to scale the rhs
    if (m_Scaler != NULL ) // we have setup a scaler
    {
        Matrix<TConfig> *m_A =  dynamic_cast<Matrix<TConfig>*>(this->m_A);
        m_Scaler->scaleMatrix( *m_A, amgx::SCALE );
        m_Scaler->scaleVector( b, amgx::SCALE, amgx::LEFT); //rescale rhs in place
        m_Scaler->scaleVector( x, amgx::UNSCALE, amgx::RIGHT); //rescale x in place
        //x.setParameter<bool>("workWithScaling", true);
    }

    // if overiding number of iterations with 0 iterations
    const int bsize = m_A->get_block_dimy();

    // Is monitoring residual, or if solver needs residual storage
    if (m_monitor_residual || is_residual_needed())
    {
        // Compute the initial residual
        if (xIsZero)
        {
            (this->m_r)->copy(b);
        }
        else
        {
            compute_residual(b, x);
        }
    }

    // If we monitor convergence, we compute the norm of the residual.
    PODVector_h last_nrm;

    if (m_monitor_convergence)
    {
        if (!m_use_scalar_norm)
        {
            assert(static_cast<int>(m_nrm.size()) >= bsize);
        }

        if (!m_use_scalar_norm)
        {
            assert(static_cast<int>(m_nrm_ini.size()) >= bsize);
        }

        // Only happens if L1 scaled norm is utilised
        Matrix<TConfig> *m_A =  dynamic_cast<Matrix<TConfig>*>(this->m_A);
        compute_norm_factor(*m_A, b, x, m_norm_type, m_norm_factor);

        compute_norm();
        last_nrm = m_nrm_ini = m_nrm;
    }

    // If store_res_history is true, it means m_monitor_residual is also true....
    if (m_store_res_history)
    {
        assert(m_monitor_residual);
        m_res_history[0] = m_nrm;
    }

    // Print solve informations if needed.
    if (m_verbosity_level > 2 && getPrintSolveStats())
    {
        std::stringstream ss;
        ss << std::setw(15) << "iter" << std::setw(20) << " Mem Usage (GB)"
           << std::setw(15) << "residual";

        for (int i = 0; i < m_nrm.size() - 1; i++)
        {
            ss << std::setw(15) << " ";
        }

        ss << std::setw(15) << "rate";

        for (int i = 0; i < m_nrm.size() - 1; i++)
        {
            ss << std::setw(15) << " ";
        }

        ss << std::endl;
        ss
                << "         ----------------------------------------------------------------------";

        for (int i = 0; i < m_nrm.size() - 1; i++)
        {
            ss << "-----------------------";    // 15 + 8
        }

        ss << std::endl;
        ss << std::setw(15) << "Ini";
        ss << std::setw(20) << MemoryInfo::getMaxMemoryUsage();
        print_norm(ss);
        ss << std::endl;
        amgx_output(ss.str().c_str(), static_cast<int>(ss.str().length()));
    }

    if ((m_verbosity_level == 1 || m_verbosity_level == 2) && getPrintSolveStats())
    {
        std::stringstream ss;
        ss << "tol. ";
        const int bsize = m_use_scalar_norm ? 1 : this->get_A().get_block_dimy();
        PODVector_h m_nrm2;
        m_nrm2.resize(bsize, types::util<PODValueB>::get_zero());
        compute_norm(b, m_nrm2);

        for (int i = 0; i < m_nrm2.size(); ++i)
        {
            ss << std::scientific << std::setprecision(4) << m_nrm2[i] * eps << " ";
        }

        ss << std::endl;
        amgx_output(ss.str().c_str(), static_cast<int>(ss.str().length()));
    }

    // Initialize convergence checker if needed.
    if (m_monitor_convergence)
    {
        m_convergence->convergence_init();
        m_convergence->convergence_update_and_check(m_nrm, m_nrm_ini);
    }

    // Initialize the solver
    bool done = m_monitor_convergence ? (converged() == AMGX_ST_CONVERGED) : false;

    if (m_max_iters == 0)
    {
        return m_monitor_convergence ? AMGX_ST_NOT_CONVERGED : AMGX_ST_CONVERGED;
    }

    if (!done)
    {
        solve_init(b, x, xIsZero);
    }

    AMGX_STATUS conv_stat = done ? AMGX_ST_CONVERGED : AMGX_ST_NOT_CONVERGED;

    // Run the iterations
    std::stringstream ss;

    for (m_curr_iter = 0; m_curr_iter < m_max_iters && !done; ++m_curr_iter)
    {
        // Run one iteration. Compute residual and its norm and decide convergence
        conv_stat = solve_iteration(b, x, xIsZero);
        // Make sure x is not zero anymore.
        xIsZero = false;
        // Is it done ?
        done = m_monitor_convergence && isDone(conv_stat);

        // If we print stats... Let's do it.
        if (m_verbosity_level > 2 && getPrintSolveStats())
        {
            ss.str(std::string());
            ss << std::setw(15) << m_curr_iter;
            ss << std::setw(20) << MemoryInfo::getMaxMemoryUsage();
            print_norm(ss);
            ss << std::setw(15);

            for (int i = 0; i < last_nrm.size(); i++)
                ss << std::fixed << std::setprecision(4) << m_nrm[i] / last_nrm[i]
                   << std::setw(8);

            ss << std::endl;
            amgx_output(ss.str().c_str(), static_cast<int>(ss.str().length()));
            last_nrm = m_nrm;
        }

        if ((m_verbosity_level == 1 || m_verbosity_level == 2) && getPrintSolveStats())
        {
            ss.str(std::string());
            ss << std::setw(4) << m_curr_iter << " ";
            print_norm2(ss);
            ss << std::endl;
            amgx_output(ss.str().c_str(), static_cast<int>(ss.str().length()));
        }

        // If we need to store the residual, do it.
        if (m_store_res_history)
        {
            assert(m_curr_iter + 1 < static_cast<int>(m_res_history.size()));
            m_res_history[m_curr_iter + 1] = m_nrm;
        }
    } // loop over iterations

    // The number of iterations.
    m_num_iters = m_curr_iter;

    // Finalize the solver if needed.
    if (m_num_iters > 0)
    {
        solve_finalize(b, x);
    }

    if ( m_Scaler != NULL) //rescale solution to match equation scaling
    {
        Matrix<TConfig> *m_A =  dynamic_cast<Matrix<TConfig>*>(this->m_A);
        m_Scaler->scaleVector( x, amgx::SCALE, amgx::RIGHT);
        m_Scaler->scaleVector( b, amgx::UNSCALE, amgx::LEFT); //rescale rhs in place
        m_Scaler->scaleMatrix( *m_A, amgx::UNSCALE );
        //x.setParameter<bool>("workWithScaling", false);
    }

    // Collect timings if needed.
    if (m_obtain_timings)
    {
        cudaEventRecord(m_solve_stop);
        cudaCheckError();
        cudaEventSynchronize(m_solve_stop);
        cudaCheckError();
        cudaEventElapsedTime(&m_solve_time, m_solve_start, m_solve_stop);
        m_solve_time *= 1e-3f;
#ifdef AMGX_WITH_MPI
#ifdef MPI_SOLVE_PROFILE

        if ( m_A && !m_A->is_matrix_singleGPU() )
        {
            MPI_Barrier(MPI_COMM_WORLD);
        }

#ifdef AMGX_USE_VAMPIR_TRACE
        int tag = VT_User_marker_def__("Solver_End", VT_MARKER_TYPE_HINT);
        VT_User_marker__(tag, "Solver End");
#endif
        cudaDeviceSynchronize();
        cudaCheckError();
#endif
#endif
    }

    // Print residual convergence information
    if (m_verbosity_level > 2 && getPrintSolveStats())
    {
        ss.str(std::string());
        ss
                << "         ----------------------------------------------------------------------";

        for (int i = 0; i < m_nrm.size() - 1; i++)
        {
            ss << "-----------------------";
        }

        ss << std::endl;
        ss << "         Total Iterations: " << m_num_iters << std::endl;
        ss << "         Avg Convergence Rate: \t\t";

        for (int i = 0; i < last_nrm.size(); i++)
            ss << std::fixed << std::setw(15)
               << ((m_nrm_ini[i] > eps) ? pow(last_nrm[i] / m_nrm_ini[i],  types::util<PODValueB>::get_one() / m_num_iters) : m_nrm_ini[i]);

        ss << std::endl;
        ss << "         Final Residual: \t\t" << std::setprecision(6);

        for (int i = 0; i < last_nrm.size(); i++)
        {
            ss << std::scientific << std::setw(15) << last_nrm[i] << std::fixed;
        }

        ss << std::endl;
        ss << "         Total Reduction in Residual: \t" << std::setprecision(6);

        for (int i = 0; i < last_nrm.size(); i++)
            ss << std::scientific << std::setw(15)
               << ((m_nrm_ini[i] > eps) ? last_nrm[i] / m_nrm_ini[i] : m_nrm_ini[i])
               << std::fixed;

        ss << std::endl;
        ss << "         Maximum Memory Usage: \t\t" << std::setprecision(3)
           << std::setw(15) << MemoryInfo::getMaxMemoryUsage() << " GB"
           << std::endl;
        ss
                << "         ----------------------------------------------------------------------";

        for (int i = 0; i < m_nrm.size() - 1; i++)
        {
            ss << "-----------------------";
        }

        ss << std::endl;
        amgx_output(ss.str().c_str(), static_cast<int>(ss.str().length()));
        ss.str(std::string());
        // what should we use as comparison of block norm?
    }

    // print grid hierarchy
    if (m_verbosity_level == 2 && getPrintGridStats())
    {
        ss.str(std::string());
        ss << std::endl;
        amgx_output(ss.str().c_str(), static_cast<int>(ss.str().length()));
        print_grid_stats2();
    }

    // extra line
    if ((m_verbosity_level == 1 || m_verbosity_level == 2) && getPrintGridStats())
    {
        ss.str(std::string());
        ss << std::endl;
        amgx_output(ss.str().c_str(), static_cast<int>(ss.str().length()));
    }

    // Print timing results
    if (m_verbosity_level > 2 && m_obtain_timings)
    {
        print_timings();
    }

    return conv_stat;
}

template<class TConfig>
AMGX_ERROR Solver<TConfig>::solve_no_throw(VVector &b, VVector &x,
        AMGX_STATUS &status, bool xIsZero)
{
    AMGX_ERROR rc = AMGX_OK;

    AMGX_TRIES()
    {
        // Check if fine level is consolidated and not a root partition
        if ( !(this->get_A().getManager() != NULL && this->get_A().getManager()->isFineLevelConsolidated() && !this->get_A().getManager()->isFineLevelRootPartition() ))
        {
            // If matrix is consolidated on fine level and not a root partition
            if (b.tag == -1 || x.tag == -1)
            {
                b.tag = this->tag * 100 + 0;
                x.tag = this->tag * 100 + 1;
            }

            status = this->solve(b, x, xIsZero);
        }

        // Exchange residual history, number of iterations, solve status if fine level consoildation was used
        if (this->get_A().getManager() != NULL && this->get_A().getManager()->isFineLevelConsolidated())
        {
            this->exchangeSolveResultsConsolidation(status);
        }
    }

    AMGX_CATCHES(rc)
    return rc;
}

template<class TConfig>
bool Solver<TConfig>::getPrintSolveStats()
{
    return m_cfg->template getParameter<int>("print_solve_stats", m_cfg_scope) != 0;
}

template<class TConfig>
bool Solver<TConfig>::getPrintGridStats()
{
    return m_cfg->template getParameter<int>("print_grid_stats", m_cfg_scope) != 0;
}

template<class TConfig>
void Solver<TConfig>::setTolerance(double tol)
{
    m_convergence->setTolerance(tol);
}

template<class TConfig>
void Solver<TConfig>::print_timings()
{
    std::stringstream ss;
    ss << "Total Time: " << m_setup_time + m_solve_time << std::endl;
    ss << "    setup: " << m_setup_time << " s\n";
    ss << "    solve: " << m_solve_time << " s\n";
    ss << "    solve(per iteration): " << ((m_num_iters == 0) ? m_num_iters : m_solve_time / m_num_iters) << " s\n";
    amgx_output(ss.str().c_str(), static_cast<int>(ss.str().length()));
}

using std::scientific;
using std::fixed;

template<class TConfig>
std::map<std::string, SolverFactory<TConfig>*> &
SolverFactory<TConfig>::getFactories()
{
    static std::map<std::string, SolverFactory<TConfig>*> s_factories;
    return s_factories;
}

template<class TConfig>
void SolverFactory<TConfig>::registerFactory(std::string name,
        SolverFactory<TConfig> *f)
{
    std::map<std::string, SolverFactory<TConfig>*> &factories = getFactories();
    typename std::map<std::string, SolverFactory<TConfig> *>::const_iterator it =
        factories.find(name);

    if (it != factories.end())
    {
        std::string error = "SolverFactory '" + name + "' has already been registered\n";
        FatalError(error.c_str(), AMGX_ERR_CORE);
    }

    factories[name] = f;
}

template<class TConfig>
void SolverFactory<TConfig>::unregisterFactory(std::string name)
{
    std::map<std::string, SolverFactory<TConfig>*> &factories = getFactories();
    typename std::map<std::string, SolverFactory<TConfig> *>::iterator it = factories.find(
                name);

    if (it == factories.end())
    {
        std::string error = "SolverFactory '" + name + "' has not been registered\n";
        FatalError(error.c_str(), AMGX_ERR_CORE);
    }

    SolverFactory<TConfig> *factory = it->second;
    assert(factory != NULL);
    delete factory;
    factories.erase(it);
}

template<class TConfig>
void SolverFactory<TConfig>::unregisterFactories()
{
    std::map<std::string, SolverFactory<TConfig>*> &factories = getFactories();
    typename std::map<std::string, SolverFactory<TConfig> *>::iterator it =
        factories.begin();

    for (; it != factories.end();)
    {
        SolverFactory<TConfig> *factory = it->second;
        assert(factory != NULL);
        it++;
        delete factory;
    }

    factories.clear();
}

template<class TConfig>
Solver<TConfig> *SolverFactory<TConfig>::allocate(AMG_Config &cfg,
        const std::string &current_scope, const std::string &solverType,
        ThreadManager *tmng)
{
    std::map<std::string, SolverFactory<TConfig>*> &factories = getFactories();
    std::string solverName, new_scope;
    cfg.getParameter<std::string>(solverType, solverName, current_scope,
                                  new_scope);

    if ( (solverType == "coarse_solver" ||
            solverType == "smoother" ||
            solverType == "preconditioner") &&
            (solverName == "AMG" || solverName == "FGMRES" ||
             solverName == "PCGF" || solverName == "PBICGSTAB" ||
             solverName == "PCG") &&
            new_scope == "default")
    {
        std::string error = "Solver " + solverName
                       + " uses an inner solver (i.e. a preconditioner, smoother or coarse_solver) and therefore cannot be used as an inner solver with the default scope due to the possibility of an infinite number of nested solvers. Please use config_version=2, and specify a new scope name for the inner solver. For example: preconditioner(amg_solver) = AMG. \n";
        FatalError( error.c_str(), AMGX_ERR_BAD_PARAMETERS);
    }

    typename std::map<std::string, SolverFactory<TConfig> *>::const_iterator it =
        factories.find(solverName);

    if (it == factories.end())
    {
        std::string error = "SolverFactory '" + solverName
                       + "' has not been registered\n";
        FatalError( error.c_str( ), AMGX_ERR_CORE);
    }

    Solver<TConfig> *solver = it->second->create(cfg, new_scope, tmng);
    solver->setName(solverName);
    return solver;
}
;

template<class TConfig>
Solver<TConfig> *SolverFactory<TConfig>::allocate(AMG_Config &cfg,
        const std::string &solverType, ThreadManager *tmng)
{
    return SolverFactory<TConfig>::allocate(cfg, "default", solverType, tmng);
}

/****************************************
 * Explict instantiations
 ***************************************/
#define AMGX_CASE_LINE(CASE) template class Solver<TemplateMode<CASE>::Type>;
AMGX_FORALL_BUILDS(AMGX_CASE_LINE)
AMGX_FORCOMPLEX_BUILDS(AMGX_CASE_LINE)
#undef AMGX_CASE_LINE

#define AMGX_CASE_LINE(CASE) template class SolverFactory<TemplateMode<CASE>::Type>;
AMGX_FORALL_BUILDS(AMGX_CASE_LINE)
AMGX_FORCOMPLEX_BUILDS(AMGX_CASE_LINE)
#undef AMGX_CASE_LINE

}// namespace amgx

