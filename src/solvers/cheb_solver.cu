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

#include <solvers/cheb_solver.h>
#include <operators/solver_operator.h>
#include <blas.h>
#include <util.h>

#include <thrust/extrema.h> // for thrust::max_element

namespace amgx
{

template<typename IndexType, typename ValueTypeA, typename ValueTypeB, int threads_per_block, int warps_per_block, bool diag>
__global__
void getLambdaEstimate(const IndexType *row_offsets, const IndexType *column_indices, const ValueTypeA *values, const IndexType *dia_indices, const int num_rows, ValueTypeB *out)

{
    int row_id = blockDim.x * blockIdx.x + threadIdx.x;
    ValueTypeB max_sum = (ValueTypeB)0.0;

    while (row_id < num_rows)
    {
        ValueTypeB cur_sum = (ValueTypeB)0.0;

        for (int j = row_offsets[row_id]; j < row_offsets[row_id + 1]; j++)
        {
            cur_sum += abs(values[j]);
        }

        if (diag)
        {
            cur_sum += abs(values[dia_indices[row_id]]);
        }

        max_sum = max(max_sum, cur_sum);
        row_id += gridDim.x * blockDim.x;
    }

    out[blockDim.x * blockIdx.x + threadIdx.x] = max_sum;
}

// Method to compute the inverse of the diagonal blocks
template <class T_Config>
void Chebyshev_Solver<T_Config>::compute_eigenmax_estimate(const Matrix<T_Config> &A, ValueTypeB &lambda)
{
#define LAMBDA_BLOCK_SIZE 256
    VVector tsum(A.get_num_rows());
    const int threads_per_block = 256;
    const int blockrows_per_cta = threads_per_block;
    const int num_blocks = std::min(AMGX_GRID_MAX_SIZE, (int) (A.get_num_rows() - 1) / blockrows_per_cta + 1);
    const IndexType *A_row_offsets_ptr = A.row_offsets.raw();
    const IndexType *A_column_indices_ptr = A.col_indices.raw();
    const IndexType *A_dia_idx_ptr = A.diag.raw();
    const ValueTypeA *A_nonzero_values_ptr = A.values.raw();

    if (A.hasProps(DIAG))
    {
        cudaFuncSetCacheConfig(getLambdaEstimate < IndexType, ValueTypeA, ValueTypeB, LAMBDA_BLOCK_SIZE, LAMBDA_BLOCK_SIZE / 32, true >, cudaFuncCachePreferL1);
        getLambdaEstimate < IndexType, ValueTypeA, ValueTypeB, LAMBDA_BLOCK_SIZE, LAMBDA_BLOCK_SIZE / 32, true > <<< num_blocks, threads_per_block>>>
        (A_row_offsets_ptr, A_column_indices_ptr, A_nonzero_values_ptr, A_dia_idx_ptr, A.get_num_rows(), tsum.raw());
    }
    else
    {
        cudaFuncSetCacheConfig(getLambdaEstimate < IndexType, ValueTypeA, ValueTypeB, LAMBDA_BLOCK_SIZE, LAMBDA_BLOCK_SIZE / 32, false >, cudaFuncCachePreferL1);
        getLambdaEstimate < IndexType, ValueTypeA, ValueTypeB, LAMBDA_BLOCK_SIZE, LAMBDA_BLOCK_SIZE / 32, false > <<< num_blocks, threads_per_block>>>
        (A_row_offsets_ptr, A_column_indices_ptr, A_nonzero_values_ptr, A_dia_idx_ptr, A.get_num_rows(), tsum.raw());
    }

    lambda = *(thrust::max_element(tsum.begin(), tsum.end()));
}

// Constructor
template< class T_Config>
Chebyshev_Solver<T_Config>::Chebyshev_Solver( AMG_Config &cfg, const std::string &cfg_scope) :
    Solver<T_Config>( cfg, cfg_scope),
    m_buffer_N(0), m_eigsolver(NULL)
{
    std::string solverName, new_scope, tmp_scope;
    cfg.getParameter<std::string>( "preconditioner", solverName, cfg_scope, new_scope );
    m_lambda_mode = cfg.AMG_Config::getParameter<int>("chebyshev_lambda_estimate_mode", cfg_scope);
    m_cheby_order = cfg.AMG_Config::getParameter<int>("chebyshev_polynomial_order", cfg_scope);
    // 0 - use eigensolver to get BOTH estimates
    // 1 - use eigensolver to get maximum estimate
    // 2 - use max sum of abs values as a rough estimate for maximum eigenvalue
    // 3 - use user provided cheby_max_lambda and cheby_min_lambda

    if (m_lambda_mode == 3)
    {
        m_user_max_lambda = cfg.AMG_Config::getParameter<double>("cheby_max_lambda", cfg_scope);
        m_user_min_lambda = cfg.AMG_Config::getParameter<double>("cheby_min_lambda", cfg_scope);
    }

    if (solverName.compare("NOSOLVER") == 0)
    {
        no_preconditioner = true;
        m_preconditioner = NULL;
    }
    else
    {
        no_preconditioner = false;
        m_preconditioner = SolverFactory<T_Config>::allocate( cfg, cfg_scope, "preconditioner" );
    }

    std::string eig_cfg_string =  "algorithm=AGGREGATION,\n"
                                  "eig_solver=LANCZOS,\n"
                                  "verbosity_level=0,\n"
                                  "eig_max_iters=128,\n"
                                  "eig_tolerance=1e-4,\n"
                                  "eig_which=largest,\n"
                                  "eig_eigenvector=0,\n"
                                  "eig_eigenvector_solver=default";

    /*std::ifstream t("/home/marsaev/work/perforce/marsaev_sw/sw/gpgpu/amgx/amg/eigen_examples/POWER_ITERATION");
    std::stringstream buffer;
    buffer << t.rdbuf();
    std::string eig_cfg_string = buffer.str();*/

    if (m_lambda_mode < 2)
    {
        AMG_Configuration eig_cfg;
        eig_cfg.parseParameterString(eig_cfg_string.c_str());
        m_eigsolver = EigenSolverFactory<T_Config>::allocate(*eig_cfg.getConfigObject(), "default", "eig_solver");
    }
}

template<class T_Config>
Chebyshev_Solver<T_Config>::~Chebyshev_Solver()
{
    if (!no_preconditioner) { delete m_preconditioner; }

    if (!m_eigsolver) { delete m_eigsolver; }
}


template<class T_Config>
void
Chebyshev_Solver<T_Config>::solver_setup(bool reuse_matrix_structure)
{
    AMGX_CPU_PROFILER( "Chebyshev_Solver::solver_setup " );
    ViewType oldView = this->m_A->currentView();
    this->m_A->setViewExterior();

    // Setup the preconditionner
    if (!no_preconditioner)
    {
        m_preconditioner->setup(*this->m_A, reuse_matrix_structure);
    }

    // The number of elements in temporary vectors.
    this->m_buffer_N = static_cast<int>( this->m_A->get_num_cols() * this->m_A->get_block_dimy() );
    Matrix<T_Config> *mtx_A = dynamic_cast<Matrix<T_Config>*>(this->m_A);
    VVector eig_solver_t_x;

    // m_lambda_mode:
    // 0: use eigensolver to get lmin and lmax estimate
    // 1: use eigensolver to get lmax estimate, set lmin = lmax/8
    // 2: use max row sum as lmax estimate, set lmin = lmax/8
    if (m_lambda_mode < 2)
    {
        if (!no_preconditioner)
        {
            SolverOperator<T_Config> *MA = new SolverOperator<T_Config> (this->m_A, m_preconditioner);
            m_eigsolver->setup(*MA);
            m_eigsolver->solve(eig_solver_t_x);
            delete MA;
        }
        else
        {
            m_eigsolver->setup(*mtx_A);
            m_eigsolver->solve(eig_solver_t_x);
        }

        const std::vector<ValueTypeB> &lambdas = m_eigsolver->get_eigenvalues();
        this->m_lmax = lambdas[0] * 1.05;

        if (m_lambda_mode == 0)
        {
            this->m_lmin = lambdas[lambdas.size() - 1] * 0.95;
        }
        else
        {
            this->m_lmin = this->m_lmax * 0.125;
        }
    }
    else if (m_lambda_mode == 2)
    {
        if (no_preconditioner)
        {
            Matrix<T_Config> *pA = dynamic_cast< Matrix<T_Config>* > (this->m_A);
            compute_eigenmax_estimate(*pA, this->m_lmax);
            this->m_lmin = this->m_lmax * 0.125;
        }
        else
        {
            // assuming that this preconditioner would be good enough to reduce spectrum to the largest eigen value = 1.0
            this->m_lmax = 0.9;
            this->m_lmin = this->m_lmax * 0.125;
        }
    }
    else if (m_lambda_mode == 3)
    {
        if (no_preconditioner)
        {
            Matrix<T_Config> *pA = dynamic_cast< Matrix<T_Config>* > (this->m_A);
            compute_eigenmax_estimate(*pA, this->m_lmax);
            this->m_lmin = this->m_lmax * 0.125;
        }
        else
        {
            // Use user input estimates
            this->m_lmax = this->m_user_max_lambda;
            this->m_lmin = this->m_user_min_lambda;
        }
    }
    else
    {
        FatalError("Not supported chebyshev_lambda_estimate_mode.", AMGX_ERR_NOT_SUPPORTED_BLOCKSIZE);
    }

    // Allocate memory needed for iterating.
    m_p.resize( this->m_buffer_N );
    m_z.resize( this->m_buffer_N );
    m_Ap.resize( this->m_buffer_N );
    m_xp.resize( this->m_buffer_N );
    m_rp.resize( this->m_buffer_N );
    m_p.set_block_dimy(this->m_A->get_block_dimy());
    m_p.set_block_dimx(1);
    m_p.dirtybit = 1;
    m_p.delayed_send = 1;
    m_p.tag = this->tag * 100 + 1;
    m_Ap.set_block_dimy(this->m_A->get_block_dimy());
    m_Ap.set_block_dimx(1);
    m_Ap.dirtybit = 1;
    m_Ap.delayed_send = 1;
    m_Ap.tag = this->tag * 100 + 2;
    m_z.set_block_dimy(this->m_A->get_block_dimy());
    m_z.set_block_dimx(1);
    m_z.dirtybit = 1;
    m_z.delayed_send = 1;
    m_z.tag = this->tag * 100 + 3;
    m_xp.set_block_dimy(this->m_A->get_block_dimy());
    m_xp.set_block_dimx(1);
    m_xp.dirtybit = 1;
    m_xp.delayed_send = 1;
    m_xp.tag = this->tag * 100 + 4;
    this->m_A->setView(oldView);
}

template<class T_Config>
void
Chebyshev_Solver<T_Config>::solve_init( VVector &b, VVector &x, bool xIsZero )
{
    AMGX_CPU_PROFILER( "Chebyshev_Solver::solve_init " );
    Operator<T_Config> &A = *this->m_A;
    ViewType oldView = A.currentView();
    A.setViewExterior();
    int offset, size;
    A.getOffsetAndSizeForView(A.getViewExterior(), &offset, &size);

    // Run one iteration of preconditioner with zero initial guess
    if (no_preconditioner)
    {
        copy(*this->m_r, m_z, offset, size);
    }
    else
    {
        m_z.delayed_send = 1;
        this->m_r->delayed_send = 1;
        m_preconditioner->solve( *this->m_r, m_z, true );
        m_z.delayed_send = 1;
        this->m_r->delayed_send = 1;
    }

    // m_p - res after precond
    copy( m_z, m_p, offset, size );
    A.setView(oldView);
    m_gamma = 0.;
    m_beta = 0.;
    first_iter = 0;
}

template<class T_Config>
bool
Chebyshev_Solver<T_Config>::solve_iteration( VVector &b, VVector &x, bool xIsZero )
{
    AMGX_CPU_PROFILER( "Chebyshev_Solver::solve_iteration " );
    Operator<T_Config> &A = *this->m_A;
    ViewType oldView = A.currentView();
    A.setViewExterior();
    int offset, size;
    A.getOffsetAndSizeForView(A.getViewExterior(), &offset, &size);
    ValueTypeB a = (this->m_lmax + this->m_lmin) / 2;
    ValueTypeB c = (this->m_lmax - this->m_lmin) / 2;

    for (int i = 0; i < m_cheby_order; i++)
    {
        // apply precond
        if (no_preconditioner)
        {
            copy(*this->m_r, m_z, offset, size);
        }
        else
        {
            m_z.delayed_send = 1;
            this->m_r->delayed_send = 1;
            m_preconditioner->solve( *this->m_r, m_z, true );
            m_z.delayed_send = 1;
            this->m_r->delayed_send = 1;
        }

        if (first_iter == 0)
        {
            m_gamma = 1. / a;
            first_iter = 1;
        }
        else
        {
            m_beta = c * c * m_gamma * m_gamma / 4.;

            if (m_gamma != ValueTypeB(0) && (a - (m_beta / m_gamma)) != ValueTypeB(0))
            {
                m_gamma = 1. / (a - m_beta / m_gamma);
            }

            axpby( m_z, m_p, m_p, ValueTypeB( 1 ), m_beta, offset, size);
        }

        axpy( m_p, x, m_gamma, offset, size );
        this->compute_residual( b, x);
    }

    // Do we converge ?
    if ( this->m_monitor_convergence && this->compute_norm_and_converged() )
    {
        A.setView(oldView);
        return true;
    }

    // No convergence so far.
    A.setView(oldView);
    return !this->m_monitor_convergence;
}

template<class T_Config>
void
Chebyshev_Solver<T_Config>::solve_finalize( VVector &b, VVector &x )
{}

template<class T_Config>
void
Chebyshev_Solver<T_Config>::printSolverParameters() const
{
    if (!no_preconditioner)
    {
        std::cout << "preconditioner: " << this->m_preconditioner->getName()
                  << " with scope name: "
                  << this->m_preconditioner->getScope() << std::endl;
    }
}

/****************************************
 * Explict instantiations
 ***************************************/
#define AMGX_CASE_LINE(CASE) template class Chebyshev_Solver<TemplateMode<CASE>::Type>;
AMGX_FORALL_BUILDS(AMGX_CASE_LINE)
#undef AMGX_CASE_LINE

} // namespace amgx
