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

#include <eigensolvers/single_iteration_eigensolver.h>
#include <solvers/solver.h>
#include <blas.h>
#include <multiply.h>
#include <norm.h>
#include <transpose.h>
#include <operators/shifted_operator.h>
#include <operators/solve_operator.h>
#include <operators/pagerank_operator.h>
#include <algorithm>
#define AMGX_EXPLICIT_SHIFT 1
namespace amgx
{

template <typename index_type, typename value_type>
__global__ void shift_diagonal(index_type num_rows,
                               const index_type *row_offsets,
                               const index_type *col_indices,
                               value_type *values,
                               value_type shift)
{
    index_type tidx = blockDim.x * blockIdx.x + threadIdx.x;

    for (int r = tidx; r < num_rows; r += blockDim.x * gridDim.x)
    {
        index_type row_start = row_offsets[r];
        index_type row_end = row_offsets[r + 1];

        for (int j = row_start; j < row_end; j++)
        {
            if (col_indices[j] == r)
            {
                values[j] += shift;
                continue;
            }
        }
    }
}
// OBSOLETE compute a when the input is H only
template <typename index_type, typename value_type>
__global__ void dangling_nodes(index_type num_rows,
                               const index_type *row_offsets,
                               value_type *aa)
{
    index_type tidx = blockDim.x * blockIdx.x + threadIdx.x;

    for (int r = tidx; r < num_rows; r += blockDim.x * gridDim.x)
    {
        index_type row_start = row_offsets[r];
        index_type row_end = row_offsets[r + 1];

        // NOTE 1 : a = alpha*a + (1-alpha)e
        // NOTE 2 : a is initialized to (1-alpha)
        if (row_start == row_end)
        {
            aa[r] = 1.0;    // NOTE 3 : alpha*1 + (1-alpha)*1 = 1.0
        }
    }
}
// used when a is given as input and the matrix is H^T
template <typename index_type, typename value_type>
__global__ void update_a(index_type num_rows,
                         value_type *aa,
                         value_type beta)
{
    index_type tidx = blockDim.x * blockIdx.x + threadIdx.x;

    for (int r = tidx; r < num_rows; r += blockDim.x * gridDim.x)
    {
        // NOTE 1 : a = alpha*a + (1-alpha)e
        if (aa[r] == 0.0)
        {
            aa[r] = beta;    // NOTE 2 : alpha*0 + (1-alpha)*1 = (1-alpha)
        }
    }
}

template <class TConfig>
SingleIteration_EigenSolver<TConfig>::SingleIteration_EigenSolver(AMG_Config &cfg,
        const std::string &cfg_scope)
    : Base(cfg, cfg_scope), m_cfg(cfg), m_operator(NULL)
{
    m_convergence_check_freq = cfg.getParameter<int>("eig_convergence_check_freq", cfg_scope);
}

template <class TConfig>
SingleIteration_EigenSolver<TConfig>::~SingleIteration_EigenSolver()
{
    /*if (m_operator)
        delete m_operator;*/
    if (this->m_which == EIG_SMALLEST)
    {
        delete m_operator;
    }

    if (this->m_which == EIG_PAGERANK)
    {
        delete m_operator;
    }

    free_allocated();
}

template <class TConfig>
void SingleIteration_EigenSolver<TConfig>::free_allocated()
{
}

template <class TConfig>
void SingleIteration_EigenSolver<TConfig>::shift_matrix()
{
    ValueTypeMat shift = this->m_shift;

    if (shift == 0)
    {
        return;
    }

    Matrix<TConfig> *pA = dynamic_cast< Matrix<TConfig>* > (this->m_A);
    Matrix<TConfig> &A = *pA;
    int num_threads = 128;
    int max_grid_size = 4096;
    int num_rows = A.get_num_rows();
    int num_blocks = std::min(max_grid_size, (num_rows + num_threads - 1) / num_rows);
    shift_diagonal <<< num_blocks, num_threads>>>(num_rows, A.row_offsets.raw(),
            A.col_indices.raw(), A.values.raw(), -shift);
    cudaCheckError();
}
// OBSOLETE compute a when the input is H only
template <class TConfig>
void SingleIteration_EigenSolver<TConfig>::get_dangling_nodes()
{
    Matrix<TConfig> *pA = dynamic_cast< Matrix<TConfig>* > (this->m_A);
    Matrix<TConfig> &A = *pA;
    int num_threads = 128;
    int max_grid_size = 4096;
    int num_rows = A.get_num_rows();
    int num_blocks = std::min(max_grid_size, (num_rows + num_threads - 1) / num_rows);
    dangling_nodes <<< num_blocks, num_threads>>>(num_rows, A.row_offsets.raw(), m_a.raw());
    cudaCheckError();
    // CPU CODE : you shouldn't use it, it is very slow
    /* for (int i = 0; i < num_rows;  ++i)
    {
        if (A.row_offsets[i] == A.row_offsets[i+1])
            m_a[i] = 1.0; // alpha*1 + (1-alpha)*1 = 1.0
        else
            m_a[i] = beta;
    } */
}

// 0 are replaced by 1-alpha
template <class TConfig>
void SingleIteration_EigenSolver<TConfig>::update_dangling_nodes()
{
    Operator<TConfig> &A = *this->m_A;
    ValueTypeVec beta = 1.0 - this->m_damping_factor;
    int num_rows = A.get_num_rows();

    // CPU
    if (TConfig::memSpace == AMGX_host)
    {
        for (int i = 0; i < num_rows; i++)
        {
            // NOTE 1 : a = alpha*a + (1-alpha)e
            if (m_a[i] == 0.0)
            {
                m_a[i] = beta;    // NOTE 2 : alpha*0 + (1-alpha)*1 = (1-alpha)
            }
        }
    }
    //GPU
    else
    {
        int num_threads = 128;
        int max_grid_size = 4096;
        int num_blocks = std::min(max_grid_size, (num_rows + num_threads - 1) / num_rows);
        update_a <<< num_blocks, num_threads>>>(num_rows, m_a.raw(), beta );
        cudaCheckError();
    }
}

template <class TConfig>
void SingleIteration_EigenSolver<TConfig>::solver_setup()
{
    Operator<TConfig> &A = *this->m_A;
    ViewType oldView = A.currentView();
    A.setViewExterior();

    if (this->m_which == EIG_PAGERANK)
    {
        PagerankOperator<TConfig> *op = new PagerankOperator<TConfig>(A, &m_a, &m_b, this->m_damping_factor);
        m_operator = op;
    }
    else if (this->m_which == EIG_SMALLEST)
    {
        Solver<TConfig> *solver = SolverFactory<TConfig>::allocate(m_cfg, "default", "solver");
#ifdef AMGX_EXPLICIT_SHIFT
        shift_matrix();
        SolveOperator<TConfig> *solve_op = new SolveOperator<TConfig>(A, *solver);
#else
        ShiftedOperator<TConfig> *op = new ShiftedOperator<TConfig>(A, -this->m_shift);
        SolveOperator<TConfig> *solve_op = new SolveOperator<TConfig>(*op, *solver);
#endif
        solve_op->setup();
        m_operator = solve_op;
    }
    else
    {
        m_operator = &A;
    }

    const int N = static_cast<int>(A.get_num_cols() * A.get_block_dimy());
    // Allocate two vectors.
    m_v.resize(N);
    m_x.resize(N);
    m_allocated_vectors.push_back(&m_v);
    m_allocated_vectors.push_back(&m_x);

    // Vectors "a" and "b" are needed only for PR
    if (this->m_which == EIG_PAGERANK)
    {
        m_a.resize(N);
        m_b.resize(N);
        m_allocated_vectors.push_back(&m_a);
        m_allocated_vectors.push_back(&m_b);
    }

    int start_tag = 100;

    for (int i = 0; i < m_allocated_vectors.size(); ++i)
    {
        VVector *v = m_allocated_vectors[i];
        v->tag = start_tag + i;
        v->set_block_dimy(A.get_block_dimy());
        v->set_block_dimx(1);
        v->dirtybit = 1;
        v->delayed_send = 1;
    }

    A.setView(oldView);
}

template <class TConfig>
void SingleIteration_EigenSolver<TConfig>::solver_pagerank_setup(VVector &a)
{
    if (this->m_which == EIG_PAGERANK)
    {
        Matrix<TConfig> *pA = dynamic_cast< Matrix<TConfig>* > (this->m_A);
        Matrix<TConfig> &A = *pA;
        ViewType oldView = A.currentView();
        A.setViewExterior();
        int offset, size;
        A.getOffsetAndSizeForView(A.getViewExterior(), &offset, &size);
        copy(a, m_a, offset, size);
        //Get the number of rows of the matrix
        int num_rows_loc = A.get_num_rows();
        int num_rows_glob = num_rows_loc;
        // MPI?
#ifdef AMGX_WITH_MPI
        int mpi_initialized = 0;
        MPI_Initialized(&mpi_initialized);

        if (mpi_initialized)
        {
            if (A.is_matrix_distributed())
            {
                A.getManager()->global_reduce_sum(&num_rows_glob);
            }
        }

#endif
        // Requiered to compute b
        // a = alpha*a + (1-alpha)e
        update_dangling_nodes();
        // b is a constant and uniform vector
        ValueTypeMat tmp = 1.0 / num_rows_glob;
        fill(m_b, tmp);
        A.setView(oldView);
    }
}

template <class TConfig>
void SingleIteration_EigenSolver<TConfig>::solve_init(VVector &x)
{
    Operator<TConfig> &A = *this->m_A;
    ViewType oldView = A.currentView();
    A.setViewExterior();
    int offset, size;
    A.getOffsetAndSizeForView(A.getViewExterior(), &offset, &size);
    copy(x, m_x, offset, size);
    std::swap(m_x, m_v);
    A.setView(oldView);
}

// One iteration of the single iteration.
// MATLAB code of the algorithm:
// while true
//   V = X / norm(X);
//   X = linsolve(A, V);
//   lambda = V' * X;
//   R = X - lambda * V;
//   if norm(R) < tol * abs(lambda)
//     break;
//   end

template <class TConfig>
bool SingleIteration_EigenSolver<TConfig>::solve_iteration(VVector &x)
{
    Operator<TConfig> &A = *this->m_A;
    ViewType oldView = A.currentView();
    A.setViewExterior();
    int offset, size;
    A.getOffsetAndSizeForView(A.getViewExterior(), &offset, &size);
    // V = X / norm(X)
    scal(m_v, ValueTypeVec(1) / get_norm(A, m_v, this->m_norm_type), offset, size);
    // X = linsolve(A, v)
    m_operator->apply(m_v, m_x);

    if ((this->m_curr_iter % this->m_convergence_check_freq) == 0)
    {
        ValueTypeVec lambda;

        if (this->m_which == EIG_PAGERANK)
        {
            // The maximum eigenvalue of G is 1.0
            lambda = 1.0;
            // v = x - v
            // axpy(m_x, m_v, ValueTypeVec(-1),offset, size);
            axpby(m_v, m_x, m_v, ValueTypeVec(-1.0), ValueTypeVec(1), offset, size);
        }
        else
        {
            // lambda = v.x
            lambda = dot(A, m_v, m_x);
            // v = x - lambda * v
            axpby(m_v, m_x, m_v, -lambda, ValueTypeVec(1), offset, size);
        }

        ValueTypeVec residual_norm = get_norm(A, m_v, this->m_norm_type);
        this->m_residuals.push_back(residual_norm / fabs(lambda));

        // Check convergence.
        if (residual_norm < this->m_tolerance * fabs(lambda))
        {
            // Normalize eigenvector.
            if (this->m_which == EIG_PAGERANK)
            {
                //Norm L1 is more fited for the output of PageRank
                ValueTypeVec norm = get_norm(A, m_x, L1);
                scal(m_x, ValueTypeVec(1) / norm, offset, size);
            }
            else
            {
                ValueTypeVec norm = get_norm(A, m_x, this->m_norm_type);
                scal(m_x, ValueTypeVec(1) / norm, offset, size);
            }

            // With inverse iteration we need to scale the eigenvector by the inverse of the eigenvalue,
            // but this doesn't seems to be needed since we already normalized the eigenvector just above.
            this->m_eigenvectors.push_back(m_x);
            copy(m_x, x, offset, size);
            this->m_eigenvalues.push_back(lambda);
            this->postprocess_eigenpairs();
            return true;
        }
    }

    std::swap(m_x, m_v);
    A.setView(oldView);
    return false;
}

template <class TConfig>
void SingleIteration_EigenSolver<TConfig>::solve_finalize()
{
}

// Explicit template instantiation.
template class SingleIteration_EigenSolver<TConfigGeneric_d>;
template class SingleIteration_EigenSolver<TConfigGeneric_h>;

};
