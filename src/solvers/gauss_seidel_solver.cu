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

#include <solvers/gauss_seidel_solver.h>
#include <solvers/block_common_solver.h>
#include <blas.h>
#include <string.h>
#include <cutil.h>
#include <miscmath.h>

namespace amgx
{

// Constructor
template<class T_Config>
GaussSeidelSolver_Base<T_Config>::GaussSeidelSolver_Base( AMG_Config &cfg, const std::string &cfg_scope) : Solver<T_Config>( cfg, cfg_scope)
{
    this->weight = cfg.AMG_Config::getParameter<double>("relaxation_factor", cfg_scope);

    if (this->weight == 0)
    {
        this->weight = 1.;
    }
}

// Destructor
template<class T_Config>
GaussSeidelSolver_Base<T_Config>::~GaussSeidelSolver_Base()
{
}

template<class T_Config>
void
GaussSeidelSolver_Base<T_Config>::printSolverParameters() const
{
    std::cout << "relaxation_factor= " << this->weight << std::endl;
}

// Solver setup
template<class T_Config>
void
GaussSeidelSolver_Base<T_Config>::solver_setup(bool reuse_matrix_structure)
{
    Matrix<T_Config> *A_as_matrix = dynamic_cast<Matrix<T_Config>*>(this->m_A);

    if (!A_as_matrix)
    {
        FatalError("GaussSeidelSolver only works with explicit matrices", AMGX_ERR_INTERNAL);
    }

    if ( A_as_matrix->get_block_size() != 1 )
    {
        FatalError("Unsupported block size for GaussSeidelSolver", AMGX_ERR_NOT_SUPPORTED_BLOCKSIZE);
    }

    this->diag.resize((int) A_as_matrix->get_num_rows());

    if (A_as_matrix->hasProps(DIAG))
    {
        amgx::thrust::copy( A_as_matrix->values.begin() + A_as_matrix->diagOffset()*A_as_matrix->get_block_size()/*block_size == 1*/, A_as_matrix->values.end(), this->diag.begin());
        cudaCheckError();
    }
    else
    {
        find_diag(*A_as_matrix);
    }
}

//
template<class T_Config>
void
GaussSeidelSolver_Base<T_Config>::solve_init( VVector &b, VVector &x, bool xIsZero )
{
}


// Solve one iteration
template<class T_Config>
bool
GaussSeidelSolver_Base<T_Config>::solve_iteration( VVector &b, VVector &x, bool xIsZero )
{
    Matrix<T_Config> *A_as_matrix = (Matrix<T_Config> *) this->m_A;

    if ( A_as_matrix->get_block_size() == 1)
    {
        if (xIsZero)
        {
            smooth_with_0_initial_guess_1x1( *A_as_matrix, b, x );
        }
        else
        {
            smooth_1x1( *A_as_matrix, b, x );
        }
    }
    else
    {
        FatalError("Unsupported block size for GaussSeidel_Solver", AMGX_ERR_NOT_SUPPORTED_BLOCKSIZE);
    }

    return this->converged( b, x );
}

template<class T_Config>
void
GaussSeidelSolver_Base<T_Config>::solve_finalize( VVector &b, VVector &x )
{
}

template<class T_Config>
void
GaussSeidelSolver_Base<T_Config>::find_diag( const Matrix<TConfig> &in_A )
{
    if (in_A.get_block_size() == 1)
    {
        find_diag_1x1(in_A);
    }
    else
    {
        FatalError("Unsupported block size for GaussSeidelSolver", AMGX_ERR_NOT_SUPPORTED_BLOCKSIZE);
    }
}

/********************************************/

// Finding diag on host, CSR format
template <AMGX_VecPrecision t_vecPrec, AMGX_MatPrecision t_matPrec, AMGX_IndPrecision t_indPrec>
void GaussSeidelSolver<TemplateConfig<AMGX_host, t_vecPrec, t_matPrec, t_indPrec> >::find_diag_1x1(const Matrix_h &A)
{
    //for each row
    for (int i = 0; i < A.get_num_rows(); i++)
    {
        //for each column
        for (int j = A.row_offsets[i]; j < A.row_offsets[i + 1]; j++)
        {
            if (A.col_indices[j] == i)
            {
                this->diag[i] = /*static_cast<ValueTypeB>(1) /*/ A.values[j];
                break;
            }

            if (j == A.row_offsets[i + 1] - 1)
            {
                std::string error = "Could not find a diagonal value at row " + i;
                FatalError(error.c_str(), AMGX_ERR_BAD_PARAMETERS);
            }
        }
    }
}

template <AMGX_VecPrecision t_vecPrec, AMGX_MatPrecision t_matPrec, AMGX_IndPrecision t_indPrec>
void GaussSeidelSolver<TemplateConfig<AMGX_host, t_vecPrec, t_matPrec, t_indPrec> >::smooth_1x1(const Matrix_h &A, const VVector &b, VVector &x)
{
    //for each row
    for (int i = 0; i < A.get_num_rows(); i++)
    {
        ValueTypeB Axi = 0.0;
        ValueTypeA diag = isNotCloseToZero(this->diag[i]) ? this->diag[i] : epsilon(this->diag[i]);

        //for each column
        for (int j = A.row_offsets[i]; j < A.row_offsets[i + 1]; j++)
        {
            Axi += A.values[j] * x[A.col_indices[j]];
        }

        x[i] = x[i] + this->weight * (b[i] - Axi) / diag;
    }
}

template <AMGX_VecPrecision t_vecPrec, AMGX_MatPrecision t_matPrec, AMGX_IndPrecision t_indPrec>
void GaussSeidelSolver<TemplateConfig<AMGX_host, t_vecPrec, t_matPrec, t_indPrec> >::smooth_with_0_initial_guess_1x1(const Matrix_h &A, const VVector &b, VVector &x)
{
    //for each row
    for (int i = 0; i < A.get_num_rows(); i++)
    {
        ValueTypeA diag = isNotCloseToZero(this->diag[i]) ? this->diag[i] : epsilon(this->diag[i]);
        x[i] = this->weight * b[i] / diag;
    }
}

template <AMGX_VecPrecision t_vecPrec, AMGX_MatPrecision t_matPrec, AMGX_IndPrecision t_indPrec>
void GaussSeidelSolver<TemplateConfig<AMGX_device, t_vecPrec, t_matPrec, t_indPrec> >::find_diag_1x1(const Matrix_d &A)
{
    typedef typename Matrix_d::index_type IndexType;
    typedef typename Matrix_d::value_type ValueType;
    const size_t THREADS_PER_BLOCK  = 128;
    const size_t NUM_BLOCKS = min(AMGX_GRID_MAX_SIZE, (int)ceil((ValueType)A.get_num_rows() / (ValueType)THREADS_PER_BLOCK));
    this->diag.resize(A.get_num_rows());
    find_diag_kernel<IndexType, ValueType> <<< (unsigned int)NUM_BLOCKS, (unsigned int)THREADS_PER_BLOCK >>>
    ((int)A.get_num_rows(),
     A.row_offsets.raw(),
     A.col_indices.raw(),
     A.values.raw(),
     this->diag.raw());
    cudaCheckError();
}

template<typename IndexType, typename ValueTypeA,  typename ValueTypeB>
__global__ void GS_smooth_kernel(const IndexType num_rows,
                                 const IndexType *Ap,
                                 const IndexType *Aj,
                                 const ValueTypeA *Ax,
                                 const ValueTypeB *diag,
                                 const ValueTypeB *b,
                                 const ValueTypeB weight,
                                 ValueTypeB *x)

{
    IndexType tidx = blockDim.x * blockIdx.x + threadIdx.x;

    for (int ridx = tidx; ridx < num_rows; ridx += blockDim.x * gridDim.x)
    {
        IndexType row_start = Ap[ridx];
        IndexType row_end   = Ap[ridx + 1];
        ValueTypeB Axi = 0.0;
        ValueTypeA mydiag = isNotCloseToZero(diag[ridx]) ? diag[ridx] : epsilon(diag[ridx]);

        for (int j = row_start; j < row_end; j++)
        {
            Axi += Ax[j] * x[Aj[j]];
        }

        ValueTypeB tmp = x[ridx] + weight * (b[ridx] - Axi) / mydiag;
        x[ridx] = tmp;
    }
}



template <AMGX_VecPrecision t_vecPrec, AMGX_MatPrecision t_matPrec, AMGX_IndPrecision t_indPrec>
void GaussSeidelSolver<TemplateConfig<AMGX_device, t_vecPrec, t_matPrec, t_indPrec> >::smooth_1x1(const Matrix_d &A, const VVector &b, VVector &x)
{
    typedef typename Matrix_d::index_type IndexType;
    typedef typename Matrix_d::value_type ValueTypeA;
    typedef typename VVector::value_type ValueTypeB;
    const size_t THREADS_PER_BLOCK  = 128;
    const size_t NUM_BLOCKS = min(AMGX_GRID_MAX_SIZE, (int)ceil((ValueTypeB)A.get_num_rows() / (ValueTypeB)THREADS_PER_BLOCK));
    GS_smooth_kernel<IndexType, ValueTypeA, ValueTypeB> <<< (unsigned int)NUM_BLOCKS, (unsigned int)THREADS_PER_BLOCK >>>
    ((int)A.get_num_rows(),
     A.row_offsets.raw(),
     A.col_indices.raw(),
     A.values.raw(),
     this->diag.raw(),
     b.raw(),
     this->weight,
     x.raw());
    cudaCheckError();
}

template <AMGX_VecPrecision t_vecPrec, AMGX_MatPrecision t_matPrec, AMGX_IndPrecision t_indPrec>
void GaussSeidelSolver<TemplateConfig<AMGX_device, t_vecPrec, t_matPrec, t_indPrec> >::smooth_with_0_initial_guess_1x1(const Matrix_d &A, const VVector &b, VVector &x)
{
    amgx::thrust::fill(x.begin(), x.end(), ValueTypeB(0));
    cudaCheckError();
    smooth_1x1(A, b, x);
}

/****************************************
 * Explict instantiations
 ***************************************/
#define AMGX_CASE_LINE(CASE) template class GaussSeidelSolver_Base<TemplateMode<CASE>::Type>;
AMGX_FORALL_BUILDS(AMGX_CASE_LINE)
#undef AMGX_CASE_LINE

#define AMGX_CASE_LINE(CASE) template class GaussSeidelSolver<TemplateMode<CASE>::Type>;
AMGX_FORALL_BUILDS(AMGX_CASE_LINE)
#undef AMGX_CASE_LINE

} // namespace amgx
