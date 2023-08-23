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

#include <solvers/chebyshev_poly.h>
#include <basic_types.h>
#include <cutil.h>
#include <util.h>
#include <string>
#include <miscmath.h>
#define _USE_MATH_DEFINES
#include "math.h"
#include <amgx_cublas.h>
#include <amgx_cusparse.h>

#include <thrust/extrema.h>

#undef DEBUG_CHEBYSHEV_OUTPUT

namespace amgx
{
namespace chebyshev_poly_smoother
{

template <typename ValueTypeA, typename ValueTypeB>
struct jacobi_presmooth_functor
{
    ValueTypeB omega;
    jacobi_presmooth_functor( ValueTypeB omega ) : omega( omega ) {}
    __host__ __device__ ValueTypeB operator()( const ValueTypeB &b, const ValueTypeA &d ) const { return isNotCloseToZero(d) ? omega * b / d : omega * b / epsilon(d); }
};

template <typename ValueTypeA, typename ValueTypeB>
struct jacobi_postsmooth_functor
{
    ValueTypeB omega;
    jacobi_postsmooth_functor( ValueTypeB omega ) : omega( omega ) {}
    template<typename Tuple> __host__ __device__  ValueTypeB operator( )( const Tuple &t ) const
    {
        ValueTypeB x = thrust::get<0>(t);
        ValueTypeA d = thrust::get<1>(t);
        ValueTypeB b = thrust::get<2>(t);
        ValueTypeB y = thrust::get<3>(t);
        // return x + omega * (b - y) / d.
        d = isNotCloseToZero(d) ? d :  epsilon(d);
        d  = ValueTypeA( 1 ) / d;
        b -= y;
        b *= omega;
        return b * d + x;
    }
};

template <typename ValueTypeB>
struct add_functor
{
    __host__ __device__  ValueTypeB operator()( const ValueTypeB &x, const ValueTypeB &y )const { return x + y; }
};


// -----------------------------------
//  KERNELS
// -----------------------------------

template <typename ValueTypeB>
__host__ __device__ ValueTypeB magicDampBeta(int m)
{
    return M_PI / (4 * (ValueTypeB)m + 2);
}

template <typename ValueTypeB>
__host__ __device__ ValueTypeB magicDamp(int n, ValueTypeB beta)
{
    return cos(beta) * cos(beta) / (cos(beta * (2 * n + 1)) * cos(beta * (2 * n + 1)) - sin(beta) * sin(beta));
}

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

//--------------------------------
// Methods
//--------------------------------

// Constructor
template<class T_Config>
ChebyshevPolySolver_Base<T_Config>::ChebyshevPolySolver_Base( AMG_Config &cfg, const std::string &cfg_scope) : Solver<T_Config>( cfg, cfg_scope)
{
    poly_order = cfg.AMG_Config::template getParameter<int>("chebyshev_polynomial_order", cfg_scope);
    poly_order = min(10, max(poly_order, 1));
    tau.resize(poly_order);
}

// Destructor
template<class T_Config>
ChebyshevPolySolver_Base<T_Config>::~ChebyshevPolySolver_Base()
{
    this->tau.resize(0);
}

template<class T_Config>
void
ChebyshevPolySolver_Base<T_Config>::printSolverParameters() const
{
    std::cout << "chebyshev_polynomial_order= " << this->poly_order << std::endl;
}

// Method to compute the inverse of the diagonal blocks
template <AMGX_VecPrecision t_vecPrec, AMGX_MatPrecision t_matPrec, AMGX_IndPrecision t_indPrec>
void ChebyshevPolySolver<TemplateConfig<AMGX_device, t_vecPrec, t_matPrec, t_indPrec> >::compute_eigenmax_estimate(const Matrix_d &A, ValueTypeB &lambda)
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
#ifdef DEBUG_CHEBYSHEV_OUTPUT
    printf("Lambda for A on row %lu is: %f\n", thrust::max_element(tsum.begin(), tsum.end()) - tsum.begin(), lambda);
#endif
}

// Method to compute the inverse of the diagonal blocks
template <AMGX_VecPrecision t_vecPrec, AMGX_MatPrecision t_matPrec, AMGX_IndPrecision t_indPrec>
void ChebyshevPolySolver<TemplateConfig<AMGX_host, t_vecPrec, t_matPrec, t_indPrec> >::compute_eigenmax_estimate(const Matrix_h &A, ValueTypeB &lambda)
{
    FatalError("compute_eigenmax_estimate is not implemented for host", AMGX_ERR_NOT_IMPLEMENTED);
}

// Solver setup
template<class T_Config>
void
ChebyshevPolySolver_Base<T_Config>::solver_setup(bool reuse_matrix_structure)
{
    Matrix<T_Config> *A_as_matrix = dynamic_cast<Matrix<T_Config>*>(this->m_A);

    if (!A_as_matrix)
    {
        FatalError("ChebyshevPolySolver only works with explicit matrices", AMGX_ERR_INTERNAL);
    }

    ValueTypeB lambda = 0.0;
    compute_eigenmax_estimate( *A_as_matrix, lambda );
    ValueTypeB beta = magicDampBeta<ValueTypeB>(poly_order);
#ifdef DEBUG_CHEBYSHEV_OUTPUT
    int lvl = 0;
    A_as_matrix->getParameter("level", lvl);
    printf("Tau values for level %d : ", lvl);
#endif

    for (int i = 0; i < poly_order; i++)
    {
        tau[i] = magicDamp(i, beta) / lambda;
#ifdef DEBUG_CHEBYSHEV_OUTPUT
        printf("%f%s", tau[i], (i == poly_order - 1) ? "\n" : " ");
#endif
    }
}

//
template<class T_Config>
void
ChebyshevPolySolver_Base<T_Config>::solve_init( VVector &b, VVector &x, bool xIsZero )
{
}

// Solve one iteration
template<class T_Config>
bool
ChebyshevPolySolver_Base<T_Config>::solve_iteration( VVector &b, VVector &x, bool xIsZero )
{
    Matrix<T_Config> *A_as_matrix = (Matrix<T_Config> *) this->m_A;

    if (xIsZero) { x.dirtybit = 0; }

    if (!A_as_matrix->is_matrix_singleGPU())
    {
        A_as_matrix->manager->exchange_halo_async(x, x.tag);

        if (A_as_matrix->getViewExterior() == A_as_matrix->getViewInterior())
        {
            A_as_matrix->manager->exchange_halo_wait(x, x.tag);
        }
    }

    ViewType oldView = A_as_matrix->currentView();
    ViewType flags;
    bool latencyHiding = true;

    if (A_as_matrix->is_matrix_singleGPU() || (x.dirtybit == 0))
    {
        latencyHiding = false;
        A_as_matrix->setViewExterior();
        flags = (ViewType)(A_as_matrix->getViewInterior() | A_as_matrix->getViewExterior());
    }
    else
    {
        flags = A_as_matrix->getViewInterior();
        A_as_matrix->setViewInterior();
    }

    if (A_as_matrix->get_block_dimx() == 1 && A_as_matrix->get_block_dimy() == 1)
    {
        smooth_1x1(*A_as_matrix, b, x, flags);
    }
    else
    {
        FatalError("Unsupported block size for BlockJacobi_Solver", AMGX_ERR_NOT_SUPPORTED_BLOCKSIZE);
    }

    if (latencyHiding)
    {
        A_as_matrix->manager->exchange_halo_wait(x, x.tag);
        A_as_matrix->setViewExterior();
        flags = (ViewType)(~(A_as_matrix->getViewInterior()) & A_as_matrix->getViewExterior());

        if (flags != 0)
        {
            if (A_as_matrix->get_block_dimx() == 1 && A_as_matrix->get_block_dimy() == 1)
            {
                smooth_1x1(*A_as_matrix, b, x, flags);
            }
            else
            {
                FatalError("Unsupported block size for BlockJacobi_Solver", AMGX_ERR_NOT_SUPPORTED_BLOCKSIZE);
            }
        }
    }

    x.dirtybit = 1;
    A_as_matrix->setView(oldView);
    return this->converged( b, x );
}

template<class T_Config>
void
ChebyshevPolySolver_Base<T_Config>::solve_finalize( VVector &b, VVector &x )
{}


template <AMGX_VecPrecision t_vecPrec, AMGX_MatPrecision t_matPrec, AMGX_IndPrecision t_indPrec>
void ChebyshevPolySolver<TemplateConfig<AMGX_host, t_vecPrec, t_matPrec, t_indPrec> >::smooth_1x1(Matrix_h &A, VVector &b, VVector &x, ViewType separation_flags)
{
    FatalError("chebyshev poly smoother not implemented with host format, exiting", AMGX_ERR_NOT_IMPLEMENTED);
}

template <typename ValueTypeB>
struct chebyshev_poly_functor
{
    ValueTypeB tau;
    chebyshev_poly_functor( ValueTypeB tau ) : tau( tau ) {}
    template<typename Tuple> __host__ __device__  ValueTypeB operator( )( const Tuple &t ) const
    {
        ValueTypeB x = thrust::get<0>(t);
        ValueTypeB b = thrust::get<1>(t);
        ValueTypeB y = thrust::get<2>(t);
        return x + tau * (b - y);
    }
};

template <AMGX_VecPrecision t_vecPrec, AMGX_MatPrecision t_matPrec, AMGX_IndPrecision t_indPrec>
void ChebyshevPolySolver<TemplateConfig<AMGX_device, t_vecPrec, t_matPrec, t_indPrec> >::smooth_1x1(Matrix_d &A, VVector &b, VVector &x, ViewType separation_flags)
{
    if (this->y.size() != b.size())
    {
        this->y.resize(b.size());
        this->y.tag = this->tag * 100 + 3;
        this->y.set_block_dimx(b.get_block_dimx());
        this->y.set_block_dimy(b.get_block_dimy());
    }

    int num_rows = A.get_num_rows();
    int offset = 0;
    A.getOffsetAndSizeForView(separation_flags, &offset, &num_rows);
    bool latencyHiding = separation_flags != A.getViewIntExt();// we change view only when do latency hiding, maybe it's better to use some explicit flag attached to matrix?

    for (int i = 0; i < this->poly_order; i++)
    {
        this->y.dirtybit = 0;
        //y = A*x
        multiply( A, x, this->y, separation_flags );
        //x += tau_i * (b - y)
        thrust::transform( thrust::make_zip_iterator(thrust::make_tuple( x.begin() + offset, b.begin() + offset, this->y.begin() + offset)),
                           thrust::make_zip_iterator(thrust::make_tuple( x.begin() + num_rows, b.begin() + num_rows, this->y.begin() + num_rows)),
                           x.begin() + offset,
                           chebyshev_poly_functor<ValueTypeB>( this->tau[i] ));
        //Cublas::axpy(num_rows, (ValueTypeB)(this->tau[i]),      b.raw(),       1, x.raw(), 1);
        //Cublas::axpy(num_rows, (ValueTypeB)(-1.0*this->tau[i]), this->y.raw(), 1, x.raw(), 1);
    }
}

template <AMGX_VecPrecision t_vecPrec, AMGX_MatPrecision t_matPrec, AMGX_IndPrecision t_indPrec>
void ChebyshevPolySolver<TemplateConfig<AMGX_host, t_vecPrec, t_matPrec, t_indPrec> >::smooth_with_0_initial_guess_1x1(const Matrix_h &A, const VVector &b, VVector &x, ViewType separation_flags)
{
    FatalError("chebyshev poly smoother not implemented with host format, exiting", AMGX_ERR_NOT_IMPLEMENTED);
}

template <AMGX_VecPrecision t_vecPrec, AMGX_MatPrecision t_matPrec, AMGX_IndPrecision t_indPrec>
void ChebyshevPolySolver<TemplateConfig<AMGX_device, t_vecPrec, t_matPrec, t_indPrec> >::smooth_with_0_initial_guess_1x1(const Matrix_d &A, const VVector &b, VVector &x, ViewType separation_flags)
{
}

template <AMGX_VecPrecision t_vecPrec, AMGX_MatPrecision t_matPrec, AMGX_IndPrecision t_indPrec>
void ChebyshevPolySolver<TemplateConfig<AMGX_host, t_vecPrec, t_matPrec, t_indPrec> >::smooth_BxB(Matrix_h &A, VVector &b, VVector &x, bool firstStep, ViewType separation_flags)
{
    FatalError("M*M chebyshev poly smoother not implemented with host format, exiting", AMGX_ERR_NOT_IMPLEMENTED);
}

template <AMGX_VecPrecision t_vecPrec, AMGX_MatPrecision t_matPrec, AMGX_IndPrecision t_indPrec>
void ChebyshevPolySolver<TemplateConfig<AMGX_device, t_vecPrec, t_matPrec, t_indPrec> >::smooth_BxB(Matrix_d &A, VVector &b, VVector &x, bool firstStep, ViewType separation_flags)
{
    FatalError("M*M chebyshev poly smoother not implemented with host format, exiting", AMGX_ERR_NOT_IMPLEMENTED);
}

/****************************************
 * Explict instantiations
 ***************************************/
#define AMGX_CASE_LINE(CASE) template class ChebyshevPolySolver_Base<TemplateMode<CASE>::Type>;
AMGX_FORALL_BUILDS(AMGX_CASE_LINE)
#undef AMGX_CASE_LINE

#define AMGX_CASE_LINE(CASE) template class ChebyshevPolySolver<TemplateMode<CASE>::Type>;
AMGX_FORALL_BUILDS(AMGX_CASE_LINE)
#undef AMGX_CASE_LINE

} // namespace chebyshev_poly_smoother
} // namespace amgx
