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

#include <solvers/cf_jacobi_solver.h>
#include <solvers/block_common_solver.h>
#include <gaussian_elimination.h>
#include <basic_types.h>
#include <cutil.h>
#include <util.h>
#include <string>
#include <miscmath.h>
#include <texture.h>
#include <amgx_cusparse.h>

#include <thrust/partition.h>

using namespace std;
namespace amgx
{
namespace cf_jacobi_solver
{

struct is_coarse
{
    __host__ __device__
    int operator()(const int &x)
    {
        return (int) (x == COARSE);
    }
};

struct is_fine
{
    __host__ __device__
    int operator()(const int &x)
    {
        return (int) (x == FINE);
    }
};

struct is_eq_minus_one
{
    __host__ __device__
    int operator()(const int &x)
    {
        return (int) (x == -1);
    }
};


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

template<typename T>
__device__ __forceinline__ T fmnaOp (T a, T b, T c)
{
    return -(a * b) + c;
}
template<typename T>
__device__ __forceinline__ T mulOp (T a, T b)
{
    return a * b;
}
template<typename T>
__device__ __forceinline__ T rcpOp (T a)
{
    return 1.0 / (isNotCloseToZero(a) ? a : epsilon(a));
}
template<typename T>
__device__ __forceinline__ T absOp (T a)
{
    return fabs(a);
}

// -----------------------------------
//  KERNELS
// -----------------------------------

template<typename IndexType, typename ValueTypeA, typename ValueTypeB>
__global__
void jacobi_zero_ini_masked_step(const int *row_ids, const int nrows_to_process, const ValueTypeA *Dinv, const ValueTypeB *b, const ValueTypeB relaxation_factor, ValueTypeB *x)
{
    int tid = blockDim.x * blockIdx.x + threadIdx.x;

    while (tid < nrows_to_process)
    {
        int i = row_ids[tid];
        ValueTypeB d = Dinv[i];
        d = isNotCloseToZero(d) ? d : epsilon(d);
        x[i] = relaxation_factor * b[i] / d;
        tid += (blockDim.x * gridDim.x);
    }
}

template<typename IndexType, typename ValueTypeA, typename ValueTypeB>
__global__
void jacobi_masked_step(const int *row_ids, const int nrows_to_process, const ValueTypeA *Dinv, const ValueTypeB *b, const ValueTypeB *y, const ValueTypeB relaxation_factor, ValueTypeB *x)
{
    int tid = blockDim.x * blockIdx.x + threadIdx.x;

    while (tid < nrows_to_process)
    {
        int i = row_ids[tid];
        ValueTypeB d = Dinv[i];
        d = isNotCloseToZero(d) ? d : epsilon(d);
        x[i] += relaxation_factor * (b[i] - y[i]) / d;
        tid += (blockDim.x * gridDim.x);
    }
}

__global__
void agg_write_agg(const int *agg_map, const int nrows, int *dst)
{
    int tid = blockDim.x * blockIdx.x + threadIdx.x;

    while (tid < nrows)
    {
        dst[agg_map[tid]] = tid;
        tid += (blockDim.x * gridDim.x);
    }
}


//--------------------------------
// Methods
//--------------------------------

// Constructor
template<class T_Config>
CFJacobiSolver_Base<T_Config>::CFJacobiSolver_Base( AMG_Config &cfg, const std::string &cfg_scope) : Solver<T_Config>( cfg, cfg_scope)
{
    weight = cfg.AMG_Config::getParameter<double>("relaxation_factor", cfg_scope);
    int param_mode = cfg.AMG_Config::getParameter<int>("cf_smoothing_mode", cfg_scope);

    switch (param_mode)
    {
        case 0:
            this->mode = CF_CF;
            break;

        case 1:
            this->mode = CF_FC;
            break;

        case 2:
            this->mode = CF_FCF;
            break;

        case 3:
            this->mode = CF_CFC;
            break;

        default:
            this->mode = CF_CF;
    }

    if (weight == 0)
    {
        weight = 1.;
        amgx_printf("Warning, setting weight to 1 instead of estimating largest_eigen_value in Block Jacobi smoother\n");
    }
}

// Destructor
template<class T_Config>
CFJacobiSolver_Base<T_Config>::~CFJacobiSolver_Base()
{
    this->Dinv.resize(0);
}

template<class T_Config>
void
CFJacobiSolver_Base<T_Config>::printSolverParameters() const
{
    std::cout << "relaxation_factor= " << this->weight << std::endl;
}

// Solver setup
template<class T_Config>
void
CFJacobiSolver_Base<T_Config>::solver_setup(bool reuse_matrix_structure)
{
    Matrix<T_Config> *A_as_matrix = dynamic_cast<Matrix<T_Config>*>(this->m_A);

    if (!A_as_matrix)
    {
        FatalError("CFJacobiSolver only works with explicit matrices", AMGX_ERR_INTERNAL);
    }

    computeDinv( *A_as_matrix );

    if ( A_as_matrix->getBlockFormat() != ROW_MAJOR )
    {
        FatalError(" CFJacobiSolver only supports row major format", AMGX_ERR_CONFIGURATION);
    }

    if (A_as_matrix->hasParameter("cf_map")) // classical case
    {
        IVector *cf_map = A_as_matrix->template getParameterPtr< IVector >("cf_map");
        int nrows = A_as_matrix->get_num_rows();
        this->num_coarse = thrust::count(cf_map->begin(), cf_map->end(), (int)COARSE);
        this->c_rows.resize(this->num_coarse);
        this->f_rows.resize(nrows - this->num_coarse);
        thrust::counting_iterator<int> zero(0);
        thrust::counting_iterator<int> zero_plus_nrows = zero + nrows;
        thrust::copy_if(zero, zero_plus_nrows, cf_map->begin(), this->c_rows.begin(), is_coarse());
        thrust::copy_if(zero, zero_plus_nrows, cf_map->begin(), this->f_rows.begin(), is_fine());
        cudaCheckError();
        // partitioning check
        /*
        {
          typedef typename TConfig::template setMemSpace<AMGX_host  >::Type TConfig_h;
          typedef typename TConfig_h::template setVecPrec<AMGX_vecInt>::Type ivec_value_type_h;
          typedef Vector<ivec_value_type_h> IVector_h;

          IVector_h cf_map_h = *cf_map;
          IVector_h cf_rows = this->cf_rows;
          for (int i = 0; i < cf_map_h.size(); i++)
          {

              if ((cf_map_h[cf_rows[i]] == FINE && i < this->num_coarse) || (cf_map_h[cf_rows[i]] == COARSE && i >= this->num_coarse))
                  printf("CFJ FAIL at i==%d: cf_rows[i]==%d, cf_map[row]==%d, num_coarse==%d\n", i, cf_rows[i], cf_map_h[cf_rows[i]], this->num_coarse);
          }
        }
        */
    }
    else if (A_as_matrix->hasParameter("aggregates_map")) // aggregation case
    {
        IVector *agg_map = A_as_matrix->template getParameterPtr< IVector >("aggregates_map");
        int agg_num = A_as_matrix->template getParameter< int >("aggregates_num");
        this->num_coarse = agg_num;
        const int threads_per_block = 256;
        const int num_blocks = min(AMGX_GRID_MAX_SIZE, (int) (A_as_matrix->get_num_rows() + threads_per_block - 1) / threads_per_block);
        int nrows = A_as_matrix->get_num_rows();
        this->c_rows.resize(this->num_coarse);
        this->f_rows.resize(nrows - this->num_coarse);
        IVector tmap(nrows, -1);
        // can use thrust permutation operator here.
        agg_write_agg <<< num_blocks, threads_per_block>>>(agg_map->raw(), agg_map->size(), this->c_rows.raw());
        cudaCheckError();
        agg_write_agg <<< num_blocks, threads_per_block>>>(this->c_rows.raw(), this->num_coarse, tmap.raw());
        cudaCheckError();
        thrust::counting_iterator<int> zero(0);
        thrust::counting_iterator<int> zero_plus_nrows = zero + nrows;
        thrust::copy_if(zero, zero_plus_nrows, tmap.begin(), this->f_rows.begin(), is_eq_minus_one());
        cudaCheckError();
    }
    else
    {
        FatalError("No info from AMG level was found for C-F separation, use different smoother or drink 1 beer", AMGX_ERR_BAD_PARAMETERS);
    }
}

//
template<class T_Config>
void
CFJacobiSolver_Base<T_Config>::solve_init( VVector &b, VVector &x, bool xIsZero )
{
}

// Solve one iteration
template<class T_Config>
bool
CFJacobiSolver_Base<T_Config>::solve_iteration( VVector &b, VVector &x, bool xIsZero )
{
    //bool done = false;
    Matrix<T_Config> *A_as_matrix = (Matrix<T_Config> *) this->m_A;
    int *smoothing_direction = A_as_matrix->template getParameterPtr<int> ("smoothing_direction");
    // smoothing direction == 0 : presmoothing
    // smoothing direction == 1 : postsmoothing
    SmoothingOrder current_order;

    switch (this->mode)
    {
        case CF_CF:
            current_order = (*smoothing_direction == 0) ? CF_CF : CF_FC;
            break;

        case CF_FC:
            current_order = (*smoothing_direction == 1) ? CF_CF : CF_FC;
            break;

        case CF_FCF:
            current_order = (*smoothing_direction == 0) ? CF_FCF : CF_CFC;
            break;

        case CF_CFC:
            current_order = (*smoothing_direction == 1) ? CF_FCF : CF_CFC;
            break;
    }

// no multi-gpu for now
    ViewType flags = OWNED;

    if (xIsZero) { x.dirtybit = 0; }

    if (A_as_matrix->get_block_dimx() == 1 && A_as_matrix->get_block_dimy() == 1)
    {
        if (xIsZero)
        {
            smooth_with_0_initial_guess_1x1(*A_as_matrix, b, x, current_order, flags);
        }
        else
        {
            smooth_1x1(*A_as_matrix, b, x, current_order, flags);
        }
    }
    else
    {
        FatalError("Unsupported block size for BlockJacobi_Solver", AMGX_ERR_NOT_SUPPORTED_BLOCKSIZE);
    }

    return this->converged( b, x );
}

template<class T_Config>
void
CFJacobiSolver_Base<T_Config>::solve_finalize( VVector &b, VVector &x )
{}

template<class T_Config>
void CFJacobiSolver_Base<T_Config>::computeDinv( Matrix<T_Config> &A)
{
    Matrix<T_Config> *A_as_matrix = (Matrix<T_Config> *) this->m_A;
    ViewType oldView = A.currentView();
    A.setView(A_as_matrix->getViewExterior());

    if (A.get_block_dimx() == 1 && A.get_block_dimy() == 1)
    {
        this->computeDinv_1x1(A);
    }

    A.setView(oldView);
}


// Method to compute the inverse of the diagonal blocks
template <AMGX_VecPrecision t_vecPrec, AMGX_MatPrecision t_matPrec, AMGX_IndPrecision t_indPrec>
void CFJacobiSolver<TemplateConfig<AMGX_device, t_vecPrec, t_matPrec, t_indPrec> >::computeDinv_1x1(const Matrix_d &A)
{
    Matrix_d *A_as_matrix = (Matrix_d *) this->m_A;
// supports both diag
    this->Dinv.resize(A.get_num_rows()*A.get_block_dimx()*A.get_block_dimy(), 0.0);

    if ( A_as_matrix->hasProps(DIAG) )
    {
        const int num_values = A_as_matrix->diagOffset() * A_as_matrix->get_block_size();
        thrust::copy( A_as_matrix->values.begin() + num_values, A_as_matrix->values.begin() + num_values + A_as_matrix->get_num_rows()*A_as_matrix->get_block_size(), this->Dinv.begin() );
        cudaCheckError();
    }
    else
    {
        find_diag( *A_as_matrix );
    }
}


// Method to compute the inverse of the diagonal blocks
template <AMGX_VecPrecision t_vecPrec, AMGX_MatPrecision t_matPrec, AMGX_IndPrecision t_indPrec>
void CFJacobiSolver<TemplateConfig<AMGX_host, t_vecPrec, t_matPrec, t_indPrec> >::computeDinv_1x1(const Matrix_h &A)
{
    // Do nothing
}


// Finding diag on device, CSR format
template <AMGX_VecPrecision t_vecPrec, AMGX_MatPrecision t_matPrec, AMGX_IndPrecision t_indPrec>
void CFJacobiSolver<TemplateConfig<AMGX_host, t_vecPrec, t_matPrec, t_indPrec> >::find_diag( const Matrix_h &A )
{
    //for each row
    for (int i = 0; i < A.get_num_rows(); i++)
    {
        //for each column
        for (int j = A.row_offsets[i]; j < A.row_offsets[i + 1]; j++)
        {
            if (A.col_indices[j] == i)
            {
                this->Dinv[i] = A.values[j];
                break;
            }

            if (j == A.row_offsets[i + 1] - 1)
            {
                FatalError("Could not find a diagonal value", AMGX_ERR_BAD_PARAMETERS);
            }
        }
    }
}


// Finding diag on device, CSR format
template <AMGX_VecPrecision t_vecPrec, AMGX_MatPrecision t_matPrec, AMGX_IndPrecision t_indPrec>
void CFJacobiSolver<TemplateConfig<AMGX_device, t_vecPrec, t_matPrec, t_indPrec> >::find_diag( const Matrix_d &A )
{
    AMGX_CPU_PROFILER( "JacobiSolver::find_diag " );
    const size_t THREADS_PER_BLOCK  = 128;
    const size_t NUM_BLOCKS = min(AMGX_GRID_MAX_SIZE, (int)ceil((ValueTypeB)A.get_num_rows() / (ValueTypeB)THREADS_PER_BLOCK));
    find_diag_kernel_indexed_dia <<< (unsigned int)NUM_BLOCKS, (unsigned int)THREADS_PER_BLOCK>>>(
        A.get_num_rows(),
        A.diag.raw(),
        A.values.raw(),
        this->Dinv.raw());
    cudaCheckError();
}

template <AMGX_VecPrecision t_vecPrec, AMGX_MatPrecision t_matPrec, AMGX_IndPrecision t_indPrec>
void CFJacobiSolver<TemplateConfig<AMGX_host, t_vecPrec, t_matPrec, t_indPrec> >::smooth_1x1(Matrix_h &A, VVector &b, VVector &x, SmoothingOrder order, ViewType separation_flags)
{
    VVector newx(x.size());

    //for each row
    for (int i = 0; i < A.get_num_rows(); i++)
    {
        ValueTypeB Axi = 0.0;
        ValueTypeB d = A.values[A.diag[i]];
        ValueTypeB mydiaginv = this->weight / (isNotCloseToZero(d) ? d : epsilon(d) );

        //for each column
        for (int j = A.row_offsets[i]; j < A.row_offsets[i + 1]; j++)
        {
            Axi += A.values[j] * x[A.col_indices[j]];
        }

        newx[i] = x[i] +  (b[i] - Axi) * mydiaginv ;
    }

    x.swap(newx);
}

template <AMGX_VecPrecision t_vecPrec, AMGX_MatPrecision t_matPrec, AMGX_IndPrecision t_indPrec>
void CFJacobiSolver<TemplateConfig<AMGX_device, t_vecPrec, t_matPrec, t_indPrec> >::smooth_1x1(Matrix_d &A, VVector &b, VVector &x, SmoothingOrder order, ViewType separation_flags)
{
    AMGX_CPU_PROFILER( "JacobiSolver::smooth_1x1 " );

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
    const int threads_per_block = 256;
    const int num_blocks = min(AMGX_GRID_MAX_SIZE, (int) (A.get_num_rows() + threads_per_block - 1) / threads_per_block);

    if (order == CF_CF || order == CF_FC)
    {
        IVector &rows_first = (order == CF_CF) ? this->c_rows : this->f_rows;
        IVector &rows_second = (order == CF_CF) ? this->f_rows : this->c_rows;
        const int threads_per_block = 256;
        const int num_blocks = min(AMGX_GRID_MAX_SIZE, (int) (A.get_num_rows() + threads_per_block - 1) / threads_per_block);
        // if use transform + permutation iterator it will yield into two separate permutation reads - for src and dst, so using simple kernel here
        multiply_masked( A, x, this->y, rows_first, separation_flags );
        jacobi_masked_step<IndexType, ValueTypeA, ValueTypeB> <<< num_blocks, threads_per_block>>>(
            rows_first.raw(),
            rows_first.size(),
            this->Dinv.raw(),
            b.raw(),
            this->y.raw(),
            this->weight,
            x.raw());
        cudaCheckError();
        multiply_masked( A, x, this->y, rows_second, separation_flags );
        jacobi_masked_step<IndexType, ValueTypeA, ValueTypeB> <<< num_blocks, threads_per_block>>>(
            rows_second.raw(),
            rows_second.size(),
            this->Dinv.raw(),
            b.raw(),
            this->y.raw(),
            this->weight,
            x.raw());
        cudaCheckError();
    }
    else
    {
        FatalError("CF_FCF and CF_CFC is not yet done", AMGX_ERR_NOT_IMPLEMENTED);
    }
}


template <AMGX_VecPrecision t_vecPrec, AMGX_MatPrecision t_matPrec, AMGX_IndPrecision t_indPrec>
void CFJacobiSolver<TemplateConfig<AMGX_host, t_vecPrec, t_matPrec, t_indPrec> >::smooth_with_0_initial_guess_1x1( Matrix_h &A, VVector &b, VVector &x, SmoothingOrder order, ViewType separation_flags)
{
    //for each row
    for (int i = 0; i < A.get_num_rows(); i++)
    {
        ValueTypeB d = A.values[A.diag[i]];
        ValueTypeB mydiag = this->weight / (isNotCloseToZero(d) ? d : epsilon(d));
        x[i] =  b[i] * mydiag;
    }
}

template <AMGX_VecPrecision t_vecPrec, AMGX_MatPrecision t_matPrec, AMGX_IndPrecision t_indPrec>
void CFJacobiSolver<TemplateConfig<AMGX_device, t_vecPrec, t_matPrec, t_indPrec> >::smooth_with_0_initial_guess_1x1(Matrix_d &A, VVector &b, VVector &x, SmoothingOrder order, ViewType separation_flags)
{
    AMGX_CPU_PROFILER( "JacobiSolver::smooth_with_0_initial_guess_1x1 " );
    int num_rows = A.get_num_rows();
    int offset = 0;
    A.getOffsetAndSizeForView(separation_flags, &offset, &num_rows);
    const int threads_per_block = 256;
    const int num_blocks = min(AMGX_GRID_MAX_SIZE, (int) (A.get_num_rows() + threads_per_block - 1) / threads_per_block);

    if (this->y.size() != b.size())
    {
        this->y.resize(b.size());
        this->y.tag = this->tag * 100 + 3;
        this->y.set_block_dimx(b.get_block_dimx());
        this->y.set_block_dimy(b.get_block_dimy());
    }

    if (order == CF_CF || order == CF_FC)
    {
//   IVector& rows_first = (order == CF_CF) ? this->c_rows : this->f_rows;
        IVector &rows_second = (order == CF_CF) ? this->f_rows : this->c_rows;
        const int threads_per_block = 256;
        const int num_blocks = min(AMGX_GRID_MAX_SIZE, (int) (A.get_num_rows() + threads_per_block - 1) / threads_per_block);
        /*jacobi_zero_ini_masked_step<IndexType, ValueTypeA, ValueTypeB><<<num_blocks,threads_per_block>>>(
          rows_first.raw(),
          rows_first.size(),
          this->Dinv.raw(),
          b.raw(),
          this->weight,
          x.raw());*/
        // it is not so much harder to initialize whole vector instead of just C or F points
        thrust::transform(b.begin( ),
                          b.begin( ) + A.get_num_rows(),
                          this->Dinv.begin( ),
                          x.begin( ),
                          jacobi_presmooth_functor<ValueTypeA, ValueTypeB>( this->weight ));
        cudaCheckError();
        multiply_masked( A, x, this->y, rows_second, separation_flags );
        cudaCheckError();
        jacobi_masked_step<IndexType, ValueTypeA, ValueTypeB> <<< num_blocks, threads_per_block>>>(
            rows_second.raw(),
            rows_second.size(),
            this->Dinv.raw(),
            b.raw(),
            this->y.raw(),
            this->weight,
            x.raw());
        cudaCheckError();
    }
    else
    {
        FatalError("CF_FCF and CF_CFC is not yet done", AMGX_ERR_NOT_IMPLEMENTED);
    }

    cudaCheckError();
}

/****************************************
 * Explict instantiations
 ***************************************/
#define AMGX_CASE_LINE(CASE) template class CFJacobiSolver_Base<TemplateMode<CASE>::Type>;
AMGX_FORALL_BUILDS(AMGX_CASE_LINE)
#undef AMGX_CASE_LINE

#define AMGX_CASE_LINE(CASE) template class CFJacobiSolver<TemplateMode<CASE>::Type>;
AMGX_FORALL_BUILDS(AMGX_CASE_LINE)
#undef AMGX_CASE_LINE

} // namespace block_jacobi
} // namespace amgx
