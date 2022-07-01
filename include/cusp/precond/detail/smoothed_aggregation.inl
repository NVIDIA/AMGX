/*
 *  Copyright 2008-2009 NVIDIA Corporation
 *
 *  Licensed under the Apache License, Version 2.0 (the "License");
 *  you may not use this file except in compliance with the License.
 *  You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 *  Unless required by applicable law or agreed to in writing, software
 *  distributed under the License is distributed on an "AS IS" BASIS,
 *  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  See the License for the specific language governing permissions and
 *  limitations under the License.
 */

#include <cusp/blas.h>
#include <cusp/elementwise.h>
#include <cusp/multiply.h>
#include <cusp/monitor.h>
#include <cusp/transpose.h>
#include <cusp/graph/maximal_independent_set.h>
#include <cusp/precond/diagonal.h>
#include <cusp/precond/aggregate.h>
#include <cusp/precond/smooth.h>
#include <cusp/precond/strength.h>
#include <cusp/krylov/arnoldi.h>

#include <cusp/detail/format_utils.h>
#include <cusp/detail/spectral_radius.h>

#include <thrust/extrema.h>
#include <thrust/transform.h>
#include <thrust/gather.h>
#include <thrust/reduce.h>

namespace cusp
{
namespace precond
{
namespace detail
{


template <typename MatrixType>
struct Dinv_A : public cusp::linear_operator<typename MatrixType::value_type, typename MatrixType::memory_space>
{
    const MatrixType& A;
    const cusp::precond::diagonal<typename MatrixType::value_type, typename MatrixType::memory_space> Dinv;

    Dinv_A(const MatrixType& A)
        : A(A), Dinv(A),
          cusp::linear_operator<typename MatrixType::value_type, typename MatrixType::memory_space>(A.num_rows, A.num_cols, A.num_entries + A.num_rows)
          {}

    template <typename Array1, typename Array2>
    void operator()(const Array1& x, Array2& y) const
    {
        cusp::multiply(A,x,y);
        cusp::multiply(Dinv,y,y);
    }
};

template <typename MatrixType>
double estimate_rho_Dinv_A(const MatrixType& A)
{
    typedef typename MatrixType::value_type   ValueType;
    typedef typename MatrixType::memory_space MemorySpace;

    Dinv_A<MatrixType> Dinv_A(A);

    return cusp::detail::ritz_spectral_radius(Dinv_A, 8);
}


template <typename T>
struct square : thrust::unary_function<T,T>
{
    __host__ __device__
    T operator()(const T& x) { return x * x; }
};

template <typename T>
struct sqrt_functor : thrust::unary_function<T,T>
{
    __host__ __device__
    T operator()(const T& x) { return sqrt(x); }
};

template <typename Array1,
          typename Array2,
          typename IndexType, typename ValueType,
          typename Array3>
void fit_candidates(const Array1& aggregates,
                    const Array2& B,
                          cusp::csr_matrix<IndexType,ValueType,cusp::host_memory>& Q_,
                          Array3& R)
{
  CUSP_PROFILE_SCOPED();
  // TODO handle case w/ unaggregated nodes (marked w/ -1)
  IndexType num_aggregates = *thrust::max_element(aggregates.begin(), aggregates.end()) + 1;

  cusp::coo_matrix<IndexType,ValueType,cusp::host_memory> Q;
  Q.resize(aggregates.size(), num_aggregates, aggregates.size());
  R.resize(num_aggregates);

  // gather values into Q
  thrust::sequence(Q.row_indices.begin(), Q.row_indices.end());
  thrust::copy(aggregates.begin(), aggregates.end(), Q.column_indices.begin());
  thrust::copy(B.begin(), B.end(), Q.values.begin());
                        
  // compute norm over each aggregate
  {
    // compute Qt
    cusp::coo_matrix<IndexType,ValueType,cusp::host_memory> Qt;  cusp::transpose(Q, Qt);

    // compute sum of squares for each column of Q (rows of Qt)
    cusp::array1d<IndexType, cusp::host_memory> temp(num_aggregates);
    thrust::reduce_by_key(Qt.row_indices.begin(), Qt.row_indices.end(),
                          thrust::make_transform_iterator(Qt.values.begin(), square<ValueType>()),
                          temp.begin(),
                          R.begin());

    // compute square root of each column sum
    thrust::transform(R.begin(), R.end(), R.begin(), sqrt_functor<ValueType>());
  }

  Q_.resize(Q.num_rows, Q.num_cols, Q.num_entries);
  thrust::copy(Q.column_indices.begin(), Q.column_indices.end(), Q_.column_indices.begin());
  cusp::detail::indices_to_offsets(Q.row_indices, Q_.row_offsets);

  // rescale columns of Q
  thrust::transform(Q.values.begin(), Q.values.end(),
                    thrust::make_permutation_iterator(R.begin(), Q.column_indices.begin()),
                    Q_.values.begin(),
                    thrust::divides<ValueType>());
}

template <typename Array1,
          typename Array2,
          typename IndexType, typename ValueType,
          typename Array3>
void fit_candidates(const Array1& aggregates,
                    const Array2& B,
                          cusp::coo_matrix<IndexType,ValueType,cusp::device_memory>& Q,
                          Array3& R)
{
  CUSP_PROFILE_SCOPED();
  // TODO handle case w/ unaggregated nodes (marked w/ -1)
  IndexType num_aggregates = *thrust::max_element(aggregates.begin(), aggregates.end()) + 1;

  Q.resize(aggregates.size(), num_aggregates, aggregates.size());
  R.resize(num_aggregates);

  // gather values into Q
  thrust::sequence(Q.row_indices.begin(), Q.row_indices.end());
  thrust::copy(aggregates.begin(), aggregates.end(), Q.column_indices.begin());
  thrust::copy(B.begin(), B.end(), Q.values.begin());
                        
  // compute norm over each aggregate
  {
    // compute Qt
    cusp::coo_matrix<IndexType,ValueType,cusp::device_memory> Qt;  cusp::transpose(Q, Qt);

    // compute sum of squares for each column of Q (rows of Qt)
    cusp::array1d<IndexType, cusp::device_memory> temp(num_aggregates);
    thrust::reduce_by_key(Qt.row_indices.begin(), Qt.row_indices.end(),
                          thrust::make_transform_iterator(Qt.values.begin(), square<ValueType>()),
                          temp.begin(),
                          R.begin());

    // compute square root of each column sum
    thrust::transform(R.begin(), R.end(), R.begin(), sqrt_functor<ValueType>());
  }

  // rescale columns of Q
  thrust::transform(Q.values.begin(), Q.values.end(),
                    thrust::make_permutation_iterator(R.begin(), Q.column_indices.begin()),
                    Q.values.begin(),
                    thrust::divides<ValueType>());
}

template <typename Matrix>
void setup_level_matrix(Matrix& dst, Matrix& src)
{	dst.swap(src);	}

template <typename Matrix1, typename Matrix2>
void setup_level_matrix(Matrix1& dst, Matrix2& src)
{	
    dst = src;

    // save space by resizing src to nothing
    src.resize(0,0,0);
}

} // end namespace detail


template <typename IndexType, typename ValueType, typename MemorySpace>
template <typename MatrixType>
smoothed_aggregation<IndexType,ValueType,MemorySpace>::smoothed_aggregation(const MatrixType& A, const ValueType theta)
    : theta(theta)
{
  CUSP_PROFILE_SCOPED();

  levels.reserve(20); // avoid reallocations which force matrix copies

  levels.push_back(typename smoothed_aggregation<IndexType,ValueType,MemorySpace>::level());
  levels.back().A_ = A; // copy
  levels.back().B.resize(A.num_rows, ValueType(1.0));

  while (levels.back().A_.num_rows > 100)
    extend_hierarchy();

  // TODO make lu_solver accept sparse input
  cusp::array2d<ValueType,cusp::host_memory> coarse_dense(levels.back().A_);
  LU = cusp::detail::lu_solver<ValueType, cusp::host_memory>(coarse_dense);

  // Setup solve matrix for each level
  levels[0].A = A;
  for( size_t lvl = 1; lvl < levels.size(); lvl++ )
    detail::setup_level_matrix( levels[lvl].A, levels[lvl].A_ );
}

template <typename IndexType, typename ValueType, typename MemorySpace>
void smoothed_aggregation<IndexType,ValueType,MemorySpace>::extend_hierarchy(void)
{
  CUSP_PROFILE_SCOPED();

  cusp::array1d<IndexType,MemorySpace> aggregates;
  {
    // compute stength of connection matrix
    SetupMatrixType C;
    detail::symmetric_strength_of_connection(levels.back().A_, C, theta);

    // compute aggregates
    aggregates.resize(C.num_rows);
    cusp::blas::fill(aggregates,IndexType(0));
    detail::standard_aggregation(C, aggregates);
  }

  // compute spectral radius of diag(C)^-1 * C
  ValueType rho_DinvA = detail::estimate_rho_Dinv_A(levels.back().A_);

  SetupMatrixType P;
  cusp::array1d<ValueType,MemorySpace>  B_coarse;
  {
    // compute tenative prolongator and coarse nullspace vector
    SetupMatrixType 				T;
    detail::fit_candidates(aggregates, levels.back().B, T, B_coarse);
  
    // compute prolongation operator
    detail::smooth_prolongator(levels.back().A_, T, P, ValueType(4.0/3.0), rho_DinvA);  // TODO if C != A then compute rho_Dinv_C
  }

  // compute restriction operator (transpose of prolongator)
  SetupMatrixType R;
  cusp::transpose(P,R);

  // construct Galerkin product R*A*P
  SetupMatrixType RAP;
  {
    // TODO test speed of R * (A * P) vs. (R * A) * P
    SetupMatrixType AP;
    cusp::multiply(levels.back().A_, P, AP);
    cusp::multiply(R, AP, RAP);
  }

  #ifndef USE_POLY_SMOOTHER
  //  4/3 * 1/rho is a good default, where rho is the spectral radius of D^-1(A)
  ValueType omega = ValueType(4.0/3.0) / rho_DinvA;
  levels.back().smoother = cusp::relaxation::jacobi<ValueType, MemorySpace>(levels.back().A_, omega);
  #else
  cusp::array1d<ValueType,cusp::host_memory> coef;
  ValueType rho = cusp::detail::ritz_spectral_radius_symmetric(levels.back().A_, 8);
  cusp::relaxation::detail::chebyshev_polynomial_coefficients(rho,coef);
  levels.back().smoother = cusp::relaxation::polynomial<ValueType, MemorySpace>(levels.back().A_,coef);
  #endif

  levels.back().aggregates.swap(aggregates);
  detail::setup_level_matrix( levels.back().R, R );
  detail::setup_level_matrix( levels.back().P, P );
  levels.back().residual.resize(levels.back().A_.num_rows);

  //resize A_ on the finest level to save space
  if( levels.size() == 1 )
    levels[0].A_.resize(0,0,0);

  //std::cout << "omega " << omega << std::endl;

  levels.push_back(level());
  levels.back().A_.swap(RAP);
  levels.back().B.swap(B_coarse);
  levels.back().x.resize(levels.back().A_.num_rows);
  levels.back().b.resize(levels.back().A_.num_rows);
}

    
template <typename IndexType, typename ValueType, typename MemorySpace>
template <typename Array1, typename Array2>
void smoothed_aggregation<IndexType,ValueType,MemorySpace>::operator()(const Array1& b, Array2& x)
{
  CUSP_PROFILE_SCOPED();

  // perform 1 V-cycle
  _solve(b, x, 0);
}

template <typename IndexType, typename ValueType, typename MemorySpace>
template <typename Array1, typename Array2>
void smoothed_aggregation<IndexType,ValueType,MemorySpace>::solve(const Array1& b, Array2& x)
{
  CUSP_PROFILE_SCOPED();

  cusp::default_monitor<ValueType> monitor(b);

  solve(b, x, monitor);
}

template <typename IndexType, typename ValueType, typename MemorySpace>
template <typename Array1, typename Array2, typename Monitor>
void smoothed_aggregation<IndexType,ValueType,MemorySpace>::solve(const Array1& b, Array2& x, Monitor& monitor )
{
  CUSP_PROFILE_SCOPED();

  const size_t n = levels[0].A.num_rows;

  // use simple iteration
  cusp::array1d<ValueType,MemorySpace> update(n);
  cusp::array1d<ValueType,MemorySpace> residual(n);

  // compute initial residual
  cusp::multiply(levels[0].A, x, residual);
  cusp::blas::axpby(b, residual, residual, ValueType(1.0), ValueType(-1.0));

  while(!monitor.finished(residual))
  {   
      _solve(residual, update, 0); 

      // x += M * r
      cusp::blas::axpy(update, x, ValueType(1.0));

      // update residual
      cusp::multiply(levels[0].A, x, residual);
      cusp::blas::axpby(b, residual, residual, ValueType(1.0), ValueType(-1.0));
      ++monitor;
  }   
}

template <typename IndexType, typename ValueType, typename MemorySpace>
template <typename Array1, typename Array2>
void smoothed_aggregation<IndexType,ValueType,MemorySpace>
::_solve(const Array1& b, Array2& x, const size_t i)
{
  CUSP_PROFILE_SCOPED();

  if (i + 1 == levels.size())
  {
    // coarse grid solve
    // TODO streamline
    cusp::array1d<ValueType,cusp::host_memory> temp_b(b);
    cusp::array1d<ValueType,cusp::host_memory> temp_x(x.size());
    LU(temp_b, temp_x);
    x = temp_x;
  }
  else
  {
    // presmooth
    levels[i].smoother.presmooth(levels[i].A, b, x);

    // compute residual <- b - A*x
    cusp::multiply(levels[i].A, x, levels[i].residual);
    cusp::blas::axpby(b, levels[i].residual, levels[i].residual, ValueType(1.0), ValueType(-1.0));

    // restrict to coarse grid
    cusp::multiply(levels[i].R, levels[i].residual, levels[i + 1].b);

    // compute coarse grid solution
    _solve(levels[i + 1].b, levels[i + 1].x, i + 1);

    // apply coarse grid correction 
    cusp::multiply(levels[i].P, levels[i + 1].x, levels[i].residual);
    cusp::blas::axpy(levels[i].residual, x, ValueType(1.0));

    // postsmooth
    levels[i].smoother.postsmooth(levels[i].A, b, x);
  }
}

template <typename IndexType, typename ValueType, typename MemorySpace>
void smoothed_aggregation<IndexType,ValueType,MemorySpace>
::print( void )
{
	IndexType num_levels = levels.size();

	std::cout << "\tNumber of Levels:\t" << num_levels << std::endl;
	std::cout << "\tOperator Complexity:\t" << operator_complexity() << std::endl;
	std::cout << "\tGrid Complexity:\t" << grid_complexity() << std::endl;
	std::cout << "\tlevel\tunknowns\tnonzeros:\t" << std::endl;

	IndexType nnz = 0;

	for(size_t index = 0; index < levels.size(); index++)
		nnz += levels[index].A.num_entries;

	for(size_t index = 0; index < levels.size(); index++)
  	{
		double percent = (double)levels[index].A.num_entries / nnz;
		std::cout << "\t" << index << "\t" << levels[index].A.num_cols << "\t\t" \
              << levels[index].A.num_entries << " \t[" << 100*percent << "%]" \
              << std::endl;
	}
} 

template <typename IndexType, typename ValueType, typename MemorySpace>
double smoothed_aggregation<IndexType,ValueType,MemorySpace>
::operator_complexity( void )
{
	size_t nnz = 0;

	for(size_t index = 0; index < levels.size(); index++)
		nnz += levels[index].A.num_entries;

	return (double) nnz / (double) levels[0].A.num_entries;
} 

template <typename IndexType, typename ValueType, typename MemorySpace>
double smoothed_aggregation<IndexType,ValueType,MemorySpace>
::grid_complexity( void )
{
	size_t unknowns = 0;
	for(size_t index = 0; index < levels.size(); index++)
		unknowns += levels[index].A.num_rows;

	return (double) unknowns / (double) levels[0].A.num_rows;
} 

} // end namespace precond
} // end namespace cusp

