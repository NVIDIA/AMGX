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

/*! \file smoothed_aggregation.h
 *  \brief Algebraic multigrid preconditoner based on smoothed aggregation.
 *  
 */

#pragma once

#include <cusp/detail/config.h>

#include <vector> // TODO replace with host_vector
#include <cusp/linear_operator.h>

#include <cusp/coo_matrix.h>
#include <cusp/csr_matrix.h>
#include <cusp/hyb_matrix.h>
#include <cusp/relaxation/jacobi.h>
#include <cusp/relaxation/polynomial.h>

#include <cusp/detail/lu.h>

namespace cusp
{
namespace precond
{

/*! \addtogroup preconditioners Preconditioners
 *  \ingroup preconditioners
 *  \{
 */

template <typename IndexType, typename ValueType, typename MemorySpace>
struct amg_container
{
};

template <typename IndexType, typename ValueType>
struct amg_container<IndexType,ValueType,cusp::host_memory>
{
    // use CSR on host
    typedef typename cusp::csr_matrix<IndexType,ValueType,cusp::host_memory> setup_type;
    typedef typename cusp::csr_matrix<IndexType,ValueType,cusp::host_memory> solve_type;
};

template <typename IndexType, typename ValueType>
struct amg_container<IndexType,ValueType,cusp::device_memory>
{
    // use COO on device
    typedef typename cusp::coo_matrix<IndexType,ValueType,cusp::device_memory> setup_type;
    typedef typename cusp::hyb_matrix<IndexType,ValueType,cusp::device_memory> solve_type;
};

/*! \p smoothed_aggregation : algebraic multigrid preconditoner based on
 *  smoothed aggregation
 *
 *  TODO
 */
template <typename IndexType, typename ValueType, typename MemorySpace>
class smoothed_aggregation : public cusp::linear_operator<ValueType, MemorySpace, IndexType>
{

    typedef typename amg_container<IndexType,ValueType,MemorySpace>::setup_type SetupMatrixType;
    typedef typename amg_container<IndexType,ValueType,MemorySpace>::solve_type SolveMatrixType;

    struct level
    {
        SetupMatrixType A_; // matrix
        SolveMatrixType R;  // restriction operator
        SolveMatrixType A;  // matrix
        SolveMatrixType P;  // prolongation operator
        cusp::array1d<IndexType,MemorySpace> aggregates;      // aggregates
        cusp::array1d<ValueType,MemorySpace> B;               // near-nullspace candidates
        cusp::array1d<ValueType,MemorySpace> x;               // per-level solution
        cusp::array1d<ValueType,MemorySpace> b;               // per-level rhs
        cusp::array1d<ValueType,MemorySpace> residual;        // per-level residual
        
	#ifndef USE_POLY_SMOOTHER
        cusp::relaxation::jacobi<ValueType,MemorySpace> smoother;
	#else
        cusp::relaxation::polynomial<ValueType,MemorySpace> smoother;
	#endif
    };

    std::vector<level> levels;
        
    cusp::detail::lu_solver<ValueType, cusp::host_memory> LU;

    ValueType theta;

    public:

    template <typename MatrixType>
    smoothed_aggregation(const MatrixType& A, const ValueType theta=0);

    
    template <typename Array1, typename Array2>
    void operator()(const Array1& x, Array2& y);

    template <typename Array1, typename Array2>
    void solve(const Array1& b, Array2& x);

    template <typename Array1, typename Array2, typename Monitor>
    void solve(const Array1& b, Array2& x, Monitor& monitor);

    void print( void );

    double operator_complexity( void );

    double grid_complexity( void );

    protected:

    void extend_hierarchy(void);

    template <typename Array1, typename Array2>
    void _solve(const Array1& b, Array2& x, const size_t i);
};
/*! \}
 */

} // end namespace precond
} // end namespace cusp

#include <cusp/precond/detail/smoothed_aggregation.inl>

