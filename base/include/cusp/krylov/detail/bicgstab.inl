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


#include <cusp/array1d.h>
#include <cusp/blas.h>
#include <cusp/multiply.h>
#include <cusp/monitor.h>
#include <cusp/linear_operator.h>

namespace blas = cusp::blas;

namespace cusp
{
namespace krylov
{

template <class LinearOperator,
          class Vector>
void bicgstab(LinearOperator& A,
              Vector& x,
              Vector& b)
{
    typedef typename LinearOperator::value_type   ValueType;

    cusp::default_monitor<ValueType> monitor(b);

    cusp::krylov::bicgstab(A, x, b, monitor);
}

template <class LinearOperator,
          class Vector,
          class Monitor>
void bicgstab(LinearOperator& A,
              Vector& x,
              Vector& b,
              Monitor& monitor)
{
    typedef typename LinearOperator::value_type   ValueType;
    typedef typename LinearOperator::memory_space MemorySpace;

    cusp::identity_operator<ValueType,MemorySpace> M(A.num_rows, A.num_cols);

    cusp::krylov::bicgstab(A, x, b, monitor, M);
}

template <class LinearOperator,
          class Vector,
          class Monitor,
          class Preconditioner>
void bicgstab(LinearOperator& A,
              Vector& x,
              Vector& b,
              Monitor& monitor,
              Preconditioner& M)
{
    CUSP_PROFILE_SCOPED();

    typedef typename LinearOperator::value_type   ValueType;
    typedef typename LinearOperator::memory_space MemorySpace;

    assert(A.num_rows == A.num_cols);        // sanity check

    const size_t N = A.num_rows;

    // allocate workspace
    cusp::array1d<ValueType,MemorySpace> y(N);

    cusp::array1d<ValueType,MemorySpace>   p(N);
    cusp::array1d<ValueType,MemorySpace>   r(N);
    cusp::array1d<ValueType,MemorySpace>   r_star(N);
    cusp::array1d<ValueType,MemorySpace>   s(N);
    cusp::array1d<ValueType,MemorySpace>  Mp(N);
    cusp::array1d<ValueType,MemorySpace> AMp(N);
    cusp::array1d<ValueType,MemorySpace>  Ms(N);
    cusp::array1d<ValueType,MemorySpace> AMs(N);

    // y <- Ax
    cusp::multiply(A, x, y);

    // r <- b - A*x
    blas::axpby(b, y, r, ValueType(1), ValueType(-1));

    // p <- r
    blas::copy(r, p);

    // r_star <- r
    blas::copy(r, r_star);

    ValueType r_r_star_old = blas::dotc(r_star, r);

    while (!monitor.finished(r))
    {
        // Mp = M*p
        cusp::multiply(M, p, Mp);

        // AMp = A*Mp
        cusp::multiply(A, Mp, AMp);

        // alpha = (r_j, r_star) / (A*M*p, r_star)
        ValueType alpha = r_r_star_old / blas::dotc(r_star, AMp);
        
        // s_j = r_j - alpha * AMp
        blas::axpby(r, AMp, s, ValueType(1), ValueType(-alpha));

	if (monitor.finished(s)){
	  // x += alpha*M*p_j
	  blas::axpby(x, Mp, x, ValueType(1), ValueType(alpha));
	  break;
	}

        // Ms = M*s_j
        cusp::multiply(M, s, Ms);
        
        // AMs = A*Ms
        cusp::multiply(A, Ms, AMs);

        // omega = (AMs, s) / (AMs, AMs)
        ValueType omega = blas::dotc(AMs, s) / blas::dotc(AMs, AMs);
        
        // x_{j+1} = x_j + alpha*M*p_j + omega*M*s_j
        blas::axpbypcz(x, Mp, Ms, x, ValueType(1), alpha, omega);

        // r_{j+1} = s_j - omega*A*M*s
        blas::axpby(s, AMs, r, ValueType(1), -omega);

        // beta_j = (r_{j+1}, r_star) / (r_j, r_star) * (alpha/omega)
        ValueType r_r_star_new = blas::dotc(r_star, r);
        ValueType beta = (r_r_star_new / r_r_star_old) * (alpha / omega);
        r_r_star_old = r_r_star_new;

        // p_{j+1} = r_{j+1} + beta*(p_j - omega*A*M*p)
        blas::axpbypcz(r, p, AMp, p, ValueType(1), beta, -beta*omega);

        ++monitor;
    }
}

} // end namespace krylov
} // end namespace cusp

