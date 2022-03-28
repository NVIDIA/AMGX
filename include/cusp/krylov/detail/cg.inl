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
void cg(LinearOperator& A,
        Vector& x,
        Vector& b)
{
    typedef typename LinearOperator::value_type   ValueType;

    cusp::default_monitor<ValueType> monitor(b);

    cusp::krylov::cg(A, x, b, monitor);
}

template <class LinearOperator,
          class Vector,
          class Monitor>
void cg(LinearOperator& A,
        Vector& x,
        Vector& b,
        Monitor& monitor)
{
    typedef typename LinearOperator::value_type   ValueType;
    typedef typename LinearOperator::memory_space MemorySpace;

    cusp::identity_operator<ValueType,MemorySpace> M(A.num_rows, A.num_cols);

    cusp::krylov::cg(A, x, b, monitor, M);
}

template <class LinearOperator,
          class Vector,
          class Monitor,
          class Preconditioner>
void cg(LinearOperator& A,
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
    cusp::array1d<ValueType,MemorySpace> z(N);
    cusp::array1d<ValueType,MemorySpace> r(N);
    cusp::array1d<ValueType,MemorySpace> p(N);
        
    // y <- Ax
    cusp::multiply(A, x, y);

    // r <- b - A*x
    blas::axpby(b, y, r, ValueType(1), ValueType(-1));
   
    // z <- M*r
    cusp::multiply(M, r, z);

    // p <- z
    blas::copy(z, p);
		
    // rz = <r^H, z>
    ValueType rz = blas::dotc(r, z);

    while (!monitor.finished(r))
    {
        // y <- Ap
        cusp::multiply(A, p, y);
        
        // alpha <- <r,z>/<y,p>
        ValueType alpha =  rz / blas::dotc(y, p);

        // x <- x + alpha * p
        blas::axpy(p, x, alpha);

        // r <- r - alpha * y		
        blas::axpy(y, r, -alpha);

        // z <- M*r
        cusp::multiply(M, r, z);
		
        ValueType rz_old = rz;

        // rz = <r^H, z>
        rz = blas::dotc(r, z);

        // beta <- <r_{i+1},r_{i+1}>/<r,r> 
        ValueType beta = rz / rz_old;
		
        // p <- r + beta*p
        blas::axpby(z, p, p, ValueType(1), beta);

        ++monitor;
    }
}

} // end namespace krylov
} // end namespace cusp

