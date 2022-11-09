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

#include <solvers/kpz_polynomial_solver.h>
#include <blas.h>
#include <string.h>
#include <cutil.h>
#include <multiply.h>
#include <miscmath.h>
#include <transpose.h>
#include <thrust_wrapper.h>

#include <thrust/transform_reduce.h>

namespace amgx
{

namespace detail
{

template< typename CsrMatrix >
struct row_sum
{
    typedef typename CsrMatrix::index_type IndexType;
    typedef typename CsrMatrix::value_type ValueType;

    // We keep a view of the offsets and values.
    const IndexType *offsets_;
    const ValueType *values_;

    // Constructor.
    row_sum( const IndexType *offsets, const ValueType *values ) : offsets_( offsets ), values_( values )
    {}

    // Sum the values in a given range.
    __device__ __host__
    ValueType operator( )( IndexType r ) const
    {
        ValueType value( 0 );

        for ( IndexType i = offsets_[r], end = offsets_[r + 1] ; i < end ; ++i )
        {
            ValueType v = values_[i];
            value += v < ValueType( 0 ) ? -v : v; // abs( v ).
        }

        return value;
    }
};

}

// Constructor
template<class T_Config>
KPZPolynomialSolver<T_Config>::KPZPolynomialSolver( AMG_Config &cfg, const std::string &cfg_scope) : Solver<T_Config>( cfg, cfg_scope)
{
    l_inf = 0;
    mu = cfg.AMG_Config::getParameter<IndexType>("kpz_mu", cfg_scope);
    poly_order = cfg.AMG_Config::getParameter<IndexType>("kpz_order", cfg_scope);
}

// Destructor
template<class T_Config>
KPZPolynomialSolver<T_Config>::~KPZPolynomialSolver()
{
}

template<class T_Config>
void
KPZPolynomialSolver<T_Config>::printSolverParameters() const
{
    std::cout << "kpz_mu = " << this->mu << std::endl;
    std::cout << "kpz_order = " << this->poly_order << std::endl;
}

// Solver setup
template<class T_Config>
void
KPZPolynomialSolver<T_Config>::solver_setup(bool reuse_matrix_structure)
{
    m_explicit_A = dynamic_cast<Matrix<T_Config>*>(Base::m_A);

    if (!m_explicit_A)
    {
        FatalError("KPZPolynomialSolver only works with explicit matrices", AMGX_ERR_INTERNAL);
    }

    if (this->m_explicit_A->hasProps(DIAG))
    {
        FatalError("Unsupported separate diag", AMGX_ERR_NOT_IMPLEMENTED);
    }

    if (this->m_explicit_A->get_block_size() != 1)
    {
        FatalError("Unsupported matrix size, KPZPolynomialSmoother", AMGX_ERR_NOT_SUPPORTED_BLOCKSIZE);
    }

    // The first step is to compute the L_inf norm of A, i.e. the max of column sums.
    // Convert the matrix to CSC format (transpose the rows).
    Matrix<T_Config>  AT_csr; // Generic Matrix is a device-side scalar csr format
    transpose( *this->m_explicit_A, AT_csr );
    // Compute the sums of columns and perform the reduction. TODO: try segmented sum then reduction.
    amgx::thrust::counting_iterator<typename Matrix<T_Config>::index_type, typename Matrix<T_Config>::memory_space> it( 0 );
    l_inf = thrust_wrapper::transform_reduce(
                it,
                it + this->m_explicit_A->get_num_rows(),
                detail::row_sum<Matrix<T_Config> >( (int *) AT_csr.row_offsets.raw(), AT_csr.values.raw() ),
                ValueTypeA( 0 ),
                amgx::thrust::maximum<ValueTypeA>( ) );
    cudaCheckError();
}

//
template<class T_Config>
void
KPZPolynomialSolver<T_Config>::solve_init( VVector &b, VVector &x, bool xIsZero )
{
}


// Solve one iteration
template<class T_Config>
bool
KPZPolynomialSolver<T_Config>::solve_iteration( VVector &b, VVector &x, bool xIsZero )
{
    if ( this->m_explicit_A->get_block_dimx() == 1 && this->m_explicit_A->get_block_dimy() == 1 )
    {
        if (xIsZero)
        {
            fill(x, ValueTypeB(0));
        }

        smooth_1x1(*this->m_explicit_A, b, x);
    }
    else
    {
        FatalError("Unsupported blocksize KPZ polynomial", AMGX_ERR_NOT_SUPPORTED_BLOCKSIZE);
    }

    return this->converged( b, x );
}

template<class T_Config>
void
KPZPolynomialSolver<T_Config>::solve_finalize( VVector &b, VVector &x )
{
}

/********************************************/


template<class T_Config>
void KPZPolynomialSolver<T_Config>::smooth_1x1(Matrix<T_Config> &A, const VVector &b, VVector &x)
{
    if (A.hasProps(DIAG))
    {
        FatalError("Unsupported separate diag", AMGX_ERR_NOT_IMPLEMENTED);
    }

    ValueTypeB smax = l_inf;
    ValueTypeB smin = smax / mu;
    ValueTypeB smu0 = ValueTypeB( 1 ) / smax;
    ValueTypeB smu1 = ValueTypeB( 1 ) / smin;
    ValueTypeB skappa = (ValueTypeB) sqrt( smax / smin );
    ValueTypeB delta = (skappa - ValueTypeB( 1 )) / (skappa + ValueTypeB( 1 ));
    ValueTypeB beta = (ValueTypeB)(sqrt(smu0) + sqrt(smu1));
    beta = beta * beta;
    ValueTypeB chi = (ValueTypeB) ((ValueTypeB(4) * smu0 * smu1) / beta);
    int offset = 0, size = 0;
    A.getOffsetAndSizeForView(A.currentView(), &offset, &size);
    VVector r(A.get_num_cols());
    VVector v(A.get_num_cols());
    VVector v0(size);
    VVector sn(size);
    r.set_block_dimy(this->m_explicit_A->get_block_dimy());
    r.set_block_dimx(1);
    r.tag = this->tag * 111 + 1;
    r.dirtybit = 1;
    r.delayed_send = 1;
    v.set_block_dimy(this->m_explicit_A->get_block_dimy());
    v.set_block_dimx(1);
    v.tag = this->tag * 111 + 2;
    v.dirtybit = 1;
    v.delayed_send = 1;
    v0.set_block_dimy(this->m_explicit_A->get_block_dimy());
    v0.set_block_dimx(1);
    sn.set_block_dimy(this->m_explicit_A->get_block_dimy());
    sn.set_block_dimx(1);
    //printf("[%d] x.tag=%d, r.tag=%d, offset=%d, size=%d, r.size()=%d, b.size()=%d, x.size()=%d\n",(A.manager == NULL) ? 0 : A.manager->global_id(), x.tag, r.tag, offset, size, r.size(), b.size(), x.size());
    // r = b - Ax (r.dirtybit will be set inside multiply to indicate the data has changed)
    multiply(A, x, r);
    axpby(b, r, r, ValueTypeB( 1 ), ValueTypeB(-1), offset, size );
    // v0 = (smu0+smu1)/2 * r (copy_async will resize v0 to r size before copying)
    v0.copy_async(r);
    //copy(r, v0);
    scal(v0, (ValueTypeB) ((smu0 + smu1) / ValueTypeB(2)), offset, size);
    // v = (sqrt(smu0) + sqrt(smu1))^2/2 * r - smu0*smu1*A*r
    multiply(A, r, v);
    axpby(r, v, v, beta / ValueTypeB(2), -smu0 * smu1, offset, size);

    for ( int i = 2 ; i <= poly_order ; ++i )
    {
        // sn = chi *(r-A*v) + delta*delta*(v-v0);
        multiply(A, v, sn);
        axpby(r, sn, sn, ValueTypeB(1), ValueTypeB(-1), offset, size);
        //cusp::blas::axpbypcz(sn, v, v0, sn, chi, delta*delta, -delta*delta);
        axpbypcz(sn, v, v0, sn, chi, delta * delta, -delta * delta, offset, size);
        // v0 <- v (copy_async will resize v0 to v size before copying)
        v0.copy_async(v);
        //copy(v, v0);
        // v  <- v + sn
        axpy(sn, v, ValueTypeB(1), offset, size);
    }

    // x = x + v
    axpy(v, x, ValueTypeB(1), offset, size);
}

/****************************************
 * Explict instantiations
 ***************************************/
#define AMGX_CASE_LINE(CASE) template class KPZPolynomialSolver<TemplateMode<CASE>::Type>;
AMGX_FORALL_BUILDS(AMGX_CASE_LINE)
#undef AMGX_CASE_LINE

} // namespace amgx
