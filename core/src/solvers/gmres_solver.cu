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

#include <solvers/gmres_solver.h>
#include <blas.h>
#include <multiply.h>
#include <cusp/blas.h>
#include <util.h>
#include <cutil.h>
#include <norm.h>

#include <amgx_types/util.h>

#include <solvers/block_common_solver.h>

#include <amgx_cublas.h>

//TODO remove synchronization from this module by moving host operations to the device
namespace amgx
{

template< class T_Config>
GMRES_Solver<T_Config>::GMRES_Solver( AMG_Config &cfg, const std::string &cfg_scope ) :
    Solver<T_Config>( cfg, cfg_scope ), m_preconditioner(0), no_preconditioner(true)

{
    std::string solverName, new_scope, tmp_scope;
    cfg.getParameter<std::string>( "preconditioner", solverName, cfg_scope, new_scope );

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

    m_R = cfg.AMG_Config::getParameter<int>("gmres_n_restart", cfg_scope);
    m_krylov_size = min( this->m_max_iters, m_R );

    if ( this->m_norm_type != L2 )
    {
        FatalError("GMRES only works with L2 norm. Other norms would require extra computations. ", AMGX_ERR_NOT_SUPPORTED_TARGET);
    }

    m_H.resize( m_krylov_size + 1, m_krylov_size );
    m_s.resize( m_krylov_size + 1 );
    m_cs.resize( m_krylov_size );
    m_sn.resize( m_krylov_size );
    m_V_vectors.resize( m_krylov_size + 1 );
}

template<class T_Config>
GMRES_Solver<T_Config>::~GMRES_Solver()
{
    if (!no_preconditioner) { delete m_preconditioner; }
}

template<class T_Config>
void
GMRES_Solver<T_Config>::printSolverParameters() const
{
    std::cout << "gmres_n_restart=" << this->m_R << std::endl;

    if (!no_preconditioner)
    {
        std::cout << "preconditioner: " << this->m_preconditioner->getName() << " with scope name: " << this->m_preconditioner->getScope() << std::endl;
    }
}

template<class T_Config>
void
GMRES_Solver<T_Config>::solver_setup(bool reuse_matrix_structure)
{
    // Setup the solver
    ViewType oldView = this->m_A->currentView();
    this->m_A->setViewExterior();

    if ( this->m_A->get_block_dimy() != 1 && !this->m_use_scalar_norm )
    {
        FatalError( "GMRES solver only works on block matrix if configuration parameter use_scalar_norm=1", AMGX_ERR_NOT_SUPPORTED_TARGET );
    }

    if (!no_preconditioner) { m_preconditioner->setup( *this->m_A, reuse_matrix_structure ); }

    // Make sure vectors already have decent sizes.
    assert( m_V_vectors.size() >= m_krylov_size + 1 );
    // The number of elements in temporary vectors.
    const int N = static_cast<int>( this->m_A->get_num_cols() * this->m_A->get_block_dimy() );

    // Allocate memory needed for iterating.
    for ( int i = 0 ; i <= m_krylov_size ; ++i )
    {
        m_V_vectors[i].resize(N);
    }

    m_Z_vector.resize(N);

    for ( int i = 0 ; i <= m_krylov_size ; ++i )
    {
        m_V_vectors[i].set_block_dimy(this->m_A->get_block_dimy());
        m_V_vectors[i].set_block_dimx(1);
        m_V_vectors[i].dirtybit = 1;
        m_V_vectors[i].delayed_send = 1;
        m_V_vectors[i].tag = this->tag * 100 + i;
    }

    m_Z_vector.set_block_dimy(this->m_A->get_block_dimy());
    m_Z_vector.set_block_dimx(1);
    m_Z_vector.dirtybit = 1;
    m_Z_vector.delayed_send = 1;
    m_Z_vector.tag = this->tag * 100;
    this->m_A->setView(oldView);
}

template <typename ValueType, bool IsComplex>
struct GeneratePlaneRotation;

template <typename ValueType>
struct GeneratePlaneRotation<ValueType, false>
{
    static __host__ void generate( ValueType &dx, ValueType &dy, ValueType &cs, ValueType &sn, ValueType &rhsx, ValueType &rhsy )
    {
        ValueType tmp;

        if (dy < ValueType(0.0))
        {
            cs = 1.0;
            sn = 0.0;
        }
        else if (abs(dy) > abs(dx))
        {
            tmp = dx / dy;
            sn = ValueType(1.0) / sqrt(ValueType(1.0) + tmp * tmp);
            cs = tmp * sn;
        }
        else
        {
            tmp = dy / dx;
            cs = ValueType(1.0) / sqrt(ValueType(1.0) + tmp * tmp);
            sn = tmp * cs;
        }

        tmp   =  cs * rhsx;
        rhsy  = -sn * rhsx;
        rhsx  = tmp;
    }
};



template <typename ValueType>
struct GeneratePlaneRotation<ValueType, true>
{
    static __host__ void generate( ValueType &dx, ValueType &dy, ValueType &cs, ValueType &sn, ValueType &rhsx, ValueType &rhsy )
    {
        typedef typename types::PODTypes<ValueType>::type PodTypeB;
        ValueType tmp;
        PodTypeB adx = types::util<ValueType>::abs(dx);
        PodTypeB ady = types::util<ValueType>::abs(dy);

        if (isCloseToZero(dx + dy))
        {
            cs = types::util<ValueType>::get_one();
            sn = types::util<ValueType>::get_zero();
        }
        else if (ady > adx)
        {
            adx = adx / ady;
            sn = types::util<ValueType>::get_one() / sqrt(PodTypeB(1.0) + adx * adx);
            cs = sn * adx;
        }
        else
        {
            ady = ady / adx;
            cs = types::util<ValueType>::get_one() / sqrt(PodTypeB(1.0) + ady * ady);
            sn = cs * ady;
        }

        tmp   = cs * rhsx;
        rhsy  = types::util<ValueType>::invert(types::util<ValueType>::conjugate(sn)) * rhsx; //-conjugate(sin)
        rhsx  = tmp;
    }
};


template <typename ValueType>
static __host__ void PlaneRotation( cusp::array2d<ValueType, cusp::host_memory, cusp::column_major> &H,
                                    cusp::array1d<ValueType, cusp::host_memory> &cs,
                                    cusp::array1d<ValueType, cusp::host_memory> &sn,
                                    cusp::array1d<ValueType, cusp::host_memory> &s,
                                    int i)
{
    ValueType temp;

    for (int k = 0; k < i; k++)
    {
        temp     =  cs[k] * H(k,  i) +                                   sn[k] * H(k + 1, i);
        H(k + 1, i) =  cs[k] * H(k + 1, i) - types::util<ValueType>::conjugate(sn[k]) * H(k, i);
        H(k, i)   = temp;
    }

    GeneratePlaneRotation<ValueType, types::util<ValueType>::is_complex>::generate(H(i, i), H(i + 1, i), cs[i], sn[i], s[i], s[i + 1]);
    H(i, i) = cs[i] * H(i, i) + sn[i] * H(i + 1, i);
    H(i + 1, i) = types::util<ValueType>::get_zero();
}

template<class T_Config>
void
GMRES_Solver<T_Config>::solve_init( VVector &b, VVector &x, bool xIsZero )
{}

template<class T_Config>
bool
GMRES_Solver<T_Config>::solve_one_iteration( VVector &b, VVector &x )
{
    ViewType oldView = this->m_A->currentView();
    this->m_A->setViewExterior();
    Operator<T_Config> &A = *this->m_A;
    int offset, size;
    A.getOffsetAndSizeForView(A.getViewExterior(), &offset, &size);
    // compute initial residual
    A.apply( x, m_V_vectors[0]); // V(0) = A*x
    axpy( b, m_V_vectors[0], types::util<ValueTypeB>::get_minus_one(), offset, size);       // V(0) = V(0) - b
    PodTypeB beta = get_norm(A, m_V_vectors[0], L2);         // beta = norm(V(0))
    Cublas::scal( size, PodTypeB(-1.0 / beta), m_V_vectors[0].raw() + offset, 1 );   // V(0) = -V(0)/beta //
    cusp::blas::fill( m_s, types::util<ValueTypeB>::get_zero() );
    m_s[0] = types::util<ValueTypeB>::get_one() * beta;

    // Run one iteration of preconditioner with zero initial guess
    if (no_preconditioner)
    {
        copy(m_V_vectors[0], m_Z_vector, offset, size);
    }
    else
    {
        m_V_vectors[0].delayed_send = 1;
        m_Z_vector.delayed_send = 1;
        m_preconditioner->solve( m_V_vectors[0], m_Z_vector, true );
        m_V_vectors[0].delayed_send = 1;
        m_Z_vector.delayed_send = 1;
    }

    A.apply(m_Z_vector, m_V_vectors[1]);
    // Modified Gram-Schmidt
    //  H(k,i) = <V(i+1),V(k)>    //
    m_H(0, 0) = dot(A, m_V_vectors[1], m_V_vectors[0]);
    // V(i+1) -= H(k, i) * V(k)  //
    axpy( m_V_vectors[0], m_V_vectors[1], types::util<ValueTypeB>::invert(m_H(0, 0)), offset, size );
    m_H(1, 0) = types::util<ValueTypeB>::get_one() * get_norm(A, m_V_vectors[1], L2);
    PlaneRotation( m_H, m_cs, m_sn, m_s, 0 );

    if ( this->m_monitor_convergence )
    {
        this->m_nrm[0] = types::util<ValueTypeB>::abs( m_s[1] );
    }

    m_s[0] = m_s[0] / m_H(0, 0);
    //  Update the solution
    axpy( m_Z_vector, x, m_s[0], offset, size );
    this->m_A->setView(oldView);
    return !this->m_monitor_convergence || this->converged();
}

template<class T_Config>
bool
GMRES_Solver<T_Config>::solve_iteration( VVector &b, VVector &x, bool xIsZero )
{
    Operator<T_Config> &A = *this->m_A;
    ViewType oldView = this->m_A->currentView();
    this->m_A->setViewExterior();
    int offset, size;
    A.getOffsetAndSizeForView(A.getViewExterior(), &offset, &size);

    if ( this->m_max_iters == 1 )
    {
        return solve_one_iteration( b, x );
    }

    bool done = false;
    int i = this->m_curr_iter % m_R; //current iteration within restart

    if (i == 0)
    {
        // compute initial residual
        A.apply(x, m_V_vectors[0]); // V(0) = A*x
        axpy( b, m_V_vectors[0], types::util<ValueTypeB>::get_minus_one(), offset, size ); // V(0) = V(0) - b
        PodTypeB beta = get_norm(A, m_V_vectors[0], L2); // beta = norm(V(0))

        if ( Base::m_monitor_convergence )
        {
            this->m_nrm[0] = beta;

            if ( this->converged() )
            {
                return true;
            }
        }

        Cublas::scal( size, PodTypeB(-1.0 / beta), m_V_vectors[0].raw() + offset, 1 );                // V(0) = -V(0)/beta //
        cusp::blas::fill( m_s, types::util<ValueTypeB>::get_zero() );
        m_s[0] = types::util<ValueTypeB>::get_one() * beta;
    }

    // Run one iteration of preconditioner with zero initial guess
    if (no_preconditioner)
    {
        copy(m_V_vectors[i], m_Z_vector, offset, size);
    }
    else
    {
        m_V_vectors[i].delayed_send = 1;
        m_Z_vector.delayed_send = 1;
        m_preconditioner->solve( m_V_vectors[i], m_Z_vector, true );
        m_V_vectors[i].delayed_send = 1;
        m_Z_vector.delayed_send = 1;
    }

    A.apply(m_Z_vector, m_V_vectors[i + 1]);

    // Modified Gram-Schmidt
    for ( int k = 0; k <= i; ++k )
    {
        //  H(k,i) = <V(i+1),V(k)>    //
        m_H(k, i) = dot(A, m_V_vectors[i + 1], m_V_vectors[k]);
        // V(i+1) -= H(k, i) * V(k)  //
        axpy( m_V_vectors[k], m_V_vectors[i + 1], types::util<ValueTypeB>::invert(m_H(k, i)), offset, size );
    }

    m_H(i + 1, i) = types::util<ValueTypeB>::get_one() * get_norm(A, m_V_vectors[i + 1], L2);
    scal( m_V_vectors[i + 1], types::util<ValueTypeB>::get_one() / m_H(i + 1, i), offset, size );
    PlaneRotation( m_H, m_cs, m_sn, m_s, i );

    // Check for convergence
    // abs(s[i+1]) = L2 norm of residual
    if ( Base::m_monitor_convergence )
    {
        this->m_nrm[0] = types::util<ValueTypeB>::abs( m_s[i + 1] );
        done = this->converged();
    }

    // If reached restart limit or last iteration or if converged, compute x vector
    if ( i == (m_R - 1) || this->is_last_iter() || done )
    {
        // Solve upper triangular system in place
        for (int j = i; j >= 0; j--)
        {
            m_s[j] = m_s[j] / m_H(j, j);

            //S(0:j) = s(0:j) - s[j] H(0:j,j)
            for (int k = j - 1; k >= 0; k--)
            {
                m_s[k] = m_s[k] - (m_H(k, j) * m_s[j]);
            }
        }

        // Accumulate sum_n V_m*y_m into m_Z_vector
        thrust::fill(m_Z_vector.begin(), m_Z_vector.end(), types::util<ValueTypeB>::get_zero());
        cudaCheckError();

        for (int j = 0; j <= i; j++)
        {
            axpy( m_V_vectors[j], m_Z_vector, m_s[j], offset, size );
        }

        // Call the preconditioner to get M^-1*(sum_m vm*ym), store in m_V_Vectors[0]
        if (no_preconditioner)
        {
            copy( m_Z_vector, m_V_vectors[0], offset, size);
        }
        else
        {
            m_V_vectors[0].delayed_send = 1;
            m_Z_vector.delayed_send = 1;
            m_preconditioner->solve( m_V_vectors[0], m_Z_vector, true );
            m_V_vectors[0].delayed_send = 1;
            m_Z_vector.delayed_send = 1;
        }

        // Update the solution
        // Add to x0
        axpy( m_V_vectors[0], x, types::util<ValueTypeB>::get_one(), offset, size );
    }

    this->m_A->setView(oldView);
    return !Base::m_monitor_convergence || done;
}

template<class T_Config>
void
GMRES_Solver<T_Config>::solve_finalize( VVector &b, VVector &x )
{}

/****************************************
* Explict instantiations
***************************************/
#define AMGX_CASE_LINE(CASE) template class GMRES_Solver<TemplateMode<CASE>::Type>;
AMGX_FORALL_BUILDS(AMGX_CASE_LINE)
AMGX_FORCOMPLEX_BUILDS(AMGX_CASE_LINE)
#undef AMGX_CASE_LINE

} // namespace amgx
