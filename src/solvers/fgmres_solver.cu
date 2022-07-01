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

#include <solvers/fgmres_solver.h>
#include <blas.h>
#include <cusp/blas.h>
#include <util.h>
#include <cutil.h>
#include <norm.h>

//TODO remove synchronization from this module by moving host operations to the device

namespace amgx
{

template <class TConfig>
void KrylovSubspaceBuffer<TConfig>::set_max_dimension(int max_dimension)
{
    m_V_vector.resize( max_dimension + 2 );
    m_Z_vector.resize( max_dimension + 1 );

    //set the pointers to NULL (they will be [re]allocated only if they are NULL)
    for (int i = 0; i < max_dimension + 1; i++)
    {
        m_V_vector[i] = NULL;
        m_Z_vector[i] = NULL;
    }

    m_V_vector[max_dimension + 1] = NULL;
    //this looks kind of hacky
    this->max_dimension = max_dimension + 1;
}

template <class TConfig>
KrylovSubspaceBuffer<TConfig>::KrylovSubspaceBuffer()
{
    this->dimension = -1;
    this->max_dimension = 0;
    this->N = 0;
}

template <class TConfig>
KrylovSubspaceBuffer<TConfig>::~KrylovSubspaceBuffer()
{
    for (int i = 0; i < m_V_vector.size(); i++)
    {
        delete m_V_vector[i];
    }

    for (int i = 0; i < m_Z_vector.size(); i++)
    {
        delete m_Z_vector[i];
    }
}

template <class TConfig>
bool KrylovSubspaceBuffer<TConfig>::set_iteration(int m)
{
    if ( m > this->iteration + 1 )
    {
        FatalError("Internal error in set_iteration: It seems like one iteration has not been set", AMGX_ERR_UNKNOWN);
    }

    //if we haven't reached this iteration yet and haven't reached the max dimension, try to increase dimension and if that fails, tell.
    if ( (m > this->iteration) && (this->dimension < this->max_dimension) && (!this->increase_dimension()))
    {
        return false;
    }

    this->iteration = m;
    return true;
}

template <class TConfig>
int KrylovSubspaceBuffer<TConfig>::get_smallest_m()
{
    return max(this->iteration + 1 - this->dimension, 0);
}


template <class TConfig>
bool KrylovSubspaceBuffer<TConfig>::increase_dimension()
{
    if ( this->N < 1 )
    {
        FatalError("N cannot be smaller than 1.", AMGX_ERR_UNKNOWN );
    }

    if (this->dimension == this->max_dimension )
    {
        return true;
    }

    //grow krylov space
    //check whether m_Z_vector of the same size has already been allocated
    if ((m_Z_vector[dimension] != NULL) && (m_Z_vector[dimension]->size() != N))
    {
        delete m_Z_vector[dimension];
        m_Z_vector[dimension] = NULL;
    }

    //check whether m_V_vector of the same size has already been allocated
    if ((m_V_vector[dimension + 1] != NULL) && (m_V_vector[dimension + 1]->size() != N))
    {
        delete m_V_vector[dimension + 1];
        m_V_vector[dimension + 1] = NULL;
    }

    //allocate the vector if it has not been allocated (or was not allocated with the same size)
    try
    {
        if (m_Z_vector[dimension] == NULL)
        {
            m_Z_vector[dimension] = new VVector(N);
        }

        if (m_V_vector[dimension + 1] == NULL)
        {
            m_V_vector[dimension + 1] = new VVector(N);
        }
    }
    catch (std::bad_alloc &e)
    {
        //inform user
        std::cout << "WARNING: Cannot allocate next Krylov vector, out of memory. Falling back to DQGMRES" << std::endl;
        //clear error from error history
        cudaGetLastError();
        //reset max dimension
        this->max_dimension = this->dimension;
        //back out and tell
        return false;
    }

    // init Z and V
    m_V_vector[dimension + 1]->set_block_dimy(this->blockdim);
    m_V_vector[dimension + 1]->set_block_dimx(1);
    m_V_vector[dimension + 1]->dirtybit = 1;
    m_V_vector[dimension + 1]->delayed_send = 1;
    m_V_vector[dimension + 1]->tag = this->tag * 100 + max_dimension + dimension + 1;
    m_Z_vector[dimension]->set_block_dimy(this->blockdim);
    m_Z_vector[dimension]->set_block_dimx(1);
    m_Z_vector[dimension]->dirtybit = 1;
    m_Z_vector[dimension]->delayed_send = 1;
    m_Z_vector[dimension]->tag = this->tag * 100 + dimension;
    dimension++;
    return true;
}

template <class TConfig>
Vector<TConfig> &KrylovSubspaceBuffer<TConfig>::V(int m)
{
    if ( m > this->iteration + 1 )
    {
        FatalError("Try to access unallocated V-vector. You have to set the iteration before accessing this vector", AMGX_ERR_BAD_PARAMETERS );
    }

    if ( m < this->get_smallest_m() )
    {
        FatalError("Try to access forgotten V-vector.", AMGX_ERR_BAD_PARAMETERS );
    }

    return *(this->m_V_vector[m % (this->dimension + 1)]);
}

template <class TConfig>
Vector<TConfig> &KrylovSubspaceBuffer<TConfig>::Z(int m)
{
    if ( m > this->iteration )
    {
        FatalError("Try to access unallocated Z-vector. You have to set the iteration before accessing this vector", AMGX_ERR_BAD_PARAMETERS );
    }

    if ( m < this->get_smallest_m() )
    {
        FatalError("Try to access forgotten Z-vector.", AMGX_ERR_BAD_PARAMETERS );
    }

    return *(this->m_Z_vector[m % this->dimension]);
}

//init the frist vector
template <class TConfig>
void KrylovSubspaceBuffer<TConfig>::setup(int N, int blockdim, int tag)
{
    this->N = N;
    this->blockdim = blockdim;
    this->tag = tag;
    this->iteration = -1;
    this->dimension = 0;

    //init V(0)
    //check whether m_V_vector of the same size has already been allocated
    if ((m_V_vector[0] != NULL) && (m_V_vector[0]->size() != N))
    {
        delete m_V_vector[0];
        m_V_vector[0] = NULL;
    }

    //allocate the vector if it has not been allocated (or was not allocated with the same size)
    if (m_V_vector[0] == NULL)
    {
        m_V_vector[0] = new VVector(N);
    }

    m_V_vector[0]->set_block_dimy(this->blockdim);
    m_V_vector[0]->set_block_dimx(1);
    m_V_vector[0]->dirtybit = 1;
    m_V_vector[0]->delayed_send = 1;
    m_V_vector[0]->tag = this->tag * 100 + max_dimension + 1;
}


template< class T_Config>
FGMRES_Solver<T_Config>::FGMRES_Solver( AMG_Config &cfg, const std::string &cfg_scope ) :
    Solver<T_Config>( cfg, cfg_scope ), m_preconditioner(0)

{
    std::string solverName, new_scope, tmp_scope;
    cfg.getParameter<std::string>( "preconditioner", solverName, cfg_scope, new_scope );

    if (solverName.compare("NOSOLVER") == 0)
    {
        use_preconditioner = false;
        m_preconditioner = NULL;
    }
    else
    {
        use_preconditioner = true;
        m_preconditioner = SolverFactory<T_Config>::allocate( cfg, cfg_scope, "preconditioner" );
    }

    m_R = cfg.AMG_Config::getParameter<int>("gmres_n_restart", cfg_scope);
    m_krylov_size = min( this->m_max_iters, m_R );
    int krylov_param = cfg.AMG_Config::getParameter<int>( "gmres_krylov_dim", cfg_scope );

    if ( krylov_param > 0 )
    {
        m_krylov_size = min( m_krylov_size, krylov_param );
    }

    //Using L2 norm is ok, however we will do the extra computations
    //if( this->m_norm_type != L2 )
    //  FatalError("FGMRES only works with L2 norm. Other norms would require extra computations. ", AMGX_ERR_NOT_SUPPORTED_TARGET);
    m_H.resize( m_R + 1, m_R );
    m_s.resize( m_R + 1 );
    m_cs.resize( m_R );
    m_sn.resize( m_R );
    gamma.resize( m_R + 1 );
    subspace.set_max_dimension( m_krylov_size );
}

template<class T_Config>
FGMRES_Solver<T_Config>::~FGMRES_Solver()
{
    if (use_preconditioner) { delete m_preconditioner; }
}

template<class T_Config>
void
FGMRES_Solver<T_Config>::printSolverParameters() const
{
    std::cout << "gmres_n_restart=" << this->m_R << std::endl;

    if (use_preconditioner)
    {
        std::cout << "preconditioner: " << this->m_preconditioner->getName() << " with scope name: " << this->m_preconditioner->getScope() << std::endl;
    }
}

template<class T_Config>
void
FGMRES_Solver<T_Config>::solver_setup(bool reuse_matrix_structure)
{
    if (use_preconditioner)
    {
        m_preconditioner->setup( *this->m_A, reuse_matrix_structure );
    }

    ViewType oldView = this->m_A->currentView();
    this->m_A->setViewExterior();
    //should we warn the user about the extra computational work?
    use_scalar_L2_norm = (this->m_nrm.size() == 1 || this->m_use_scalar_norm) && this->m_norm_type == L2;
    subspace.setup(this->m_A->get_num_cols()*this->m_A->get_block_dimy(), this->m_A->get_block_dimy(), this->tag);
    residual.tag = (this->tag + 1) * 100 - 2;

    if ( this->m_R == 1 || this->m_max_iters == 1 )
    {
        update_x_every_iteration = true;
        update_r_every_iteration = false;
    }
    else
    {
        // The update of x is needed only if running the truncated gmres
        update_x_every_iteration = (m_krylov_size < m_R);
        update_r_every_iteration = (!use_scalar_L2_norm || (m_krylov_size < m_R)) && Base::m_monitor_convergence;
    }

    this->m_A->setView(oldView);
}

template <typename ValueType>
static __host__ void GeneratePlaneRotation( ValueType &dx, ValueType &dy, ValueType &cs, ValueType &sn )
{
    if (dy < ValueType(0.0))
    {
        cs = 1.0;
        sn = 0.0;
    }
    else if (abs(dy) > abs(dx))
    {
        ValueType tmp = dx / dy;
        sn = ValueType(1.0) / sqrt(ValueType(1.0) + tmp * tmp);
        cs = tmp * sn;
    }
    else
    {
        ValueType tmp = dy / dx;
        cs = ValueType(1.0) / sqrt(ValueType(1.0) + tmp * tmp);
        sn = tmp * cs;
    }
}

template <typename ValueType>
void PlaneRotation( cusp::array2d<ValueType, cusp::host_memory, cusp::column_major> &H,
                    cusp::array1d<ValueType, cusp::host_memory> &cs,
                    cusp::array1d<ValueType, cusp::host_memory> &sn,
                    cusp::array1d<ValueType, cusp::host_memory> &s,
                    int i)
{
    ValueType temp;

    for (int k = 0; k < i; k++)
    {
        temp     =  cs[k] * H(k, i) + sn[k] * H(k + 1, i);
        H(k + 1, i) = -sn[k] * H(k, i) + cs[k] * H(k + 1, i);
        H(k, i)   = temp;
    }

    GeneratePlaneRotation(H(i, i), H(i + 1, i), cs[i], sn[i]);
    H(i, i) = cs[i] * H(i, i) + sn[i] * H(i + 1, i);
    H(i + 1, i) = 0.0;
    temp = cs[i] * s[i];
    s[i + 1] = -sn[i] * s[i];
    s[i] = temp;
}

template<class T_Config>
void
FGMRES_Solver<T_Config>::solve_init( VVector &b, VVector &x, bool xIsZero )
{
    //init residual, even if we don't plan to use it, we might need it, so make sure we have enough memory to store it now
    residual.resize( b.size() );
    residual.set_block_dimx( 1 );
    residual.set_block_dimy( this->m_A->get_block_dimy() );
    residual.dirtybit = 1;
    residual.delayed_send = 1;
}

//check for convergence
//al the complicated stuff happens here
template <class TConfig>
bool FGMRES_Solver<TConfig>::checkConvergenceGMRES(bool check_V_0)
{
    if ( Base::m_monitor_convergence)
    {
        //enable blas operations
        Operator<TConfig> &A = *this->m_A;
        int offset, size;
        A.getOffsetAndSizeForView(A.getViewExterior(), &offset, &size);

        if ( this->use_scalar_L2_norm && !update_r_every_iteration )
        {
            this->m_nrm[0] = this->beta;
            return this->converged();
        }
        else
        {
            if ( check_V_0 )
            {
                get_norm( A, subspace.V(0), A.get_block_dimy(), this->m_norm_type, this->m_nrm );
                return this->converged();
            }
            else
            {
                if ( !update_r_every_iteration )
                {
                    FatalError("have to compute the residual every iteration to compute a norm other than scalar L2", AMGX_ERR_BAD_PARAMETERS );
                }

                //compute norm of r
                get_norm( A, residual, A.get_block_dimy(), this->m_norm_type, this->m_nrm );
                return this->converged();
            }
        }
    }
    else
    {
        return false;
    }
}


//Run preconditioned GMRES
template<class T_Config>
bool
FGMRES_Solver<T_Config>::solve_iteration( VVector &b, VVector &x, bool xIsZero )
{
    Operator<T_Config> &A = *this->m_A;
    ViewType oldView = A.currentView();
    A.setViewExterior();
    int offset, size;
    A.getOffsetAndSizeForView(A.getViewExterior(), &offset, &size);
    bool done = false;
    int m = this->m_curr_iter % m_R; //current iteration within restart

    if (m == 0)
    {
        //initialize gmres
        subspace.set_iteration(0);
        // compute initial residual r0 = b - Ax
        axmb( A, x, b, subspace.V(0), offset, size );
        // compute initial residual norm
        this->beta = get_norm(A, subspace.V(0), L2);

        // check for convergence (do we need it? leave it here for now)
        if ((this->m_curr_iter == 0) && checkConvergenceGMRES( true ))
        {
            return true;
        }

        // normalize initial residual
        scal( subspace.V(0), ValueTypeB(1.0 / this->beta), offset, size );
        //set reduced system rhs = beta*e1
        cusp::blas::fill( m_s, ValueTypeB(0.0) );
        m_s[0] = this->beta;
    }

    //our krylov space is now smaller than m!
    //because we hadn't updated x before, we have to form
    //change the base formed by Z
    if ( !subspace.set_iteration(m) )
    {
        //we have to start updating x from now on, so prepare Z for that now
        if ( !update_x_every_iteration )
        {
            //TODO: There could be a more efficient way to do this
            for (int k = 1; k < m; k++)
            {
                // This can be written as [Z] * inv(R), where R=Q*H. It is more efficient to do M-M product.
                // But this would require the memory to be in consecutive chunk. This is not what
                // the lazy memory allocation does.
                //p_k = [z_k - sum( h_ik*p_i )] / h_kk
                for (int i = 0; i < k; i++)
                {
                    axpy( subspace.Z(i), subspace.Z(k), -m_H(i, k), offset, size );
                }

                scal( subspace.Z(k), ValueTypeB(1.0) / m_H(k, k), offset, size );
                // This can be written as dense M-V product, [Z]*gamma[1:m-1]
                //x_k = x_k-1 + gamma_k*p_k
                axpy( subspace.Z(k), x, gamma[k], offset, size );
            }

            update_x_every_iteration = true;
        }

        if ( !update_r_every_iteration && Base::m_monitor_convergence)
        {
            //compute residual
            axmb( A, x, b, residual, offset, size );
            update_r_every_iteration = true;
        }

        subspace.set_iteration(m); //if the allocation failed, we have to set the iteration again
    }

    // Run one iteration of preconditioner with zero initial guess and v_m as rhs, i.e. solve Az_m=v_m
    if (use_preconditioner)
    {
        m_preconditioner->solve( subspace.V(m), subspace.Z(m), true ); //TODO: check if using zero as initial solution when solving for residual inside subspace is correct 
    }
    else
    {
        copy(subspace.V(m), subspace.Z(m), offset, size);
    }

    //obtain v_m+1 := A*z_m
    A.apply( subspace.Z(m), subspace.V(m + 1) );

    // Modified Gram-Schmidt
    for ( int i = subspace.get_smallest_m(); i <= m; i++ )
    {
        // H(i,m) = <V(i),V(m+1)>
        m_H(i, m) = dot(A, subspace.V(i), subspace.V(m + 1));
        // V(m+1) -= H(i, m) * V(i)
        axpy( subspace.V(i), subspace.V(m + 1), -m_H(i, m), offset, size );
    }

    //H(m+1,m) = || v_m+1 ||
    m_H(m + 1, m) = get_norm(A, subspace.V(m + 1), L2);
    //normalize v_m+1
    scal( subspace.V(m + 1), ValueTypeB(1.0) / m_H(m + 1, m), offset, size );
    this->gamma[m] = m_s[m];
    PlaneRotation( m_H, m_cs, m_sn, m_s, m );

    if ( update_x_every_iteration )
    {
        //p_m = [z_m - sum( h_im*p_i )] / h_mm
        // This is dense [Z]*[-H(smallest_m:m-1,m); 1] / m_H(m,m)
        for (int i = subspace.get_smallest_m(); i < m; i++)
        {
            axpy( subspace.Z(i), subspace.Z(m), -m_H(i, m), offset, size );
        }

        scal( subspace.Z(m), ValueTypeB(1.0) / m_H(m, m), offset, size );
        //x_m = x_m-1 + gamma_m*pm
        axpy( subspace.Z(m), x, m_s[m], offset, size );
    }

    if ( update_r_every_iteration )
    {
        // This is the recursion in Christophe's presentation
        // r_m = gamma_m+1*( c_m*v_m+1 - s_m*r_m-1/gamma_m )
        // r_m = (gamma_m+1*c_m)*v_m+1 + (-gamma_m+1*s_m/gamma_m)*r_m-1)
        if ( m == 0 )
        {
            axpby( subspace.V(1), subspace.V(0), residual, m_s[m + 1]*m_cs[m], ValueTypeB(-1.0 * m_s[m + 1]*m_sn[m]), offset, size );
        }
        else
        {
            axpby( subspace.V(m + 1), residual, residual, m_s[m + 1]*m_cs[m], ValueTypeB(-1.0 * m_s[m + 1]*m_sn[m] / gamma[m]), offset, size );
        }
    }

    // Check for convergence
    // abs(s[m+1]) = L2 norm of residual
    this->beta = abs( m_s[m + 1] );
    done = checkConvergenceGMRES( false );

    // If reached restart limit or last iteration or if converged, compute x vector
    if ( !update_x_every_iteration && (m == m_R - 1 || this->is_last_iter() || done ))
    {
        // Solve upper triangular system in place
        for (int j = m; j >= 0; j--)
        {
            m_s[j] /= m_H(j, j);

            //S(0:j) = s(0:j) - s[j] H(0:j,j)
            for (int k = j - 1; k >= 0; k--)
            {
                m_s[k] -= m_H(k, j) * m_s[j];
            }
        }

        //    Update the solution
        // This is dense M-V, x += [Z]*m_s
        for (int j = 0; j <= m; j++)
        {
            axpy( subspace.Z(j), x, m_s[j], offset, size );
        }
    }

    A.setView(oldView);
    return !Base::m_monitor_convergence || done;
}

template<class T_Config>
void
FGMRES_Solver<T_Config>::solve_finalize( VVector &b, VVector &x )
{
    residual.resize(0);
}

/****************************************
* Explict instantiations
***************************************/
#define AMGX_CASE_LINE(CASE) template class FGMRES_Solver<TemplateMode<CASE>::Type>;
AMGX_FORALL_BUILDS(AMGX_CASE_LINE)
#undef AMGX_CASE_LINE

} // namespace amgx
