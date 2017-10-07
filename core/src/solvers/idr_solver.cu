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

#include <solvers/idr_solver.h>
#include <amgx_cublas.h>
#include <blas.h>
#include <multiply.h>
#include <util.h>
#include <iostream>
#include <fstream>
#include <sstream>
#include <curand.h>

using namespace std;
namespace amgx
{
namespace idr_solver
{
// Constructor
template< class T_Config>
IDR_Solver_Base<T_Config>::IDR_Solver_Base( AMG_Config &cfg, const std::string &cfg_scope) :
    Solver<T_Config>( cfg, cfg_scope),
    m_buffer_N(0)
{
    std::string solverName, new_scope, tmp_scope;
    cfg.getParameter<std::string>( "preconditioner", solverName, cfg_scope, new_scope );
    s = cfg.AMG_Config::getParameter<int>("subspace_dim_s", cfg_scope);

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
}

template<class T_Config>
IDR_Solver_Base<T_Config>::~IDR_Solver_Base()
{
    if (!no_preconditioner) { delete m_preconditioner; }
}

template<class T_Config>
void
IDR_Solver_Base<T_Config>::solver_setup(bool reuse_matrix_structure)
{
    AMGX_CPU_PROFILER( "IDR_Solver::solver_setup " );
    Operator<T_Config> &A = *this->m_A;
    ViewType oldView = A.currentView();
    A.setViewExterior();
    // The number of elements in temporary vectors.
    this->m_buffer_N = static_cast<int>( this->m_A->get_num_cols() * this->m_A->get_block_dimy() );
    const int N = this->m_buffer_N;
    s = this->s;
    // Allocate memory needed for iterating.
    m_z.resize(N);
    m_Ax.resize(N);
    m_v.resize(N);
    tempg.resize(N);
    tempu.resize(N);
    temp.resize(N);
    t_idr.resize(N);
    c.resize(s);
    m_f.resize(s);
    h_chk.resize(N * s);
    svec_chk.resize(s);
    G.resize(N * s);
    G.set_lda(N);
    G.set_num_rows(N);
    G.set_num_cols(s);
    U.resize(N * s);
    U.set_lda(N);
    U.set_num_rows(N);
    U.set_num_cols(s);
    P.resize(N * s);
    P.set_lda(N);
    P.set_num_rows(N);
    P.set_num_cols(s);
    M.resize(s * s);
    M.set_lda(s);
    M.set_num_rows(s);
    M.set_num_cols(s);
    m_Ax.set_block_dimy(this->m_A->get_block_dimy());
    m_Ax.set_block_dimx(1);
    m_Ax.dirtybit = 1;
    m_Ax.delayed_send = 1;
    m_Ax.tag = this->tag * 100 + 2;
    m_z.set_block_dimy(this->m_A->get_block_dimy());
    m_z.set_block_dimx(1);
    m_z.dirtybit = 1;
    m_z.delayed_send = 1;
    m_z.tag = this->tag * 100 + 3;
    m_v.set_block_dimy(this->m_A->get_block_dimy());
    m_v.set_block_dimx(1);
    m_v.dirtybit = 1;
    m_v.delayed_send = 1;
    m_v.tag = this->tag * 100 + 3;
    m_f.set_block_dimx(1);
    m_f.set_block_dimy(1);
    c.set_block_dimx(1);
    c.set_block_dimy(1);
    temp.set_block_dimx(1);
    temp.set_block_dimy(this->m_A->get_block_dimy());
    tempu.set_block_dimx(1);
    tempu.set_block_dimy(this->m_A->get_block_dimy());
    tempg.set_block_dimx(1);
    tempg.set_block_dimy(this->m_A->get_block_dimy());
    t_idr.set_block_dimx(1);
    t_idr.set_block_dimy(this->m_A->get_block_dimy());
    U.set_block_dimx(1);
    U.set_block_dimy(this->m_A->get_block_dimy());
    G.set_block_dimx(1);
    G.set_block_dimy(this->m_A->get_block_dimy());
    P.set_block_dimx(1);
    P.set_block_dimy(this->m_A->get_block_dimy());
    M.set_block_dimx(1);
    M.set_block_dimy(1);

    // Setup the preconditionner
    if (!no_preconditioner)
    {
        m_preconditioner->setup(A, reuse_matrix_structure);
    }

    A.setView(oldView);
}

template<class T_Config>
void
IDR_Solver_Base<T_Config>::solve_init( VVector &b, VVector &x, bool xIsZero )
{
    AMGX_CPU_PROFILER( "IDR_Solver::solve_init " );
    int s;
    Operator<T_Config> &A = *this->m_A;
    ViewType oldView = A.currentView();
    A.setViewExterior();
    int offset, size, N;
    A.getOffsetAndSizeForView(A.getViewExterior(), &offset, &size);
    N = this->m_buffer_N;
    s = this->s;
    this->omega = 1;
    // M is identity
    fill(G, (ValueTypeB)0, 0, N * s);
    fill(U, (ValueTypeB)0, 0, N * s);
    fill(M, (ValueTypeB)0, 0, s * s);
    fill(P, (ValueTypeB)0, 0, N * s);
    fill(tempg, (ValueTypeB)0, 0, N);
    fill(tempu, (ValueTypeB)0, 0, N);
    fill(temp, (ValueTypeB)0, 0, N);
    fill(t_idr, (ValueTypeB)0, 0, N);
    fill(m_f, (ValueTypeB)0, 0, s);
    fill(c, (ValueTypeB)0, 0, s);
    fill(m_v, (ValueTypeB)0, 0, N);
    setup_arrays(P, M, b, x, h_chk, s, N);
    A.setView(oldView);
}

template<class T_Config>
bool
IDR_Solver_Base<T_Config>::solve_iteration( VVector &b, VVector &x, bool xIsZero )
{
    AMGX_CPU_PROFILER( "IDR_Solver::solve_iteration " );
    Operator<T_Config> &A = *this->m_A;
    ViewType oldView = A.currentView();
    A.setViewExterior();
    bool transposed = false;
    int offset, i, s, k, N, size;
    ValueTypeB alpha_blas(1), beta_blas(0), malpha_blas(-1);// malpha_blas=-1.0f;
    ValueTypeB alpha, beta;
    ValueTypeB ns, nt, ts, rho, angle(0.7);
    A.getOffsetAndSizeForView(A.getViewExterior(), &offset, &size);
    N = A.get_num_rows();
    s = this->s;

    if (s == 1) { angle = (ValueTypeB) 0; }

    // New right-hand size for small system:
    // f = (r'*P)';
    dot_ina_loop(P, *this->m_r, 0, 0, m_f, svec_chk, 0, N, 0, s);

    // solving the small system  and making v orth. to P
    for (k = 0; k < s; k++)
    {
        // c = M(k:s,k:s)\f(k:s); getrf+trsv_v2
        copy_ext(m_f, c, k, 0, s - k );
        transposed = false;
        trsv_extnd(transposed, M, s, c, s - k, 1, k + s * k);
        //  v = r - G(:,k:s)*c; dense matvec then vector update
        transposed = false;
        gemv_extnd(transposed, G, c, temp, N, s - k, alpha_blas, beta_blas, 1, 1, N, k * N, 0, 0);
        axpby(*this->m_r, temp, m_v, alpha_blas, malpha_blas, 0, N);

        if (no_preconditioner)
        {
            ;
        }
        else
        {
            m_z.delayed_send = 1;
            m_v.delayed_send = 1;
            m_preconditioner->solve( m_v, m_z, true );
            m_z.delayed_send = 1;
            m_z.delayed_send = 1;
            copy(m_z, m_v, 0, N);
        }

        // U(:,k) = U(:,k:s)*c + om*v; matvec + axpy
        transposed = false;
        gemv_extnd(transposed, U, c, temp, N, s - k, alpha_blas, beta_blas, 1, 1, N, k * N, 0, 0);
        copy_ext(temp, U, 0, k * N, N);
        axpy(m_v, U, this->omega, 0, k * N, N);
        // G(:,k) = A*U(:,k); matvec
        copy_ext(U, tempu, k * N, 0, N);
        A.apply(tempu, tempg);
        copy_ext(tempg, G, 0, k * N, N);

        // Bi-Orthogonalise the new basis vectors:
        for (i = 0; i < k; i++)
        {
            //( P(:,i)'*G(:,k) )/M(i,i);
            dotc_div(P, G, i * N, k * N, N, M, i, s, &alpha);

            if (alpha == (ValueTypeB) 0)
            {
                FatalError("M(i,i)=0 breakdown condition (alpha):IDR", AMGX_ERR_INTERNAL);
            }

            // G(:,k) = G(:,k) - alpha*G(:,i);
            axpy(G, G, -alpha, i * N, k * N, N);
            // U(:,k) = U(:,k) - alpha*U(:,i);
            axpy(U, U, -alpha, i * N, k * N, N);
        }

        // New column of M = P'*G  (first k-1 entries are zero)
        // M(k:s,k) = (G(:,k)'*P(:,k:s))';

        transposed = true;
        gemv_div(transposed, P, G, M, N, s - k, alpha_blas, beta_blas, 1, 1, N, k * N, k * N, k * s + k, m_f, k, s, &beta, svec_chk);

        if (beta == (ValueTypeB) 0)
        {
            FatalError("M(k,k)=0 breakdown condition (beta):IDR", AMGX_ERR_INTERNAL);
        }

        // r = r - beta*G(:,k);
        axpy(G, *this->m_r, -beta, k * N, 0, N);
        // x = x + beta*U(:,k);
        axpy(U, x, beta, k * N, 0, N);
        // Do we converge ?
        this->m_curr_iter = this->m_curr_iter + 1;

        if ( this->m_monitor_convergence && this->compute_norm_and_converged() )
        {
            A.setView(oldView);
            return true;
        }

        //Early exit: last iteration, no need to prepare the next one.
        if ( this->is_last_iter() )
        {
            A.setView(oldView);
            return !this->m_monitor_convergence;
        }

        // New f = P'*r (first k  components are zero)
        // if ( k < s )
        //     f(k+1:s)   = f(k+1:s) - beta*M(k+1:s,k);
        // end
        if (k < s - 1)
        {
            axpy(M, m_f, -beta, k * s + k + 1, k + 1, s - k - 1);
        }
    }/// for ends for smaller space

    //check for convergence once again. If converged just leave the function
    if ( this->m_monitor_convergence && this->compute_norm_and_converged() )
    {
        A.setView(oldView);
        return true;
    }

    copy( *this->m_r, m_v, 0, N);

    if (no_preconditioner)
    {
        ;
    }
    else
    {
        m_z.delayed_send = 1;
        m_v.delayed_send = 1;
        m_preconditioner->solve( m_v, m_z, true );
        m_z.delayed_send = 1;
        m_v.delayed_send = 1;
        copy( m_z, m_v, 0, N);
    }

    A.apply(m_v, t_idr );
    // calculate new omega
    ns = nrm2(*this->m_r, 0, N);
    nt = nrm2(t_idr, 0, N);
    ts = dotc(t_idr, *this->m_r, 0, N);
    rho = abs(ts / (nt * ns));
    this->omega = ts / (nt * nt);

    if (rho < angle)
    {
        this->omega = this->omega * angle / rho;
    }

    if (this->omega == 0)
    {
        cout << "Error happened in this->omega==0" << endl;
        exit(1);
    }

    //     r = r - omega*t;
    axpy( t_idr, *this->m_r, -(this->omega), 0, N );
    axpy( m_v, x, this->omega, 0, N );
    // No convergence so far.
    A.setView(oldView);
    return !this->m_monitor_convergence;
}

template<class T_Config>
void
IDR_Solver_Base<T_Config>::solve_finalize( VVector &b, VVector &x )
{}

template<class T_Config>
void
IDR_Solver_Base<T_Config>::printSolverParameters() const
{
    if (!no_preconditioner)
    {
        std::cout << "preconditioner: " << this->m_preconditioner->getName()
                  << " with scope name: "
                  << this->m_preconditioner->getScope() << std::endl;
    }
}

template <AMGX_VecPrecision t_vecPrec, AMGX_MatPrecision t_matPrec, AMGX_IndPrecision t_indPrec>
void IDR_Solver<TemplateConfig<AMGX_host, t_vecPrec, t_matPrec, t_indPrec> >::dot_ina_loop(const VVector &a, const VVector &b, int offseta, int offsetb, VVector &res, VVector &hres, int offsetres, int size, int k, int s)
{
    int i;

    for (i = k; i < s; i++)
    {
        (res.raw())[i + offsetres] = dotc(a, b, offseta + i * size, offsetb, size);
    }
}

template <AMGX_VecPrecision t_vecPrec, AMGX_MatPrecision t_matPrec, AMGX_IndPrecision t_indPrec>
void IDR_Solver<TemplateConfig<AMGX_device, t_vecPrec, t_matPrec, t_indPrec> >::dot_ina_loop(const VVector &a, const VVector &b, int offseta, int offsetb, VVector &res, Vector_h &hres,  int offsetres, int size, int k, int s)
{
    int i;

    for (i = k; i < s; i++)
    {
        hres.raw()[i + offsetres] = dotc(a, b, offseta + i * size, offsetb, size);
    }

    cudaMemcpy((void *) res.raw(),       (void *) hres.raw(),       (s - k)*sizeof(ValueTypeB),   cudaMemcpyHostToDevice);
}

template <AMGX_VecPrecision t_vecPrec, AMGX_MatPrecision t_matPrec, AMGX_IndPrecision t_indPrec>
void IDR_Solver<TemplateConfig<AMGX_device, t_vecPrec, t_matPrec, t_indPrec> >::gemv_div(bool trans, const VVector &A, const VVector &x, VVector &y, int m, int n,
        ValueTypeB alpha, ValueTypeB beta, int incx, int incy, int lda,
        int offsetA, int offsetx, int offsety, VVector &nume, int k, int s, ValueTypeB *ratio, Vector_h &svec_chk)
{
    int j, col;
    ValueTypeB numer, gemv_res;
    ValueTypeB denom;

    if (s == 1)
    {
        gemv_res = dotc(A, x, 0, k * m, m);
        cudaMemcpy((void *) & (y.raw()[k * s + k]),       (void *) &gemv_res,       sizeof(ValueTypeB),   cudaMemcpyHostToDevice);

        if (gemv_res != (ValueTypeB) 0)
        {
            cudaMemcpy((void *) &numer,
                       (void *) & ((nume.raw())[k]),
                       sizeof(ValueTypeB),   cudaMemcpyDeviceToHost);
            *ratio = numer / gemv_res;
        }
        else
        {
            *ratio = (ValueTypeB) 0;
        }
    }
    else
    {
        for (col = k, j = 0; j < s - k; j++, col++)
        {
            (svec_chk.raw())[j] = dotc(A, x, col * m, k * m, m);
        }

        cudaMemcpy((void *) & (y.raw())[k * s + k],       (void *) svec_chk.raw(),       (s - k)*sizeof(ValueTypeB),   cudaMemcpyHostToDevice);
        cudaMemcpy((void *) &denom, (void *) & (y.raw())[k + s * k],  sizeof(ValueTypeB),   cudaMemcpyDeviceToHost);

        if (denom != (ValueTypeB) 0)
        {
            cudaMemcpy((void *) &numer,
                       (void *) & ((nume.raw())[k]),
                       sizeof(ValueTypeB),   cudaMemcpyDeviceToHost);
            *ratio = numer / denom;
        }
        else
        {
            *ratio = (ValueTypeB) 0;
        }
    }
}

template <AMGX_VecPrecision t_vecPrec, AMGX_MatPrecision t_matPrec, AMGX_IndPrecision t_indPrec>
void IDR_Solver<TemplateConfig<AMGX_host, t_vecPrec, t_matPrec, t_indPrec> >::gemv_div(bool trans, const VVector &A, const VVector &x, VVector &y, int m, int n,
        ValueTypeB alpha, ValueTypeB beta, int incx, int incy, int lda,
        int offsetA, int offsetx, int offsety, VVector &nume, int k, int s, ValueTypeB *ratio, Vector_h &svec)
{
    int j, col;
    ValueTypeB gemv_res;
    ValueTypeB denom;

    if (s == 1)
    {
        gemv_res = dotc(A, x, 0, k * m, m);
        (y.raw()[k * s + k]) = gemv_res;

        if (gemv_res != (ValueTypeB) 0)
        {
            *ratio = ((nume.raw())[k]) / gemv_res;
        }
        else
        {
            *ratio = (ValueTypeB) 0;
        }
    }
    else
    {
        for (col = k, j = 0; j < s - k; j++, col++)
        {
            (y.raw())[k + s * k + j] = dotc(A, x, col * m, k * m, m);
        }

        denom = (y.raw())[k + s * k];

        if (denom != (ValueTypeB) 0)
        {
            *ratio = ((nume.raw())[k]) / denom;
        }
        else
        {
            *ratio = (ValueTypeB) 0;
        }
    }
}


template <AMGX_VecPrecision t_vecPrec, AMGX_MatPrecision t_matPrec, AMGX_IndPrecision t_indPrec>
void IDR_Solver<TemplateConfig<AMGX_device, t_vecPrec, t_matPrec, t_indPrec> >::dotc_div(VVector &a, VVector &b, int offseta, int offsetb, int size, VVector &denom, int i, int s, ValueTypeB *ratio)
{
    ValueTypeB dnr;
    cudaMemcpy((void *) &dnr, (void *) & (denom.raw())[i + s * i],  sizeof(ValueTypeB),   cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();

    if (dnr != (ValueTypeB) 0)
    {
        *ratio = dotc(a, b, offseta, offsetb, size) / dnr;
    }
    else
    {
        *ratio = (ValueTypeB) 0;
    }
}


template <AMGX_VecPrecision t_vecPrec, AMGX_MatPrecision t_matPrec, AMGX_IndPrecision t_indPrec>
void IDR_Solver<TemplateConfig<AMGX_host, t_vecPrec, t_matPrec, t_indPrec> >::dotc_div(VVector &a, VVector &b, int offseta, int offsetb, int size, VVector &denom, int i, int s, ValueTypeB *ratio)
{
    ValueTypeB alpha_iter;

    if ((denom.raw())[i * s + i] != (ValueTypeB) 0)
    {
        alpha_iter = dotc(a, b, offseta, offsetb, size) / denom[i * s + i];
        *ratio = alpha_iter;
    }
    else
    {
        *ratio = (ValueTypeB) 0;
    }
}


template <AMGX_VecPrecision t_vecPrec, AMGX_MatPrecision t_matPrec, AMGX_IndPrecision t_indPrec>
void IDR_Solver<TemplateConfig<AMGX_device, t_vecPrec, t_matPrec, t_indPrec> >::setup_arrays(VVector &P, VVector &M, VVector &b, VVector &x, Vector_h &hbuff,
        int s, int N)
{
    int i;

    for (i = 0; i < s; i++) { (hbuff.raw())[i * s + i] = (ValueTypeB) 1.0; }

    cudaMemcpy((void *)M.raw(), (void *)hbuff.raw(), s * s * sizeof(ValueTypeB), cudaMemcpyHostToDevice);

    if (s == 1)
    {
        cudaMemcpy((void *)P.raw(), (void *)b.raw(), N * s * sizeof(ValueTypeB), cudaMemcpyDeviceToDevice);
    }
    else
    {
        srand(0);

        for (i = 0; i < N * s; i++)
        {
            (hbuff.raw())[i] = (ValueTypeB) rand() / (ValueTypeB (RAND_MAX));
        }
        cudaMemcpy((void *)P.raw(), (void *)hbuff.raw(), N * s * sizeof(ValueTypeB), cudaMemcpyHostToDevice);
    }

//
}

template <AMGX_VecPrecision t_vecPrec, AMGX_MatPrecision t_matPrec, AMGX_IndPrecision t_indPrec>
void IDR_Solver<TemplateConfig<AMGX_host, t_vecPrec, t_matPrec, t_indPrec> >::setup_arrays(VVector &P, VVector &M, VVector &b, VVector &x, VVector &hbuff,
        int s, int N)
{
    int i;

    for (i = 0; i < s; i++) { (M.raw())[i * s + i] = (ValueTypeB) 1.0; }

    if (s == 1)
    {
        copy(b, P, 0, N);    // copying b into P if s=1;
    }
    else
    {
        srand(0);

        for (i = 0; i < N * s; i++)
        {
            (P.raw())[i] = (ValueTypeB) rand() / (ValueTypeB (RAND_MAX));
        }
    }
}
/****************************************
 * Explict instantiations
 ***************************************/

#define AMGX_CASE_LINE(CASE) template class IDR_Solver_Base<TemplateMode<CASE>::Type>;
AMGX_FORALL_BUILDS(AMGX_CASE_LINE)
#undef AMGX_CASE_LINE

#define AMGX_CASE_LINE(CASE) template class IDR_Solver<TemplateMode<CASE>::Type>;
AMGX_FORALL_BUILDS(AMGX_CASE_LINE)
#undef AMGX_CASE_LINE

} // namespace idr_solver
} // namespace amgx
