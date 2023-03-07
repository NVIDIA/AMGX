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

#pragma once

#include <solvers/solver.h>
#include <amgx_types/util.h>

namespace amgx
{
namespace block_jacobi_solver
{

template <class T_Config> class BlockJacobiSolver;

template<class T_Config>
class BlockJacobiSolver_Base : public Solver<T_Config>
{
    public:
        typedef Solver<T_Config> Base;

        typedef T_Config TConfig;
        static const AMGX_VecPrecision vecPrec = TConfig::vecPrec;
        static const AMGX_MatPrecision matPrec = TConfig::matPrec;
        static const AMGX_IndPrecision indPrec = TConfig::indPrec;
        typedef AMG<vecPrec, matPrec, indPrec> AMG_Class;
        typedef typename T_Config::MatPrec ValueTypeA;
        typedef typename T_Config::VecPrec ValueTypeB;
        typedef typename T_Config::IndPrec IndexType;
        typedef Vector<T_Config> VVector;
        typedef Vector<TemplateConfig<AMGX_host, vecPrec, matPrec, indPrec> > Vector_h;
        typedef typename Matrix<TConfig>::MVector MVector;

    private:
        void computeDinv( Matrix<T_Config> &A);

    protected:
        double weight;
        MVector Dinv;

        VVector t_res; // temporary storage for latency hiding case, lazy allocation

        VVector y;     // permanent temporary storage for 1x1 solver, lazy allocation

        virtual void find_diag( const Matrix<T_Config> &A ) = 0;
        virtual void smooth_1x1(Matrix<T_Config> &A, VVector &b, VVector &x, ViewType separation_flags) = 0;
        virtual void smooth_with_0_initial_guess_1x1(const Matrix<T_Config> &A, const VVector &b, VVector &x, ViewType separation_flags) = 0;
        virtual void smooth_4x4(const Matrix<T_Config> &A, const VVector &b, VVector &x, ViewType separation_flags) = 0;
        virtual void smooth_with_0_initial_guess_4x4(const Matrix<T_Config> &A, const VVector &b, VVector &x, ViewType separation_flags) = 0;
        template <int bsize> void computeDinv_bxb(const Matrix<T_Config> &A);
        virtual void computeDinv_1x1(const Matrix<T_Config> &A) = 0;
        virtual void computeDinv_4x4(const Matrix<T_Config> &A) = 0;
        virtual void computeDinv_3x3(const Matrix<T_Config> &A) = 0;
        virtual void computeDinv_Big(const Matrix<T_Config> &A, const int bsize) = 0;
        virtual void smooth_BxB(Matrix<T_Config> &A, VVector &b, VVector &x, bool xIsZero, ViewType separation_flags) = 0;

    public:
        // Constructor.
        BlockJacobiSolver_Base( AMG_Config &cfg, const std::string &cfg_scope);

        // Destructor
        ~BlockJacobiSolver_Base();

        bool is_residual_needed() const { return false; }

        bool isColoringNeeded( ) const { return false; }

        bool getReorderColsByColorDesired() const { return false; }

        bool getInsertDiagonalDesired() const { return false; }

        // Print the solver parameters
        void printSolverParameters() const;
        // Setup the solver
        void solver_setup(bool reuse_matrix_structure);
        // Initialize the solver before running the iterations.
        void solve_init( VVector &b, VVector &x, bool xIsZero );
        // Run a single iteration. Compute the residual and its norm and decide convergence.
        bool solve_iteration( VVector &b, VVector &x, bool xIsZero );
        // Finalize the solver after running the iterations.
        void solve_finalize( VVector &b, VVector &x );
};

// ----------------------------
//  specialization for host
// ----------------------------

template <AMGX_VecPrecision t_vecPrec, AMGX_MatPrecision t_matPrec, AMGX_IndPrecision t_indPrec>
class BlockJacobiSolver< TemplateConfig<AMGX_host, t_vecPrec, t_matPrec, t_indPrec> > : public BlockJacobiSolver_Base< TemplateConfig<AMGX_host, t_vecPrec, t_matPrec, t_indPrec> >
{
    public:
        typedef TemplateConfig<AMGX_host, t_vecPrec, t_matPrec, t_indPrec> TConfig_h;
        typedef Matrix<TConfig_h> Matrix_h;
        typedef typename TConfig_h::MatPrec ValueTypeA;
        typedef typename TConfig_h::VecPrec ValueTypeB;
        typedef Vector<TConfig_h> VVector;
        typedef typename TConfig_h::IndPrec IndexType;
        typedef typename Matrix<TConfig_h>::MVector MVector;

        BlockJacobiSolver(AMG_Config &cfg, const std::string &cfg_scope) : BlockJacobiSolver_Base<TConfig_h>(cfg, cfg_scope) {}
    private:
        void smooth_1x1_const(const Matrix_h &A, const VVector &b, VVector &x, ViewType separation_flags);
        void smooth_1x1(Matrix_h &A, VVector &b, VVector &x, ViewType separation_flags);
        void smooth_with_0_initial_guess_1x1(const Matrix_h &A, const VVector &b, VVector &x, ViewType separation_flags);
        void smooth_4x4(const Matrix_h &A, const VVector &b, VVector &x, ViewType separation_flags);
        void smooth_with_0_initial_guess_4x4(const Matrix_h &A, const VVector &b, VVector &x, ViewType separation_flags);
        void computeDinv_1x1(const Matrix_h &A);
        void computeDinv_4x4(const Matrix_h &A);
        void computeDinv_3x3(const Matrix_h &A);
        void computeDinv_Big(const Matrix_h &A, const int bsize);
        void smooth_BxB(Matrix_h &A, VVector &b, VVector &x, bool xIsZero, ViewType separation_flags);
        void find_diag( const Matrix_h &A );

};

// ----------------------------
//  specialization for device
// ----------------------------

template <AMGX_VecPrecision t_vecPrec, AMGX_MatPrecision t_matPrec, AMGX_IndPrecision t_indPrec>
class BlockJacobiSolver< TemplateConfig<AMGX_device, t_vecPrec, t_matPrec, t_indPrec> >: public BlockJacobiSolver_Base< TemplateConfig<AMGX_device, t_vecPrec, t_matPrec, t_indPrec> >
{
    public:
        typedef TemplateConfig<AMGX_device, t_vecPrec, t_matPrec, t_indPrec> TConfig_d;
        typedef Matrix<TConfig_d> Matrix_d;
        typedef typename TConfig_d::MatPrec ValueTypeA;
        typedef typename TConfig_d::VecPrec ValueTypeB;
        typedef typename TConfig_d::IndPrec IndexType;
        typedef Vector<TConfig_d> VVector;
        typedef typename Matrix_d ::IVector IVector;
        typedef typename Matrix<TConfig_d>::MVector MVector;

        BlockJacobiSolver(AMG_Config &cfg, const std::string &cfg_scope) : BlockJacobiSolver_Base<TConfig_d>(cfg, cfg_scope) {}
    private:

        void smooth_1x1(Matrix_d &A, VVector &b, VVector &x, ViewType separation_flags);
        void smooth_with_0_initial_guess_1x1(const Matrix_d &A, const VVector &b, VVector &x, ViewType separation_flags);
        void smooth_4x4(const Matrix_d &A, const VVector &b, VVector &x, ViewType separation_flags);
        void smooth_with_0_initial_guess_4x4(const Matrix_d &A, const VVector &b, VVector &x, ViewType separation_flags);
        void computeDinv_1x1(const Matrix_d &A);
        void computeDinv_4x4(const Matrix_d &A);
        void computeDinv_3x3(const Matrix_d &A);
        void computeDinv_Big(const Matrix_d &A, const int bsize);
        void smooth_BxB(Matrix_d &A, VVector &b, VVector &x, bool firstStep, ViewType separation_flags); // first step is false when smoothing second step (exterior-interior) view in latency hiding
        void find_diag( const Matrix_d &A );

};

template<class T_Config>
class BlockJacobiSolverFactory : public SolverFactory<T_Config>
{
    public:
        Solver<T_Config> *create( AMG_Config &cfg, const std::string &cfg_scope, ThreadManager *tmng ) { return new BlockJacobiSolver<T_Config>( cfg, cfg_scope); }
};

} // namespace block_jacobi
} // namespace amgx
