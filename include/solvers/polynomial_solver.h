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

#include<solvers/solver.h>
#include <basic_types.h>

#ifdef _WIN32
#pragma warning (push)
#pragma warning (disable : 4244 4267 4521)
#endif
#include <cusp/relaxation/polynomial.h>
#ifdef _WIN32
#pragma warning (pop)
#endif


namespace amgx
{
namespace polynomial_solver
{

template <class T_Config> class PolynomialSolver;

template<class T_Config>
class PolynomialSolverBase: public Solver<T_Config>
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
        typedef Vector<T_Config> VVector;
        typedef Vector<TemplateConfig<AMGX_host, vecPrec, matPrec, indPrec> > Vector_h;

        typedef typename T_Config::IndPrec IndexType;

    protected:
        Matrix<T_Config> *m_explicit_A;

        IndexType ndeg0;
        ValueTypeA k[6];
        VVector R, V0, V, Rbar, Sn, Dinv;
        cusp::relaxation::polynomial<ValueTypeA, typename T_Config::MemSpace> poly;
        virtual void smooth_1x1(const Matrix<T_Config> &A, const VVector &b, VVector &x) = 0;
        virtual void smooth_with_0_initial_guess_1x1(const Matrix<T_Config> &A, const VVector &b, VVector &x) = 0;
        virtual void smooth_common_sqblocks(Matrix<T_Config> &A, const VVector &b, VVector &x) = 0;

    public:
        // Constructor.
        PolynomialSolverBase( AMG_Config &cfg, const std::string &cfg_scope);

        // Destructor
        ~PolynomialSolverBase();

        bool is_residual_needed() const { return false; }

        // Print the solver parameters
        void printSolverParameters() const;

        // Setup the solver
        void solver_setup(bool reuse_matrix_structure);

        bool isColoringNeeded() const { return false; }

        bool getReorderColsByColorDesired() const { return false; }

        bool getInsertDiagonalDesired() const { return false; }

        // Initialize the solver before running the iterations.
        void solve_init( VVector &b, VVector &x, bool xIsZero );
        // Run a single iteration. Compute the residual and its norm and decide convergence.
        bool solve_iteration( VVector &b, VVector &x, bool xIsZero );
        // Finalize the solver after running the iterations.
        void solve_finalize( VVector &b, VVector &x );
};


// specialization for host
template <AMGX_VecPrecision t_vecPrec, AMGX_MatPrecision t_matPrec, AMGX_IndPrecision t_indPrec>
class PolynomialSolver< TemplateConfig<AMGX_host, t_vecPrec, t_matPrec, t_indPrec> > : public PolynomialSolverBase< TemplateConfig<AMGX_host, t_vecPrec, t_matPrec, t_indPrec> >
{
    public:
        typedef TemplateConfig<AMGX_host, t_vecPrec, t_matPrec, t_indPrec> TConfig_h;
        typedef Matrix<TConfig_h> Matrix_h;
        typedef typename TConfig_h::MatPrec ValueTypeA;
        typedef typename TConfig_h::VecPrec ValueTypeB;
        typedef Vector<TConfig_h> VVector;
        typedef typename TConfig_h::IndPrec IndexType;
        PolynomialSolver(AMG_Config &cfg, const std::string &cfg_scope) : PolynomialSolverBase<TConfig_h>(cfg, cfg_scope) {;};
        //PolynomialSmoother(AMG_Config& cfg, const std::string &cfg_scope, Matrix_h& Ainit) : BlockJacobiSmootherBase<TConfig_h>(cfg, cfg_scope, Ainit) {  computeDinv(Ainit); }
    protected:
        void smooth_1x1(const Matrix_h &A, const VVector &b, VVector &x);
        void smooth_with_0_initial_guess_1x1(const Matrix_h &A, const VVector &b, VVector &x);
        void smooth_common_sqblocks(Matrix_h &A, const VVector &b, VVector &x);
};

// specialization for device
template <AMGX_VecPrecision t_vecPrec, AMGX_MatPrecision t_matPrec, AMGX_IndPrecision t_indPrec>
class PolynomialSolver< TemplateConfig<AMGX_device, t_vecPrec, t_matPrec, t_indPrec> >: public PolynomialSolverBase< TemplateConfig<AMGX_device, t_vecPrec, t_matPrec, t_indPrec> >
{
    public:
        typedef TemplateConfig<AMGX_device, t_vecPrec, t_matPrec, t_indPrec> TConfig_d;
        typedef Matrix<TConfig_d> Matrix_d;
        typedef typename TConfig_d::MatPrec ValueTypeA;
        typedef typename TConfig_d::VecPrec ValueTypeB;
        typedef typename TConfig_d::IndPrec IndexType;
        typedef Vector<TConfig_d> VVector;
        PolynomialSolver(AMG_Config &cfg, const std::string &cfg_scope) : PolynomialSolverBase<TConfig_d>(cfg, cfg_scope) {};
    protected:
        void smooth_1x1(const Matrix_d &A, const VVector &b, VVector &x);
        void smooth_with_0_initial_guess_1x1(const Matrix_d &A, const VVector &b, VVector &x);
        void smooth_common_sqblocks(Matrix_d &A, const VVector &b, VVector &x);
};








template<class T_Config>
class PolynomialSolverFactory : public SolverFactory<T_Config>
{
    public:
        Solver<T_Config> *create( AMG_Config &cfg, const std::string &cfg_scope, ThreadManager *tmng ) { return new PolynomialSolver<T_Config>( cfg, cfg_scope); }
};

} // namespace polynomial-solver
} // namespace amgx
