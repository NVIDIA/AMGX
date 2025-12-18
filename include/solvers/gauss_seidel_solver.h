// SPDX-FileCopyrightText: 2011 - 2025 NVIDIA CORPORATION. All Rights Reserved.
//
// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#include<solvers/solver.h>

namespace amgx
{

template <class T_Config> class GaussSeidelSolver;

template<class T_Config>
class GaussSeidelSolver_Base : public Solver<T_Config>
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

    private:

        void find_diag(const Matrix<T_Config> &A);

    protected:
        ValueTypeB weight;
        VVector diag;

        virtual void find_diag_1x1(const Matrix<T_Config> &A) = 0;
        virtual void smooth_1x1(const Matrix<T_Config> &A, const VVector &b, VVector &x) = 0;
        virtual void smooth_with_0_initial_guess_1x1(const Matrix<T_Config> &A, const VVector &b, VVector &x) = 0;

    public:
        // Constructor.
        GaussSeidelSolver_Base( AMG_Config &cfg, const std::string &cfg_scope);

        // Destructor
        ~GaussSeidelSolver_Base();

        bool is_residual_needed() const { return false; }

        // Print the solver parameters
        void printSolverParameters() const;
        // Setup the solver
        void solver_setup(bool reuse_matrix_structure);

        bool isColoringNeeded( ) const { return false; }

        bool getReorderColsByColorDesired() const { return false; }

        bool getInsertDiagonalDesired() const { return false; }

        // Initialize the solver before running the iterations.
        void solve_init( VVector &b, VVector &x, bool xIsZero );
        // Run a single iteration. Compute the residual and its norm and decide convergence.
        AMGX_STATUS solve_iteration( VVector &b, VVector &x, bool xIsZero );
        // Finalize the solver after running the iterations.
        void solve_finalize( VVector &b, VVector &x );
};

// ----------------------------
//  specialization for host
// ----------------------------

template <AMGX_VecPrecision t_vecPrec, AMGX_MatPrecision t_matPrec, AMGX_IndPrecision t_indPrec>
class GaussSeidelSolver< TemplateConfig<AMGX_host, t_vecPrec, t_matPrec, t_indPrec> > : public GaussSeidelSolver_Base< TemplateConfig<AMGX_host, t_vecPrec, t_matPrec, t_indPrec> >
{
    public:
        typedef TemplateConfig<AMGX_host, t_vecPrec, t_matPrec, t_indPrec> TConfig_h;
        typedef Matrix<TConfig_h> Matrix_h;
        typedef typename TConfig_h::MatPrec ValueTypeA;
        typedef typename TConfig_h::VecPrec ValueTypeB;
        typedef Vector<TConfig_h> VVector;
        typedef typename TConfig_h::IndPrec IndexType;
        GaussSeidelSolver(AMG_Config &cfg, const std::string &cfg_scope) : GaussSeidelSolver_Base<TConfig_h>(cfg, cfg_scope) {}
    private:
        void find_diag_1x1(const Matrix_h &A);
        void smooth_1x1(const Matrix_h &A, const VVector &b, VVector &x);
        void smooth_with_0_initial_guess_1x1(const Matrix_h &A, const VVector &b, VVector &x);
};

// ----------------------------
//  specialization for device
// ----------------------------

template <AMGX_VecPrecision t_vecPrec, AMGX_MatPrecision t_matPrec, AMGX_IndPrecision t_indPrec>
class GaussSeidelSolver< TemplateConfig<AMGX_device, t_vecPrec, t_matPrec, t_indPrec> >: public GaussSeidelSolver_Base< TemplateConfig<AMGX_device, t_vecPrec, t_matPrec, t_indPrec> >
{
    public:
        typedef TemplateConfig<AMGX_device, t_vecPrec, t_matPrec, t_indPrec> TConfig_d;
        typedef Matrix<TConfig_d> Matrix_d;
        typedef typename TConfig_d::MatPrec ValueTypeA;
        typedef typename TConfig_d::VecPrec ValueTypeB;
        typedef typename TConfig_d::IndPrec IndexType;
        typedef Vector<TConfig_d> VVector;
        typedef typename Matrix_d ::IVector IVector;

        GaussSeidelSolver(AMG_Config &cfg, const std::string &cfg_scope) : GaussSeidelSolver_Base<TConfig_d>(cfg, cfg_scope) {}
    private:
        void find_diag_1x1(const Matrix_d &A);
        void smooth_1x1(const Matrix_d &A, const VVector &b, VVector &x);
        void smooth_with_0_initial_guess_1x1(const Matrix_d &A, const VVector &b, VVector &x);
};

template<class T_Config>
class GaussSeidelSolverFactory : public SolverFactory<T_Config>
{
    public:
        Solver<T_Config> *create( AMG_Config &cfg, const std::string &cfg_scope, ThreadManager *tmng ) { return new GaussSeidelSolver<T_Config>( cfg, cfg_scope); }
};

} // namespace amgx
