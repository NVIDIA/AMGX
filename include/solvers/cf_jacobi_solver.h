// SPDX-FileCopyrightText: 2011 - 2024 NVIDIA CORPORATION. All Rights Reserved.
//
// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#include<solvers/solver.h>

namespace amgx
{
namespace cf_jacobi_solver
{

enum SmoothingOrder
{
    CF_CF,
    CF_FC,
    CF_FCF,
    CF_CFC
};

template <class T_Config> class CFJacobiSolver;

template<class T_Config>
class CFJacobiSolver_Base : public Solver<T_Config>
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
        typedef typename Matrix<TConfig>::IVector IVector;

    private:
        void computeDinv( Matrix<T_Config> &A);

    protected:
        ValueTypeB weight;
        SmoothingOrder mode;
        MVector Dinv;
        VVector t_res; // temporary storage for latency hiding case, lazy allocation
        VVector y;     // permanent temporary storage for 1x1 solver, lazy allocation
        int num_coarse; // number of coarse points
        IVector c_rows;  // cf_rows[0..num_coarse-1] - rows numbers corresponding to coarse points
        IVector f_rows;                  // cf_rows[num_coarse..N-1] - rows numbers corresponding to fine points

        virtual void find_diag( const Matrix<T_Config> &A ) = 0;
        virtual void smooth_1x1(Matrix<T_Config> &A, VVector &b, VVector &x, SmoothingOrder order, ViewType separation_flags) = 0;
        virtual void smooth_with_0_initial_guess_1x1( Matrix<T_Config> &A, VVector &b, VVector &x, SmoothingOrder order, ViewType separation_flags) = 0;
        virtual void computeDinv_1x1(const Matrix<T_Config> &A) = 0;

    public:
        // Constructor.
        CFJacobiSolver_Base( AMG_Config &cfg, const std::string &cfg_scope);

        // Destructor
        ~CFJacobiSolver_Base();

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
        AMGX_STATUS solve_iteration( VVector &b, VVector &x, bool xIsZero );
        // Finalize the solver after running the iterations.
        void solve_finalize( VVector &b, VVector &x );
};

// ----------------------------
//  specialization for host
// ----------------------------

template <AMGX_VecPrecision t_vecPrec, AMGX_MatPrecision t_matPrec, AMGX_IndPrecision t_indPrec>
class CFJacobiSolver< TemplateConfig<AMGX_host, t_vecPrec, t_matPrec, t_indPrec> > : public CFJacobiSolver_Base< TemplateConfig<AMGX_host, t_vecPrec, t_matPrec, t_indPrec> >
{
    public:
        typedef TemplateConfig<AMGX_host, t_vecPrec, t_matPrec, t_indPrec> TConfig_h;
        typedef Matrix<TConfig_h> Matrix_h;
        typedef typename TConfig_h::MatPrec ValueTypeA;
        typedef typename TConfig_h::VecPrec ValueTypeB;
        typedef Vector<TConfig_h> VVector;
        typedef typename TConfig_h::IndPrec IndexType;
        typedef typename Matrix<TConfig_h>::MVector MVector;
        typedef typename Matrix<TConfig_h>::IVector IVector;

        CFJacobiSolver(AMG_Config &cfg, const std::string &cfg_scope) : CFJacobiSolver_Base<TConfig_h>(cfg, cfg_scope) {}
    private:
        void smooth_1x1(Matrix_h &A, VVector &b, VVector &x, SmoothingOrder order, ViewType separation_flags);
        void smooth_with_0_initial_guess_1x1( Matrix_h &A, VVector &b, VVector &x, SmoothingOrder order, ViewType separation_flags);
        void computeDinv_1x1(const Matrix_h &A);
        void find_diag( const Matrix_h &A );

};

// ----------------------------
//  specialization for device
// ----------------------------

template <AMGX_VecPrecision t_vecPrec, AMGX_MatPrecision t_matPrec, AMGX_IndPrecision t_indPrec>
class CFJacobiSolver< TemplateConfig<AMGX_device, t_vecPrec, t_matPrec, t_indPrec> >: public CFJacobiSolver_Base< TemplateConfig<AMGX_device, t_vecPrec, t_matPrec, t_indPrec> >
{
    public:
        typedef TemplateConfig<AMGX_device, t_vecPrec, t_matPrec, t_indPrec> TConfig_d;
        typedef Matrix<TConfig_d> Matrix_d;
        typedef typename TConfig_d::MatPrec ValueTypeA;
        typedef typename TConfig_d::VecPrec ValueTypeB;
        typedef typename TConfig_d::IndPrec IndexType;
        typedef Vector<TConfig_d> VVector;
        typedef typename Matrix<TConfig_d>::MVector MVector;
        typedef typename Matrix<TConfig_d>::IVector IVector;

        CFJacobiSolver(AMG_Config &cfg, const std::string &cfg_scope) : CFJacobiSolver_Base<TConfig_d>(cfg, cfg_scope) {}
    private:

        void smooth_1x1(Matrix_d &A, VVector &b, VVector &x, SmoothingOrder order, ViewType separation_flags);
        void smooth_with_0_initial_guess_1x1( Matrix_d &A, VVector &b, VVector &x, SmoothingOrder order, ViewType separation_flags);
        void computeDinv_1x1(const Matrix_d &A);
        void find_diag( const Matrix_d &A );

};

template<class T_Config>
class CFJacobiSolverFactory : public SolverFactory<T_Config>
{
    public:
        Solver<T_Config> *create( AMG_Config &cfg, const std::string &cfg_scope, ThreadManager *tmng ) { return new CFJacobiSolver<T_Config>( cfg, cfg_scope); }
};

} // namespace block_jacobi
} // namespace amgx
