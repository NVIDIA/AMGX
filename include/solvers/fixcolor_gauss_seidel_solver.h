// SPDX-FileCopyrightText: 2011 - 2025 NVIDIA CORPORATION. All Rights Reserved.
//
// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#include <string>
#include<solvers/solver.h>
#include <matrix_coloring/matrix_coloring.h>
#include <basic_types.h>
#include <matrix.h>

namespace amgx
{
namespace fixcolor_gauss_seidel_solver
{

template <class T_Config> class FixcolorGaussSeidelSolver;

template<class T_Config>
class FixcolorGaussSeidelSolver_Base : public Solver<T_Config>
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
        typedef typename Matrix<TConfig>::MVector MVector;

    private:

        void computeDinv(Matrix<T_Config> &A);

    protected:

        virtual void smooth_BxB(Matrix<T_Config> &A, VVector &b, VVector &x, ViewType separation_flag) = 0;
        virtual void smooth_1x1(const Matrix<T_Config> &A, const VVector &b, VVector &x, ViewType separation_flag) = 0;
        virtual void smooth_3x3(const Matrix<T_Config> &A, const VVector &b, VVector &x, ViewType separation_flag) = 0;
        virtual void smooth_4x4(const Matrix<T_Config> &A, const VVector &b, VVector &x, ViewType separation_flag) = 0;
        virtual void computeDinv_1x1(const Matrix<T_Config> &A) = 0;
        virtual void computeDinv_2x2(const Matrix<T_Config> &A) = 0;
        virtual void computeDinv_3x3(const Matrix<T_Config> &A) = 0;
        virtual void computeDinv_4x4(const Matrix<T_Config> &A) = 0;
        virtual void computeDinv_5x5(const Matrix<T_Config> &A) = 0;
        virtual void computeDinv_bxb(const Matrix<T_Config> &A, const int bsize) = 0;

        ValueTypeB weight;
        MVector Dinv;
        int symFlag;
        int use_bsrxmv;
        int num_colors;

        MatrixColoring<TConfig> *matrix_coloring_scheme;

    public:
        // Constructor.
        FixcolorGaussSeidelSolver_Base( AMG_Config &cfg, const std::string &cfg_scope);

        // Destructor
        ~FixcolorGaussSeidelSolver_Base();

        bool isResidualNeeded() const { return false;}

        bool isColoringNeeded() const { return false; }

        bool getReorderColsByColorDesired() const { return false; }

        bool getInsertDiagonalDesired() const { return false; }

        // Print the solver parameters
        void printSolverParameters() const;

        // Setup the solver
        void solver_setup (bool reuse_matrix_structure);

        // Initialize the solver before running the iterations.
        void solve_init( VVector &b, VVector &x, bool xIsZero);

        // Run a single iteration. Compute the residual and its norm and decide convergence.
        AMGX_STATUS solve_iteration( VVector &b, VVector &x, bool xIsZero);

        // Finalize the solver after running the iterations.
        void solve_finalize( VVector &b, VVector &x);
};

// ----------------------------
//  specialization for host
// ----------------------------

template <AMGX_VecPrecision t_vecPrec, AMGX_MatPrecision t_matPrec, AMGX_IndPrecision t_indPrec>
class FixcolorGaussSeidelSolver< TemplateConfig<AMGX_host, t_vecPrec, t_matPrec, t_indPrec> > : public FixcolorGaussSeidelSolver_Base< TemplateConfig<AMGX_host, t_vecPrec, t_matPrec, t_indPrec> >
{
    public:
        typedef TemplateConfig<AMGX_host, t_vecPrec, t_matPrec, t_indPrec> TConfig_h;
        typedef Matrix<TConfig_h> Matrix_h;
        typedef typename TConfig_h::MatPrec ValueTypeA;
        typedef typename TConfig_h::VecPrec ValueTypeB;
        typedef Vector<TConfig_h> VVector;
        typedef typename TConfig_h::IndPrec IndexType;
        typedef typename Matrix<TConfig_h>::MVector MVector;
        FixcolorGaussSeidelSolver(AMG_Config &cfg, const std::string &cfg_scope) : FixcolorGaussSeidelSolver_Base<TConfig_h>(cfg, cfg_scope) {}
    private:
        void smooth_BxB(Matrix_h &A, VVector &b, VVector &x, ViewType separation_flag);
        void smooth_1x1(const Matrix_h &A, const VVector &b, VVector &x, ViewType separation_flag);
        void smooth_3x3(const Matrix_h &A, const VVector &b, VVector &x, ViewType separation_flag);
        void smooth_4x4(const Matrix_h &A, const VVector &b, VVector &x, ViewType separation_flag);
        void computeDinv_1x1(const Matrix_h &A);
        void computeDinv_2x2(const Matrix_h &A);
        void computeDinv_3x3(const Matrix_h &A);
        void computeDinv_4x4(const Matrix_h &A);
        void computeDinv_5x5(const Matrix_h &A);
        void computeDinv_bxb(const Matrix_h &A, const int bsize);


};

// ----------------------------
//  specialization for device
// ----------------------------

template <AMGX_VecPrecision t_vecPrec, AMGX_MatPrecision t_matPrec, AMGX_IndPrecision t_indPrec>
class FixcolorGaussSeidelSolver< TemplateConfig<AMGX_device, t_vecPrec, t_matPrec, t_indPrec> >: public FixcolorGaussSeidelSolver_Base< TemplateConfig<AMGX_device, t_vecPrec, t_matPrec, t_indPrec> >
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

        FixcolorGaussSeidelSolver(AMG_Config &cfg, const std::string &cfg_scope) : FixcolorGaussSeidelSolver_Base<TConfig_d>(cfg, cfg_scope) {}
    private:
        void smooth_BxB(Matrix_d &A, VVector &b, VVector &x, ViewType separation_flag);
        void smooth_1x1(const Matrix_d &A, const VVector &b, VVector &x, ViewType separation_flag);
        void smooth_3x3(const Matrix_d &A, const VVector &b, VVector &x, ViewType separation_flag);
        void smooth_4x4(const Matrix_d &A, const VVector &b, VVector &x, ViewType separation_flag);
        void computeDinv_1x1(const Matrix_d &A);
        void computeDinv_2x2(const Matrix_d &A);
        void computeDinv_3x3(const Matrix_d &A);
        void computeDinv_4x4(const Matrix_d &A);
        void computeDinv_5x5(const Matrix_d &A);
        void computeDinv_bxb(const Matrix_d &A, const int bsize);
};

template<class T_Config>
class FixcolorGaussSeidelSolverFactory : public SolverFactory<T_Config>
{
    public:
        Solver<T_Config> *create( AMG_Config &cfg, const std::string &cfg_scope, ThreadManager *tmng ) { return new FixcolorGaussSeidelSolver<T_Config>( cfg, cfg_scope); }
};

} // namespace multicolor_dilu
} // namespace amgx
