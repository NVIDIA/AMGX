// SPDX-FileCopyrightText: 2011 - 2024 NVIDIA CORPORATION. All Rights Reserved.
//
// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#include <string>
#include <solvers/solver.h>
#include <basic_types.h>
#include <matrix.h>

namespace amgx
{
namespace multicolor_gauss_seidel_solver
{

enum KernelMethod
{
    DEFAULT      = 0,   // default, let implementation choose appropriate kernel
    NAIVE        = 1,   // use the "naive" implementation, thread per row - previously default
    WARP_PER_ROW = 2,   // each row is processed by a warp
    T32_PER_ROW  = 3,   // each row is processed by 32 threads (specialization of N_PER_ROW as opposed to fixed WARP_PER_ROW)
    T4_PER_ROW   = 4    // each row is processed by  4 threads 
};

template <class T_Config> class MulticolorGaussSeidelSolver;

template<class T_Config>
class MulticolorGaussSeidelSolver_Base : public Solver<T_Config>
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
        virtual void smooth_1x1_naive(const Matrix<T_Config> &A, const VVector &b, VVector &x, ViewType separation_flag) = 0;
        virtual void smooth_3x3(const Matrix<T_Config> &A, const VVector &b, VVector &x, ViewType separation_flag) = 0;
        virtual void smooth_4x4(const Matrix<T_Config> &A, const VVector &b, VVector &x, ViewType separation_flag) = 0;
        virtual void computeDinv_1x1(const Matrix<T_Config> &A) = 0;
        virtual void computeDinv_2x2(const Matrix<T_Config> &A) = 0;
        virtual void computeDinv_3x3(const Matrix<T_Config> &A) = 0;
        virtual void computeDinv_4x4(const Matrix<T_Config> &A) = 0;
        virtual void computeDinv_5x5(const Matrix<T_Config> &A) = 0;
        virtual void computeDinv_bxb(const Matrix<T_Config> &A, const int bsize) = 0;

        Matrix<T_Config> *m_explicit_A;
        ValueTypeB weight;
        MVector Dinv;
        int symFlag;
        int use_bsrxmv;
        KernelMethod gs_method;
        bool m_reorder_cols_by_color_desired;
        bool m_insert_diagonal_desired;

        cudaStream_t get_aux_stream();
        cudaEvent_t m_start, m_end;

    public:
        // Constructor.
        MulticolorGaussSeidelSolver_Base( AMG_Config &cfg, const std::string &cfg_scope);

        // Destructor
        ~MulticolorGaussSeidelSolver_Base();

        bool is_residual_needed() const { return false; }

        // Print the solver parameters
        void printSolverParameters() const;

        // Setup the solver
        void solver_setup(bool reuse_matrix_structure);

        bool isColoringNeeded() const { return true; }

        void getColoringScope( std::string &cfg_scope_for_coloring) const  { cfg_scope_for_coloring = this->m_cfg_scope; }

        bool getReorderColsByColorDesired() const { return m_reorder_cols_by_color_desired;}

        bool getInsertDiagonalDesired() const { return m_insert_diagonal_desired;}

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
class MulticolorGaussSeidelSolver< TemplateConfig<AMGX_host, t_vecPrec, t_matPrec, t_indPrec> > : public MulticolorGaussSeidelSolver_Base< TemplateConfig<AMGX_host, t_vecPrec, t_matPrec, t_indPrec> >
{
    public:
        typedef TemplateConfig<AMGX_host, t_vecPrec, t_matPrec, t_indPrec> TConfig_h;
        typedef Matrix<TConfig_h> Matrix_h;
        typedef typename TConfig_h::MatPrec ValueTypeA;
        typedef typename TConfig_h::VecPrec ValueTypeB;
        typedef Vector<TConfig_h> VVector;
        typedef typename TConfig_h::IndPrec IndexType;
        typedef typename Matrix<TConfig_h>::MVector MVector;
        MulticolorGaussSeidelSolver(AMG_Config &cfg, const std::string &cfg_scope) : MulticolorGaussSeidelSolver_Base<TConfig_h>(cfg, cfg_scope) {}
    private:
        void smooth_BxB(Matrix_h &A, VVector &b, VVector &x, ViewType separation_flag);
        void smooth_1x1(const Matrix_h &A, const VVector &b, VVector &x, ViewType separation_flag);
        void smooth_1x1_naive(const Matrix_h &A, const VVector &b, VVector &x, ViewType separation_flag);
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
class MulticolorGaussSeidelSolver< TemplateConfig<AMGX_device, t_vecPrec, t_matPrec, t_indPrec> >: public MulticolorGaussSeidelSolver_Base< TemplateConfig<AMGX_device, t_vecPrec, t_matPrec, t_indPrec> >
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
        void batch_smooth_1x1(const Matrix_d &A, int batch_sz, const VVector &b, VVector &x);
        void batch_smooth_1x1_fast(const Matrix_d &A, int batch_sz, const VVector &b, VVector &x);
        MulticolorGaussSeidelSolver(AMG_Config &cfg, const std::string &cfg_scope) : MulticolorGaussSeidelSolver_Base<TConfig_d>(cfg, cfg_scope)
        {}
    private:
        void smooth_BxB(Matrix_d &A, VVector &b, VVector &x, ViewType separation_flag);
        void smooth_1x1(const Matrix_d &A, const VVector &b, VVector &x, ViewType separation_flag);
        void smooth_1x1_naive(const Matrix_d &A, const VVector &b, VVector &x, ViewType separation_flag);
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
class MulticolorGaussSeidelSolverFactory : public SolverFactory<T_Config>
{
    public:
        Solver<T_Config> *create( AMG_Config &cfg, const std::string &cfg_scope, ThreadManager *tmng ) { return new MulticolorGaussSeidelSolver<T_Config>( cfg, cfg_scope); }
};

} // namespace multicolor_dilu
} // namespace amgx
