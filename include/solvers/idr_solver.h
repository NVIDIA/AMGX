// SPDX-FileCopyrightText: 2011 - 2025 NVIDIA CORPORATION. All Rights Reserved.
//
// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#include<solvers/solver.h>

namespace amgx
{

namespace idr_solver
{

template <class T_Config> class IDR_Solver;

template<class T_Config>
class IDR_Solver_Base : public Solver<T_Config>
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
        // Temporary vectors needed for the computation.
        VVector m_z, m_Ax, m_v, m_f, c, tempg, tempu, temp, t_idr;
        VVector G, U, P, M;
        Vector_h h_chk, svec_chk;
        int m_buffer_N;
        // The dot product between z and the residual.
        ValueTypeB omega;
        int s;
        bool no_preconditioner;
        Solver<TConfig> *m_preconditioner;

    public:
        // Constructor.
        IDR_Solver_Base( AMG_Config &cfg, const std::string &cfg_scope);

        ~IDR_Solver_Base();

        // Print the solver parameters
        void printSolverParameters() const;

        // Setup the solver
        void solver_setup(bool reuse_matrix_structure);

        bool isColoringNeeded() const { if (m_preconditioner != NULL) return m_preconditioner->isColoringNeeded(); return false; }

        void getColoringScope( std::string &cfg_scope_for_coloring) const { if (m_preconditioner != NULL) m_preconditioner->getColoringScope(cfg_scope_for_coloring); }

        bool getReorderColsByColorDesired() const { if (m_preconditioner != NULL) return m_preconditioner->getReorderColsByColorDesired(); return false; }

        bool getInsertDiagonalDesired() const { if (m_preconditioner != NULL) return m_preconditioner->getInsertDiagonalDesired(); return false; }

        // Initialize the solver before running the iterations.
        void solve_init( VVector &b, VVector &x, bool xIsZero );
        // Run a single iteration. Compute the residual and its norm and decide convergence.
        AMGX_STATUS solve_iteration( VVector &b, VVector &x, bool xIsZero );
        // Finalize the solver after running the iterations.
        void solve_finalize( VVector &b, VVector &x );

    protected:
        virtual void gemv_div(bool trans, const VVector &A, const VVector &x, VVector &y, int m, int n,
                              ValueTypeB alpha, ValueTypeB beta, int incx, int incy, int lda,
                              int offsetA, int offsetx, int offsety, VVector &nume, int k, int s, ValueTypeB *ratio, Vector_h &svec) = 0;

        virtual void dotc_div(VVector &a, VVector &b, int offseta, int offsetb, int size, VVector &denom, int i, int s, ValueTypeB *ratio) = 0;
        virtual void dot_ina_loop(const VVector &a, const VVector &b, int offseta, int offsetb, VVector &res, Vector_h &hres, int offsetres, int size, int k, int s) = 0;
        virtual void setup_arrays(VVector &P, VVector &M, VVector &b, VVector &x, Vector_h &hdevbuff, int s, int N) = 0;

};

// ----------------------------
//  specialization for host
// ----------------------------

template <AMGX_VecPrecision t_vecPrec, AMGX_MatPrecision t_matPrec, AMGX_IndPrecision t_indPrec>
class IDR_Solver< TemplateConfig<AMGX_host, t_vecPrec, t_matPrec, t_indPrec> > : public IDR_Solver_Base< TemplateConfig<AMGX_host, t_vecPrec, t_matPrec, t_indPrec> >
{
    public:
        typedef TemplateConfig<AMGX_host, t_vecPrec, t_matPrec, t_indPrec> TConfig_h;
        typedef Matrix<TConfig_h> Matrix_h;
        typedef typename TConfig_h::MatPrec ValueTypeA;
        typedef typename TConfig_h::VecPrec ValueTypeB;
        typedef Vector<TConfig_h> VVector;
        typedef typename TConfig_h::IndPrec IndexType;
        typedef typename Matrix<TConfig_h>::MVector MVector;
        typedef Vector<TConfig_h> Vector_h;

        IDR_Solver(AMG_Config &cfg, const std::string &cfg_scope) : IDR_Solver_Base<TConfig_h>(cfg, cfg_scope) {}
    private:
        void gemv_div(bool trans, const VVector &A, const VVector &x, VVector &y, int m, int n,
                      ValueTypeB alpha, ValueTypeB beta, int incx, int incy, int lda,
                      int offsetA, int offsetx, int offsety, VVector &nume, int k, int s, ValueTypeB *ratio, Vector_h &svec);


        void dotc_div(VVector &a, VVector &b, int offseta, int offsetb, int size, VVector &denom, int i, int s, ValueTypeB *ratio);
        void dot_ina_loop(const VVector &a, const VVector &b, int offseta, int offsetb, VVector &res, VVector &hres, int offsetres, int size, int k, int s);
        void setup_arrays(VVector &P, VVector &M, VVector &b, VVector &x, Vector_h &hdevbuff, int s, int N);
};

// ----------------------------
//  specialization for device
// ----------------------------

template <AMGX_VecPrecision t_vecPrec, AMGX_MatPrecision t_matPrec, AMGX_IndPrecision t_indPrec>
class IDR_Solver< TemplateConfig<AMGX_device, t_vecPrec, t_matPrec, t_indPrec> >: public IDR_Solver_Base< TemplateConfig<AMGX_device, t_vecPrec, t_matPrec, t_indPrec> >
{
    public:
        typedef TemplateConfig<AMGX_device, t_vecPrec, t_matPrec, t_indPrec> TConfig_d;
        typedef TemplateConfig<AMGX_host, t_vecPrec, t_matPrec, t_indPrec> TConfig_h;
        typedef Matrix<TConfig_d> Matrix_d;
        typedef typename TConfig_d::MatPrec ValueTypeA;
        typedef typename TConfig_d::VecPrec ValueTypeB;
        typedef typename TConfig_d::IndPrec IndexType;
        typedef Vector<TConfig_d> VVector;
        typedef typename Matrix_d ::IVector IVector;
        typedef typename Matrix<TConfig_d>::MVector MVector;
        typedef Vector<TConfig_h> Vector_h;

        IDR_Solver(AMG_Config &cfg, const std::string &cfg_scope) : IDR_Solver_Base<TConfig_d>(cfg, cfg_scope) {}
    private:
        void gemv_div(bool trans, const VVector &A, const VVector &x, VVector &y, int m, int n,
                      ValueTypeB alpha, ValueTypeB beta, int incx, int incy, int lda,
                      int offsetA, int offsetx, int offsety, VVector &nume, int k, int s, ValueTypeB *ratio, Vector_h &svec);

        void dotc_div(VVector &a, VVector &b, int offseta, int offsetb, int size, VVector &denom, int i, int s, ValueTypeB *ratio);
        void dot_ina_loop(const VVector &a, const VVector &b, int offseta, int offsetb, VVector &res, Vector_h &hres, int offsetres, int size, int k, int s);
        void setup_arrays(VVector &P, VVector &M, VVector &b, VVector &x, Vector_h &hdevbuff, int s, int N);
};


template<class T_Config>
class IDR_SolverFactory : public SolverFactory<T_Config>
{
    public:
        Solver<T_Config> *create( AMG_Config &cfg, const std::string &cfg_scope, ThreadManager *tmng ) { return new IDR_Solver<T_Config>( cfg, cfg_scope); }
};

} // namespace idr_solver
} // namespace amgx
