// SPDX-FileCopyrightText: 2011 - 2024 NVIDIA CORPORATION. All Rights Reserved.
//
// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#include <string>
#include<solvers/solver.h>
#include <basic_types.h>
#include <matrix.h>

namespace amgx
{
namespace multicolor_ilu_solver
{

template <class T_Config> class MulticolorILUSolver;

template<class T_Config>
class MulticolorILUSolver_Base : public Solver<T_Config>
{
    public:
        typedef Solver<T_Config> Base;

        typedef T_Config TConfig;
        static const AMGX_VecPrecision vecPrec = TConfig::vecPrec;
        static const AMGX_MatPrecision matPrec = TConfig::matPrec;
        static const AMGX_IndPrecision indPrec = TConfig::indPrec;

        typedef typename T_Config::MatPrec ValueTypeA;
        typedef typename T_Config::VecPrec ValueTypeB;
        typedef typename T_Config::IndPrec IndexType;

        typedef Vector<T_Config> VVector;
        typedef typename Matrix<T_Config>::IVector IVector;
        typedef Vector<TemplateConfig<AMGX_host, vecPrec, matPrec, indPrec> > Vector_h;

    private:

        // Setup the solver
        void pre_setup();

        //void reorderRowsAndColsByColor(Matrix<T_Config> &M, IVector &color_permutation_vector, bool permute_values_flag);
        //void reorderColsByColor(Matrix<T_Config> &M, IVector &color_permutation_vector, bool permute_values_falg);
        //void reorderColsByColor(Matrix<T_Config> &M, bool permute_values_flag);

    protected:

        virtual void computeLUSparsityPattern(void) = 0;
        virtual void computeAtoLUmapping(void) = 0;
        virtual void fillLUValuesWithAValues(void) = 0;
        virtual void computeLUFactors(void) = 0;
        virtual void smooth_4x4(const VVector &b, VVector &x, bool xIsZero) = 0;
        virtual void smooth_bxb(const VVector &b, VVector &x, bool xIsZero) = 0;

        Matrix<T_Config> *m_explicit_A;
        int m_use_bsrxmv;
        IndexType m_sparsity_level;
        ValueTypeB m_weight;
        IVector m_A_to_LU_mapping;
        Matrix<TConfig> m_LU;
        void *sparsity_wk;
        bool m_reorder_cols_by_color_desired;
        bool m_insert_diagonal_desired;

        VVector m_delta, m_Delta;

    public:
        // Constructor.
        MulticolorILUSolver_Base( AMG_Config &cfg, const std::string &cfg_scope);

        // Destructor
        virtual ~MulticolorILUSolver_Base();

        bool is_residual_needed() const { return false; }

        // Print the solver parameters
        void printSolverParameters() const;
        // Setup the solver
        void solver_setup(bool reuse_matrix_structure);

        bool isColoringNeeded() const { return true; }

        void getColoringScope( std::string &cfg_scope_for_coloring) const { cfg_scope_for_coloring = this->m_cfg_scope; }

        bool getReorderColsByColorDesired() const { return m_reorder_cols_by_color_desired;}

        bool getInsertDiagonalDesired() const { return m_insert_diagonal_desired;}

        // Initialize the solver before running the iterations.
        void solve_init( VVector &b, VVector &x, bool xIsZero);

        // Run a single iteration. Compute the residual and its norm and decide convergence.
        AMGX_STATUS solve_iteration( VVector &b, VVector &x, bool xIsZero );

        // Finalize the solver after running the iterations.
        void solve_finalize( VVector &b, VVector &x );
};

// ----------------------------
//  specialization for host
// ----------------------------

template <AMGX_VecPrecision t_vecPrec, AMGX_MatPrecision t_matPrec, AMGX_IndPrecision t_indPrec>
class MulticolorILUSolver< TemplateConfig<AMGX_host, t_vecPrec, t_matPrec, t_indPrec> > : public MulticolorILUSolver_Base< TemplateConfig<AMGX_host, t_vecPrec, t_matPrec, t_indPrec> >
{
    public:
        typedef TemplateConfig<AMGX_host, t_vecPrec, t_matPrec, t_indPrec> TConfig_h;
        typedef Matrix<TConfig_h> Matrix_h;
        typedef typename TConfig_h::MatPrec ValueTypeA;
        typedef typename TConfig_h::VecPrec ValueTypeB;
        typedef Vector<TConfig_h> VVector;
        typedef typename TConfig_h::IndPrec IndexType;
        MulticolorILUSolver(AMG_Config &cfg, const std::string &cfg_scope) : MulticolorILUSolver_Base<TConfig_h>(cfg, cfg_scope) {}
        ~MulticolorILUSolver() {};

    private:
        void computeLUSparsityPattern(void);
        void computeAtoLUmapping(void);
        void fillLUValuesWithAValues(void);
        void computeLUFactors(void);
        void smooth_4x4( const VVector &b, VVector &x, bool xIsZero);
        void smooth_bxb( const VVector &b, VVector &x, bool xIsZero);
};

// ----------------------------
//  specialization for device
// ----------------------------

template <AMGX_VecPrecision t_vecPrec, AMGX_MatPrecision t_matPrec, AMGX_IndPrecision t_indPrec>
class MulticolorILUSolver< TemplateConfig<AMGX_device, t_vecPrec, t_matPrec, t_indPrec> >: public MulticolorILUSolver_Base< TemplateConfig<AMGX_device, t_vecPrec, t_matPrec, t_indPrec> >
{
    public:
        typedef TemplateConfig<AMGX_host, t_vecPrec, t_matPrec, t_indPrec> TConfig_h;
        typedef TemplateConfig<AMGX_device, t_vecPrec, t_matPrec, t_indPrec> TConfig_d;
        typedef Matrix<TConfig_h> Matrix_h;
        typedef Matrix<TConfig_d> Matrix_d;

        typedef typename TConfig_d::MatPrec ValueTypeA;
        typedef typename TConfig_d::VecPrec ValueTypeB;
        typedef typename TConfig_d::IndPrec IndexType;
        typedef Vector<TConfig_d> VVector;
        typedef typename Matrix_d ::IVector IVector;

        MulticolorILUSolver(AMG_Config &cfg, const std::string &cfg_scope) : MulticolorILUSolver_Base<TConfig_d>(cfg, cfg_scope) {}
        ~MulticolorILUSolver() {};
    private:

        void computeLUSparsityPattern(void);
        void computeAtoLUmapping(void);
        void fillLUValuesWithAValues(void);
        void computeLUFactors(void);
        void smooth_4x4(const VVector &b, VVector &x, bool xIsZero);
        void smooth_bxb(const VVector &b, VVector &x, bool xIsZero);
};

template<class T_Config>
class MulticolorILUSolverFactory : public SolverFactory<T_Config>
{
    public:
        Solver<T_Config> *create( AMG_Config &cfg, const std::string &cfg_scope, ThreadManager *tmng ) { return new MulticolorILUSolver<T_Config>( cfg, cfg_scope); }
};

} // namespace multicolor_dilu
} // namespace amgx
