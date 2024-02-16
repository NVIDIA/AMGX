// SPDX-FileCopyrightText: 2011 - 2024 NVIDIA CORPORATION. All Rights Reserved.
//
// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#include<solvers/solver.h>
#include <blas.h>
#include <multiply.h>
#include <transpose.h>

namespace amgx
{

template<class T_Config>
class KPZPolynomialSolver: public Solver<T_Config>
{
    public:
        typedef Solver<T_Config> Base;
        typedef T_Config TConfig;
        static const AMGX_VecPrecision vecPrec = TConfig::vecPrec;
        static const AMGX_MatPrecision matPrec = TConfig::matPrec;
        static const AMGX_IndPrecision indPrec = TConfig::indPrec;
        typedef typename T_Config::MatPrec ValueTypeA;
        typedef typename T_Config::VecPrec ValueTypeB;
        typedef Vector<T_Config> VVector;
        typedef typename T_Config::IndPrec IndexType;
        typedef Vector<TemplateConfig<AMGX_host, vecPrec, matPrec, indPrec> > Vector_h;

    private:
        // Override parent attribute.
        Matrix<T_Config> *m_explicit_A;

    private:

        ValueTypeB l_inf;
        IndexType mu;
        IndexType poly_order;

        void smooth_1x1(Matrix<T_Config> &A, const VVector &b, VVector &x);

    public:
        // Constructor.
        KPZPolynomialSolver( AMG_Config &cfg, const std::string &cfg_scope);

        // Destructor
        ~KPZPolynomialSolver();

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
        AMGX_STATUS solve_iteration( VVector &b, VVector &x, bool xIsZero );
        // Finalize the solver after running the iterations.
        void solve_finalize( VVector &b, VVector &x );
};

template<class T_Config>
class KPZPolynomialSolverFactory : public SolverFactory<T_Config>
{
    public:
        Solver<T_Config> *create( AMG_Config &cfg, const std::string &cfg_scope, ThreadManager *tmng ) { return new KPZPolynomialSolver<T_Config>( cfg, cfg_scope); }
};

} // namespace amgx
