// SPDX-FileCopyrightText: 2011 - 2024 NVIDIA CORPORATION. All Rights Reserved.
//
// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#include <solvers/solver.h>
#include <eigensolvers/eigensolver.h>

namespace amgx
{

template<class T_Config>
class Chebyshev_Solver : public Solver<T_Config>
{
    public:
        typedef Solver<T_Config> Base;

        typedef typename Base::VVector VVector;
        typedef typename Base::Vector_h Vector_h;
        typedef typename Base::ValueTypeB ValueTypeB;
        typedef typename T_Config::MatPrec ValueTypeA;
        typedef typename T_Config::IndPrec IndexType;

    private:
        // Temporary vectors needed for the computation.
        VVector m_p, m_Ap, m_z, m_xp, m_rp;
        // The dot product between z and the residual.
        ValueTypeB m_r_z, m_lmax, m_lmin, m_gamma, m_beta;
        int m_buffer_N, first_iter;

        int m_lambda_mode, m_cheby_order;

        double m_user_max_lambda, m_user_min_lambda;

        bool no_preconditioner;
        Solver<T_Config> *m_preconditioner;
        EigenSolver<T_Config> *m_eigsolver;

        virtual void compute_eigenmax_estimate( const Matrix<T_Config> &A, ValueTypeB &lambda );

    public:
        // Constructor.
        Chebyshev_Solver( AMG_Config &cfg, const std::string &cfg_scope);

        ~Chebyshev_Solver();

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
};

template<class T_Config>
class Chebyshev_SolverFactory : public SolverFactory<T_Config>
{
    public:
        Solver<T_Config> *create( AMG_Config &cfg, const std::string &cfg_scope, ThreadManager *tmng ) { return new Chebyshev_Solver<T_Config>( cfg, cfg_scope); }
};

} // namespace amgx
