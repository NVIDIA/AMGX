// SPDX-FileCopyrightText: 2013 - 2025 NVIDIA CORPORATION. All Rights Reserved.
//
// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#include<solvers/solver.h>

namespace amgx
{

template<class T_Config>
class BiCGStab_Solver : public Solver<T_Config>
{
    public:
        typedef Solver<T_Config> Base;

        typedef typename Base::VVector VVector;
        typedef typename Base::Vector_h Vector_h;
        typedef typename Base::ValueTypeB ValueTypeB;

    private:
        // Scalar temporary variables.
        ValueTypeB m_rho;
        // Vector temporary variables.
        VVector m_p, m_Mp, m_s, m_Ms, m_t, m_v, m_r_tilde;
        // Norm of s.
        Vector_h m_s_norm;

    public:
        // Constructor.
        BiCGStab_Solver( AMG_Config &cfg, const std::string &cfg_scope );
        // Destructor.
        ~BiCGStab_Solver();

        // Setup the solver
        void solver_setup(bool reuse_matrix_structure);

        bool isColoringNeeded( ) const { return false; }

        bool getReorderColsByColorDesired() const { return false; }

        bool getInsertDiagonalDesired() const { return false; }

        // Initialize the solver before running the iterations.
        void solve_init( VVector &b, VVector &x, bool xIsZero );
        // Run a single iteration. Compute the residual and its norm and decide convergence
        AMGX_STATUS solve_iteration( VVector &b, VVector &x, bool xIsZero );
        // Finalize the solver after running the iterations.
        void solve_finalize( VVector &b, VVector &x );
};

template<class T_Config>
class BiCGStab_SolverFactory : public SolverFactory<T_Config>
{
    public:
        Solver<T_Config> *create( AMG_Config &cfg, const std::string &cfg_scope, ThreadManager *tmng ) { return new BiCGStab_Solver<T_Config>( cfg, cfg_scope); }
};

} // namespace amgx
