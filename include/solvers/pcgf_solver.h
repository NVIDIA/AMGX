// SPDX-FileCopyrightText: 2011 - 2024 NVIDIA CORPORATION. All Rights Reserved.
//
// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#include<solvers/solver.h>

namespace amgx
{

template<class T_Config>
class PCGF_Solver : public Solver<T_Config>
{
    public:
        typedef Solver<T_Config> Base;

        typedef typename Base::VVector VVector;
        typedef typename Base::Vector_h Vector_h;
        typedef typename Base::ValueTypeB ValueTypeB;

    private:
        // Temporary vectors needed for the computation.
        VVector m_p, m_Ap, m_z, m_d;
        // The dot product between z and the residual.
        ValueTypeB m_r_z;
        int m_buffer_N;

        bool no_preconditioner;
        Solver<T_Config> *m_preconditioner;

    public:
        // Constructor.
        PCGF_Solver( AMG_Config &cfg, const std::string &cfg_scope );

        // Dtor.
        ~PCGF_Solver();

        // Print the solver parameters
        void printSolverParameters() const;
        // Setup the solver
        void solver_setup(bool reuse_matrix_structure);

        bool isColoringNeeded() const { if (m_preconditioner != NULL) return m_preconditioner->isColoringNeeded(); return false; }

        void getColoringScope( std::string &cfg_scope_for_coloring) const  { if (m_preconditioner != NULL) m_preconditioner->getColoringScope(cfg_scope_for_coloring); }

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
class PCGF_SolverFactory : public SolverFactory<T_Config>
{
    public:
        Solver<T_Config> *create( AMG_Config &cfg, const std::string &cfg_scope, ThreadManager *tmng ) { return new PCGF_Solver<T_Config>( cfg, cfg_scope ); }
};

} // namespace amgx
