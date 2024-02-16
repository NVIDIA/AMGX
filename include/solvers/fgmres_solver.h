// SPDX-FileCopyrightText: 2011 - 2024 NVIDIA CORPORATION. All Rights Reserved.
//
// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#include<solvers/solver.h>
#include<cusp/array1d.h>
#include<cusp/array2d.h>

namespace amgx
{


//data structure that manages the krylov vectors and takes care of all the modulo calculation etc.
template <class TConfig>
class KrylovSubspaceBuffer
{
    public:
        typedef Vector<TConfig> VVector;

        KrylovSubspaceBuffer();
        ~KrylovSubspaceBuffer();

        VVector &V(int m);
        VVector &Z(int m);

        void set_max_dimension(int max_dimension);
        void setup(int N, int blockdim, int tag);
        int get_dimension() {return this->dimension;}
        bool set_iteration(int m);
        int get_smallest_m();

    private:
        std::vector<VVector *> m_V_vector;
        std::vector<VVector *> m_Z_vector;

        int dimension;
        int max_dimension;
        int iteration;
        int N, tag, blockdim;

        bool increase_dimension();
};



template<class T_Config>
class FGMRES_Solver : public Solver<T_Config>
{

    public:

        typedef Solver<T_Config> Base;
        typedef typename Base::VVector VVector;
        typedef typename Base::Vector_h Vector_h;
        typedef typename Base::ValueTypeA ValueTypeA;
        typedef typename Base::ValueTypeB ValueTypeB;

        FGMRES_Solver( AMG_Config &cfg, const std::string &cfg_scope );
        ~FGMRES_Solver();

        bool is_residual_needed() const { return false; }
        void printSolverParameters() const;
        void solver_setup(bool reuse_matrix_structure);

        bool isColoringNeeded( ) const { if (m_preconditioner != NULL) return m_preconditioner->isColoringNeeded(); else return false; }
        void getColoringScope( std::string &cfg_scope_for_coloring) const { if (m_preconditioner != NULL) m_preconditioner->getColoringScope(cfg_scope_for_coloring); }
        bool getReorderColsByColorDesired() const { if (m_preconditioner != NULL) return m_preconditioner->getReorderColsByColorDesired(); return false; }

        bool getInsertDiagonalDesired() const
        {
            if (m_preconditioner != NULL)
            {
                return m_preconditioner->getInsertDiagonalDesired();
            }

            return false;
        }

        void solve_init( VVector &b, VVector &x, bool xIsZero );
        // Run a single iteration. Compute the residual and its norm and decide convergence.
        bool solve_one_iteration( VVector &b, VVector &x );
        // Run a single iteration. Compute the residual and its norm and decide convergence.
        AMGX_STATUS solve_iteration( VVector &b, VVector &x, bool xIsZero );
        void solve_finalize( VVector &b, VVector &x );

    private:

        int m_R;  //Iterations between restarts, do we still need restart?
        int m_krylov_size;
        bool use_preconditioner;
        bool use_scalar_L2_norm;
        bool update_x_every_iteration; // x is solution
        bool update_r_every_iteration; // r is residual
        Solver<T_Config> *m_preconditioner;

        //DEVICE WORKSPACE
        KrylovSubspaceBuffer<T_Config> subspace;

        //HOST WORKSPACE
        //TODO: move those to device
        cusp::array2d<ValueTypeB, cusp::host_memory, cusp::column_major> m_H; //Hessenberg matrix
        cusp::array1d<ValueTypeB, cusp::host_memory> m_s; // rotated right-hand side vector, size=m+1
        cusp::array1d<ValueTypeB, cusp::host_memory> m_cs; // Givens rotation cosine
        cusp::array1d<ValueTypeB, cusp::host_memory> m_sn; // Givens rotation sine
        cusp::array1d<ValueTypeB, cusp::host_memory> gamma; // recursion for residual calculation

        ValueTypeA beta;

        VVector residual; //compute the whole residual recursively

        AMGX_STATUS checkConvergenceGMRES(bool check_V_0);

};

template<class T_Config>
class FGMRES_SolverFactory : public SolverFactory<T_Config>
{
    public:
        Solver<T_Config> *create( AMG_Config &cfg, const std::string &cfg_scope, ThreadManager *tmng ) { return new FGMRES_Solver<T_Config>( cfg, cfg_scope ); }
};

} // namespace amgx
