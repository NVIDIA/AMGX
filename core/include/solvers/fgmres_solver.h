/* Copyright (c) 2011-2017, NVIDIA CORPORATION. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *  * Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *  * Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *  * Neither the name of NVIDIA CORPORATION nor the names of its
 *    contributors may be used to endorse or promote products derived
 *    from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
 * EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 * PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
 * CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 * EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 * PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
 * OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

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
        bool solve_iteration( VVector &b, VVector &x, bool xIsZero );
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

        bool checkConvergenceGMRES(bool check_V_0);

};

template<class T_Config>
class FGMRES_SolverFactory : public SolverFactory<T_Config>
{
    public:
        Solver<T_Config> *create( AMG_Config &cfg, const std::string &cfg_scope, ThreadManager *tmng ) { return new FGMRES_Solver<T_Config>( cfg, cfg_scope ); }
};

} // namespace amgx
