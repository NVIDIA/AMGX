/* Copyright (c) 2013-2017, NVIDIA CORPORATION. All rights reserved.
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

#include <basic_types.h>
#include <solvers/solver.h>

namespace amgx
{

template<class T_Config>
class AlgebraicMultigrid_Solver: public Solver<T_Config>
{
        typedef Solver<T_Config> Base;

        typedef T_Config TConfig;
        static const AMGX_VecPrecision vecPrec = TConfig::vecPrec;
        static const AMGX_MatPrecision matPrec = TConfig::matPrec;
        static const AMGX_IndPrecision indPrec = TConfig::indPrec;

        typedef typename MemorySpaceMap<AMGX_host>::Type host_memory;
        typedef typename MemorySpaceMap<AMGX_device>::Type device_memory;

        typedef TemplateConfig<AMGX_host, vecPrec, matPrec, indPrec> TConfig_h;
        typedef Matrix<TConfig_h> Matrix_h;
        typedef TemplateConfig<AMGX_device, vecPrec, matPrec, indPrec> TConfig_d;
        typedef Matrix<TConfig_d> Matrix_d;

        typedef typename T_Config::MatPrec ValueTypeA;
        typedef typename T_Config::VecPrec ValueTypeB;
        typedef Vector<T_Config> VVector;
        typedef Vector<TemplateConfig<AMGX_host, vecPrec, matPrec, indPrec> > Vector_h;

        typedef typename TConfig::MemSpace MemorySpace;
        MemorySpace memorySpaceTag;

    private:
        AMG<vecPrec, matPrec, indPrec> m_amg;

    protected:
        // Override parent attribute.
        Matrix<T_Config> *m_A;
    public:
        // Constructor.
        AlgebraicMultigrid_Solver( AMG_Config &cfg, const std::string &cfg_scope, ThreadManager *tmng );

        void SetThreadManager(ThreadManager *tmng) { m_amg.tmng = tmng; }

        // Destructor
        ~AlgebraicMultigrid_Solver();

        // That solver doesn't need a residual to operate.
        bool is_residual_needed() const { return false; }

        // Print the solver parameters
        void printSolverParameters() const;

        // Setup the solver
        void solver_setup(bool reuse_matrix_structure);

        // Coloring is postponed. If the smoother needs a coloring, it will do it, itself. It's
        // better for asynchronous execution today.
        bool isColoringNeeded() const { return m_amg.getFinestLevel(memorySpaceTag)->getSmoother()->isColoringNeeded(); } //amg setup will color matrix for particular level if needed.

        void getColoringScope(std::string &cfg_scope_for_coloring) const { return m_amg.getFinestLevel(memorySpaceTag)->getSmoother()->getColoringScope(cfg_scope_for_coloring);} //amg setup will color matrix for particular level if needed.

        bool getReorderColsByColorDesired() const { return m_amg.getFinestLevel(memorySpaceTag)->getSmoother()->getReorderColsByColorDesired();}

        bool getInsertDiagonalDesired() const { return m_amg.getFinestLevel(memorySpaceTag)->getSmoother()->getInsertDiagonalDesired();}

        // Initialize the solver before running the iterations.
        void solve_init( VVector &b, VVector &x, bool xIsZero );
        // Run a single iteration. Compute the residual and its norm and decide convergence.
        bool solve_iteration( VVector &b, VVector &x, bool xIsZero );
        // Finalize the solver after running the iterations.
        void solve_finalize( VVector &b, VVector &x );

        // Print data.
        void print_grid_stats();
        void print_grid_stats2();
        void print_vis_data();
};

template<class T_Config>
class AlgebraicMultigrid_SolverFactory : public SolverFactory<T_Config>
{
    public:
        Solver<T_Config> *create( AMG_Config &cfg, const std::string &cfg_scope, ThreadManager *tmng ) { return new AlgebraicMultigrid_Solver<T_Config>( cfg, cfg_scope, tmng ); }
};

} // namespace amgx
