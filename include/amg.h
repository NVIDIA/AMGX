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

#include <basic_types.h>
namespace amgx
{
template< AMGX_VecPrecision t_vecPrec, AMGX_MatPrecision t_matPrec, AMGX_IndPrecision t_indPrec >
class AMG;
}

template<AMGX_MemorySpace memspace>
struct opposite_memspace;

template<>
struct opposite_memspace<AMGX_host>
{
    static const AMGX_MemorySpace memspace = AMGX_device;
};

template<>
struct opposite_memspace<AMGX_device>
{
    static const AMGX_MemorySpace memspace = AMGX_host;
};


#include <amg_solver.h>
#include <amg_config.h>
#include <solvers/solver.h>
#include <convergence/convergence.h>
#include <amg_level.h>
#include <misc.h>
#include <sstream>

namespace amgx
{

template< AMGX_VecPrecision t_vecPrec, AMGX_MatPrecision t_matPrec, AMGX_IndPrecision t_indPrec >
class AMG_Setup;

template< typename TConfig>
class AMG_Solve;

template < AMGX_VecPrecision t_vecPrec, AMGX_MatPrecision t_matPrec, AMGX_IndPrecision t_indPrec>
class AMG
{
        typedef TemplateConfig<AMGX_host, t_vecPrec, t_matPrec, t_indPrec> TConfig_h;
        typedef TemplateConfig<AMGX_device, t_vecPrec, t_matPrec, t_indPrec> TConfig_d;

        typedef typename MemorySpaceMap<AMGX_host>::Type host_memory;
        typedef typename MemorySpaceMap<AMGX_device>::Type device_memory;

        typedef Matrix<TConfig_h> Matrix_h;
        typedef Matrix<TConfig_d> Matrix_d;

        typedef Vector<TConfig_h> Vector_h;
        typedef Vector<TConfig_d> Vector_d;

        typedef typename IndPrecisionMap<t_indPrec>::Type IndexType;
        typedef typename MatPrecisionMap<t_matPrec>::Type ValueTypeA;
        typedef typename VecPrecisionMap<t_vecPrec>::Type ValueTypeB;

        friend class AMG_Setup<t_vecPrec, t_matPrec, t_indPrec>;
        friend class AMG_Solve<TConfig_h>;
        friend class AMG_Solve<TConfig_d>;

        friend class AMG_Level<TConfig_h>;
        friend class AMG_Level<TConfig_d>;

    public:
        AMG_Config *m_cfg;
        std::string m_cfg_scope;

        ThreadManager *tmng;

        AMG(AMG_Config &cfg, const std::string &cfg_scope);
        ~AMG();

        void printSettings() const ;

        // special call to allocate fine level
        void allocate_fine_level();

        // Setup the hierarchy starting from a level.
        void setup( AMG_Level<TConfig_h> *level );
        void setup( AMG_Level<TConfig_d> *level );

        // Setup the hierarchy to solve on host.
        void setup( Matrix_h &A );
        void setup( Matrix_d &A );

        void solve_iteration(  Vector_h &b, Vector_h &x);
        void solve_iteration(  Vector_d &b, Vector_d &x);

        void solve_init(  Vector_h &b, Vector_h &x, bool xIsZero);
        void solve_init(  Vector_d &b, Vector_d &x, bool xIsZero);

        void getGridStatisticsString(std::stringstream &ss);
        void getGridStatisticsString2(std::stringstream &ss);
        void printGridStatistics();
        void printGridStatistics2();
        // void printStatistics();

        // profiling & debug output
        // void printProfile();
        void printCoarsePoints();
        void printConnections();

        inline int getCycleIters() const {return cycle_iters;}
        inline int getNumPresweeps() const {return m_cfg->template getParameter<int>("presweeps", m_cfg_scope);}
        inline int getNumCoarsestsweeps() const {return m_cfg->template getParameter<int>("coarsest_sweeps", m_cfg_scope);}
        inline int getNumFinestsweeps() const {return m_cfg->template getParameter<int>("finest_sweeps", m_cfg_scope);}
        inline int getNumPostsweeps() const {return m_cfg->template getParameter<int>("postsweeps", m_cfg_scope);}
        inline bool getIntensiveSmoothing() const {return m_cfg->template getParameter<int>("intensive_smoothing", m_cfg_scope) != 0;}
        inline int getIters() const {return iterations;}

        inline NormType getNormType() const {return norm; }

        int ref_count;

        inline AMG_Level<TConfig_h> *getFinestLevel( host_memory ) const { return fine_h; }
        inline void setFinestLevel( AMG_Level<TConfig_h> *level ) { fine_h = level; }
        inline void resetFinestLevel( host_memory ) { fine_h = 0L; }

        inline AMG_Level<TConfig_d> *getFinestLevel( device_memory ) const { return fine_d; }
        inline void setFinestLevel( AMG_Level<TConfig_d> *level ) { fine_d = level; }
        inline void resetFinestLevel( device_memory ) { fine_d = 0L; }

        inline Solver<TConfig_h> *getCoarseSolver( host_memory ) { return coarse_solver_h; }
        inline void setCoarseSolver( Solver<TConfig_h> *s, host_memory ) { coarse_solver_h = s; }
        inline Solver<TConfig_d> *getCoarseSolver( device_memory ) { return coarse_solver_d; }
        inline void setCoarseSolver( Solver<TConfig_d> *s, device_memory ) { coarse_solver_d = s; }

        void *getD2Workspace() { return d2_workspace; }
        void  setD2Workspace(void *workspace) { d2_workspace = workspace; }
        void *getCsrWorkspace() { return csr_workspace; }
        void  setCsrWorkspace(void *workspace) { csr_workspace = workspace; }
    private:

        AMG_Level<TConfig_d> *fine_d;
        Solver<TConfig_d> *coarse_solver_d;

        AMG_Level<TConfig_h> *fine_h;
        Solver<TConfig_h> *coarse_solver_h;

        // These will be filled in based on whether the Convergence needs block data or scalar data
        Vector_h nrm;
        Vector_h nrm_ini;

        std::vector<Vector_h> res_history;

        int iterations;
        int max_iters;
        int max_levels;
        double coarsen_threshold;

        int m_sum_stopping_criteria;
        int m_structure_reuse_levels;
        int m_amg_host_levels_rows;

        int min_fine_rows;
        int min_coarse_rows;
        double min_coarse_rows_prct;

        int max_coarse_iters;
        int cycle_iters;
        int num_levels;

        NormType norm;

        // The specified number of rows to tune dense LU. If we don't use DENSE LU or we want to
        // use min_coarse_rows as the input parameter, that value will be 0. If it is > 0 we use
        // that value to build the heuristic.
        int m_dense_lu_num_rows, m_dense_lu_max_rows;

        // Pimpl to csr_multiply workspace.
        void *csr_workspace, *d2_workspace;
};

} // namespace amgx
