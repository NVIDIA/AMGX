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

#include <string>
#include <solvers/solver.h>
#include <matrix_coloring/matrix_coloring.h>
#include <basic_types.h>
#include <matrix.h>

#include <amgx_types/util.h>

namespace amgx
{
namespace multicolor_dilu_solver
{

template <class T_Config> class MulticolorDILUSolver;

template<class T_Config>
class MulticolorDILUSolver_Base : public Solver<T_Config>
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
        typedef Vector<T_Config> VVector;
        typedef Vector<TemplateConfig<AMGX_host, vecPrec, matPrec, indPrec> > Vector_h;
        typedef typename Matrix<TConfig>::MVector MVector;

        typedef typename types::PODTypes<ValueTypeB>::type WeightType;

    private:

        void computeEinv(Matrix<T_Config> &A);

    protected:

        virtual void computeEinv_NxN(const Matrix<T_Config> &A, const int block_size) = 0;
        virtual void smooth_NxN(const Matrix<T_Config> &A, VVector &b, VVector &x, ViewType separation_flag) = 0;

        Matrix<T_Config> *m_explicit_A;

        WeightType weight;
        ColoringType m_boundary_coloring;
        bool always_obey_coloring;

        bool m_reorder_cols_by_color_desired;
        bool m_insert_diagonal_desired;

        VVector m_delta, m_Delta;

    public:
        MVector Einv;

        // Constructor.
        MulticolorDILUSolver_Base( AMG_Config &cfg, const std::string &cfg_scope, ThreadManager *tmng );

        // Destructor
        virtual ~MulticolorDILUSolver_Base();

        bool is_residual_needed() const { return false; }

        // Print the solver parameters
        void printSolverParameters() const;

        bool isColoringNeeded() const { return true; }

        void getColoringScope( std::string &cfg_scope_for_coloring) const  { cfg_scope_for_coloring = this->m_cfg_scope; }

        bool getReorderColsByColorDesired() const { return m_reorder_cols_by_color_desired;}

        bool getInsertDiagonalDesired() const { return m_insert_diagonal_desired;}

        // Setup the solver
        void solver_setup(bool reuse_matrix_structure);
        // Initialize the solver before running the iterations.
        void solve_init( VVector &b, VVector &x, bool xIsZero );
        // Run a single iteration. Compute the residual and its norm and decide convergence.
        bool solve_iteration( VVector &b, VVector &x, bool xIsZero );
        // Finalize the solver after running the iterations.
        void solve_finalize( VVector &b, VVector &x );
};

// ----------------------------
//  specialization for host
// ----------------------------

template< AMGX_VecPrecision V, AMGX_MatPrecision M, AMGX_IndPrecision I >
class MulticolorDILUSolver<TemplateConfig<AMGX_host, V, M, I> > : public MulticolorDILUSolver_Base< TemplateConfig<AMGX_host, V, M, I> >
{
    public:
        typedef TemplateConfig<AMGX_host, V, M, I> TConfig_h;
        typedef Matrix<TConfig_h> Matrix_h;
        typedef typename TConfig_h::MatPrec ValueTypeA;
        typedef typename TConfig_h::VecPrec ValueTypeB;
        typedef Vector<TConfig_h> VVector;
        typedef typename TConfig_h::IndPrec IndexType;
        typedef typename Matrix<TConfig_h>::MVector MVector;

        typedef typename types::PODTypes<ValueTypeB>::type WeightType;

        MulticolorDILUSolver(AMG_Config &cfg, const std::string &cfg_scope, ThreadManager *tmng) :
            MulticolorDILUSolver_Base<TConfig_h>(cfg, cfg_scope, tmng)
        {}

        ~MulticolorDILUSolver() {}

    private:
        void smooth_NxN(const Matrix_h &A, VVector &b, VVector &x, ViewType separation_flag);

        void computeEinv_NxN(const Matrix_h &A, const int block_size);
};

// ----------------------------
//  specialization for device
// ----------------------------

template< AMGX_VecPrecision V, AMGX_MatPrecision M, AMGX_IndPrecision I >
class MulticolorDILUSolver<TemplateConfig<AMGX_device, V, M, I> >: public MulticolorDILUSolver_Base< TemplateConfig<AMGX_device, V, M, I> >
{
    public:
        typedef TemplateConfig<AMGX_device, V, M, I> TConfig_d;
        typedef Matrix<TConfig_d> Matrix_d;
        typedef typename TConfig_d::MatPrec ValueTypeA;
        typedef typename TConfig_d::VecPrec ValueTypeB;
        typedef typename TConfig_d::IndPrec IndexType;
        typedef Vector<TConfig_d> VVector;
        typedef typename Matrix_d ::IVector IVector;
        typedef typename Matrix<TConfig_d>::MVector MVector;

        typedef typename types::PODTypes<ValueTypeB>::type WeightType;

        MulticolorDILUSolver(AMG_Config &cfg, const std::string &cfg_scope, ThreadManager *tmng);
        ~MulticolorDILUSolver() {}
    private:
        void smooth_NxN(const Matrix_d &A, VVector &b, VVector &x, ViewType separation_flag);

        void computeEinv_NxN(const Matrix_d &A, const int bsize);

        bool m_is_kepler;
};

template<class T_Config>
class MulticolorDILUSolverFactory : public SolverFactory<T_Config>
{
    public:
        Solver<T_Config> *create( AMG_Config &cfg, const std::string &cfg_scope, ThreadManager *tmng ) { return new MulticolorDILUSolver<T_Config>( cfg, cfg_scope, tmng ); }
};

} // namespace multicolor_dilu
} // namespace amgx

