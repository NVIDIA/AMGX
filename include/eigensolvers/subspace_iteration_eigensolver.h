// SPDX-FileCopyrightText: 2013 - 2024 NVIDIA CORPORATION. All Rights Reserved.
//
// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#include <eigensolvers/eigensolver.h>

namespace amgx
{

template <class TConfig>
class SubspaceIteration_EigenSolver : public EigenSolver<TConfig>
{
    public:
        typedef EigenSolver<TConfig> Base;

        typedef typename Base::TConfig_h TConfig_h;
        typedef typename Base::VVector VVector;
        typedef typename Base::MMatrix MMatrix;
        typedef typename Base::Vector_h Vector_h;
        typedef typename Base::Matrix_h Matrix_h;
        typedef typename Base::ValueTypeMat ValueTypeMat;
        typedef typename Base::ValueTypeVec ValueTypeVec;

        SubspaceIteration_EigenSolver(AMG_Config &cfg, const std::string &cfg_scope);
        ~SubspaceIteration_EigenSolver();

        void solver_setup();
        void solver_pagerank_setup(VVector &a);
        void solve_init(VVector &x);
        bool solve_iteration(VVector &x);
        void solve_finalize();
    private:
        void orthonormalize(VVector &V);
    private:
        VVector m_X;
        VVector m_V;
        VVector m_H;
        VVector m_R;
        int m_subspace_size;
        int m_wanted_count;
        ValueTypeVec m_initial_residual;
};

template<class TConfig>
class SubspaceIteration_EigenSolverFactory : public EigenSolverFactory<TConfig>
{
    public:
        EigenSolver<TConfig> *create(AMG_Config &cfg, const std::string &cfg_scope, ThreadManager *tmng)
        {
            return new SubspaceIteration_EigenSolver<TConfig>(cfg, cfg_scope);
        }
};

}
