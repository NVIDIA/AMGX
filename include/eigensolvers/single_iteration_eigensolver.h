// SPDX-FileCopyrightText: 2013 - 2025 NVIDIA CORPORATION. All Rights Reserved.
//
// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#include <amg_solver.h>
#include <eigensolvers/eigensolver.h>

namespace amgx
{

template <class TConfig>
class SingleIteration_EigenSolver : public EigenSolver<TConfig>
{
    public:
        typedef EigenSolver<TConfig> Base;

        typedef typename Base::VVector VVector;
        typedef typename Base::MMatrix MMatrix;
        typedef typename Base::Vector_h Vector_h;
        typedef typename Base::Matrix_h Matrix_h;
        typedef typename Base::ValueTypeMat ValueTypeMat;
        typedef typename Base::ValueTypeVec ValueTypeVec;
        typedef typename Base::IndType IndType;

        SingleIteration_EigenSolver(AMG_Config &cfg,
                                    const std::string &cfg_scope);
        ~SingleIteration_EigenSolver();

        void solver_setup();
        void solver_pagerank_setup(VVector &a);
        void solve_init(VVector &x);
        bool solve_iteration(VVector &x);
        void solve_finalize();
    private:
        void free_allocated();
        void shift_matrix();
        void get_dangling_nodes();
        void update_dangling_nodes();
    private:
        int m_convergence_check_freq;
        AMG_Config m_cfg;
        Operator<TConfig> *m_operator;
        VVector m_v;
        VVector m_x;
        VVector m_a;
        VVector m_b;
        std::vector<VVector *> m_allocated_vectors;
};

template<class TConfig>
class SingleIteration_EigenSolverFactory : public EigenSolverFactory<TConfig>
{
    public:
        EigenSolver<TConfig> *create(AMG_Config &cfg, const std::string &cfg_scope, ThreadManager *tmng)
        {
            return new SingleIteration_EigenSolver<TConfig>(cfg, cfg_scope);
        }
};

}
