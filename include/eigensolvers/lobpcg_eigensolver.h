// SPDX-FileCopyrightText: 2013 - 2025 NVIDIA CORPORATION. All Rights Reserved.
//
// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#include <amg_solver.h>
#include <eigensolvers/eigensolver.h>

namespace amgx
{

template <class TConfig>
class LOBPCG_EigenSolver : public EigenSolver<TConfig>
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

        LOBPCG_EigenSolver(AMG_Config &cfg,
                           const std::string &cfg_scope);
        ~LOBPCG_EigenSolver();

        void solver_setup();
        void solve_init(VVector &x);
        void solver_pagerank_setup(VVector &a);
        bool solve_iteration(VVector &x);
        void solve_finalize();

    private:
        void free_allocated();
    private:
        int m_subspace_size;
        Vector_h m_work;
        VVector X;
        VVector AX;
        VVector W;
        VVector W_cpy;
        VVector AW;
        VVector P;
        VVector AP;
        ValueTypeVec m_lambda;
        Solver<TConfig> *m_solver;
};

template<class TConfig>
class LOBPCG_EigenSolverFactory : public EigenSolverFactory<TConfig>
{
    public:
        EigenSolver<TConfig> *create(AMG_Config &cfg, const std::string &cfg_scope, ThreadManager *tmng)
        {
            return new LOBPCG_EigenSolver<TConfig>(cfg, cfg_scope);
        }
};

}
