// SPDX-FileCopyrightText: 2013 - 2025 NVIDIA CORPORATION. All Rights Reserved.
//
// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#include <eigensolvers/eigensolver.h>
#include <vector>
#include <operators/deflated_multiply_operator.h>

namespace amgx
{

template <class TConfig>
class JacobiDavidson_EigenSolver : public EigenSolver<TConfig>
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

        JacobiDavidson_EigenSolver(AMG_Config &cfg, const std::string &cfg_scope);
        ~JacobiDavidson_EigenSolver();

        void solver_setup();
        void solver_pagerank_setup(VVector &a);
        void solve_init(VVector &x);
        bool solve_iteration(VVector &x);
        void solve_finalize();
    private:
        void free_allocated();
        void orthonormalize(VVector &V);
        void set_subspace_size(int size);
    private:
        AMG_Config m_cfg;
        Operator<TConfig> *m_operator;
        DeflatedMultiplyOperator<TConfig> *m_deflated_operator;
        VVector m_x;
        VVector m_Ax;
        VVector m_d;
        VVector m_work;
        VVector m_V;
        VVector m_AV;
        VVector m_H;
        VVector m_s;
        Vector<TConfig_h> m_subspace_eigenvalues;
        ValueTypeVec m_lambda;
};

template <class TConfig>
class JacobiDavidson_EigenSolverFactory : public EigenSolverFactory<TConfig>
{
    public:
        EigenSolver<TConfig> *create(AMG_Config &cfg, const std::string &cfg_scope, ThreadManager *tmng)
        {
            return new JacobiDavidson_EigenSolver<TConfig>(cfg, cfg_scope);
        }
};

}
