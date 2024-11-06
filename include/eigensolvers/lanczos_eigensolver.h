// SPDX-FileCopyrightText: 2013 - 2024 NVIDIA CORPORATION. All Rights Reserved.
//
// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#include <eigensolvers/eigensolver.h>

namespace amgx
{

template <class TConfig>
class Lanczos_EigenSolver : public EigenSolver<TConfig>
{
    public:
        typedef EigenSolver<TConfig> Base;

        typedef typename Base::TConfig_h TConfig_h;
        typedef typename Base::VVector VVector;
        typedef typename Base::MMatrix MMatrix;
        typedef typename Base::Vector_h Vector_h;
        typedef typename Base::Vector_d Vector_d;
        typedef typename Base::Matrix_h Matrix_h;
        typedef typename Base::ValueTypeMat ValueTypeMat;
        typedef typename Base::ValueTypeVec ValueTypeVec;

        Lanczos_EigenSolver(AMG_Config &cfg, const std::string &cfg_scope);
        ~Lanczos_EigenSolver();

        void solver_setup();
        void solver_pagerank_setup(VVector &a);
        void solve_init(VVector &x);
        bool solve_iteration(VVector &x);
        void solve_finalize();

    private:
        void free_allocated();
    private:
        VVector m_w;
        VVector m_v;
        VVector m_v_prev;
        Vector_h m_ritz_eigenvectors;
        ValueTypeVec m_beta;
        Vector_h m_diagonal;
        Vector_h m_subdiagonal;
        Vector_h m_diagonal_tmp;
        Vector_h m_subdiagonal_tmp;
        Vector_d m_dwork;
};

template <class TConfig>
class Lanczos_EigenSolverFactory : public EigenSolverFactory<TConfig>
{
    public:
        EigenSolver<TConfig> *create(AMG_Config &cfg, const std::string &cfg_scope, ThreadManager *tmng)
        {
            return new Lanczos_EigenSolver<TConfig>(cfg, cfg_scope);
        }
};

}
