// SPDX-FileCopyrightText: 2013 - 2024 NVIDIA CORPORATION. All Rights Reserved.
//
// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#include <eigensolvers/eigensolver.h>
#include <vector>
#include <cusp/array2d.h>

namespace amgx
{

template <class TConfig>
class Arnoldi_EigenSolver : public EigenSolver<TConfig>
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

        Arnoldi_EigenSolver(AMG_Config &cfg, const std::string &cfg_scope);
        ~Arnoldi_EigenSolver();

        void solver_setup();
        void solver_pagerank_setup(VVector &a);
        void solve_init(VVector &x);
        bool solve_iteration(VVector &x);
        void solve_finalize();

    private:
        void free_allocated();
    private:
        int m_krylov_size;
        std::vector<VVector *> m_V_vectors;
        Vector_h m_H;
        Vector_h m_H_tmp;
        Vector_h m_ritz_eigenvalues;
        Vector_h m_ritz_eigenvectors;
        ValueTypeVec m_beta;
};

template<class TConfig>
class Arnoldi_EigenSolverFactory : public EigenSolverFactory<TConfig>
{
    public:
        EigenSolver<TConfig> *create(AMG_Config &cfg, const std::string &cfg_scope, ThreadManager *tmng)
        {
            return new Arnoldi_EigenSolver<TConfig>(cfg, cfg_scope);
        }
};

}
