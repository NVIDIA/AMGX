// SPDX-FileCopyrightText: 2013 - 2025 NVIDIA CORPORATION. All Rights Reserved.
//
// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#include <matrix.h>
#include <eigensolvers/eigensolver.h>

namespace amgx
{

template <class TConfig>
class EigenVectorSolver
{
    public:
        typedef Matrix<TConfig> MMatrix;
        typedef Vector<TConfig> VVector;

        typedef typename TConfig::template setMemSpace<AMGX_host  >::Type TConfig_h;
        typedef typename TConfig::template setMemSpace<AMGX_device>::Type TConfig_d;

        typedef Matrix<TConfig_h> Matrix_h;
        typedef Matrix<TConfig_d> Matrix_d;

        typedef Vector<TConfig_h> Vector_h;
        typedef Vector<TConfig_d> Vector_d;

        typedef typename TConfig::MatPrec ValueTypeMat;
        typedef typename TConfig::VecPrec ValueTypeVec;
        typedef typename TConfig::IndPrec IndType;

        EigenVectorSolver(AMG_Config &cfg, const std::string &cfg_scope);
        ~EigenVectorSolver();

        void setup(Operator<TConfig> &A);
        AMGX_STATUS solve(ValueTypeVec eigenvalue, VVector &eigenvector);
    private:
        AMG_Config m_cfg;
        Operator<TConfig> *m_A;
        EigenSolver<TConfig> *m_solver;
};

template <class TConfig>
class EigenVectorSolverFactory
{
    public:
        static EigenVectorSolver<TConfig> *create(std::string &name);
    private:
        static EigenVectorSolver<TConfig> *create_inverse_iteration();
};

}
