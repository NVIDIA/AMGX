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
