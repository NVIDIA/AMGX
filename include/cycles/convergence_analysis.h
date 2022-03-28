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

#include <basic_types.h>
#include <amg_config.h>

#include <amg_level.h>


namespace amgx
{





template<class T_Config>
class ConvergenceAnalysis
{

    public:
        typedef T_Config TConfig;
        typedef Matrix<T_Config> Matrix;
        typedef typename T_Config::VecPrec ValueTypeV;
        typedef typename T_Config::MatPrec ValueTypeM;
        typedef typename T_Config::IndPrec IndexType;
        typedef typename T_Config::MemSpace MemorySpace;
        typedef typename Matrix::IVector IVector;
        typedef typename Matrix::VVector VVector;
        typedef typename Matrix::MVector MVector;

        ConvergenceAnalysis( AMG_Config &cfg, std::string &cfg_scope, AMG_Level<T_Config> &curLevel );

        void startPresmoothing(Matrix &A, VVector &x, VVector &b );
        void stopPresmoothing(Matrix &A, VVector &x, VVector &b );
        void startCoarseGridCorrection(Matrix &A, VVector &x, VVector &b );
        void stopCoarseGridCorrection(Matrix &A, VVector &x, VVector &b );
        void startPostsmoothing(Matrix &A, VVector &x, VVector &b );
        void stopPostsmoothing(Matrix &A, VVector &x, VVector &b );
        void printConvergenceAnalysis(Matrix &A, VVector &x, VVector &b );



    private:

        void computeResidual( Matrix &A, VVector &x, VVector &b, ValueTypeV &res );

        //contains the number of levels to analyse
        int convergence_analysis;

        //current level
        AMG_Level<T_Config> *level;

        //error vectors
        VVector delta_e_smoother;
        VVector delta_e_coarse;

        //residuals
        ValueTypeV res_before_pre, res_after_pre,
                   res_before_coarse, res_after_coarse,
                   res_before_post, res_after_post;

};















}
