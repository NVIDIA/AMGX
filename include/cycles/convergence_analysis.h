// SPDX-FileCopyrightText: 2013 - 2024 NVIDIA CORPORATION. All Rights Reserved.
//
// SPDX-License-Identifier: BSD-3-Clause

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
