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

#include <blas.h>
#include <cycles/convergence_analysis.h>

namespace amgx
{

//Constructor
template<class T_Config>
ConvergenceAnalysis<T_Config>::ConvergenceAnalysis( AMG_Config &cfg, std::string &cfg_scope, AMG_Level<T_Config> &curLevel )
{
    this->convergence_analysis = cfg.AMG_Config::getParameter<int>( "convergence_analysis", cfg_scope );
    this->level = &curLevel;
}

////////////////////////////////////////
//            Methods                 //
///////////////////////////////////////

template<class T_Config>
void ConvergenceAnalysis<T_Config>::startPresmoothing(Matrix &A, VVector &x, VVector &b )
{
    if ( this->level->getLevelIndex() >= this->convergence_analysis )
    {
        return;
    }

    int offset, size;
    A.getOffsetAndSizeForView(OWNED, &offset, &size);

    //set rhs to zero so x=e
    if ( this->level->isFinest() && this->level->isInitCycle() )
    {
        fill( b, 0.0, offset, size );
    }

    computeResidual( A, x, b, this->res_before_pre );
    //init delta_e_smoother
    delta_e_smoother.resize( b.size() );
    delta_e_smoother.set_block_dimy( b.get_block_dimy() );
    delta_e_smoother.set_block_dimx( 1 );

    if ( x.size() == b.size() )
    {
        copy( x, delta_e_smoother, offset, size );
    }
    else
    {
        fill( delta_e_smoother, 0.0, offset, size );
    }
}


template<class T_Config>
void ConvergenceAnalysis<T_Config>::stopPresmoothing(Matrix &A, VVector &x, VVector &b )
{
    if ( this->level->getLevelIndex() >= this->convergence_analysis )
    {
        return;
    }

    computeResidual( A, x, b, this->res_after_pre );
}

template<class T_Config>
void ConvergenceAnalysis<T_Config>::startCoarseGridCorrection(Matrix &A, VVector &x, VVector &b )
{
    if ( this->level->getLevelIndex() >= this->convergence_analysis )
    {
        return;
    }

    computeResidual( A, x, b, this->res_before_coarse );
    //init delta_e_coarse
    delta_e_coarse.resize( b.size() );
    delta_e_coarse.set_block_dimy( b.get_block_dimy() );
    delta_e_coarse.set_block_dimx( 1 );
    int offset, size;
    A.getOffsetAndSizeForView(OWNED, &offset, &size);
    copy( x, delta_e_coarse, offset, size );
}


template<class T_Config>
void ConvergenceAnalysis<T_Config>::stopCoarseGridCorrection(Matrix &A, VVector &x, VVector &b )
{
    if ( this->level->getLevelIndex() >= this->convergence_analysis )
    {
        return;
    }

    computeResidual( A, x, b, this->res_after_coarse );
    //get size and offets from A
    int offset, size;
    A.getOffsetAndSizeForView(OWNED, &offset, &size);
    axpy( x, this->delta_e_coarse, ValueTypeV(-1.0), offset, size );
}


template<class T_Config>
void ConvergenceAnalysis<T_Config>::startPostsmoothing(Matrix &A, VVector &x, VVector &b )
{
    if ( this->level->getLevelIndex() >= this->convergence_analysis )
    {
        return;
    }

    computeResidual( A, x, b, this->res_before_post );
    //get size and offets from A
    int offset, size;
    A.getOffsetAndSizeForView(OWNED, &offset, &size);
    axpy( x, delta_e_smoother, ValueTypeV(1.0), offset, size );
}

template<class T_Config>
void ConvergenceAnalysis<T_Config>::stopPostsmoothing(Matrix &A, VVector &x, VVector &b )
{
    if ( this->level->getLevelIndex() >= this->convergence_analysis )
    {
        return;
    }

    computeResidual( A, x, b, this->res_after_post );
    //get size and offets from A
    int offset, size;
    A.getOffsetAndSizeForView(OWNED, &offset, &size);
    axpy( x, delta_e_smoother, ValueTypeV(-1.0), offset, size );
}


template<class T_Config>
void ConvergenceAnalysis<T_Config>::computeResidual( Matrix &A, VVector &x, VVector &b, ValueTypeV &res )
{
    //get size and offets from A
    int offset, size;
    A.getOffsetAndSizeForView(OWNED, &offset, &size);

    if ( x.size() == b.size() )
    {
        //init temporary residual vector
        VVector r( b.size() );
        r.set_block_dimy(b.get_block_dimy());
        r.set_block_dimx(1);
        //compute residual
        cudaDeviceSynchronize();
        cudaCheckError();
        axmb(A, x, b, r, offset, size);
        res = nrm2( r, offset, size );
    }
    else //if x is empty, the residual is simply b
    {
        res = nrm2( b, offset, size );
    }
}

template<class T_Config>
void ConvergenceAnalysis<T_Config>::printConvergenceAnalysis(Matrix &A, VVector &x, VVector &b)
{
    if ( this->level->getLevelIndex() >= this->convergence_analysis )
    {
        return;
    }

    VVector r( b.size() );
    r.set_block_dimy(b.get_block_dimy());
    r.set_block_dimx(1);
    int offset, size;
    A.getOffsetAndSizeForView(OWNED, &offset, &size);
    axmb(A, x, b, r, offset, size);
    //compute error change norms and angle between error changes
    ValueTypeV norm_e_smoother = nrm2( this->delta_e_smoother, offset, size );
    ValueTypeV norm_e_coarse = nrm2( this->delta_e_coarse, offset, size );
    ValueTypeV dot_smoother_coarse;

    if ( norm_e_smoother == 0.0 || norm_e_coarse == 0.0 )
    {
        dot_smoother_coarse = 0.0;
    }
    else
    {
        dot_smoother_coarse = dotc( this->delta_e_smoother, this->delta_e_coarse, offset, size ) / (norm_e_smoother * norm_e_coarse);
    }

    axpy( this->delta_e_smoother, this->delta_e_coarse, ValueTypeV(1.0), offset, size );
    ValueTypeV norm_e_total = nrm2( this->delta_e_coarse, offset, size );
    //compute standard deviation of current residual
    ValueTypeV res_mean = thrust::reduce( r.begin(), r.end(), 0.0, thrust::plus<ValueTypeV>() );
    cudaCheckError();
    VVector mean_vec( r.size(), res_mean );
    axpy( r, mean_vec, ValueTypeV(-1.0), offset, size );
    ValueTypeV standard_dev = nrm2( mean_vec, offset, size ) / std::sqrt(ValueTypeV(r.size()));
    ValueTypeV norm_res = nrm2( r, offset, size );
    double pi = 4 * std::atan(double(1.0));
    std::cout << "number of rows: " << A.get_num_rows() << std::endl;
    std::cout << "error change by smoothers      : " << norm_e_smoother << std::endl;
    std::cout << "error change by coarse         : " << norm_e_coarse << std::endl;
    std::cout << "error change total             : " << norm_e_total << std::endl;
    std::cout << "angle between smoother/coarse  : " << 180 * std::acos(dot_smoother_coarse) / pi << std::endl;
    std::cout << "locality measure of residual   : " << (standard_dev / norm_res) << std::endl;
    std::cout << "residual reduction by coarse   : " << res_after_coarse / res_before_coarse << std::endl;
    std::cout << "residual reduction by smoothers: " << (res_after_pre / res_before_pre)*(res_after_post / res_before_post) << std::endl;
    std::cout << "residual reduction total       : " << res_after_post / res_before_pre << std::endl;
}

/////////////////////////////////////
//     Explicit instantiations     //
/////////////////////////////////////
#define AMGX_CASE_LINE(CASE) template class ConvergenceAnalysis<TemplateMode<CASE>::Type>;
AMGX_FORALL_BUILDS(AMGX_CASE_LINE)
#undef AMGX_CASE_LINE


}//namespace amgx






