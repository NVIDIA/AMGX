/* Copyright (c) 2011-2017, NVIDIA CORPORATION. All rights reserved.
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

#include<classical/interpolators/interpolator.h>

namespace amgx
{

template <class T_Config> class  Distance1_Interpolator;

template <class T_Config>
class Distance1_InterpolatorBase : public Interpolator<T_Config>
{
        typedef T_Config TConfig;
        typedef typename TConfig::MatPrec ValueType;
        typedef typename TConfig::IndPrec IndexType;
        typedef Vector<typename TConfig::template setVecPrec<AMGX_vecInt>::Type> IntVector;
        typedef Vector<typename TConfig::template setVecPrec<AMGX_vecBool>::Type> BVector;
        typedef typename Matrix<T_Config>::MVector VVector;
        typedef typename Matrix<T_Config>::IVector IVector;

    public:
        void generateInterpolationMatrix(Matrix<T_Config> &A,
                                         IntVector &cf_map,
                                         BVector &s_con,
                                         IntVector &scratch,
                                         Matrix<T_Config> &P);
    protected:
        virtual void generateInterpolationMatrix_1x1(   Matrix<T_Config> &A,
                IntVector &cf_map,
                BVector &s_con,
                IntVector &scratch,
                Matrix<T_Config> &P) = 0;
};

// specialization for host
template <AMGX_VecPrecision t_vecPrec, AMGX_MatPrecision t_matPrec, AMGX_IndPrecision t_indPrec>
class Distance1_Interpolator< TemplateConfig<AMGX_host, t_vecPrec, t_matPrec, t_indPrec> > : public Distance1_InterpolatorBase< TemplateConfig<AMGX_host, t_vecPrec, t_matPrec, t_indPrec> >
{
        typedef TemplateConfig<AMGX_host, t_vecPrec, t_matPrec, t_indPrec> TConfig_h;
        typedef typename TConfig_h::MatPrec ValueType;
        typedef typename TConfig_h::IndPrec IndexType;
        typedef Vector<typename TConfig_h::template setVecPrec<AMGX_vecInt>::Type> IntVector;
        typedef Vector<typename TConfig_h::template setVecPrec<AMGX_vecBool>::Type> BVector;
        typedef typename Matrix<TConfig_h>::MVector VVector;
        typedef Matrix<TConfig_h> Matrix_h;
        typedef typename Matrix_h::IVector IVector;
    public:
    private:
        void generateInterpolationMatrix_1x1(   Matrix_h &A,
                                                IntVector &cf_map,
                                                BVector &s_con,
                                                IntVector &scratch,
                                                Matrix_h &P);
};

// specialization for device
template <AMGX_VecPrecision t_vecPrec, AMGX_MatPrecision t_matPrec, AMGX_IndPrecision t_indPrec>
class Distance1_Interpolator< TemplateConfig<AMGX_device, t_vecPrec, t_matPrec, t_indPrec> >: public Distance1_InterpolatorBase< TemplateConfig<AMGX_device, t_vecPrec, t_matPrec, t_indPrec> >
{
        typedef TemplateConfig<AMGX_device, t_vecPrec, t_matPrec, t_indPrec> TConfig_d;
        typedef typename TConfig_d::MatPrec ValueType;
        typedef typename TConfig_d::IndPrec IndexType;
        typedef Vector<typename TConfig_d::template setVecPrec<AMGX_vecInt>::Type> IntVector;
        typedef Vector<typename TConfig_d::template setVecPrec<AMGX_vecBool>::Type> BVector;
        typedef typename Matrix<TConfig_d>::MVector VVector;
        typedef Matrix<TConfig_d> Matrix_d;
        typedef typename Matrix_d::IVector IVector;
    public:
    private:
        void generateInterpolationMatrix_1x1(   Matrix_d &A,
                                                IntVector &cf_map,
                                                BVector &s_con,
                                                IntVector &scratch,
                                                Matrix_d &P);
        void setSetsGPU(const Matrix_d &A, const BVector &s_con,
                        const IntVector &cf_map,
                        IntVector &set_fields);
};

template<class T_Config>
class Distance1_InterpolatorFactory : public InterpolatorFactory<T_Config>
{
    public:
        Interpolator<T_Config> *create(AMG_Config &cfg, const std::string &cfg_scope) { return new Distance1_Interpolator<T_Config>; }
};

} // namespace amgx
