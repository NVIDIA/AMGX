// SPDX-FileCopyrightText: 2011 - 2024 NVIDIA CORPORATION. All Rights Reserved.
//
// SPDX-License-Identifier: BSD-3-Clause

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
