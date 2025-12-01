// SPDX-FileCopyrightText: 2011 - 2025 NVIDIA CORPORATION. All Rights Reserved.
//
// SPDX-License-Identifier: BSD-3-Clause

#pragma once
namespace amgx
{
template<class T_Config> class Strength;
}

#include <getvalue.h>
#include <error.h>
#include <basic_types.h>
#include <amg_config.h>
#include <classical/strength/strength.h>

namespace amgx
{

template <class T_Config> class  Strength_Base;

template<class T_Config>
class Strength_BaseBase : public Strength<T_Config>
{
        typedef T_Config TConfig;
        typedef typename TConfig::MatPrec ValueType;
        typedef typename T_Config::VecPrec ValueTypeB;
        typedef typename TConfig::IndPrec IndexType;
        typedef Vector<typename TConfig::template setVecPrec<AMGX_vecBool>::Type> BVector;
        typedef Vector<typename TConfig::template setVecPrec<AMGX_vecFloat>::Type> FVector;
        typedef typename Matrix<TConfig>::MVector MVector;

    public:
        Strength_BaseBase(AMG_Config &cfg, const std::string &cfg_scope);
        void computeStrongConnectionsAndWeights(Matrix<T_Config> &A,
                                                BVector &s_con,
                                                FVector &weights,
                                                const double max_row_sum);

        void computeWeights(Matrix<T_Config> &S,
                            FVector &weights);

        __host__ __device__
        virtual bool strongly_connected(ValueType val, ValueType threshold, ValueType diagonal) = 0;
    protected:
        virtual void computeStrongConnectionsAndWeights_1x1(Matrix<T_Config> &A,
                BVector &s_con,
                FVector &weights,
                const double max_row_sum) = 0;
        virtual void computeWeights_1x1(Matrix<T_Config> &S,
                                        FVector &weights) = 0;
        ValueType alpha;
        int m_use_opt_kernels = 0;
};

// specialization for host
template <AMGX_VecPrecision t_vecPrec, AMGX_MatPrecision t_matPrec, AMGX_IndPrecision t_indPrec>
class  Strength_Base< TemplateConfig<AMGX_host, t_vecPrec, t_matPrec, t_indPrec> > : public  Strength_BaseBase< TemplateConfig<AMGX_host, t_vecPrec, t_matPrec, t_indPrec> >
{
        typedef TemplateConfig<AMGX_host, t_vecPrec, t_matPrec, t_indPrec> TConfig_h;
        typedef typename TConfig_h::MatPrec ValueType;
        typedef typename TConfig_h::IndPrec IndexType;
        typedef Vector<typename TConfig_h::template setVecPrec<AMGX_vecBool>::Type> BVector;
        typedef Vector<typename TConfig_h::template setVecPrec<AMGX_vecFloat>::Type> FVector;
        typedef typename Matrix<TConfig_h>::MVector VVector;
        typedef Matrix<TConfig_h> Matrix_h;
    public:
        Strength_Base(AMG_Config &cfg, const std::string &cfg_scope) : Strength_BaseBase<TConfig_h>(cfg, cfg_scope) {}
    private:
        void computeStrongConnectionsAndWeights_1x1(Matrix_h &A,
                BVector &s_con,
                FVector &weights,
                const double max_row_sum);
        void computeWeights_1x1(Matrix_h &S,
                                FVector &weights);
};

// specialization for device
template <AMGX_VecPrecision t_vecPrec, AMGX_MatPrecision t_matPrec, AMGX_IndPrecision t_indPrec>
class  Strength_Base< TemplateConfig<AMGX_device, t_vecPrec, t_matPrec, t_indPrec> >: public  Strength_BaseBase< TemplateConfig<AMGX_device, t_vecPrec, t_matPrec, t_indPrec> >
{
        typedef TemplateConfig<AMGX_device, t_vecPrec, t_matPrec, t_indPrec> TConfig_d;
        typedef typename TConfig_d::MatPrec ValueType;
        typedef typename TConfig_d::IndPrec IndexType;
        typedef Vector<typename TConfig_d::template setVecPrec<AMGX_vecBool>::Type> BVector;
        typedef Vector<typename TConfig_d::template setVecPrec<AMGX_vecFloat>::Type> FVector;
        typedef typename Matrix<TConfig_d>::MVector VVector;
        typedef Matrix<TConfig_d> Matrix_d;
    public:
        Strength_Base(AMG_Config &cfg, const std::string &cfg_scope) : Strength_BaseBase<TConfig_d>(cfg, cfg_scope) {}
    private:
        void computeStrongConnectionsAndWeights_1x1(Matrix_d &A,
                BVector &s_con,
                FVector &weights,
                const double max_row_sum);
        void computeWeights_1x1(Matrix_d &S,
                                FVector &weights);
};

} // namespace amgx

