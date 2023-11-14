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

