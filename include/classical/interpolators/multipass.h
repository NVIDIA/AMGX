// SPDX-FileCopyrightText: 2011 - 2024 NVIDIA CORPORATION. All Rights Reserved.
//
// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#include <classical/interpolators/interpolator.h>
#include <set>
#include <vector>
#include <amg.h>

namespace amgx
{

template <class T_Config> class  Multipass_Interpolator;

template <class T_Config>
class Multipass_InterpolatorBase : public Interpolator<T_Config>
{
        typedef T_Config TConfig;
        typedef typename TConfig::MatPrec ValueType;
        typedef typename TConfig::IndPrec IndexType;
        typedef Vector<typename TConfig::template setVecPrec<AMGX_vecInt>::Type> IntVector;
        typedef Vector<typename TConfig::template setVecPrec<AMGX_vecBool>::Type> BVector;
        typedef typename Matrix<T_Config>::MVector VVector;
        typedef typename Matrix<T_Config>::IVector IVector;

        typedef typename TConfig::template setVecPrec<AMGX_vecInt64>::Type i64vec_value_type;
        typedef Vector<i64vec_value_type> I64Vector;
        typedef typename TConfig::template setVecPrec<AMGX_vecUInt64>::Type u64vec_value_type;
        typedef Vector<u64vec_value_type> U64Vector;


    public:
        Multipass_InterpolatorBase(AMG_Config &, const std::string &) {}
        void generateInterpolationMatrix( Matrix<T_Config> &A,
                                          IntVector &cf_map,
                                          BVector &s_con,
                                          IntVector &scratch,
                                          Matrix<T_Config> &P );
    protected:

        int m_use_opt_kernels = 0;

        virtual void generateInterpolationMatrix_1x1( Matrix<T_Config> &A,
                IntVector &cf_map,
                BVector &s_con,
                IntVector &scratch,
                Matrix<T_Config> &P ) = 0;
};

template <AMGX_VecPrecision t_vecPrec, AMGX_MatPrecision t_matPrec, AMGX_IndPrecision t_indPrec>
class Multipass_Interpolator< TemplateConfig<AMGX_host, t_vecPrec, t_matPrec, t_indPrec> > : public Multipass_InterpolatorBase< TemplateConfig<AMGX_host, t_vecPrec, t_matPrec, t_indPrec> >
{
        typedef TemplateConfig<AMGX_host, t_vecPrec, t_matPrec, t_indPrec> TConfig_h;
        typedef Multipass_InterpolatorBase<TConfig_h> Base;

        typedef typename TConfig_h::MatPrec ValueType;
        typedef typename TConfig_h::IndPrec IndexType;
        typedef Vector<typename TConfig_h::template setVecPrec<AMGX_vecInt>::Type> IntVector;
        typedef Vector<typename TConfig_h::template setVecPrec<AMGX_vecBool>::Type> BVector;
        typedef typename Matrix<TConfig_h>::MVector VVector;
        typedef Matrix<TConfig_h> Matrix_h;
        typedef typename Matrix_h::IVector IVector;

        typedef typename TConfig_h::template setVecPrec<AMGX_vecInt64>::Type i64vec_value_type_h;

        typedef Vector<i64vec_value_type_h> I64Vector_h;

    public:
        Multipass_Interpolator(AMG_Config &cfg, const std::string &cfg_scope) : Base(cfg, cfg_scope) {}

    private:
        void generateInterpolationMatrix_1x1( Matrix_h &A,
                                              IntVector &cf_map,
                                              BVector &s_con,
                                              IntVector &scratch,
                                              Matrix_h &P );
};

// specialization for device
template <AMGX_VecPrecision t_vecPrec, AMGX_MatPrecision t_matPrec, AMGX_IndPrecision t_indPrec>
class Multipass_Interpolator< TemplateConfig<AMGX_device, t_vecPrec, t_matPrec, t_indPrec> >: public Multipass_InterpolatorBase< TemplateConfig<AMGX_device, t_vecPrec, t_matPrec, t_indPrec> >
{
        typedef TemplateConfig<AMGX_device, t_vecPrec, t_matPrec, t_indPrec> TConfig_d;
        typedef TemplateConfig<AMGX_host, t_vecPrec, t_matPrec, t_indPrec> TConfig_h;
        typedef Multipass_InterpolatorBase<TConfig_d> Base;

        typedef typename TConfig_d::MatPrec ValueType;
        typedef typename TConfig_d::IndPrec IndexType;
        typedef Vector<typename TConfig_d::template setVecPrec<AMGX_vecInt>::Type> IntVector;
        typedef Vector<typename TConfig_d::template setVecPrec<AMGX_vecBool>::Type> BVector;
        typedef typename Matrix<TConfig_d>::MVector VVector;
        typedef Matrix<TConfig_d> Matrix_d;
        typedef typename Matrix_d::IVector IVector;

        typedef typename TConfig_d::template setVecPrec<AMGX_vecInt64>::Type i64vec_value_type_d;
        typedef typename TConfig_h::template setVecPrec<AMGX_vecInt>::Type ivec_value_type_h;
        typedef Vector<ivec_value_type_h> IVector_h;

        typedef Vector<i64vec_value_type_d> I64Vector_d;
        typedef typename TConfig_h::template setVecPrec<AMGX_vecInt64>::Type i64vec_value_type_h;
        typedef Vector<i64vec_value_type_h> I64Vector_h;

    public:
        Multipass_Interpolator(AMG_Config &cfg, const std::string &cfg_scope);
        ~Multipass_Interpolator();

    private:
        void generateInterpolationMatrix_1x1( Matrix_d &A,
                                              IntVector &cf_map,
                                              BVector &s_con,
                                              IntVector &scratch,
                                              Matrix_d &P );

        template <int hash_size, int nthreads_per_group, class KeyType>
            void compute_c_hat_opt_dispatch(
                    const Matrix_d &A,
                    const bool *s_con,
                    const int *C_hat_start,
                    int *C_hat_size,
                    KeyType *C_hat,
                    int *C_hat_pos,
                    int *assigned,
                    IntVector &assigned_set,
                    int pass );
};

template<class T_Config>
class Multipass_InterpolatorFactory : public InterpolatorFactory<T_Config>
{
    public:
        Interpolator<T_Config> *create( AMG_Config &cfg, const std::string &cfg_scope ) { return new Multipass_Interpolator<T_Config>( cfg, cfg_scope ); }
};

} // namespace amgx
