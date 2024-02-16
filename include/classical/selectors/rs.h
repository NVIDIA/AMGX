// SPDX-FileCopyrightText: 2011 - 2024 NVIDIA CORPORATION. All Rights Reserved.
//
// SPDX-License-Identifier: BSD-3-Clause

#pragma once
//namespace amgx {
//namespace classical {
//template <class Matrix, class Vector> class RS_Selector;
//}
//}

#include <classical/selectors/selector.h>

namespace amgx
{

namespace classical
{

template <class T_Config> class  RS_Selector;

template <class T_Config>
class RS_SelectorBase : public Selector<T_Config>
{
        typedef T_Config TConfig;
        typedef typename TConfig::MatPrec ValueType;
        typedef typename TConfig::IndPrec IndexType;
        typedef Vector<typename TConfig::template setVecPrec<AMGX_vecBool>::Type> BVector;
        typedef Vector<typename TConfig::template setVecPrec<AMGX_vecFloat>::Type> FVector;
        typedef typename Matrix<T_Config>::MVector VVector;
        typedef typename Matrix<T_Config>::IVector IVector;
        typedef Vector<typename TConfig::template setVecPrec<AMGX_vecInt>::Type> IntVector;
    public:
        void markCoarseFinePoints(Matrix<T_Config> &A,
                                  FVector &weights,
                                  const BVector &s_con,
                                  IVector &cf_map,
                                  IVector &scratch,
                                  int cf_map_init = 0);

        RS_SelectorBase(AMG_Config &cfg, const std::string &cfg_scope) : 
          Selector<T_Config>(cfg, cfg_scope) {}

    protected:
        virtual void markCoarseFinePoints_1x1(Matrix<T_Config> &A,
                                              FVector &weights,
                                              const BVector &s_con,
                                              IVector &cf_map,
                                              IVector &scratch,
                                              int cf_map_init = 0) = 0;
};


// specialization for host
template <AMGX_VecPrecision t_vecPrec, AMGX_MatPrecision t_matPrec, AMGX_IndPrecision t_indPrec>
class RS_Selector< TemplateConfig<AMGX_host, t_vecPrec, t_matPrec, t_indPrec> > : public RS_SelectorBase< TemplateConfig<AMGX_host, t_vecPrec, t_matPrec, t_indPrec> >
{
        typedef TemplateConfig<AMGX_host, t_vecPrec, t_matPrec, t_indPrec> TConfig_h;
        typedef typename TConfig_h::MatPrec ValueType;
        typedef typename TConfig_h::IndPrec IndexType;
        typedef Vector<typename TConfig_h::template setVecPrec<AMGX_vecBool>::Type> BVector;
        typedef Vector<typename TConfig_h::template setVecPrec<AMGX_vecFloat>::Type> FVector;
        typedef typename Matrix<TConfig_h>::MVector VVector;
        typedef Matrix<TConfig_h> Matrix_h;
        typedef typename Matrix_h::IVector IVector;
        typedef Vector<typename TConfig_h::template setVecPrec<AMGX_vecInt>::Type> IntVector;
    public:
        RS_Selector(AMG_Config &cfg, const std::string &cfg_scope) : 
          RS_SelectorBase<TConfig_h>(cfg, cfg_scope) {}
    private:
        void markCoarseFinePoints_1x1(Matrix_h &A,
                                      FVector &weights,
                                      const BVector &s_con,
                                      IVector &cf_map,
                                      IVector &scratch,
                                      int cf_map_init = 0);
};

// specialization for device
template <AMGX_VecPrecision t_vecPrec, AMGX_MatPrecision t_matPrec, AMGX_IndPrecision t_indPrec>
class RS_Selector< TemplateConfig<AMGX_device, t_vecPrec, t_matPrec, t_indPrec> >: public RS_SelectorBase< TemplateConfig<AMGX_device, t_vecPrec, t_matPrec, t_indPrec> >
{
        typedef TemplateConfig<AMGX_device, t_vecPrec, t_matPrec, t_indPrec> TConfig_d;
        typedef typename TConfig_d::template setMemSpace<AMGX_host>::Type TConfig_h;

        typedef typename TConfig_d::MatPrec ValueType;
        typedef typename TConfig_d::IndPrec IndexType;
        typedef Vector<typename TConfig_d::template setVecPrec<AMGX_vecBool>::Type> BVector;
        typedef Vector<typename TConfig_d::template setVecPrec<AMGX_vecFloat>::Type> FVector;
        typedef Vector<typename TConfig_h::template setVecPrec<AMGX_vecBool>::Type> BVector_h;
        typedef Vector<typename TConfig_h::template setVecPrec<AMGX_vecFloat>::Type> FVector_h;
        typedef typename Matrix<TConfig_d>::MVector VVector;
        typedef Matrix<TConfig_d> Matrix_d;
        typedef Matrix<TConfig_h> Matrix_h;
        typedef typename Matrix_d::IVector IVector;
        typedef typename Matrix_h::IVector IVector_h;
        typedef Vector<typename TConfig_d::template setVecPrec<AMGX_vecInt>::Type> IntVector;
    public:
        RS_Selector(AMG_Config &cfg, const std::string &cfg_scope) : 
          RS_SelectorBase<TConfig_d>(cfg, cfg_scope) {}
    private:
        void markCoarseFinePoints_1x1(Matrix_d &A,
                                      FVector &weights,
                                      const BVector &s_con,
                                      IVector &cf_map,
                                      IVector &scratch,
                                      int cf_map_init = 0);
};

template<class T_Config>
class RS_SelectorFactory : public SelectorFactory<T_Config>
{
    public:
        Selector<T_Config> *create(AMG_Config &cfg, const std::string &cfg_scope) 
        { 
          return new RS_Selector<T_Config>(cfg, cfg_scope);
        }
};

} // namespace classical

} // namespace amgx

