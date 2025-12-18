// SPDX-FileCopyrightText: 2011 - 2025 NVIDIA CORPORATION. All Rights Reserved.
//
// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#include <classical/selectors/selector.h>

namespace amgx
{

namespace classical
{

template <class T_Config> class  Dummy_Selector;

template <class T_Config>
class Dummy_SelectorBase : public Selector<T_Config>
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

        void demoteStrongEdges(const Matrix<TConfig> &A,
                               const FVector &weights,
                               BVector &s_con,
                               const IVector &cf_map,
                               const IndexType offset) {};

        Dummy_SelectorBase(AMG_Config &cfg, const std::string &cfg_scope) : 
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
class Dummy_Selector< TemplateConfig<AMGX_host, t_vecPrec, t_matPrec, t_indPrec> > : public Dummy_SelectorBase< TemplateConfig<AMGX_host, t_vecPrec, t_matPrec, t_indPrec> >
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
        Dummy_Selector(AMG_Config &cfg, const std::string &cfg_scope) : 
          Dummy_SelectorBase<TConfig_h>(cfg, cfg_scope) {}
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
class Dummy_Selector< TemplateConfig<AMGX_device, t_vecPrec, t_matPrec, t_indPrec> >: public Dummy_SelectorBase< TemplateConfig<AMGX_device, t_vecPrec, t_matPrec, t_indPrec> >
{
        typedef TemplateConfig<AMGX_device, t_vecPrec, t_matPrec, t_indPrec> TConfig_d;
        typedef typename TConfig_d::MatPrec ValueType;
        typedef typename TConfig_d::IndPrec IndexType;
        typedef Vector<typename TConfig_d::template setVecPrec<AMGX_vecBool>::Type> BVector;
        typedef Vector<typename TConfig_d::template setVecPrec<AMGX_vecFloat>::Type> FVector;
        typedef typename Matrix<TConfig_d>::MVector VVector;
        typedef Matrix<TConfig_d> Matrix_d;
        typedef typename Matrix_d::IVector IVector;
        typedef Vector<typename TConfig_d::template setVecPrec<AMGX_vecInt>::Type> IntVector;
    public:
        Dummy_Selector(AMG_Config &cfg, const std::string &cfg_scope) : 
          Dummy_SelectorBase<TConfig_d>(cfg, cfg_scope) {}
    private:
        void markCoarseFinePoints_1x1(Matrix_d &A,
                                      FVector &weights,
                                      const BVector &s_con,
                                      IVector &cf_map,
                                      IVector &scratch,
                                      int cf_map_init = 0);

};

template<class T_Config>
class Dummy_SelectorFactory : public SelectorFactory<T_Config>
{
    public:
        Selector<T_Config> *create(AMG_Config &cfg, const std::string &cfg_scope) 
        { 
          return new Dummy_Selector<T_Config>(cfg, cfg_scope); 
        }
};

} // namespace classical

} // namespace amgx

