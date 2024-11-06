// SPDX-FileCopyrightText: 2011 - 2024 NVIDIA CORPORATION. All Rights Reserved.
//
// SPDX-License-Identifier: BSD-3-Clause

#pragma once
#include <string>
#include <matrix_coloring/matrix_coloring.h>
#include <basic_types.h>
#include <matrix.h>

namespace amgx
{

template <class T_Config> class UniformMatrixColoring;

template<class T_Config>
class UniformMatrixColoringBase: public MatrixColoring<T_Config>
{
        typedef T_Config TConfig;
    public:
        typedef typename TConfig::MatPrec ValueType;
        typedef typename Matrix<T_Config>::index_type IndexType;

        UniformMatrixColoringBase(AMG_Config &cfg, const std::string &cfg_scope);

        void colorMatrix(Matrix<TConfig> &A);

    protected:
        ValueType m_uncolored_fraction;
        ColoringType m_halo_coloring;
};

// specialization for host
template <AMGX_VecPrecision t_vecPrec, AMGX_MatPrecision t_matPrec, AMGX_IndPrecision t_indPrec>
class UniformMatrixColoring< TemplateConfig<AMGX_host, t_vecPrec, t_matPrec, t_indPrec> > : public UniformMatrixColoringBase< TemplateConfig<AMGX_host, t_vecPrec, t_matPrec, t_indPrec> >
{
    public:
        typedef TemplateConfig<AMGX_host, t_vecPrec, t_matPrec, t_indPrec> TConfig_h;
        typedef Matrix<TConfig_h> Matrix_h;
        typedef typename Matrix_h::value_type ValueType;
        typedef typename Matrix_h::index_type IndexType;
        //UniformMatrixColoring() : UniformMatrixColoringBase<TConfig_h>() {}
        UniformMatrixColoring(AMG_Config &cfg, const std::string &cfg_scope) : UniformMatrixColoringBase<TConfig_h>(cfg, cfg_scope)
        {  }

};

// specialization for device
template <AMGX_VecPrecision t_vecPrec, AMGX_MatPrecision t_matPrec, AMGX_IndPrecision t_indPrec>
class UniformMatrixColoring< TemplateConfig<AMGX_device, t_vecPrec, t_matPrec, t_indPrec> >: public UniformMatrixColoringBase< TemplateConfig<AMGX_device, t_vecPrec, t_matPrec, t_indPrec> >
{
    public:
        typedef TemplateConfig<AMGX_device, t_vecPrec, t_matPrec, t_indPrec> TConfig_d;
        typedef Matrix<TConfig_d> Matrix_d;
        typedef typename Matrix_d::value_type ValueType;
        typedef typename Matrix_d::index_type IndexType;
        typedef typename Matrix_d ::IVector IVector;
        //UniformMatrixColoring() : UniformMatrixColoringBase<TConfig_d>() {}
        UniformMatrixColoring(AMG_Config &cfg, const std::string &cfg_scope) : UniformMatrixColoringBase<TConfig_d>(cfg, cfg_scope)
        {  }

};


template<class T_Config>
class UniformMatrixColoringFactory : public MatrixColoringFactory<T_Config>
{
    public:
        MatrixColoring<T_Config> *create(AMG_Config &cfg, const std::string &cfg_scope) { return new UniformMatrixColoring<T_Config>(cfg, cfg_scope); }
};

} // namespace amgx

//#include <matrix_coloring.inl>

