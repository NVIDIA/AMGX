// SPDX-FileCopyrightText: 2011 - 2025 NVIDIA CORPORATION. All Rights Reserved.
//
// SPDX-License-Identifier: BSD-3-Clause

#pragma once
#include <string>
#include <matrix_coloring/matrix_coloring.h>
#include <basic_types.h>
#include <matrix.h>

namespace amgx
{

template <class T_Config> class MinMaxMatrixColoring;

template<class T_Config>
class MinMaxMatrixColoringBase: public MatrixColoring<T_Config>
{
        typedef T_Config TConfig;
    public:
        typedef typename TConfig::MatPrec ValueType;
        typedef typename Matrix<T_Config>::index_type IndexType;

        MinMaxMatrixColoringBase(AMG_Config &cfg, const std::string &cfg_scope);

        virtual void colorMatrix(Matrix<TConfig> &A);

    protected:
        ValueType m_uncolored_fraction;

    private:
        virtual void colorMatrixOneRing(Matrix<T_Config> &A) = 0;
        virtual void colorMatrixTwoRing(Matrix<T_Config> &A) = 0;
};

// specialization for host
template <AMGX_VecPrecision t_vecPrec, AMGX_MatPrecision t_matPrec, AMGX_IndPrecision t_indPrec>
class MinMaxMatrixColoring< TemplateConfig<AMGX_host, t_vecPrec, t_matPrec, t_indPrec> > : public MinMaxMatrixColoringBase< TemplateConfig<AMGX_host, t_vecPrec, t_matPrec, t_indPrec> >
{
    public:
        typedef TemplateConfig<AMGX_host, t_vecPrec, t_matPrec, t_indPrec> TConfig_h;
        typedef Matrix<TConfig_h> Matrix_h;
        typedef typename Matrix_h::value_type ValueType;
        typedef typename Matrix_h::index_type IndexType;
        //MinMaxMatrixColoring() : MinMaxMatrixColoringBase<TConfig_h>() {}
        MinMaxMatrixColoring(AMG_Config &cfg, const std::string &cfg_scope) : MinMaxMatrixColoringBase<TConfig_h>(cfg, cfg_scope)
        { }

        void colorMatrixOneRing(Matrix_h &A);
        void colorMatrixTwoRing(Matrix_h &A);

};

// specialization for device
template <AMGX_VecPrecision t_vecPrec, AMGX_MatPrecision t_matPrec, AMGX_IndPrecision t_indPrec>
class MinMaxMatrixColoring< TemplateConfig<AMGX_device, t_vecPrec, t_matPrec, t_indPrec> >: public MinMaxMatrixColoringBase< TemplateConfig<AMGX_device, t_vecPrec, t_matPrec, t_indPrec> >
{
    public:
        typedef TemplateConfig<AMGX_device, t_vecPrec, t_matPrec, t_indPrec> TConfig_d;
        typedef Matrix<TConfig_d> Matrix_d;
        typedef typename Matrix_d::value_type ValueType;
        typedef typename Matrix_d::index_type IndexType;
        typedef typename Matrix_d ::IVector IVector;
        //MinMaxMatrixColoring() : MinMaxMatrixColoringBase<TConfig_d>() {}
        MinMaxMatrixColoring(AMG_Config &cfg, const std::string &cfg_scope) : MinMaxMatrixColoringBase<TConfig_d>(cfg, cfg_scope)
        { }

        void colorMatrixOneRing(Matrix_d &A);
        void colorMatrixTwoRing(Matrix_d &A);
};


template<class T_Config>
class MinMaxMatrixColoringFactory : public MatrixColoringFactory<T_Config>
{
    public:
        MatrixColoring<T_Config> *create(AMG_Config &cfg, const std::string &cfg_scope) { return new MinMaxMatrixColoring<T_Config>(cfg, cfg_scope); }
};

} // namespace amgx

//#include <matrix_coloring.inl>

