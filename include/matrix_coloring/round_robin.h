// SPDX-FileCopyrightText: 2011 - 2024 NVIDIA CORPORATION. All Rights Reserved.
//
// SPDX-License-Identifier: BSD-3-Clause

#pragma once
#include <string>
#include <matrix_coloring/matrix_coloring.h>
#include <basic_types.h>

namespace amgx
{

template <class T_Config> class RoundRobinMatrixColoring;

template<class T_Config>
class RoundRobinMatrixColoringBase: public MatrixColoring<T_Config>
{
        typedef T_Config TConfig;
    public:
        typedef typename TConfig::MatPrec ValueType;
        typedef typename Matrix<T_Config>::index_type IndexType;

        RoundRobinMatrixColoringBase(AMG_Config &cfg, const std::string &cfg_scope);

        virtual void colorMatrix(Matrix<T_Config> &A);

    protected:
        int num_colors;
        int reorder_matrix;
        ColoringType m_halo_coloring;

    private:
        virtual void colorMatrixOneRing(Matrix<T_Config> &A) = 0;

};

// specialization for host
template <AMGX_VecPrecision t_vecPrec, AMGX_MatPrecision t_matPrec, AMGX_IndPrecision t_indPrec>
class RoundRobinMatrixColoring< TemplateConfig<AMGX_host, t_vecPrec, t_matPrec, t_indPrec> > : public RoundRobinMatrixColoringBase< TemplateConfig<AMGX_host, t_vecPrec, t_matPrec, t_indPrec> >
{
    public:
        typedef TemplateConfig<AMGX_host, t_vecPrec, t_matPrec, t_indPrec> TConfig_h;
        typedef Matrix<TConfig_h> Matrix_h;
        typedef typename Matrix_h::value_type ValueType;
        typedef typename Matrix_h::index_type IndexType;
//    RoundRobinMatrixColoring() : RoundRobinMatrixColoringBase<TConfig_h>() {}
        RoundRobinMatrixColoring(AMG_Config &cfg, const std::string &cfg_scope) : RoundRobinMatrixColoringBase<TConfig_h>(cfg, cfg_scope)
        { }
        void colorMatrixOneRing(Matrix_h &A);
};

// specialization for device
template <AMGX_VecPrecision t_vecPrec, AMGX_MatPrecision t_matPrec, AMGX_IndPrecision t_indPrec>
class RoundRobinMatrixColoring< TemplateConfig<AMGX_device, t_vecPrec, t_matPrec, t_indPrec> >: public RoundRobinMatrixColoringBase< TemplateConfig<AMGX_device, t_vecPrec, t_matPrec, t_indPrec> >
{
    public:
        typedef TemplateConfig<AMGX_device, t_vecPrec, t_matPrec, t_indPrec> TConfig_d;
        typedef Matrix<TConfig_d> Matrix_d;
        typedef typename Matrix_d::value_type ValueType;
        typedef typename Matrix_d::index_type IndexType;
        typedef typename Matrix_d ::IVector IVector;
        //RoundRobinMatrixColoring() : RoundRobinMatrixColoringBase<TConfig_d>() {}
        RoundRobinMatrixColoring(AMG_Config &cfg, const std::string &cfg_scope) : RoundRobinMatrixColoringBase<TConfig_d>(cfg, cfg_scope)
        {}

        void colorMatrixOneRing(Matrix_d &A);
};

template<class T_Config>
class RoundRobinMatrixColoringFactory : public MatrixColoringFactory<T_Config>
{
    public:
        MatrixColoring<T_Config> *create(AMG_Config &cfg, const std::string &cfg_scope) { return new RoundRobinMatrixColoring<T_Config>(cfg, cfg_scope); }
};

} // namespace amgx

//#include <matrix_coloring.inl>

