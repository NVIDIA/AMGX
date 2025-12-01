// SPDX-FileCopyrightText: 2011 - 2025 NVIDIA CORPORATION. All Rights Reserved.
//
// SPDX-License-Identifier: BSD-3-Clause

#pragma once
#include <string>
#include <matrix_coloring/matrix_coloring.h>
#include <basic_types.h>

namespace amgx
{

template <class T_Config> class MultiHashMatrixColoring;

template<class T_Config>
class MultiHashMatrixColoringBase: public MatrixColoring<T_Config>
{
        typedef T_Config TConfig;
    public:
        typedef typename TConfig::MatPrec ValueType;
        typedef typename Matrix<T_Config>::index_type IndexType;

        MultiHashMatrixColoringBase(AMG_Config &cfg, const std::string &cfg_scope);

        virtual void colorMatrix(Matrix<TConfig> &A);

    protected:

        ValueType m_uncolored_fraction;
        int num_hash;
        int max_num_hash;
        ColoringType m_halo_coloring;
        int reorder_matrix;

    private:
        virtual void colorMatrixOneRing(Matrix<T_Config> &A) = 0;

};

// specialization for host
template <AMGX_VecPrecision t_vecPrec, AMGX_MatPrecision t_matPrec, AMGX_IndPrecision t_indPrec>
class MultiHashMatrixColoring< TemplateConfig<AMGX_host, t_vecPrec, t_matPrec, t_indPrec> > : public MultiHashMatrixColoringBase< TemplateConfig<AMGX_host, t_vecPrec, t_matPrec, t_indPrec> >
{
    public:
        typedef TemplateConfig<AMGX_host, t_vecPrec, t_matPrec, t_indPrec> TConfig_h;
        typedef Matrix<TConfig_h> Matrix_h;
        typedef typename Matrix_h::value_type ValueType;
        typedef typename Matrix_h::index_type IndexType;
        //MultiHashMatrixColoring() : MultiHashMatrixColoringBase<TConfig_h>() {}
        MultiHashMatrixColoring(AMG_Config &cfg, const std::string &cfg_scope) : MultiHashMatrixColoringBase<TConfig_h>(cfg, cfg_scope) { }

        void colorMatrixOneRing(Matrix_h &A);

};

// specialization for device
template <AMGX_VecPrecision t_vecPrec, AMGX_MatPrecision t_matPrec, AMGX_IndPrecision t_indPrec>
class MultiHashMatrixColoring< TemplateConfig<AMGX_device, t_vecPrec, t_matPrec, t_indPrec> >: public MultiHashMatrixColoringBase< TemplateConfig<AMGX_device, t_vecPrec, t_matPrec, t_indPrec> >
{
    public:
        typedef TemplateConfig<AMGX_device, t_vecPrec, t_matPrec, t_indPrec> TConfig_d;
        typedef Matrix<TConfig_d> Matrix_d;
        typedef typename Matrix_d::value_type ValueType;
        typedef typename Matrix_d::index_type IndexType;
        typedef typename Matrix_d ::IVector IVector;
        //MultiHashMatrixColoring() : MultiHashMatrixColoringBase<TConfig_d>() {}
        MultiHashMatrixColoring(AMG_Config &cfg, const std::string &cfg_scope) : MultiHashMatrixColoringBase<TConfig_d>(cfg, cfg_scope)
        { }

        void colorMatrixOneRing(Matrix_d &A);

};

template<class T_Config>
class MultiHashMatrixColoringFactory : public MatrixColoringFactory<T_Config>
{
    public:
        MatrixColoring<T_Config> *create(AMG_Config &cfg, const std::string &cfg_scope) { return new MultiHashMatrixColoring<T_Config>(cfg, cfg_scope); }
};

} // namespace amgx

//#include <matrix_coloring.inl>

