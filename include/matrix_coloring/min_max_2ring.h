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

template< typename T_Config >
class Min_Max_2Ring_Matrix_Coloring_Base : public MatrixColoring<T_Config>
{
        typedef T_Config TConfig;

    public:
        typedef typename TConfig::MatPrec ValueType;
        typedef typename Matrix<T_Config>::index_type IndexType;

        Min_Max_2Ring_Matrix_Coloring_Base( AMG_Config &cfg, const std::string &cfg_scope);

        virtual void colorMatrix( Matrix<TConfig> &A ) = 0;

    protected:
        ValueType m_uncolored_fraction;
        int m_weakness_bound;
        bool m_late_rejection;
        ColoringType m_halo_coloring;
};

// Declare the class.
template< typename T_Config >
class Min_Max_2Ring_Matrix_Coloring
{};

// specialization for host
template< AMGX_VecPrecision V, AMGX_MatPrecision M, AMGX_IndPrecision I >
class Min_Max_2Ring_Matrix_Coloring<TemplateConfig<AMGX_host, V, M, I> > : public Min_Max_2Ring_Matrix_Coloring_Base<TemplateConfig<AMGX_host, V, M, I> >
{
    public:
        typedef TemplateConfig<AMGX_host, V, M, I> TConfig_h;
        typedef Matrix<TConfig_h> Matrix_h;
        typedef typename Matrix_h::value_type ValueType;
        typedef typename Matrix_h::index_type IndexType;

        Min_Max_2Ring_Matrix_Coloring( AMG_Config &cfg, const std::string &cfg_scope) : Min_Max_2Ring_Matrix_Coloring_Base<TConfig_h>( cfg, cfg_scope)
        {
            FatalError( "Min_Max_2Ring not available on the host", AMGX_ERR_NOT_SUPPORTED_TARGET );
        }

        void colorMatrix( Matrix<TConfig_h> &A )
        {
            FatalError( "Min_Max_2Ring not available on the host", AMGX_ERR_NOT_SUPPORTED_TARGET );
        }

};

// specialization for device
template< AMGX_VecPrecision V, AMGX_MatPrecision M, AMGX_IndPrecision I >
class Min_Max_2Ring_Matrix_Coloring<TemplateConfig<AMGX_device, V, M, I> > : public Min_Max_2Ring_Matrix_Coloring_Base<TemplateConfig<AMGX_device, V, M, I> >
{
    public:
        typedef TemplateConfig<AMGX_device, V, M, I> TConfig_d;
        typedef Matrix<TConfig_d> Matrix_d;
        typedef typename Matrix_d::value_type ValueType;
        typedef typename Matrix_d::index_type IndexType;
        typedef typename Matrix_d ::IVector IVector;

        Min_Max_2Ring_Matrix_Coloring( AMG_Config &cfg, const std::string &cfg_scope) : Min_Max_2Ring_Matrix_Coloring_Base<TConfig_d>( cfg, cfg_scope) { }

        void colorMatrix( Matrix<TConfig_d> &A );
};


template< class T_Config >
class Min_Max_2Ring_Matrix_Coloring_Factory : public MatrixColoringFactory<T_Config>
{
    public:
        MatrixColoring<T_Config> *create( AMG_Config &cfg, const std::string &cfg_scope )
        {
            return new Min_Max_2Ring_Matrix_Coloring<T_Config>(cfg, cfg_scope);
        }
};

} // namespace amgx

