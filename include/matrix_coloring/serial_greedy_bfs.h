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

#include <string>
#include <matrix_coloring/matrix_coloring.h>
#include <basic_types.h>
#include <matrix.h>

//MatrixColoringFactory<T_Config>::registerFactory("SERIAL_GREEDY_BFS", new Serial_Greedy_BFS_MatrixColoring_Factory<T_Config>);
//MatrixColoringFactory<T_Config>::registerFactory("GREEDY_RECOLOR", new Greedy_Recolor_Coloring_Factory<T_Config>);

namespace amgx
{

template< typename T_Config >
class Serial_Greedy_BFS_MatrixColoring_Base : public MatrixColoring<T_Config>
{
        typedef T_Config TConfig;

    public:
        typedef typename TConfig::MatPrec ValueType;
        typedef typename Matrix<T_Config>::index_type IndexType;

        Serial_Greedy_BFS_MatrixColoring_Base( AMG_Config &cfg, const std::string &cfg_scope );

        virtual void colorMatrix( Matrix<TConfig> &A ) = 0;


    protected:
        AMG_Config fallback_config;
        std::string fallback_config_scope;

        std::string m_coloring_custom_arg;
        int m_coloring_try_remove_last_color_;
        ColoringType m_halo_coloring;

        thrust::host_vector<int> aggregates; //for experimental aggregator
        bool dummy_coloring;
        bool ready_for_coloring_arrays;
};

// Declare the class.
template< typename T_Config >
class Serial_Greedy_BFS_MatrixColoring
{};

// specialization for host
template< AMGX_VecPrecision V, AMGX_MatPrecision M, AMGX_IndPrecision I >
class Serial_Greedy_BFS_MatrixColoring<TemplateConfig<AMGX_host, V, M, I> > : public Serial_Greedy_BFS_MatrixColoring_Base<TemplateConfig<AMGX_host, V, M, I> >
{
    public:
        typedef TemplateConfig<AMGX_host, V, M, I> TConfig_h;
        typedef Matrix<TConfig_h> Matrix_h;
        typedef typename Matrix_h::value_type ValueType;
        typedef typename Matrix_h::index_type IndexType;

        Serial_Greedy_BFS_MatrixColoring( AMG_Config &cfg, const std::string &cfg_scope ) : Serial_Greedy_BFS_MatrixColoring_Base<TConfig_h>( cfg, cfg_scope)
        {
            FatalError( "Serial_Greedy_BFS not available on the host", AMGX_ERR_NOT_SUPPORTED_TARGET );
        }

        void colorMatrix( Matrix<TConfig_h> &A )
        {
            FatalError( "Serial_Greedy_BFS not available on the host", AMGX_ERR_NOT_SUPPORTED_TARGET );
        }
};

// specialization for device
template< AMGX_VecPrecision V, AMGX_MatPrecision M, AMGX_IndPrecision I >
class Serial_Greedy_BFS_MatrixColoring<TemplateConfig<AMGX_device, V, M, I> > : public Serial_Greedy_BFS_MatrixColoring_Base<TemplateConfig<AMGX_device, V, M, I> >
{
    public:
        typedef TemplateConfig<AMGX_device, V, M, I> TConfig_d;
        typedef Matrix<TConfig_d> Matrix_d;
        typedef typename Matrix_d::value_type ValueType;
        typedef typename Matrix_d::index_type IndexType;
        typedef typename Matrix_d ::IVector IVector;

        Serial_Greedy_BFS_MatrixColoring( AMG_Config &cfg, const std::string &cfg_scope ) : Serial_Greedy_BFS_MatrixColoring_Base<TConfig_d>( cfg, cfg_scope )
        {  }

        void colorMatrix( Matrix<TConfig_d> &A );
        void createColorArrays(Matrix<TConfig_d> &A);

        void run_createColorArrays_on_cpu(int num_rows, int *color, int *sorted_rows_by_color);
        void run_cpu_parallel_greedy_equivalent(Matrix_d &A, int *color, int *queue, int *A_row_offsets, int *A_col_indices);
        void run_cpu_serial_bfs_greedy(Matrix_d &A, int *color, int *queue, int *A_row_offsets, int *A_col_indices);
        void run_cpu_serial_IDO_ordering_greedy(Matrix_d &A, int *color, int *queue, int *A_row_offsets, int *A_col_indices);

};


template< class T_Config >
class Serial_Greedy_BFS_MatrixColoring_Factory : public MatrixColoringFactory<T_Config>
{
    public:
        MatrixColoring<T_Config> *create( AMG_Config &cfg, const std::string &cfg_scope )
        {
            return new Serial_Greedy_BFS_MatrixColoring<T_Config>(cfg, cfg_scope);
        }
};

} // namespace amgx

