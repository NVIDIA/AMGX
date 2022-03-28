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

#include <aggregation/coarseAgenerators/coarse_A_generator.h>
#include <matrix.h>

namespace amgx
{
namespace aggregation
{


template <class T_Config> class SerialCoarseAGenerator;

template <class T_Config>
class SerialCoarseAGeneratorBase : public CoarseAGenerator<T_Config>
{
    public:
        typedef T_Config TConfig;
        typedef typename T_Config::MatPrec ValueType;
        typedef typename T_Config::IndPrec IndexType;
        typedef typename Matrix<T_Config>::IVector IVector;

        SerialCoarseAGeneratorBase();

        void computeAOperator(const Matrix<T_Config> &A, Matrix<T_Config> &Ac, const IVector &aggregates, const IVector &R_row_offsets, const IVector &R_column_indices, const int num_aggregates);

    protected:
        virtual void computeAOperator_1x1(const Matrix<T_Config> &A, Matrix<T_Config> &Ac, const IVector &aggregates, const IVector &R_row_offsets, const IVector &R_column_indices, const int num_aggregates) = 0;
        virtual void computeAOperator_common_sqblock(const Matrix<T_Config> &A, Matrix<T_Config> &Ac, const IVector &aggregates, const IVector &R_row_offsets, const IVector &R_column_indices, const int num_aggregates) = 0;
};

// specialization for host
template <AMGX_VecPrecision t_vecPrec, AMGX_MatPrecision t_matPrec, AMGX_IndPrecision t_indPrec>
class SerialCoarseAGenerator<TemplateConfig<AMGX_host, t_vecPrec, t_matPrec, t_indPrec> > : public SerialCoarseAGeneratorBase< TemplateConfig<AMGX_host, t_vecPrec, t_matPrec, t_indPrec> >
{
    public:
        typedef TemplateConfig<AMGX_host, t_vecPrec, t_matPrec, t_indPrec> TConfig;
        typedef Matrix< TemplateConfig<AMGX_host, t_vecPrec, t_matPrec, t_indPrec> > Matrix_h;
        typedef Matrix< TemplateConfig<AMGX_device, t_vecPrec, t_matPrec, t_indPrec> > Matrix_d;
        typedef typename Matrix_h::value_type ValueType;
        typedef typename Matrix_h::index_type IndexType;
        typedef typename Matrix_h::IVector IVector;
        SerialCoarseAGenerator() : SerialCoarseAGeneratorBase<TConfig>() {}
    private:
        void computeAOperator_1x1(const Matrix_h &A, Matrix_h &Ac, const IVector &aggregates, const IVector &R_row_offsets, const IVector &R_column_indices, const int num_aggregates);
        void computeAOperator_common_sqblock(const Matrix_h &A, Matrix_h &Ac, const IVector &aggregates, const IVector &R_row_offsets, const IVector &R_column_indices, const int num_aggregates);
};

// specialization for device
template <AMGX_VecPrecision t_vecPrec, AMGX_MatPrecision t_matPrec, AMGX_IndPrecision t_indPrec>
class SerialCoarseAGenerator<TemplateConfig<AMGX_device, t_vecPrec, t_matPrec, t_indPrec> >: public SerialCoarseAGeneratorBase< TemplateConfig<AMGX_device, t_vecPrec, t_matPrec, t_indPrec> >
{
    public:
        typedef TemplateConfig<AMGX_device, t_vecPrec, t_matPrec, t_indPrec> TConfig;
        typedef Matrix< TemplateConfig<AMGX_host, t_vecPrec, t_matPrec, t_indPrec> > Matrix_h;
        typedef Matrix< TemplateConfig<AMGX_device, t_vecPrec, t_matPrec, t_indPrec> > Matrix_d;
        typedef typename Matrix_d::value_type ValueType;
        typedef typename Matrix_d::index_type IndexType;
        typedef typename Matrix_d::IVector IVector;
        SerialCoarseAGenerator() : SerialCoarseAGeneratorBase<TConfig>() {}
    private:
        void computeAOperator_1x1(const Matrix_d &A, Matrix_d &Ac, const IVector &aggregates, const IVector &R_row_offsets, const IVector &R_column_indices, const int num_aggregates);
        void computeAOperator_common_sqblock(const Matrix_d &A, Matrix_d &Ac, const IVector &aggregates, const IVector &R_row_offsets, const IVector &R_column_indices, const int num_aggregates);
};

template<class T_Config>
class SerialCoarseAGeneratorFactory : public CoarseAGeneratorFactory<T_Config>
{
    public:
        CoarseAGenerator<T_Config> *create(AMG_Config &, const std::string &) { return new SerialCoarseAGenerator<T_Config>; }
};
}
}
