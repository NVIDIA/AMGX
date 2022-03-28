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
#include <aggregation/selectors/agg_selector.h>

namespace amgx
{
namespace aggregation
{
namespace multi_pairwise
{

template <class T_Config> class MultiPairwiseSelector;

template <class T_Config>
class MultiPairwiseSelectorBase: public Selector<T_Config>
{

    public:
        typedef T_Config TConfig;
        typedef typename T_Config::MatPrec ValueType;
        typedef typename T_Config::IndPrec IndexType;
        typedef typename T_Config::MemSpace MemorySpace;
        typedef typename Matrix<TConfig>::IVector IVector;
        typedef typename Matrix<TConfig>::VVector VVector;
        typedef typename Matrix<TConfig>::MVector MVector;

        //typedefs for the weight matrix and vectors
        typedef TemplateMode<AMGX_mode_dFFI>::Type TConfig_dFFI;//possible issue: the indextype does not match the actual indextype
        typedef typename TConfig_dFFI::template setVecPrec<AMGX_vecInt>::Type ivec_value_type_dFFI;
        typedef Vector<ivec_value_type_dFFI> IVector_dFFI;
        typedef Vector<TConfig_dFFI> VVector_dFFI;


        MultiPairwiseSelectorBase(AMG_Config &cfg, const std::string &cfg_scope);
        void setAggregates( Matrix<T_Config> &A,
                            IVector &aggregates, IVector &aggregates_global, int &num_aggregates);

        // this method should really be private
        virtual void setAggregates_common_sqblocks(Matrix<T_Config> &A,
                IVector &aggregates, IVector &aggregates_global, int &num_aggregates, MVector &edge_weights, IVector &sizes) = 0;
    protected:
        int max_iterations;
        int deterministic;
        ValueType numUnassigned_tol;
        bool two_phase;
        int m_aggregation_edge_weight_component;
        int aggregation_passes; //double the size of the aggregates until target_size is reached or exceeded
        int filter_weights; //use alternate weight formula and apply weight filter
        ValueType filter_weights_alpha; //parameter for weigth filter. must be between 0 and 1. typical value: 0.25
        bool notay_weights; //use the weights that are proposed by napov and notay in "an algebraic multigrid method with guaranteed convergence rate"
        bool full_ghost_level; //compute ghost level from the original matrix (full) or from the weight matrix (not full).
        int ghost_offdiag_limit; //0 means no limit, full galerkin operator is computed. if greater than 0 then the ghost level matrix will not have more off-diagonals per row than this number.
        int merge_singletons; //decides wether to merge singletons into their strongest neighbor aggregate
        int aggregation_post_processing; //the number of post processing steps
        int weight_formula; //which weight formula is used.
        double diagonal_dominance; // diagonal dominance parameter. set to 0.0 to disable dd checking.
        bool serial_matching;// use serial matching algorithm?
        int max_aggregate_size; // maximum size for an aggregate
        bool modified_handshake; // run modified handshake?

        AMG_Config mCfg;
        std::string mCfg_scope;

        void assertRestriction( const IVector &R_row_offsets, const IVector &R_col_indices, const IVector &aggregates );
        virtual void computeIncompleteGalerkin( const Matrix<TConfig> &A,
                                                Matrix<TConfig> &Ac,
                                                const IVector &aggregates,
                                                const IVector &R_row_offsets,
                                                const IVector &R_column_indices,
                                                const int num_aggregates ) = 0;
};

// specialization for host
template <AMGX_VecPrecision t_vecPrec, AMGX_MatPrecision t_matPrec, AMGX_IndPrecision t_indPrec>
class MultiPairwiseSelector<TemplateConfig<AMGX_host, t_vecPrec, t_matPrec, t_indPrec> > : public MultiPairwiseSelectorBase< TemplateConfig<AMGX_host, t_vecPrec, t_matPrec, t_indPrec> >
{
    public:
        typedef TemplateConfig<AMGX_host, t_vecPrec, t_matPrec, t_indPrec> TConfig;
        typedef Matrix<TConfig> Matrix_h;
        typedef typename TConfig::MatPrec ValueType;
        typedef typename TConfig::IndPrec IndexType;
        typedef typename Matrix_h::IVector IVector;
        typedef typename Matrix_h::VVector VVector;
        typedef typename Matrix_h::MVector MVector;

        //typedefs for the weight matrix and vectors
        typedef TemplateMode<AMGX_mode_dFFI>::Type TConfig_dFFI;//possible issue: the indextype does not match the actual indextype
        typedef typename TConfig_dFFI::template setVecPrec<AMGX_vecInt>::Type ivec_value_type_dFFI;
        typedef Vector<ivec_value_type_dFFI> IVector_dFFI;
        typedef Vector<TConfig_dFFI> VVector_dFFI;

        MultiPairwiseSelector(AMG_Config &cfg, const std::string &cfg_scope) : MultiPairwiseSelectorBase<TConfig>(cfg, cfg_scope) {}
        void setAggregates_common_sqblocks(Matrix_h &A,
                                           IVector &aggregates, IVector &aggregates_global, int &num_aggregates, MVector &edge_weights, IVector &sizes);
    protected:
        void computeIncompleteGalerkin( const Matrix_h &A,
                                        Matrix_h &Ac,
                                        const IVector &aggregates,
                                        const IVector &R_row_offsets,
                                        const IVector &R_column_indices,
                                        const int num_aggregates );


};

// specialization for device
template <AMGX_VecPrecision t_vecPrec, AMGX_MatPrecision t_matPrec, AMGX_IndPrecision t_indPrec>
class MultiPairwiseSelector<TemplateConfig<AMGX_device, t_vecPrec, t_matPrec, t_indPrec> >: public MultiPairwiseSelectorBase< TemplateConfig<AMGX_device, t_vecPrec, t_matPrec, t_indPrec> >
{
    public:
        typedef TemplateConfig<AMGX_device, t_vecPrec, t_matPrec, t_indPrec> TConfig;
        typedef Matrix<TemplateConfig<AMGX_host, t_vecPrec, t_matPrec, t_indPrec> > Matrix_h;
        typedef Matrix<TConfig> Matrix_d;
        typedef typename TConfig::MatPrec ValueType;
        typedef typename TConfig::IndPrec IndexType;
        typedef typename Matrix_d::IVector IVector;
        typedef typename Matrix_d::VVector VVector;
        typedef typename Matrix_d::MVector MVector;

        //typedefs for the weight matrix and vectors
        typedef TemplateMode<AMGX_mode_dFFI>::Type TConfig_dFFI;//possible issue: the indextype does not match the actual indextype
        typedef typename TConfig_dFFI::template setVecPrec<AMGX_vecInt>::Type ivec_value_type_dFFI;
        typedef Vector<ivec_value_type_dFFI> IVector_dFFI;
        typedef Vector<TConfig_dFFI> VVector_dFFI;


        MultiPairwiseSelector(AMG_Config &cfg, const std::string &cfg_scope) : MultiPairwiseSelectorBase<TConfig>(cfg, cfg_scope) {}

        void setAggregates_common_sqblocks(Matrix_d &A,
                                           IVector &aggregates, IVector &aggregates_global, int &num_aggregates, MVector &edge_weights, IVector &sizes);

    protected:
        void computeIncompleteGalerkin( const Matrix_d &A,
                                        Matrix_d &Ac,
                                        const IVector &aggregates,
                                        const IVector &R_row_offsets,
                                        const IVector &R_column_indices,
                                        const int num_aggregates );

        void computeMatchingSerialGreedy( const Matrix_d &A, IVector &aggregates, int &num_aggregates, MVector &edge_weights);

};

template<class T_Config>
class MultiPairwiseSelectorFactory : public SelectorFactory<T_Config>
{
    public:
        Selector<T_Config> *create(AMG_Config &cfg, const std::string &cfg_scope) { return new MultiPairwiseSelector<T_Config>(cfg, cfg_scope); }
};
}
}
}
