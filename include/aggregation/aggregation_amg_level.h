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
namespace amgx
{
namespace aggregation
{
template <class T_Config> class Aggregation_AMG_Level;
}
}

#include <string>
#include <string.h>
#include <basic_types.h>
#include <amg_level.h>
#include <aggregation/selectors/agg_selector.h>
#include <aggregation/coarseAgenerators/coarse_A_generator.h>

namespace amgx
{
/***************************************************
 * Aggregation AMG Base Class
 *  Defines the AMG solve algorithm, decendents must
 *  define markCoarseFinePoints() and
 *  generateInterpoloationMatrix()
 **************************************************/
namespace aggregation
{

template <class T_Config> class Aggregation_AMG_Level;

template <class T_Config>
class Aggregation_AMG_Level_Base : public AMG_Level<T_Config>
{
    public:
        typedef T_Config TConfig;
//    typedef typename TraitsFromMatrix<Matrix>::Traits MatrixTraits;
        typedef typename TConfig::MatPrec ValueTypeA;
        typedef typename TConfig::IndPrec IndexType;
        typedef typename TConfig::MemSpace MemorySpace;
        typedef typename TConfig::VecPrec ValueTypeB;
        static const AMGX_VecPrecision vecPrec = T_Config::vecPrec;
        static const AMGX_MatPrecision matPrec = T_Config::matPrec;
        static const AMGX_IndPrecision indPrec = T_Config::indPrec;
        typedef AMG<vecPrec, matPrec, indPrec> AMG_Class;
        typedef Vector<TConfig> VVector;
        typedef typename Matrix<TConfig>::IVector IVector;
        typedef typename Matrix<TConfig>::MVector MVector;
        typedef typename TConfig::template setMemSpace<AMGX_host>::Type TConfig_h;
        typedef typename Matrix<TConfig_h>::IVector IVector_h;
        typedef typename TConfig::template setVecPrec<AMGX_vecInt>::Type ivec_value_type;
        typedef typename TConfig_h::template setVecPrec<AMGX_vecInt>::Type ivec_value_type_h;
        typedef typename ivec_value_type_h::VecPrec VecInt_t;

        typedef typename MemorySpaceMap<AMGX_host>::Type host_memory;
        typedef typename MemorySpaceMap<AMGX_device>::Type device_memory;
        static const AMGX_MemorySpace other_memspace = MemorySpaceMap<opposite_memspace<TConfig::memSpace>::memspace>::id;
        typedef TemplateConfig<other_memspace, vecPrec, matPrec, indPrec> TConfig1;
        typedef TConfig1 T_Config1;

        friend class Aggregation_AMG_Level_Base<TConfig1>;

        Aggregation_AMG_Level_Base(AMG_Class *amg, ThreadManager *tmng) : AMG_Level<T_Config>(amg, tmng)
        {
            m_selector = SelectorFactory<T_Config>::allocate(*(amg->m_cfg), amg->m_cfg_scope);
            m_coarseAGenerator = CoarseAGeneratorFactory<T_Config>::allocate(*(amg->m_cfg), amg->m_cfg_scope);
            m_matrix_halo_exchange = amg->m_cfg->AMG_Config::template getParameter<int>("matrix_halo_exchange", amg->m_cfg_scope);
            m_print_aggregation_info = amg->m_cfg->AMG_Config::template getParameter<int>("print_aggregation_info", amg->m_cfg_scope) != 0;
            m_error_scaling = amg->m_cfg->AMG_Config::template getParameter<int>("error_scaling", amg->m_cfg_scope );
            reuse_scale = amg->m_cfg->AMG_Config::template getParameter<int>("reuse_scale", amg->m_cfg_scope );
            scaling_smoother_steps = amg->m_cfg->AMG_Config::template getParameter<int>("scaling_smoother_steps", amg->m_cfg_scope );
            scale_counter = 0;
        }

        virtual void transfer_level(AMG_Level<TConfig1> *ref_lvl);

        virtual ~Aggregation_AMG_Level_Base()
        {
            delete m_selector;
            delete m_coarseAGenerator;
        }

        void createCoarseVertices();
        void createCoarseMatrices();
        bool isClassicalAMGLevel() { return false; }
        IndexType getNumCoarseVertices()
        {
            return this->m_num_aggregates;
        }

        void restrictResidual(VVector &r, VVector &rr);
        void prolongateAndApplyCorrection( VVector &c, VVector &bc, VVector &x, VVector &tmp);
        void computeRestrictionOperator();
        void consolidateVector(VVector &x);
        void unconsolidateVector(VVector &x);


    protected:

        typedef Vector<TemplateConfig<AMGX_device, AMGX_vecInt, matPrec, indPrec> > IntVector_d;

        void computeAOperator();
        void computeAOperatorThrust();
        void setNeighborAggregates();
        void createCoarseB2LMaps(std::vector<IVector> &in_coarse_B2L_maps);
        void prepareNextLevelMatrix(const Matrix<TConfig> &A, Matrix<TConfig> &Ac);
        void prepareNextLevelMatrix_full(const Matrix<TConfig> &A, Matrix<TConfig> &Ac);
        void prepareNextLevelMatrix_diag(const Matrix<TConfig> &A, Matrix<TConfig> &Ac);
        void prepareNextLevelMatrix_none(const Matrix<TConfig> &A, Matrix<TConfig> &Ac);
        void fillRowOffsetsAndColIndices(const int R_num_cols);

        void consolidationBookKeeping();
        void consolidateCoarseGridMatrix();

        typename TConfig::VecPrec computeAlpha(const VVector &e, const VVector &bc, const VVector &tmp);

        void dumpMatrices(IntVector_d &Ac_row_offsets, IntVector_d &A_row_offsets, IntVector_d &A_column_indices, VVector &A_dia_values, VVector &A_nonzero_values, IntVector_d &R_row_offsets, IntVector_d &R_column_indices, IntVector_d &aggregates, IndexType num_aggregates);

        int m_num_aggregates;
        int m_num_all_aggregates;
        int m_matrix_halo_exchange;
        bool m_print_aggregation_info;
        int m_error_scaling;
        int reuse_scale;
        int scaling_smoother_steps;

        int scale_counter;
        ValueTypeB scale;

        IVector m_R_row_offsets;
        IVector m_R_column_indices;
        IVector m_aggregates;
        IVector m_aggregates_fine_idx;

        Selector<TConfig> *m_selector;
        CoarseAGenerator<TConfig> *m_coarseAGenerator;

        void computeRestrictionOperator_common();

        // consolidation related bookkeeping
        IVector_h              m_fine_parts_to_consolidate;
        std::vector<IVector_h> m_vertex_counts;

        // temporary storage for consolidation related bookkeeping
        // that is stored in the DistributedManager of the coarse
        // matrix after it has been constructed the first time
        int m_total_interior_rows_in_merged = -1;
        int m_total_boundary_rows_in_merged = -1;
        IVector_h            m_consolidated_neighbors;
        std::vector<IVector> m_consolidated_B2L_maps;
        IVector_h            m_consolidated_halo_offsets;

    private:
        virtual void prolongateAndApplyCorrection_4x4(VVector &c, VVector &bc, VVector &x, VVector &tmp) = 0;
        virtual void prolongateAndApplyCorrection_1x1(VVector &c, VVector &bc, VVector &x, VVector &tmp) = 0;
        virtual void restrictResidual_1x1(const VVector &r, VVector &rr) = 0;
        virtual void restrictResidual_4x4(const VVector &r, VVector &rr) = 0;
        virtual void computeRestrictionOperator_1x1() = 0;
        virtual void computeRestrictionOperator_4x4() = 0 ;

};

// specialization for host
template <AMGX_VecPrecision t_vecPrec, AMGX_MatPrecision t_matPrec, AMGX_IndPrecision t_indPrec>
class Aggregation_AMG_Level< TemplateConfig<AMGX_host, t_vecPrec, t_matPrec, t_indPrec> > : public Aggregation_AMG_Level_Base< TemplateConfig<AMGX_host, t_vecPrec, t_matPrec, t_indPrec> >
{
    public:
        typedef TemplateConfig<AMGX_host, t_vecPrec, t_matPrec, t_indPrec> TConfig;
        typedef typename TConfig::MatPrec ValueTypeA;
        typedef typename TConfig::IndPrec IndexType;
        typedef typename TConfig::MemSpace MemorySpace;
        typedef typename TConfig::VecPrec ValueTypeB;
        typedef AMG<t_vecPrec, t_matPrec, t_indPrec> AMG_Class;
        typedef Vector<TConfig> VVector;
        typedef typename Matrix<TConfig>::IVector IVector;
        //constructors
        //Aggregation_AMG_Level() : Aggregation_AMG_Level_Base<TConfig>() {}
        Aggregation_AMG_Level(AMG_Class *amg, ThreadManager *tmng) : Aggregation_AMG_Level_Base<TConfig>(amg, tmng) {}
    private:
        virtual void prolongateAndApplyCorrection_4x4(VVector &c, VVector &bc, VVector &x, VVector &tmp);
        virtual void prolongateAndApplyCorrection_1x1(VVector &c, VVector &bc, VVector &x, VVector &tmp);
        virtual void restrictResidual_1x1(const VVector &r, VVector &rr);
        virtual void restrictResidual_4x4(const VVector &r, VVector &rr);
        virtual void computeRestrictionOperator_1x1();
        virtual void computeRestrictionOperator_4x4();
};

// specialization for device
template <AMGX_VecPrecision t_vecPrec, AMGX_MatPrecision t_matPrec, AMGX_IndPrecision t_indPrec>
class Aggregation_AMG_Level< TemplateConfig<AMGX_device, t_vecPrec, t_matPrec, t_indPrec> > : public Aggregation_AMG_Level_Base< TemplateConfig<AMGX_device, t_vecPrec, t_matPrec, t_indPrec> >
{
    public:
        typedef TemplateConfig<AMGX_device, t_vecPrec, t_matPrec, t_indPrec> TConfig;
        typedef typename TConfig::MatPrec ValueTypeA;
        typedef typename TConfig::IndPrec IndexType;
        typedef typename TConfig::MemSpace MemorySpace;
        typedef typename TConfig::VecPrec ValueTypeB;
        typedef AMG<t_vecPrec, t_matPrec, t_indPrec> AMG_Class;
        typedef Vector<TConfig> VVector;
        typedef typename Matrix<TConfig>::IVector IVector;
        //constructors
        //Aggregation_AMG_Level() : Aggregation_AMG_Level_Base<TConfig>() {}
        Aggregation_AMG_Level(AMG_Class *amg, ThreadManager *tmng) : Aggregation_AMG_Level_Base<TConfig>(amg, tmng) {}
    private:
        virtual void prolongateAndApplyCorrection_4x4(VVector &c, VVector &bc, VVector &x, VVector &tmp);
        virtual void prolongateAndApplyCorrection_1x1(VVector &c, VVector &bc, VVector &x, VVector &tmp);
        virtual void restrictResidual_1x1(const VVector &r, VVector &rr);
        virtual void restrictResidual_4x4(const VVector &r, VVector &rr);
        virtual void computeRestrictionOperator_1x1();
        virtual void computeRestrictionOperator_4x4();
};
}

template<typename T_Config>
class Aggregation_AMG_LevelFactory : public AMG_LevelFactory<T_Config>
{
    public:
        static const AMGX_VecPrecision vecPrec = T_Config::vecPrec;
        static const AMGX_MatPrecision matPrec = T_Config::matPrec;
        static const AMGX_IndPrecision indPrec = T_Config::indPrec;
        typedef AMG<vecPrec, matPrec, indPrec> AMG_Class;
        AMG_Level<T_Config> *create(AMG_Class *amg, ThreadManager *tmng) { return new aggregation::Aggregation_AMG_Level<T_Config>(amg, tmng); }
};


}
