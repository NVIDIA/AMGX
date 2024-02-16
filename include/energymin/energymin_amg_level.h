// SPDX-FileCopyrightText: 2011 - 2024 NVIDIA CORPORATION. All Rights Reserved.
//
// SPDX-License-Identifier: BSD-3-Clause

#pragma once
namespace amgx
{
namespace energymin
{
template <class T_Config> class Energymin_AMG_Level;
}
}

#include <amg_level.h>
#include <energymin/selectors/em_selector.h>
#include <energymin/interpolators/em_interpolator.h>
#include <classical/selectors/selector.h>
#include <classical/strength/strength.h>

namespace amgx
{

/***************************************************
 * Energymin AMG Base Class
 *  Defines the AMG solve algorithm
 **************************************************/
namespace energymin
{

template <class T_Config>
class Energymin_AMG_Level_Base : public AMG_Level<T_Config>
{
        //typedef typename TraitsFromMatrix<Matrix>::Traits MatrixTraits;
        typedef T_Config TConfig;
        typedef typename TConfig::template setMemSpace<AMGX_host>::Type TConfig_h;
        static const AMGX_VecPrecision vecPrec = TConfig::vecPrec;
        static const AMGX_MatPrecision matPrec = TConfig::matPrec;
        static const AMGX_IndPrecision indPrec = TConfig::indPrec;
        typedef AMG<vecPrec, matPrec, indPrec> AMG_Class;
        typedef typename TConfig::IndPrec IndexType;
        typedef typename TConfig::MatPrec ValueType;
        typedef Vector<TConfig> VVector;

        typedef typename Matrix<TConfig_h>::IVector IVector_h;

        typedef typename MemorySpaceMap<AMGX_host>::Type host_memory;
        typedef typename MemorySpaceMap<AMGX_device>::Type device_memory;
        static const AMGX_MemorySpace other_memspace = MemorySpaceMap<opposite_memspace<TConfig::memSpace>::memspace>::id;
        typedef TemplateConfig<other_memspace, vecPrec, matPrec, indPrec> TConfig1;
        typedef TConfig1 T_Config1;

    public:
        typedef Vector<typename TConfig::template setVecPrec<AMGX_vecInt>::Type> IVector;

    public:
        Energymin_AMG_Level_Base(AMG_Class *amg);
        virtual ~Energymin_AMG_Level_Base();

        virtual void transfer_level(AMG_Level<TConfig1> *ref_lvl) {FatalError("transfer levels is not implemented for energymin", AMGX_ERR_NOT_IMPLEMENTED);};

        void createCoarseVertices();
        void createCoarseMatrices();
        IndexType getNumCoarseVertices()
        {
            return this->m_num_coarse_vertices;
        }

        void setNumCoarseVertices(int num_owned_coarse_pts)
        {
            this->m_num_coarse_vertices = num_owned_coarse_pts;
        }

        bool isClassicalAMGLevel() { return false; }
        void restrictResidual(VVector &r, VVector &rr);
        void prolongateAndApplyCorrection( VVector &c, VVector &bc, VVector &x, VVector &tmp);

        virtual void computeAOperator_1x1() = 0;
        virtual void computeAOperator_1x1_distributed() = 0;
        void prepareNextLevelMatrix(const Matrix<TConfig> &A, Matrix<TConfig> &Ac) {};
        void consolidateVector(VVector &x) {};
        void unconsolidateVector(VVector &x) {};
        void prolongateAndApplyCorrectionRescale(VVector &ec, VVector &bf, VVector &xf, VVector &ef, VVector &Aef);

    protected:
        typedef Vector<typename T_Config::template setVecPrec<AMGX_vecBool>::Type> BVector;

        void computeProlongationOperator();
        void markCoarseFinePoints();
        void computeRestrictionOperator();
        void computeAOperator();
        void computeAOperator_distributed();
        Matrix<TConfig> P, R;

        amgx::classical::Selector<TConfig> *selector;
        amgx::Strength<TConfig> *strength;
        int max_row_sum;
        Interpolator<TConfig> *interpolator;
        //Strength<TConfig> *strength;

        //double max_row_sum;
        //double trunc_factor;
        //int max_elmts;
        int m_num_coarse_vertices;
        //int num_aggressive_levels;
        //BVector m_s_con;
        //IVector m_scratch;
        IVector m_cf_map;

};

// specialization for host
template <AMGX_VecPrecision t_vecPrec, AMGX_MatPrecision t_matPrec, AMGX_IndPrecision t_indPrec>
class Energymin_AMG_Level< TemplateConfig<AMGX_host, t_vecPrec, t_matPrec, t_indPrec> > : public Energymin_AMG_Level_Base< TemplateConfig<AMGX_host, t_vecPrec, t_matPrec, t_indPrec> >
{
        typedef TemplateConfig<AMGX_host, t_vecPrec, t_matPrec, t_indPrec> TConfig_h;
        typedef Energymin_AMG_Level_Base<TConfig_h> Base;

        typedef AMG<t_vecPrec, t_matPrec, t_indPrec> AMG_Class;
        typedef typename TConfig_h::IndPrec IndexType;
        typedef typename TConfig_h::MatPrec ValueType;
        typedef Vector<TConfig_h> VVector;
        typedef typename MemorySpaceMap<AMGX_host>::Type host_memory;
        typedef typename Base::IVector IVector;
        typedef typename Matrix<TConfig_h>::IVector IVector_h;


    public:
        Energymin_AMG_Level(AMG_Class *amg) : Energymin_AMG_Level_Base<TConfig_h>(amg) {}
        void computeAOperator_1x1();
        void computeAOperator_1x1_distributed();
};

// specialization for device
template <AMGX_VecPrecision t_vecPrec, AMGX_MatPrecision t_matPrec, AMGX_IndPrecision t_indPrec>
class Energymin_AMG_Level< TemplateConfig<AMGX_device, t_vecPrec, t_matPrec, t_indPrec> > : public Energymin_AMG_Level_Base< TemplateConfig<AMGX_device, t_vecPrec, t_matPrec, t_indPrec> >
{
        typedef TemplateConfig<AMGX_device, t_vecPrec, t_matPrec, t_indPrec> TConfig_d;
        typedef TemplateConfig<AMGX_host, t_vecPrec, t_matPrec, t_indPrec> TConfig_h;
        typedef Energymin_AMG_Level_Base<TConfig_d> Base;

        typedef AMG<t_vecPrec, t_matPrec, t_indPrec> AMG_Class;
        typedef typename TConfig_d::IndPrec IndexType;
        typedef typename TConfig_d::MatPrec ValueType;
        typedef Vector<TConfig_d> VVector;
        typedef typename MemorySpaceMap<AMGX_device>::Type device_memory;
        typedef typename Base::IVector IVector;
        typedef typename Matrix<TConfig_h>::IVector IVector_h;
        typedef typename Matrix<TConfig_d>::IVector IVector_d;


        typedef typename TConfig_d::template setVecPrec<AMGX_vecInt64>::Type i64vec_value_type;
        typedef typename TConfig_h::template setVecPrec<AMGX_vecInt64>::Type i64vec_value_type_h;
        typedef typename TConfig_d::template setVecPrec<AMGX_vecInt64>::Type i64vec_value_type_d;

        typedef Vector<i64vec_value_type> I64Vector;
        typedef Vector<i64vec_value_type_h> I64Vector_h;
        typedef Vector<i64vec_value_type_d> I64Vector_d;

    public:
        Energymin_AMG_Level(AMG_Class *amg) : Energymin_AMG_Level_Base<TConfig_d>(amg) {}
        void computeAOperator_1x1();
        void computeAOperator_1x1_distributed();
};

} // namespace energymin

template<class T_Config>
class Energymin_AMG_LevelFactory : public AMG_LevelFactory<T_Config>
{
    public:
        static const AMGX_VecPrecision vecPrec = T_Config::vecPrec;
        static const AMGX_MatPrecision matPrec = T_Config::matPrec;
        static const AMGX_IndPrecision indPrec = T_Config::indPrec;
        typedef AMG<vecPrec, matPrec, indPrec> AMG_Class;
        AMG_Level<T_Config> *create(AMG_Class *amg, ThreadManager *tmng) { return new energymin::Energymin_AMG_Level<T_Config>(amg); }
};

} // namespace amgx

