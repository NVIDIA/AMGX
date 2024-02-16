// SPDX-FileCopyrightText: 2011 - 2024 NVIDIA CORPORATION. All Rights Reserved.
//
// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#include <energymin/interpolators/em_interpolator.h>
#include "cusolverDn.h"

#define TEST_EM

namespace amgx
{
namespace energymin
{

template <class T_Config> class  EM_Interpolator;

template <class T_Config>
class EM_InterpolatorBase : public Interpolator<T_Config>
{
        typedef T_Config TConfig;
        typedef typename TConfig::MatPrec ValueType;
        typedef typename TConfig::IndPrec IndexType;
        typedef Vector<typename TConfig::template setVecPrec<AMGX_vecInt>::Type> IntVector;
        typedef Vector<typename TConfig::template setVecPrec<AMGX_vecBool>::Type> BVector;
        typedef typename Matrix<T_Config>::MVector VVector;
        typedef typename Matrix<T_Config>::IVector IVector;

        typedef typename TConfig::template setMemSpace<AMGX_host>::Type TConfig_h;
        typedef typename TConfig_h::template setVecPrec<AMGX_vecInt>::Type ivec_value_type_h;
        typedef Vector<ivec_value_type_h> IVector_h;

    public:
        void generateInterpolationMatrix( const Matrix<T_Config> &A, const IntVector &cf_map,
                                          Matrix<T_Config> &P, void *amg);

        EM_InterpolatorBase() {};
        virtual ~EM_InterpolatorBase() {};

    protected:
        virtual void generateInterpolationMatrix_1x1( const Matrix<T_Config> &A, const IntVector &cf_map,
                Matrix<T_Config> &P, void *amg) = 0;
};


// specialization for host
template <AMGX_VecPrecision t_vecPrec, AMGX_MatPrecision t_matPrec, AMGX_IndPrecision t_indPrec>
class EM_Interpolator< TemplateConfig<AMGX_host, t_vecPrec, t_matPrec, t_indPrec> > :
    public EM_InterpolatorBase< TemplateConfig<AMGX_host, t_vecPrec, t_matPrec, t_indPrec> >
{
        typedef TemplateConfig<AMGX_host, t_vecPrec, t_matPrec, t_indPrec> TConfig_h;
        typedef typename TConfig_h::MatPrec ValueType;
        typedef typename TConfig_h::IndPrec IndexType;
        typedef Vector<typename TConfig_h::template setVecPrec<AMGX_vecInt>::Type> IntVector;
        typedef Vector<typename TConfig_h::template setVecPrec<AMGX_vecBool>::Type> BVector;
        typedef typename Matrix<TConfig_h>::MVector VVector;
        typedef Matrix<TConfig_h> Matrix_h;
        typedef typename Matrix_h::IVector IVector;

        typedef typename TConfig_h::template setVecPrec<AMGX_vecInt>::Type ivec_value_type_h;
        typedef Vector<ivec_value_type_h> IVector_h;

    private:
        void generateInterpolationMatrix_1x1( const Matrix_h &A, const IntVector &cf_map,
                                              Matrix_h &P, void *amg)
        {
            FatalError("Energymin InterpolationMatrix is not implemented on host", AMGX_ERR_NOT_IMPLEMENTED);
        }
};


// specialization for device
template <AMGX_VecPrecision t_vecPrec, AMGX_MatPrecision t_matPrec, AMGX_IndPrecision t_indPrec>
class EM_Interpolator< TemplateConfig<AMGX_device, t_vecPrec, t_matPrec, t_indPrec> >:
    public EM_InterpolatorBase< TemplateConfig<AMGX_device, t_vecPrec, t_matPrec, t_indPrec> >
{
        typedef TemplateConfig<AMGX_device, t_vecPrec, t_matPrec, t_indPrec> TConfig_d;
        typedef typename TConfig_d::MatPrec ValueType;
        typedef typename TConfig_d::IndPrec IndexType;
        typedef Vector<typename TConfig_d::template setVecPrec<AMGX_vecInt>::Type> IntVector;
        typedef Vector<typename TConfig_d::template setVecPrec<AMGX_vecBool>::Type> BVector;
        typedef typename Matrix<TConfig_d>::MVector VVector;
        typedef Matrix<TConfig_d> Matrix_d;
        typedef Vector<TConfig_d> Vector_d;
        typedef typename Matrix_d::IVector IVector;

        typedef typename TConfig_d::template setMemSpace<AMGX_host>::Type TConfig_h;
        typedef typename TConfig_h::template setVecPrec<AMGX_vecInt>::Type ivec_value_type_h;
        typedef Vector<ivec_value_type_h> IVector_h;

        typedef typename TConfig_h::template setVecPrec<(AMGX_VecPrecision)AMGX_GET_MODE_VAL(AMGX_MatPrecision, TConfig_h::mode)>::Type hmvec_value_type;
        typedef Vector<hmvec_value_type> VVector_h;


    public:
        EM_Interpolator();
        virtual ~EM_Interpolator();

    private:
        cusolverDnHandle_t m_cuds_handle;
        ValueType *m_dense_Aijs;    // store dense submatrices of A
        ValueType *m_dense_invAijs; // store dense inverses of submatrices Aij
        int *m_ipiv;                // device pointer for pivot sequence from getrf()
        ValueType *m_cuds_wspace;   // device pointer for workspace needed by getrf()
        int *m_cuds_info;           // host pointer for debug info from getrf() and getrs()
        //ValueType m_Mae_norm;

        void generateInterpolationMatrix_1x1( const Matrix_d &A, const IntVector &cf_map,
                                              Matrix_d &P, void *amg );

        void computePsparsity(const Matrix_d &A, const IVector &cf_map, const IVector &coarse_idx,
                              Matrix_d &P, IVector &PnnzPerCol, BVector &Ma_nzDiagRows);

        void computeAijSubmatrices( const Matrix_d &A, const int numCoarse, const Matrix_d &P,
                                    ValueType *dense_Aijs, ValueType *dense_invAijs,
                                    const IntVector &AijOffsets, int *ipiv,
                                    cusolverDnHandle_t &cuds_handle, int *cuds_info = 0 );

        void computeMa( Matrix_d &Ma, const int AnumRows, const int numCoarse, const Matrix_d &P,
                        const ValueType *dense_invAijs, const IntVector &AijOffsets, const BVector &Ma_nzDiagRows,
                        const bool perturb_Ma_diag = 1, const ValueType perturb_mag = 1.0e-8 );

        void solveMa_e( Matrix_d &Ma, const int AnumRows, Vector<TConfig_d> &v_x );

        void computePvalues(const int AnumRows, const int numCoarse, Matrix_d &P, const Vector_d &v_x,
                            const ValueType *dense_Aijs, const IntVector &AijOffsets, const int *ipiv,
                            cusolverDnHandle_t &cuds_handle, int *cuds_info = 0);

        void computePvalues(const int AnumRows, const int numCoarse, Matrix_d &P, const Vector_d &v_x,
                            const ValueType *dense_invAijs, const IntVector &AijOffsets);
};

template<class T_Config>
class EM_InterpolatorFactory : public InterpolatorFactory<T_Config>
{
    public:
        Interpolator<T_Config> *create(AMG_Config &cfg, const std::string &cfg_scope)
        { return new EM_Interpolator<T_Config>; }
};

} // namespace energymin
} // namespace amgx
