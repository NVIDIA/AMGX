// SPDX-FileCopyrightText: 2011 - 2025 NVIDIA CORPORATION. All Rights Reserved.
//
// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#include <classical/interpolators/interpolator.h>
#include <set>
#include <vector>
#include <amg.h>

namespace amgx
{

namespace distance2
{

template< int CTA_SIZE, int WARP_SIZE >
__global__ __launch_bounds__( CTA_SIZE )
void
estimate_c_hat_size_kernel( const int A_num_rows,
                            const int *A_rows,
                            const int *A_cols,
                            const int *cf_map,
                            const bool *s_con,
                            int *C_hat_offsets );

template< int NUM_THREADS_PER_ROW, int CTA_SIZE, int WARP_SIZE >
__global__ __launch_bounds__( CTA_SIZE )
void
estimate_c_hat_size_kernel( const int A_num_rows,
                            const int *A_rows,
                            const int *A_cols,
                            const int *cf_map,
                            const bool *s_con,
                            int *C_hat_offsets );

template< int CTA_SIZE, int SMEM_SIZE, int WARP_SIZE >
__global__ __launch_bounds__( CTA_SIZE )
void
compute_c_hat_kernel( int A_num_rows,
                      const int *__restrict A_rows,
                      const int *__restrict A_cols,
                      const int *__restrict cf_map,
                      const bool *__restrict s_con,
                      const int *__restrict C_hat_start,
                      int *__restrict C_hat_end,
                      int *__restrict C_hat,
                      int *__restrict C_hat_pos,
                      int gmem_size,
                      int *g_keys,
                      int *wk_work_queue,
                      int *wk_status );

template< int NUM_THREADS_PER_ROW, int CTA_SIZE, int SMEM_SIZE, int WARP_SIZE >
__global__ __launch_bounds__( CTA_SIZE )
void
compute_c_hat_kernel( int A_num_rows,
                      const int *__restrict A_rows,
                      const int *__restrict A_cols,
                      const int *__restrict cf_map,
                      const bool *__restrict s_con,
                      const int *__restrict C_hat_start,
                      int *__restrict C_hat_end,
                      int *__restrict C_hat,
                      int *__restrict C_hat_pos,
                      int gmem_size,
                      int *g_keys,
                      int *wk_work_queue,
                      int *wk_status );

} // namespace distance2

template <class T_Config> class  Distance2_Interpolator;

template <class T_Config>
class Distance2_InterpolatorBase : public Interpolator<T_Config>
{
        typedef T_Config TConfig;
        typedef typename TConfig::MatPrec ValueType;
        typedef typename TConfig::IndPrec IndexType;
        typedef Vector<typename TConfig::template setVecPrec<AMGX_vecInt>::Type> IntVector;
        typedef Vector<typename TConfig::template setVecPrec<AMGX_vecBool>::Type> BVector;
        typedef typename Matrix<T_Config>::MVector VVector;
        typedef typename Matrix<T_Config>::IVector IVector;

        typedef typename TConfig::template setVecPrec<AMGX_vecInt64>::Type i64vec_value_type;

        typedef Vector<i64vec_value_type> I64Vector;

    public:
        Distance2_InterpolatorBase(AMG_Config &, const std::string &) {}
        void generateInterpolationMatrix( Matrix<T_Config> &A,
                                          IntVector &cf_map,
                                          BVector &s_con,
                                          IntVector &scratch,
                                          Matrix<T_Config> &P);
    protected:
        virtual void generateInterpolationMatrix_1x1( Matrix<T_Config> &A,
                IntVector &cf_map,
                BVector &s_con,
                IntVector &scratch,
                Matrix<T_Config> &P) = 0;

};

template <AMGX_VecPrecision t_vecPrec, AMGX_MatPrecision t_matPrec, AMGX_IndPrecision t_indPrec>
class Distance2_Interpolator< TemplateConfig<AMGX_host, t_vecPrec, t_matPrec, t_indPrec> > : public Distance2_InterpolatorBase< TemplateConfig<AMGX_host, t_vecPrec, t_matPrec, t_indPrec> >
{
        typedef TemplateConfig<AMGX_host, t_vecPrec, t_matPrec, t_indPrec> TConfig_h;
        typedef Distance2_InterpolatorBase<TConfig_h> Base;

        typedef typename TConfig_h::MatPrec ValueType;
        typedef typename TConfig_h::IndPrec IndexType;
        typedef Vector<typename TConfig_h::template setVecPrec<AMGX_vecInt>::Type> IntVector;
        typedef Vector<typename TConfig_h::template setVecPrec<AMGX_vecBool>::Type> BVector;
        typedef typename Matrix<TConfig_h>::MVector VVector;
        typedef Matrix<TConfig_h> Matrix_h;
        typedef typename Matrix_h::IVector IVector;

        typedef typename TConfig_h::template setVecPrec<AMGX_vecInt64>::Type i64vec_value_type_h;

        typedef Vector<i64vec_value_type_h> I64Vector_h;



    public:
        Distance2_Interpolator(AMG_Config &cfg, const std::string &cfg_scope) : Base(cfg, cfg_scope) {}

    private:
        void generateInterpolationMatrix_1x1( Matrix_h &A,
                                              IntVector &cf_map,
                                              BVector &s_con,
                                              IntVector &scratch,
                                              Matrix_h &P);

        void calculateD( const Matrix_h &A, IVector &cf_map, BVector &s_con,
                         std::vector<int> *C_hat, VVector &diag, VVector &D,
                         std::set<int> *weak_lists, VVector &innerSum,
                         IVector &innerSumOffsets );
        void generateInnerSum( Matrix_h &A, const IntVector &cf_map,
                               const BVector &s_con,
                               std::vector<int> *C_hat, VVector &diag,
                               VVector &innerSum,
                               IntVector &innerSumOffsets );
};

// specialization for device
template <AMGX_VecPrecision t_vecPrec, AMGX_MatPrecision t_matPrec, AMGX_IndPrecision t_indPrec>
class Distance2_Interpolator< TemplateConfig<AMGX_device, t_vecPrec, t_matPrec, t_indPrec> >: public Distance2_InterpolatorBase< TemplateConfig<AMGX_device, t_vecPrec, t_matPrec, t_indPrec> >
{
        typedef TemplateConfig<AMGX_device, t_vecPrec, t_matPrec, t_indPrec> TConfig_d;
        typedef Distance2_InterpolatorBase<TConfig_d> Base;

        typedef typename TConfig_d::MatPrec ValueType;
        typedef typename TConfig_d::IndPrec IndexType;
        typedef Vector<typename TConfig_d::template setVecPrec<AMGX_vecInt>::Type> IntVector;
        typedef Vector<typename TConfig_d::template setVecPrec<AMGX_vecBool>::Type> BVector;
        typedef typename Matrix<TConfig_d>::MVector VVector;
        typedef Matrix<TConfig_d> Matrix_d;
        typedef typename Matrix_d::IVector IVector;

        typedef typename TConfig_d::template setVecPrec<AMGX_vecInt64>::Type i64vec_value_type_d;

        typedef Vector<i64vec_value_type_d> I64Vector_d;

    public:
        Distance2_Interpolator(AMG_Config &cfg, const std::string &cfg_scope);
        ~Distance2_Interpolator();

    private:
        void generateInterpolationMatrix_1x1( Matrix_d &A,
                                              IntVector &cf_map,
                                              BVector &s_con,
                                              IntVector &scratch,
                                              Matrix_d &P);

};

template<class T_Config>
class Distance2_InterpolatorFactory : public InterpolatorFactory<T_Config>
{
    public:
        Interpolator<T_Config> *create( AMG_Config &cfg, const std::string &cfg_scope ) { return new Distance2_Interpolator<T_Config>( cfg, cfg_scope ); }
};

} // namespace amgx
