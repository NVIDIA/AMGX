// SPDX-FileCopyrightText: 2011 - 2024 NVIDIA CORPORATION. All Rights Reserved.
//
// SPDX-License-Identifier: BSD-3-Clause

#include <generic_spmv_csr.h>
#include <basic_types.h>
#include <matrix.h>
#include <vector.h>

#pragma once


namespace amgx
{

struct default_state
{
};

template <typename T>
struct row_sum
{
    T val;

    __host__ __device__
    row_sum() : val(0) {};

    __host__ __device__
    row_sum(T v) : val(v) {};

    template <typename valType1, typename valType2, typename indType, typename state>
    __host__ __device__ __forceinline__
    static row_sum binary_op(const valType1 &a, const valType2 b, const indType row, const indType col,
                             const state &st)
    {
        return row_sum(a);
    }

    __host__ __device__ __forceinline__
    static row_sum reduce_op(const row_sum &a, const row_sum &b)
    {
        return row_sum(a.val + b.val);
    }

    __host__ __device__ __forceinline__
    static row_sum post_op(const row_sum &a)
    {
        return row_sum(a.val);
    }

    template <typename outType>
    __host__ __device__ __forceinline__
    void retrieve(outType &out) const
    {
        out = val;
    }
};

template <typename T>
struct abs_row_sum
{
    T val;

    __host__ __device__
    abs_row_sum() : val(0) {};

    __host__ __device__
    abs_row_sum(T v) : val(v) {};

    template <typename valType1, typename valType2, typename indType, typename state>
    __host__ __device__ __forceinline__
    static abs_row_sum binary_op(const valType1 &a, const valType2 b, const indType row, const indType col,
                                 const state &st)
    {
        return abs_row_sum(fabs(a));
    }

    __host__ __device__ __forceinline__
    static abs_row_sum reduce_op(const abs_row_sum &a, const abs_row_sum &b)
    {
        return abs_row_sum(a.val + b.val);
    }

    __host__ __device__ __forceinline__
    static abs_row_sum post_op(const abs_row_sum &a)
    {
        return abs_row_sum(a.val);
    }

    template <typename outType>
    __host__ __device__ __forceinline__
    void retrieve(outType &out) const
    {
        out = val;
    }
};

template <typename T>
struct weighted_row_sum
{
    T val, diag;

    __host__ __device__
    weighted_row_sum() : val(0), diag(0) {};

    __host__ __device__
    weighted_row_sum(T v) : val(v), diag(0) {};

    __host__ __device__
    weighted_row_sum(T v, T d) : val(v), diag(d) {};

    template <typename valType1, typename valType2, typename indType, typename state>
    __host__ __device__ __forceinline__
    static weighted_row_sum binary_op(const valType1 &a, const valType2 b, const indType row, const indType col,
                                      const state &st)
    {
        double d = 0;

        if (row == col) { d = a; }

        return weighted_row_sum(a, d);
    }

    __host__ __device__ __forceinline__
    static weighted_row_sum reduce_op(const weighted_row_sum &a, const weighted_row_sum &b)
    {
        double d = 0;

        if (a.diag != 0) { d = a.diag; }

        if (b.diag != 0) { d = b.diag; }

        return weighted_row_sum(a.val + b.val, d);
    }

    __host__ __device__ __forceinline__
    static weighted_row_sum post_op(const weighted_row_sum &a)
    {
        return weighted_row_sum(fabs(a.val / a.diag), a.diag);
    }

    template <typename outType>
    __host__ __device__ __forceinline__
    void retrieve(outType &out) const
    {
        out = val;
    }
};

template <typename T>
struct max_coef
{
    T coef;
    __host__ __device__
    max_coef() : coef(0) {};

    __host__ __device__
    max_coef(const T v) : coef(v) {};

    template <typename valType1, typename valType2, typename indType, typename state>
    __host__ __device__ __forceinline__
    static max_coef binary_op(const valType1 &a, const valType2 b, const indType row, const indType col,
                              const state &st)
    {
        const double temp = fabs(a);
        return max_coef(temp);
    }

    __host__ __device__ __forceinline__
    static max_coef reduce_op(const max_coef &a, const max_coef &b)
    {
        return max_coef(max(a.coef, b.coef));
    }

    __host__ __device__ __forceinline__
    static max_coef post_op(const max_coef &a)
    {
        return max_coef(a.coef);
    }

    template <typename outType>
    __host__ __device__ __forceinline__
    void retrieve(outType &out) const
    {
        out = coef;
    }
};

template <typename T>
struct max_coef_sum
{
    T coef, sum;
    __host__ __device__
    max_coef_sum() : coef(0), sum(0) {};
    __host__ __device__
    max_coef_sum(T v) : coef(v), sum(0) {};
    __host__ __device__
    max_coef_sum(T c, T s) : coef(c), sum(s) {};

    template <typename valType1, typename valType2, typename indType, typename state>
    __host__ __device__ __forceinline__
    static max_coef_sum binary_op(const valType1 &a, const valType2 b, const indType row, const indType col,
                                  const state &st)
    {
        const double temp = fabs(a);
        return max_coef_sum(temp, a);
    }

    __host__ __device__ __forceinline__
    static max_coef_sum reduce_op(const max_coef_sum &a, const max_coef_sum &b)
    {
        return max_coef_sum(max(a.coef, b.coef), a.sum + b.sum);
    }

    __host__ __device__ __forceinline__
    static max_coef_sum post_op(const max_coef_sum &a)
    {
        return max_coef_sum(a.coef, a.sum);
    }

    template <typename T1, typename T2>
    __host__ __device__ __forceinline__
    void retrieve(T1 &out1, T2 &out2) const
    {
        out1 = coef;
        out2 = sum;
    }
};

template <typename I, typename T>
struct count_trunc_elements
{
    I count;
    T threshold, new_sum;

    __host__ __device__
    count_trunc_elements() : count(0), threshold(0), new_sum(0) {};

    __host__ __device__
    count_trunc_elements(T t) : count(0), threshold(t), new_sum(0) {};

    __host__ __device__
    count_trunc_elements(I c, T t, T s) : count(c), threshold(t), new_sum(s) {};

    template <typename valType1, typename valType2, typename indType, typename state>
    __host__ __device__ __forceinline__
    static count_trunc_elements binary_op(const valType1 &a, const valType2 x, const indType &row,
                                          const indType &col, const state &st)
    {
        if (fabs(a) >= x[row]*st.val) { return count_trunc_elements(1, x[row], a); }
        else { return count_trunc_elements(0, x[row], 0); }
    }

    __host__ __device__ __forceinline__
    static count_trunc_elements reduce_op(const count_trunc_elements &a,
                                          const count_trunc_elements &b)
    {
        return count_trunc_elements(a.count + b.count, a.threshold, a.new_sum + b.new_sum);
    }

    __host__ __device__ __forceinline__
    static count_trunc_elements post_op(const count_trunc_elements &a)
    {
        return count_trunc_elements(a.count, a.threshold, a.new_sum);
    }

    template <typename T1, typename T2>
    __host__ __device__ __forceinline__
    void retrieve(T1 &out1, T2 &out2) const
    {
        out1 = count;
        out2 = new_sum;
    }
};

template <AMGX_VecPrecision t_vecPrec, AMGX_VecPrecision t_vecPrec2, AMGX_MatPrecision t_matPrec, AMGX_IndPrecision t_indPrec>
void rowSum(const Matrix<TemplateConfig<AMGX_device, t_vecPrec, t_matPrec, t_indPrec> > &A,
            Vector<TemplateConfig<AMGX_device, t_vecPrec2, t_matPrec, t_indPrec> > &row_sums)
{
    typedef TemplateConfig<AMGX_device, t_vecPrec, t_matPrec, t_indPrec> TConfig_d;
    typedef typename Matrix<TConfig_d>::VVector VVector;
    typedef typename Matrix<TConfig_d>::MVector MVector;
    VVector x(A.get_num_rows());
    device_vector_alloc<row_sum<typename TConfig_d::VecPrec> > y(A.get_num_rows(), 0);
    default_state st;
    genericSpmvCSR(A, x, y, st);
    retrieveOneArgument(y, row_sums);
}

template <AMGX_VecPrecision t_vecPrec, AMGX_VecPrecision t_vecPrec2, AMGX_MatPrecision t_matPrec, AMGX_IndPrecision t_indPrec>
void rowSum(const Matrix<TemplateConfig<AMGX_host, t_vecPrec, t_matPrec, t_indPrec> > &A,
            Vector<TemplateConfig<AMGX_host, t_vecPrec2, t_matPrec, t_indPrec> > &row_sums)
{
    typedef TemplateConfig<AMGX_host, t_vecPrec, t_matPrec, t_indPrec> TConfig_h;
    typedef typename Matrix<TConfig_h>::VVector VVector;
    VVector x(A.get_num_rows());
    amgx::thrust::host_vector<row_sum<typename TConfig_h::VecPrec> > y(A.get_num_rows(), 0);
    default_state st;
    genericSpmvCSR(A, x, y, st);
    retrieveOneArgument(y, row_sums);
}

template <AMGX_VecPrecision t_vecPrec, AMGX_VecPrecision t_vecPrec2, AMGX_MatPrecision t_matPrec, AMGX_IndPrecision t_indPrec>
void absRowSum(const Matrix<TemplateConfig<AMGX_device, t_vecPrec, t_matPrec, t_indPrec> > &A,
               Vector<TemplateConfig<AMGX_device, t_vecPrec2, t_matPrec, t_indPrec> > &row_sums)
{
    typedef TemplateConfig<AMGX_device, t_vecPrec, t_matPrec, t_indPrec> TConfig_d;
    typedef typename Matrix<TConfig_d>::VVector VVector;
    typedef typename Matrix<TConfig_d>::MVector MVector;
    VVector x(A.get_num_rows());
    device_vector_alloc<abs_row_sum<typename TConfig_d::VecPrec> > y(A.get_num_rows(), 0);
    default_state st;
    genericSpmvCSR(A, x, y, st);
    retrieveOneArgument(y, row_sums);
}

template <AMGX_VecPrecision t_vecPrec, AMGX_VecPrecision t_vecPrec2, AMGX_MatPrecision t_matPrec, AMGX_IndPrecision t_indPrec>
void absRowSum(const Matrix<TemplateConfig<AMGX_host, t_vecPrec, t_matPrec, t_indPrec> > &A,
               Vector<TemplateConfig<AMGX_host, t_vecPrec2, t_matPrec, t_indPrec> > &row_sums)
{
    typedef TemplateConfig<AMGX_host, t_vecPrec, t_matPrec, t_indPrec> TConfig_h;
    typedef typename Matrix<TConfig_h>::VVector VVector;
    VVector x(A.get_num_rows());
    amgx::thrust::host_vector<abs_row_sum<typename TConfig_h::VecPrec> > y(A.get_num_rows(), 0);
    default_state st;
    genericSpmvCSR(A, x, y, st);
    retrieveOneArgument(y, row_sums);
}

template <AMGX_VecPrecision t_vecPrec, AMGX_VecPrecision t_vecPrec2, AMGX_MatPrecision t_matPrec, AMGX_IndPrecision t_indPrec>
void weightedRowSum(const Matrix<TemplateConfig<AMGX_device, t_vecPrec, t_matPrec, t_indPrec> > &A,
                    Vector<TemplateConfig<AMGX_device, t_vecPrec2, t_matPrec, t_indPrec> > &row_sums)
{
    typedef TemplateConfig<AMGX_device, t_vecPrec, t_matPrec, t_indPrec> TConfig_d;
    typedef typename Matrix<TConfig_d>::VVector VVector;
    VVector x(A.get_num_rows());
    device_vector_alloc<weighted_row_sum<typename TConfig_d::VecPrec> > y(A.get_num_rows(), 0);
    default_state st;
    genericSpmvCSR(A, x, y, st);
    retrieveOneArgument(y, row_sums);
}

template <AMGX_VecPrecision t_vecPrec, AMGX_VecPrecision t_vecPrec2, AMGX_MatPrecision t_matPrec, AMGX_IndPrecision t_indPrec>
void weightedRowSum(const Matrix<TemplateConfig<AMGX_host, t_vecPrec, t_matPrec, t_indPrec> > &A,
                    Vector<TemplateConfig<AMGX_host, t_vecPrec2, t_matPrec, t_indPrec> > &row_sums)
{
    typedef TemplateConfig<AMGX_host, t_vecPrec, t_matPrec, t_indPrec> TConfig_h;
    typedef typename Matrix<TConfig_h>::VVector VVector;
    VVector x(A.get_num_rows());
    amgx::thrust::host_vector<weighted_row_sum<typename TConfig_h::VecPrec> > y(A.get_num_rows(), 0);
    default_state st;
    genericSpmvCSR(A, x, y, st);
    retrieveOneArgument(y, row_sums);
}

template <AMGX_VecPrecision t_vecPrec, AMGX_VecPrecision t_vecPrec2, AMGX_MatPrecision t_matPrec, AMGX_IndPrecision t_indPrec>
void maxCoef(const Matrix<TemplateConfig<AMGX_device, t_vecPrec, t_matPrec, t_indPrec> > &A,
             Vector<TemplateConfig<AMGX_device, t_vecPrec2, t_matPrec, t_indPrec> > &max_coefs)
{
    typedef TemplateConfig<AMGX_device, t_vecPrec, t_matPrec, t_indPrec> TConfig_d;
    typedef typename Matrix<TConfig_d>::VVector VVector;
    VVector x(A.get_num_rows());
    device_vector_alloc<max_coef<typename TConfig_d::VecPrec> > y(A.get_num_rows());
    default_state st;
    genericSpmvCSR(A, x, y, st);
    retrieveOneArgument(y, max_coefs);
}

template <AMGX_VecPrecision t_vecPrec, AMGX_VecPrecision t_vecPrec2, AMGX_MatPrecision t_matPrec, AMGX_IndPrecision t_indPrec>
void maxCoef(const Matrix<TemplateConfig<AMGX_host, t_vecPrec, t_matPrec, t_indPrec> > &A,
             Vector<TemplateConfig<AMGX_host, t_vecPrec2, t_matPrec, t_indPrec> > &max_coefs)
{
    typedef TemplateConfig<AMGX_host, t_vecPrec, t_matPrec, t_indPrec> TConfig_h;
    typedef typename Matrix<TConfig_h>::VVector VVector;
    VVector x(A.get_num_rows());
    amgx::thrust::host_vector<max_coef<typename TConfig_h::VecPrec> > y(A.get_num_rows());
    default_state st;
    genericSpmvCSR(A, x, y, st);
    retrieveOneArgument(y, max_coefs);
}

template <AMGX_VecPrecision t_vecPrec, AMGX_VecPrecision t_vecPrec2, AMGX_MatPrecision t_matPrec, AMGX_IndPrecision t_indPrec>
void maxCoefAndSum(const Matrix<TemplateConfig<AMGX_device, t_vecPrec, t_matPrec, t_indPrec> > &A,
                   Vector<TemplateConfig<AMGX_device, t_vecPrec2, t_matPrec, t_indPrec> > &max_coefs,
                   Vector<TemplateConfig<AMGX_device, t_vecPrec2, t_matPrec, t_indPrec> > &row_sums)
{
    typedef TemplateConfig<AMGX_device, t_vecPrec, t_matPrec, t_indPrec> TConfig_d;
    typedef typename Matrix<TConfig_d>::VVector VVector;
    VVector x(A.get_num_rows());
    device_vector_alloc<max_coef_sum<typename TConfig_d::VecPrec> > y(A.get_num_rows());
    default_state st;
    genericSpmvCSR(A, x, y, st);
    retrieveTwoArguments(y, max_coefs, row_sums);
}

template <AMGX_VecPrecision t_vecPrec, AMGX_VecPrecision t_vecPrec2, AMGX_MatPrecision t_matPrec, AMGX_IndPrecision t_indPrec>
void maxCoefAndSum(const Matrix<TemplateConfig<AMGX_host, t_vecPrec, t_matPrec, t_indPrec> > &A,
                   Vector<TemplateConfig<AMGX_host, t_vecPrec2, t_matPrec, t_indPrec> > &max_coefs,
                   Vector<TemplateConfig<AMGX_host, t_vecPrec2, t_matPrec, t_indPrec> > &row_sums)
{
    typedef TemplateConfig<AMGX_host, t_vecPrec, t_matPrec, t_indPrec> TConfig_h;
    typedef typename Matrix<TConfig_h>::VVector VVector;
    VVector x(A.get_num_rows());
    amgx::thrust::host_vector<max_coef_sum<typename TConfig_h::VecPrec> > y(A.get_num_rows());
    default_state st;
    genericSpmvCSR(A, x, y, st);
    retrieveTwoArguments(y, max_coefs, row_sums);
}

struct trunc_state
{
    double val;
};

template <AMGX_VecPrecision t_vecPrec, AMGX_VecPrecision t_vecPrecIn, AMGX_VecPrecision t_vecPrecOut, AMGX_MatPrecision t_matPrec,
          AMGX_IndPrecision t_indPrec>
void countTruncElements(const Matrix<TemplateConfig<AMGX_device, t_vecPrec, t_matPrec, t_indPrec>  > &A,
                        const double                                                                    &val,
                        const Vector<TemplateConfig<AMGX_device, t_vecPrecIn, t_matPrec, t_indPrec>  > &trunc,
                        Vector<TemplateConfig<AMGX_device, t_vecPrecOut, t_matPrec, t_indPrec> > &count,
                        Vector<TemplateConfig<AMGX_device, t_vecPrecIn, t_matPrec, t_indPrec>  > &new_row_sums)
{
    typedef TemplateConfig<AMGX_device, t_vecPrec, t_matPrec, t_indPrec> TConfig_d;
    device_vector_alloc<count_trunc_elements<typename TConfig_d::IndPrec, typename TConfig_d::VecPrec> > y(A.get_num_rows());
    trunc_state st;
    st.val = val;
    genericSpmvCSR(A, trunc, y, st);
    retrieveTwoArguments(y, count, new_row_sums);
}

template <AMGX_VecPrecision t_vecPrec, AMGX_VecPrecision t_vecPrecIn, AMGX_VecPrecision t_vecPrecOut, AMGX_MatPrecision t_matPrec,
          AMGX_IndPrecision t_indPrec>
void countTruncElements(const Matrix<TemplateConfig<AMGX_host, t_vecPrec, t_matPrec, t_indPrec>  > &A,
                        const double                                                                  &val,
                        const Vector<TemplateConfig<AMGX_host, t_vecPrecIn, t_matPrec, t_indPrec>  > &trunc,
                        Vector<TemplateConfig<AMGX_host, t_vecPrecOut, t_matPrec, t_indPrec> > &count)
{
    typedef TemplateConfig<AMGX_host, t_vecPrec, t_matPrec, t_indPrec> TConfig_h;
    device_vector_alloc<count_trunc_elements<typename TConfig_h::IndPrec, typename TConfig_h::VecPrec> > y(A.get_num_rows());
    trunc_state st;
    st.val = val;
    genericSpmvCSR(A, trunc, y, st);
    retrieveOneArgument(y, count);
}

}
