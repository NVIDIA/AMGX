// SPDX-FileCopyrightText: 2011 - 2025 NVIDIA CORPORATION. All Rights Reserved.
//
// SPDX-License-Identifier: BSD-3-Clause

#include <blas.h>
#include <basic_types.h>

#ifdef _WIN32
#pragma warning (push)
#pragma warning (disable : 4244 4267 4521)
#endif

#ifdef _WIN32
#pragma warning (pop)
#endif

#include <vector.h>
#include <matrix.h>
#include <complex>
#include <thrust/inner_product.h>
#include <thrust_wrapper.h>
#include <amgx_cublas.h>
#ifdef AMGX_USE_LAPACK
#include "mkl.h"
#endif

#include "amgx_types/util.h"
#include "amgx_types/math.h"

namespace amgx
{

// conjugate<T> computes the complex conjugate of a number f(a + b * i) -> a - b * i
template <typename T>
struct conjugate
{
    __host__ __device__
    T operator()(T x)
    {
        return types::util<T>::conjugate(x);
    }
};

// square<T> computes the square of a number f(x) -> x*conj(x)
template <typename T>
struct norm_squared
{
    __host__ __device__
    typename types::PODTypes<T>::type operator()(T x)
    {
        return types::util<T>::abs(x * conjugate<T>()(x));
    }
};
template <typename T>
struct SCAL
{
    T alpha;

    SCAL(T _alpha)
        : alpha(_alpha) {}

    template <typename T2>
    __host__ __device__
    void operator()(T2 &x)
    {
        x = x * alpha;
    }
};


template <typename T>
struct AXPY
{
    T alpha;

    AXPY(T _alpha)
        : alpha(_alpha) {}

    template <typename Tuple>
    __host__ __device__
    void operator()(Tuple t)
    {
        amgx::thrust::get<1>(t) = amgx::thrust::get<0>(t) * alpha +
                            amgx::thrust::get<1>(t);
    }
};

template <typename T1, typename T2>
struct AXPBY
{
    T1 alpha;
    T2 beta;

    AXPBY(T1 _alpha, T2 _beta)
        : alpha(_alpha), beta(_beta) {}

    template <typename Tuple>
    __host__ __device__
    void operator()(Tuple t)
    {
        amgx::thrust::get<2>(t) = amgx::thrust::get<0>(t) * alpha +
                            amgx::thrust::get<1>(t) * beta;
    }
};

template <typename T1, typename T2, typename T3>
struct AXPBYPCZ
{
    T1 alpha;
    T2 beta;
    T3 gamma;

    AXPBYPCZ(T1 _alpha, T2 _beta, T3 _gamma)
        : alpha(_alpha), beta(_beta), gamma(_gamma) {}

    template <typename Tuple>
    __host__ __device__
    void operator()(Tuple t)
    {
        amgx::thrust::get<3>(t) = amgx::thrust::get<0>(t) * alpha +
                            amgx::thrust::get<1>(t) * beta +
                            amgx::thrust::get<2>(t) * gamma;
    }
};

// absolute<T> computes the absolute value of a number f(x) -> |x|
template <typename T>
struct absolute
{
    __host__ __device__
    typename types::PODTypes<T>::type operator()(T x)
    {
        return types::util<T>::abs(x);
    }
};

// maximum<T> returns the largest of two numbers
template <typename T>
struct maximum
{
    __host__ __device__
    T operator()(T x, T y)
    {
        return amgx::thrust::maximum<T>()(x, y);
    }
};

template <typename ForwardIterator1,
          typename ForwardIterator2,
          typename ScalarType>
void thrust_axpy(ForwardIterator1 first1,
                 ForwardIterator1 last1,
                 ForwardIterator2 first2,
                 ScalarType alpha)
{
    size_t N = last1 - first1;
    amgx::thrust::for_each(amgx::thrust::make_zip_iterator(amgx::thrust::make_tuple(first1, first2)),
                     amgx::thrust::make_zip_iterator(amgx::thrust::make_tuple(first1, first2)) + N,
                     AXPY<ScalarType>(alpha));
}

template <int MemSpace,
          typename InputIterator1,
          typename InputIterator2,
          typename OutputIterator,
          typename ScalarType1,
          typename ScalarType2>
void thrust_axpby(InputIterator1 first1,
                  InputIterator1 last1,
                  InputIterator2 first2,
                  OutputIterator output,
                  ScalarType1 alpha,
                  ScalarType2 beta)
{
    size_t N = last1 - first1;
    thrust_wrapper::for_each<MemSpace>(amgx::thrust::make_zip_iterator(amgx::thrust::make_tuple(first1, first2, output)),
                     amgx::thrust::make_zip_iterator(amgx::thrust::make_tuple(first1, first2, output)) + N,
                     AXPBY<ScalarType1, ScalarType2>(alpha, beta));
}

template <typename InputIterator1,
          typename InputIterator2,
          typename InputIterator3,
          typename OutputIterator,
          typename ScalarType1,
          typename ScalarType2,
          typename ScalarType3>
void thrust_axpbypcz(InputIterator1 first1,
                     InputIterator1 last1,
                     InputIterator2 first2,
                     InputIterator3 first3,
                     OutputIterator output,
                     ScalarType1 alpha,
                     ScalarType2 beta,
                     ScalarType3 gamma)
{
    size_t N = last1 - first1;
    amgx::thrust::for_each(amgx::thrust::make_zip_iterator(amgx::thrust::make_tuple(first1, first2, first3, output)),
                     amgx::thrust::make_zip_iterator(amgx::thrust::make_tuple(first1, first2, first3, output)) + N,
                     AXPBYPCZ<ScalarType1, ScalarType2, ScalarType3>(alpha, beta, gamma));
}

template <typename InputIterator1,
          typename InputIterator2>
typename amgx::thrust::iterator_traits<InputIterator1>::value_type
thrust_dotc(InputIterator1 first1,
            InputIterator1 last1,
            InputIterator2 first2)
{
    typedef typename amgx::thrust::iterator_traits<InputIterator1>::value_type OutputType;
    return amgx::thrust::inner_product(amgx::thrust::make_transform_iterator(first1, conjugate<OutputType>()),
                                 amgx::thrust::make_transform_iterator(last1,  conjugate<OutputType>()),
                                 first2,
                                 types::util<OutputType>::get_zero());
}

template <int MemSpace, typename InputIterator>
typename types::PODTypes<typename amgx::thrust::iterator_traits<InputIterator>::value_type>::type
thrust_nrm1(InputIterator first,
            InputIterator last)
{
    typedef typename amgx::thrust::iterator_traits<InputIterator>::value_type ValueType;
    typedef typename types::PODTypes<typename amgx::thrust::iterator_traits<InputIterator>::value_type>::type OutType;
    absolute<ValueType> unary_op;
    amgx::thrust::plus<OutType> binary_op;
    OutType init = types::util<OutType>::get_zero(); // OutType is always scalar, we could just typecast
    OutType result = thrust_wrapper::transform_reduce<MemSpace>(first, last, unary_op, init, binary_op);
    cudaCheckError();
    return result;
}

template <int MemSpace, typename InputIterator>
typename types::PODTypes<typename amgx::thrust::iterator_traits<InputIterator>::value_type>::type
thrust_nrm2(InputIterator first,
            InputIterator last)
{
    typedef typename amgx::thrust::iterator_traits<InputIterator>::value_type ValueType;
    typedef typename types::PODTypes<ValueType>::type OutType;
    norm_squared<ValueType> unary_op;
    amgx::thrust::plus<OutType> binary_op;
    OutType init = types::util<OutType>::get_zero(); // OutType is always scalar, we could just typecast
    OutType result = thrust_wrapper::transform_reduce<MemSpace>(first, last, unary_op, init, binary_op);
    cudaCheckError();
    return std::sqrt( result );
}

template <int MemSpace, typename InputIterator>
typename types::PODTypes<typename amgx::thrust::iterator_traits<InputIterator>::value_type>::type
thrust_nrmmax(InputIterator first,
              InputIterator last)
{
    typedef typename amgx::thrust::iterator_traits<InputIterator>::value_type ValueType;
    typedef typename types::PODTypes<typename amgx::thrust::iterator_traits<InputIterator>::value_type>::type OutType;
    absolute<ValueType>  unary_op;
    maximum<OutType>   binary_op;
    OutType init = types::util<OutType>::get_zero(); // OutType is always scalar, we could just typecast
    OutType result = thrust_wrapper::transform_reduce<MemSpace>(first, last, unary_op, init, binary_op);
    cudaCheckError();
    return result;
}

template <typename ForwardIterator,
          typename ScalarType>
void thrust_scal(ForwardIterator first,
                 ForwardIterator last,
                 ScalarType alpha)
{
    amgx::thrust::for_each(first,
                     last,
                     SCAL<ScalarType>(alpha));
    cudaCheckError();
}

template<class Vector, class Scalar>
void axpbypcz(const Vector &x, const Vector &y, const Vector &z, Vector &out, Scalar a, Scalar b, Scalar c, int offset, int size)
{
    if (size == -1) { size = x.size() / x.get_block_size(); }

#ifndef NDEBUG

    if (x.get_block_dimx() == -1) { FatalError("x block dims not set", AMGX_ERR_NOT_IMPLEMENTED); }

    if (y.get_block_dimx() == -1) { FatalError("y block dims not set", AMGX_ERR_NOT_IMPLEMENTED); }

    if (z.get_block_dimx() == -1) { FatalError("z block dims not set", AMGX_ERR_NOT_IMPLEMENTED); }

    if (out.get_block_dimx() == -1) { FatalError("out block dims not set", AMGX_ERR_NOT_IMPLEMENTED); }

#endif
    thrust_axpbypcz(x.begin() + offset * x.get_block_size(),
                    x.begin() + (offset + size) * x.get_block_size(),
                    y.begin() + offset * x.get_block_size(),
                    z.begin() + offset * x.get_block_size(),
                    out.begin() + offset * x.get_block_size(), a, b, c);
    out.dirtybit = 1;
    cudaCheckError();
}

//out=a*x+b*y
template<class Vector, class Scalar>
void axpby(const Vector &x, const Vector &y, Vector &out, Scalar a, Scalar b, int offset, int size)
{
    if (size == -1) { size = x.size() / x.get_block_size(); }

#ifndef NDEBUG

    if (x.get_block_dimx() == -1) { FatalError("x block dims not set", AMGX_ERR_NOT_IMPLEMENTED); }

    if (y.get_block_dimx() == -1) { FatalError("y block dims not set", AMGX_ERR_NOT_IMPLEMENTED); }

    if (out.get_block_dimx() == -1) { FatalError("out block dims not set", AMGX_ERR_NOT_IMPLEMENTED); }

#endif
    thrust_axpby<Vector::TConfig::memSpace>(x.begin() + offset * x.get_block_size(),
                 x.begin() + (offset + size) * x.get_block_size(),
                 y.begin() + offset * x.get_block_size(),
                 out.begin() + offset * x.get_block_size(),
                 a, b);
    out.dirtybit = 1;
    cudaCheckError();
}

/// gemv_Ext for host and device begins
namespace
{
#ifdef AMGX_USE_LAPACK

/*extern "C"
void cblas_dgemv(const CBLAS_ORDER order, const CBLAS_TRANSPOSE TransA, const int M,
                 const int N, const double alpha, const double *A, const int lda, const double *X,
                 const int incX, const double beta, double *Y, const int incY);

extern "C"
void cblas_sgemv(const CBLAS_ORDER order, const CBLAS_TRANSPOSE TransA, const int M,
                 const int N, const float alpha, const float *A, const int lda, const float *X,
                 const int incX, const float beta, float *Y, const int incY);

extern "C"
void cblas_cgemv(const CBLAS_ORDER order, const CBLAS_TRANSPOSE TransA, const int M,
                 const int N, const cuComplex alpha, const cuComplex *A, const int lda, const cuComplex *X,
                 const int incX, const cuComplex beta, cuComplex *Y, const int incY);

extern "C"
void cblas_zgemv(const CBLAS_ORDER order, const CBLAS_TRANSPOSE TransA, const int M,
                 const int N, const cuDoubleComplex alpha, const cuDoubleComplex *A, const int lda, const cuDoubleComplex *X,
                 const int incX, const cuDoubleComplex beta, cuDoubleComplex *Y, const int incY);*/

void mkl_gemv_dispatch(const CBLAS_ORDER order, const CBLAS_TRANSPOSE TransA, const int M,
                       const int N, const double alpha, const double *A, const int lda, const double *X,
                       const int incX, const double beta, double *Y, const int incY)
{
    cblas_dgemv(order, TransA, M, N, alpha, A, lda, X, incX, beta, Y, incY);
}

void mkl_gemv_dispatch(const CBLAS_ORDER order, const CBLAS_TRANSPOSE TransA, const int M,
                       const int N, const float alpha, const float *A, const int lda, const float *X,
                       const int incX, const float beta, float *Y, const int incY)
{
    cblas_sgemv(order, TransA, M, N, alpha, A, lda, X, incX, beta, Y, incY);
}

void mkl_gemv_dispatch(const CBLAS_ORDER order, const CBLAS_TRANSPOSE TransA, const int M,
                       const int N, const cuComplex alpha, const cuComplex *A, const int lda, const cuComplex *X,
                       const int incX, const cuComplex beta, cuComplex *Y, const int incY)
{
    cblas_cgemv(order, TransA, M, N, 
                reinterpret_cast<const void*>(&alpha), 
                reinterpret_cast<const void*>(A), 
                lda, 
                reinterpret_cast<const void*>(X), 
                incX, 
                reinterpret_cast<const void*>(&beta), 
                reinterpret_cast<void*>(Y), 
                incY);
}

void mkl_gemv_dispatch(const CBLAS_ORDER order, const CBLAS_TRANSPOSE TransA, const int M,
                       const int N, const cuDoubleComplex alpha, const cuDoubleComplex *A, const int lda, const cuDoubleComplex *X,
                       const int incX, const cuDoubleComplex beta, cuDoubleComplex *Y, const int incY)
{
    cblas_zgemv(order, TransA, M, N, 
                reinterpret_cast<const void*>(&alpha), 
                reinterpret_cast<const void*>(A), 
                lda, 
                reinterpret_cast<const void*>(X), 
                incX, 
                reinterpret_cast<const void*>(&beta), 
                reinterpret_cast<void*>(Y), 
                incY);
}

template <typename T>
void mkl_gemv(CBLAS_ORDER order, CBLAS_TRANSPOSE TransA, int M,
              int N, T *alpha, const T *A, int lda, const T *X,
              int incX, T *beta, T *Y, int incY)
{
    mkl_gemv_dispatch(order, TransA, M, N, *alpha, A, lda, X, incX, *beta, Y, incY);
}
#endif

}

template <class Vector, class Scalar>
void gemv_extnd(bool trans, const Vector &A, const Vector &x, Vector &y, int m, int n,
                Scalar alpha, Scalar beta, int incx, int incy, int lda,
                int offsetA, int offsetx, int offsety)
{
    typedef typename Vector::TConfig TConfig;
    typedef typename Vector::value_type value_type;

    if (TConfig::memSpace == AMGX_host)
    {
#ifdef AMGX_USE_LAPACK
        CBLAS_TRANSPOSE TransA = trans ? CblasTrans : CblasNoTrans;
        CBLAS_ORDER order = CblasColMajor;
        mkl_gemv(order, TransA, m, n, &alpha, A.raw() + offsetA, lda, x.raw() + offsetx, incx, &beta, y.raw() + offsety, incy); // add mkl call here
#endif
    }
    else
    {
        Cublas::gemv_ext(trans, m, n, &alpha, A.raw(), lda, x.raw(), incx, &beta, y.raw(), incy, offsetx, offsety, offsetA);
    }
}
/// gemv_Ext for host and device ends

/// trsv for host and device begins
namespace
{
#ifdef AMGX_USE_LAPACK
/*extern "C" void cblas_strsv (const CBLAS_ORDER order, const CBLAS_UPLO Uplo, const CBLAS_TRANSPOSE TransA, const CBLAS_DIAG Diag,
                             const int N, const float *A, const int lda, float *X, const int incX);

extern "C" void cblas_dtrsv (const CBLAS_ORDER order, const CBLAS_UPLO Uplo, const CBLAS_TRANSPOSE TransA, const CBLAS_DIAG Diag,
                             const int N, const double *A, const int lda, double *X, const int incX);

extern "C" void cblas_ctrsv (const CBLAS_ORDER order, const CBLAS_UPLO Uplo, const CBLAS_TRANSPOSE TransA, const CBLAS_DIAG Diag,
                             const int N, const cuComplex *A, const int lda, cuComplex *X, const int incX);

extern "C" void cblas_ztrsv (const CBLAS_ORDER order, const CBLAS_UPLO Uplo, const CBLAS_TRANSPOSE TransA, const CBLAS_DIAG Diag,
                             const int N, const cuDoubleComplex *A, const int lda, cuDoubleComplex *X, const int incX);*/

void mkl_trsv_dispatch(const CBLAS_ORDER order, const CBLAS_UPLO Uplo, const CBLAS_TRANSPOSE TransA, const CBLAS_DIAG Diag,
                       const int N, const float *A, const int lda, float *X, const int incX)
{   
    cblas_strsv (order, Uplo, TransA, Diag, N, A, lda, X, incX);
}

void mkl_trsv_dispatch(const CBLAS_ORDER order, const CBLAS_UPLO Uplo, const CBLAS_TRANSPOSE TransA, const CBLAS_DIAG Diag,
                       const int N, const double *A, const int lda, double *X, const int incX)
{   
    cblas_dtrsv (order, Uplo, TransA, Diag, N, A, lda, X, incX);
}

void mkl_trsv_dispatch(const CBLAS_ORDER order, const CBLAS_UPLO Uplo, const CBLAS_TRANSPOSE TransA, const CBLAS_DIAG Diag,
                       const int N, const cuComplex *A, const int lda, cuComplex *X, const int incX)
{   
    cblas_ctrsv (order, Uplo, TransA, Diag, N, A, lda, X, incX);
}

void mkl_trsv_dispatch(const CBLAS_ORDER order, const CBLAS_UPLO Uplo, const CBLAS_TRANSPOSE TransA, const CBLAS_DIAG Diag,
                       const int N, const cuDoubleComplex *A, const int lda, cuDoubleComplex *X, const int incX)
{   
    cblas_ztrsv (order, Uplo, TransA, Diag, N, A, lda, X, incX);
}

template <typename T>
void mkl_trsv(CBLAS_ORDER order, CBLAS_UPLO Uplo, CBLAS_TRANSPOSE TransA, const CBLAS_DIAG Diag,
              int N, const T *A, int lda, T *X, int incX)
{   
    mkl_trsv_dispatch(order, Uplo, TransA, Diag, N, A, lda, X, incX);
}
#endif
}

template <class Vector>
void trsv_extnd(bool trans, const Vector &A, int lda, Vector &x, int n, int incx, int offsetA)
{
    typedef typename Vector::TConfig TConfig;

//   typedef typename Vector::value_type value_type;
    if (TConfig::memSpace == AMGX_host)
    {
#ifdef AMGX_USE_LAPACK
        CBLAS_TRANSPOSE TransA = trans ? CblasTrans : CblasNoTrans;
        CBLAS_ORDER order = CblasColMajor;
        CBLAS_UPLO  Uplo = CblasLower;
        CBLAS_DIAG Diag = CblasNonUnit;
        //mkl_gemv(order, TransA, m, n, &alpha, A.raw()+offsetA, lda, x.raw()+offsetx, incx, &beta, y.raw()+offsety, incy);// add mkl call here
        mkl_trsv(order, Uplo, TransA, Diag, n, A.raw() + offsetA, lda, x.raw(), incx);
#endif
    }
    else
    {
        Cublas::trsv_v2(CUBLAS_FILL_MODE_LOWER, CUBLAS_OP_N, CUBLAS_DIAG_NON_UNIT, n, A.raw(), lda, x.raw(), incx, offsetA);
        cudaCheckError();
    }
}

/// trsv for host and device ends
//y=y+a*x
template<class Vector, class Scalar>
void axpy(const Vector &x, Vector &y, Scalar a, int offset, int size)
{
    if (size == -1) { size = x.size() / x.get_block_size(); }

#ifndef NDEBUG

    if (x.get_block_dimx() == -1) { FatalError("x block dims not set", AMGX_ERR_NOT_IMPLEMENTED); }

    if (y.get_block_dimx() == -1) { FatalError("y block dims not set", AMGX_ERR_NOT_IMPLEMENTED); }

#endif
    typedef typename Vector::TConfig TConfig;
    typedef typename Vector::value_type value_type;
    int x_first = offset * x.get_block_size();
    int x_last = (offset + size) * x.get_block_size();
    int y_first = offset * y.get_block_size();

    if (TConfig::memSpace == AMGX_host)
        thrust_axpy(x.begin() + x_first,
                    x.begin() + x_last, y.begin() + y_first, a);
    else
    {
        size = x_last - x_first;
        const value_type *x_raw = x.raw() + x_first;
        value_type *y_raw = y.raw() + y_first;
        Cublas::axpy(size, a, x_raw, 1, y_raw, 1);
    }

    cudaCheckError();
    y.dirtybit = 1;
}


template<class Vector, class Scalar>
void axpy(Vector &x, Vector &y, Scalar a, int offsetx, int offsety, int size)
{
    if (size == -1) { size = x.size() / x.get_block_size(); }

#ifndef NDEBUG

    if (x.get_block_dimx() == -1) { FatalError("x block dims not set", AMGX_ERR_NOT_IMPLEMENTED); }

    if (y.get_block_dimx() == -1) { FatalError("y block dims not set", AMGX_ERR_NOT_IMPLEMENTED); }

#endif
    typedef typename Vector::TConfig TConfig;
    typedef typename Vector::value_type value_type;
    int x_first = offsetx * x.get_block_size();
    int x_last = (offsetx + size) * x.get_block_size();
    int y_first = offsety * y.get_block_size();

    if (TConfig::memSpace == AMGX_host)
        thrust_axpy(x.begin() + x_first,
                    x.begin() + x_last, y.begin() + y_first, a);
    else
    {
        size = x_last - x_first;
        const value_type *x_raw = x.raw() + x_first;
        value_type *y_raw = y.raw() + y_first;
        Cublas::axpy(size, a, x_raw, 1, y_raw, 1);
        cudaCheckError();
    }

    y.dirtybit = 1;
}

//x=a*x
template<class Vector, class Scalar>
void scal(Vector &x, Scalar a, int offset, int size)
{
    if (size == -1) { size = x.size() / x.get_block_size(); }

#ifndef NDEBUG

    if (x.get_block_dimx() == -1) { FatalError("x block dims not set", AMGX_ERR_NOT_IMPLEMENTED); }

#endif
    typedef typename Vector::TConfig TConfig;
    typedef typename Vector::value_type value_type;
    int x_first = offset * x.get_block_size();
    int x_last = (offset + size) * x.get_block_size();

    if (TConfig::memSpace == AMGX_host)
    {
        thrust_scal(x.begin() + x_first, x.begin() + x_last, a);
    }
    else
    {
        size = x_last - x_first;
        value_type *x_raw = x.raw() + x_first;
        Cublas::scal(size, a, x_raw, 1);
        cudaCheckError();
    }

    x.dirtybit = 1;
}


//r=A*x-b
template <class Matrix, class Vector>
void axmb(Matrix &A, Vector &x, Vector &b, Vector &r, int offset, int size)
{
    if (size == -1) { size = x.size() / x.get_block_size(); }

#ifndef NDEBUG

    if (x.get_block_dimx() == -1) { FatalError("x block dims not set", AMGX_ERR_NOT_IMPLEMENTED); }

    if (b.get_block_dimx() == -1) { FatalError("b block dims not set", AMGX_ERR_NOT_IMPLEMENTED); }

    if (r.get_block_dimx() == -1) { FatalError("r block dims not set", AMGX_ERR_NOT_IMPLEMENTED); }

#endif
    typedef typename Matrix::value_type ValueType;
    A.apply(x, r);
    thrust_axpby<Vector::TConfig::memSpace>(
                 r.begin() + offset * x.get_block_size(),
                 r.begin() + (offset + size) * x.get_block_size(),
                 b.begin() + offset * x.get_block_size(),
                 r.begin() + offset * x.get_block_size(),
                 types::util<ValueType>::get_minus_one(), types::util<ValueType>::get_one());
}

template <class Vector>
typename Vector::value_type dotc(const Vector &a, const Vector &b, int offset, int size)
{
    if (size == -1) { size = a.size() / a.get_block_size(); }

#ifndef NDEBUG

    if (a.get_block_dimx() == -1) { FatalError("a block dims not set", AMGX_ERR_NOT_IMPLEMENTED); }

    if (b.get_block_dimx() == -1) { FatalError("b block dims not set", AMGX_ERR_NOT_IMPLEMENTED); }

#endif
    typedef typename Vector::TConfig TConfig;
    typedef typename Vector::value_type value_type;
    int a_first = offset * a.get_block_size();
    int a_last = (offset + size) * a.get_block_size();
    int b_first = offset * a.get_block_size();

    if (TConfig::memSpace == AMGX_host)
    {
        return thrust_dotc(a.begin() + a_first, a.begin() + a_last, b.begin() + b_first);
    }
    else
    {
        size = a_last - a_first;
        const value_type *a_raw = a.raw() + a_first;
        const value_type *b_raw = b.raw() + b_first;
        value_type result;
        Cublas::dot(size, a_raw, 1, b_raw, 1, &result);
        cudaCheckError();
        return result;
    }
}

template <class Vector>
typename Vector::value_type dotc(const Vector &a, const Vector &b, int offseta, int offsetb, int size)
{
    if (size == -1) { size = a.size() / a.get_block_size(); }

#ifndef NDEBUG

    if (a.get_block_dimx() == -1) { FatalError("a block dims not set", AMGX_ERR_NOT_IMPLEMENTED); }

    if (b.get_block_dimx() == -1) { FatalError("b block dims not set", AMGX_ERR_NOT_IMPLEMENTED); }

#endif
    typedef typename Vector::TConfig TConfig;
    typedef typename Vector::value_type value_type;
    int a_first = offseta * a.get_block_size();
    int a_last = (offseta + size) * a.get_block_size();
    int b_first = offsetb * a.get_block_size();

    if (TConfig::memSpace == AMGX_host)
    {
        return thrust_dotc(a.begin() + a_first, a.begin() + a_last, b.begin() + b_first);
    }
    else
    {
        size = a_last - a_first;
        const value_type *a_raw = a.raw() + a_first;
        const value_type *b_raw = b.raw() + b_first;
        value_type result;
        Cublas::dot(size, a_raw, 1, b_raw, 1, &result);
        cudaCheckError();
        return result;
    }
}

// This function is similar to dotc above but hides necessary MPI calls to the user.
template <class Matrix, class Vector>
typename Vector::value_type dot(const Matrix &A, const Vector &x, const Vector &y)
{
    typedef typename Vector::value_type value_type;
    int offset, size;
    A.getOffsetAndSizeForView(A.getViewExterior(), &offset, &size);
    value_type reduce = dotc(x, y, offset, size);

    if (A.is_matrix_distributed())
    {
        A.getManager()->global_reduce_sum(&reduce);
    }

    cudaCheckError();
    return reduce;
}

template <class Matrix, class Vector>
typename Vector::value_type dot(const Matrix &A, const Vector &x, const Vector &y, int offsetx, int offsety)
{
    typedef typename Vector::value_type value_type;
    int offset, size;
    A.getOffsetAndSizeForView(A.getViewExterior(), &offset, &size);
    value_type reduce = dotc(x, y, offsetx, offsety, size);

    if (A.is_matrix_distributed())
    {
        A.getManager()->global_reduce_sum(&reduce);
    }

    cudaCheckError();
    return reduce;
}

//b=a
template <class Vector>
void copy(const Vector &a, Vector &b, int offset, int size)
{
    if (size == -1) { size = a.size() / a.get_block_size(); }

#ifndef NDEBUG

    if (a.get_block_dimx() == -1) { FatalError("a block dims not set", AMGX_ERR_NOT_IMPLEMENTED); }

    if (b.get_block_dimx() == -1) { FatalError("b block dims not set", AMGX_ERR_NOT_IMPLEMENTED); }

#endif
    typedef typename Vector::TConfig TConfig;
    typedef typename Vector::value_type value_type;
    int a_first = offset * a.get_block_size();
    int a_last = (offset + size) * a.get_block_size();
    int b_first = offset * a.get_block_size();

    if (TConfig::memSpace == AMGX_host)
    {
        amgx::thrust::copy(a.begin() + a_first, a.begin() + a_last, b.begin() + b_first);
    }
    else
    {
        size = a_last - a_first;
        const value_type *a_raw = a.raw() + a_first;
        value_type *b_raw = b.raw() + b_first;
        Cublas::copy(size, a_raw, 1, b_raw, 1);
        cudaCheckError();
    }

    b.dirtybit = 1;
}

template <class Vector>
void copy_ext(Vector &a, Vector &b, int offseta,  int offsetb, int size)
{
    if (size == -1) { size = a.size() / a.get_block_size(); }

#ifndef NDEBUG

    if (a.get_block_dimx() == -1) { FatalError("a block dims not set", AMGX_ERR_NOT_IMPLEMENTED); }

    if (b.get_block_dimx() == -1) { FatalError("b block dims not set", AMGX_ERR_NOT_IMPLEMENTED); }

#endif
    typedef typename Vector::TConfig TConfig;
    typedef typename Vector::value_type value_type;
    int a_first = offseta * a.get_block_size();
    int a_last = (offseta + size) * a.get_block_size();
    int b_first = offsetb * a.get_block_size();

    if (TConfig::memSpace == AMGX_host)
    {
        amgx::thrust::copy(a.begin() + a_first, a.begin() + a_last, b.begin() + b_first);
    }
    else
    {
        size = a_last - a_first;
        const value_type *a_raw = a.raw() + a_first;
        value_type *b_raw = b.raw() + b_first;
        Cublas::copy(size, a_raw, 1, b_raw, 1);
        cudaCheckError();
    }

    b.dirtybit = 1;
}

//x_i=val
template <class Vector>
void fill(Vector &x, typename Vector::value_type val, int offset, int size)
{
    if (size == -1) { size = x.size() / x.get_block_size(); }

#ifndef NDEBUG

    if (x.get_block_dimx() == -1) { FatalError("x block dims not set", AMGX_ERR_NOT_IMPLEMENTED); }

#endif
    thrust_wrapper::fill<Vector::TConfig::memSpace>(x.begin() + offset * x.get_block_size(),
                 x.begin() + (offset + size) * x.get_block_size(), val);
    x.dirtybit = 1;
    cudaCheckError();
}

template <class Vector>
typename types::PODTypes<typename Vector::value_type>::type
nrm1(const Vector &x, int offset, int size)
{
    if (size == -1) { size = x.size() / x.get_block_size(); }

#ifndef NDEBUG

    if (x.get_block_dimx() == -1) { FatalError("x block dims not set", AMGX_ERR_NOT_IMPLEMENTED); }

#endif
    typename types::PODTypes<typename Vector::value_type>::type out =
        thrust_nrm1<Vector::TConfig::memSpace>(x.begin() + offset * x.get_block_size(),
                    x.begin() + (offset + size) * x.get_block_size());
    cudaCheckError();
    return out;
}

template <AMGX_VecPrecision t_vecPrec, AMGX_MatPrecision t_matPrec, AMGX_IndPrecision t_indPrec>
typename types::PODTypes<typename Vector<TemplateConfig<AMGX_host, t_vecPrec, t_matPrec, t_indPrec> >::value_type>::type
nrm1(const Vector<TemplateConfig<AMGX_host, t_vecPrec, t_matPrec, t_indPrec> > &x, int offset, int size)
{
    if (size == -1) { size = x.size() / x.get_block_size(); }

#ifndef NDEBUG

    if (x.get_block_dimx() == -1) { FatalError("x block dims not set", AMGX_ERR_NOT_IMPLEMENTED); }

#endif
    int bsize = x.get_block_size();
    typedef typename Vector<TemplateConfig<AMGX_host, t_vecPrec, t_matPrec, t_indPrec> >::value_type ValueType;
    typedef typename types::PODTypes<ValueType>::type OutType;
    OutType out = types::util<OutType>::get_zero();

    for (int i = 0; i < size * bsize; i++)
    {
        out += types::util<ValueType>::abs(x[offset * bsize + i]);
    }

    return out;
}

template <class Vector>
typename types::PODTypes<typename Vector::value_type>::type
nrm2(const Vector &x, int offset, int size)
{
    if (size == -1) { size = x.size() / x.get_block_size(); }

#ifndef NDEBUG

    if (x.get_block_dimx() == -1) { FatalError("x block dims not set", AMGX_ERR_NOT_IMPLEMENTED); }

#endif
    typedef typename Vector::TConfig TConfig;
    typedef typename Vector::value_type value_type;
    int x_first = offset * x.get_block_size();
    int x_last = (offset + size) * x.get_block_size();
    // We are not using CUBLAS for nrm2 since the implementation is slower.
    return thrust_nrm2<Vector::TConfig::memSpace>(x.begin() + x_first,
                       x.begin() + x_last);
}

template <AMGX_VecPrecision t_vecPrec, AMGX_MatPrecision t_matPrec, AMGX_IndPrecision t_indPrec>
typename types::PODTypes<typename Vector<TemplateConfig<AMGX_host, t_vecPrec, t_matPrec, t_indPrec> >::value_type>::type
nrm2(const Vector<TemplateConfig<AMGX_host, t_vecPrec, t_matPrec, t_indPrec> > &x, int offset, int size)
{
    if (size == -1) { size = x.size() / x.get_block_size(); }

#ifndef NDEBUG

    if (x.get_block_dimx() == -1) { FatalError("x block dims not set", AMGX_ERR_NOT_IMPLEMENTED); }

#endif
    int bsize = x.get_block_size();
    typedef typename Vector<TemplateConfig<AMGX_host, t_vecPrec, t_matPrec, t_indPrec> >::value_type ValueType;
    typedef typename types::PODTypes<ValueType>::type OutType;
    OutType out = types::util<OutType>::get_zero();

    for (int i = 0; i < size * bsize; i++)
    {
        out += types::util<ValueType>::abs(x[offset * bsize + i] * x[offset * bsize + i]);
    }

    return sqrt(out);
}


template <class Vector>
typename types::PODTypes<typename Vector::value_type>::type
nrmmax(const Vector &x, int offset, int size)
{
    if (size == -1) { size = x.size() / x.get_block_size(); }

#ifndef NDEBUG

    if (x.get_block_dimx() == -1) { FatalError("x block dims not set", AMGX_ERR_NOT_IMPLEMENTED); }

#endif
    typename types::PODTypes<typename Vector::value_type>::type out =
        thrust_nrmmax<Vector::TConfig::memSpace>(x.begin() + offset * x.get_block_size(),
                      x.begin() + (offset + size) * x.get_block_size());
    cudaCheckError();
    return out;
}



/****************************************
 * Explict instantiations
 ***************************************/

#define AMGX_CASE_LINE(CASE) template void axpbypcz(const Vector<TemplateMode<CASE>::Type> &x, const Vector<TemplateMode<CASE>::Type>& y, const Vector<TemplateMode<CASE>::Type>& z, Vector<TemplateMode<CASE>::Type> &out, Vector<TemplateMode<CASE>::Type>::value_type a, Vector<TemplateMode<CASE>::Type>::value_type b, Vector<TemplateMode<CASE>::Type>::value_type c, int, int);
AMGX_FORALL_BUILDS(AMGX_CASE_LINE)
AMGX_FORCOMPLEX_BUILDS(AMGX_CASE_LINE)
#undef AMGX_CASE_LINE

#define AMGX_CASE_LINE(CASE) template void axpby(const Vector<TemplateMode<CASE>::Type> &x, const Vector<TemplateMode<CASE>::Type>& y, Vector<TemplateMode<CASE>::Type> &out, double a, double b, int, int);
AMGX_FORALL_BUILDS(AMGX_CASE_LINE)
AMGX_FORCOMPLEX_BUILDS(AMGX_CASE_LINE)
#undef AMGX_CASE_LINE

#define AMGX_CASE_LINE(CASE) template void axpby(const Vector<TemplateMode<CASE>::Type> &x, const Vector<TemplateMode<CASE>::Type>& y, Vector<TemplateMode<CASE>::Type> &out, float a, float b, int, int);
AMGX_FORALL_BUILDS(AMGX_CASE_LINE)
AMGX_FORCOMPLEX_BUILDS(AMGX_CASE_LINE)
#undef AMGX_CASE_LINE

#define AMGX_CASE_LINE(CASE) template void axpby(const Vector<TemplateMode<CASE>::Type> &x, const Vector<TemplateMode<CASE>::Type>& y, Vector<TemplateMode<CASE>::Type> &out, cuDoubleComplex a, cuDoubleComplex b, int, int);
AMGX_FORCOMPLEX_BUILDS(AMGX_CASE_LINE)
#undef AMGX_CASE_LINE

#define AMGX_CASE_LINE(CASE) template void axpby(const Vector<TemplateMode<CASE>::Type> &x, const Vector<TemplateMode<CASE>::Type>& y, Vector<TemplateMode<CASE>::Type> &out, cuComplex a, cuComplex b, int, int);
AMGX_FORCOMPLEX_BUILDS(AMGX_CASE_LINE)
#undef AMGX_CASE_LINE

#define AMGX_CASE_LINE(CASE) template void axpy(const Vector<TemplateMode<CASE>::Type> &x, Vector<TemplateMode<CASE>::Type>& y, Vector<TemplateMode<CASE>::Type>::value_type a, int, int);
AMGX_FORALL_BUILDS(AMGX_CASE_LINE)
AMGX_FORCOMPLEX_BUILDS(AMGX_CASE_LINE)
#undef AMGX_CASE_LINE


#define AMGX_CASE_LINE(CASE) template void axpy(Vector<TemplateMode<CASE>::Type> &x, Vector<TemplateMode<CASE>::Type>& y, Vector<TemplateMode<CASE>::Type>::value_type a, int, int, int);
AMGX_FORALL_BUILDS(AMGX_CASE_LINE)
AMGX_FORCOMPLEX_BUILDS(AMGX_CASE_LINE)
#undef AMGX_CASE_LINE

#define AMGX_CASE_LINE(CASE) template void gemv_extnd(bool, const Vector<TemplateMode<CASE>::Type> &, const Vector<TemplateMode<CASE>::Type> &, Vector<TemplateMode<CASE>::Type> &, int, int, Vector<TemplateMode<CASE>::Type>::value_type,  Vector<TemplateMode<CASE>::Type>::value_type, int, int, int, int, int, int);
AMGX_FORALL_BUILDS(AMGX_CASE_LINE)
AMGX_FORCOMPLEX_BUILDS(AMGX_CASE_LINE)
#undef AMGX_CASE_LINE

#define AMGX_CASE_LINE(CASE) template void trsv_extnd(bool, const Vector<TemplateMode<CASE>::Type> &, int, Vector<TemplateMode<CASE>::Type> &, int, int, int);
AMGX_FORALL_BUILDS(AMGX_CASE_LINE)
AMGX_FORCOMPLEX_BUILDS(AMGX_CASE_LINE)
#undef AMGX_CASE_LINE

#define AMGX_CASE_LINE(CASE) template void scal(Vector<TemplateMode<CASE>::Type> &x, Vector<TemplateMode<CASE>::Type>::value_type a, int, int);
AMGX_FORALL_BUILDS(AMGX_CASE_LINE)
AMGX_FORCOMPLEX_BUILDS(AMGX_CASE_LINE)
#undef AMGX_CASE_LINE

#define AMGX_CASE_LINE(CASE) template void axmb(Matrix<TemplateMode<CASE>::Type>& A, Vector<TemplateMode<CASE>::Type>& x, Vector<TemplateMode<CASE>::Type>& b, Vector<TemplateMode<CASE>::Type>& r, int, int);
AMGX_FORALL_BUILDS(AMGX_CASE_LINE)
AMGX_FORCOMPLEX_BUILDS(AMGX_CASE_LINE)
#undef AMGX_CASE_LINE

#define AMGX_CASE_LINE(CASE) template void axmb(Operator<TemplateMode<CASE>::Type>& A, Vector<TemplateMode<CASE>::Type>& x, Vector<TemplateMode<CASE>::Type>& b, Vector<TemplateMode<CASE>::Type>& r, int, int);
AMGX_FORALL_BUILDS(AMGX_CASE_LINE)
AMGX_FORCOMPLEX_BUILDS(AMGX_CASE_LINE)
#undef AMGX_CASE_LINE

#define AMGX_CASE_LINE(CASE) template typename Vector<TemplateMode<CASE>::Type>::value_type dotc(const Vector<TemplateMode<CASE>::Type>& a, const Vector<TemplateMode<CASE>::Type> &b, int, int);
AMGX_FORALL_BUILDS(AMGX_CASE_LINE)
AMGX_FORCOMPLEX_BUILDS(AMGX_CASE_LINE)
#undef AMGX_CASE_LINE

#define AMGX_CASE_LINE(CASE) template typename Vector<TemplateMode<CASE>::Type>::value_type dotc(const Vector<TemplateMode<CASE>::Type>& a, const Vector<TemplateMode<CASE>::Type> &b, int, int, int);
AMGX_FORALL_BUILDS(AMGX_CASE_LINE)
AMGX_FORCOMPLEX_BUILDS(AMGX_CASE_LINE)
#undef AMGX_CASE_LINE

#define AMGX_CASE_LINE(CASE) template typename Vector<TemplateMode<CASE>::Type>::value_type dot(const Matrix<TemplateMode<CASE>::Type>& A,const Vector<TemplateMode<CASE>::Type>& x, const Vector<TemplateMode<CASE>::Type> &y);
AMGX_FORALL_BUILDS(AMGX_CASE_LINE)
AMGX_FORCOMPLEX_BUILDS(AMGX_CASE_LINE)
#undef AMGX_CASE_LINE

#define AMGX_CASE_LINE(CASE) template typename Vector<TemplateMode<CASE>::Type>::value_type dot(const Operator<TemplateMode<CASE>::Type>& A,const Vector<TemplateMode<CASE>::Type>& x, const Vector<TemplateMode<CASE>::Type> &y);
AMGX_FORALL_BUILDS(AMGX_CASE_LINE)
AMGX_FORCOMPLEX_BUILDS(AMGX_CASE_LINE)
#undef AMGX_CASE_LINE

#define AMGX_CASE_LINE(CASE) template typename Vector<TemplateMode<CASE>::Type>::value_type dot(const Matrix<TemplateMode<CASE>::Type>& A,const Vector<TemplateMode<CASE>::Type>& x, const Vector<TemplateMode<CASE>::Type> &y, int, int);
AMGX_FORALL_BUILDS(AMGX_CASE_LINE)
AMGX_FORCOMPLEX_BUILDS(AMGX_CASE_LINE)
#undef AMGX_CASE_LINE

#define AMGX_CASE_LINE(CASE) template typename Vector<TemplateMode<CASE>::Type>::value_type dot(const Operator<TemplateMode<CASE>::Type>& A,const Vector<TemplateMode<CASE>::Type>& x, const Vector<TemplateMode<CASE>::Type> &y, int, int);
AMGX_FORALL_BUILDS(AMGX_CASE_LINE)
AMGX_FORCOMPLEX_BUILDS(AMGX_CASE_LINE)
#undef AMGX_CASE_LINE

#define AMGX_CASE_LINE(CASE) template void copy(const Vector<TemplateMode<CASE>::Type> & a, Vector<TemplateMode<CASE>::Type>  &b, int, int);
AMGX_FORALL_BUILDS(AMGX_CASE_LINE)
AMGX_FORCOMPLEX_BUILDS(AMGX_CASE_LINE)
#undef AMGX_CASE_LINE


#define AMGX_CASE_LINE(CASE) template void copy_ext(Vector<TemplateMode<CASE>::Type> & a, Vector<TemplateMode<CASE>::Type>  &b, int, int, int);
AMGX_FORALL_BUILDS(AMGX_CASE_LINE)
AMGX_FORCOMPLEX_BUILDS(AMGX_CASE_LINE)
#undef AMGX_CASE_LINE

#define AMGX_CASE_LINE(CASE) template void fill(Vector<TemplateMode<CASE>::Type>& x, Vector<TemplateMode<CASE>::Type>::value_type val, int, int);
AMGX_FORALL_BUILDS(AMGX_CASE_LINE)
AMGX_FORCOMPLEX_BUILDS(AMGX_CASE_LINE)
#undef AMGX_CASE_LINE

#define AMGX_CASE_LINE(CASE) template typename types::PODTypes<typename Vector<TemplateMode<CASE>::Type>::value_type>::type nrm1(const Vector<TemplateMode<CASE>::Type>& x, int, int);
AMGX_FORALL_BUILDS(AMGX_CASE_LINE)
AMGX_FORCOMPLEX_BUILDS(AMGX_CASE_LINE)
#undef AMGX_CASE_LINE

#define AMGX_CASE_LINE(CASE) template typename types::PODTypes<typename Vector<TemplateMode<CASE>::Type>::value_type>::type nrm2(const Vector<TemplateMode<CASE>::Type>& x, int, int);
AMGX_FORALL_BUILDS(AMGX_CASE_LINE)
AMGX_FORCOMPLEX_BUILDS(AMGX_CASE_LINE)
#undef AMGX_CASE_LINE

#define AMGX_CASE_LINE(CASE) template typename types::PODTypes<typename Vector<TemplateMode<CASE>::Type>::value_type>::type nrmmax(const Vector<TemplateMode<CASE>::Type>& x, int, int);
AMGX_FORALL_BUILDS(AMGX_CASE_LINE)
AMGX_FORCOMPLEX_BUILDS(AMGX_CASE_LINE)
#undef AMGX_CASE_LINE

} // namespace amgx
