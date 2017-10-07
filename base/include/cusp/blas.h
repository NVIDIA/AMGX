/*
 *  Copyright 2008-2009 NVIDIA Corporation
 *
 *  Licensed under the Apache License, Version 2.0 (the "License");
 *  you may not use this file except in compliance with the License.
 *  You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 *  Unless required by applicable law or agreed to in writing, software
 *  distributed under the License is distributed on an "AS IS" BASIS,
 *  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  See the License for the specific language governing permissions and
 *  limitations under the License.
 */

/*! \file blas.h
 *  \brief BLAS-like functions
 */


#pragma once

#include <cusp/detail/config.h>

#include <cusp/complex.h>

#include <thrust/iterator/iterator_traits.h>

namespace cusp
{
namespace blas
{

/*! \addtogroup algorithms Algorithms
 */

/*! \addtogroup blas BLAS
 *  \ingroup algorithms
 *  \{
 */

template <typename ForwardIterator1,
          typename ForwardIterator2,
          typename ScalarType>
CUSP_DEPRECATED
void axpy(ForwardIterator1 first1,
          ForwardIterator1 last1,
          ForwardIterator2 first2,
          ScalarType alpha);

/*! \p axpy : scaled vector addition (y = alpha * x + y)
 */
template <typename Array1,
          typename Array2,
          typename ScalarType>
void axpy(const Array1& x,
                Array2& y,
          ScalarType alpha);

/*! \p axpy : scaled vector addition (y = alpha * x + y)
 */
template <typename Array1,
          typename Array2,
          typename ScalarType>
void axpy(const Array1& x,
          const Array2& y,
          ScalarType alpha);


template <typename InputIterator1,
          typename InputIterator2,
          typename OutputIterator,
          typename ScalarType>
CUSP_DEPRECATED
void axpby(InputIterator1 first1,
           InputIterator1 last1,
           InputIterator2 first2,
           OutputIterator output,
           ScalarType alpha,
           ScalarType beta);

/*! \p axpby : linear combination of two vectors (output = alpha * x + beta * y)
 */
template <typename Array1,
          typename Array2,
          typename Array3,
          typename ScalarType1,
          typename ScalarType2>
void axpby(const Array1& x,
           const Array2& y,
                 Array3& output,
           ScalarType1 alpha,
           ScalarType2 beta);

/*! \p axpby : linear combination of two vectors (output = alpha * x + beta * y)
 */
template <typename Array1,
          typename Array2,
          typename Array3,
          typename ScalarType1,
          typename ScalarType2>
void axpby(const Array1& x,
           const Array2& y,
           const Array3& output,
           ScalarType1 alpha,
           ScalarType2 beta);


template <typename InputIterator1,
          typename InputIterator2,
          typename InputIterator3,
          typename OutputIterator,
          typename ScalarType>
CUSP_DEPRECATED
void axpbypcz(InputIterator1 first1,
              InputIterator1 last1,
              InputIterator2 first2,
              InputIterator3 first3,
              OutputIterator output,
              ScalarType alpha,
              ScalarType beta,
              ScalarType gamma);

/*! \p axpbycz : linear combination of three vectors (output = alpha * x + beta * y + gamma * z)
 */
template <typename Array1,
          typename Array2,
          typename Array3,
          typename Array4,
          typename ScalarType1,
          typename ScalarType2,
          typename ScalarType3>
void axpbypcz(const Array1& x,
              const Array2& y,
              const Array3& z,
                    Array4& output,
              ScalarType1 alpha,
              ScalarType2 beta,
              ScalarType3 gamma);

/*! \p axpbycz : linear combination of three vectors (output = alpha * x + beta * y + gamma * z)
 */
template <typename Array1,
          typename Array2,
          typename Array3,
          typename Array4,
          typename ScalarType1,
          typename ScalarType2,
          typename ScalarType3>
void axpbypcz(const Array1& x,
              const Array2& y,
              const Array3& z,
              const Array4& output,
              ScalarType1 alpha,
              ScalarType2 beta,
              ScalarType3 gamma);


template <typename InputIterator1,
          typename InputIterator2,
          typename OutputIterator,
          typename ScalarType>
CUSP_DEPRECATED
void xmy(InputIterator1 first1,
         InputIterator1 last1,
         InputIterator2 first2,
         OutputIterator output);

/*! \p xmy : elementwise multiplication of two vectors (output[i] = x[i] * y[i])
 */
template <typename Array1,
          typename Array2,
          typename Array3>
void xmy(const Array1& x,
         const Array2& y,
               Array3& output);

/*! \p xmy : elementwise multiplication of two vectors (output[i] = x[i] * y[i])
 */
template <typename Array1,
          typename Array2,
          typename Array3>
void xmy(const Array1& x,
         const Array2& y,
         const Array3& output);


template <typename InputIterator,
          typename ForwardIterator>
CUSP_DEPRECATED
void copy(InputIterator   first1,
          InputIterator   last1,
          ForwardIterator first2);

/*! \p copy : vector copy (y = x)
 */
template <typename Array1,
          typename Array2>
void copy(const Array1& array1,
                Array2& array2);

/*! \p copy : vector copy (y = x)
 */
template <typename Array1,
          typename Array2>
void copy(const Array1& array1,
          const Array2& array2);


template <typename InputIterator1,
          typename InputIterator2>
CUSP_DEPRECATED
typename thrust::iterator_value<InputIterator1>::type
    dot(InputIterator1 first1,
        InputIterator1 last1,
        InputIterator2 first2);

/*! \p dot : dot product (x^T * y)
 */
template <typename Array1,
          typename Array2>
typename Array1::value_type
    dot(const Array1& x,
        const Array2& y);


template <typename InputIterator1,
          typename InputIterator2>
CUSP_DEPRECATED
typename thrust::iterator_value<InputIterator1>::type
    dotc(InputIterator1 first1,
         InputIterator1 last1,
         InputIterator2 first2);

/*! \p dotc : conjugate dot product (conjugate(x)^T * y)
 */
template <typename Array1,
          typename Array2>
typename Array1::value_type
    dotc(const Array1& x,
         const Array2& y);


template <typename ForwardIterator,
          typename ScalarType>
CUSP_DEPRECATED
void fill(ForwardIterator first,
          ForwardIterator last,
          ScalarType alpha);

/*! \p fill : vector fill (x[i] = alpha)
 */
template <typename Array,
          typename ScalarType>
void fill(Array& array,
          ScalarType alpha);

/*! \p fill : vector fill (x[i] = alpha)
 */
template <typename Array,
          typename ScalarType>
void fill(const Array& array,
          ScalarType alpha);


template <typename InputIterator>
CUSP_DEPRECATED
typename norm_type<typename thrust::iterator_value<InputIterator>::type>::type
    nrm1(InputIterator first,
         InputIterator last);

/*! \p nrm1 : vector 1-norm (sum abs(x[i]))
 */
template <typename Array>
typename norm_type<typename Array::value_type>::type
    nrm1(const Array& array);


template <typename InputIterator>
CUSP_DEPRECATED
typename norm_type<typename thrust::iterator_value<InputIterator>::type>::type
    nrm2(InputIterator first,
         InputIterator last);

/*! \p nrm2 : vector 2-norm (sqrt(sum x[i] * x[i] )
 */
template <typename Array>
typename norm_type<typename Array::value_type>::type
    nrm2(const Array& array);


template <typename InputIterator>
CUSP_DEPRECATED
typename thrust::iterator_value<InputIterator>::type
    nrmmax(InputIterator first,
           InputIterator last);

/*! \p nrmmax : vector infinity norm
 */
template <typename Array>
typename Array::value_type
    nrmmax(const Array& array);


template <typename ForwardIterator,
          typename ScalarType>
CUSP_DEPRECATED
void scal(ForwardIterator first,
          ForwardIterator last,
          ScalarType alpha);

/*! \p nrmmax : scale vector (x[i] = alpha * x[i])
 */
template <typename Array,
          typename ScalarType>
void scal(Array& x,
          ScalarType alpha);

/*! \p nrmmax : scale vector (x[i] = alpha * x[i])
 */
template <typename Array,
          typename ScalarType>
void scal(const Array& x,
          ScalarType alpha);

/*! \}
 */

} // end namespace blas
} // end namespace cusp

#include <cusp/detail/blas.inl>

