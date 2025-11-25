// SPDX-FileCopyrightText: 2008 - 2024 NVIDIA CORPORATION. All Rights Reserved.
//
// SPDX-License-Identifier: BSD-3-Clause

#include <cusp/array1d.h>

#include <cusp/exception.h>

#include <thrust/copy.h>
#include <thrust/fill.h>
#include <thrust/functional.h>
#include <thrust/transform.h>
#include <thrust/transform_reduce.h>
#include <thrust/inner_product.h>

#include <thrust/iterator/transform_iterator.h>
#include <thrust_wrapper.h>

#include <cmath>

namespace cusp
{
namespace blas
{

namespace detail
{
    template <typename Array1, typename Array2>
    void assert_same_dimensions(const Array1& array1,
                                const Array2& array2)
    {
        if(array1.size() != array2.size())
            throw cusp::invalid_input_exception("array dimensions do not match");
    }
    
    template <typename Array1, typename Array2, typename Array3>
    void assert_same_dimensions(const Array1& array1,
                                const Array2& array2,
                                const Array3& array3)
    {
        assert_same_dimensions(array1, array2);
        assert_same_dimensions(array2, array3);
    }
    
    template <typename Array1, typename Array2, typename Array3, typename Array4>
    void assert_same_dimensions(const Array1& array1,
                                const Array2& array2,
                                const Array3& array3,
                                const Array4& array4)
    {
        assert_same_dimensions(array1, array2);
        assert_same_dimensions(array2, array3);
        assert_same_dimensions(array3, array4);
    }

    // square<T> computes the square of a number f(x) -> x*x
    template <typename T>
        struct square
        {
            __host__ __device__
                T operator()(T x)
                { 
                    return x * x;
                }
        };
        
    // absolute<T> computes the absolute value of a number f(x) -> sqrt(x)
    template <typename T>
        struct sqrt
        {
            __host__ __device__
                T operator()(T x)
                { 
		    return std::sqrt(x);
                }
        };
        
    // absolute<T> computes the absolute value of a number f(x) -> |x|
    template <typename T>
        struct absolute
        {
            __host__ __device__
                T operator()(T x)
                { 
		    return abs(x);
                }
        };

    // maximum<T> returns the largest of two numbers
    template <typename T>
        struct maximum
        {
            __host__ __device__
  	        T operator()(T x, T y)
                { 
		  return amgx::thrust::maximum<T>()(x,y);
                }
        };

    // maximum<T> returns the number with the largest real part
    template <typename T>
        struct maximum<cusp::complex<T> >
        {
            __host__ __device__
	    cusp::complex<T> operator()(cusp::complex<T> x, cusp::complex<T> y)
                { 
		  return amgx::thrust::maximum<T>()(x.real(),y.real());
                }
        };
    
    // conjugate<T> computes the complex conjugate of a number f(a + b * i) -> a - b * i
    template <typename T>
        struct conjugate
        {
            __host__ __device__
                T operator()(T x)
                { 
                    return x;
                }
        };

    template <typename T>
        struct conjugate<cusp::complex<T> >
        {
            __host__ __device__
	        cusp::complex<T> operator()(cusp::complex<T> x)
                { 
		    return cusp::conj(x);
                }
        };

    // square<T> computes the square of a number f(x) -> x*conj(x)
    template <typename T>
        struct norm_squared
        {
            __host__ __device__
                T operator()(T x)
                { 
  		    return x * conjugate<T>()(x);
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
                void operator()(T2 & x)
                { 
		  x = alpha * x;
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
                    amgx::thrust::get<1>(t) = alpha * amgx::thrust::get<0>(t) +
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
                    amgx::thrust::get<2>(t) = alpha * amgx::thrust::get<0>(t) +
		                        beta  * amgx::thrust::get<1>(t);
                }
        };

  template <typename T1,typename T2,typename T3>
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
                    amgx::thrust::get<3>(t) = alpha * amgx::thrust::get<0>(t) +
                                        beta  * amgx::thrust::get<1>(t) +
                                        gamma * amgx::thrust::get<2>(t);
                }
        };

    template <typename T>
        struct XMY
        {
            __host__ __device__
                T operator()(T x, T y)
                { 
                    return x * y;
                }
        };

  template <typename ForwardIterator1,
	    typename ForwardIterator2,
	    typename ScalarType>
  void axpy(ForwardIterator1 first1,
	    ForwardIterator1 last1,
	    ForwardIterator2 first2,
	    ScalarType alpha)
  {
    size_t N = last1 - first1;
    amgx::thrust::for_each(amgx::thrust::make_zip_iterator(amgx::thrust::make_tuple(first1, first2)),
                     amgx::thrust::make_zip_iterator(amgx::thrust::make_tuple(first1, first2)) + N,
                     detail::AXPY<ScalarType>(alpha));
  }

  template <typename InputIterator1,
	    typename InputIterator2,
	    typename OutputIterator,
	    typename ScalarType1,
	    typename ScalarType2>
  void axpby(InputIterator1 first1,
	     InputIterator1 last1,
	     InputIterator2 first2,
	     OutputIterator output,
	     ScalarType1 alpha,
	     ScalarType2 beta)
  {
    size_t N = last1 - first1;
    amgx::thrust::for_each(amgx::thrust::make_zip_iterator(amgx::thrust::make_tuple(first1, first2, output)),
                     amgx::thrust::make_zip_iterator(amgx::thrust::make_tuple(first1, first2, output)) + N,
                     detail::AXPBY<ScalarType1,ScalarType2>(alpha, beta));
  }

  template <typename InputIterator1,
	    typename InputIterator2,
	    typename InputIterator3,
	    typename OutputIterator,
	    typename ScalarType1,
	    typename ScalarType2,
	    typename ScalarType3>
  void axpbypcz(InputIterator1 first1,
		InputIterator1 last1,
		InputIterator2 first2,
		InputIterator3 first3,
		OutputIterator output,
		ScalarType1 alpha,
		ScalarType2 beta,
		ScalarType3 gamma)
  {
    CUSP_PROFILE_SCOPED();
    size_t N = last1 - first1;
    amgx::thrust::for_each(amgx::thrust::make_zip_iterator(amgx::thrust::make_tuple(first1, first2, first3, output)),
                     amgx::thrust::make_zip_iterator(amgx::thrust::make_tuple(first1, first2, first3, output)) + N,
                     detail::AXPBYPCZ<ScalarType1,ScalarType2,ScalarType3>(alpha, beta, gamma));
  }
  
  template <typename InputIterator1,
	    typename InputIterator2,
	    typename OutputIterator>
  void xmy(InputIterator1 first1,
	   InputIterator1 last1,
	   InputIterator2 first2,
	   OutputIterator output)
  {
    typedef typename amgx::thrust::iterator_traits<OutputIterator>::value_type ScalarType;
    amgx::thrust::transform(first1, last1, first2, output, detail::XMY<ScalarType>());
  }
  
  template <typename InputIterator,
	    typename ForwardIterator>
  void copy(InputIterator   first1,
	    InputIterator   last1,
	    ForwardIterator first2)
  {
      amgx::thrust::copy(first1, last1, first2);
  }
  
  template <typename InputIterator1,
	    typename InputIterator2>
  typename amgx::thrust::iterator_traits<InputIterator1>::value_type
  dot(InputIterator1 first1,
      InputIterator1 last1,
      InputIterator2 first2)
  {
    typedef typename amgx::thrust::iterator_traits<InputIterator1>::value_type OutputType;
    return amgx::thrust::inner_product(first1, last1, first2, OutputType(0));
  }

  template <typename InputIterator1,
	    typename InputIterator2>
  typename amgx::thrust::iterator_traits<InputIterator1>::value_type
  dotc(InputIterator1 first1,
       InputIterator1 last1,
       InputIterator2 first2)
  {
    typedef typename amgx::thrust::iterator_traits<InputIterator1>::value_type OutputType;
    return amgx::thrust::inner_product(amgx::thrust::make_transform_iterator(first1, detail::conjugate<OutputType>()),
                                 amgx::thrust::make_transform_iterator(last1,  detail::conjugate<OutputType>()),
                                 first2,
                                 OutputType(0));
  }

  template <typename ForwardIterator,
	    typename ScalarType>
  void fill(ForwardIterator first,
	    ForwardIterator last,
	    ScalarType alpha)
  {
      typedef typename amgx::thrust::iterator_traits<ForwardIterator>::memory_space MemorySpace;
      thrust_wrapper::fill<CuspMemMap<MemorySpace>::value>(first, last, alpha);
  }
  
  template <typename InputIterator>
  typename norm_type<typename amgx::thrust::iterator_traits<InputIterator>::value_type>::type
  nrm1(InputIterator first,
       InputIterator last)
  {
    typedef typename amgx::thrust::iterator_traits<InputIterator>::value_type ValueType;
    
    detail::absolute<ValueType> unary_op;
    amgx::thrust::plus<ValueType>     binary_op;
    
    ValueType init = 0;
    
    return abs(thrust_wrapper::transform_reduce(first, last, unary_op, init, binary_op));
  }

  template <typename InputIterator>
  typename norm_type<typename amgx::thrust::iterator_traits<InputIterator>::value_type>::type
  nrm2(InputIterator first,
       InputIterator last)
  {
    typedef typename amgx::thrust::iterator_traits<InputIterator>::value_type ValueType;

    detail::norm_squared<ValueType> unary_op;
    amgx::thrust::plus<ValueType>   binary_op;

    ValueType init = 0;

    return std::sqrt( abs(amgx::thrust::transform_reduce(first, last, unary_op, init, binary_op)) );
  }

  template <typename InputIterator>
  typename amgx::thrust::iterator_traits<InputIterator>::value_type
  nrmmax(InputIterator first,
	 InputIterator last)
  {
    typedef typename amgx::thrust::iterator_traits<InputIterator>::value_type ValueType;

    detail::absolute<ValueType>  unary_op;
    detail::maximum<ValueType>   binary_op;

    ValueType init = 0;

    return amgx::thrust::transform_reduce(first, last, unary_op, init, binary_op);
  }

  template <typename ForwardIterator,
	    typename ScalarType>
  void scal(ForwardIterator first,
	    ForwardIterator last,
	    ScalarType alpha)
  {
      amgx::thrust::for_each(first,
                     last,
                     detail::SCAL<ScalarType>(alpha));
  }
} // end namespace detail


template <typename ForwardIterator1,
          typename ForwardIterator2,
          typename ScalarType>
void axpy(ForwardIterator1 first1,
          ForwardIterator1 last1,
          ForwardIterator2 first2,
          ScalarType alpha)
{
  cusp::blas::detail::axpy(first1, last1, first2, alpha);
}

template <typename Array1,
          typename Array2, 
	  typename ScalarType>
void axpy(const Array1& x,
                Array2& y,
          ScalarType alpha)
{
    CUSP_PROFILE_SCOPED();
    detail::assert_same_dimensions(x, y);
    cusp::blas::detail::axpy(x.begin(), x.end(), y.begin(), alpha);
}

template <typename Array1,
          typename Array2, 
	  typename ScalarType>
void axpy(const Array1& x,
          const Array2& y,
          ScalarType alpha)
{
    CUSP_PROFILE_SCOPED();
    detail::assert_same_dimensions(x, y);
    cusp::blas::detail::axpy(x.begin(), x.end(), y.begin(), alpha);
}


template <typename InputIterator1,
          typename InputIterator2,
          typename OutputIterator,
          typename ScalarType>
void axpby(InputIterator1 first1,
           InputIterator1 last1,
           InputIterator2 first2,
           OutputIterator output,
           ScalarType alpha,
           ScalarType beta)
{
  cusp::blas::detail::axpby(first1, last1, first2, output, alpha, beta);
}

template <typename Array1,
          typename Array2,
          typename Array3,
	  typename ScalarType1,
	  typename ScalarType2>
void axpby(const Array1& x,
           const Array2& y,
                 Array3& z,
           ScalarType1 alpha,
           ScalarType2 beta)
{
    CUSP_PROFILE_SCOPED();
    detail::assert_same_dimensions(x, y, z);
    cusp::blas::detail::axpby(x.begin(), x.end(), y.begin(), z.begin(), alpha, beta);
}

template <typename Array1,
          typename Array2,
          typename Array3,
	  typename ScalarType1,
	  typename ScalarType2>
void axpby(const Array1& x,
           const Array2& y,
           const Array3& z,
           ScalarType1 alpha,
           ScalarType2 beta)
{
    CUSP_PROFILE_SCOPED();
    detail::assert_same_dimensions(x, y, z);
    cusp::blas::detail::axpby(x.begin(), x.end(), y.begin(), z.begin(), alpha, beta);
}

template <typename InputIterator1,
          typename InputIterator2,
          typename InputIterator3,
          typename OutputIterator,
          typename ScalarType>
void axpbypcz(InputIterator1 first1,
              InputIterator1 last1,
              InputIterator2 first2,
              InputIterator3 first3,
              OutputIterator output,
              ScalarType alpha,
              ScalarType beta,
              ScalarType gamma)
{
    CUSP_PROFILE_SCOPED();
    cusp::blas::detail::axpbypcz(first1, last1, first2, first3, output.begin(), alpha, beta, gamma);
}

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
	      ScalarType3 gamma)
{
    CUSP_PROFILE_SCOPED();
    detail::assert_same_dimensions(x, y, z, output);
    cusp::blas::detail::axpbypcz(x.begin(), x.end(), y.begin(), z.begin(), output.begin(), alpha, beta, gamma);
}

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
	      ScalarType3 gamma)
{
    CUSP_PROFILE_SCOPED();
    detail::assert_same_dimensions(x, y, z, output);
    cusp::blas::detail::axpbypcz(x.begin(), x.end(), y.begin(), z.begin(), output.begin(), alpha, beta, gamma);
}
    

template <typename InputIterator1,
          typename InputIterator2,
          typename OutputIterator>
void xmy(InputIterator1 first1,
         InputIterator1 last1,
         InputIterator2 first2,
         OutputIterator output)
{
    typedef typename amgx::thrust::iterator_traits<OutputIterator>::value_type ScalarType;
    typedef typename amgx::thrust::iterator_traits<InputIterator1>::memory_space MemorySpace;
    thrust_wrapper::transform<CuspMemMap<MemorySpace>::value>(first1, last1, first2, output, detail::XMY<ScalarType>());
}

template <typename Array1,
          typename Array2,
          typename Array3>
void xmy(const Array1& x,
         const Array2& y,
               Array3& output)
{
    CUSP_PROFILE_SCOPED();
    detail::assert_same_dimensions(x, y, output);
    cusp::blas::detail::xmy(x.begin(), x.end(), y.begin(), output.begin());
}

template <typename Array1,
          typename Array2,
          typename Array3>
void xmy(const Array1& x,
         const Array2& y,
         const Array3& output)
{
    CUSP_PROFILE_SCOPED();
    detail::assert_same_dimensions(x, y, output);
    cusp::blas::detail::xmy(x.begin(), x.end(), y.begin(), output.begin());
}

template <typename InputIterator,
          typename ForwardIterator>
void copy(InputIterator   first1,
          InputIterator   last1,
          ForwardIterator first2)
{
    amgx::thrust::copy(first1, last1, first2);
}

template <typename Array1,
          typename Array2>
void copy(const Array1& x,
                Array2& y)
{
    CUSP_PROFILE_SCOPED();
    detail::assert_same_dimensions(x, y);
    cusp::blas::detail::copy(x.begin(), x.end(), y.begin());
}

template <typename Array1,
          typename Array2>
void copy(const Array1& x,
          const Array2& y)
{
    CUSP_PROFILE_SCOPED();
    detail::assert_same_dimensions(x, y);
    cusp::blas::detail::copy(x.begin(), x.end(), y.begin());
}


// TODO properly harmonize heterogenous types
template <typename InputIterator1,
          typename InputIterator2>
typename amgx::thrust::iterator_traits<InputIterator1>::value_type
    dot(InputIterator1 first1,
        InputIterator1 last1,
        InputIterator2 first2)
{
    typedef typename amgx::thrust::iterator_traits<InputIterator1>::value_type OutputType;
    return amgx::thrust::inner_product(first1, last1, first2, OutputType(0));
}

// TODO properly harmonize heterogenous types
template <typename Array1,
          typename Array2>
typename Array1::value_type
    dot(const Array1& x,
        const Array2& y)
{
    CUSP_PROFILE_SCOPED();
    detail::assert_same_dimensions(x, y);
    return cusp::blas::detail::dot(x.begin(), x.end(), y.begin());
}

// TODO properly harmonize heterogenous types
template <typename InputIterator1,
          typename InputIterator2>
typename amgx::thrust::iterator_traits<InputIterator1>::value_type
    dotc(InputIterator1 first1,
         InputIterator1 last1,
         InputIterator2 first2)
{
    typedef typename amgx::thrust::iterator_traits<InputIterator1>::value_type OutputType;
    return amgx::thrust::inner_product(amgx::thrust::make_transform_iterator(first1, detail::conjugate<OutputType>()),
                                 amgx::thrust::make_transform_iterator(last1,  detail::conjugate<OutputType>()),
                                 first2,
                                 OutputType(0));
}

// TODO properly harmonize heterogenous types
template <typename Array1,
          typename Array2>
typename Array1::value_type
    dotc(const Array1& x,
         const Array2& y)
{
    CUSP_PROFILE_SCOPED();
    detail::assert_same_dimensions(x, y);
    return cusp::blas::detail::dotc(x.begin(), x.end(), y.begin());
}



template <typename ForwardIterator,
          typename ScalarType>
void fill(ForwardIterator first,
          ForwardIterator last,
          ScalarType alpha)
{
    thrust_wrapper::fill(first, last, alpha);
}

template <typename Array,
          typename ScalarType>
void fill(Array& x,
	  ScalarType alpha)
{
    CUSP_PROFILE_SCOPED();
    cusp::blas::detail::fill(x.begin(), x.end(), alpha);
}

template <typename Array,
          typename ScalarType>
void fill(const Array& x,
          ScalarType alpha)
{
    CUSP_PROFILE_SCOPED();
    cusp::blas::detail::fill(x.begin(), x.end(), alpha);
}


template <typename InputIterator>
typename norm_type<typename amgx::thrust::iterator_traits<InputIterator>::value_type>::type
    nrm1(InputIterator first,
         InputIterator last)
{
    typedef typename amgx::thrust::iterator_traits<InputIterator>::value_type ValueType;

    detail::absolute<ValueType> unary_op;
    amgx::thrust::plus<ValueType>     binary_op;

    ValueType init = 0;

    return thrust_wrapper::transform_reduce(first, last, unary_op, init, binary_op);
}

template <typename Array>
typename norm_type<typename Array::value_type>::type
    nrm1(const Array& x)
{
    CUSP_PROFILE_SCOPED();
    return cusp::blas::detail::nrm1(x.begin(), x.end());
}


template <typename InputIterator>
typename norm_type<typename amgx::thrust::iterator_traits<InputIterator>::value_type>::type
    nrm2(InputIterator first,
         InputIterator last)
{
    typedef typename amgx::thrust::iterator_traits<InputIterator>::value_type ValueType;

    detail::norm_squared<ValueType> unary_op;
    amgx::thrust::plus<ValueType>   binary_op;

    ValueType init = 0;

    return std::sqrt( thrust_wrapper::transform_reduce(first, last, unary_op, init, binary_op) );
}

template <typename Array>
typename norm_type<typename Array::value_type>::type
    nrm2(const Array& x)
{
    CUSP_PROFILE_SCOPED();
    return cusp::blas::detail::nrm2(x.begin(), x.end());
}


template <typename InputIterator>
typename amgx::thrust::iterator_traits<InputIterator>::value_type
    nrmmax(InputIterator first,
           InputIterator last)
{
    typedef typename amgx::thrust::iterator_traits<InputIterator>::value_type ValueType;

    detail::absolute<ValueType>  unary_op;
    amgx::thrust::maximum<ValueType>   binary_op;

    ValueType init = 0;

    return thrust_wrapper::transform_reduce(first, last, unary_op, init, binary_op);
}

template <typename Array>
typename Array::value_type
    nrmmax(const Array& x)
{
    CUSP_PROFILE_SCOPED();
    return cusp::blas::detail::nrmmax(x.begin(), x.end());
}


template <typename ForwardIterator,
          typename ScalarType>
void scal(ForwardIterator first,
          ForwardIterator last,
          ScalarType alpha)
{
    typedef typename amgx::thrust::iterator_traits<ForwardIterator>::value_type ValueType;
    thrust_wrapper::transform(first, last, first, detail::SCAL<ValueType>(alpha));
}

template <typename Array,
          typename ScalarType>
void scal(Array& x,
          ScalarType alpha)
{
    CUSP_PROFILE_SCOPED();
    cusp::blas::detail::scal(x.begin(), x.end(), alpha);
}

template <typename Array,
          typename ScalarType>
void scal(const Array& x,
          ScalarType alpha)
{
    CUSP_PROFILE_SCOPED();
    cusp::blas::detail::scal(x.begin(), x.end(), alpha);
}

} // end namespace blas
} // end namespace cusp

