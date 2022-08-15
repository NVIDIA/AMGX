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

/*! \file jacobi.inl
 *  \brief Inline file for jacobi.h
 */

#include <cusp/multiply.h>
#include <cusp/detail/format_utils.h>

#include <thrust/functional.h>
#include <thrust/transform.h>
#include <thrust/iterator/zip_iterator.h>

namespace cusp
{
namespace relaxation
{
namespace detail
{

template <typename ValueType>
struct jacobi_presmooth_functor
{
    ValueType omega;

    jacobi_presmooth_functor(ValueType omega) : omega(omega) {}

    __host__ __device__
    ValueType operator()(const ValueType& b, const ValueType& d) const
    {
        return omega * b / d;
    }
};

template <typename ValueType>
struct jacobi_postsmooth_functor
{
    ValueType omega;

    jacobi_postsmooth_functor(ValueType omega) : omega(omega) {}

    template <typename Tuple>
    __host__ __device__
    ValueType operator()(const Tuple& t)
    {
        const ValueType x = thrust::get<0>(t);
        const ValueType d = thrust::get<1>(t);
        const ValueType b = thrust::get<2>(t);
        const ValueType y = thrust::get<3>(t);

        return x + omega * (b - y) / d;
    }
};

} // end namespace detail


// constructor
template <typename ValueType, typename MemorySpace>
    jacobi<ValueType,MemorySpace>
    ::jacobi() : default_omega(0.0)
    {
    }

template <typename ValueType, typename MemorySpace>
template<typename MatrixType>
    jacobi<ValueType,MemorySpace>
    ::jacobi(const MatrixType& A, ValueType omega)
        : default_omega(omega), temp(A.num_rows)
    {
        CUSP_PROFILE_SCOPED();

        // extract the main diagonal
        cusp::detail::extract_diagonal(A, diagonal);
    }

// linear_operator
template <typename ValueType, typename MemorySpace>
template<typename MatrixType, typename VectorType1, typename VectorType2>
    void jacobi<ValueType,MemorySpace>
    ::operator()(const MatrixType& A, const VectorType1& b, VectorType2& x)
    {
        jacobi<ValueType,MemorySpace>::operator()(A,b,x,default_omega);
    }

// override default omega
template <typename ValueType, typename MemorySpace>
template<typename MatrixType, typename VectorType1, typename VectorType2>
    void jacobi<ValueType,MemorySpace>
    ::operator()(const MatrixType& A, const VectorType1& b, VectorType2& x, ValueType omega)
    {
        CUSP_PROFILE_SCOPED();

        // y <- A*x
        cusp::multiply(A, x, temp);
        
        // x <- x + omega * D^-1 * (b - y)
        thrust::transform(thrust::make_zip_iterator(thrust::make_tuple(x.begin(), diagonal.begin(), b.begin(), temp.begin())),
                          thrust::make_zip_iterator(thrust::make_tuple(x.end(),   diagonal.end(),   b.end(),   temp.end())),
                          x.begin(),
                          detail::jacobi_postsmooth_functor<ValueType>(omega));
    }

template <typename ValueType, typename MemorySpace>
template<typename MatrixType, typename VectorType1, typename VectorType2>
    void jacobi<ValueType,MemorySpace>
    ::presmooth(const MatrixType&, const VectorType1& b, VectorType2& x)
    {
        CUSP_PROFILE_SCOPED();
        
        // x <- omega * D^-1 * b 
        thrust::transform(b.begin(), b.end(),
                          diagonal.begin(),
                          x.begin(),
                          detail::jacobi_presmooth_functor<ValueType>(default_omega));
    }

template <typename ValueType, typename MemorySpace>
template<typename MatrixType, typename VectorType1, typename VectorType2>
    void jacobi<ValueType,MemorySpace>
    ::postsmooth(const MatrixType& A, const VectorType1& b, VectorType2& x)
    {
        CUSP_PROFILE_SCOPED();

        // y <- A*x
        cusp::multiply(A, x, temp);
        
        // x <- x + omega * D^-1 * (b - y)
        thrust::transform(thrust::make_zip_iterator(thrust::make_tuple(x.begin(), diagonal.begin(), b.begin(), temp.begin())),
                          thrust::make_zip_iterator(thrust::make_tuple(x.end(),   diagonal.end(),   b.end(),   temp.end())),
                          x.begin(),
                          detail::jacobi_postsmooth_functor<ValueType>(default_omega));
    }


} // end namespace relaxation
} // end namespace cusp

