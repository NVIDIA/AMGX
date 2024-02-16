// SPDX-FileCopyrightText: 2008 - 2024 NVIDIA CORPORATION. All Rights Reserved.
//
// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#include <cusp/blas.h>
#include <cusp/multiply.h>
#include <cusp/krylov/arnoldi.h>
#include <cusp/array1d.h>
#include <cusp/array2d.h>
#include <cusp/detail/random.h>
#include <cusp/detail/format_utils.h>

#include <thrust/extrema.h>
#include <thrust/transform.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/reduce.h>

#include <thrust/detail/integer_traits.h>

namespace cusp
{
namespace detail
{

// TODO move this to a shared header
// absolute<T> computes the absolute value of a number f(x) -> |x|
template <typename T>
struct absolute : public amgx::thrust::unary_function<T,T>
{
    __host__ __device__
	T operator()(T x)
	{
	    return x < 0 ? -x : x;
	}
};


template <typename Matrix>    
double estimate_spectral_radius(const Matrix& A, size_t k = 20)
{
    CUSP_PROFILE_SCOPED();

    typedef typename Matrix::index_type   IndexType;
    typedef typename Matrix::value_type   ValueType;
    typedef typename Matrix::memory_space MemorySpace;

    const IndexType N = A.num_rows;

    cusp::array1d<ValueType, MemorySpace> x(N);
    cusp::array1d<ValueType, MemorySpace> y(N);

    // initialize x to random values in [0,1)
    cusp::copy(cusp::detail::random_reals<ValueType>(N), x);

    for(size_t i = 0; i < k; i++)
    {
        cusp::blas::scal(x, ValueType(1.0) / cusp::blas::nrmmax(x));
        cusp::multiply(A, x, y);
        x.swap(y);
    }
   
    if (k == 0)
        return 0;
    else
        return cusp::blas::nrm2(x) / cusp::blas::nrm2(y);
}

template <typename Matrix>    
double ritz_spectral_radius(const Matrix& A, size_t k = 10)
{
    CUSP_PROFILE_SCOPED();

    typedef typename Matrix::value_type ValueType;

    cusp::array2d<ValueType,cusp::host_memory> H;
    cusp::krylov::arnoldi(A, H, k);

    return estimate_spectral_radius(H);
}

template <typename Matrix>    
double ritz_spectral_radius_symmetric(const Matrix& A, size_t k = 10)
{
    CUSP_PROFILE_SCOPED();

    typedef typename Matrix::value_type ValueType;

    cusp::array2d<ValueType,cusp::host_memory> H;
    cusp::krylov::lanczos(A, H, k);

    return estimate_spectral_radius(H);
}

template <typename IndexType, typename ValueType, typename MemorySpace>    
double disks_spectral_radius(const cusp::coo_matrix<IndexType,ValueType,MemorySpace>& A)
{
    CUSP_PROFILE_SCOPED();

    const IndexType N = A.num_rows;

    // compute sum of absolute values for each row of A
    cusp::array1d<IndexType, MemorySpace> row_sums(N);

    {
      cusp::array1d<IndexType, MemorySpace> temp(N);
      amgx::thrust::reduce_by_key
        (A.row_indices.begin(), A.row_indices.end(),
         amgx::thrust::make_transform_iterator(A.values.begin(), absolute<ValueType>()),
         temp.begin(),
         row_sums.begin());
    }

    return *amgx::thrust::max_element(row_sums.begin(), row_sums.end());
}

template <typename Matrix> 
double disks_spectral_radius(const Matrix& A)
{
    typedef typename Matrix::index_type   IndexType;
    typedef typename Matrix::value_type   ValueType;
    typedef typename Matrix::memory_space MemorySpace;

    const cusp::coo_matrix<IndexType,ValueType,MemorySpace> C(A);

    return disks_spectral_radius(C);
}

} // end namespace detail
} // end namespace cusp

