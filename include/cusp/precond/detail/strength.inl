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


#include <cusp/copy.h>
#include <cusp/array1d.h>
#include <cusp/convert.h>
#include <cusp/csr_matrix.h>
#include <cusp/detail/format_utils.h>

#include <thrust/count.h>
#include <thrust/functional.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/iterator/permutation_iterator.h>

namespace cusp
{
namespace precond
{
namespace detail
{

template <typename ValueType>
__host__ __device__
ValueType absolute_value(const ValueType& x)
{
  return (x < 0) ? -x : x;
}

////////////////
// Host Paths //
////////////////

template <typename Matrix1, typename Matrix2>
void symmetric_strength_of_connection(const Matrix1& A, Matrix2& S, const double theta,
                                      cusp::csr_format, cusp::host_memory,
                                      cusp::csr_format, cusp::host_memory)
{
  typedef typename Matrix1::index_type IndexType;
  typedef typename Matrix1::value_type ValueType;

  // extract matrix diagonal
  cusp::array1d<ValueType,cusp::host_memory> diagonal;
  cusp::detail::extract_diagonal(A, diagonal);

  IndexType num_entries = 0;
  
  // count num_entries in output
  for(size_t i = 0; i < A.num_rows; i++)
  {
    const ValueType Aii = diagonal[i];

    for(IndexType jj = A.row_offsets[i]; jj < A.row_offsets[i + 1]; jj++)
    {
      const IndexType   j = A.column_indices[jj];
      const ValueType Aij = A.values[jj];
      const ValueType Ajj = diagonal[j];

      //  |A(i,j)| >= theta * sqrt(|A(i,i)|*|A(j,j)|)
      if(Aij*Aij >= (theta * theta) * absolute_value(Aii * Ajj))
        num_entries++;
    }
  }
  
  // resize output
  S.resize(A.num_rows, A.num_cols, num_entries);

  // reset counter for second pass
  num_entries = 0;

  // copy strong connections to output
  for(size_t i = 0; i < A.num_rows; i++)
  {
    const ValueType Aii = diagonal[i];
  
    S.row_offsets[i] = num_entries;

    for(IndexType jj = A.row_offsets[i]; jj < A.row_offsets[i + 1]; jj++)
    {
      const IndexType   j = A.column_indices[jj];
      const ValueType Aij = A.values[jj];
      const ValueType Ajj = diagonal[j];
      
      //  |A(i,j)| >= theta * sqrt(|A(i,i)|*|A(j,j)|)
      if(Aij*Aij >= (theta * theta) * absolute_value(Aii * Ajj))
      {
        S.column_indices[num_entries] =   j;
        S.values[num_entries]         = Aij;
        num_entries++;
      }
    }
  }
    
  S.row_offsets[S.num_rows] = num_entries;
}

//////////////////
// Device Paths //
//////////////////

/* none for now */

///////////////////
// Generic Paths //
///////////////////

template <typename ValueType>
struct is_strong_connection
{
  ValueType theta;

  is_strong_connection(const ValueType theta) : theta(theta) {}

  template <typename Tuple>
    __host__ __device__
  bool operator()(const Tuple& t) const
  {
    ValueType Aij = thrust::get<0>(t);
    ValueType Aii = thrust::get<1>(t);
    ValueType Ajj = thrust::get<2>(t);

    // square everything to eliminate the sqrt()
    return (Aij * Aij) >= (theta * theta) * absolute_value(Aii * Ajj);
  }
};

template <typename Matrix1, typename Matrix2, typename MemorySpace>
void symmetric_strength_of_connection(const Matrix1& A, Matrix2& S, const double theta,
                                      cusp::coo_format, MemorySpace,
                                      cusp::coo_format, MemorySpace)
{
  typedef typename Matrix1::index_type IndexType;
  typedef typename Matrix1::value_type ValueType;

  cusp::array1d<ValueType,MemorySpace> diagonal;
  cusp::detail::extract_diagonal(A, diagonal);

  is_strong_connection<ValueType> pred(theta);

  // compute number of entries in output
  IndexType num_entries = thrust::count_if
    (thrust::make_zip_iterator(thrust::make_tuple
       (A.values.begin(),
        thrust::make_permutation_iterator(diagonal.begin(), A.row_indices.begin()),
        thrust::make_permutation_iterator(diagonal.begin(), A.column_indices.begin()))),
     thrust::make_zip_iterator(thrust::make_tuple
       (A.values.begin(),
        thrust::make_permutation_iterator(diagonal.begin(), A.row_indices.begin()),
        thrust::make_permutation_iterator(diagonal.begin(), A.column_indices.begin()))) + A.num_entries,
     pred);

  // this is just zipping up (A[i,j],A[i,i],A[j,j]) and applying is_strong_connection to each tuple

  // resize output
  S.resize(A.num_rows, A.num_cols, num_entries);

  // copy strong connections to output
  thrust::copy_if(thrust::make_zip_iterator(thrust::make_tuple(A.row_indices.begin(), A.column_indices.begin(), A.values.begin())),
                  thrust::make_zip_iterator(thrust::make_tuple(A.row_indices.begin(), A.column_indices.begin(), A.values.begin())) + A.num_entries,
                  thrust::make_zip_iterator(thrust::make_tuple
                    (A.values.begin(),
                     thrust::make_permutation_iterator(diagonal.begin(), A.row_indices.begin()),
                     thrust::make_permutation_iterator(diagonal.begin(), A.column_indices.begin()))),
                  thrust::make_zip_iterator(thrust::make_tuple(S.row_indices.begin(), S.column_indices.begin(), S.values.begin())),
                  pred);
}

//////////////////
// Default Path //
//////////////////

template <typename Matrix1, typename Matrix2,
          typename Format1, typename MemorySpace1,
          typename Format2, typename MemorySpace2>
void symmetric_strength_of_connection(const Matrix1& A, Matrix2& S, const double theta,
                                      Format1, MemorySpace1,
                                      Format2, MemorySpace2)
{
  typedef typename Matrix1::index_type IndexType1;
  typedef typename Matrix1::value_type ValueType1;
  typedef typename Matrix2::index_type IndexType2;
  typedef typename Matrix2::value_type ValueType2;

  // do everything on the host using the CSR format
  cusp::csr_matrix<IndexType1,ValueType1,cusp::host_memory> A_csr(A);
  cusp::csr_matrix<IndexType2,ValueType2,cusp::host_memory> S_csr;

  symmetric_strength_of_connection(A_csr, S_csr, theta);

  cusp::convert(S_csr, S);
}

/////////////////
// Entry Point //
/////////////////

template <typename Matrix1, typename Matrix2>
void symmetric_strength_of_connection(const Matrix1& A, Matrix2& S, const double theta)
{
  CUSP_PROFILE_SCOPED();

  if (theta == 0.0)
  {
    // everything is a strong connection
    cusp::copy(A,S);
  }
  else
  {
    // dispatch based on format and memory_space
    symmetric_strength_of_connection
      (A, S, theta,
       typename Matrix1::format(), typename Matrix1::memory_space(), 
       typename Matrix2::format(), typename Matrix2::memory_space());
  }
}

} // end namepace detail
} // end namespace precond
} // end namespace cusp

