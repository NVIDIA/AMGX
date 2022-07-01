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

#include <cusp/convert.h>

namespace cusp
{

//////////////////
// Constructors //
//////////////////

// construct from a different matrix
template<typename ValueType, class MemorySpace, class Orientation>
template <typename MatrixType>
array2d<ValueType,MemorySpace,Orientation>
  ::array2d(const MatrixType& matrix)
  {
    cusp::convert(matrix, *this);
  }

//////////////////////
// Member Functions //
//////////////////////

template <typename ValueType, class MemorySpace, class Orientation>
array2d<ValueType,MemorySpace,Orientation>&
array2d<ValueType,MemorySpace,Orientation>
  ::operator=(const array2d<ValueType,MemorySpace,Orientation>& matrix)
  {
    cusp::convert(matrix, *this);

    return *this;
  }

template <typename ValueType, class MemorySpace, class Orientation>
template <typename MatrixType>
array2d<ValueType,MemorySpace,Orientation>&
array2d<ValueType,MemorySpace,Orientation>
  ::operator=(const MatrixType& matrix)
  {
    cusp::convert(matrix, *this);

    return *this;
  }

/////////////////////
// Other Functions //
/////////////////////

namespace detail
{
template <typename Array1, typename Array2>
bool array2d_equal(const Array1& lhs, const Array2& rhs)
{
  typedef typename Array1::orientation Orientation1;
  typedef typename Array2::orientation Orientation2;

  if (lhs.num_rows != rhs.num_rows || lhs.num_cols != rhs.num_cols)
      return false;
  
  thrust::counting_iterator<size_t> begin(0);
  thrust::counting_iterator<size_t> end(lhs.num_entries);

  cusp::detail::logical_to_physical_functor      <size_t, Orientation1>               func1(lhs.num_rows, lhs.num_cols, lhs.pitch);
  cusp::detail::logical_to_other_physical_functor<size_t, Orientation1, Orientation2> func2(rhs.num_rows, rhs.num_cols, rhs.pitch);

  // like a boss
  return thrust::equal(thrust::make_permutation_iterator(lhs.values.begin(), thrust::make_transform_iterator(begin, func1)),
                       thrust::make_permutation_iterator(lhs.values.begin(), thrust::make_transform_iterator(end,   func1)),
                       thrust::make_permutation_iterator(rhs.values.begin(), thrust::make_transform_iterator(begin, func2)));
}
  
} // end namespace detail

template<typename ValueType1, typename MemorySpace1, typename Orientation1,
         typename ValueType2, typename MemorySpace2, typename Orientation2>
bool operator==(const array2d<ValueType1,MemorySpace1,Orientation1>& lhs,
                const array2d<ValueType2,MemorySpace2,Orientation2>& rhs)
{
  return cusp::detail::array2d_equal(lhs, rhs);
}

template<typename Array1, typename Orientation1,
         typename Array2, typename Orientation2>
bool operator==(const array2d_view<Array1,Orientation1>& lhs,
                const array2d_view<Array2,Orientation2>& rhs)
{
  return cusp::detail::array2d_equal(lhs, rhs);
}

template<typename ValueType1, typename MemorySpace1, typename Orientation1,
         typename Array2, typename Orientation2>
bool operator==(const array2d<ValueType1,MemorySpace1,Orientation1>& lhs,
                const array2d_view<Array2,Orientation2>& rhs)
{
  return cusp::detail::array2d_equal(lhs, rhs);
}

template<typename Array1, typename Orientation1,
         typename ValueType2, typename MemorySpace2, typename Orientation2>
bool operator==(const array2d_view<Array1,Orientation1>& lhs,
                const array2d<ValueType2,MemorySpace2,Orientation2>& rhs)
{
  return cusp::detail::array2d_equal(lhs, rhs);
}

template<typename ValueType1, typename MemorySpace1, typename Orientation1,
         typename ValueType2, typename MemorySpace2, typename Orientation2>
bool operator!=(const array2d<ValueType1,MemorySpace1,Orientation1>& lhs,
                const array2d<ValueType2,MemorySpace2,Orientation2>& rhs)
{
    return !(lhs == rhs);
}

template<typename Array1, typename Orientation1,
         typename Array2, typename Orientation2>
bool operator!=(const array2d_view<Array1,Orientation1>& lhs,
                const array2d_view<Array2,Orientation2>& rhs)
{
    return !(lhs == rhs);
}

template<typename ValueType1, typename MemorySpace1, typename Orientation1,
         typename Array2, typename Orientation2>
bool operator!=(const array2d<ValueType1,MemorySpace1,Orientation1>& lhs,
                const array2d_view<Array2,Orientation2>& rhs)
{
    return !(lhs == rhs);
}

template<typename Array1, typename Orientation1,
         typename ValueType2, typename MemorySpace2, typename Orientation2>
bool operator!=(const array2d_view<Array1,Orientation1>& lhs,
                const array2d<ValueType2,MemorySpace2,Orientation2>& rhs)
{
    return !(lhs == rhs);
}
    
} // end namespace cusp

