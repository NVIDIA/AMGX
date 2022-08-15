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

/*! \file array2d.h
 *  \brief Two-dimensional array
 */

#pragma once

#include <cusp/detail/config.h>

#include <cusp/memory.h>
#include <cusp/format.h>
#include <cusp/array1d.h>
#include <cusp/detail/matrix_base.h>

#include <thrust/functional.h>

// TODO move indexing stuff to /detail/

namespace cusp
{

struct row_major    {};
struct column_major {};

// forward definitions
template<typename Array, class Orientation> class array2d_view;
    
namespace detail
{
  // (i,j) -> major dimension
  // (i,j) -> minor dimension
  // logical n -> (i,j)
  // (i,j) -> logical n
  // (i,j) -> physical n
  // logical n -> physical n
  // logical n -> physical n (translated)
  
  template <typename IndexType>
  __host__ __device__
  IndexType minor_dimension(IndexType num_rows, IndexType num_cols, row_major)    { return num_cols; }
  
  template <typename IndexType>
  __host__ __device__
  IndexType minor_dimension(IndexType num_rows, IndexType num_cols, column_major) { return num_rows; }

  template <typename IndexType>
  __host__ __device__
  IndexType major_dimension(IndexType num_rows, IndexType num_cols, row_major)    { return num_rows; }
  
  template <typename IndexType>
  __host__ __device__
  IndexType major_dimension(IndexType num_rows, IndexType num_cols, column_major) { return num_cols; }

  // convert logical linear index into a logical (i,j) index
  template <typename IndexType>
  __host__ __device__
  IndexType linear_index_to_row_index(IndexType linear_index, IndexType num_rows, IndexType num_cols, row_major)    { return linear_index / num_cols; }
      
  template <typename IndexType>
  __host__ __device__
  IndexType linear_index_to_col_index(IndexType linear_index, IndexType num_rows, IndexType num_cols, row_major)    { return linear_index % num_cols; }
  
  template <typename IndexType>
  __host__ __device__
  IndexType linear_index_to_row_index(IndexType linear_index, IndexType num_rows, IndexType num_cols, column_major)    { return linear_index % num_rows; }
      
  template <typename IndexType>
  __host__ __device__
  IndexType linear_index_to_col_index(IndexType linear_index, IndexType num_rows, IndexType num_cols, column_major)    { return linear_index / num_rows; }

  // convert a logical (i,j) index into a physical linear index
  template <typename IndexType>
  __host__ __device__
  IndexType index_of(IndexType i, IndexType j, IndexType pitch, row_major)    { return i * pitch + j; }
      
  template <typename IndexType>
  __host__ __device__
  IndexType index_of(IndexType i, IndexType j, IndexType pitch, column_major) { return j * pitch + i; }
  
  template <typename IndexType, typename Orientation>
  __host__ __device__
  IndexType logical_to_physical(IndexType linear_index, IndexType num_rows, IndexType num_cols, IndexType pitch, Orientation)
  {
    IndexType i = linear_index_to_row_index(linear_index, num_rows, num_cols, Orientation());
    IndexType j = linear_index_to_col_index(linear_index, num_rows, num_cols, Orientation());
    
    return index_of(i, j, pitch, Orientation());
  }

  // convert logical linear index in the source into a physical linear index in the destination
  template <typename IndexType, typename Orientation1, typename Orientation2>
  __host__ __device__
  IndexType logical_to_other_physical(IndexType linear_index, IndexType num_rows, IndexType num_cols, IndexType pitch, Orientation1, Orientation2)
  { 
    IndexType i = linear_index_to_row_index(linear_index, num_rows, num_cols, Orientation1());
    IndexType j = linear_index_to_col_index(linear_index, num_rows, num_cols, Orientation1());

    return index_of(i, j, pitch, Orientation2());
  }
  
  // functors 
  template <typename IndexType, typename Orientation>
  struct logical_to_physical_functor : public thrust::unary_function<IndexType,IndexType>
  {
    IndexType num_rows, num_cols, pitch;
  
    logical_to_physical_functor(IndexType num_rows, IndexType num_cols, IndexType pitch)
      : num_rows(num_rows), num_cols(num_cols), pitch(pitch) {}
  
    __host__ __device__
    IndexType operator()(const IndexType i) const
    {
      return logical_to_physical(i, num_rows, num_cols, pitch, Orientation());
    }
  };

  template <typename IndexType, typename Orientation1, typename Orientation2>
  struct logical_to_other_physical_functor : public thrust::unary_function<IndexType,IndexType>
  {
    IndexType num_rows, num_cols, pitch;
  
    logical_to_other_physical_functor(IndexType num_rows, IndexType num_cols, IndexType pitch)
      : num_rows(num_rows), num_cols(num_cols), pitch(pitch) {}
  
    __host__ __device__
    IndexType operator()(const IndexType i) const
    {
      return logical_to_other_physical(i, num_rows, num_cols, pitch, Orientation1(), Orientation2());
    }
  };
  
  template <typename Iterator, class Orientation>
  struct row_view {};

  template <typename Iterator>
  struct row_view<Iterator,cusp::row_major> : public cusp::array1d_view<Iterator>
  {
    template <typename Array>
    row_view(Array& A, size_t n)
      : cusp::array1d_view<Iterator>(A.values.begin() + A.pitch * n,
                                     A.values.begin() + A.pitch * n + A.num_cols)
    {}
  };
  
  template <typename Iterator, class Orientation>
  struct column_view {};

  template <typename Iterator>
  struct column_view<Iterator,cusp::column_major> : public cusp::array1d_view<Iterator>
  {
    template <typename Array>
    column_view(Array& A, size_t n)
      : cusp::array1d_view<Iterator>(A.values.begin() + A.pitch * n,
                                     A.values.begin() + A.pitch * n + A.num_rows)
    {}
  };
} // end namespace detail

// TODO document mapping of (i,j) onto values[pitch * i + j] or values[pitch * j + i]
// TODO document that array2d operations will try to respect .pitch of destination argument

/*! \addtogroup arrays Arrays
 */

/*! \addtogroup array_containers Array Containers
 *  \ingroup arrays
 *  \{
 */

/*! \p array2d : One-dimensional array container
 * 
 * \tparam T value_type of the array
 * \tparam MemorySpace memory space of the array (cusp::host_memory or cusp::device_memory)
 * \tparam Orientation orientation of the array (cusp::row_major or cusp::column_major)
 *
 * \TODO example
 */
template<typename ValueType, class MemorySpace, class Orientation = cusp::row_major>
class array2d : public cusp::detail::matrix_base<int,ValueType,MemorySpace,cusp::array2d_format>
{
  typedef typename cusp::detail::matrix_base<int,ValueType,MemorySpace,cusp::array2d_format> Parent;

  public:
  typedef Orientation orientation;

  template<typename MemorySpace2>
    struct rebind { typedef cusp::array2d<ValueType, MemorySpace2, Orientation> type; };
  
  typedef typename cusp::array1d<ValueType, MemorySpace> values_array_type;

  /*! equivalent container type
   */
  typedef typename cusp::array2d<ValueType, MemorySpace, Orientation> container;
  
  /*! equivalent view type
   */
  typedef typename cusp::array2d_view<typename values_array_type::view, Orientation> view;
  
  /*! equivalent const_view type
   */
  typedef typename cusp::array2d_view<typename values_array_type::const_view, Orientation> const_view;

  /*! array1d_view of a single row
   */
  typedef typename cusp::detail::row_view<typename values_array_type::iterator,Orientation> row_view;
  
  /*! array1d_view of a single column
   */
  typedef typename cusp::detail::column_view<typename values_array_type::iterator,Orientation> column_view;

  values_array_type values;

  // minor_dimension + padding
  size_t pitch;

  // construct empty matrix
  array2d()
    : Parent(), pitch(0), values(0) {}

  // construct matrix with given shape and number of entries
  array2d(size_t num_rows, size_t num_cols)
    : Parent(num_rows, num_cols, num_rows * num_cols),
      pitch(cusp::detail::minor_dimension(num_rows, num_cols, orientation())),
      values(num_rows * num_cols) {}

  // construct matrix with given shape and number of entries and fill with a given value
  array2d(size_t num_rows, size_t num_cols, const ValueType& value)
    : Parent(num_rows, num_cols, num_rows * num_cols),
      pitch(cusp::detail::minor_dimension(num_rows, num_cols, orientation())),
      values(num_rows * num_cols, value) {}

  // construct from another matrix
  template <typename MatrixType>
  array2d(const MatrixType& matrix);

  typename values_array_type::reference operator()(const size_t i, const size_t j)
  { 
    return values[cusp::detail::index_of(i, j, pitch, orientation())];
  }

  typename values_array_type::const_reference operator()(const size_t i, const size_t j) const
  { 
    return values[cusp::detail::index_of(i, j, pitch, orientation())];
  }

  void resize(size_t num_rows, size_t num_cols, size_t pitch)
  {
    if (pitch < cusp::detail::minor_dimension(num_rows, num_cols, orientation()))
      throw cusp::invalid_input_exception("pitch cannot be less than minor dimension");

    values.resize(pitch * cusp::detail::major_dimension(num_rows, num_cols, orientation()));

    this->num_rows    = num_rows;
    this->num_cols    = num_cols;
    this->pitch       = pitch; 
    this->num_entries = num_rows * num_cols;
  }

  void resize(size_t num_rows, size_t num_cols)
  {
    // preserve .pitch if possible
    if (this->num_rows == num_rows && this->num_cols == num_cols)
      return;

    resize(num_rows, num_cols, cusp::detail::minor_dimension(num_rows, num_cols, orientation()));
  }

  void swap(array2d& matrix)
  {
    Parent::swap(matrix);
    thrust::swap(this->pitch, matrix.pitch);
    values.swap(matrix.values);
  }

  row_view row(size_t i)
  {
    return row_view(*this, i);
  }
  
  column_view column(size_t i)
  {
    return column_view(*this, i);
  }
  
  array2d& operator=(const array2d& matrix);

  template <typename MatrixType>
  array2d& operator=(const MatrixType& matrix);

}; // class array2d
/*! \}
 */
  
/*! \addtogroup array_views Array Views
 *  \ingroup arrays
 *  \{
 */

/*! \p array2d_view : One-dimensional array view
 *
 * \tparam Array Underlying one-dimensional array view
 * \tparam Orientation orientation of the array (cusp::row_major or cusp::column_major)
 *
 * \TODO example
 */
template<typename Array, class Orientation = cusp::row_major>
class array2d_view : public cusp::detail::matrix_base<int, typename Array::value_type,typename Array::memory_space, cusp::array2d_format>
{
  typedef cusp::detail::matrix_base<int, typename Array::value_type,typename Array::memory_space, cusp::array2d_format> Parent;
  public:
  typedef Orientation orientation;

  typedef Array values_array_type;

  values_array_type values;
  
  /*! equivalent container type
   */
  typedef typename cusp::array2d<typename Parent::value_type, typename Parent::memory_space, Orientation> container;
  
  /*! equivalent view type
   */
  typedef typename cusp::array2d_view<Array, Orientation> view;

  /*! array1d_view of a single row
   */
  typedef typename cusp::detail::row_view<typename values_array_type::iterator,Orientation> row_view;
  
  /*! array1d_view of a single column
   */
  typedef typename cusp::detail::column_view<typename values_array_type::iterator,Orientation> column_view;
  
  // minor_dimension + padding
  size_t pitch;

  // construct empty view
  array2d_view(void)
    : Parent(), values(0), pitch(0) {}

  array2d_view(const array2d_view& a)
    : Parent(a), values(a.values), pitch(a.pitch) {}

  // TODO handle different Orientation (pitch = major)
  //template <typename Array2, typename Orientation2>
  //array2d_view(const array2d_view<Array2,Orientation2>& A)
  
  // TODO check values.size()

  // construct from array2d container
  array2d_view(      array2d<typename Parent::value_type, typename Parent::memory_space, orientation>& a)
    : Parent(a), values(a.values), pitch(a.pitch) {}

  array2d_view(const array2d<typename Parent::value_type, typename Parent::memory_space, orientation>& a)
    : Parent(a), values(a.values), pitch(a.pitch) {}

  template <typename Array2>
  array2d_view(size_t num_rows, size_t num_cols, size_t pitch, Array2& values)
   : Parent(num_rows, num_cols, num_rows * num_cols), pitch(pitch), values(values) {}
  
  template <typename Array2>
  array2d_view(size_t num_rows, size_t num_cols, size_t pitch, const Array2& values)
   : Parent(num_rows, num_cols, num_rows * num_cols), pitch(pitch), values(values) {}

  typename values_array_type::reference operator()(const size_t i, const size_t j) const
  { 
    return values[detail::index_of(i, j, pitch, orientation())];
  }

  void resize(size_t num_rows, size_t num_cols, size_t pitch)
  {
    if (pitch < cusp::detail::minor_dimension(num_rows, num_cols, orientation()))
      throw cusp::invalid_input_exception("pitch cannot be less than minor dimension");

    values.resize(pitch * cusp::detail::major_dimension(num_rows, num_cols, orientation()));

    this->num_rows    = num_rows;
    this->num_cols    = num_cols;
    this->pitch       = pitch; 
    this->num_entries = num_rows * num_cols;
  }

  void resize(size_t num_rows, size_t num_cols)
  {
    // preserve .pitch if possible
    if (this->num_rows == num_rows && this->num_cols == num_cols)
      return;

    resize(num_rows, num_cols, cusp::detail::minor_dimension(num_rows, num_cols, orientation()));
  }
  
  row_view row(size_t i) const
  {
    return row_view(*this, i);
  }
  
  column_view column(size_t i) const
  {
    return column_view(*this, i);
  }
}; // class array2d_view



template <typename Iterator, typename Orientation>
array2d_view<typename cusp::array1d_view<Iterator>,Orientation>
make_array2d_view(size_t num_rows, size_t num_cols, size_t pitch, const cusp::array1d_view<Iterator>& values, Orientation)
{
  return array2d_view<typename cusp::array1d_view<Iterator>,Orientation>(num_rows, num_cols, pitch, values);
}
  
template <typename Array, typename Orientation>
array2d_view<Array,Orientation>
make_array2d_view(const array2d_view<Array, Orientation>& a)
{
  return array2d_view<Array,Orientation>(a);
}

template<typename ValueType, class MemorySpace, class Orientation>
array2d_view<typename cusp::array1d_view<typename cusp::array1d<ValueType,MemorySpace>::iterator >, Orientation>
make_array2d_view(cusp::array2d<ValueType,MemorySpace,Orientation>& a)
{
  return cusp::make_array2d_view(a.num_rows, a.num_cols, a.pitch, cusp::make_array1d_view(a.values), Orientation());
}

template<typename ValueType, class MemorySpace, class Orientation>
array2d_view<typename cusp::array1d_view<typename cusp::array1d<ValueType,MemorySpace>::const_iterator >, Orientation>
make_array2d_view(const cusp::array2d<ValueType,MemorySpace,Orientation>& a)
{
  return cusp::make_array2d_view(a.num_rows, a.num_cols, a.pitch, cusp::make_array1d_view(a.values), Orientation());
}
/*! \}
 */

} // end namespace cusp

#include <cusp/detail/array2d.inl>

