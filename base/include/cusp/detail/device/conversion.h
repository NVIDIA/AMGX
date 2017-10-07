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


#pragma once

#include <cusp/blas.h>
#include <cusp/copy.h>
#include <cusp/format.h>
#include <cusp/coo_matrix.h>
#include <cusp/csr_matrix.h>

#include <cusp/detail/format_utils.h>
#include <cusp/detail/device/conversion_utils.h>
#include <cusp/detail/host/convert.h>

#include <thrust/count.h>
#include <thrust/gather.h>
#include <thrust/inner_product.h>
#include <thrust/replace.h>
#include <thrust/scan.h>
#include <thrust/scatter.h>
#include <thrust/sequence.h>
#include <thrust/tuple.h>

#include <thrust/iterator/constant_iterator.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/zip_iterator.h>

#include <cassert>

namespace cusp
{
namespace detail
{
namespace device
{

// Native Device Conversion Functions
// COO <- CSR
//     <- ELL
//     <- DIA
//     <- HYB
// CSR <- COO
//     <- ELL
//     <- DIA
// ELL <- CSR
//     <- COO
// DIA <- CSR
//     <- COO
// HYB <- CSR
//     <- COO
//     <- ELL

template <typename IndexType>
struct is_valid_ell_index
{
  const IndexType num_rows;

  is_valid_ell_index(const IndexType num_rows)
    : num_rows(num_rows) {}

  template <typename Tuple>
    __host__ __device__
  bool operator()(const Tuple& t) const
  {
    const IndexType i = thrust::get<0>(t);
    const IndexType j = thrust::get<1>(t);

    return i < num_rows && j != IndexType(-1);
  }
};

template <typename IndexType, typename ValueType>
struct is_valid_coo_index
{
  const IndexType num_rows;
  const IndexType num_cols;

  is_valid_coo_index(const IndexType num_rows, const IndexType num_cols)
    : num_rows(num_rows), num_cols(num_cols) {}

  template <typename Tuple>
    __host__ __device__
  bool operator()(const Tuple& t) const
  {
    const IndexType i = thrust::get<0>(t);
    const IndexType j = thrust::get<1>(t);
    const ValueType value = thrust::get<2>(t);

    return ( i > IndexType(-1) && i < num_rows ) && 
	   ( j > IndexType(-1) && j < num_cols ) && 
	   ( value != ValueType(0) ) ;
  }
};

template <typename T>
struct transpose_index_functor : public thrust::unary_function<T,T>
{
  const T num_entries_per_row;
  const T pitch;

  transpose_index_functor(const T pitch, const T num_entries_per_row)
    : num_entries_per_row(num_entries_per_row), pitch(pitch) {}

    __host__ __device__
  T operator()(const T& n) const
  {
    return pitch * (n % num_entries_per_row) + n / num_entries_per_row;
  }
};

template <typename IndexType>
struct occupied_diagonal_functor
{
  typedef IndexType result_type;

  const   IndexType num_rows;

  occupied_diagonal_functor(const IndexType num_rows)
    : num_rows(num_rows) {}

  template <typename Tuple>
    __host__ __device__
  IndexType operator()(const Tuple& t) const
  {
    const IndexType i = thrust::get<0>(t);
    const IndexType j = thrust::get<1>(t);

    return num_rows-i+j;
  }
};

template <typename IndexType>
struct diagonal_index_functor : public thrust::unary_function<IndexType,IndexType>
{
  const IndexType pitch;

  diagonal_index_functor(const IndexType pitch)
    : pitch(pitch) {}

  template <typename Tuple>
    __host__ __device__
  IndexType operator()(const Tuple& t) const
  {
    const IndexType row  = thrust::get<0>(t);
    const IndexType diag = thrust::get<1>(t);

    return (diag * pitch) + row;
  }
};

template <typename T>
struct sum_tuple_functor : public thrust::unary_function<T,T>
{
  template <typename Tuple>
    __host__ __device__
  T operator()(const Tuple& t) const
  {
    const T offset  = thrust::get<0>(t);
    const T modulus = thrust::get<1>(t);

    return offset + modulus;
  }
};

template <typename T>
struct multiply_value : public thrust::unary_function<T,T>
{
  const T value;

  multiply_value(const T value)
    : value(value) {}

    __host__ __device__
  T operator()(const T& x) const
  {
    return x * value;
  }
};

template <typename T>
struct divide_value : public thrust::unary_function<T,T>
{
  const T value;

  divide_value(const T value)
    : value(value) {}

    __host__ __device__
  T operator()(const T& x) const
  {
    return x / value;
  }
};

template <typename T>
struct modulus_value : public thrust::unary_function<T,T>
{
  const T value;

  modulus_value(const T value)
    : value(value) {}

    __host__ __device__
  T operator()(const T& x) const
  {
    return x % value;
  }
};

template <typename T>
struct greater_than_or_equal_to
{
  const T num;

  greater_than_or_equal_to(const T num)
    : num(num) {}

  __host__ __device__ bool operator()(const T &x) const {return x >= num;}
};

template <typename T>
struct less_than
{
  const T num;

  less_than(const T num)
    : num(num) {}

  __host__ __device__ bool operator()(const T &x) const {return x < num;}
};

template <typename IndexType>
struct is_positive
{
  __host__ __device__
  bool operator()(const IndexType x)
  {
    return x > 0;
  }
};

/////////
// COO //
/////////
template <typename Matrix1, typename Matrix2>
void csr_to_coo(const Matrix1& src, Matrix2& dst)
{
    dst.resize(src.num_rows, src.num_cols, src.num_entries);

    cusp::detail::offsets_to_indices(src.row_offsets, dst.row_indices);
    cusp::copy(src.column_indices, dst.column_indices);
    cusp::copy(src.values,         dst.values);
}


template <typename Matrix1, typename Matrix2>
void ell_to_coo(const Matrix1& src, Matrix2& dst)
{
   typedef typename Matrix1::index_type IndexType;
   
   const IndexType pitch               = src.column_indices.pitch;
   const IndexType num_entries_per_row = src.column_indices.num_cols;

   // define types used to programatically generate row_indices
   typedef typename thrust::counting_iterator<IndexType> IndexIterator;
   typedef typename thrust::transform_iterator<modulus_value<IndexType>, IndexIterator> RowIndexIterator;

   RowIndexIterator row_indices_begin(IndexIterator(0), modulus_value<IndexType>(pitch));

   // compute true number of nonzeros in ELL
   const IndexType num_entries = 
     thrust::count_if
      (thrust::make_zip_iterator(thrust::make_tuple(row_indices_begin, src.column_indices.values.begin())),
       thrust::make_zip_iterator(thrust::make_tuple(row_indices_begin, src.column_indices.values.begin())) + src.column_indices.values.size(),
       is_valid_ell_index<IndexType>(src.num_rows));

   // allocate output storage
   dst.resize(src.num_rows, src.num_cols, num_entries);

   // copy valid entries to COO format
   thrust::copy_if
     (thrust::make_permutation_iterator(thrust::make_zip_iterator(thrust::make_tuple(row_indices_begin, src.column_indices.values.begin(), src.values.values.begin())), thrust::make_transform_iterator(thrust::counting_iterator<IndexType>(0), transpose_index_functor<IndexType>(pitch, num_entries_per_row))),
      thrust::make_permutation_iterator(thrust::make_zip_iterator(thrust::make_tuple(row_indices_begin, src.column_indices.values.begin(), src.values.values.begin())), thrust::make_transform_iterator(thrust::counting_iterator<IndexType>(0), transpose_index_functor<IndexType>(pitch, num_entries_per_row))) + src.column_indices.values.size(),
      thrust::make_zip_iterator(thrust::make_tuple(dst.row_indices.begin(), dst.column_indices.begin(), dst.values.begin())),
      is_valid_ell_index<IndexType>(src.num_rows));
}

template <typename Matrix1, typename Matrix2>
void dia_to_coo(const Matrix1& src, Matrix2& dst)
{
   typedef typename Matrix1::index_type IndexType;
   typedef typename Matrix1::value_type ValueType;
   
   // allocate output storage
   dst.resize(src.num_rows, src.num_cols, src.num_entries);

   if( src.num_entries == 0 ) return;

   const IndexType pitch = src.values.pitch;
   const size_t num_entries   = src.values.values.size();
   const size_t num_diagonals = src.diagonal_offsets.size();

   // define types used to programatically generate row_indices
   typedef typename thrust::counting_iterator<IndexType> IndexIterator;
   typedef typename thrust::transform_iterator<modulus_value<IndexType>, IndexIterator> RowIndexIterator;

   RowIndexIterator row_indices_begin(IndexIterator(0), modulus_value<IndexType>(pitch));

   // define types used to programatically generate column_indices
   typedef typename thrust::device_vector<IndexType>::const_iterator ConstElementIterator;
   typedef typename thrust::transform_iterator<divide_value<IndexType>, IndexIterator> DivideIterator;
   typedef typename thrust::permutation_iterator<ConstElementIterator,DivideIterator> OffsetsPermIterator;
   typedef typename thrust::tuple<OffsetsPermIterator, RowIndexIterator> IteratorTuple;
   typedef typename thrust::zip_iterator<IteratorTuple> ZipIterator;
   typedef typename thrust::transform_iterator<sum_tuple_functor<IndexType>, ZipIterator> ColumnIndexIterator;

   DivideIterator gather_indices_begin(IndexIterator(0), divide_value<IndexType>(pitch));
   OffsetsPermIterator offsets_begin(src.diagonal_offsets.begin(), gather_indices_begin);
   ZipIterator offset_modulus_tuple(thrust::make_tuple(offsets_begin, row_indices_begin));
   ColumnIndexIterator column_indices_begin(offset_modulus_tuple, sum_tuple_functor<IndexType>());

   // copy valid entries to COO format
   //thrust::copy_if
   //  (thrust::make_permutation_iterator(thrust::make_zip_iterator(thrust::make_tuple(row_indices_begin, column_indices_begin, src.values.values.begin())), thrust::make_transform_iterator(thrust::counting_iterator<IndexType>(0), transpose_index_functor<IndexType>(pitch, num_diagonals))),
   //   thrust::make_permutation_iterator(thrust::make_zip_iterator(thrust::make_tuple(row_indices_begin, column_indices_begin, src.values.values.begin())), thrust::make_transform_iterator(thrust::counting_iterator<IndexType>(0), transpose_index_functor<IndexType>(pitch, num_diagonals))) + num_entries,
   //   thrust::make_zip_iterator(thrust::make_tuple(dst.row_indices.begin(), dst.column_indices.begin(), dst.values.begin())),
   //   is_valid_coo_index<IndexType,ValueType>(src.num_rows,src.num_cols));
   {
     // TODO remove this when copy_if can accept more parameters
     cusp::array1d<IndexType, cusp::device_memory> temp0(num_entries);
     cusp::array1d<IndexType, cusp::device_memory> temp1(num_entries);
     cusp::array1d<ValueType, cusp::device_memory> temp2(num_entries);
     thrust::copy(thrust::make_permutation_iterator(thrust::make_zip_iterator(thrust::make_tuple(row_indices_begin, column_indices_begin, src.values.values.begin())), thrust::make_transform_iterator(thrust::counting_iterator<IndexType>(0), transpose_index_functor<IndexType>(pitch, num_diagonals))),
                  thrust::make_permutation_iterator(thrust::make_zip_iterator(thrust::make_tuple(row_indices_begin, column_indices_begin, src.values.values.begin())), thrust::make_transform_iterator(thrust::counting_iterator<IndexType>(0), transpose_index_functor<IndexType>(pitch, num_diagonals))) + num_entries,
                  thrust::make_zip_iterator(thrust::make_tuple(temp0.begin(), temp1.begin(), temp2.begin())));
     thrust::copy_if
       (thrust::make_zip_iterator(thrust::make_tuple(temp0.begin(), temp1.begin(), temp2.begin())),
        thrust::make_zip_iterator(thrust::make_tuple(temp0.begin(), temp1.begin(), temp2.begin())) + num_entries,
        thrust::make_zip_iterator(thrust::make_tuple(dst.row_indices.begin(), dst.column_indices.begin(), dst.values.begin())),
        is_valid_coo_index<IndexType,ValueType>(src.num_rows,src.num_cols));
   }
}

template <typename Matrix1, typename Matrix2>
void hyb_to_coo(const Matrix1& src, Matrix2& dst)
{
   typedef typename Matrix1::coo_matrix_type  CooMatrixType;
   typedef typename CooMatrixType::container  CooMatrix;

   // convert ell portion to coo
   CooMatrix temp;
   ell_to_coo(src.ell, temp);
   
   // resize output
   dst.resize(src.num_rows, src.num_cols, temp.num_entries + src.coo.num_entries);

   // merge coo matrices together
   thrust::copy(temp.row_indices.begin(),    temp.row_indices.end(),    dst.row_indices.begin());
   thrust::copy(temp.column_indices.begin(), temp.column_indices.end(), dst.column_indices.begin());
   thrust::copy(temp.values.begin(),         temp.values.end(),         dst.values.begin());
   thrust::copy(src.coo.row_indices.begin(),    src.coo.row_indices.end(),    dst.row_indices.begin()    + temp.num_entries);
   thrust::copy(src.coo.column_indices.begin(), src.coo.column_indices.end(), dst.column_indices.begin() + temp.num_entries);
   thrust::copy(src.coo.values.begin(),         src.coo.values.end(),         dst.values.begin()         + temp.num_entries);

   if (temp.num_entries > 0 && src.coo.num_entries > 0)
     cusp::detail::sort_by_row_and_column(dst.row_indices, dst.column_indices, dst.values); 
}
   


/////////
// CSR //
/////////
template <typename Matrix1, typename Matrix2>
void coo_to_csr(const Matrix1& src, Matrix2& dst)
{
    dst.resize(src.num_rows, src.num_cols, src.num_entries);

    cusp::detail::indices_to_offsets(src.row_indices, dst.row_offsets);
    cusp::copy(src.column_indices, dst.column_indices);
    cusp::copy(src.values,         dst.values);
}

template <typename Matrix1, typename Matrix2>
void ell_to_csr(const Matrix1& src, Matrix2& dst)
{
   typedef typename Matrix1::index_type IndexType;
   
   const IndexType pitch               = src.column_indices.pitch;
   const IndexType num_entries_per_row = src.column_indices.num_cols;

   // define types used to programatically generate row_indices
   typedef typename thrust::counting_iterator<IndexType> IndexIterator;
   typedef typename thrust::transform_iterator<modulus_value<IndexType>, IndexIterator> RowIndexIterator;

   RowIndexIterator row_indices_begin(IndexIterator(0), modulus_value<IndexType>(pitch));

   // compute true number of nonzeros in ELL
   const IndexType num_entries = 
     thrust::count_if
      (thrust::make_zip_iterator(thrust::make_tuple(row_indices_begin, src.column_indices.values.begin())),
       thrust::make_zip_iterator(thrust::make_tuple(row_indices_begin, src.column_indices.values.begin())) + src.column_indices.values.size(),
       is_valid_ell_index<IndexType>(src.num_rows));

   // allocate output storage
   dst.resize(src.num_rows, src.num_cols, num_entries);

   // create temporary row_indices array to capture valid ELL row indices
   cusp::array1d<IndexType, cusp::device_memory> row_indices(num_entries);

   // copy valid entries to mixed COO/CSR format
   thrust::copy_if
     (thrust::make_permutation_iterator(thrust::make_zip_iterator(thrust::make_tuple(row_indices_begin, src.column_indices.values.begin(), src.values.values.begin())), thrust::make_transform_iterator(thrust::counting_iterator<IndexType>(0), transpose_index_functor<IndexType>(pitch, num_entries_per_row))),
      thrust::make_permutation_iterator(thrust::make_zip_iterator(thrust::make_tuple(row_indices_begin, src.column_indices.values.begin(), src.values.values.begin())), thrust::make_transform_iterator(thrust::counting_iterator<IndexType>(0), transpose_index_functor<IndexType>(pitch, num_entries_per_row))) + src.column_indices.values.size(),
      thrust::make_zip_iterator(thrust::make_tuple(row_indices.begin(), dst.column_indices.begin(), dst.values.begin())),
      is_valid_ell_index<IndexType>(src.num_rows));

   // convert COO row_indices to CSR row_offsets
   cusp::detail::indices_to_offsets(row_indices, dst.row_offsets);
}

template <typename Matrix1, typename Matrix2>
void dia_to_csr(const Matrix1& src, Matrix2& dst)
{
   typedef typename Matrix1::index_type IndexType;
   typedef typename Matrix1::value_type ValueType;
   
   // allocate output storage
   dst.resize(src.num_rows, src.num_cols, src.num_entries);

   if( src.num_entries == 0 ) return;

   const IndexType pitch = src.values.pitch;
   const size_t num_entries   = src.values.values.size();
   const size_t num_diagonals = src.diagonal_offsets.size();

   // define types used to programatically generate row_indices
   typedef typename thrust::counting_iterator<IndexType> IndexIterator;
   typedef typename thrust::transform_iterator<modulus_value<IndexType>, IndexIterator> RowIndexIterator;

   RowIndexIterator row_indices_begin(IndexIterator(0), modulus_value<IndexType>(pitch));

   // define types used to programatically generate column_indices
   typedef typename thrust::device_vector<IndexType>::const_iterator ConstElementIterator;
   typedef typename thrust::transform_iterator<divide_value<IndexType>, IndexIterator> DivideIterator;
   typedef typename thrust::permutation_iterator<ConstElementIterator,DivideIterator> OffsetsPermIterator;
   typedef typename thrust::tuple<OffsetsPermIterator, RowIndexIterator> IteratorTuple;
   typedef typename thrust::zip_iterator<IteratorTuple> ZipIterator;
   typedef typename thrust::transform_iterator<sum_tuple_functor<IndexType>, ZipIterator> ColumnIndexIterator;

   DivideIterator gather_indices_begin(IndexIterator(0), divide_value<IndexType>(pitch));
   OffsetsPermIterator offsets_begin(src.diagonal_offsets.begin(), gather_indices_begin);
   ZipIterator offset_modulus_tuple(thrust::make_tuple(offsets_begin, row_indices_begin));
   ColumnIndexIterator column_indices_begin(offset_modulus_tuple, sum_tuple_functor<IndexType>());

   cusp::array1d<IndexType, cusp::device_memory> row_indices(src.num_entries);

   // copy valid entries to COO format
   {
     // TODO remove this when copy_if can accept more parameters
     cusp::array1d<IndexType, cusp::device_memory> temp0(num_entries);
     cusp::array1d<IndexType, cusp::device_memory> temp1(num_entries);
     cusp::array1d<ValueType, cusp::device_memory> temp2(num_entries);
     thrust::copy(thrust::make_permutation_iterator(thrust::make_zip_iterator(thrust::make_tuple(row_indices_begin, column_indices_begin, src.values.values.begin())), thrust::make_transform_iterator(thrust::counting_iterator<IndexType>(0), transpose_index_functor<IndexType>(pitch, num_diagonals))),
                  thrust::make_permutation_iterator(thrust::make_zip_iterator(thrust::make_tuple(row_indices_begin, column_indices_begin, src.values.values.begin())), thrust::make_transform_iterator(thrust::counting_iterator<IndexType>(0), transpose_index_functor<IndexType>(pitch, num_diagonals))) + num_entries,
                  thrust::make_zip_iterator(thrust::make_tuple(temp0.begin(), temp1.begin(), temp2.begin())));
     thrust::copy_if
       (thrust::make_zip_iterator(thrust::make_tuple(temp0.begin(), temp1.begin(), temp2.begin())),
        thrust::make_zip_iterator(thrust::make_tuple(temp0.begin(), temp1.begin(), temp2.begin())) + num_entries,
        thrust::make_zip_iterator(thrust::make_tuple(row_indices.begin(), dst.column_indices.begin(), dst.values.begin())),
        is_valid_coo_index<IndexType,ValueType>(src.num_rows,src.num_cols));
   }
   //thrust::copy_if
   //  (thrust::make_permutation_iterator(thrust::make_zip_iterator(thrust::make_tuple(row_indices_begin, column_indices_begin, src.values.values.begin())), thrust::make_transform_iterator(thrust::counting_iterator<IndexType>(0), transpose_index_functor<IndexType>(pitch, num_diagonals))),
   //   thrust::make_permutation_iterator(thrust::make_zip_iterator(thrust::make_tuple(row_indices_begin, column_indices_begin, src.values.values.begin())), thrust::make_transform_iterator(thrust::counting_iterator<IndexType>(0), transpose_index_functor<IndexType>(pitch, num_diagonals))) + num_entries,
   //   thrust::make_zip_iterator(thrust::make_tuple(row_indices.begin(), dst.column_indices.begin(), dst.values.begin())),
   //   is_valid_coo_index<IndexType,ValueType>(src.num_rows,src.num_cols));

    cusp::detail::indices_to_offsets( row_indices, dst.row_offsets );
}


/////////
// DIA //
/////////
template <typename Matrix1, typename Matrix2>
void coo_to_dia(const Matrix1& src, Matrix2& dst,
                const size_t alignment = 32)
{
    typedef typename Matrix2::index_type IndexType;
    typedef typename Matrix2::value_type ValueType;

    // compute number of occupied diagonals and enumerate them
    cusp::array1d<IndexType,cusp::device_memory> diag_map(src.num_entries);
    thrust::transform(thrust::make_zip_iterator( thrust::make_tuple( src.row_indices.begin(), src.column_indices.begin() ) ), 
		      thrust::make_zip_iterator( thrust::make_tuple( src.row_indices.end()  , src.column_indices.end() ) )  ,
		      diag_map.begin(),
		      occupied_diagonal_functor<IndexType>(src.num_rows)); 

    // place ones in diagonals array locations with occupied diagonals
    cusp::array1d<IndexType,cusp::device_memory> diagonals(src.num_rows+src.num_cols,IndexType(0));
    thrust::scatter(thrust::constant_iterator<IndexType>(1), 
		    thrust::constant_iterator<IndexType>(1)+src.num_entries, 
		    diag_map.begin(),
		    diagonals.begin());

    const IndexType num_diagonals = thrust::reduce(diagonals.begin(), diagonals.end());

    // allocate DIA structure
    dst.resize(src.num_rows, src.num_cols, src.num_entries, num_diagonals, alignment);

    // fill in values array
    thrust::fill(dst.values.values.begin(), dst.values.values.end(), ValueType(0));

    // fill in diagonal_offsets array
    thrust::copy_if(thrust::counting_iterator<IndexType>(0), 
		    thrust::counting_iterator<IndexType>(src.num_rows+src.num_cols),
		    diagonals.begin(),
		    dst.diagonal_offsets.begin(), 
		    is_positive<IndexType>()); 

    // replace shifted diagonals with index of diagonal in offsets array
    cusp::array1d<IndexType,cusp::host_memory> diagonal_offsets( dst.diagonal_offsets );
    for( IndexType num_diag = 0; num_diag < num_diagonals; num_diag++ )
	thrust::replace(diag_map.begin(), diag_map.end(), diagonal_offsets[num_diag], num_diag);

    // copy values to dst
    thrust::scatter(src.values.begin(), src.values.end(),
		    thrust::make_transform_iterator(
				thrust::make_zip_iterator( thrust::make_tuple( src.row_indices.begin(), diag_map.begin() ) ), 
				diagonal_index_functor<IndexType>(dst.values.pitch)), 
                    dst.values.values.begin());


    typedef typename cusp::array1d_view< thrust::constant_iterator<IndexType> > ConstantView;
    ConstantView constant_view(thrust::constant_iterator<IndexType>(dst.num_rows),
			       thrust::constant_iterator<IndexType>(dst.num_rows)+num_diagonals);
    cusp::blas::axpy(constant_view,
		     dst.diagonal_offsets,
		     IndexType(-1));
}

template <typename Matrix1, typename Matrix2>
void csr_to_dia(const Matrix1& src, Matrix2& dst,
                const size_t alignment = 32)
{
    typedef typename Matrix2::index_type IndexType;
    typedef typename Matrix2::value_type ValueType;

    // compute number of occupied diagonals and enumerate them
    cusp::array1d<IndexType,cusp::device_memory> row_indices(src.num_entries);
    cusp::detail::offsets_to_indices(src.row_offsets, row_indices);

    cusp::array1d<IndexType,cusp::device_memory> diag_map(src.num_entries);
    thrust::transform(thrust::make_zip_iterator( thrust::make_tuple( row_indices.begin(), src.column_indices.begin() ) ), 
		      thrust::make_zip_iterator( thrust::make_tuple( row_indices.end()  , src.column_indices.end() ) )  ,
		      diag_map.begin(),
		      occupied_diagonal_functor<IndexType>(src.num_rows)); 

    // place ones in diagonals array locations with occupied diagonals
    cusp::array1d<IndexType,cusp::device_memory> diagonals(src.num_rows+src.num_cols,IndexType(0));
    thrust::scatter(thrust::constant_iterator<IndexType>(1), 
		    thrust::constant_iterator<IndexType>(1)+src.num_entries, 
		    diag_map.begin(),
		    diagonals.begin());

    const IndexType num_diagonals = thrust::reduce(diagonals.begin(), diagonals.end());

    // allocate DIA structure
    dst.resize(src.num_rows, src.num_cols, src.num_entries, num_diagonals, alignment);

    // fill in values array
    thrust::fill(dst.values.values.begin(), dst.values.values.end(), ValueType(0));

    // fill in diagonal_offsets array
    thrust::copy_if(thrust::counting_iterator<IndexType>(0), 
		    thrust::counting_iterator<IndexType>(src.num_rows+src.num_cols),
		    diagonals.begin(),
		    dst.diagonal_offsets.begin(), 
		    is_positive<IndexType>()); 

    // replace shifted diagonals with index of diagonal in offsets array
    cusp::array1d<IndexType,cusp::host_memory> diagonal_offsets( dst.diagonal_offsets );
    for( IndexType num_diag = 0; num_diag < num_diagonals; num_diag++ )
	thrust::replace(diag_map.begin(), diag_map.end(), diagonal_offsets[num_diag], num_diag);

    // copy values to dst
    thrust::scatter(src.values.begin(), src.values.end(),
		    thrust::make_transform_iterator(
				thrust::make_zip_iterator( thrust::make_tuple( row_indices.begin(), diag_map.begin() ) ), 
				diagonal_index_functor<IndexType>(dst.values.pitch)), 
                    dst.values.values.begin());

    // shift diagonal_offsets by num_rows 

    typedef typename cusp::array1d_view< thrust::constant_iterator<IndexType> > ConstantView;
    ConstantView constant_view(thrust::constant_iterator<IndexType>(dst.num_rows),
			       thrust::constant_iterator<IndexType>(dst.num_rows)+num_diagonals);
    cusp::blas::axpy(constant_view,
		     dst.diagonal_offsets,
		     IndexType(-1));

}

/////////
// ELL //
/////////
template <typename Matrix1, typename Matrix2>
void coo_to_ell(const Matrix1& src, Matrix2& dst,
                const size_t num_entries_per_row, const size_t alignment = 32)
{
  typedef typename Matrix2::index_type IndexType;
  typedef typename Matrix2::value_type ValueType;

  if (src.num_entries == 0)
  {
    dst.resize(src.num_rows, src.num_cols, src.num_entries, 0);
    return;
  }

  // allocate output storage
  dst.resize(src.num_rows, src.num_cols, src.num_entries, num_entries_per_row, alignment);

  // compute permutation from COO index to ELL index
  // first enumerate the entries within each row, e.g. [0, 1, 2, 0, 1, 2, 3, ...]
  cusp::array1d<IndexType, cusp::device_memory> permutation(src.num_entries);

  thrust::exclusive_scan_by_key(src.row_indices.begin(), src.row_indices.end(),
                                thrust::constant_iterator<IndexType>(1),
                                permutation.begin(),
                                IndexType(0));
 
  // next, scale by pitch and add row index
  cusp::blas::axpby(permutation, src.row_indices,
                    permutation,
                    IndexType(dst.column_indices.pitch),
                    IndexType(1));

  // fill output with padding
  thrust::fill(dst.column_indices.values.begin(), dst.column_indices.values.end(), IndexType(-1));
  thrust::fill(dst.values.values.begin(),         dst.values.values.end(),         ValueType(0));

  // scatter COO entries to ELL
  thrust::scatter(src.column_indices.begin(), src.column_indices.end(),
                  permutation.begin(),
                  dst.column_indices.values.begin());
  thrust::scatter(src.values.begin(), src.values.end(),
                  permutation.begin(),
                  dst.values.values.begin());
}

template <typename Matrix1, typename Matrix2>
void csr_to_ell(const Matrix1& src, Matrix2& dst,
                const size_t num_entries_per_row, const size_t alignment = 32)
{
  typedef typename Matrix2::index_type IndexType;
  typedef typename Matrix2::value_type ValueType;

  // allocate output storage
  dst.resize(src.num_rows, src.num_cols, src.num_entries, num_entries_per_row, alignment);

  // expand row offsets into row indices
  cusp::array1d<IndexType, cusp::device_memory> row_indices(src.num_entries);
  cusp::detail::offsets_to_indices(src.row_offsets, row_indices);

  // compute permutation from CSR index to ELL index
  // first enumerate the entries within each row, e.g. [0, 1, 2, 0, 1, 2, 3, ...]
  cusp::array1d<IndexType, cusp::device_memory> permutation(src.num_entries);
  thrust::exclusive_scan_by_key(row_indices.begin(), row_indices.end(),
                                thrust::constant_iterator<IndexType>(1),
                                permutation.begin(),
                                IndexType(0));
  
  // next, scale by pitch and add row index
  cusp::blas::axpby(permutation, row_indices,
                    permutation,
                    IndexType(dst.column_indices.pitch),
                    IndexType(1));

  // fill output with padding
  thrust::fill(dst.column_indices.values.begin(), dst.column_indices.values.end(), IndexType(-1));
  thrust::fill(dst.values.values.begin(),         dst.values.values.end(),         ValueType(0));

  // scatter CSR entries to ELL
  thrust::scatter(src.column_indices.begin(), src.column_indices.end(),
                  permutation.begin(),
                  dst.column_indices.values.begin());
  thrust::scatter(src.values.begin(), src.values.end(),
                  permutation.begin(),
                  dst.values.values.begin());
}


/////////
// HYB //
/////////

template <typename Matrix1, typename Matrix2>
void coo_to_hyb(const Matrix1& src, Matrix2& dst,
                const size_t num_entries_per_row, const size_t alignment = 32)
{
  typedef typename Matrix2::index_type IndexType;
  typedef typename Matrix2::value_type ValueType;

  cusp::array1d<IndexType, cusp::device_memory> indices(src.num_entries);
  thrust::exclusive_scan_by_key(src.row_indices.begin(), src.row_indices.end(),
                                thrust::constant_iterator<IndexType>(1),
                                indices.begin(),
                                IndexType(0));

  size_t num_coo_entries = thrust::count_if(indices.begin(), indices.end(), greater_than_or_equal_to<size_t>(num_entries_per_row));
  size_t num_ell_entries = src.num_entries - num_coo_entries;

  // allocate output storage
  dst.resize(src.num_rows, src.num_cols, num_ell_entries, num_coo_entries, num_entries_per_row, alignment);

  // fill output with padding
  thrust::fill(dst.ell.column_indices.values.begin(), dst.ell.column_indices.values.end(), IndexType(-1));
  thrust::fill(dst.ell.values.values.begin(),         dst.ell.values.values.end(),         ValueType(0));

  // write tail of each row to COO portion
  thrust::copy_if
      (thrust::make_zip_iterator( thrust::make_tuple( src.row_indices.begin(), src.column_indices.begin(), src.values.begin() ) ),
       thrust::make_zip_iterator( thrust::make_tuple( src.row_indices.end()  , src.column_indices.end()  , src.values.end()   ) ),
       indices.begin(),
       thrust::make_zip_iterator( thrust::make_tuple( dst.coo.row_indices.begin(), dst.coo.column_indices.begin(), dst.coo.values.begin() ) ),
       greater_than_or_equal_to<size_t>(num_entries_per_row) );

  assert(dst.ell.column_indices.pitch == dst.ell.values.pitch);

  size_t pitch = dst.ell.column_indices.pitch;

  // next, scale by pitch and add row index
  cusp::blas::axpby(indices, src.row_indices,
                    indices,
                    IndexType(pitch),
                    IndexType(1));

  // scatter COO entries to ELL
  thrust::scatter_if(src.column_indices.begin(), src.column_indices.end(),
                     indices.begin(),
                     indices.begin(),
                     dst.ell.column_indices.values.begin(),
                     less_than<size_t>(dst.ell.column_indices.values.size()));
  thrust::scatter_if(src.values.begin(), src.values.end(),
                     indices.begin(),
                     indices.begin(),
                     dst.ell.values.values.begin(),
                     less_than<size_t>(dst.ell.values.values.size()));
//// fused version appears to be slightly slower                     
//  thrust::scatter_if(thrust::make_zip_iterator(thrust::make_tuple(src.column_indices.begin(), src.values.begin())),
//                     thrust::make_zip_iterator(thrust::make_tuple(src.column_indices.end(),   src.values.end())),
//                     indices.begin(),
//                     indices.begin(),
//                     thrust::make_zip_iterator(thrust::make_tuple(dst.ell.column_indices.values.begin(), dst.ell.values.values.begin())),
//                     less_than<size_t>(dst.ell.column_indices.values.size()));
}

template <typename Matrix1, typename Matrix2>
void csr_to_hyb(const Matrix1& src, Matrix2& dst,
                const size_t num_entries_per_row, const size_t alignment = 32)
{
  typedef typename Matrix2::index_type IndexType;
  typedef typename Matrix2::value_type ValueType;

  // expand row offsets into row indices
  cusp::array1d<IndexType, cusp::device_memory> row_indices(src.num_entries);
  cusp::detail::offsets_to_indices(src.row_offsets, row_indices);

  // TODO call coo_to_hyb with a coo_matrix_view

  cusp::array1d<IndexType, cusp::device_memory> indices(src.num_entries);
  thrust::exclusive_scan_by_key(row_indices.begin(), row_indices.end(),
                                thrust::constant_iterator<IndexType>(1),
                                indices.begin(),
                                IndexType(0));

  size_t num_coo_entries = thrust::count_if(indices.begin(), indices.end(), greater_than_or_equal_to<size_t>(num_entries_per_row));
  size_t num_ell_entries = src.num_entries - num_coo_entries;

  // allocate output storage
  dst.resize(src.num_rows, src.num_cols, num_ell_entries, num_coo_entries, num_entries_per_row, alignment);

  // fill output with padding
  thrust::fill(dst.ell.column_indices.values.begin(), dst.ell.column_indices.values.end(), IndexType(-1));
  thrust::fill(dst.ell.values.values.begin(),         dst.ell.values.values.end(),         ValueType(0));

  thrust::copy_if(thrust::make_zip_iterator( thrust::make_tuple( row_indices.begin(), src.column_indices.begin(), src.values.begin() ) ),
  		  thrust::make_zip_iterator( thrust::make_tuple( row_indices.end()  , src.column_indices.end()  , src.values.end()   ) ),
                  indices.begin(),
  		  thrust::make_zip_iterator( thrust::make_tuple( dst.coo.row_indices.begin(), dst.coo.column_indices.begin(), dst.coo.values.begin() ) ),
		  greater_than_or_equal_to<size_t>(num_entries_per_row) );

  // next, scale by pitch and add row index
  cusp::blas::axpby(indices, row_indices,
                    indices,
                    IndexType(dst.ell.column_indices.pitch),
                    IndexType(1));

  // scatter CSR entries to ELL
  thrust::scatter_if(src.column_indices.begin(), src.column_indices.end(),
                     indices.begin(),
                     indices.begin(),
                     dst.ell.column_indices.values.begin(),
		     less_than<size_t>(dst.ell.column_indices.values.size()));
  thrust::scatter_if(src.values.begin(), src.values.end(),
                     indices.begin(),
                     indices.begin(),
                     dst.ell.values.values.begin(),
		     less_than<size_t>(dst.ell.values.values.size()));
}

template <typename Matrix1, typename Matrix2>
void ell_to_hyb(const Matrix1& src, Matrix2& dst)
{
  // just copy into ell part of destination
  dst.resize(src.num_rows, src.num_cols,
             src.num_entries, 0,
             src.column_indices.num_cols);

  cusp::copy(src, dst.ell);
}

///////////
// Array //
///////////


} // end namespace device
} // end namespace detail
} // end namespace cusp

