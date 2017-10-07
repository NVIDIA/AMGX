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
#include <cusp/format.h>
#include <cusp/array1d.h>

#include <thrust/fill.h>
#include <thrust/extrema.h>
#include <thrust/binary_search.h>
#include <thrust/transform.h>
#include <thrust/gather.h>
#include <thrust/scatter.h>
#include <thrust/sequence.h>
#include <thrust/scan.h>
#include <thrust/sort.h>

namespace cusp
{
namespace detail
{

template <typename IndexType>
struct empty_row_functor
{
  typedef bool result_type;

  template <typename Tuple>
    __host__ __device__
  bool operator()(const Tuple& t) const
  {
    const IndexType a = thrust::get<0>(t);
    const IndexType b = thrust::get<1>(t);

    return a != b;
  }
};


template <typename OffsetArray, typename IndexArray>
void offsets_to_indices(const OffsetArray& offsets, IndexArray& indices)
{
    CUSP_PROFILE_SCOPED();

    typedef typename OffsetArray::value_type OffsetType;

    // convert compressed row offsets into uncompressed row indices
    thrust::fill(indices.begin(), indices.end(), OffsetType(0));
    thrust::scatter_if( thrust::counting_iterator<OffsetType>(0),
			thrust::counting_iterator<OffsetType>(offsets.size()-1),
			offsets.begin(),
                    	thrust::make_transform_iterator(
                                thrust::make_zip_iterator( thrust::make_tuple( offsets.begin(), offsets.begin()+1 ) ),
                                empty_row_functor<OffsetType>()),
                    	indices.begin());
    thrust::inclusive_scan(indices.begin(), indices.end(), indices.begin(), thrust::maximum<OffsetType>());
}

template <typename IndexArray, typename OffsetArray>
void indices_to_offsets(const IndexArray& indices, OffsetArray& offsets)
{
    CUSP_PROFILE_SCOPED();

    typedef typename OffsetArray::value_type OffsetType;

    // convert uncompressed row indices into compressed row offsets
    thrust::lower_bound(indices.begin(),
                        indices.end(),
                        thrust::counting_iterator<OffsetType>(0),
                        thrust::counting_iterator<OffsetType>(offsets.size()),
                        offsets.begin());
}

template<typename IndexType>
struct row_operator : public std::unary_function<size_t,IndexType>
{
    size_t pitch;

    row_operator(size_t pitch)
        : pitch(pitch) {}

    __host__ __device__
    IndexType operator()(const size_t & linear_index) const
    {
        return linear_index % pitch;
    }
};


template <typename IndexType>
struct tuple_equal_to : public thrust::unary_function<thrust::tuple<IndexType,IndexType>,bool>
{
    __host__ __device__
    bool operator()(const thrust::tuple<IndexType,IndexType>& t) const
    {
        return thrust::get<0>(t) == thrust::get<1>(t);
    }
};

template <typename Matrix, typename Array>
void extract_diagonal(const Matrix& A, Array& output, cusp::coo_format)
{
    CUSP_PROFILE_SCOPED();

    typedef typename Matrix::index_type  IndexType;
    typedef typename Array::value_type   ValueType;
    
    // initialize output to zero
    thrust::fill(output.begin(), output.end(), ValueType(0));

    // scatter the diagonal values to output
    thrust::scatter_if(A.values.begin(), A.values.end(),
                       A.row_indices.begin(),
                       thrust::make_transform_iterator(thrust::make_zip_iterator(thrust::make_tuple(A.row_indices.begin(), A.column_indices.begin())), tuple_equal_to<IndexType>()),
                       output.begin());
}


template <typename Matrix, typename Array>
void extract_diagonal(const Matrix& A, Array& output, cusp::csr_format)
{
    typedef typename Matrix::index_type  IndexType;
    typedef typename Array::value_type   ValueType;
    typedef typename Array::memory_space MemorySpace;
    
    // first expand the compressed row offsets into row indices
    cusp::array1d<IndexType,MemorySpace> row_indices(A.num_entries);
    offsets_to_indices(A.row_offsets, row_indices);

    // initialize output to zero
    thrust::fill(output.begin(), output.end(), ValueType(0));

    // scatter the diagonal values to output
    thrust::scatter_if(A.values.begin(), A.values.end(),
                       row_indices.begin(),
                       thrust::make_transform_iterator(thrust::make_zip_iterator(thrust::make_tuple(row_indices.begin(), A.column_indices.begin())), tuple_equal_to<IndexType>()),
                       output.begin());
}


template <typename Matrix, typename Array>
void extract_diagonal(const Matrix& A, Array& output, cusp::dia_format)
{
    typedef typename Matrix::index_type  IndexType;
    typedef typename Array::value_type   ValueType;
    
    // copy diagonal_offsets to host (sometimes unnecessary)
    cusp::array1d<IndexType,cusp::host_memory> diagonal_offsets(A.diagonal_offsets);

    for(size_t i = 0; i < diagonal_offsets.size(); i++)
    {
        if(diagonal_offsets[i] == 0)
        {
            // diagonal found, copy to output and return
            thrust::copy(A.values.values.begin() + A.values.pitch * i,
                         A.values.values.begin() + A.values.pitch * i + output.size(),
                         output.begin());
            return;
        }
    }

    // no diagonal found
    thrust::fill(output.begin(), output.end(), ValueType(0));
}


template <typename Matrix, typename Array>
void extract_diagonal(const Matrix& A, Array& output, cusp::ell_format)
{
    typedef typename Matrix::index_type  IndexType;
    typedef typename Array::value_type   ValueType;
    
    // initialize output to zero
    thrust::fill(output.begin(), output.end(), ValueType(0));

    thrust::scatter_if
        (A.values.values.begin(), A.values.values.end(),
         thrust::make_transform_iterator(thrust::counting_iterator<size_t>(0), row_operator<IndexType>(A.column_indices.pitch)),
         thrust::make_zip_iterator(thrust::make_tuple
             (thrust::make_transform_iterator(thrust::counting_iterator<size_t>(0), row_operator<IndexType>(A.column_indices.pitch)),
              A.column_indices.values.begin())),
         output.begin(),
         tuple_equal_to<IndexType>());

    // TODO ignore padded values in column_indices
}

template <typename Matrix, typename Array>
void extract_diagonal(const Matrix& A, Array& output, cusp::hyb_format)
{
    typedef typename Matrix::index_type  IndexType;
    typedef typename Array::value_type   ValueType;
    
    // extract COO diagonal
    cusp::detail::extract_diagonal(A.coo, output);

    // extract ELL diagonal
    thrust::scatter_if
        (A.ell.values.values.begin(), A.ell.values.values.end(),
         thrust::make_transform_iterator(thrust::counting_iterator<size_t>(0), row_operator<IndexType>(A.ell.column_indices.pitch)),
         thrust::make_zip_iterator(thrust::make_tuple
             (thrust::make_transform_iterator(thrust::counting_iterator<size_t>(0), row_operator<IndexType>(A.ell.column_indices.pitch)),
              A.ell.column_indices.values.begin())),
         output.begin(),
         tuple_equal_to<IndexType>());

    // TODO ignore padded values in column_indices
}


template <typename Matrix, typename Array>
void extract_diagonal(const Matrix& A, Array& output)
{
    CUSP_PROFILE_SCOPED();

    output.resize(thrust::min(A.num_rows, A.num_cols));

    // dispatch on matrix format
    extract_diagonal(A, output, typename Matrix::format());
}

template <typename Array1, typename Array2, typename Array3>
void sort_by_row(Array1& rows, Array2& columns, Array3& values)
{
    CUSP_PROFILE_SCOPED();

    typedef typename Array1::value_type IndexType;
    typedef typename Array3::value_type ValueType;
    typedef typename Array1::memory_space MemorySpace;
        
    size_t N = rows.size();

    cusp::array1d<IndexType,MemorySpace> permutation(N);
    thrust::sequence(permutation.begin(), permutation.end());
  
    // compute permutation that sorts the rows
    thrust::sort_by_key(rows.begin(), rows.end(), permutation.begin());

    // copy columns and values to temporary buffers
    cusp::array1d<IndexType,MemorySpace> temp1(columns);
    cusp::array1d<ValueType,MemorySpace> temp2(values);
        
    // use permutation to reorder the values
    thrust::gather(permutation.begin(), permutation.end(),
                   thrust::make_zip_iterator(thrust::make_tuple(temp1.begin(),   temp2.begin())),
                   thrust::make_zip_iterator(thrust::make_tuple(columns.begin(), values.begin())));
}

template <typename Array1, typename Array2, typename Array3>
void sort_by_row_and_column(Array1& rows, Array2& columns, Array3& values)
{
    CUSP_PROFILE_SCOPED();

    typedef typename Array1::value_type IndexType;
    typedef typename Array3::value_type ValueType;
    typedef typename Array1::memory_space MemorySpace;
        
    size_t N = rows.size();

    cusp::array1d<IndexType,MemorySpace> permutation(N);
    thrust::sequence(permutation.begin(), permutation.end());
  
    // compute permutation and sort by (I,J)
    {
        cusp::array1d<IndexType,MemorySpace> temp(columns);
        thrust::stable_sort_by_key(temp.begin(), temp.end(), permutation.begin());

        cusp::copy(rows, temp);
        thrust::gather(permutation.begin(), permutation.end(), temp.begin(), rows.begin());
        thrust::stable_sort_by_key(rows.begin(), rows.end(), permutation.begin());

        cusp::copy(columns, temp);
        thrust::gather(permutation.begin(), permutation.end(), temp.begin(), columns.begin());
    }

    // use permutation to reorder the values
    {
        cusp::array1d<ValueType,MemorySpace> temp(values);
        thrust::gather(permutation.begin(), permutation.end(), temp.begin(), values.begin());
    }
}

} // end namespace detail
} // end namespace cusp

