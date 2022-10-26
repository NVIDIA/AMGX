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

#include <cusp/array1d.h>

#include <cusp/detail/format_utils.h>

#include <thrust/gather.h>
#include <thrust/scan.h>
#include <thrust/scatter.h>
#include <thrust/transform.h>
#include <thrust/reduce.h>
#include <thrust/inner_product.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/iterator/permutation_iterator.h>

#include <list>

namespace cusp
{
namespace detail
{
namespace device
{

template <typename Matrix1,
          typename Matrix2,
          typename Matrix3,
          typename Array1,
          typename Array2>
void coo_spmm_helper(size_t workspace_size,
                     size_t begin_row,
                     size_t end_row,
                     size_t begin_segment,
                     size_t end_segment,
                     const Matrix1& A,
                     const Matrix2& B,
                           Matrix3& C,
                     const Array1& B_row_offsets,
                     const Array1& segment_lengths,
                     const Array1& output_ptr,
                           Array1& A_gather_locations,
                           Array1& B_gather_locations,
                           Array1& I,
                           Array1& J,
                           Array2& V)
{
    typedef typename Array1::value_type IndexType;
    typedef typename Array2::value_type ValueType;

    A_gather_locations.resize(workspace_size);
    B_gather_locations.resize(workspace_size);
    I.resize(workspace_size);
    J.resize(workspace_size);
    V.resize(workspace_size);
  
    // nothing to do
    if (workspace_size == 0)
    {
        C.resize(A.num_rows, B.num_cols, 0);
        return;
    }

    // compute gather locations of intermediate format
    amgx::thrust::fill(A_gather_locations.begin(), A_gather_locations.end(), 0);
    amgx::thrust::scatter_if(amgx::thrust::counting_iterator<IndexType>(begin_segment), amgx::thrust::counting_iterator<IndexType>(end_segment),
                       output_ptr.begin() + begin_segment, 
                       segment_lengths.begin() + begin_segment,
                       A_gather_locations.begin() - output_ptr[begin_segment]);
    amgx::thrust::inclusive_scan(A_gather_locations.begin(), A_gather_locations.end(), A_gather_locations.begin(), amgx::thrust::maximum<IndexType>());
  
    // compute gather locations of intermediate format
    amgx::thrust::fill(B_gather_locations.begin(), B_gather_locations.end(), 1);
    amgx::thrust::scatter_if(amgx::thrust::make_permutation_iterator(B_row_offsets.begin(), A.column_indices.begin()) + begin_segment,
                       amgx::thrust::make_permutation_iterator(B_row_offsets.begin(), A.column_indices.begin()) + end_segment,
                       output_ptr.begin() + begin_segment,
//                       amgx::thrust::make_transform_iterator(output_ptr.begin(), subtract_constant<IndexType>(begin + begin_segment,
                       segment_lengths.begin() + begin_segment,
                       B_gather_locations.begin() - output_ptr[begin_segment]);
                       amgx::thrust::inclusive_scan_by_key(A_gather_locations.begin(), A_gather_locations.end(),
                                  B_gather_locations.begin(),
                                  B_gather_locations.begin());

    
    thrust_wrapper::gather(A_gather_locations.begin(), A_gather_locations.end(),
                   A.row_indices.begin(),
                   I.begin());
    thrust_wrapper::gather(B_gather_locations.begin(), B_gather_locations.end(),
                   B.column_indices.begin(),
                   J.begin());

    amgx::thrust::transform(amgx::thrust::make_permutation_iterator(A.values.begin(), A_gather_locations.begin()),
                      amgx::thrust::make_permutation_iterator(A.values.begin(), A_gather_locations.end()),
                      amgx::thrust::make_permutation_iterator(B.values.begin(), B_gather_locations.begin()),
                      V.begin(),
                      amgx::thrust::multiplies<ValueType>());

    // sort (I,J,V) tuples by (I,J)
    cusp::detail::sort_by_row_and_column(I, J, V);

    // compute unique number of nonzeros in the output
    IndexType NNZ = amgx::thrust::inner_product(amgx::thrust::make_zip_iterator(amgx::thrust::make_tuple(I.begin(), J.begin())),
                                          amgx::thrust::make_zip_iterator(amgx::thrust::make_tuple(I.end (),  J.end()))   - 1,
                                          amgx::thrust::make_zip_iterator(amgx::thrust::make_tuple(I.begin(), J.begin())) + 1,
                                          IndexType(0),
                                          amgx::thrust::plus<IndexType>(),
                                          amgx::thrust::not_equal_to< amgx::thrust::tuple<IndexType,IndexType> >()) + 1;

    // allocate space for output
    C.resize(A.num_rows, B.num_cols, NNZ);

    // sum values with the same (i,j)
    amgx::thrust::reduce_by_key
        (amgx::thrust::make_zip_iterator(amgx::thrust::make_tuple(I.begin(), J.begin())),
         amgx::thrust::make_zip_iterator(amgx::thrust::make_tuple(I.end(),   J.end())),
         V.begin(),
         amgx::thrust::make_zip_iterator(amgx::thrust::make_tuple(C.row_indices.begin(), C.column_indices.begin())),
         C.values.begin(),
         amgx::thrust::equal_to< amgx::thrust::tuple<IndexType,IndexType> >(),
         amgx::thrust::plus<ValueType>());
}


template <typename Matrix1,
          typename Matrix2,
          typename Matrix3>
void spmm_coo(const Matrix1& A,
              const Matrix2& B,
                    Matrix3& C)
{
    CUSP_PROFILE_SCOPED();

    typedef typename Matrix3::index_type   IndexType;
    typedef typename Matrix3::value_type   ValueType;
    typedef typename Matrix3::memory_space MemorySpace;

    // check whether matrices are empty
    if (A.num_entries == 0 || B.num_entries == 0)
    {
        C.resize(A.num_rows, B.num_cols, 0);
        return;
    }

    // compute row offsets for B
    cusp::array1d<IndexType,MemorySpace> B_row_offsets(B.num_rows + 1);
    cusp::detail::indices_to_offsets(B.row_indices, B_row_offsets);

    // compute row lengths for B
    cusp::array1d<IndexType,MemorySpace> B_row_lengths(B.num_rows);
    amgx::thrust::transform(B_row_offsets.begin() + 1, B_row_offsets.end(), B_row_offsets.begin(), B_row_lengths.begin(), amgx::thrust::minus<IndexType>());

    // for each element A(i,j) compute the number of nonzero elements in B(j,:)
    cusp::array1d<IndexType,MemorySpace> segment_lengths(A.num_entries);
    thrust_wrapper::gather(A.column_indices.begin(), A.column_indices.end(),
                   B_row_lengths.begin(),
                   segment_lengths.begin());
    
    // output pointer
    cusp::array1d<IndexType,MemorySpace> output_ptr(A.num_entries + 1);
    amgx::thrust::exclusive_scan(segment_lengths.begin(), segment_lengths.end(),
                           output_ptr.begin(),
                           IndexType(0));
    output_ptr[A.num_entries] = output_ptr[A.num_entries - 1] + segment_lengths[A.num_entries - 1]; // XXX is this necessary?

    size_t coo_num_nonzeros = output_ptr[A.num_entries];

    size_t workspace_capacity = amgx::thrust::min<size_t>(coo_num_nonzeros, 16 << 20);
    
    {
      // TODO abstract this
      size_t free, total;
      cudaMemGetInfo(&free, &total);

      // divide free bytes by the size of each workspace unit
      size_t max_workspace_capacity = free / (4 * sizeof(IndexType) + sizeof(ValueType));

      // use at most one third of the remaining capacity
      workspace_capacity = amgx::thrust::min<size_t>(max_workspace_capacity / 3, workspace_capacity);
    }

    // workspace arrays
    cusp::array1d<IndexType,MemorySpace> A_gather_locations;
    cusp::array1d<IndexType,MemorySpace> B_gather_locations;
    cusp::array1d<IndexType,MemorySpace> I;
    cusp::array1d<IndexType,MemorySpace> J;
    cusp::array1d<ValueType,MemorySpace> V;

    if (coo_num_nonzeros <= workspace_capacity)
    {
        // compute C = A * B in one step
        size_t begin_row      = 0;
        size_t end_row        = A.num_rows;
        size_t begin_segment  = 0;
        size_t end_segment    = A.num_entries;
        size_t workspace_size = coo_num_nonzeros;

        coo_spmm_helper(workspace_size,
                        begin_row, end_row,
                        begin_segment, end_segment,
                        A, B, C,
                        B_row_offsets,
                        segment_lengths, output_ptr,
                        A_gather_locations, B_gather_locations,
                        I, J, V);
    }
    else
    {
        // decompose C = A * B into several C[slice,:] = A[slice,:] * B operations
        typedef typename cusp::coo_matrix<IndexType,ValueType,MemorySpace> Container;
        typedef typename std::list<Container> ContainerList;

        // storage for C[slice,:] partial results
        ContainerList slices;

        // compute row offsets for A
        cusp::array1d<IndexType,MemorySpace> A_row_offsets(A.num_rows + 1);
        cusp::detail::indices_to_offsets(A.row_indices, A_row_offsets);
    
        // compute worspace requirements for each row
        cusp::array1d<IndexType,MemorySpace> cummulative_row_workspace(A.num_rows);
        thrust_wrapper::gather(A_row_offsets.begin() + 1, A_row_offsets.end(),
                       output_ptr.begin(),
                       cummulative_row_workspace.begin());

        size_t begin_row = 0;
        size_t total_work = 0;

        while (begin_row < size_t(A.num_rows))
        {
            Container C_slice;
    
            // find largest end_row such that the capacity of [begin_row, end_row) fits in the workspace_capacity
            size_t end_row = amgx::thrust::upper_bound(cummulative_row_workspace.begin() + begin_row, cummulative_row_workspace.end(),
                                                 total_work + IndexType(workspace_capacity)) - cummulative_row_workspace.begin();

            size_t begin_segment = A_row_offsets[begin_row];
            size_t end_segment   = A_row_offsets[end_row];
        
            // TODO throw exception signaling that there is insufficient memory (not necessarily bad_alloc)
            //if (begin_row == end_row)
            //    // workspace wasn't large enough, throw cusp::memory_allocation_failure?

            size_t workspace_size = output_ptr[end_segment] - output_ptr[begin_segment];
            
            total_work += workspace_size;

            // TODO remove these when an exception is in place
            assert(end_row > begin_row);
            assert(workspace_size <= workspace_capacity);

            coo_spmm_helper(workspace_size,
                            begin_row, end_row,
                            begin_segment, end_segment,
                            A, B, C_slice,
                            B_row_offsets,
                            segment_lengths, output_ptr,
                            A_gather_locations, B_gather_locations,
                            I, J, V);

            slices.push_back(Container());
            slices.back().swap(C_slice);

            begin_row = end_row;
        }

        // deallocate workspace
        A_gather_locations.clear(); A_gather_locations.shrink_to_fit();  
        B_gather_locations.clear(); B_gather_locations.shrink_to_fit();
        I.clear();                  I.shrink_to_fit();
        J.clear();                  J.shrink_to_fit();
        V.clear();                  V.shrink_to_fit();

        // compute total output size
        size_t C_num_entries = 0;
        for(typename ContainerList::iterator iter = slices.begin(); iter != slices.end(); ++iter)
            C_num_entries += iter->num_entries;

        // resize output
        C.resize(A.num_rows, B.num_cols, C_num_entries);
       
        // copy slices into output
        size_t base = 0;
        for(typename ContainerList::iterator iter = slices.begin(); iter != slices.end(); ++iter)
        {
            amgx::thrust::copy(iter->row_indices.begin(),    iter->row_indices.end(),    C.row_indices.begin()    + base);
            amgx::thrust::copy(iter->column_indices.begin(), iter->column_indices.end(), C.column_indices.begin() + base);
            amgx::thrust::copy(iter->values.begin(),         iter->values.end(),         C.values.begin()         + base);
            base += iter->num_entries;
        }
    }
}

} // end namespace device
} // end namespace detail
} // end namespace cusp

