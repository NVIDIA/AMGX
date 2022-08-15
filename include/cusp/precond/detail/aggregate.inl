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

#include <cusp/coo_matrix.h>
#include <cusp/graph/maximal_independent_set.h>
#include <cusp/detail/device/generalized_spmv/coo_flat.h>

#include <thrust/count.h>
#include <thrust/fill.h>
#include <thrust/scan.h>
#include <thrust/gather.h>
#include <thrust/transform.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/iterator/counting_iterator.h>

namespace cusp
{
namespace precond
{
namespace detail
{

template <typename IndexType, typename ValueType, typename MemorySpace,
          typename ArrayType>
void mis_to_aggregates(const cusp::coo_matrix<IndexType,ValueType,MemorySpace>& C,
                       const ArrayType& mis,
                             ArrayType& aggregates)
{
  CUSP_PROFILE_SCOPED();
  const IndexType N = C.num_rows;

  // (2,i) mis (0,i) non-mis

  // current (ring,index)
  ArrayType mis1(N);
  ArrayType idx1(N);
  ArrayType mis2(N);
  ArrayType idx2(N);

  typedef typename ArrayType::value_type T;
  typedef thrust::tuple<T,T> Tuple;

  // find the largest (mis[j],j) 1-ring neighbor for each node
  cusp::detail::device::cuda::spmv_coo
      (C.num_rows, C.num_entries,
       C.row_indices.begin(), C.column_indices.begin(), thrust::constant_iterator<int>(1),  // XXX should we mask explicit zeros? (e.g. DIA, array2d)
       thrust::make_zip_iterator(thrust::make_tuple(mis.begin(), thrust::counting_iterator<IndexType>(0))),
       thrust::make_zip_iterator(thrust::make_tuple(mis.begin(), thrust::counting_iterator<IndexType>(0))),
       thrust::make_zip_iterator(thrust::make_tuple(mis1.begin(), idx1.begin())),
       thrust::project2nd<Tuple,Tuple>(), thrust::maximum<Tuple>());

  // boost mis0 values so they win in second round
  thrust::transform(mis.begin(), mis.end(), mis1.begin(), mis1.begin(), thrust::plus<typename ArrayType::value_type>());

  // find the largest (mis[j],j) 2-ring neighbor for each node
  cusp::detail::device::cuda::spmv_coo
      (C.num_rows, C.num_entries,
       C.row_indices.begin(), C.column_indices.begin(), thrust::constant_iterator<int>(1),  // XXX should we mask explicit zeros? (e.g. DIA, array2d)
       thrust::make_zip_iterator(thrust::make_tuple(mis1.begin(), idx1.begin())),
       thrust::make_zip_iterator(thrust::make_tuple(mis1.begin(), idx1.begin())),
       thrust::make_zip_iterator(thrust::make_tuple(mis2.begin(), idx2.begin())),
       thrust::project2nd<Tuple,Tuple>(), thrust::maximum<Tuple>());

  // enumerate the MIS nodes
  cusp::array1d<IndexType,MemorySpace> mis_enum(N);
  thrust::exclusive_scan(mis.begin(), mis.end(), mis_enum.begin());

  thrust::gather(idx2.begin(), idx2.end(),
                 mis_enum.begin(),
                 aggregates.begin());
} // mis_to_aggregates()


template <typename IndexType, typename ValueType,
          typename ArrayType>
void standard_aggregation(const cusp::coo_matrix<IndexType,ValueType,cusp::device_memory>& C,
                          ArrayType& aggregates)
{
  CUSP_PROFILE_SCOPED();
  // TODO check sizes
  // TODO label singletons with a -1

  const size_t N = C.num_rows;

  // compute MIS(2)
  cusp::array1d<IndexType,cusp::device_memory> mis(N);
  cusp::graph::maximal_independent_set(C, mis, 2);

  // compute aggregates from MIS(2)
  mis_to_aggregates(C, mis, aggregates);
}

template <typename IndexType, typename ValueType,
	  typename ArrayType>
void standard_aggregation(const cusp::csr_matrix<IndexType,ValueType,cusp::host_memory>& C,
              		  ArrayType& aggregates)
{
  CUSP_PROFILE_SCOPED();

  IndexType next_aggregate = 1; // number of aggregates + 1

  // initialize aggregates to 0
  thrust::fill(aggregates.begin(), aggregates.end(), 0);

  IndexType n_row = C.num_rows;

  //Pass #1
  for(IndexType i = 0; i < n_row; i++)
  {
    if(aggregates[i]){ continue; } //already marked

    const IndexType row_start = C.row_offsets[i];
    const IndexType row_end   = C.row_offsets[i+1];

    //Determine whether all neighbors of this node are free (not already aggregates)
    bool has_aggregated_neighbors = false;
    bool has_neighbors            = false;

    for(IndexType jj = row_start; jj < row_end; jj++)
    {
      const IndexType j = C.column_indices[jj];
      if( i != j )
      {
        has_neighbors = true;
        if( aggregates[j] )
        {
          has_aggregated_neighbors = true;
          break;
        }
      }
    }

    if(!has_neighbors)
    {
      //isolated node, do not aggregate
      aggregates[i] = -n_row;
    }
    else if (!has_aggregated_neighbors)
    {
      //Make an aggregate out of this node and its neighbors
      aggregates[i] = next_aggregate;
      for(IndexType jj = row_start; jj < row_end; jj++){
        aggregates[C.column_indices[jj]] = next_aggregate;
      }
      next_aggregate++;
    }
  }

  //Pass #2
  // Add unaggregated nodes to any neighboring aggregate
  for(IndexType i = 0; i < n_row; i++){
    if(aggregates[i]){ continue; } //already marked

    for(IndexType jj = C.row_offsets[i]; jj < C.row_offsets[i+1]; jj++){
      const IndexType j = C.column_indices[jj];

      const IndexType tj = aggregates[j];
      if(tj > 0){
        aggregates[i] = -tj;
        break;
      }
    }
  }

  next_aggregate--;

  //Pass #3
  for(IndexType i = 0; i < n_row; i++){
    const IndexType ti = aggregates[i];

    if(ti != 0){
      // node i has been aggregated
      if(ti > 0)
        aggregates[i] = ti - 1;
      else if(ti == -n_row)
        aggregates[i] = -1;
      else
        aggregates[i] = -ti - 1;
      continue;
    }

    // node i has not been aggregated
    const IndexType row_start = C.row_offsets[i];
    const IndexType row_end   = C.row_offsets[i+1];

    aggregates[i] = next_aggregate;

    for(IndexType jj = row_start; jj < row_end; jj++){
      const IndexType j = C.column_indices[jj];

      if(aggregates[j] == 0){ //unmarked neighbors
        aggregates[j] = next_aggregate;
      }
    }
    next_aggregate++;
  }

  if( next_aggregate == 0 )
  {
    thrust::fill( aggregates.begin(), aggregates.end(), 0 );
  }
  else
  {
    // TODO Handle unaggregated nodes
    ValueType agg_min = *thrust::min_element(aggregates.begin(), aggregates.end());
    if( agg_min == -1 )
    {
      IndexType num_unaggregated = thrust::count(aggregates.begin(), aggregates.end(), -1.0);
      printf("number unaggregated nodes: %d\n",num_unaggregated);
      for( size_t i = 0; i < aggregates.size(); i++ )
      {
	if( aggregates[i] == -1 ) 
	{
	  aggregates[i] = next_aggregate++;
	}
      }
    }
  }
}

} // end namespace detail
} // end namespace precond
} // end namespace cusp

