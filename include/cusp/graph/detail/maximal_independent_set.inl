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

#include <cusp/detail/device/generalized_spmv/csr_scalar.h>

#include <cusp/copy.h>
#include <cusp/array1d.h>
#include <cusp/exception.h>
#include <cusp/coo_matrix.h>
#include <cusp/csr_matrix.h>
#include <cusp/detail/random.h>
#include <cusp/detail/format_utils.h>

//#include <cusp/print.h>

#include <thrust/count.h>
#include <thrust/transform.h>
#include <thrust/transform_scan.h>
#include <thrust/iterator/constant_iterator.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/zip_iterator.h>

namespace cusp
{
namespace graph
{
namespace detail
{

struct process_mis_nodes
{
    template <typename Tuple>
    __host__ __device__
    void operator()(Tuple t)
    {
        if (thrust::get<1>(t) == 1)                     // undecided node
        {
            if (thrust::get<0>(t) == thrust::get<3>(t)) // i == maximal_index
              thrust::get<1>(t) = 2;                    // mis_node
        }
    }
};

struct process_non_mis_nodes
{
    template <typename Tuple>
    __host__ __device__
    void operator()(Tuple t)
    {
        if (thrust::get<0>(t) == 1)            // undecided node
        {
            if (thrust::get<1>(t) == 2)        // maximal_state == mis_node
              thrust::get<0>(t) = 0;           // non_mis_node
        }
    }
};

  
template <typename Matrix, typename IndexType>
void propagate_distances(const Matrix& A,
                         const IndexType i,
                         const size_t d,
                         const size_t k,
                         cusp::array1d<size_t,cusp::host_memory>& distance)
{
  distance[i] = d;

  if (d < k)
  {
    for(IndexType jj = A.row_offsets[i]; jj < A.row_offsets[i + 1]; jj++)
    {
      IndexType j = A.column_indices[jj];

      // update only if necessary
      if (d + 1 < distance[j])
        propagate_distances(A, j, d + 1, k, distance);
    }
  }
}

struct is_subgraph_edge
{
  template <typename Tuple>
  __host__ __device__
  bool operator()(const Tuple& t) const
  {
    return thrust::get<0>(t) && thrust::get<1>(t);
  }
};

template <typename NodeStateType>
struct is_active_node
{
  __host__ __device__
  bool operator()(const NodeStateType& s) const
  {
    return s == 1;
  }
};


template <typename Array1,
          typename Array2,
          typename Array3,
          typename Array4>
void compute_mis_states(const size_t k,
                        const Array1& row_indices,
                        const Array2& column_indices,
                        const Array3& random_values,
                              Array4& states)
{
    typedef typename Array1::value_type   IndexType;
    typedef typename Array3::value_type   RandomType;
    typedef typename Array4::value_type   NodeStateType;
    typedef typename Array1::memory_space MemorySpace;

    typedef typename thrust::tuple<NodeStateType,RandomType,IndexType> Tuple;
    
    const size_t N = states.size();

    const IndexType num_rows    = states.size();
    //const IndexType num_entries = row_indices.size();

    // TODO remove this WAR when generalize COO SpMV problem is resolved
    cusp::array1d<IndexType,MemorySpace> row_offsets(num_rows + 1);
    cusp::detail::indices_to_offsets(row_indices, row_offsets);
    
    cusp::array1d<NodeStateType,MemorySpace> maximal_states(N);
    cusp::array1d<RandomType,MemorySpace>    maximal_values(N);
    cusp::array1d<IndexType,MemorySpace>     maximal_indices(N);

    cusp::array1d<NodeStateType,MemorySpace> last_states;
    cusp::array1d<RandomType,MemorySpace>    last_values;
    cusp::array1d<IndexType,MemorySpace>     last_indices;;
    
    // TODO choose threshold in a more principled manner
//    size_t compaction_threshold = (N < 10000) ? 0 : N / 10;
    size_t active_nodes = N;
    
//    size_t num_iters = 0;

    do
    {
        // find the largest (state,value,index) 1-ring neighbor for each node
        cusp::detail::device::cuda::spmv_csr_scalar
            (num_rows,
             row_offsets.begin(), column_indices.begin(), thrust::constant_iterator<Tuple>(Tuple(0,0)),  // XXX should we mask explicit zeros? (e.g. DIA, array2d)
             thrust::make_zip_iterator(thrust::make_tuple(states.begin(), random_values.begin(), thrust::counting_iterator<IndexType>(0))),
             thrust::make_zip_iterator(thrust::make_tuple(states.begin(), random_values.begin(), thrust::counting_iterator<IndexType>(0))),
             thrust::make_zip_iterator(thrust::make_tuple(maximal_states.begin(), maximal_values.begin(), maximal_indices.begin())),
             thrust::project2nd<Tuple,Tuple>(), thrust::maximum<Tuple>());
        //cusp::detail::device::cuda::spmv_coo
        //    (num_rows, num_entries,
        //     row_indices.begin(), column_indices.begin(), thrust::constant_iterator<Tuple>(Tuple(0,0)),  // XXX should we mask explicit zeros? (e.g. DIA, array2d)
        //     thrust::make_zip_iterator(thrust::make_tuple(states.begin(), random_values.begin(), thrust::counting_iterator<IndexType>(0))),
        //     thrust::make_zip_iterator(thrust::make_tuple(states.begin(), random_values.begin(), thrust::counting_iterator<IndexType>(0))),
        //     thrust::make_zip_iterator(thrust::make_tuple(maximal_states.begin(), maximal_values.begin(), maximal_indices.begin())),
        //     thrust::project2nd<Tuple,Tuple>(), thrust::maximum<Tuple>());

        // find the largest (state,value,index) k-ring neighbor for each node (if k > 1)
        for(size_t ring = 1; ring < k; ring++)
        {
            last_states.resize (N); last_states.swap (maximal_states);
            last_values.resize (N); last_values.swap (maximal_values);
            last_indices.resize(N); last_indices.swap(maximal_indices);

            // TODO replace with call to generalized method
            cusp::detail::device::cuda::spmv_csr_scalar
                (num_rows,
                 row_offsets.begin(), column_indices.begin(), thrust::constant_iterator<Tuple>(Tuple(0,0)),  // XXX should we mask explicit zeros? (e.g. DIA, array2d)
                 thrust::make_zip_iterator(thrust::make_tuple(last_states.begin(), last_values.begin(), last_indices.begin())),
                 thrust::make_zip_iterator(thrust::make_tuple(last_states.begin(), last_values.begin(), last_indices.begin())),
                 thrust::make_zip_iterator(thrust::make_tuple(maximal_states.begin(), maximal_values.begin(), maximal_indices.begin())),
                 thrust::project2nd<Tuple,Tuple>(), thrust::maximum<Tuple>());
        }
       
        // label local maxima as MIS nodes
        thrust::for_each(thrust::make_zip_iterator(thrust::make_tuple(thrust::counting_iterator<IndexType>(0), states.begin(), maximal_states.begin(), maximal_indices.begin())),
                         thrust::make_zip_iterator(thrust::make_tuple(thrust::counting_iterator<IndexType>(0), states.begin(), maximal_states.begin(), maximal_indices.begin())) + N,
                         process_mis_nodes());
        
        // label k-ring neighbors of MIS nodes as non-MIS nodes
        thrust::for_each(thrust::make_zip_iterator(thrust::make_tuple(states.begin(), thrust::make_permutation_iterator(states.begin(), maximal_indices.begin()))),
                         thrust::make_zip_iterator(thrust::make_tuple(states.begin(), thrust::make_permutation_iterator(states.begin(), maximal_indices.begin()))) + N,
                         process_non_mis_nodes());

        active_nodes = thrust::count(states.begin(), states.end(), 1);
        
//        num_iters++;
//        std::cout << "(iter " <<  num_iters << "," << (double(active_nodes) / double(N)) << ")" << std::endl;
//        std::cout << "N= " << N << " iteration=" << num_iters << " active_nodes=" << active_nodes << " compaction_threshold=" << compaction_threshold << "\n";
//        std::cout << "states\n";
//        cusp::print(states);
//
//        if (active_nodes < compaction_threshold)
//        {
//            cusp::array1d<bool,MemorySpace> retained_nodes(N);
//            cusp::array1d<bool,MemorySpace> last_retained_nodes(N);
//
//            thrust::transform(maximal_states.begin(), maximal_states.end(), thrust::constant_iterator<NodeStateType>(1), retained_nodes.begin(), thrust::equal_to<NodeStateType>());
//
//            // propagate retained region outward
//            for(size_t ring = 1; 2*ring <= k; ring++)
//            {
//                retained_nodes.swap(last_retained_nodes);
//
//                // TODO replace with call to generalized method
//                cusp::detail::device::cuda::spmv_coo
//                    (num_rows, num_entries,
//                     row_indices.begin(), column_indices.begin(), thrust::constant_iterator<bool>(false), 
//                     last_retained_nodes.begin(),
//                     last_retained_nodes.begin(),
//                     retained_nodes.begin(),
//                     thrust::project2nd<bool,bool>(), thrust::logical_or<bool>());
//            }
//        
//            std::cout << "retained nodes\n";
//            cusp::print(retained_nodes);
//
//            size_t num_subgraph_nodes = thrust::count(retained_nodes.begin(), retained_nodes.end(), true);
//            size_t num_subgraph_edges = thrust::count
//                (thrust::make_zip_iterator(thrust::make_tuple(thrust::make_permutation_iterator(retained_nodes.begin(), row_indices.begin()),
//                                                              thrust::make_permutation_iterator(retained_nodes.begin(), column_indices.begin()))),
//                 thrust::make_zip_iterator(thrust::make_tuple(thrust::make_permutation_iterator(retained_nodes.begin(), row_indices.end()),
//                                                              thrust::make_permutation_iterator(retained_nodes.begin(), column_indices.end()))),
//                 thrust::make_tuple(true,true));
//
//
//            std::cout << "subgraph nodes: " << double(100*num_subgraph_nodes)/N << "% edges " << double(100*num_subgraph_edges)/num_entries << "%" << std::endl;
//
//            // map old indices into subgraph indices
//            cusp::array1d<IndexType, MemorySpace> index_map(N);
//            thrust::transform_exclusive_scan(retained_nodes.begin(), retained_nodes.end(), index_map.begin(), thrust::identity<IndexType>(), IndexType(0), thrust::plus<IndexType>());
//            
//            std::cout << "index map\n";
//            cusp::print(index_map);
//    
//            // storage for subgraph
//            cusp::array1d<IndexType,     MemorySpace> subgraph_row_indices(num_subgraph_edges);
//            cusp::array1d<IndexType,     MemorySpace> subgraph_column_indices(num_subgraph_edges);
//            cusp::array1d<NodeStateType, MemorySpace> subgraph_states(num_subgraph_nodes);
//            cusp::array1d<RandomType,    MemorySpace> subgraph_random_values(num_subgraph_nodes);
//            
//            thrust::copy_if
//                (thrust::make_zip_iterator(thrust::make_tuple(thrust::make_permutation_iterator(index_map.begin(), row_indices.begin()),
//                                                              thrust::make_permutation_iterator(index_map.begin(), column_indices.begin()))),
//                 thrust::make_zip_iterator(thrust::make_tuple(thrust::make_permutation_iterator(index_map.begin(), row_indices.end()),
//                                                              thrust::make_permutation_iterator(index_map.begin(), column_indices.end()))),
//                 thrust::make_zip_iterator(thrust::make_tuple(thrust::make_permutation_iterator(retained_nodes.begin(), row_indices.begin()),
//                                                              thrust::make_permutation_iterator(retained_nodes.begin(), column_indices.begin()))),
//                 thrust::make_zip_iterator(thrust::make_tuple(subgraph_row_indices.begin(),
//                                                              subgraph_column_indices.begin())),
//                 is_subgraph_edge());
//
//            thrust::scatter_if
//                (thrust::make_zip_iterator(thrust::make_tuple(states.begin(), random_values.begin())),
//                 thrust::make_zip_iterator(thrust::make_tuple(states.end(),   random_values.end())),
//                 index_map.begin(),
//                 retained_nodes.begin(),
//                 thrust::make_zip_iterator(thrust::make_tuple(subgraph_states.begin(), subgraph_random_values.begin())));
//        
//    
//            compute_mis_states(k, subgraph_row_indices, subgraph_column_indices, subgraph_random_values, subgraph_states);
//
//            // update active node states from subgraph
//            thrust::gather_if(index_map.begin(), index_map.end(),
//                              retained_nodes.begin(),
//                              subgraph_states.begin(),
//                              states.begin());
//            return;
//        }
    } while (active_nodes > 0);
}





////////////////
// Host Paths //
////////////////
      
template <typename Matrix, typename Array>
size_t maximal_independent_set(const Matrix& A, Array& stencil, size_t k,
                               cusp::csr_format, cusp::host_memory)
{
  typedef typename Matrix::index_type   IndexType;
  
  const IndexType N = A.num_rows;

  // distance to nearest MIS node
  cusp::array1d<size_t,cusp::host_memory> distance(N, k + 1);

  // count number of MIS nodes
  size_t set_nodes = 0;
  
  // pick MIS-k nodes greedily and deactivate all their k-neighbors
  for(IndexType i = 0; i < N; i++)
  {
    if (distance[i] > k)
    {
      set_nodes++;

      // reset distances on all k-ring neighbors 
      propagate_distances(A, i, 0, k, distance);
    }
  }
  
  // write output
  stencil.resize(N);

  for (IndexType i = 0; i < N; i++)
      stencil[i] = distance[i] == 0;

  return set_nodes;
}

//////////////////
// Device Paths //
//////////////////

template <typename Matrix, typename Array>
size_t maximal_independent_set(const Matrix& A, Array& stencil, size_t k,
                               cusp::coo_format, cusp::device_memory)
{
    typedef typename Matrix::index_type   IndexType;
    typedef typename Matrix::value_type   ValueType;
    typedef typename Matrix::memory_space MemorySpace;
    typedef unsigned int  RandomType;
    typedef unsigned char NodeStateType;
        
    const IndexType N = A.num_rows;
    
    cusp::array1d<RandomType,MemorySpace> random_values(N);
    cusp::copy(cusp::detail::random_integers<RandomType>(N), random_values);

    cusp::array1d<NodeStateType,MemorySpace> states(N, 1);

    compute_mis_states(k, A.row_indices, A.column_indices, random_values, states);
    
    // resize output
    stencil.resize(N);

    // mark all mis nodes
    thrust::transform(states.begin(), states.end(), thrust::constant_iterator<NodeStateType>(2), stencil.begin(), thrust::equal_to<NodeStateType>());

    // return the size of the MIS
    return thrust::count(stencil.begin(), stencil.end(), typename Array::value_type(true));
}


//////////////
// General Path //
//////////////////

template <typename Matrix, typename Array,
          typename Format, typename MemorySpace>
size_t maximal_independent_set(const Matrix& A, Array& stencil, size_t k,
                               Format, MemorySpace)
{
  typedef typename Matrix::index_type   IndexType;
  typedef typename Matrix::value_type   ValueType;

  // convert matrix to CSR format and compute on the host
  cusp::csr_matrix<IndexType,ValueType,cusp::host_memory> A_csr(A);

  return cusp::graph::maximal_independent_set(A_csr, stencil, k);
}

} // end namespace detail

/////////////////
// Entry Point //
/////////////////

template <typename Matrix, typename Array>
size_t maximal_independent_set(const Matrix& A, Array& stencil, size_t k)
{
    CUSP_PROFILE_SCOPED();

    if(A.num_rows != A.num_cols)
        throw cusp::invalid_input_exception("matrix must be square");

    if (k == 0)
    {
        stencil.resize(A.num_rows);
        thrust::fill(stencil.begin(), stencil.end(), typename Array::value_type(1));
        return stencil.size();
    }
    else
    {
        return cusp::graph::detail::maximal_independent_set(A, stencil, k, typename Matrix::format(), typename Matrix::memory_space());
    }
}

} // end namespace graph
} // end namespace cusp

