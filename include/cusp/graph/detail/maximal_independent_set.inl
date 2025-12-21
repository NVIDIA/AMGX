// SPDX-FileCopyrightText: 2008 - 2025 NVIDIA CORPORATION. All Rights Reserved.
//
// SPDX-License-Identifier: BSD-3-Clause

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
        if (amgx::thrust::get<1>(t) == 1)                     // undecided node
        {
            if (amgx::thrust::get<0>(t) == amgx::thrust::get<3>(t)) // i == maximal_index
              amgx::thrust::get<1>(t) = 2;                    // mis_node
        }
    }
};

struct process_non_mis_nodes
{
    template <typename Tuple>
    __host__ __device__
    void operator()(Tuple t)
    {
        if (amgx::thrust::get<0>(t) == 1)            // undecided node
        {
            if (amgx::thrust::get<1>(t) == 2)        // maximal_state == mis_node
              amgx::thrust::get<0>(t) = 0;           // non_mis_node
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
    return amgx::thrust::get<0>(t) && amgx::thrust::get<1>(t);
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

    typedef typename amgx::thrust::tuple<NodeStateType,RandomType,IndexType> Tuple;
    
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
             row_offsets.begin(), column_indices.begin(), amgx::thrust::constant_iterator<Tuple>(Tuple(0,0)),  // XXX should we mask explicit zeros? (e.g. DIA, array2d)
             amgx::thrust::make_zip_iterator(amgx::thrust::make_tuple(states.begin(), random_values.begin(), amgx::thrust::counting_iterator<IndexType>(0))),
             amgx::thrust::make_zip_iterator(amgx::thrust::make_tuple(states.begin(), random_values.begin(), amgx::thrust::counting_iterator<IndexType>(0))),
             amgx::thrust::make_zip_iterator(amgx::thrust::make_tuple(maximal_states.begin(), maximal_values.begin(), maximal_indices.begin())),
             amgx::thrust::project2nd<Tuple,Tuple>(), amgx::thrust::maximum<Tuple>());
        //cusp::detail::device::cuda::spmv_coo
        //    (num_rows, num_entries,
        //     row_indices.begin(), column_indices.begin(), amgx::thrust::constant_iterator<Tuple>(Tuple(0,0)),  // XXX should we mask explicit zeros? (e.g. DIA, array2d)
        //     amgx::thrust::make_zip_iterator(amgx::thrust::make_tuple(states.begin(), random_values.begin(), amgx::thrust::counting_iterator<IndexType>(0))),
        //     amgx::thrust::make_zip_iterator(amgx::thrust::make_tuple(states.begin(), random_values.begin(), amgx::thrust::counting_iterator<IndexType>(0))),
        //     amgx::thrust::make_zip_iterator(amgx::thrust::make_tuple(maximal_states.begin(), maximal_values.begin(), maximal_indices.begin())),
        //     amgx::thrust::project2nd<Tuple,Tuple>(), amgx::thrust::maximum<Tuple>());

        // find the largest (state,value,index) k-ring neighbor for each node (if k > 1)
        for(size_t ring = 1; ring < k; ring++)
        {
            last_states.resize (N); last_states.swap (maximal_states);
            last_values.resize (N); last_values.swap (maximal_values);
            last_indices.resize(N); last_indices.swap(maximal_indices);

            // TODO replace with call to generalized method
            cusp::detail::device::cuda::spmv_csr_scalar
                (num_rows,
                 row_offsets.begin(), column_indices.begin(), amgx::thrust::constant_iterator<Tuple>(Tuple(0,0)),  // XXX should we mask explicit zeros? (e.g. DIA, array2d)
                 amgx::thrust::make_zip_iterator(amgx::thrust::make_tuple(last_states.begin(), last_values.begin(), last_indices.begin())),
                 amgx::thrust::make_zip_iterator(amgx::thrust::make_tuple(last_states.begin(), last_values.begin(), last_indices.begin())),
                 amgx::thrust::make_zip_iterator(amgx::thrust::make_tuple(maximal_states.begin(), maximal_values.begin(), maximal_indices.begin())),
                 amgx::thrust::project2nd<Tuple,Tuple>(), amgx::thrust::maximum<Tuple>());
        }
       
        // label local maxima as MIS nodes
        amgx::thrust::for_each(amgx::thrust::make_zip_iterator(amgx::thrust::make_tuple(amgx::thrust::counting_iterator<IndexType>(0), states.begin(), maximal_states.begin(), maximal_indices.begin())),
                         amgx::thrust::make_zip_iterator(amgx::thrust::make_tuple(amgx::thrust::counting_iterator<IndexType>(0), states.begin(), maximal_states.begin(), maximal_indices.begin())) + N,
                         process_mis_nodes());
        
        // label k-ring neighbors of MIS nodes as non-MIS nodes
        amgx::thrust::for_each(amgx::thrust::make_zip_iterator(amgx::thrust::make_tuple(states.begin(), amgx::thrust::make_permutation_iterator(states.begin(), maximal_indices.begin()))),
                         amgx::thrust::make_zip_iterator(amgx::thrust::make_tuple(states.begin(), amgx::thrust::make_permutation_iterator(states.begin(), maximal_indices.begin()))) + N,
                         process_non_mis_nodes());

        active_nodes = amgx::thrust::count(states.begin(), states.end(), 1);
        
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
//            thrust_wrapper::transform(maximal_states.begin(), maximal_states.end(), amgx::thrust::constant_iterator<NodeStateType>(1), retained_nodes.begin(), amgx::thrust::equal_to<NodeStateType>());
//
//            // propagate retained region outward
//            for(size_t ring = 1; 2*ring <= k; ring++)
//            {
//                retained_nodes.swap(last_retained_nodes);
//
//                // TODO replace with call to generalized method
//                cusp::detail::device::cuda::spmv_coo
//                    (num_rows, num_entries,
//                     row_indices.begin(), column_indices.begin(), amgx::thrust::constant_iterator<bool>(false), 
//                     last_retained_nodes.begin(),
//                     last_retained_nodes.begin(),
//                     retained_nodes.begin(),
//                     amgx::thrust::project2nd<bool,bool>(), amgx::thrust::logical_or<bool>());
//            }
//        
//            std::cout << "retained nodes\n";
//            cusp::print(retained_nodes);
//
//            size_t num_subgraph_nodes = amgx::thrust::count(retained_nodes.begin(), retained_nodes.end(), true);
//            size_t num_subgraph_edges = amgx::thrust::count
//                (amgx::thrust::make_zip_iterator(amgx::thrust::make_tuple(amgx::thrust::make_permutation_iterator(retained_nodes.begin(), row_indices.begin()),
//                                                              amgx::thrust::make_permutation_iterator(retained_nodes.begin(), column_indices.begin()))),
//                 amgx::thrust::make_zip_iterator(amgx::thrust::make_tuple(amgx::thrust::make_permutation_iterator(retained_nodes.begin(), row_indices.end()),
//                                                              amgx::thrust::make_permutation_iterator(retained_nodes.begin(), column_indices.end()))),
//                 amgx::thrust::make_tuple(true,true));
//
//
//            std::cout << "subgraph nodes: " << double(100*num_subgraph_nodes)/N << "% edges " << double(100*num_subgraph_edges)/num_entries << "%" << std::endl;
//
//            // map old indices into subgraph indices
//            cusp::array1d<IndexType, MemorySpace> index_map(N);
//            amgx::thrust::transform_exclusive_scan(retained_nodes.begin(), retained_nodes.end(), index_map.begin(), identity_function<IndexType>(), IndexType(0), amgx::thrust::plus<IndexType>());
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
//            amgx::thrust::copy_if
//                (amgx::thrust::make_zip_iterator(amgx::thrust::make_tuple(amgx::thrust::make_permutation_iterator(index_map.begin(), row_indices.begin()),
//                                                              amgx::thrust::make_permutation_iterator(index_map.begin(), column_indices.begin()))),
//                 amgx::thrust::make_zip_iterator(amgx::thrust::make_tuple(amgx::thrust::make_permutation_iterator(index_map.begin(), row_indices.end()),
//                                                              amgx::thrust::make_permutation_iterator(index_map.begin(), column_indices.end()))),
//                 amgx::thrust::make_zip_iterator(amgx::thrust::make_tuple(amgx::thrust::make_permutation_iterator(retained_nodes.begin(), row_indices.begin()),
//                                                              amgx::thrust::make_permutation_iterator(retained_nodes.begin(), column_indices.begin()))),
//                 amgx::thrust::make_zip_iterator(amgx::thrust::make_tuple(subgraph_row_indices.begin(),
//                                                              subgraph_column_indices.begin())),
//                 is_subgraph_edge());
//
//            amgx::thrust::scatter_if
//                (amgx::thrust::make_zip_iterator(amgx::thrust::make_tuple(states.begin(), random_values.begin())),
//                 amgx::thrust::make_zip_iterator(amgx::thrust::make_tuple(states.end(),   random_values.end())),
//                 index_map.begin(),
//                 retained_nodes.begin(),
//                 amgx::thrust::make_zip_iterator(amgx::thrust::make_tuple(subgraph_states.begin(), subgraph_random_values.begin())));
//        
//    
//            compute_mis_states(k, subgraph_row_indices, subgraph_column_indices, subgraph_random_values, subgraph_states);
//
//            // update active node states from subgraph
//            amgx::thrust::gather_if(index_map.begin(), index_map.end(),
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
    thrust_wrapper::transform(states.begin(), states.end(), amgx::thrust::constant_iterator<NodeStateType>(2), stencil.begin(), amgx::thrust::equal_to<NodeStateType>());

    // return the size of the MIS
    return amgx::thrust::count(stencil.begin(), stencil.end(), typename Array::value_type(true));
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
        thrust_wrapper::fill(stencil.begin(), stencil.end(), typename Array::value_type(1));
        return stencil.size();
    }
    else
    {
        return cusp::graph::detail::maximal_independent_set(A, stencil, k, typename Matrix::format(), typename Matrix::memory_space());
    }
}

} // end namespace graph
} // end namespace cusp

