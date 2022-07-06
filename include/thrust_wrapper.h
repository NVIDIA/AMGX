#pragma once 

#include <thrust/scan.h>
#include <thrust/reduce.h>
#include <thrust/sort.h>
#include <thrust/gather.h>
#include <thrust/count.h>
#include <thrust/transform.h>
#include <thrust/iterator/iterator_traits.h>
#include <vector_thrust_allocator.h>
#include <thrust/transform_reduce.h>
#include <thrust/adjacent_difference.h>

namespace thrust_wrapper
{
  template<typename InputIterator, typename OutputIterator>
    inline void exclusive_scan(InputIterator first, InputIterator last, OutputIterator result)
    {
      amgx::thrust_amgx_allocator<typename thrust::iterator_traits<InputIterator>::value_type> alloc;
#if THRUST_VERSION >= 101600
      thrust::exclusive_scan(thrust::cuda::par_nosync(alloc), first, last, result);
#else
      thrust::exclusive_scan(thrust::cuda::par(alloc), first, last, result);
#endif
    }

  template<typename InputIterator, typename OutputIterator, typename T>
    inline void exclusive_scan(InputIterator first, InputIterator last, OutputIterator result, T init)
    {
      amgx::thrust_amgx_allocator<typename thrust::iterator_traits<InputIterator>::value_type> alloc;
#if THRUST_VERSION >= 101600
      thrust::exclusive_scan(thrust::cuda::par_nosync(alloc), first, last, result, init);
#else
      thrust::exclusive_scan(thrust::cuda::par(alloc), first, last, result, init);
#endif
    }

  template<typename InputIterator, typename OutputIterator>
    inline void inclusive_scan(InputIterator first, InputIterator last, OutputIterator result)
    {
      amgx::thrust_amgx_allocator<typename thrust::iterator_traits<InputIterator>::value_type> alloc;
#if THRUST_VERSION >= 101600
      thrust::inclusive_scan(thrust::cuda::par_nosync(alloc), first, last, result);
#else
      thrust::inclusive_scan(thrust::cuda::par(alloc), first, last, result);
#endif
    }

  template<typename InputIterator, typename OutputIterator, typename AssociativeOperator>
    inline void inclusive_scan(InputIterator first, InputIterator last, OutputIterator result, AssociativeOperator binary_op)
    {
      amgx::thrust_amgx_allocator<typename thrust::iterator_traits<InputIterator>::value_type> alloc;
#if THRUST_VERSION >= 101600
      thrust::inclusive_scan(thrust::cuda::par_nosync(alloc), first, last, result, binary_op);
#else
      thrust::inclusive_scan(thrust::cuda::par(alloc), first, last, result, binary_op);
#endif
    }

  template<typename RandomAccessIterator>
    inline void sort(RandomAccessIterator first, RandomAccessIterator last)
    {
      amgx::thrust_amgx_allocator<typename thrust::iterator_traits<RandomAccessIterator>::value_type> alloc;
#if THRUST_VERSION >= 101600
      thrust::sort(thrust::cuda::par_nosync(alloc), first, last);
#else
      thrust::sort(thrust::cuda::par(alloc), first, last);
#endif
    }

  template<typename RandomAccessIterator1, typename RandomAccessIterator2>
    inline void sort_by_key(RandomAccessIterator1 keys_first, RandomAccessIterator1 keys_last, RandomAccessIterator2 values_first)
    {
      amgx::thrust_amgx_allocator<typename thrust::iterator_traits<RandomAccessIterator1>::value_type> alloc;
#if THRUST_VERSION >= 101600
      thrust::sort_by_key(thrust::cuda::par_nosync(alloc), keys_first, keys_last, values_first);
#else
      thrust::sort_by_key(thrust::cuda::par(alloc), keys_first, keys_last, values_first);
#endif
    }

  template<typename RandomAccessIterator1, typename RandomAccessIterator2>
    inline void stable_sort_by_key(RandomAccessIterator1 keys_first, RandomAccessIterator1 keys_last, RandomAccessIterator2 values_first)
    {
      amgx::thrust_amgx_allocator<typename thrust::iterator_traits<RandomAccessIterator1>::value_type> alloc;
#if THRUST_VERSION >= 101600
      thrust::stable_sort_by_key(thrust::cuda::par_nosync(alloc), keys_first, keys_last, values_first);
#else
      thrust::stable_sort_by_key(thrust::cuda::par(alloc), keys_first, keys_last, values_first);
#endif
    }

  template<typename InputIterator>
    inline typename thrust::iterator_traits<InputIterator>::value_type reduce(InputIterator first, InputIterator last)
    {
      amgx::thrust_amgx_allocator<typename thrust::iterator_traits<InputIterator>::value_type> alloc;
#if THRUST_VERSION >= 101600
      return thrust::reduce(thrust::cuda::par_nosync(alloc), first, last);
#else
      return thrust::reduce(thrust::cuda::par(alloc), first, last);
#endif
    }

  template<typename InputIterator, typename T, typename BinaryFunction>
    inline T reduce(InputIterator first, InputIterator last, T init, BinaryFunction binary_op)
    {
      amgx::thrust_amgx_allocator<typename thrust::iterator_traits<InputIterator>::value_type> alloc;
#if THRUST_VERSION >= 101600
      return thrust::reduce(thrust::cuda::par_nosync(alloc), first, last, init, binary_op);
#else
      return thrust::reduce(thrust::cuda::par(alloc), first, last, init, binary_op);
#endif
    }

  template<typename InputIterator, typename OutputIterator, typename UnaryFunction>
    inline OutputIterator transform(InputIterator first, InputIterator last, OutputIterator result, UnaryFunction op)
    {
      amgx::thrust_amgx_allocator<typename thrust::iterator_traits<InputIterator>::value_type> alloc;
#if THRUST_VERSION >= 101600
      return thrust::transform(thrust::cuda::par_nosync(alloc), first, last, result, op);
#else
      return thrust::transform(thrust::cuda::par(alloc), first, last, result, op);
#endif
    }

  template<typename InputIterator, typename OutputIterator, typename RandomAccessIterator>
    inline OutputIterator gather(InputIterator map_first, InputIterator map_last, RandomAccessIterator input_first, OutputIterator result)
    {
      amgx::thrust_amgx_allocator<typename thrust::iterator_traits<InputIterator>::value_type> alloc;
#if THRUST_VERSION >= 101600
      return thrust::gather(thrust::cuda::par_nosync(alloc), map_first, map_last, input_first, result);
#else
      return thrust::gather(thrust::cuda::par(alloc), map_first, map_last, input_first, result);
#endif
    }

  template<typename InputIterator , typename UnaryFunction , typename OutputType , typename BinaryFunction >
    inline OutputType transform_reduce(InputIterator first, InputIterator last, UnaryFunction unary_op, OutputType init, BinaryFunction binary_op)
    {
      amgx::thrust_amgx_allocator<typename thrust::iterator_traits<InputIterator>::value_type> alloc;
#if THRUST_VERSION >= 101600
      return thrust::transform_reduce(thrust::cuda::par_nosync(alloc), first, last, unary_op, init, binary_op);
#else
      return thrust::transform_reduce(thrust::cuda::par(alloc), first, last, unary_op, init, binary_op);
#endif
    }

  template<typename InputIterator, typename UnaryFunction>
    inline InputIterator for_each(InputIterator first, InputIterator last, UnaryFunction f)
    {
      amgx::thrust_amgx_allocator<typename thrust::iterator_traits<InputIterator>::value_type> alloc;
#if THRUST_VERSION >= 101600
      return thrust::for_each(thrust::cuda::par_nosync(alloc), first, last, f);
#else
      return thrust::for_each(thrust::cuda::par(alloc), first, last, f);
#endif
    }

  template<typename InputIterator , typename OutputIterator >
    inline OutputIterator copy(InputIterator first, InputIterator last, OutputIterator result, cudaStream_t stream = 0, bool sync_default = false)
    {
      if(sync_default) {
        cudaStreamSynchronize(0);
      }
#if THRUST_VERSION >= 101600
      OutputIterator res = thrust::copy(thrust::cuda::par_nosync.on(stream), first, last, result);
#else
      OutputIterator res = thrust::copy(thrust::cuda::par.on(stream), first, last, result);
#endif
      cudaStreamSynchronize(stream);
      return res;
    }

  template<typename InputIterator, typename EqualityComparable>
    inline typename thrust::iterator_traits<InputIterator>::difference_type count(InputIterator first, InputIterator last, const EqualityComparable& value) 
    {
      amgx::thrust_amgx_allocator<typename thrust::iterator_traits<InputIterator>::value_type> alloc;
#if THRUST_VERSION >= 101600
      return thrust::count(thrust::cuda::par_nosync(alloc), first, last, value);
#else
      return thrust::count(thrust::cuda::par(alloc), first, last, value);
#endif
    }

  template<typename InputIterator, typename Predicate >
    inline typename thrust::iterator_traits<InputIterator>::difference_type count_if(InputIterator first, InputIterator last, Predicate pred)
    {
      amgx::thrust_amgx_allocator<typename thrust::iterator_traits<InputIterator>::value_type> alloc;
#if THRUST_VERSION >= 101600
      return thrust::count_if(thrust::cuda::par_nosync(alloc), first, last, pred);
#else
      return thrust::count_if(thrust::cuda::par(alloc), first, last, pred);
#endif
    }

  template<typename InputIterator , typename OutputIterator >
    inline OutputIterator adjacent_difference(InputIterator first, InputIterator last, OutputIterator result)	
    {
      amgx::thrust_amgx_allocator<typename thrust::iterator_traits<InputIterator>::value_type> alloc;
#if THRUST_VERSION >= 101600
      return thrust::adjacent_difference(thrust::cuda::par_nosync(alloc), first, last, result);
#else
      return thrust::adjacent_difference(thrust::cuda::par(alloc), first, last, result);
#endif
    }
}
