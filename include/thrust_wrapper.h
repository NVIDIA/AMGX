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
      amgx::thrust_amgx_allocator<typename amgx::thrust::iterator_traits<InputIterator>::value_type> alloc;
#if THRUST_VERSION >= 101600
      amgx::thrust::exclusive_scan(amgx::thrust::cuda::par_nosync(alloc), first, last, result);
#else
      amgx::thrust::exclusive_scan(amgx::thrust::cuda::par(alloc), first, last, result);
#endif
    }

  template<typename InputIterator, typename OutputIterator, typename T>
    inline void exclusive_scan(InputIterator first, InputIterator last, OutputIterator result, T init)
    {
      amgx::thrust_amgx_allocator<typename amgx::thrust::iterator_traits<InputIterator>::value_type> alloc;
#if THRUST_VERSION >= 101600
      amgx::thrust::exclusive_scan(amgx::thrust::cuda::par_nosync(alloc), first, last, result, init);
#else
      amgx::thrust::exclusive_scan(amgx::thrust::cuda::par(alloc), first, last, result, init);
#endif
    }

  template<typename InputIterator, typename OutputIterator>
    inline void inclusive_scan(InputIterator first, InputIterator last, OutputIterator result)
    {
      amgx::thrust_amgx_allocator<typename amgx::thrust::iterator_traits<InputIterator>::value_type> alloc;
#if THRUST_VERSION >= 101600
      amgx::thrust::inclusive_scan(amgx::thrust::cuda::par_nosync(alloc), first, last, result);
#else
      amgx::thrust::inclusive_scan(amgx::thrust::cuda::par(alloc), first, last, result);
#endif
    }

  template<typename InputIterator, typename OutputIterator, typename AssociativeOperator>
    inline void inclusive_scan(InputIterator first, InputIterator last, OutputIterator result, AssociativeOperator binary_op)
    {
      amgx::thrust_amgx_allocator<typename amgx::thrust::iterator_traits<InputIterator>::value_type> alloc;
#if THRUST_VERSION >= 101600
      amgx::thrust::inclusive_scan(amgx::thrust::cuda::par_nosync(alloc), first, last, result, binary_op);
#else
      amgx::thrust::inclusive_scan(amgx::thrust::cuda::par(alloc), first, last, result, binary_op);
#endif
    }

  template<typename RandomAccessIterator>
    inline void sort(RandomAccessIterator first, RandomAccessIterator last)
    {
      amgx::thrust_amgx_allocator<typename amgx::thrust::iterator_traits<RandomAccessIterator>::value_type> alloc;
#if THRUST_VERSION >= 101600
      amgx::thrust::sort(amgx::thrust::cuda::par_nosync(alloc), first, last);
#else
      amgx::thrust::sort(amgx::thrust::cuda::par(alloc), first, last);
#endif
    }

  template<typename RandomAccessIterator1, typename RandomAccessIterator2>
    inline void sort_by_key(RandomAccessIterator1 keys_first, RandomAccessIterator1 keys_last, RandomAccessIterator2 values_first)
    {
      amgx::thrust_amgx_allocator<typename amgx::thrust::iterator_traits<RandomAccessIterator1>::value_type> alloc;
#if THRUST_VERSION >= 101600
      amgx::thrust::sort_by_key(amgx::thrust::cuda::par_nosync(alloc), keys_first, keys_last, values_first);
#else
      amgx::thrust::sort_by_key(amgx::thrust::cuda::par(alloc), keys_first, keys_last, values_first);
#endif
    }

  template<typename RandomAccessIterator1, typename RandomAccessIterator2>
    inline void stable_sort_by_key(RandomAccessIterator1 keys_first, RandomAccessIterator1 keys_last, RandomAccessIterator2 values_first)
    {
      amgx::thrust_amgx_allocator<typename amgx::thrust::iterator_traits<RandomAccessIterator1>::value_type> alloc;
#if THRUST_VERSION >= 101600
      amgx::thrust::stable_sort_by_key(amgx::thrust::cuda::par_nosync(alloc), keys_first, keys_last, values_first);
#else
      amgx::thrust::stable_sort_by_key(amgx::thrust::cuda::par(alloc), keys_first, keys_last, values_first);
#endif
    }

  template<typename InputIterator>
    inline typename amgx::thrust::iterator_traits<InputIterator>::value_type reduce(InputIterator first, InputIterator last)
    {
      amgx::thrust_amgx_allocator<typename amgx::thrust::iterator_traits<InputIterator>::value_type> alloc;
#if THRUST_VERSION >= 101600
      return amgx::thrust::reduce(amgx::thrust::cuda::par_nosync(alloc), first, last);
#else
      return amgx::thrust::reduce(amgx::thrust::cuda::par(alloc), first, last);
#endif
    }

  template<typename InputIterator, typename T, typename BinaryFunction>
    inline T reduce(InputIterator first, InputIterator last, T init, BinaryFunction binary_op)
    {
      amgx::thrust_amgx_allocator<typename amgx::thrust::iterator_traits<InputIterator>::value_type> alloc;
#if THRUST_VERSION >= 101600
      return amgx::thrust::reduce(amgx::thrust::cuda::par_nosync(alloc), first, last, init, binary_op);
#else
      return amgx::thrust::reduce(amgx::thrust::cuda::par(alloc), first, last, init, binary_op);
#endif
    }

  template<typename InputIterator, typename OutputIterator, typename UnaryFunction>
    inline OutputIterator transform(InputIterator first, InputIterator last, OutputIterator result, UnaryFunction op)
    {
      amgx::thrust_amgx_allocator<typename amgx::thrust::iterator_traits<InputIterator>::value_type> alloc;
#if THRUST_VERSION >= 101600
      return amgx::thrust::transform(amgx::thrust::cuda::par_nosync(alloc), first, last, result, op);
#else
      return amgx::thrust::transform(amgx::thrust::cuda::par(alloc), first, last, result, op);
#endif
    }

  template<typename InputIterator, typename OutputIterator, typename RandomAccessIterator>
    inline OutputIterator gather(InputIterator map_first, InputIterator map_last, RandomAccessIterator input_first, OutputIterator result)
    {
      amgx::thrust_amgx_allocator<typename amgx::thrust::iterator_traits<InputIterator>::value_type> alloc;
#if THRUST_VERSION >= 101600
      return amgx::thrust::gather(amgx::thrust::cuda::par_nosync(alloc), map_first, map_last, input_first, result);
#else
      return amgx::thrust::gather(amgx::thrust::cuda::par(alloc), map_first, map_last, input_first, result);
#endif
    }

  template<typename InputIterator , typename UnaryFunction , typename OutputType , typename BinaryFunction >
    inline OutputType transform_reduce(InputIterator first, InputIterator last, UnaryFunction unary_op, OutputType init, BinaryFunction binary_op)
    {
      amgx::thrust_amgx_allocator<typename amgx::thrust::iterator_traits<InputIterator>::value_type> alloc;
#if THRUST_VERSION >= 101600
      return amgx::thrust::transform_reduce(amgx::thrust::cuda::par_nosync(alloc), first, last, unary_op, init, binary_op);
#else
      return amgx::thrust::transform_reduce(amgx::thrust::cuda::par(alloc), first, last, unary_op, init, binary_op);
#endif
    }

  template<typename InputIterator, typename UnaryFunction>
    inline InputIterator for_each(InputIterator first, InputIterator last, UnaryFunction f)
    {
      amgx::thrust_amgx_allocator<typename amgx::thrust::iterator_traits<InputIterator>::value_type> alloc;
#if THRUST_VERSION >= 101600
      return amgx::thrust::for_each(amgx::thrust::cuda::par_nosync(alloc), first, last, f);
#else
      return amgx::thrust::for_each(amgx::thrust::cuda::par(alloc), first, last, f);
#endif
    }

  template<typename InputIterator , typename OutputIterator >
    inline OutputIterator copy(InputIterator first, InputIterator last, OutputIterator result, cudaStream_t stream = 0, bool sync_default = false)
    {
      if(sync_default) {
        cudaStreamSynchronize(0);
      }
#if THRUST_VERSION >= 101600
      OutputIterator res = amgx::thrust::copy(amgx::thrust::cuda::par_nosync.on(stream), first, last, result);
#else
      OutputIterator res = amgx::thrust::copy(amgx::thrust::cuda::par.on(stream), first, last, result);
#endif
      cudaStreamSynchronize(stream);
      return res;
    }

  template<typename InputIterator, typename EqualityComparable>
    inline typename amgx::thrust::iterator_traits<InputIterator>::difference_type count(InputIterator first, InputIterator last, const EqualityComparable& value) 
    {
      amgx::thrust_amgx_allocator<typename amgx::thrust::iterator_traits<InputIterator>::value_type> alloc;
#if THRUST_VERSION >= 101600
      return amgx::thrust::count(amgx::thrust::cuda::par_nosync(alloc), first, last, value);
#else
      return amgx::thrust::count(amgx::thrust::cuda::par(alloc), first, last, value);
#endif
    }

  template<typename InputIterator, typename Predicate >
    inline typename amgx::thrust::iterator_traits<InputIterator>::difference_type count_if(InputIterator first, InputIterator last, Predicate pred)
    {
      amgx::thrust_amgx_allocator<typename amgx::thrust::iterator_traits<InputIterator>::value_type> alloc;
#if THRUST_VERSION >= 101600
      return amgx::thrust::count_if(amgx::thrust::cuda::par_nosync(alloc), first, last, pred);
#else
      return amgx::thrust::count_if(amgx::thrust::cuda::par(alloc), first, last, pred);
#endif
    }

  template<typename InputIterator , typename OutputIterator >
    inline OutputIterator adjacent_difference(InputIterator first, InputIterator last, OutputIterator result)	
    {
      amgx::thrust_amgx_allocator<typename amgx::thrust::iterator_traits<InputIterator>::value_type> alloc;
#if THRUST_VERSION >= 101600
      return amgx::thrust::adjacent_difference(amgx::thrust::cuda::par_nosync(alloc), first, last, result);
#else
      return amgx::thrust::adjacent_difference(amgx::thrust::cuda::par(alloc), first, last, result);
#endif
    }
}
