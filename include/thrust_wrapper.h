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
#include <type_traits>

namespace thrust_wrapper
{
  template<int MemSpace, typename InputIterator, typename OutputIterator>
    inline void exclusive_scan(InputIterator first, InputIterator last, OutputIterator result)
    {
        if(MemSpace == AMGX_host) {
            amgx::thrust::exclusive_scan(first, last, result);
        }
        else {
            amgx::thrust_amgx_allocator<typename amgx::thrust::iterator_traits<InputIterator>::value_type> alloc;
            amgx::thrust::exclusive_scan(amgx::thrust::cuda::par_nosync(alloc), first, last, result);
        }
    }

  template<int MemSpace, typename InputIterator, typename OutputIterator, typename T>
    inline void exclusive_scan(InputIterator first, InputIterator last, OutputIterator result, T init)
    {
        if(MemSpace == AMGX_host) {
            amgx::thrust::exclusive_scan(first, last, result, init);
        }
        else {
            amgx::thrust_amgx_allocator<typename amgx::thrust::iterator_traits<InputIterator>::value_type> alloc;
            amgx::thrust::exclusive_scan(amgx::thrust::cuda::par_nosync(alloc), first, last, result, init);
        }
    }

  template<int MemSpace, typename InputIterator1, typename InputIterator2, typename OutputIterator, typename T, typename BinaryPredicate, typename AssociativeOperator>
    OutputIterator exclusive_scan_by_key(InputIterator1 first1, InputIterator1 last1, InputIterator2 first2, OutputIterator result, T init, BinaryPredicate binary_pred, AssociativeOperator binary_op)
    {
        if(MemSpace == AMGX_host) {
            amgx::thrust::exclusive_scan_by_key(first1, last1, first2, result, init, binary_pred, binary_op);
        }
        else {
            amgx::thrust_amgx_allocator<typename amgx::thrust::iterator_traits<InputIterator1>::value_type> alloc;
            amgx::thrust::exclusive_scan_by_key(amgx::thrust::cuda::par_nosync(alloc), first1, last1, first2, result, init, binary_pred, binary_op);
        }
    }

  template<int MemSpace, typename InputIterator, typename OutputIterator>
    inline void inclusive_scan(InputIterator first, InputIterator last, OutputIterator result)
    {
        if(MemSpace == AMGX_host) {
            amgx::thrust::inclusive_scan(first, last, result);
        }
        else {
            amgx::thrust_amgx_allocator<typename amgx::thrust::iterator_traits<InputIterator>::value_type> alloc;
            amgx::thrust::inclusive_scan(amgx::thrust::cuda::par_nosync(alloc), first, last, result);
        }
    }

  template<int MemSpace, typename InputIterator, typename OutputIterator, typename AssociativeOperator>
    inline void inclusive_scan(InputIterator first, InputIterator last, OutputIterator result, AssociativeOperator binary_op)
    {
        if(MemSpace == AMGX_host) {
            amgx::thrust::inclusive_scan(first, last, result, binary_op);
        }
        else {
            amgx::thrust_amgx_allocator<typename amgx::thrust::iterator_traits<InputIterator>::value_type> alloc;
            amgx::thrust::inclusive_scan(amgx::thrust::cuda::par_nosync(alloc), first, last, result, binary_op);
        }
    }

  template<int MemSpace, typename RandomAccessIterator>
    inline void sort(RandomAccessIterator first, RandomAccessIterator last)
    {
        if(MemSpace == AMGX_host) {
            amgx::thrust::sort(first, last);
        }
        else {
            amgx::thrust_amgx_allocator<typename amgx::thrust::iterator_traits<RandomAccessIterator>::value_type> alloc;
            amgx::thrust::sort(amgx::thrust::cuda::par_nosync(alloc), first, last);
        }
    }

  template<int MemSpace, typename RandomAccessIterator1, typename RandomAccessIterator2>
    inline void sort_by_key(RandomAccessIterator1 keys_first, RandomAccessIterator1 keys_last, RandomAccessIterator2 values_first)
    {
        if(MemSpace == AMGX_host) {
            amgx::thrust::sort_by_key(keys_first, keys_last, values_first);
        }
        else {
            amgx::thrust_amgx_allocator<typename amgx::thrust::iterator_traits<RandomAccessIterator1>::value_type> alloc;
            amgx::thrust::sort_by_key(amgx::thrust::cuda::par_nosync(alloc), keys_first, keys_last, values_first);
        }
    }

  template<int MemSpace, typename RandomAccessIterator1, typename RandomAccessIterator2>
    inline void stable_sort_by_key(RandomAccessIterator1 keys_first, RandomAccessIterator1 keys_last, RandomAccessIterator2 values_first)
    {
        if(MemSpace == AMGX_host) {
            amgx::thrust::stable_sort_by_key(keys_first, keys_last, values_first);
        }
        else {
            amgx::thrust_amgx_allocator<typename amgx::thrust::iterator_traits<RandomAccessIterator1>::value_type> alloc;
            amgx::thrust::stable_sort_by_key(amgx::thrust::cuda::par_nosync(alloc), keys_first, keys_last, values_first);
        }
    }

  template<int MemSpace, typename InputIterator>
    inline typename amgx::thrust::iterator_traits<InputIterator>::value_type reduce(InputIterator first, InputIterator last)
    {
        if(MemSpace == AMGX_host) {
            return amgx::thrust::reduce(first, last);
        }
        else {
            amgx::thrust_amgx_allocator<typename amgx::thrust::iterator_traits<InputIterator>::value_type> alloc;
            return amgx::thrust::reduce(amgx::thrust::cuda::par(alloc), first, last);
        }
    }

  template<int MemSpace, typename InputIterator, typename T, typename BinaryFunction>
    inline T reduce(InputIterator first, InputIterator last, T init, BinaryFunction binary_op)
    {
        if(MemSpace == AMGX_host) {
            return amgx::thrust::reduce(first, last, init, binary_op);
        }
        else {
            amgx::thrust_amgx_allocator<typename amgx::thrust::iterator_traits<InputIterator>::value_type> alloc;
            return amgx::thrust::reduce(amgx::thrust::cuda::par(alloc), first, last, init, binary_op);
        }
    }

  template<int MemSpace, typename InputIterator, typename OutputIterator, typename UnaryFunction>
    inline OutputIterator transform(InputIterator first, InputIterator last, OutputIterator result, UnaryFunction op)
    {
        if(MemSpace == AMGX_host) {
          return amgx::thrust::transform(first, last, result, op);
        }
        else {
          amgx::thrust_amgx_allocator<typename amgx::thrust::iterator_traits<InputIterator>::value_type> alloc;
          return amgx::thrust::transform(amgx::thrust::cuda::par_nosync(alloc), first, last, result, op);
        }
    }

  template<int MemSpace, typename InputIterator1, typename InputIterator2, typename OutputIterator, typename UnaryFunction>
    inline OutputIterator transform(InputIterator1 first1, InputIterator1 last1, InputIterator2 first2, OutputIterator result, UnaryFunction op)
    {
        if(MemSpace == AMGX_host) {
            return amgx::thrust::transform(first1, last1, first2, result, op);
        }
        else {
            amgx::thrust_amgx_allocator<typename amgx::thrust::iterator_traits<InputIterator1>::value_type> alloc;
            return amgx::thrust::transform(amgx::thrust::cuda::par_nosync(alloc), first1, last1, first2, result, op);
        }
    }

  template<int MemSpace, typename InputIterator, typename OutputIterator, typename RandomAccessIterator>
    inline OutputIterator gather(InputIterator map_first, InputIterator map_last, RandomAccessIterator input_first, OutputIterator result)
    {
        if(MemSpace == AMGX_host) {
            return amgx::thrust::gather(map_first, map_last, input_first, result);
        }
        else {
            amgx::thrust_amgx_allocator<typename amgx::thrust::iterator_traits<InputIterator>::value_type> alloc;
            return amgx::thrust::gather(amgx::thrust::cuda::par_nosync(alloc), map_first, map_last, input_first, result);
        }
    }

  template<int MemSpace, typename InputIterator , typename UnaryFunction , typename OutputType , typename BinaryFunction >
    inline OutputType transform_reduce(InputIterator first, InputIterator last, UnaryFunction unary_op, OutputType init, BinaryFunction binary_op)
    {
        if(MemSpace == AMGX_host) {
            return amgx::thrust::transform_reduce(first, last, unary_op, init, binary_op);
        }
        else {
            amgx::thrust_amgx_allocator<typename amgx::thrust::iterator_traits<InputIterator>::value_type> alloc;
            return amgx::thrust::transform_reduce(amgx::thrust::cuda::par(alloc), first, last, unary_op, init, binary_op);
        }
    }

  template<int MemSpace, typename InputIterator, typename UnaryFunction>
    inline InputIterator for_each(InputIterator first, InputIterator last, UnaryFunction f)
    {
        if(MemSpace == AMGX_host) {
            return amgx::thrust::for_each(first, last, f);
        }
        else {
            amgx::thrust_amgx_allocator<typename amgx::thrust::iterator_traits<InputIterator>::value_type> alloc;
            return amgx::thrust::for_each(amgx::thrust::cuda::par_nosync(alloc), first, last, f);
        }
    }

  template<int MemSpace, typename InputIterator , typename OutputIterator >
    inline OutputIterator copy(InputIterator first, InputIterator last, OutputIterator result, cudaStream_t stream = 0, bool sync_default = false)
    {
        if(MemSpace == AMGX_host) {
            OutputIterator res = amgx::thrust::copy(first, last, result);
            return res;
        }
        else {
            if(sync_default) {
                cudaStreamSynchronize(0);
            }
            OutputIterator res = amgx::thrust::copy(amgx::thrust::cuda::par_nosync.on(stream), first, last, result);
            cudaStreamSynchronize(stream);
            return res;
        }
    }

  template<int MemSpace, typename InputIterator, typename EqualityComparable>
    inline typename amgx::thrust::iterator_traits<InputIterator>::difference_type count(InputIterator first, InputIterator last, const EqualityComparable& value) 
    {
        if(MemSpace == AMGX_host) {
            return amgx::thrust::count(first, last, value);
        }
        else {
            amgx::thrust_amgx_allocator<typename amgx::thrust::iterator_traits<InputIterator>::value_type> alloc;
            return amgx::thrust::count(amgx::thrust::cuda::par_nosync(alloc), first, last, value);
        }
    }

  template<int MemSpace, typename InputIterator, typename Predicate >
    inline typename amgx::thrust::iterator_traits<InputIterator>::difference_type count_if(InputIterator first, InputIterator last, Predicate pred)
    {
        if(MemSpace == AMGX_host) {
            return amgx::thrust::count_if(first, last, pred);
        }
        else {
            amgx::thrust_amgx_allocator<typename amgx::thrust::iterator_traits<InputIterator>::value_type> alloc;
            return amgx::thrust::count_if(amgx::thrust::cuda::par_nosync(alloc), first, last, pred);
        }
    }

  template<int MemSpace, typename InputIterator , typename OutputIterator >
    inline OutputIterator adjacent_difference(InputIterator first, InputIterator last, OutputIterator result)	
    {
        if(MemSpace == AMGX_host) {
            return amgx::thrust::adjacent_difference(first, last, result);
        }
        else {
            amgx::thrust_amgx_allocator<typename amgx::thrust::iterator_traits<InputIterator>::value_type> alloc;
            return amgx::thrust::adjacent_difference(amgx::thrust::cuda::par_nosync(alloc), first, last, result);
        }
    }

  template<int MemSpace, typename ForwardIterator, typename T>
      inline void fill(ForwardIterator first, ForwardIterator last, const T &value)
      {
          if(MemSpace == AMGX_host) {
            amgx::thrust::fill(first, last, value);
          }
          else {
            amgx::thrust_amgx_allocator<typename amgx::thrust::iterator_traits<ForwardIterator>::value_type> alloc;
            amgx::thrust::fill(amgx::thrust::cuda::par_nosync(alloc), first, last, value);
          }
      }

  template<int MemSpace, typename ForwardIterator>
      void sequence(ForwardIterator first, ForwardIterator last)
      {
          if(MemSpace == AMGX_host) {
              amgx::thrust::sequence(first, last);
          }
          else {
              amgx::thrust_amgx_allocator<typename amgx::thrust::iterator_traits<ForwardIterator>::value_type> alloc;
              amgx::thrust::sequence(amgx::thrust::cuda::par_nosync(alloc), first, last);
          }
      }

  template<int MemSpace, typename ForwardIterator, typename T>
      void sequence(ForwardIterator first, ForwardIterator last, T init)
      {
          if(MemSpace == AMGX_host) {
              amgx::thrust::sequence(first, last, init);
          }
          else {
              amgx::thrust_amgx_allocator<typename amgx::thrust::iterator_traits<ForwardIterator>::value_type> alloc;
              amgx::thrust::sequence(amgx::thrust::cuda::par_nosync(alloc), first, last, init);
          }
      }
}
