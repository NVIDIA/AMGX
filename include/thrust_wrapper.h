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
#include <basic_types.h>

template <class T>
using amgx_thrust_host_allocator = amgx::thrust_amgx_allocator<typename amgx::thrust::iterator_traits<T>::value_type, AMGX_host>;

// Picks device allocator to pass to execution policy
template <class T>
using amgx_thrust_device_allocator = amgx::thrust_amgx_allocator<typename amgx::thrust::iterator_traits<T>::value_type, AMGX_device>;

template<class InputIterator>
auto amgx_thrust_get_allocator(std::false_type)
{
    return amgx::thrust::host(amgx_thrust_host_allocator<InputIterator>());
}

template<class InputIterator>
auto amgx_thrust_get_allocator(std::true_type)
{
    return amgx::thrust::cuda::par_nosync(amgx_thrust_device_allocator<InputIterator>());
}

namespace thrust_wrapper
{
  template<int MemSpace, typename InputIterator, typename OutputIterator>
    inline typename std::enable_if<MemSpace == AMGX_host, void>::type 
    exclusive_scan(InputIterator first, InputIterator last, OutputIterator result)
    {
        amgx::thrust::exclusive_scan(first, last, result);
    }

  template<int MemSpace, typename InputIterator, typename OutputIterator>
    inline typename std::enable_if<MemSpace == AMGX_device, void>::type 
    exclusive_scan(InputIterator first, InputIterator last, OutputIterator result)
    {
        amgx::thrust::exclusive_scan(amgx_thrust_get_allocator<InputIterator>(std::integral_constant<bool, MemSpace == AMGX_device>()), first, last, result);
    }

  template<int MemSpace, typename InputIterator, typename OutputIterator, typename T>
    inline void exclusive_scan(InputIterator first, InputIterator last, OutputIterator result, T init)
    {
        amgx::thrust::exclusive_scan(amgx_thrust_get_allocator<InputIterator>(std::integral_constant<bool, MemSpace == AMGX_device>()), first, last, result, init);
    }

  template<int MemSpace, typename InputIterator1, typename InputIterator2, typename OutputIterator, typename T, typename BinaryPredicate, typename AssociativeOperator>
    OutputIterator exclusive_scan_by_key(InputIterator1 first1, InputIterator1 last1, InputIterator2 first2, OutputIterator result, T init, BinaryPredicate binary_pred, AssociativeOperator binary_op)
    {
        amgx::thrust::exclusive_scan_by_key(amgx_thrust_get_allocator<InputIterator1>(std::integral_constant<bool, MemSpace == AMGX_device>()), first1, last1, first2, result, init, binary_pred, binary_op);
    }

  template<int MemSpace, typename InputIterator, typename OutputIterator>
    inline void inclusive_scan(InputIterator first, InputIterator last, OutputIterator result)
    {
        amgx::thrust::inclusive_scan(amgx_thrust_get_allocator<InputIterator>(std::integral_constant<bool, MemSpace == AMGX_device>()), first, last, result);
    }

  template<int MemSpace, typename InputIterator, typename OutputIterator, typename AssociativeOperator>
    inline void inclusive_scan(InputIterator first, InputIterator last, OutputIterator result, AssociativeOperator binary_op)
    {
        amgx::thrust::inclusive_scan(amgx_thrust_get_allocator<InputIterator>(std::integral_constant<bool, MemSpace == AMGX_device>()), first, last, result, binary_op);
    }

  template<int MemSpace, typename RandomAccessIterator>
    inline void sort(RandomAccessIterator first, RandomAccessIterator last)
    {
        amgx::thrust::sort(amgx_thrust_get_allocator<RandomAccessIterator>(std::integral_constant<bool, MemSpace == AMGX_device>()), first, last);
    }

  template<int MemSpace, typename RandomAccessIterator1, typename RandomAccessIterator2>
    inline void sort_by_key(RandomAccessIterator1 keys_first, RandomAccessIterator1 keys_last, RandomAccessIterator2 values_first)
    {
        amgx::thrust::sort_by_key(amgx_thrust_get_allocator<RandomAccessIterator1>(std::integral_constant<bool, MemSpace == AMGX_device>()), keys_first, keys_last, values_first);
    }

  template<int MemSpace, typename RandomAccessIterator1, typename RandomAccessIterator2>
    inline void stable_sort_by_key(RandomAccessIterator1 keys_first, RandomAccessIterator1 keys_last, RandomAccessIterator2 values_first)
    {
        amgx::thrust::stable_sort_by_key(amgx_thrust_get_allocator<RandomAccessIterator1>(std::integral_constant<bool, MemSpace == AMGX_device>()), keys_first, keys_last, values_first);
    }

  template<int MemSpace, typename InputIterator>
    inline typename amgx::thrust::iterator_traits<InputIterator>::value_type reduce(InputIterator first, InputIterator last)
    {
        return amgx::thrust::reduce(amgx_thrust_get_allocator<InputIterator>(std::integral_constant<bool, MemSpace == AMGX_device>()), first, last);
    }

  template<int MemSpace, typename InputIterator, typename T, typename BinaryFunction>
    inline T reduce(InputIterator first, InputIterator last, T init, BinaryFunction binary_op)
    {
        return amgx::thrust::reduce(amgx_thrust_get_allocator<InputIterator>(std::integral_constant<bool, MemSpace == AMGX_device>()), first, last, init, binary_op);
    }

  template<int MemSpace, typename InputIterator, typename OutputIterator, typename UnaryFunction>
    inline typename std::enable_if<MemSpace == AMGX_host, OutputIterator>::type 
    transform(InputIterator first, InputIterator last, OutputIterator result, UnaryFunction op)
    {
        return amgx::thrust::transform(first, last, result, op);
    }

  template<int MemSpace, typename InputIterator, typename OutputIterator, typename UnaryFunction>
    inline typename std::enable_if<MemSpace == AMGX_device, OutputIterator>::type 
    transform(InputIterator first, InputIterator last, OutputIterator result, UnaryFunction op)
    {
        return amgx::thrust::transform(amgx_thrust_get_allocator<InputIterator>(std::integral_constant<bool, MemSpace == AMGX_device>()), first, last, result, op);
    }

  template<int MemSpace, typename InputIterator1, typename InputIterator2, typename OutputIterator, typename UnaryFunction>
    inline typename std::enable_if<MemSpace == AMGX_host, OutputIterator>::type 
    transform(InputIterator1 first1, InputIterator1 last1, InputIterator2 first2, OutputIterator result, UnaryFunction op)
    {
        return amgx::thrust::transform(first1, last1, first2, result, op);
    }

  template<int MemSpace, typename InputIterator1, typename InputIterator2, typename OutputIterator, typename UnaryFunction>
    inline typename std::enable_if<MemSpace == AMGX_device, OutputIterator>::type 
    transform(InputIterator1 first1, InputIterator1 last1, InputIterator2 first2, OutputIterator result, UnaryFunction op)
    {
        return amgx::thrust::transform(amgx_thrust_get_allocator<InputIterator1>(std::integral_constant<bool, MemSpace == AMGX_device>()), first1, last1, first2, result, op);
    }

  template<int MemSpace, typename InputIterator, typename OutputIterator, typename RandomAccessIterator>
    inline OutputIterator gather(InputIterator map_first, InputIterator map_last, RandomAccessIterator input_first, OutputIterator result)
    {
        return amgx::thrust::gather(amgx_thrust_get_allocator<InputIterator>(std::integral_constant<bool, MemSpace == AMGX_device>()), map_first, map_last, input_first, result);
    }

  template<int MemSpace, typename InputIterator , typename UnaryFunction , typename OutputType , typename BinaryFunction >
    inline OutputType transform_reduce(InputIterator first, InputIterator last, UnaryFunction unary_op, OutputType init, BinaryFunction binary_op)
    {
        return amgx::thrust::transform_reduce(amgx_thrust_get_allocator<InputIterator>(std::integral_constant<bool, MemSpace == AMGX_device>()), first, last, unary_op, init, binary_op);
    }

  template<int MemSpace, typename InputIterator, typename UnaryFunction>
    inline InputIterator for_each(InputIterator first, InputIterator last, UnaryFunction f)
    {
        return amgx::thrust::for_each(amgx_thrust_get_allocator<InputIterator>(std::integral_constant<bool, MemSpace == AMGX_device>()), first, last, f);
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
        return amgx::thrust::count(amgx_thrust_get_allocator<InputIterator>(std::integral_constant<bool, MemSpace == AMGX_device>()), first, last, value);
    }

  template<int MemSpace, typename InputIterator, typename Predicate >
    inline typename amgx::thrust::iterator_traits<InputIterator>::difference_type count_if(InputIterator first, InputIterator last, Predicate pred)
    {
        return amgx::thrust::count_if(amgx_thrust_get_allocator<InputIterator>(std::integral_constant<bool, MemSpace == AMGX_device>()), first, last, pred);
    }

  template<int MemSpace, typename InputIterator , typename OutputIterator >
    inline OutputIterator adjacent_difference(InputIterator first, InputIterator last, OutputIterator result)	
    {
        return amgx::thrust::adjacent_difference(amgx_thrust_get_allocator<InputIterator>(std::integral_constant<bool, MemSpace == AMGX_device>()), first, last, result);
    }

  template<int MemSpace, typename ForwardIterator, typename T>
      inline void fill(ForwardIterator first, ForwardIterator last, const T &value)
      {
          amgx::thrust::fill(amgx_thrust_get_allocator<ForwardIterator>(std::integral_constant<bool, MemSpace == AMGX_device>()), first, last, value);
      }

  template<int MemSpace, typename ForwardIterator>
      inline typename std::enable_if<MemSpace == AMGX_host, void>::type 
      sequence(ForwardIterator first, ForwardIterator last)
      {
          amgx::thrust::sequence(first, last);
      }

  template<int MemSpace, typename ForwardIterator>
      inline typename std::enable_if<MemSpace == AMGX_device, void>::type 
      sequence(ForwardIterator first, ForwardIterator last)
      {
          amgx::thrust::sequence(amgx_thrust_get_allocator<ForwardIterator>(std::integral_constant<bool, MemSpace == AMGX_device>()), first, last);
      }

  template<int MemSpace, typename ForwardIterator, typename T>
      inline typename std::enable_if<MemSpace == AMGX_host, void>::type 
      sequence(ForwardIterator first, ForwardIterator last, T init)
      {
          amgx::thrust::sequence(first, last, init);
      }

  template<int MemSpace, typename ForwardIterator, typename T>
      inline typename std::enable_if<MemSpace == AMGX_device, void>::type 
      sequence(ForwardIterator first, ForwardIterator last, T init)
      {
          amgx::thrust::sequence(amgx_thrust_get_allocator<ForwardIterator>(std::integral_constant<bool, MemSpace == AMGX_device>()), first, last, init);
      }
}
