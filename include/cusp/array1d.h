// SPDX-FileCopyrightText: 2008 - 2025 NVIDIA CORPORATION. All Rights Reserved.
//
// SPDX-License-Identifier: BSD-3-Clause

/*! \file array1d.h
 *  \brief One-dimensional array
 */

#pragma once

#include <cusp/detail/config.h>

#include <cusp/memory.h>
#include <cusp/format.h>
#include <cusp/exception.h>

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/detail/vector_base.h>

#include <type_traits>

namespace cusp
{
  // forward definitions
  template <typename RandomAccessIterator> class array1d_view;

/*! \addtogroup arrays Arrays
 */

/*! \addtogroup array_containers Array Containers
 *  \ingroup arrays
 *  \{
 */

/*! \p array1d : One-dimensional array container
 * 
 * \tparam T value_type of the array
 * \tparam MemorySpace memory space of the array (cusp::host_memory or cusp::device_memory)
 *
 * \TODO example
 */
template <typename T, typename MemorySpace>
class array1d : public amgx::thrust::detail::vector_base<T, typename cusp::default_memory_allocator<T, MemorySpace>::type>
{
    private:
        typedef typename cusp::default_memory_allocator<T, MemorySpace>::type Alloc;
        typedef typename amgx::thrust::detail::vector_base<T,Alloc> Parent;

    public:
        typedef MemorySpace memory_space;
        typedef cusp::array1d_format format;

        template<typename MemorySpace2>
          struct rebind { typedef cusp::array1d<T, MemorySpace2> type; };

        /*! equivalent container type
         */
        typedef typename cusp::array1d<T,MemorySpace> container;
        
        /*! equivalent view type
         */
        typedef typename cusp::array1d_view<typename Parent::iterator> view;
        
        /*! equivalent const_view type
         */
        typedef typename cusp::array1d_view<typename Parent::const_iterator> const_view;

        typedef typename Parent::size_type  size_type;
        typedef typename Parent::value_type value_type;

        array1d(void) : Parent() {}

        explicit array1d(size_type n)
            : Parent()
        {
            if(n > 0)
            {
                Parent::m_storage.allocate(n);
                Parent::m_size = n;
            }
        }
        
        array1d(size_type n, const value_type &value) 
          : Parent(n, value) {}

        template<typename Array>
          array1d(const Array& a, typename std::enable_if<!std::is_convertible<Array,size_type>::value>::type * = 0)
          : Parent(a.begin(), a.end()) {}

        template<typename InputIterator>
          array1d(InputIterator first, InputIterator last)
          : Parent(first, last) {}

        template<typename Array>
          array1d &operator=(const Array& a)
          { Parent::assign(a.begin(), a.end()); return *this; }

        // TODO specialize resize()
}; // class array1d
/*! \}
 */
  
/*! \addtogroup array_views Array Views
 *  \ingroup arrays
 *  \{
 */

/*! \p array1d_view : One-dimensional array view
 *
 * \tparam RandomAccessIterator Underlying iterator type
 *
 * \TODO example
 */
template <typename RandomAccessIterator>
class array1d_view
{
  public:
    // what about const_iterator and const_reference?
    typedef RandomAccessIterator                                             iterator;
    typedef cusp::array1d_format                                             format;
    typedef typename amgx::thrust::iterator_traits<RandomAccessIterator>::reference  reference;
    typedef typename amgx::thrust::iterator_traits<RandomAccessIterator>::difference_type difference_type;
    typedef typename amgx::thrust::iterator_traits<RandomAccessIterator>::value_type      value_type;
#if THRUST_VERSION >= 100600
    typedef typename amgx::thrust::iterator_system<RandomAccessIterator>::type     memory_space;
#else
    typedef typename amgx::thrust::iterator_space<RandomAccessIterator>::type      memory_space;
#endif
    typedef typename amgx::thrust::iterator_traits<RandomAccessIterator>::pointer    pointer;
        
    /*! equivalent container type
     */
    typedef typename cusp::array1d<value_type,memory_space> container;
    
    /*! equivalent view type
     */
    typedef typename cusp::array1d_view<RandomAccessIterator> view;

    // is this right?
    typedef size_t size_type;
    
    array1d_view(void)
      : m_begin(), m_size(0), m_capacity(0) {}

    template <typename Array>
    explicit array1d_view(Array& a)
      : m_begin(a.begin()), m_size(a.size()), m_capacity(a.capacity()) {}
    
    template <typename Array>
    explicit array1d_view(const Array& a)
      : m_begin(a.begin()), m_size(a.size()), m_capacity(a.capacity()) {}

    // should these be templated?
    array1d_view(RandomAccessIterator first, RandomAccessIterator last)
      : m_begin(first), m_size(last - first), m_capacity(last - first) {}
   
    array1d_view& operator=(const array1d_view& a)
    {
      m_begin    = a.begin();
      m_size     = a.size();
      m_capacity = a.capacity();
      return *this;
    }

    //template <typename Array>
    //array1d_view &operator=(Array &a)
    //{
    //  m_begin    = a.begin();
    //  m_size     = a.size();
    //  m_capacity = a.capacity();
    //  return *this;
    //}
  
    reference front(void) const
    {
      return m_begin[0];
    }
    
    reference back(void) const
    {
      return m_begin[size() - 1];
    }

    reference operator[](difference_type n) const
    {
      return m_begin[n];
    }

    iterator begin(void) const
    {
      return m_begin;
    }

    iterator end(void) const
    {
      return m_begin + m_size;
    }

    size_type size(void) const
    {
      return m_size;
    }

    size_type capacity(void) const
    {
      return m_capacity;
    }

    
    pointer data(void)
    {
      return &front();
    }
    // TODO is there any value in supporting the two-argument form?
    //      i.e.  void resize(size_type new_size, value_type x = value_type())
    void resize(size_type new_size)
    {
      if (new_size <= m_capacity)
        m_size = new_size;
      else
        // XXX is not_implemented_exception the right choice?
        throw cusp::not_implemented_exception("array1d_view cannot resize() larger than capacity()");
    }

  protected:
    iterator  m_begin;
    size_type m_size;
    size_type m_capacity;
};

/* Convenience functions */
  
template <typename Iterator>
array1d_view<Iterator> make_array1d_view(Iterator first, Iterator last)
{
  return array1d_view<Iterator>(first, last);
}

template <typename Iterator>
array1d_view<Iterator> make_array1d_view(const array1d_view<Iterator>& a)
{
  return make_array1d_view(a.begin(), a.end());
}

template <typename T, typename MemorySpace>
typename array1d<T,MemorySpace>::view make_array1d_view(array1d<T,MemorySpace>& a)
{
  return make_array1d_view(a.begin(), a.end());
}

template <typename T, typename MemorySpace>
typename array1d<T,MemorySpace>::const_view make_array1d_view(const array1d<T,MemorySpace>& a)
{
  return make_array1d_view(a.begin(), a.end());
}
/*! \}
 */
  
} // end namespace cusp

#include <cusp/detail/array1d.inl>

