/* Copyright (c) 2011-2017, NVIDIA CORPORATION. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *  * Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *  * Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *  * Neither the name of NVIDIA CORPORATION nor the names of its
 *    contributors may be used to endorse or promote products derived
 *    from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
 * EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 * PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
 * CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 * EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 * PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
 * OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#pragma once

#include <thrust/detail/config.h>
#include <thrust/device_ptr.h>
#include <thrust/device_reference.h>
#include <thrust/device_malloc.h>
#include <thrust/device_free.h>
#include <thrust/device_vector.h>
#include <limits>
#include <stdexcept>

#include <global_thread_handle.h>

// Memory allocator based on the thrust default memory allocator but uses AMGX bits

namespace amgx
{

template<typename T>
class thrust_amgx_allocator
{
    public:
        /*! Type of element allocated, \c T. */
        typedef T                                   value_type;

        /*! Pointer to allocation, \c device_ptr<T>. */
        typedef amgx::thrust::device_ptr<T>               pointer;

        /*! \c const pointer to allocation, \c device_ptr<const T>. */
        typedef amgx::thrust::device_ptr<const T>         const_pointer;

        /*! Reference to allocated element, \c device_reference<T>. */
        typedef amgx::thrust::device_reference<T>         reference;

        /*! \c const reference to allocated element, \c device_reference<const T>. */
        typedef amgx::thrust::device_reference<const T>   const_reference;

        /*! Type of allocation size, \c std::size_t. */
        typedef std::size_t                         size_type;

        /*! Type of allocation difference, \c pointer::difference_type. */
        typedef typename pointer::difference_type difference_type;

        /*! The \p rebind metafunction provides the type of a \p thrust_amgx_allocator
         *  instantiated with another type.
         *
         *  \tparam U The other type to use for instantiation.
         */
        template<typename U>
        struct rebind
        {
            /*! The typedef \p other gives the type of the rebound \p thrust_amgx_allocator.
             */
            typedef thrust_amgx_allocator<U> other;
        }; // end rebind

        /*! No-argument constructor has no effect. */
        __host__ __device__
        inline thrust_amgx_allocator() {}

        /*! No-argument destructor has no effect. */
        __host__ __device__
        inline ~thrust_amgx_allocator() {}

        /*! Copy constructor has no effect. */
        __host__ __device__
        inline thrust_amgx_allocator(thrust_amgx_allocator const &) {}

        /*! Constructor from other \p thrust_amgx_allocator has no effect. */
        template<typename U>
        __host__ __device__
        inline thrust_amgx_allocator(thrust_amgx_allocator<U> const &) {}

        /*! Returns the address of an allocated object.
         *  \return <tt>&r</tt>.
         */
        __host__ __device__
        inline pointer address(reference r) { return &r; }

        /*! Returns the address an allocated object.
         *  \return <tt>&r</tt>.
         */
        __host__ __device__
        inline const_pointer address(const_reference r) { return &r; }

        /*! Allocates storage for \p cnt objects.
         *  \param cnt The number of objects to allocate.
         *  \return A \p pointer to uninitialized storage for \p cnt objects.
         *  \note Memory allocated by this function must be deallocated with \p deallocate.
         */
        __host__
        inline pointer allocate(size_type cnt,
                                const_pointer = const_pointer(static_cast<T *>(0)))
        {
            void *ptr;
            cudaMallocAsync(&ptr, sizeof(T)*cnt, 0);
            cudaStreamSynchronize(0);
            return pointer((T *)ptr);
        } // end allocate()

        /*! Deallocates storage for objects allocated with \p allocate.
         *  \param p A \p pointer to the storage to deallocate.
         *  \param cnt The size of the previous allocation.
         *  \note Memory deallocated by this function must previously have been
         *        allocated with \p allocate.
         */
        __host__
        inline void deallocate(pointer p, size_type cnt)
        {
            cudaFreeAsync((void *)p.get(), 0);
        } // end deallocate()

        /*! Compares against another \p thrust_amgx_allocator for equality.
         *  \return \c true
         */
        __host__ __device__
        inline bool operator==(thrust_amgx_allocator const &) { return true; }

        /*! Compares against another \p thrust_amgx_allocator for inequality.
         *  \return \c false
         */
        __host__ __device__
        inline bool operator!=(thrust_amgx_allocator const &a) {return !operator==(a); }
}; // end thrust_amgx_allocator

template<class T>
using device_vector_alloc = amgx::thrust::device_vector<T, thrust_amgx_allocator<T>>;

} // end amgx
