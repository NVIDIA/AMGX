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


namespace cusp
{
namespace detail
{

template <typename Array1, typename Array2>
bool array1d_equal(const Array1& lhs, const Array2& rhs)
{
  return lhs.size() == rhs.size() && thrust::detail::vector_equal(lhs.begin(), lhs.end(), rhs.begin());
}

} // end namespace detail
    
////////////////////////
// Equality Operators //
////////////////////////

// containers
template<typename T, typename Alloc,
         typename Array>
bool operator==(const array1d<T,Alloc>& lhs,
                const Array&            rhs)
{
    return cusp::detail::array1d_equal(lhs, rhs);
}

template<typename T, typename Alloc,
         typename Array>
bool operator==(const Array&            lhs,
                const array1d<T,Alloc>& rhs)
{
    return cusp::detail::array1d_equal(lhs, rhs);
}

template<typename T1, typename Alloc1,
         typename T2, typename Alloc2>
bool operator==(const array1d<T1,Alloc1>& lhs,
                const array1d<T2,Alloc2>& rhs)
{
    return cusp::detail::array1d_equal(lhs, rhs);
}

template<typename T, typename Alloc,
         typename Array>
bool operator!=(const array1d<T,Alloc>& lhs,
                const Array&            rhs)
{
    return !(lhs == rhs);
}

template<typename T, typename Alloc,
         typename Array>
bool operator!=(const Array&            lhs,
                const array1d<T,Alloc>& rhs)
{
    return !(lhs == rhs);
}

template<typename T1, typename Alloc1,
         typename T2, typename Alloc2>
bool operator!=(const array1d<T1,Alloc1>& lhs,
                const array1d<T2,Alloc2>& rhs)
{
    return !(lhs == rhs);
}

// views
template<typename I,
         typename Array>
bool operator==(const array1d_view<I>& lhs,
                const Array&           rhs)
{
    return cusp::detail::array1d_equal(lhs, rhs);
}

template<typename I,
         typename Array>
bool operator==(const Array&           lhs,
                const array1d_view<I>& rhs)
{
    return cusp::detail::array1d_equal(lhs, rhs);
}

template<typename I1,
         typename I2>
bool operator==(const array1d_view<I1>& lhs,
                const array1d_view<I2>& rhs)
{
    return cusp::detail::array1d_equal(lhs, rhs);
}

template<typename I,
         typename Array>
bool operator!=(const array1d_view<I>& lhs,
                const Array&           rhs)
{
    return !(lhs == rhs);
}

template<typename I,
         typename Array>
bool operator!=(const Array&           lhs,
                const array1d_view<I>& rhs)
{
    return !(lhs == rhs);
}

template<typename I1,
         typename I2>
bool operator!=(const array1d_view<I1>& lhs,
                const array1d_view<I2>& rhs)
{
    return !(lhs == rhs);
}


// mixed containers and views (to resolve ambiguity)
template<typename I,
         typename T, typename Alloc>
bool operator==(const array1d_view<I>&  lhs,
                const array1d<T,Alloc>& rhs)
{
    return cusp::detail::array1d_equal(lhs, rhs);
}

template<typename I,
         typename T, typename Alloc>
bool operator==(const array1d<T,Alloc>& lhs,
                const array1d_view<I>&  rhs)
{
    return cusp::detail::array1d_equal(lhs, rhs);
}

template<typename I,
         typename T, typename Alloc>
bool operator!=(const array1d_view<I>&  lhs,
                const array1d<T,Alloc>& rhs)
{
    return !(lhs == rhs);
}

template<typename I,
         typename T, typename Alloc>
bool operator!=(const array1d<T,Alloc>& lhs,
                const array1d_view<I>&  rhs)
{
    return !(lhs == rhs);
}

} // end namespace cusp

