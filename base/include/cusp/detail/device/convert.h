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


#pragma once

#include <cusp/format.h>
#include <cusp/array2d.h>
#include <cusp/coo_matrix.h>

#include <cusp/detail/device/conversion.h>
#include <cusp/detail/device/conversion_utils.h>

namespace cusp
{
namespace detail
{
namespace device
{

/////////
// COO //
/////////
template <typename Matrix1, typename Matrix2>
void convert(const Matrix1& src, Matrix2& dst,
             cusp::csr_format,
             cusp::coo_format)
{    cusp::detail::device::csr_to_coo(src, dst);    }

template <typename Matrix1, typename Matrix2>
void convert(const Matrix1& src, Matrix2& dst,
             cusp::ell_format,
             cusp::coo_format)
{    cusp::detail::device::ell_to_coo(src, dst);    }

template <typename Matrix1, typename Matrix2>
void convert(const Matrix1& src, Matrix2& dst,
             cusp::dia_format,
             cusp::coo_format)
{    cusp::detail::device::dia_to_coo(src, dst);    }

template <typename Matrix1, typename Matrix2>
void convert(const Matrix1& src, Matrix2& dst,
             cusp::hyb_format,
             cusp::coo_format)
{    cusp::detail::device::hyb_to_coo(src, dst);    }

/////////
// CSR //
/////////
template <typename Matrix1, typename Matrix2>
void convert(const Matrix1& src, Matrix2& dst,
             cusp::coo_format,
             cusp::csr_format)
{    cusp::detail::device::coo_to_csr(src, dst);    }

template <typename Matrix1, typename Matrix2>
void convert(const Matrix1& src, Matrix2& dst,
             cusp::ell_format,
             cusp::csr_format)
{    cusp::detail::device::ell_to_csr(src, dst);    }

template <typename Matrix1, typename Matrix2>
void convert(const Matrix1& src, Matrix2& dst,
             cusp::dia_format,
             cusp::csr_format)
{    cusp::detail::device::dia_to_csr(src, dst);    }


/////////
// DIA //
/////////
template <typename Matrix1, typename Matrix2>
void convert(const Matrix1& src, Matrix2& dst,
             cusp::coo_format,
             cusp::dia_format,
             const float  max_fill  = 3.0,
             const size_t alignment = 32)
{
    const size_t occupied_diagonals = cusp::detail::device::count_diagonals(src);

    const float threshold  = 1e6; // 1M entries
    const float size       = float(occupied_diagonals) * float(src.num_rows);
    const float fill_ratio = size / std::max(1.0f, float(src.num_entries));

    if (max_fill < fill_ratio && size > threshold)
        throw cusp::format_conversion_exception("dia_matrix fill-in would exceed maximum tolerance");

    cusp::detail::device::coo_to_dia(src, dst, alignment);
}

template <typename Matrix1, typename Matrix2>
void convert(const Matrix1& src, Matrix2& dst,
             cusp::csr_format,
             cusp::dia_format,
             const float  max_fill  = 3.0,
             const size_t alignment = 32)
{
    const size_t occupied_diagonals = cusp::detail::device::count_diagonals(src);

    const float threshold  = 1e6; // 1M entries
    const float size       = float(occupied_diagonals) * float(src.num_rows);
    const float fill_ratio = size / std::max(1.0f, float(src.num_entries));

    if (max_fill < fill_ratio && size > threshold)
        throw cusp::format_conversion_exception("dia_matrix fill-in would exceed maximum tolerance");

    cusp::detail::device::csr_to_dia(src, dst, alignment);
}

/////////
// ELL //
/////////
template <typename Matrix1, typename Matrix2>
void convert(const Matrix1& src, Matrix2& dst,
             cusp::coo_format,
             cusp::ell_format,
             const float  max_fill  = 3.0,
             const size_t alignment = 32)
{
    const size_t max_entries_per_row = cusp::detail::device::compute_max_entries_per_row(src);    

    const float threshold  = 1e6; // 1M entries
    const float size       = float(max_entries_per_row) * float(src.num_rows);
    const float fill_ratio = size / std::max(1.0f, float(src.num_entries));

    if (max_fill < fill_ratio && size > threshold)
        throw cusp::format_conversion_exception("ell_matrix fill-in would exceed maximum tolerance");

    cusp::detail::device::coo_to_ell(src, dst, max_entries_per_row, alignment);
}

template <typename Matrix1, typename Matrix2>
void convert(const Matrix1& src, Matrix2& dst,
             cusp::csr_format,
             cusp::ell_format,
             const float  max_fill  = 3.0,
             const size_t alignment = 32)
{
    const size_t max_entries_per_row = cusp::detail::device::compute_max_entries_per_row(src);
    
    const float threshold  = 1e6; // 1M entries
    const float size       = float(max_entries_per_row) * float(src.num_rows);
    const float fill_ratio = size / std::max(1.0f, float(src.num_entries));

    if (max_fill < fill_ratio && size > threshold)
        throw cusp::format_conversion_exception("ell_matrix fill-in would exceed maximum tolerance");

    cusp::detail::device::csr_to_ell(src, dst, max_entries_per_row, alignment);
}


/////////
// HYB //
/////////

template <typename Matrix1, typename Matrix2>
void convert(const Matrix1& src, Matrix2& dst,
             cusp::coo_format,
             cusp::hyb_format,
             const float  relative_speed      = 3.0,
             const size_t breakeven_threshold = 4096)
{
    const size_t num_entries_per_row = cusp::detail::device::compute_optimal_entries_per_row(src, relative_speed, breakeven_threshold);
    cusp::detail::device::coo_to_hyb(src, dst, num_entries_per_row);
}

template <typename Matrix1, typename Matrix2>
void convert(const Matrix1& src, Matrix2& dst,
             cusp::csr_format,
             cusp::hyb_format,
             const float  relative_speed      = 3.0,
             const size_t breakeven_threshold = 4096)
{
    const size_t num_entries_per_row = cusp::detail::device::compute_optimal_entries_per_row(src, relative_speed, breakeven_threshold);
    cusp::detail::device::csr_to_hyb(src, dst, num_entries_per_row);
}

template <typename Matrix1, typename Matrix2>
void convert(const Matrix1& src, Matrix2& dst,
             cusp::ell_format,
             cusp::hyb_format)
{
    cusp::detail::device::ell_to_hyb(src, dst);
}

/////////////
// Array1d //
/////////////

template <typename Matrix1, typename Matrix2>
void convert(const Matrix1& src, Matrix2& dst,
             cusp::array2d_format,
             cusp::array1d_format)
{
  if (src.num_rows == 0 && src.num_cols == 0)
  {
    dst.resize(0);
  }
  else if (src.num_cols == 1)
  {
    dst.resize(src.num_rows);
    
    // interpret dst as a Nx1 column matrix and copy from src
    typedef cusp::array2d_view<typename Matrix2::view, cusp::column_major> View;
    View view(src.num_rows, 1, src.num_rows, cusp::make_array1d_view(dst));
    
    cusp::copy(src, view);
  }
  else if (src.num_rows == 1)
  {
    dst.resize(src.num_cols);
    
    // interpret dst as a 1xN row matrix and copy from src
    typedef cusp::array2d_view<typename Matrix2::view, cusp::row_major> View;
    View view(1, src.num_cols, src.num_cols, cusp::make_array1d_view(dst));
    
    cusp::copy(src, view);
  }
  else
  {
    throw cusp::format_conversion_exception("array2d to array1d conversion is only defined for row or column vectors");
  }
}

/////////////
// Array2d //
/////////////
template <typename Matrix1, typename Matrix2>
void convert(const Matrix1& src, Matrix2& dst,
             cusp::array1d_format,
             cusp::array2d_format)
{
  // interpret src as a Nx1 column matrix and copy to dst
  cusp::copy(cusp::make_array2d_view
              (src.size(), 1, src.size(),
               cusp::make_array1d_view(src),
               cusp::column_major()),
             dst);
}

////////////////////
// Dense<->Sparse //
////////////////////
template <typename Matrix1, typename Matrix2>
void convert(const Matrix1& src, Matrix2& dst,
             cusp::sparse_format,
             cusp::array2d_format)
{
  // TODO do this natively on the device
  
  // transfer to host, convert on host, and transfer back to device
  typedef typename Matrix1::container SourceContainerType;
  typedef typename Matrix2::container DestinationContainerType;
  typedef typename DestinationContainerType::template rebind<cusp::host_memory>::type HostDestinationContainerType;
  typedef typename SourceContainerType::template      rebind<cusp::host_memory>::type HostSourceContainerType;

  HostSourceContainerType tmp1(src);

  HostDestinationContainerType tmp2;

  cusp::detail::host::convert(tmp1, tmp2);

  cusp::copy(tmp2, dst);
}

template <typename Matrix1, typename Matrix2>
void convert(const Matrix1& src, Matrix2& dst,
             cusp::array2d_format,
             cusp::sparse_format)
{
  // TODO do this natively on the device
  
  // transfer to host, convert on host, and transfer back to device
  typedef typename Matrix1::container SourceContainerType;
  typedef typename Matrix2::container DestinationContainerType;
  typedef typename DestinationContainerType::template rebind<cusp::host_memory>::type HostDestinationContainerType;
  typedef typename SourceContainerType::template      rebind<cusp::host_memory>::type HostSourceContainerType;

  HostSourceContainerType tmp1(src);

  HostDestinationContainerType tmp2;

  cusp::detail::host::convert(tmp1, tmp2);

  cusp::copy(tmp2, dst);
}

template <typename Matrix1, typename Matrix2>
void convert(const Matrix1& src, Matrix2& dst,
             cusp::sparse_format,
             cusp::dense_format)
{
    typedef typename Matrix1::value_type ValueType;
    cusp::array2d<ValueType,cusp::device_memory> tmp;
    cusp::convert(src, tmp);
    cusp::convert(tmp, dst);
}

template <typename Matrix1, typename Matrix2>
void convert(const Matrix1& src, Matrix2& dst,
             cusp::dense_format,
             cusp::sparse_format)
{
    typedef typename Matrix1::value_type ValueType;
    cusp::array2d<ValueType,cusp::device_memory> tmp;
    cusp::convert(src, tmp);
    cusp::convert(tmp, dst);
}

/////////////////////////////
// Sparse->Sparse Fallback //
/////////////////////////////
template <typename Matrix1, typename Matrix2>
void convert(const Matrix1& src, Matrix2& dst,
             cusp::sparse_format,
             cusp::sparse_format)
{
   typedef typename Matrix1::index_type IndexType;
   typedef typename Matrix1::value_type ValueType;

   // convert src -> coo_matrix -> dst
   cusp::coo_matrix<IndexType, ValueType, cusp::device_memory> tmp;
   cusp::convert(src, tmp);
   cusp::convert(tmp, dst);
}

/////////////////
// Entry Point //
/////////////////
template <typename Matrix1, typename Matrix2>
void convert(const Matrix1& src, Matrix2& dst)
{
    cusp::detail::device::convert(src, dst,
            typename Matrix1::format(),
            typename Matrix2::format());
}
            
} // end namespace device
} // end namespace detail
} // end namespace cusp

