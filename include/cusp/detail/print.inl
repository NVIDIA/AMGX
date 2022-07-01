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


#include <cusp/format.h>
#include <cusp/complex.h>
#include <cusp/coo_matrix.h>

#include <iostream>
#include <iomanip>

namespace cusp
{
namespace detail
{

template <typename Printable, typename Stream>
void print(const Printable& p, Stream& s, cusp::coo_format)
{
  s << "sparse matrix <" << p.num_rows << ", " << p.num_cols << "> with " << p.num_entries << " entries\n";

  for(size_t n = 0; n < p.num_entries; n++)
  {
    s << " " << std::setw(14) << p.row_indices[n];
    s << " " << std::setw(14) << p.column_indices[n];
    s << " " << std::setw(14) << p.values[n] << "\n";
  }
}

template <typename Printable, typename Stream>
void print(const Printable& p, Stream& s, cusp::sparse_format)
{
  // general sparse fallback
  cusp::coo_matrix<typename Printable::index_type, typename Printable::value_type, cusp::host_memory> coo(p);
  cusp::print(coo, s);
}

template <typename Printable, typename Stream>
void print(const Printable& p, Stream& s, cusp::array2d_format)
{
  s << "array2d <" << p.num_rows << ", " << p.num_cols << ">\n";

  for(size_t i = 0; i < p.num_rows; i++)
  {
    for(size_t j = 0; j < p.num_cols; j++)
    {
      s << std::setw(14) << p(i,j);
    }

    s << "\n";
  }
}

template <typename Printable, typename Stream>
void print(const Printable& p, Stream& s, cusp::array1d_format)
{
  s << "array1d <" << p.size() << ">\n";

  for(size_t i = 0; i < p.size(); i++)
    s << std::setw(14) << p[i] << "\n";
}

} // end namespace detail


/////////////////
// Entry Point //
/////////////////

template <typename Printable>
void print(const Printable& p)
{
  cusp::print(p, std::cout);
}

template <typename Printable, typename Stream>
void print(const Printable& p, Stream& s)
{
  cusp::detail::print(p, s, typename Printable::format());
}

template <typename Matrix>
void print_matrix(const Matrix& A)
{
  cusp::print(A);
}

} // end namespace cusp

