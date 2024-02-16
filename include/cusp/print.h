// SPDX-FileCopyrightText: 2008 - 2024 NVIDIA CORPORATION. All Rights Reserved.
//
// SPDX-License-Identifier: BSD-3-Clause

/*! \file print.h
 *  \brief Print textual representation of an object
 */

#pragma once

#include <cusp/detail/config.h>

namespace cusp
{

/*! \addtogroup utilities Utilities
 *  \ingroup utilities
 *  \{
 */

/*! \p print : print a textual representation of an object
 *
 * \param p matrix, array, or other printable object
 *
 * \tparam Printable printable type
 *
 *  The following code snippet demonstrates how to use \p print.
 *
 *  \code
 *  #include <cusp/array2d.h>
 *  #include <cusp/print.h>
 *  
 *  int main(void)
 *  {
 *      // initialize a 2x3 matrix
 *      cusp::array2d<float, cusp::host_memory> A(2,3);
 *      A(0,0) = 10;  A(0,1) = 20;  A(0,2) = 30;
 *      A(1,0) = 40;  A(1,1) = 50;  A(1,2) = 60;
 *  
 *      // print A
 *      cusp::print(A);
 *  
 *      return 0;
 *  }
 *  \endcode
 */
template <typename Printable>
void print(const Printable& p);

/*! \p print : print a textual representation of an object on a given stream
 *
 * \param p matrix, array, or other printable object
 * \param s stream on which to write the output
 *
 * \tparam Printable printable type
 * \tparam Stream output stream type
 *
 *  The following code snippet demonstrates how to use \p print.
 *
 *  \code
 *  #include <cusp/array2d.h>
 *  #include <cusp/print.h>
 *
 *  #include <sstream>
 *  
 *  int main(void)
 *  {
 *      // initialize a 2x3 matrix
 *      cusp::array2d<float, cusp::host_memory> A(2,3);
 *      A(0,0) = 10;  A(0,1) = 20;  A(0,2) = 30;
 *      A(1,0) = 40;  A(1,1) = 50;  A(1,2) = 60;
 *  
 *      std::ostringstream oss;
 *
 *      // print A to stream
 *      cusp::print(A, oss);
 *  
 *      return 0;
 *  }
 *  \endcode
 */
template <typename Printable, typename Stream>
void print(const Printable& p, Stream& s);

template <typename Matrix>
CUSP_DEPRECATED
void print_matrix(const Matrix& matrix);

/*! \}
 */

} // end namespace cusp

#include <cusp/detail/print.inl>

