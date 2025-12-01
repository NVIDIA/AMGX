// SPDX-FileCopyrightText: 2008 - 2025 NVIDIA CORPORATION. All Rights Reserved.
//
// SPDX-License-Identifier: BSD-3-Clause

/*! \file transpose.h
 *  \brief Matrix transpose
 */

#pragma once

#include <cusp/detail/config.h>

namespace cusp
{

/*! \addtogroup algorithms Algorithms
 *  \ingroup algorithms
 *  \{
 */

/*! \p transpose : transpose a matrix
 *
 * \param A input matrix
 * \param At output matrix (transpose of A)
 *
 * \tparam MatrixType1 matrix
 * \tparam MatrixType2 matrix
 *
 *  The following code snippet demonstrates how to use \p transpose.
 *
 *  \code
 *  #include <cusp/transpose.h>
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
 *      // compute the transpose
 *      cusp::array2d<float, cusp::host_memory> At;
 *      cusp::transpose(A, At);
 *  
 *      // print A^T
 *      cusp::print(At);
 *  
 *      return 0;
 *  }
 *  \endcode
 */
template <typename MatrixType1, typename MatrixType2>
void transpose(const MatrixType1& A, MatrixType2& At);

/*! \}
 */

} // end namespace cusp

#include <cusp/detail/transpose.inl>

