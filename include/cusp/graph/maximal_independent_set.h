// SPDX-FileCopyrightText: 2008 - 2025 NVIDIA CORPORATION. All Rights Reserved.
//
// SPDX-License-Identifier: BSD-3-Clause

/*! \file maximal_independent_set.h
 *  \brief Maximal independent set of a graph
 */

#pragma once

#include <cusp/detail/config.h>

namespace cusp
{
namespace graph
{
/*! \addtogroup algorithms Algorithms
 *  \ingroup algorithms
 *  \{
 */

/*! \p maximal_independent_set : computes a maximal independent set (MIS)
 * a graph.  The MIS is a set of vertices such that (1) no two vertices
 * are adjacent and (2) it is not possible to add another vertex to thes
 * set without violating the first property.  The MIS(k) is a generalization
 * of the MIS with the property that no two vertices in the set are joined
 * by a path of \p k edges or less.  The standard MIS is therefore a MIS(1).
 *
 * The MIS(k) is represented by an array of {0,1} values.  Specifically,  
 * <tt>stencil[i]</tt> is 1 if vertex \p i is a member of the MIS(k) and
 * 0 otherwise.
 *
 * \param A symmetric matrix that represents a graph
 * \param stencil array to hold the MIS(k)
 * \param k radius of independence
 *
 * \tparam Matrix matrix
 * \tparam Array array
 *
 *  \see http://en.wikipedia.org/wiki/Maximal_independent_set
 */
    
template <typename Matrix, typename Array>
size_t maximal_independent_set(const Matrix& A, Array& stencil, size_t k = 1);

/*! \}
 */


} // end namespace graph
} // end namespace cusp

#include <cusp/graph/detail/maximal_independent_set.inl>

