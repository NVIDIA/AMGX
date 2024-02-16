// SPDX-FileCopyrightText: 2008 - 2024 NVIDIA CORPORATION. All Rights Reserved.
//
// SPDX-License-Identifier: BSD-3-Clause

/*! \file cg_m.h
 *  \brief Multi-mass Conjugate Gradient (CG-M) method
 */

#pragma once

#include <cusp/detail/config.h>

namespace cusp
{
namespace krylov
{

/*! \addtogroup iterative_solvers Iterative Solvers
 *  \addtogroup krylov_methods Krylov Methods
 *  \ingroup iterative_solvers
 *  \{
 */

/*! \p cg_m : Multi-mass Conjugate Gradient method
 * 
 * Solves the symmetric, positive-definited linear system (A+\sigma) x = b
 * for some set of constant shifts \p sigma for the price of the smallest shift
 * using the default monitor
 *
 * \param A matrix of the linear system
 * \param x solutions of the system
 * \param b right-hand side of the linear system
 * \param sigma array of shifts
 * 
 */
template <class LinearOperator,
          class VectorType1,
          class VectorType2,
          class VectorType3>
void cg_m(LinearOperator& A,
          VectorType1& x,
          VectorType2& b,
          VectorType3& sigma);

/*! \p cg_m : Multi-mass Conjugate Gradient method
 * 
 * Solves the symmetric, positive-definited linear system (A+\sigma) x = b
 * for some set of constant shifts \p sigma for the price of the smallest shift
 *
 * \param A matrix of the linear system
 * \param x solutions of the system
 * \param b right-hand side of the linear system
 * \param sigma array of shifts
 * \param monitor monitors interation and determines stoppoing conditions
 *
 */
template <class LinearOperator,
          class VectorType1,
          class VectorType2,
          class VectorType3,
          class Monitor>
void cg_m(LinearOperator& A,
          VectorType1& x,
          VectorType2& b,
          VectorType3& sigma,
          Monitor& monitor);
/*! \}
 */

} // end namespace krylov
} // end namespace cusp

#include <cusp/krylov/detail/cg_m.inl>

