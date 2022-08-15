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

