/* Copyright (c) 2013-2017, NVIDIA CORPORATION. All rights reserved.
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

namespace amgx
{

template <class T_Config> class Operator;

}

#include <operators/solve_operator.h>
#include <solvers/solver.h>
#include <blas.h>

#include "amgx_types/util.h"

namespace amgx
{

template <typename TConfig>
SolveOperator<TConfig>::~SolveOperator()
{
    delete m_solver;
}

template <typename TConfig>
void SolveOperator<TConfig>::apply(const Vector<TConfig> &v, Vector<TConfig> &res, ViewType view)
{
    Operator<TConfig> &A = *this->m_A;
    Vector<TConfig> &v_ = const_cast<Vector<TConfig>&>(v);
    int offset, size;
    A.getOffsetAndSizeForView(view, &offset, &size);
    // Fill initial solution with 0s before solving.
    fill(res, types::util<typename Vector<TConfig>::value_type>::get_zero(), offset, size);
    AMGX_STATUS solve_status = m_solver->solve(v_, res, false);

    if (solve_status != AMGX_ST_CONVERGED)
    {
        FatalError("OperatorSolve: solver did not converge.", AMGX_ERR_CONFIGURATION);
    }
}

template <typename TConfig>
void SolveOperator<TConfig>::setup()
{
    assert(m_A);
    assert(m_solver);
    m_solver->setup(*m_A, false);
}

template class SolveOperator<TConfigGeneric_d>;
template class SolveOperator<TConfigGeneric_h>;

}
