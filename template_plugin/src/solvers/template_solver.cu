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


#include <solvers/template_solver.h>

namespace amgx
{
namespace template_plugin
{

template<class T_Config>
TemplateSolver<T_Config>::TemplateSolver( AMG_Config &cfg, const std::string &cfg_scope) : Solver<T_Config>( cfg, cfg_scope) {}

template<class T_Config>
void
TemplateSolver<T_Config>::solver_setup(bool reuse_matrix_structure)
{
    // Allocations and initalizations should be done in setup
}

template<class T_Config>
bool
TemplateSolver<T_Config>::solve_iteration( VVector &b, VVector &x, bool xIsZero )
{
    // One iterations of this linear solver. Modify x somehow to reduce residual norm | b - A*x |. A and b should not be modified here.
    return this->converged( b, x );
};

/****************************************
 * Explict instantiations
 ***************************************/
#define AMGX_CASE_LINE(CASE) template class TemplateSolver<TemplateMode<CASE>::Type>;
AMGX_FORALL_BUILDS(AMGX_CASE_LINE)
#undef AMGX_CASE_LINE
} // namespace template_plugin
} // namespace amgx
