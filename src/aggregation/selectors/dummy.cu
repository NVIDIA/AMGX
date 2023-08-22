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

// ---------------------------------------------------------
//  Dummy selector (permits comparison between host and device codes
// --------------------------------------------------------

#include <aggregation/selectors/dummy.h>
#include <cutil.h>
#include <types.h>
#include <basic_types.h>

#ifdef _WIN32
#pragma warning (push)
#pragma warning (disable : 4244 4267 4521)
#endif

#ifdef _WIN32
#pragma warning (pop)
#endif

namespace amgx
{
namespace aggregation
{

// Constructor
template<class T_Config>
DUMMY_Selector<T_Config>::DUMMY_Selector(AMG_Config &cfg, const std::string &cfg_scope)
{
    aggregate_size = cfg.AMG_Config::template getParameter<int>("aggregate_size", cfg_scope);
}

//  setAggregates for block_dia_csr_matrix_h format
template <class T_Config>
void DUMMY_Selector<T_Config>::setAggregates(Matrix<T_Config> &A,
        IVector &aggregates, IVector &aggregates_global, int &num_aggregates)
{
    if (!A.is_matrix_singleGPU())
    {
        aggregates.resize(A.manager->halo_offset(A.manager->num_neighbors()));
    }
    else
    {
        aggregates.resize(A.get_num_rows());
    }

    for (int i = 0; i < A.get_num_rows(); i++)
    {
        aggregates[i] = i / aggregate_size;
    }

    this->renumberAndCountAggregates(aggregates, aggregates_global, A.get_num_rows(), num_aggregates);
    cudaCheckError();
}
// ---------------------------
//  Explict instantiations
// ---------------------------
#define AMGX_CASE_LINE(CASE) template class DUMMY_Selector<TemplateMode<CASE>::Type>;
AMGX_FORALL_BUILDS(AMGX_CASE_LINE)
#undef AMGX_CASE_LINE

}
}
