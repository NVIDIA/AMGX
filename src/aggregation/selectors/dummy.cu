// SPDX-FileCopyrightText: 2011 - 2024 NVIDIA CORPORATION. All Rights Reserved.
//
// SPDX-License-Identifier: BSD-3-Clause

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
