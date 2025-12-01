// SPDX-FileCopyrightText: 2011 - 2025 NVIDIA CORPORATION. All Rights Reserved.
//
// SPDX-License-Identifier: BSD-3-Clause

#include <distributed/distributed_comms.h>
#include <basic_types.h>
#include <error.h>
#include <types.h>
#include <assert.h>

namespace amgx
{

/***************************************
 * Source Definitions
 ***************************************/
template<class T_Config>
DistributedComms<T_Config>::~DistributedComms()
{
};

/****************************************
 * Explict instantiations
 ***************************************/
#define AMGX_CASE_LINE(CASE) template class DistributedComms<TemplateMode<CASE>::Type >;
AMGX_FORALL_BUILDS(AMGX_CASE_LINE)
AMGX_FORCOMPLEX_BUILDS(AMGX_CASE_LINE)
#undef AMGX_CASE_LINE

} // namespace amgx
