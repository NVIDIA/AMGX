// SPDX-FileCopyrightText: 2011 - 2025 NVIDIA CORPORATION. All Rights Reserved.
//
// SPDX-License-Identifier: BSD-3-Clause

#include "amgx_c_common.h"

namespace amgx
{

AMGX_RC getCAPIerror_x(AMGX_ERROR err)
{
    return (AMGX_RC)((int)(err));
}

AMGX_ERROR getAMGXerror(AMGX_RC err)
{
    return (AMGX_ERROR)((int)(err));
}


void amgx_error_exit(Resources *rsc, int err)
{
#ifdef AMGX_WITH_MPI
    int isInitialized = 0;
    MPI_Initialized(&isInitialized);

    if (isInitialized)
        if (rsc != NULL)
        {
            //Resources * res = (Resources*)(((amgx::CWrapper<AMGX_resources_handle>*) rsc)->hdl);
            MPI_Abort(*(rsc->getMpiComm()), err);
        }
        else
        {
            MPI_Abort(MPI_COMM_WORLD, err);
            //MPI_Finalize();
        }
    else
    {
        exit(err);
    }

#else
    exit(err);
#endif
}

MemCArrManager &get_c_arr_mem_manager(void)
{
    static MemCArrManager man_;
    return man_;
}


} // namespace amgx