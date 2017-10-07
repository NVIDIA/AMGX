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