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

#pragma once

#include <async_event.h>
#include <distributed/distributed_comms.h>
#include <distributed/amgx_mpi.h>
#include <vector.h>

namespace amgx
{

template <class T_Config> class CommsMPI; // forward...

template <class T_Config> class CommsMPIHostBufferStream; // forward...

template <class T_Config> class CommsMPIDirect; // forward...

template<typename T>
struct VisitorBase
{
    virtual void VisitComms(CommsMPI<T> & ) = 0;
    virtual void VisitCommsHostbuffer(CommsMPIHostBufferStream<T> & ) = 0;
    virtual void VisitCommsMPIDirect(CommsMPIDirect<T> & ) = 0;

    //add here new visitor methods for other objects

    virtual ~VisitorBase(void)
    {
    }
};

template <class T_Config>
class CommsMPI : public DistributedComms<T_Config>
{
    public:
        typedef typename T_Config::template setMemSpace<AMGX_host>::Type TConfig_h;
        typedef typename TConfig_h::template setVecPrec<AMGX_vecInt>::Type hivec_value_type;
        typedef Vector<hivec_value_type> HIVector;
        typedef std::vector<HIVector> HIVector_Array;

        CommsMPI(AMG_Config &cfg, const std::string &cfg_scope) : DistributedComms<T_Config>(cfg, cfg_scope)
        {
        };

        CommsMPI() : DistributedComms<T_Config>() {};

        virtual DistributedComms<T_Config> *CloneSubComm(HIVector &coarse_part_to_fine_part, bool is_root_partition) = 0;

        virtual ~CommsMPI()
        {
        }

        virtual void Accept(VisitorBase<T_Config> &v)
        {
            //purposely empty... not pure,
        }
};

class CommsMPIFactory : public DistributedCommsFactory
{
    public:
        CommsMPIFactory() : DistributedCommsFactory() {}
};

} // namespace amgx
