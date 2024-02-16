// SPDX-FileCopyrightText: 2011 - 2024 NVIDIA CORPORATION. All Rights Reserved.
//
// SPDX-License-Identifier: BSD-3-Clause

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
