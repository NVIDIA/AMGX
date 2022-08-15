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

#include <distributed/comms_mpi_hostbuffer_stream.h>//use CommsMPIHostbufferStream as base
#include <vector.h>

namespace amgx
{

template <class T_Config>
class CommsMPIDirect : public CommsMPIHostBufferStream<T_Config>
{
    public:
        typedef DistributedComms<T_Config> Base;
        typedef T_Config TConfig;
        typedef typename Base::MVector MVector;
        typedef typename Base::MVector_Array MVector_Array;

        typedef typename Base::DVector DVector;
        typedef typename Base::DVector_Array DVector_Array;
        typedef typename Base::FVector FVector;
        typedef typename Base::FVector_Array FVector_Array;
        typedef typename Base::IVector IVector;
        typedef typename Base::IVector_h IVector_h;
        typedef typename Base::IVector_Array IVector_Array;
        typedef typename Base::BVector BVector;
        typedef typename Base::BVector_Array BVector_Array;
        typedef typename Base::I64Vector I64Vector;
        typedef typename Base::I64Vector_Array I64Vector_Array;

        typedef typename Base::HMVector HMVector;

        typedef typename Base::HDVector HDVector;
        typedef typename Base::HDVector_Array HDVector_Array;
        typedef typename Base::HFVector HFVector;
        typedef typename Base::HFVector_Array HFVector_Array;
        typedef typename Base::HIVector HIVector;
        typedef typename Base::HI64Vector HI64Vector;
        typedef typename Base::DIVector DIVector;
        typedef typename Base::DFVector DFVector;
        typedef typename Base::DDVector DDVector;
        typedef typename Base::HIVector_Array HIVector_Array;
        typedef typename Base::HI64Vector_Array HI64Vector_Array;
        typedef typename Base::HBVector HBVector;
        typedef typename Base::HBVector_Array HBVector_Array;

        typedef typename Base::Matrix_Array Matrix_Array;
        typedef typename Base::DistributedManager_Array DistributedManager_Array;

        typedef TemplateConfig<AMGX_host, TConfig::vecPrec, TConfig::matPrec, TConfig::indPrec> TConfig_h;
        typedef typename TConfig_h::IndPrec IndexType_h;

        typedef typename Base::Matrix_h Matrix_h;
        typedef DistributedManager<TConfig_h> Manager_h;

#ifdef AMGX_WITH_MPI
        CommsMPIDirect(AMG_Config &cfg, const std::string &cfg_scope, const MPI_Comm *mpi_communicator) :
            CommsMPIHostBufferStream<T_Config>(cfg, cfg_scope, mpi_communicator)
        {
        }
#endif

        CommsMPIDirect(AMG_Config &cfg, const std::string &cfg_scope) :
            CommsMPIHostBufferStream<T_Config>(cfg, cfg_scope)
        {
        }

        CommsMPIDirect<TConfig> *Clone() const
        {
            return new CommsMPIDirect<TConfig>(*this);
        }

        DistributedComms<T_Config> *CloneSubComm(HIVector &coarse_part_to_fine_part, bool is_root_partition)
        {
            return NULL;
        }

    public:
        void exchange_matrix_halo(Matrix_Array &halo_rows, DistributedManager_Array &halo_btl, const Matrix<TConfig> &m);

        void exchange_matrix_halo(IVector_Array &row_offsets,
                                  I64Vector_Array &col_indices,
                                  MVector_Array &values,
                                  I64Vector_Array &halo_row_ids,
                                  IVector_h &neighbors_list,
                                  int global_id);


        void Accept(VisitorBase<T_Config> &v)
        {
            v.VisitCommsMPIDirect(*this);
        }
    protected:

//befriend Visitors:
        //
        template<typename T, typename VecType, typename VecHost> friend struct ReceiverVisitor;

        template<typename T, typename VecType, typename VecHost> friend struct SenderVisitor;

        template<typename T, typename VecType, typename VecHost> friend struct SynchronizerVisitor;

        template<typename T, typename Tb> friend struct SynchSendVecVisitor;

        template<typename T, typename Tb> friend struct AsynchSendVecVisitor;

        template<typename T, typename Tb> friend struct SynchRecvVecVisitor;

        template<typename T, typename Tb> friend struct AsynchRecvVecVisitor;

};
} // namespace amgx
