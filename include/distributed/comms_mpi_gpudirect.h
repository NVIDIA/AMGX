// SPDX-FileCopyrightText: 2011 - 2024 NVIDIA CORPORATION. All Rights Reserved.
//
// SPDX-License-Identifier: BSD-3-Clause

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
