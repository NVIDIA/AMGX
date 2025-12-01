// SPDX-FileCopyrightText: 2011 - 2025 NVIDIA CORPORATION. All Rights Reserved.
//
// SPDX-License-Identifier: BSD-3-Clause

#include <distributed/comms_mpi_hostbuffer_stream.h>//
#include <distributed/comms_mpi_gpudirect.h>//
#include <cassert>//
#include <cutil.h>//
#include <tuple>//

namespace amgx
{

namespace
{
template <typename TConfig>
static void error_vector_too_small(const Vector<TConfig> &v, int required_size)
{
    std::stringstream ss;
    ss << "Vector size too small: not enough space for halo elements." << std::endl;
    ss << "Vector: {tag = " << v.tag << ", " << "size = " << v.size() << "}" << std::endl;
    ss << "Required size: " << required_size << std::endl;
    FatalError(ss.str(), AMGX_ERR_INTERNAL);
}

}//unnamed


//required by send_vec:
//{
template<typename TConfig, typename Tb>
void SynchSendVecVisitor<TConfig, Tb>::VisitCommsHostbuffer(CommsMPIHostBufferStream<TConfig> &comm)
{
    Tb &b = get_vec();

    if (b.host_send_recv_buffer == NULL)
    {
        b.init_host_send_recv_buffer();
    }
    else if (b.host_send_recv_buffer->size() != b.size())
    {
        b.host_send_recv_buffer->resize(b.size());
    }

    int offset = get_offset();
    int size = get_size();
    int tag = get_tag();
    int destination = get_rank();
#ifdef AMGX_WITH_MPI
    amgx::thrust::copy(b.begin() + offset, b.begin() + offset + size, b.host_send_recv_buffer->begin() + offset);
    cudaCheckError();
    MPI_Send(b.host_send_recv_buffer->raw() + offset, size * (b.host_send_recv_buffer->bytes() / b.host_send_recv_buffer->size()), MPI_BYTE, destination, tag, comm.mpi_comm);
#else
    FatalError("MPI Comms module requires compiling with MPI", AMGX_ERR_NOT_IMPLEMENTED);
#endif
}

template<typename TConfig, typename Tb>
void SynchSendVecVisitor<TConfig, Tb>::VisitCommsMPIDirect(CommsMPIDirect<TConfig> &comm)
{
    Tb &b = get_vec();
    int offset = get_offset();
    int tag = get_tag();
    int dest = get_rank();
    int size = get_size();
#ifdef AMGX_WITH_MPI
    MPI_Send(b.raw() + offset, size * (b.bytes() / b.size()), MPI_BYTE, dest, tag, comm.mpi_comm);
#else
    FatalError("MPI Comms module requires compiling with MPI", AMGX_ERR_NOT_IMPLEMENTED);
#endif
}
//}end send_vec

//required by send_vec_async:
//{
template<typename TConfig, typename Tb>
void AsynchSendVecVisitor<TConfig, Tb>::VisitCommsHostbuffer(CommsMPIHostBufferStream<TConfig> &comm)
{
    Tb &b = get_vec();

    if (b.host_send_recv_buffer == NULL)
    {
        b.init_host_send_recv_buffer();
    }
    else if (b.host_send_recv_buffer->size() != b.size())
    {
        b.host_send_recv_buffer->resize(b.size());
    }

    int offset = get_offset();
    int size = get_size();
    int tag = get_tag();
    int destination = get_rank();
#ifdef AMGX_WITH_MPI
    amgx::thrust::copy(b.begin() + offset, b.begin() + offset + size, b.host_send_recv_buffer->begin() + offset);
    cudaCheckError();
    MPI_Isend(b.host_send_recv_buffer->raw() + offset, size * (b.host_send_recv_buffer->bytes() / b.host_send_recv_buffer->size()), MPI_BYTE, destination, tag, comm.mpi_comm, &b.send_requests[b.send_requests.size() - 1]);
#else
    FatalError("MPI Comms module requires compiling with MPI", AMGX_ERR_NOT_IMPLEMENTED);
#endif
}

template<typename TConfig, typename Tb>
void AsynchSendVecVisitor<TConfig, Tb>::VisitCommsMPIDirect(CommsMPIDirect<TConfig> &comm)
{
    Tb &b = get_vec();
    int offset = get_offset();
    int tag = get_tag();
    int dest = get_rank();
    int size = get_size();
#ifdef AMGX_WITH_MPI
    MPI_Isend(b.raw() + offset, size * (b.bytes() / b.size()), MPI_BYTE, dest, tag, comm.mpi_comm, &b.send_requests[b.send_requests.size() - 1]);
#else
    FatalError("MPI Comms module requires compiling with MPI", AMGX_ERR_NOT_IMPLEMENTED);
#endif
}
//}end send_vec_async

//required by recv_vec:
//{
template<typename TConfig, typename Tb>
void SynchRecvVecVisitor<TConfig, Tb>::VisitCommsHostbuffer(CommsMPIHostBufferStream<TConfig> &comm)
{
    Tb &b = get_vec();

    if (b.host_send_recv_buffer == NULL)
    {
        b.init_host_send_recv_buffer();
    }
    else if (b.host_send_recv_buffer->size() != b.size())
    {
        b.host_send_recv_buffer->resize(b.size());
    }

    int offset = get_offset();
    int size = get_size();
    int tag = get_tag();
    int source = get_rank();
#ifdef AMGX_WITH_MPI
    MPI_Recv(b.host_send_recv_buffer->raw() + offset, size * (b.host_send_recv_buffer->bytes() / b.host_send_recv_buffer->size()), MPI_BYTE, source, tag, comm.mpi_comm, /*&recv_status*/MPI_STATUSES_IGNORE);
    amgx::thrust::copy(b.host_send_recv_buffer->begin() + offset, b.host_send_recv_buffer->begin() + offset + size, b.begin() + offset);
    cudaCheckError();
#else
    FatalError("MPI Comms module requires compiling with MPI", AMGX_ERR_NOT_IMPLEMENTED);
#endif
}

template<typename TConfig, typename Tb>
void SynchRecvVecVisitor<TConfig, Tb>::VisitCommsMPIDirect(CommsMPIDirect<TConfig> &comm)
{
    Tb &b = get_vec();
    int offset = get_offset();
    int tag = get_tag();
    int source = get_rank();
    int size = get_size();
#ifdef AMGX_WITH_MPI
    MPI_Status recv_status;
    MPI_Recv(b.raw() + offset, size * (b.bytes() / b.size()), MPI_BYTE, source, tag, comm.mpi_comm, &recv_status);
#else
    FatalError("MPI Comms module requires compiling with MPI", AMGX_ERR_NOT_IMPLEMENTED);
#endif
}
//}end recv_vec

//required by recv_vec_async:
//{
template<typename TConfig, typename Tb>
void AsynchRecvVecVisitor<TConfig, Tb>::VisitCommsHostbuffer(CommsMPIHostBufferStream<TConfig> &comm)
{
    Tb &b = get_vec();

    if (b.host_send_recv_buffer == NULL)
    {
        b.init_host_send_recv_buffer();
    }
    else if (b.host_send_recv_buffer->size() != b.size())
    {
        b.host_send_recv_buffer->resize(b.size());
    }

    int offset = get_offset();
    int size = get_size();
    int tag = get_tag();
    int source = get_rank();
#ifdef AMGX_WITH_MPI
    MPI_Irecv(b.host_send_recv_buffer->raw() + offset, size * (b.host_send_recv_buffer->bytes() / b.host_send_recv_buffer->size()), MPI_BYTE, source, tag, comm.mpi_comm, &(b.recv_requests[b.recv_requests.size() - 1]));
#else
    FatalError("MPI Comms module requires compiling with MPI", AMGX_ERR_NOT_IMPLEMENTED);
#endif
}

template<typename TConfig, typename Tb>
void AsynchRecvVecVisitor<TConfig, Tb>::VisitCommsMPIDirect(CommsMPIDirect<TConfig> &comm)
{
    Tb &b = get_vec();
    int offset = get_offset();
    int tag = get_tag();
    int source = get_rank();
    int size = get_size();
#ifdef AMGX_WITH_MPI
    MPI_Irecv(b.raw() + offset, size * (b.bytes() / b.size()), MPI_BYTE, source, tag, comm.mpi_comm, &b.recv_requests[b.recv_requests.size() - 1]);
#else
    FatalError("MPI Comms module requires compiling with MPI", AMGX_ERR_NOT_IMPLEMENTED);
#endif
}
//}end recv_vec_async

//###################################### Explicit Instantiations: #############################
// (must be in same translation unit, no effect if in another translation unit...)

#include <distributed/comms_visitors2_eti.h>

}  // namespace amgx
