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

//required by do_vec_exchange():
//{
template<typename T, typename VecType, typename VecHost>
void ReceiverVisitor<T, VecType, VecHost>::VisitCommsHostbuffer(CommsMPIHostBufferStream<T> &comm)
{
    std::vector<VecType> &b = get_rhs();
    std::vector<VecHost> &host_recv = get_host_vec();
    const std::vector<int> &recv_sizes = get_sizes();
    const Matrix<T> &m = get_matrix();
#ifdef AMGX_WITH_MPI

    for (int i = 0; i < comm.neighbors; i++)
    {
        host_recv[i].resize(recv_sizes[i]);
        MPI_Irecv(host_recv[i].raw(), host_recv[i].bytes(), MPI_BYTE, m.manager->neighbors[i], m.manager->neighbors[i], comm.mpi_comm, &comm.requests[comm.neighbors + i]);
    }

#else
    FatalError("MPI Comms module requires compiling with MPI", AMGX_ERR_NOT_IMPLEMENTED);
#endif
}

template<typename T, typename VecType, typename VecHost>
void ReceiverVisitor<T, VecType, VecHost>::VisitCommsMPIDirect(CommsMPIDirect<T> &comm)
{
    std::vector<VecType> &b = get_rhs();
    const std::vector<int> &recv_sizes = get_sizes();
    const Matrix<T> &m = get_matrix();
    m_local_copy.resize(b.size());
#ifdef AMGX_WITH_MPI

    for (int i = 0; i < comm.neighbors; i++)
    {
        m_local_copy[i].resize(recv_sizes[i]);
        MPI_Irecv(m_local_copy[i].raw(),
                  m_local_copy[i].bytes(),
                  MPI_BYTE,
                  m.manager->neighbors[i],
                  m.manager->neighbors[i],
                  comm.mpi_comm,
                  &comm.requests[comm.neighbors + i]);
    }

#else
    FatalError("MPI Comms module requires compiling with MPI", AMGX_ERR_NOT_IMPLEMENTED);
#endif
}

template<typename T, typename VecType, typename VecHost>
void SenderVisitor<T, VecType, VecHost>::VisitCommsHostbuffer(CommsMPIHostBufferStream<T> &comm)
{
    std::vector<VecType> &b = get_rhs();
    std::vector<VecHost> &host_send = get_host_vec();
    const Matrix<T> &m = get_matrix();
#ifdef AMGX_WITH_MPI

    for (int i = 0; i < comm.neighbors; i++)
    {
        MPI_Isend(host_send[i].raw(), host_send[i].bytes(), MPI_BYTE, m.manager->neighbors[i], m.manager->global_id(), comm.mpi_comm, &comm.requests[i]);
    }

#else
    FatalError("MPI Comms module requires compiling with MPI", AMGX_ERR_NOT_IMPLEMENTED);
#endif
}

template<typename T, typename VecType, typename VecHost>
void SenderVisitor<T, VecType, VecHost>::VisitCommsMPIDirect(CommsMPIDirect<T> &comm)
{
    std::vector<VecType> &b = get_rhs();
    const Matrix<T> &m = get_matrix();
#ifdef AMGX_WITH_MPI

    for (int i = 0; i < comm.neighbors; i++)
    {
        MPI_Isend(b[i].raw(), b[i].bytes(), MPI_BYTE, m.manager->neighbors[i], m.manager->global_id(), comm.mpi_comm, &comm.requests[i]);
    }

#else
    FatalError("MPI Comms module requires compiling with MPI", AMGX_ERR_NOT_IMPLEMENTED);
#endif
}

template<typename T, typename VecType, typename VecHost>
void SynchronizerVisitor<T, VecType, VecHost>::VisitCommsHostbuffer(CommsMPIHostBufferStream<T> &comm)
{
    std::vector<VecType> &b = get_rhs();
    std::vector<VecHost> &host_vec = get_host_vec();

    if ( m_dir == Direction::DeviceToHost )
        for (int i = 0; i < comm.neighbors; i++ )
        {
            host_vec[i] = b[i];
        }
    else if ( m_dir == Direction::HostToDevice )
        for (int i = 0; i < comm.neighbors; i++ )
        {
            b[i] = host_vec[i];
        }
    else
    {
        std::stringstream ss;
        ss << "ERROR: Host-Device memory sync operation is None."
           << __FILE__
           << "; "
           << __LINE__;
        throw std::runtime_error(ss.str());
    }
}

template<typename T, typename VecType, typename VecHost>
void SynchronizerVisitor<T, VecType, VecHost>::VisitCommsMPIDirect(CommsMPIDirect<T> &comm)
{
    if ( m_dir == Direction::HostToDevice )
    {
        std::vector<VecType> &b = get_rhs();
        b.swap(get_local());
    }

    //else nothing
}
//}//end do_vec_exchange

//###################################### Explicit Instantiations: #############################
// (must be in same translation unit, no effect if in another translation unit...)

#include <distributed/comms_visitors1_eti.h>

}  // namespace amgx
