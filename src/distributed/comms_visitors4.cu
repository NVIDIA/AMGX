// SPDX-FileCopyrightText: 2011 - 2024 NVIDIA CORPORATION. All Rights Reserved.
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
}

//required by do_add_from_halo()
//{
template<typename TConfig, typename Tb>
void AddFromHalo1Functor<TConfig, Tb>::operator()(CommsMPIHostBufferStream<TConfig> &comm)
{
#ifdef AMGX_WITH_MPI
    Tb &b = get_b();
    const Matrix<TConfig> &m = get_m();
    int num_rings = get_num_rings();
    int neighbors = comm.get_neighbors();
    int bsize = b.get_block_size();
    int num_cols = b.get_num_cols();
    cudaStream_t &stream = get_stream();
    int tag = get_tag();
    assert( tag == m.manager->global_id() );
    get_send_size() = (m.manager->halo_offset(neighbors * num_rings) - m.manager->halo_offset(0)) * bsize;

    if (get_send_size() != 0)
    {
        cudaMemcpyAsync(&(b.explicit_host_buffer[0]), b.buffer->raw(), get_send_size()*sizeof(typename Tb::value_type), cudaMemcpyDefault, stream);
        cudaStreamSynchronize(stream);
    }

    MPI_Comm mpi_comm = comm.get_mpi_comm();
    int offset = 0;

    for (int i = 0; i < neighbors; i++)
    {
        // Count total size to send to my neighbor
        int size = 0;

        for (int j = 0; j < num_rings; j++)
        {
            size += (m.manager->halo_offset(j * neighbors + i + 1) - m.manager->halo_offset(j * neighbors + i)) * bsize;
        }

        if (size != 0)
            MPI_Isend(&(b.explicit_host_buffer[offset]),
                      size * sizeof(typename Tb::value_type),
                      MPI_BYTE,
                      m.manager->neighbors[i],
                      tag,
                      mpi_comm,
                      &b.requests[i]);
        else
        {
            MPI_Isend(&(b.host_buffer[0]), size * sizeof(typename Tb::value_type), MPI_BYTE, m.manager->neighbors[i], tag, mpi_comm, &b.requests[i]);
        }

        offset += size;
    }

#endif
}

template<typename TConfig, typename Tb>
void AddFromHalo1Functor<TConfig, Tb>::operator()(CommsMPIDirect<TConfig> &comm)
{
#ifdef AMGX_WITH_MPI
    Tb &b = get_b();
    const Matrix<TConfig> &m = get_m();
    int bsize = b.get_block_size();
    int neighbors = m.manager->num_neighbors();
    int num_rings = get_num_rings();
    int num_cols = b.get_num_cols();
    cudaStream_t &stream = get_stream();
    int tag = get_tag();
    assert( tag == m.manager->global_id() );
    get_send_size() = (m.manager->halo_offset(neighbors * num_rings) - m.manager->halo_offset(0)) * bsize;

    if (get_send_size() != 0)
    {
        cudaStreamSynchronize(stream);
    }

    MPI_Comm mpi_comm = comm.get_mpi_comm();
    int offset = 0;

    for (int i = 0; i < neighbors; i++)
    {
        // Count total size to send to my neighbor
        int size = 0;

        for (int j = 0; j < num_rings; j++)
        {
            size += (m.manager->halo_offset(j * neighbors + i + 1) - m.manager->halo_offset(j * neighbors + i)) * bsize;
        }

        MPI_Isend(b.buffer->raw() + offset,
                  size * sizeof(typename Tb::value_type),
                  MPI_BYTE,
                  m.manager->neighbors[i],
                  tag,
                  mpi_comm,
                  &b.requests[i]);
        offset += size;
    }

#endif
}

//step 2:
//
template<typename TConfig, typename Tb>
void AddFromHalo2Functor<TConfig, Tb>::operator()(CommsMPIHostBufferStream<TConfig> &comm)
{
#ifdef AMGX_WITH_MPI
    Tb &b = get_b();
    const Matrix<TConfig> &m = get_m();
    int num_rings = get_num_rings();
    int neighbors = comm.get_neighbors();
    int bsize = b.get_block_size();
    int num_cols = b.get_num_cols();
    int send_size = get_send_size();
    int offset = 0;
    MPI_Comm mpi_comm = comm.get_mpi_comm();

    for (int i = 0; i < neighbors; i++)
    {
        // Count total size to receive from one neighbor
        int size = m.manager->getB2Lrings()[i][num_rings] * bsize;

        if (size != 0)
            MPI_Irecv(&(b.explicit_host_buffer[send_size + offset]),
                      size * sizeof(typename Tb::value_type),
                      MPI_BYTE,
                      m.manager->neighbors[i],
                      m.manager->neighbors[i],
                      mpi_comm,
                      &b.requests[neighbors + i]);
        else
        {
            MPI_Irecv(&(b.host_buffer[0]), size * sizeof(typename Tb::value_type), MPI_BYTE, m.manager->neighbors[i], m.manager->neighbors[i], mpi_comm, &b.requests[neighbors + i]);
        }

        offset += size;
    }

    get_offset() = offset;
#endif
}

template<typename TConfig, typename Tb>
void AddFromHalo2Functor<TConfig, Tb>::operator()(CommsMPIDirect<TConfig> &comm)
{
#ifdef AMGX_WITH_MPI
    Tb &b = get_b();
    const Matrix<TConfig> &m = get_m();
    int num_rings = get_num_rings();
    int neighbors = m.manager->num_neighbors();
    int bsize = b.get_block_size();
    int num_cols = b.get_num_cols();
    int send_size = get_send_size();
    int offset = 0;
    MPI_Comm mpi_comm = comm.get_mpi_comm();

    for (int i = 0; i < neighbors; i++)
    {
        int size = m.manager->getB2Lrings()[i][num_rings] * bsize;
        MPI_Irecv(b.buffer->raw() + send_size + offset,
                  size * sizeof(typename Tb::value_type),
                  MPI_BYTE,
                  m.manager->neighbors[i],
                  m.manager->neighbors[i],
                  mpi_comm,
                  &b.requests[neighbors + i]);
        offset += size;
    }

    get_offset() = offset;
#endif
}

//step 3:
//
template<typename TConfig, typename Tb>
void AddFromHalo3Functor<TConfig, Tb>::operator()(CommsMPIHostBufferStream<TConfig> &comm)
{
    Tb &b = get_b();
    b.in_transfer = IDLE;
#ifdef AMGX_WITH_MPI
    const Matrix<TConfig> &m = get_m();
    int recv_size = get_recv_size();
    int send_size = get_send_size();
    cudaStream_t &stream = get_stream();
    int neighbors = comm.get_neighbors();
    int bsize = b.get_block_size();
    int num_rings = get_num_rings();
    typedef typename Tb::value_type vtyp;

    // Copy into b.buffer, single copy
    if (recv_size != 0)
    {
        cudaMemcpyAsync(b.buffer->raw() + send_size, &(b.explicit_host_buffer[send_size]), recv_size * sizeof(typename Tb::value_type), cudaMemcpyDefault, stream);
    }

    int offset = 0;
    bool linear_buffers_changed = false;

    for (int i = 0 ; i < neighbors; i++)
    {
        if (b.linear_buffers[i] != b.buffer->raw() + send_size + offset)
        {
            linear_buffers_changed = true;
            b.linear_buffers[i] = b.buffer->raw() + send_size + offset;
        }

        offset += m.manager->getB2Lrings()[i][num_rings] * bsize;
    }

    // Copy host to device:
    //
    if (linear_buffers_changed)
    {
        b.linear_buffers_ptrs.resize(neighbors);
        cudaMemcpyAsync(amgx::thrust::raw_pointer_cast(&b.linear_buffers_ptrs[0]), &(b.linear_buffers[0]), neighbors * sizeof(vtyp *), cudaMemcpyDefault, stream);
    }

    // If we are on a stream synchronise the copies
    if(stream != 0)
    {
        cudaStreamSynchronize(stream);
    }

#endif
}

template<typename TConfig, typename Tb>
void AddFromHalo3Functor<TConfig, Tb>::operator()(CommsMPIDirect<TConfig> &comm)
{
    Tb &b = get_b();
    b.in_transfer = IDLE;
#ifdef AMGX_WITH_MPI
    int recv_size = get_recv_size();
    int send_size = get_send_size();
    cudaStream_t &stream = get_stream();
    const Matrix<TConfig> &m = get_m();
    int neighbors = comm.get_neighbors();
    int bsize = b.get_block_size();
    int num_rings = get_num_rings();
    typedef typename Tb::value_type vtyp;

    if (recv_size != 0)
    {
        cudaStreamSynchronize(stream);
    }

    int offset = 0;
    bool linear_buffers_changed = false;

    for (int i = 0 ; i < neighbors; i++)
    {
        if (b.linear_buffers[i] != b.buffer->raw() + send_size + offset)
        {
            linear_buffers_changed = true;
            b.linear_buffers[i] = b.buffer->raw() + send_size + offset;
        }

        offset += m.manager->getB2Lrings()[i][num_rings] * bsize;
    }

    // Copy device to device:
    //
    if (linear_buffers_changed)
    {
        b.linear_buffers_ptrs.resize(neighbors);
        cudaMemcpyAsync(amgx::thrust::raw_pointer_cast(&b.linear_buffers_ptrs[0]),
                        & (b.linear_buffers[0]),
                        neighbors * sizeof(vtyp *),
                        cudaMemcpyDefault, stream);
        cudaStreamSynchronize(stream);
    }

#endif
}
//} end do_add_from_halo()

//required by do_send_receive_wait()
//{
template<typename TConfig, typename Tb>
void SendRecvWait1Functor<TConfig, Tb>::operator()(CommsMPIHostBufferStream<TConfig> &comm)
{
#ifdef AMGX_WITH_MPI
    Tb &b = get_b();
    const Matrix<TConfig> &m = get_m();
    int neighbors = comm.get_neighbors();
    int bsize = b.get_block_size();
    int num_cols = b.get_num_cols();
    cudaStream_t &stream = get_stream();
    int tag = get_tag();
    const int ring1 = 1;
    get_send_size() = b.buffer_size;

    if (get_send_size() != 0)
    {
        cudaMemcpyAsync(&(b.explicit_host_buffer[0]), b.buffer->raw(), get_send_size()*sizeof(typename Tb::value_type), cudaMemcpyDefault, stream);
        cudaStreamSynchronize(stream);
    }

    MPI_Comm mpi_comm = comm.get_mpi_comm();
    int offset = 0;

    for (int i = 0; i < neighbors; i++)
    {
        int size = m.manager->getB2Lrings()[i][ring1] * bsize;

        if (size != 0)
            MPI_Isend(&(b.explicit_host_buffer[offset]),
                      size * sizeof(typename Tb::value_type),
                      MPI_BYTE,
                      m.manager->neighbors[i],
                      tag,
                      mpi_comm,
                      &b.requests[i]);
        else
        {
            b.requests[i] = MPI_REQUEST_NULL;
        }

        offset += size;
    }

#endif
}

template<typename TConfig, typename Tb>
void SendRecvWait1Functor<TConfig, Tb>::operator()(CommsMPIDirect<TConfig> &comm)
{
#ifdef AMGX_WITH_MPI
    Tb &b = get_b();
    const Matrix<TConfig> &m = get_m();
    int neighbors = m.manager->num_neighbors();
    int bsize = b.get_block_size();
    cudaStream_t &stream = get_stream();
    int tag = get_tag();
    const int ring1 = 1;
    get_send_size() = b.buffer_size;

    if (get_send_size() != 0)
    {
        cudaStreamSynchronize(stream);
    }

    MPI_Comm mpi_comm = comm.get_mpi_comm();
    int offset = 0;

    for (int i = 0; i < neighbors; i++)
    {
        int size = m.manager->getB2Lrings()[i][ring1] * bsize;

        if (size != 0)
            MPI_Isend(b.buffer->raw() + offset,
                      size * sizeof(typename Tb::value_type),
                      MPI_BYTE,
                      m.manager->neighbors[i],
                      tag,
                      mpi_comm,
                      &b.requests[i]);
        else
        {
            b.requests[i] = MPI_REQUEST_NULL;
        }

        offset += size;
    }

#endif
}

template<typename TConfig, typename Tb>
void SendRecvWait2Functor<TConfig, Tb>::operator()(CommsMPIHostBufferStream<TConfig> &comm)
{
#ifdef AMGX_WITH_MPI
    Tb &b = get_b();
    const Matrix<TConfig> &m = get_m();
    int neighbors = comm.get_neighbors();
    int bsize = b.get_block_size();
    int tag = get_tag();
    MPI_Comm mpi_comm = comm.get_mpi_comm();
    int offset = 0;

    for (int i = 0; i < neighbors; i++)
    {
        int size = (m.manager->halo_offset(i + 1) - m.manager->halo_offset(i)) * bsize;

        if (size != 0)
            MPI_Irecv(&(b.explicit_host_buffer[b.buffer_size + offset]),
                      size * sizeof(typename Tb::value_type),
                      MPI_BYTE,
                      m.manager->neighbors[i],
                      tag,
                      mpi_comm,
                      &b.requests[neighbors + i]);
        else
        {
            b.requests[neighbors + i] = MPI_REQUEST_NULL;
        }

        offset += size;
        int required_size = m.manager->halo_offset(i) * bsize + size;

        if (required_size > b.size())
        {
            error_vector_too_small(b, required_size);
        }
    }

#endif
}

template<typename TConfig, typename Tb>
void SendRecvWait2Functor<TConfig, Tb>::operator()(CommsMPIDirect<TConfig> &comm)
{
#ifdef AMGX_WITH_MPI
    Tb &b = get_b();
    const Matrix<TConfig> &m = get_m();
    int neighbors = m.manager->num_neighbors();
    int bsize = b.get_block_size();
    int tag = get_tag();
    int send_size = m.manager->halo_offset(0) * bsize;
    MPI_Comm mpi_comm = comm.get_mpi_comm();
    int offset = 0;

    for (int i = 0; i < neighbors; i++)
    {
        int size = (m.manager->halo_offset(i + 1) - m.manager->halo_offset(i)) * bsize;

        if (size != 0)
            MPI_Irecv(b.raw() + send_size + offset,
                      size * sizeof(typename Tb::value_type),
                      MPI_BYTE,
                      m.manager->neighbors[i],
                      tag,
                      mpi_comm,
                      &b.requests[neighbors + i]);
        else
        {
            b.requests[neighbors + i] = MPI_REQUEST_NULL;
        }

        offset += size;
        int required_size = m.manager->halo_offset(i) * bsize + size;

        if (required_size > b.size())
        {
            error_vector_too_small(b, required_size);
        }
    }

#endif
}

template<typename TConfig, typename Tb>
void SendRecvWait3Functor<TConfig, Tb>::operator()(CommsMPIHostBufferStream<TConfig> &comm)
{
#ifdef AMGX_WITH_MPI
    Tb &b = get_b();
    const Matrix<TConfig> &m = get_m();
    int neighbors = comm.get_neighbors();
    int bsize = b.get_block_size();
    cudaStream_t &stream = get_stream();
    int size = (m.manager->halo_offset(neighbors) - m.manager->halo_offset(0)) * bsize;

    if (size != 0)
    {
        cudaMemcpyAsync(b.raw() + m.manager->halo_offset(0)*bsize, &(b.explicit_host_buffer[b.buffer_size]), size * sizeof(typename Tb::value_type), cudaMemcpyDefault, stream);
        cudaStreamSynchronize(stream);
    }

    b.dirtybit = 0;
    b.in_transfer = IDLE;
#endif
}

template<typename TConfig, typename Tb>
void SendRecvWait3Functor<TConfig, Tb>::operator()(CommsMPIDirect<TConfig> &comm)
{
#ifdef AMGX_WITH_MPI
    Tb &b = get_b();
    const Matrix<TConfig> &m = get_m();
    int neighbors = m.manager->num_neighbors();
    int bsize = b.get_block_size();
    cudaStream_t &stream = get_stream();
    int size = (m.manager->halo_offset(neighbors) - m.manager->halo_offset(0)) * bsize;

    if (size != 0)
    {
        cudaStreamSynchronize(stream);
    }

    b.dirtybit = 0;
    b.in_transfer = IDLE;
#endif
}
//} end do_send_receive_wait()

//###################################### Explicit Instantiations: #############################
// (must be in same translation unit, no effect if in another translation unit...)

#include <distributed/comms_visitors4_eti.h>

}  // namespace amgx
