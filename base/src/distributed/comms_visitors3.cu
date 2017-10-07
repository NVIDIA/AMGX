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



//required by do_exchange_halo()
//{
//
//step 1:
//
template<typename TConfig, typename Tb>
void ExcHalo1Functor<TConfig, Tb>::operator()(CommsMPIHostBufferStream<TConfig> &comm)
{
    Tb &b = get_b();

    if (b.buffer_size != 0)
    {
        cudaMemcpy(&(b.explicit_host_buffer[0]), b.buffer->raw(), b.buffer_size * sizeof(typename Tb::value_type), cudaMemcpyDefault);
    }

#ifdef AMGX_WITH_MPI
    const Matrix<TConfig> &m = get_m();
    int neighbors = comm.get_neighbors();
    int bsize = b.get_block_size();
    int num_cols = b.get_num_cols();
    int offset = 0;
    MPI_Comm mpi_comm = comm.get_mpi_comm();

    for (int i = 0; i < neighbors; i++)
    {
        int size = m.manager->getB2Lrings()[i][m_num_rings] * bsize * num_cols;

        if (size != 0)
            MPI_Isend(&(b.explicit_host_buffer[offset]),
                      size * sizeof(typename Tb::value_type),
                      MPI_BYTE,
                      m.manager->neighbors[i],
                      m_tag,
                      mpi_comm,
                      &b.requests[i]);
        else
        {
            MPI_Isend(&(b.host_buffer[0]),
                      size * sizeof(typename Tb::value_type),
                      MPI_BYTE,
                      m.manager->neighbors[i],
                      m_tag,
                      mpi_comm,
                      &b.requests[i]);
        }

        offset += size;
    }

#endif
}

template<typename TConfig, typename Tb>
void ExcHalo1Functor<TConfig, Tb>::operator()(CommsMPIDirect<TConfig> &comm)
{
#ifdef AMGX_WITH_MPI
    Tb &b = get_b();
    const Matrix<TConfig> &m = get_m();
    int bsize = b.get_block_size();
    int neighbors = m.manager->num_neighbors();
    int num_cols = b.get_num_cols();
    int num_rings = get_num_rings();
    MPI_Comm mpi_comm = comm.get_mpi_comm();
    int offset = 0;
    //#####################################################################
    //#     RULE: for send use what Hostbuffer copies from, in step 1;    #
    //#               start position: where it starts copying from;       #
    //#               size: same as for hostbuffer;                       #
    //#                                                                   #
    //#           for recv use what Hostbuffer copies into, in step 3;    #
    //#               start position: where it starts copying to;         #
    //#               size: same as for hostbuffer;                       #
    //#####################################################################
    for (int i = 0; i < neighbors; i++)
    {
        int size = m.manager->getB2Lrings()[i][num_rings] * bsize * num_cols;
        MPI_Isend(b.buffer->raw() + offset,
                  size * sizeof(typename Tb::value_type),
                  MPI_BYTE,
                  m.manager->neighbors[i],
                  m_tag,
                  mpi_comm,
                  &b.requests[i]);
        offset += size;
    }

#endif
}

//step 2:
//
template<typename TConfig, typename Tb>
void ExcHalo2Functor<TConfig, Tb>::operator()(CommsMPIHostBufferStream<TConfig> &comm)
{
#ifdef AMGX_WITH_MPI
    Tb &b = get_b();
    const Matrix<TConfig> &m = get_m();
    int num_rings = get_num_rings();
    int neighbors = comm.get_neighbors();
    int bsize = b.get_block_size();
    int num_cols = b.get_num_cols();
    int offset = 0;
    MPI_Comm mpi_comm = comm.get_mpi_comm();

    for (int i = 0; i < neighbors; i++)
    {
        // Count total size to receive from one neighbor
        int size = 0;

        for (int j = 0; j < num_rings; j++)
        {
            size += (m.manager->halo_offset(j * neighbors + i + 1) - m.manager->halo_offset(j * neighbors + i)) * bsize * num_cols;
        }

        if (size != 0)
            MPI_Irecv(&(b.explicit_host_buffer[b.buffer_size + offset]),
                      size * sizeof(typename Tb::value_type),
                      MPI_BYTE,
                      m.manager->neighbors[i],
                      m.manager->neighbors[i],
                      mpi_comm,
                      &b.requests[neighbors + i]);
        else
            MPI_Irecv(&(b.host_buffer[0]),
                      size * sizeof(typename Tb::value_type),
                      MPI_BYTE,
                      m.manager->neighbors[i],
                      m.manager->neighbors[i],
                      mpi_comm,
                      &b.requests[neighbors + i]);

        offset += size;
        int required_size = m.manager->halo_offset(0) * bsize * num_cols + offset;

        if (required_size > b.size())
        {
            error_vector_too_small(b, required_size);
        }
    }

    get_offset() = offset;
#endif
}

template<typename TConfig, typename Tb>
void ExcHalo2Functor<TConfig, Tb>::operator()(CommsMPIDirect<TConfig> &comm)
{
#ifdef AMGX_WITH_MPI
    Tb &b = get_b();
    const Matrix<TConfig> &m = get_m();
    int num_rings = get_num_rings();
    int bsize = b.get_block_size();
    int neighbors = m.manager->num_neighbors();
    int num_cols = b.get_num_cols();
    int offset = 0;
    MPI_Comm mpi_comm = comm.get_mpi_comm();
    //#####################################################################
    //#     RULE: for send use what Hostbuffer copies from, in step 1;    #
    //#               start position: where it starts copying from;       #
    //#               size: same as for hostbuffer;                       #
    //#                                                                   #
    //#           for recv use what Hostbuffer copies into, in step 3;    #
    //#               start position: where it starts copying to;         #
    //#               size: same as for hostbuffer;                       #
    //#####################################################################
    for (int i = 0; i < neighbors; i++)
    {
        // Count total size to receive from one neighbor
        int size = 0;

        for (int j = 0; j < num_rings; j++)
        {
            size += (m.manager->halo_offset(j * neighbors + i + 1) - m.manager->halo_offset(j * neighbors + i)) * bsize * num_cols;
        }

        //Why b.raw()?
        //It's because receive was happening in the same buffer that sending
        //was done from: b.buffer->raw().
        //So, for GPU Direct, receive should be done directly into b.raw().
        //Hence: Isend from b.buffer->raw(), while Irecv into b.raw()!
        MPI_Irecv(b.raw() + m.manager->halo_offset(0)*bsize + offset,
                  size * sizeof(typename Tb::value_type),
                  MPI_BYTE,
                  m.manager->neighbors[i],
                  m.manager->neighbors[i],
                  mpi_comm,
                  &b.requests[neighbors + i]);
        offset += size;
        int required_size =  m.manager->halo_offset(i) * bsize * num_cols + size;

        if (required_size > b.size())
        {
            error_vector_too_small(b, required_size);
        }

        get_offset() = offset;
    }

#endif
}

//step 3:
//
template<typename TConfig, typename Tb>
void ExcHalo3Functor<TConfig, Tb>::operator()(CommsMPIHostBufferStream<TConfig> &comm)
{
#ifdef AMGX_WITH_MPI
    Tb &b = get_b();
    const Matrix<TConfig> &m = get_m();
    int num_rings = get_num_rings();
    int neighbors = comm.get_neighbors();
    int bsize = b.get_block_size();
    int num_cols = b.get_num_cols();
    int offset = get_offset();
    typedef typename Tb::value_type value_type;
    b.in_transfer = IDLE;

    // copy on host ring by ring
    if (num_rings == 1)
    {
        if (num_cols == 1)
        {
            if (offset != 0)
            {
                cudaMemcpy(b.raw() + m.manager->halo_offset(0)*bsize, &(b.explicit_host_buffer[b.buffer_size]), offset * sizeof(typename Tb::value_type), cudaMemcpyDefault);
            }
        }
        else
        {
            int lda = b.get_lda();
            value_type *rank_start = &(b.explicit_host_buffer[b.buffer_size]);

            for (int i = 0; i < neighbors; ++i)
            {
                int halo_size = m.manager->halo_offset(i + 1) - m.manager->halo_offset(i);

                for (int s = 0; s < num_cols; ++s)
                {
                    value_type *halo_start = b.raw() + lda * s + m.manager->halo_offset(i);
                    value_type *received_halo = rank_start + s * halo_size;
                    cudaMemcpy(halo_start, received_halo, halo_size * sizeof(value_type), cudaMemcpyDefault);
                }

                rank_start += num_cols * halo_size;
            }
        }
    }
    else
    {
        if (num_cols == 1)
        {
            offset = 0;

            // Copy into b, one neighbor at a time, one ring at a time
            for (int i = 0 ; i < neighbors ; i++)
            {
                for (int j = 0; j < num_rings; j++)
                {
                    int size = m.manager->halo_offset(j * neighbors + i + 1) * bsize - m.manager->halo_offset(j * neighbors + i) * bsize;

                    if (size != 0)
                    {
                        cudaMemcpy(b.raw() + m.manager->halo_offset(j * neighbors + i)*bsize, &(b.explicit_host_buffer[b.buffer_size + offset]), size * sizeof(typename Tb::value_type), cudaMemcpyDefault);
                    }

                    offset += size;
                }
            }
        }
        else
        {
            FatalError("num_rings != 1 && num_cols != 1 not supported\n", AMGX_ERR_NOT_IMPLEMENTED);
        }
    }

#endif
}

template<typename TConfig, typename Tb>
void ExcHalo3Functor<TConfig, Tb>::operator()(CommsMPIDirect<TConfig> &comm)
{
#ifdef AMGX_WITH_MPI
    Tb &b = get_b();
    // NOOP
    b.in_transfer = IDLE;
#endif
}

//} end do_exchange_halo()

//do_exchange_halo_async():
//{
//step 2:
//
template<typename TConfig, typename Tb>
void ExcHalo2AsyncFunctor<TConfig, Tb>::operator()(CommsMPIHostBufferStream<TConfig> &comm)
{
#ifdef AMGX_WITH_MPI
    Tb &b = get_b();
    const Matrix<TConfig> &m = get_m();
    int neighbors = comm.get_neighbors();
    int bsize = b.get_block_size();
    int num_cols = b.get_num_cols();
    int offset = 0;
    int tag = get_tag();
    MPI_Comm mpi_comm = comm.get_mpi_comm();

    for (int i = 0; i < neighbors; i++)
    {
        int size = (m.manager->halo_offset(i + 1) - m.manager->halo_offset(i)) * bsize * num_cols;

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
            MPI_Irecv(&(b.host_buffer[0]),
                      size * sizeof(typename Tb::value_type),
                      MPI_BYTE,
                      m.manager->neighbors[i],
                      tag,
                      mpi_comm,
                      &b.requests[neighbors + i]);
        }

        offset += size;
        int required_size = m.manager->halo_offset(0) * bsize * num_cols + offset;

        if (required_size > b.size())
        {
            error_vector_too_small(b, required_size);
        }
    }

#endif
}

template<typename TConfig, typename Tb>
void ExcHalo2AsyncFunctor<TConfig, Tb>::operator()(CommsMPIDirect<TConfig> &comm)
{
    //#####################################################################
    //#     RULE: for send use what Hostbuffer copies from, in step 1;    #
    //#               start position: where it starts copying from;       #
    //#               size: same as for hostbuffer;                       #
    //#                                                                   #
    //#           for recv use what Hostbuffer copies into, in step 3;    #
    //#               start position: where it starts copying to;         #
    //#               size: same as for hostbuffer;                       #
    //#####################################################################
#ifdef AMGX_WITH_MPI
    Tb &b = get_b();
    const Matrix<TConfig> &m = get_m();
    int bsize = b.get_block_size();
    int neighbors = m.manager->num_neighbors();
    int tag = get_tag();
    int num_cols = b.get_num_cols();
    int offset = 0;
    MPI_Comm mpi_comm = comm.get_mpi_comm();

    for (int i = 0; i < neighbors; i++)
    {
        int size = (m.manager->halo_offset(i + 1) - m.manager->halo_offset(i)) * bsize * num_cols;
        MPI_Irecv(b.raw() + m.manager->halo_offset(0)*bsize + offset,
                  size * sizeof(typename Tb::value_type),
                  MPI_BYTE,
                  m.manager->neighbors[i],
                  tag,
                  mpi_comm,
                  &b.requests[neighbors + i]);
        offset += size;
        int required_size = m.manager->halo_offset(i) * bsize * num_cols + size;

        if (required_size > b.size())
        {
            error_vector_too_small(b, required_size);
        }
    }

#endif
}

//step 3:
//
template<typename TConfig, typename Tb>
void ExcHalo3AsyncFunctor<TConfig, Tb>::operator()(CommsMPIHostBufferStream<TConfig> &comm)
{
    Tb &b = get_b();
    const Matrix<TConfig> &m = get_m();
    int num_rings = get_num_rings();
    int neighbors = comm.get_neighbors();
    int bsize = b.get_block_size();
    int num_cols = b.get_num_cols();
    int offset = get_offset();
    typedef typename Tb::value_type value_type;
#ifdef AMGX_WITH_MPI
    b.in_transfer = IDLE;
    int size = (m.manager->halo_offset(neighbors) - m.manager->halo_offset(0)) * bsize;

    if (size != 0)
    {
        cudaMemcpy(b.raw() + m.manager->halo_offset(0)*bsize, &(b.explicit_host_buffer[b.buffer_size]), size * sizeof(typename Tb::value_type), cudaMemcpyDefault);
    }

#endif
}

template<typename TConfig, typename Tb>
void ExcHalo3AsyncFunctor<TConfig, Tb>::operator()(CommsMPIDirect<TConfig> &comm)
{
    Tb &b = get_b();
    b.in_transfer = IDLE;
}
//} end do_exchange_halo_async()

//###################################### Explicit Instantiations: #############################
// (must be in same translation unit, no effect if in another translation unit...)

#include <distributed/comms_visitors3_eti.h>

}  // namespace amgx
