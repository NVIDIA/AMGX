// SPDX-FileCopyrightText: 2011 - 2025 NVIDIA CORPORATION. All Rights Reserved.
//
// SPDX-License-Identifier: BSD-3-Clause

#include <distributed/comms_mpi_hostbuffer_stream.h>
#include <basic_types.h>
#include <cutil.h>

namespace amgx
{

/***************************************
 * Source Definitions
 ***************************************/

template <typename value_type>
__global__
void copy_buffer(value_type *__restrict__ dest, const value_type *__restrict__ src, int size)
{
    for (int tidx = threadIdx.x + blockIdx.x * blockDim.x; tidx < size; tidx += blockDim.x * gridDim.x)
    {
        dest[tidx] = src[tidx];
    }
}

template <typename TConfig>
static void error_vector_too_small(const Vector<TConfig> &v, int required_size)
{
    std::stringstream ss;
    ss << "Vector size too small: not enough space for halo elements." << std::endl;
    ss << "Vector: {tag = " << v.tag << ", " << "size = " << v.size() << "}" << std::endl;
    ss << "Required size: " << required_size << std::endl;
    FatalError(ss.str(), AMGX_ERR_INTERNAL);
}

template <class T_Config>
void CommsMPIHostBufferStream<T_Config>::printString(const std::string &str)
{
#ifdef AMGX_WITH_MPI
    int total = 0;
    int part = 0;
    MPI_Comm_size( mpi_comm, &total );
    MPI_Comm_rank( mpi_comm, &part);

    for (int i = 0; i < total; i++)
    {
        MPI_Barrier(mpi_comm);

        if (i == part)
        {
            std::cout << "my_id=" << i << std::endl;
            std::cout << str << std::endl;
        }
    }

#else
    FatalError("MPI Comms module requires compiling with MPI", AMGX_ERR_NOT_IMPLEMENTED);
#endif
}

template <class T_Config>
template <class T, class T2>
void CommsMPIHostBufferStream<T_Config>::do_reduction(T &arrays, T2 &data, const Operator<TConfig> &m)
{
#ifdef AMGX_WITH_MPI
    int numRanks = 0;
    MPI_Comm_size( mpi_comm, &numRanks );
    typedef typename T2::value_type value_type;
    std::vector<value_type> linear_buffer(numRanks * data.size());
    // Allgather() copy data from all ranks into a 1D array (linear_buffer)
    MPI_Allgather(&data[0], sizeof(value_type)*data.size(), MPI_BYTE,
                  &linear_buffer[0], sizeof(value_type)*data.size(), MPI_BYTE, mpi_comm);
    // then it is copied from 1D array to the 2D array (called arrays)
    arrays.resize(numRanks);

    // first copy data from other ranks
    for (int i = 0; i < numRanks; i++)
    {
        if (i != m.getManager()->global_id())
        {
            arrays[i].assign(linear_buffer.begin() + i * data.size(), linear_buffer.begin() + (i + 1)*data.size());
        }
    }

    // then copy data from its won rank
    arrays[m.getManager()->global_id()] = data;
#else
    FatalError("MPI Comms module requires compiling with MPI", AMGX_ERR_NOT_IMPLEMENTED);
#endif
}

template<typename T>
__global__
void print_data(T **data, int size)
{
    int tid = threadIdx.x + blockDim.x * blockIdx.x;
    T *loc = data[0];
    printf("data = %d\n", loc[tid]);
}


template<typename T>
__global__
void print_data_f(T **data, int size)
{
    int tid = threadIdx.x + blockDim.x * blockIdx.x;
    T *loc = data[0];
    printf("data = %f\n", loc[tid]);
}

template <class T_Config>
template <class T>
void CommsMPIHostBufferStream<T_Config>::do_setup(T &b, const Matrix<TConfig> &m, int num_rings)
{
#ifdef AMGX_WITH_MPI
    int bsize = b.get_block_size();
    int num_cols = b.get_num_cols();

    if (bsize != 1 && num_cols != 1)
        FatalError("Error: vector cannot have block size and subspace size.",
                   AMGX_ERR_INTERNAL);

    // set num neighbors = size of b2l_rings
    // need to do this because comms might have more neighbors than our matrix knows about
    set_neighbors(m.manager->B2L_rings.size());

    if (b.in_transfer & SENDING)
    {
        b.in_transfer = IDLE;
    }

    b.requests.resize(2 * neighbors); //first part is sends second is receives
    b.statuses.resize(2 * neighbors);

    for (int i = 0; i < 2 * neighbors; i++)
    {
        b.requests[i] = MPI_REQUEST_NULL;
    }

    typedef typename T::value_type vtyp;

    if (b.linear_buffers_size < neighbors)
    {
        if (b.linear_buffers_size != 0) { amgx::memory::cudaFreeHost(b.linear_buffers); }
        cudaCheckError();

        amgx::memory::cudaMallocHost((void **) & (b.linear_buffers), neighbors * sizeof(vtyp *));
        cudaCheckError();
        b.linear_buffers_size = neighbors;
        b.linear_buffers_ptrs.resize(neighbors);
    }

    cudaCheckError();

    int send_size = 0;
    for (int i = 0; i < neighbors; i++)
    {
        send_size += m.manager->B2L_rings[i][num_rings] * bsize * num_cols;
    }

    int total_size = send_size + (m.manager->halo_offsets[num_rings * neighbors] - m.manager->halo_offsets[0]) * bsize * num_cols;

    b.buffer_size = send_size;

    if (b.buffer == NULL)
    {
        b.buffer = new T(total_size);
    }
    else if (total_size > b.buffer->size())
    {
        b.buffer->resize(total_size);

        // It is more efficient to synchronise only when linear buffers change
        cudaStreamSynchronize(0);
        cudaCheckError();
    }

    int offset = 0;
    for (int i = 0; i < neighbors; i++)
    {
        b.linear_buffers[i] = b.buffer->raw() + offset;
        offset += m.manager->B2L_rings[i][num_rings] * bsize * num_cols;
    }

    // Copy to device
    {
        cudaMemcpy(amgx::thrust::raw_pointer_cast(&b.linear_buffers_ptrs[0]), &(b.linear_buffers[0]), neighbors * sizeof(vtyp *), cudaMemcpyDefault);
        cudaCheckError();
    }

    if (total_size != 0)
    {
        if (b.explicit_host_buffer == NULL)
        {
            b.host_buffer.resize(1);
            cudaEventCreateWithFlags(&b.mpi_event, cudaEventDisableTiming);
            amgx::memory::cudaMallocHost((void **)&b.explicit_host_buffer, total_size * sizeof(vtyp));
            cudaCheckError();
        }
        else if (total_size > b.explicit_buffer_size)
        {
            amgx::memory::cudaFreeHost(b.explicit_host_buffer);
            cudaCheckError();
            amgx::memory::cudaMallocHost((void **)&b.explicit_host_buffer, total_size * sizeof(vtyp));
        }

        cudaCheckError();
        b.explicit_buffer_size = total_size;
    }

#else
    FatalError("MPI Comms module requires compiling with MPI", AMGX_ERR_NOT_IMPLEMENTED);
#endif
}

template <class T_Config>
template <class T>
void CommsMPIHostBufferStream<T_Config>::do_setup_L2H(T &b, Matrix<TConfig> &m, int num_rings)
{
#ifdef AMGX_WITH_MPI
    int num_neighbors = m.manager->neighbors.size();
    int bsize = b.get_block_size();
    b.requests.resize(2 * num_neighbors);
    b.statuses.resize(2 * num_neighbors);

    for (int i = 0; i < 2 * num_neighbors; i++)
    {
        b.requests[i] = MPI_REQUEST_NULL;
    }

    // compute new sizes
    int send_size = (m.manager->halo_offsets[num_rings * num_neighbors] - m.manager->halo_offsets[0]) * bsize;
    int recv_size = 0;

    for (int i = 0; i < num_neighbors; i++)
    {
        recv_size += m.manager->B2L_rings[i][num_rings] * bsize;
    }

    int size = send_size + recv_size;
    // resize device and host buffers
    b.buffer_size = size;

    if (b.buffer == NULL)
    {
        b.buffer = new T(size);
    }
    else if (size > b.buffer->size())
    {
        b.buffer->resize(size);
        cudaStreamSynchronize(0);
        cudaCheckError();
    }

    b.host_buffer.resize(send_size + recv_size);
    typedef typename T::value_type vtyp;

    if (size != 0)
    {
        if (b.explicit_host_buffer == NULL)
        {
            b.host_buffer.resize(1);
            cudaEventCreateWithFlags(&b.mpi_event, cudaEventDisableTiming);
            amgx::memory::cudaMallocHost((void **)&b.explicit_host_buffer, size * sizeof(vtyp));
            cudaCheckError();
        }
        else if (size > b.explicit_buffer_size)
        {
            amgx::memory::cudaFreeHost(b.explicit_host_buffer);
            cudaCheckError();
            amgx::memory::cudaMallocHost((void **)&b.explicit_host_buffer, size * sizeof(vtyp));
        }

        cudaCheckError();
        b.explicit_buffer_size = size;
    }

    // setup linear buffers
    if (b.linear_buffers_size < neighbors)
    {
        if (b.linear_buffers_size != 0) { amgx::memory::cudaFreeHost(b.linear_buffers); }
        cudaCheckError();

        amgx::memory::cudaMallocHost((void **) & (b.linear_buffers), neighbors * sizeof(vtyp *));
        cudaCheckError();
        b.linear_buffers_size = neighbors;
    }

    int total = 0;

    for (int i = 0; i < num_neighbors; i++)
    {
        b.linear_buffers[i] = b.buffer->raw() + total;

        for (int j = 0; j < num_rings; j++)
        {
            total += (m.manager->halo_offsets[j * num_neighbors + i + 1] - m.manager->halo_offsets[j * num_neighbors + i]) * bsize;
        }
    }

#else
    FatalError("MPI Comms module requires compiling with MPI", AMGX_ERR_NOT_IMPLEMENTED);
#endif
}

template <class T_Config>
template <AMGX_VecPrecision t_vecPrec, AMGX_MatPrecision t_matPrec, AMGX_IndPrecision t_indPrec>
void CommsMPIHostBufferStream<T_Config>::do_vec_exchange(std::vector<Vector<TemplateConfig<TConfig::memSpace, t_vecPrec, t_matPrec, t_indPrec> > > &b, const Matrix<TConfig> &m)
{
#ifdef AMGX_WITH_MPI
    using VecHost = Vector<TemplateConfig<AMGX_host, t_vecPrec, t_matPrec, t_indPrec>>;
    using VecType = Vector<TemplateConfig<TConfig::memSpace, t_vecPrec, t_matPrec, t_indPrec>>;
    std::vector<VecHost> host_send(b.size());
    std::vector<VecHost> host_recv(b.size());
    std::vector<int> send_sizes(b.size());
    std::vector<int> recv_sizes(b.size());

    for (int i = 0; i < neighbors; i++)
    {
        send_sizes[i] = b[i].size();
        MPI_Irecv(&recv_sizes[i], 1, MPI_INT, m.manager->neighbors[i], m.manager->neighbors[i], mpi_comm, &requests[neighbors + i]);
    }

    for (int i = 0; i < neighbors; i++ )
    {
        MPI_Isend(&send_sizes[i], 1, MPI_INT, m.manager->neighbors[i], m.manager->global_id(), mpi_comm, &requests[i]);
    }

    SynchronizerVisitor<T_Config, VecType, VecHost> syncV(Direction::DeviceToHost, b, host_send, recv_sizes, m);
    Accept(syncV);
    MPI_Waitall(2 * neighbors, &requests[0], MPI_STATUSES_IGNORE); //I wait for data to be sent too, as I will deallocate it upon exit
    ReceiverVisitor<T_Config, VecType, VecHost> receiverV(b, host_recv, recv_sizes, m);
    Accept(receiverV);
    SenderVisitor<T_Config, VecType, VecHost> senderV(b, host_send, send_sizes, m);
    Accept(senderV);
    MPI_Waitall(2 * neighbors, &requests[0], MPI_STATUSES_IGNORE); //I wait for data to be sent too, as I will deallocate it upon exit
    syncV.switch_direction();
    syncV.set_local(receiverV.get_local());//only CUDA-aware MPI needs this
    syncV.set_host_vec(host_recv);//only Normal MPI needs this
    Accept(syncV);
#else
    FatalError("MPI Comms module requires compiling with MPI", AMGX_ERR_NOT_IMPLEMENTED);
#endif
}


template <class T_Config>
template <class T>
void CommsMPIHostBufferStream<T_Config>::send_vec(T &b, int destination, int tag, int offset, int size)
{
#ifdef AMGX_WITH_MPI

    if (size == -1) { size = b.size(); }

    if (size == 0) { return; }

    {
        // Wait for any outstanding receive calls
        int num_recv_requests = b.recv_requests.size();

        if (num_recv_requests != 0)
        {
            MPI_Waitall(num_recv_requests, &b.recv_requests[0], MPI_STATUSES_IGNORE);
            b.recv_requests.clear();
            b.recv_statuses.clear();
        }
    }

    SynchSendVecVisitor<T_Config, T> sv(b, destination, tag, offset, size);
    Accept(sv);
#else
    FatalError("MPI Comms module requires compiling with MPI", AMGX_ERR_NOT_IMPLEMENTED);
#endif
}

template <class T_Config>
template <class T>
void CommsMPIHostBufferStream<T_Config>::send_vec_async(T &b, int destination, int tag, int offset, int size)
{
#ifdef AMGX_WITH_MPI

    if (size == -1) { size = b.size(); }

    if (size == 0) { return; }

    // Wait for any outstanding receive calls
    int num_recv_requests = b.recv_requests.size();

    if (num_recv_requests != 0)
    {
        // Wait for any outstanding sends, receives to finish
        MPI_Waitall(num_recv_requests, &b.recv_requests[0], MPI_STATUSES_IGNORE);
        b.recv_requests.clear();
        b.recv_statuses.clear();
    }

    MPI_Request new_req;
    MPI_Status new_status;
    b.send_requests.push_back(new_req);
    b.send_statuses.push_back(new_status);
    int num_send_requests = b.send_requests.size();
    AsynchSendVecVisitor<T_Config, T> sv(b, destination, tag, offset, size);
    Accept(sv);
#else
    FatalError("MPI Comms module requires compiling with MPI", AMGX_ERR_NOT_IMPLEMENTED);
#endif
}

template <class T_Config>
template <class T>
void CommsMPIHostBufferStream<T_Config>::send_vec_wait_all(T &b)
{
#ifdef AMGX_WITH_MPI
    int num_send_requests = b.send_requests.size();
    MPI_Waitall(num_send_requests, &b.send_requests[0], MPI_STATUSES_IGNORE);
    b.send_requests.clear();
    b.send_statuses.clear();
#else
    FatalError("MPI Comms module requires compiling with MPI", AMGX_ERR_NOT_IMPLEMENTED);
#endif
}

template <class T_Config>
template <class T>
void CommsMPIHostBufferStream<T_Config>::recv_vec(T &b, int source, int tag, int offset, int size)
{
#ifdef AMGX_WITH_MPI

    if (size == -1) { size = b.size(); }

    if (size == 0) { return; }

    // Wait for any outstanding sendcalls
    int num_send_requests = b.send_requests.size();

    if (num_send_requests != 0)
    {
        // Wait for any outstanding sends, receives to finish
        MPI_Waitall(num_send_requests, &b.send_requests[0], MPI_STATUSES_IGNORE);
        b.send_requests.clear();
        b.send_statuses.clear();
    }

    SynchRecvVecVisitor<T_Config, T> rv(b, source, tag, offset, size);
    Accept(rv);
#else
    FatalError("MPI Comms module requires compiling with MPI", AMGX_ERR_NOT_IMPLEMENTED);
#endif
}

template <class T_Config>
template <class T>
void CommsMPIHostBufferStream<T_Config>::recv_vec_async(T &b, int source, int tag, int offset, int size)
{
#ifdef AMGX_WITH_MPI

    if (size == -1) { size = b.size(); }

    if (size == 0) { return; }

    // Wait for any outstanding sendcalls
    int num_send_requests = b.send_requests.size();

    if (num_send_requests != 0)
    {
        // Wait for any outstanding sends, receives to finish
        MPI_Waitall(num_send_requests, &b.send_requests[0], MPI_STATUSES_IGNORE);
        b.send_requests.clear();
        b.send_statuses.clear();
    }

    MPI_Request new_req;
    MPI_Status new_status;
    b.recv_requests.push_back(new_req);
    b.recv_statuses.push_back(new_status);
    int num_recv_requests = b.recv_requests.size();
    AsynchRecvVecVisitor<T_Config, T> rv(b, source, tag, offset, size);
    Accept(rv);
#else
    FatalError("MPI Comms module requires compiling with MPI", AMGX_ERR_NOT_IMPLEMENTED);
#endif
}

template <class T_Config>
template <class T>
void CommsMPIHostBufferStream<T_Config>::recv_vec_wait_all(T &b)
{
#ifdef AMGX_WITH_MPI
    int num_recv_requests = b.recv_requests.size();
    MPI_Waitall(num_recv_requests, &b.recv_requests[0], MPI_STATUSES_IGNORE);
    b.recv_requests.clear();
    b.recv_statuses.clear();
    CopyHostFunctor<T_Config, T> cpyf(b);
    LightVisitor<T_Config> lvcpy(cpyf);
    Accept(lvcpy);
#else
    FatalError("MPI Comms module requires compiling with MPI", AMGX_ERR_NOT_IMPLEMENTED);
#endif
}

template <class T_Config>
template <class T>
void CommsMPIHostBufferStream<T_Config>::do_gather_L2H(T &b, const Matrix<TConfig> &m, int num_rings, cudaStream_t stream)
{
#ifdef AMGX_WITH_MPI
    // Gather L2H
    int num_neighbors = m.manager->neighbors.size();
    int bsize = b.get_block_size();

    for (int i = 0; i < num_neighbors; i++)
    {
        int total = 0;

        for (int j = 0; j < num_rings; j++)
        {
            int size = (m.manager->halo_offsets[j * num_neighbors + i + 1] - m.manager->halo_offsets[j * num_neighbors + i]) * bsize;

            if (size != 0)
            {
                // we need to use new indices after renumbering - these are stored in L2H_maps
                cudaMemcpyAsync(b.linear_buffers[i] + total, b.raw() + m.manager->halo_offsets[j * num_neighbors + i], size*sizeof(typename T::value_type), cudaMemcpyDefault, stream);
                cudaCheckError();
                total += size;
            }
        }

        cudaCheckError();
    }
    cudaStreamSynchronize(stream);
    cudaCheckError();

#else
    FatalError("MPI Comms module requires compiling with MPI", AMGX_ERR_NOT_IMPLEMENTED);
#endif
}

template <class T_Config>
template <class T>
void CommsMPIHostBufferStream<T_Config>::do_gather_L2H_v2(T &b, const Matrix<TConfig> &m, int num_rings, cudaStream_t stream)
{
#ifdef AMGX_WITH_MPI

    if (num_rings != 1)
    {
        FatalError("Halo exchange with ring > 1 not implemented in do_gather_L2H", AMGX_ERR_NOT_IMPLEMENTED);
    }

    // Gather L2H
    int num_neighbors = m.manager->neighbors.size();
    int bsize = b.get_block_size();
    int total = 0;

    for (int i = 0; i < num_neighbors; i++)
    {
        total += (m.manager->halo_offsets[i + 1] - m.manager->halo_offsets[i]) * bsize;
    }

    if (total != 0)
    {
        cudaMemcpyAsync(b.linear_buffers[0], b.raw() + m.manager->halo_offsets[0], total*sizeof(typename T::value_type), cudaMemcpyDefault, stream);
        cudaCheckError();
    }

#else
    FatalError("MPI Comms module requires compiling with MPI", AMGX_ERR_NOT_IMPLEMENTED);
#endif
}

template <class T_Config>
template <class T>
void CommsMPIHostBufferStream<T_Config>::do_add_from_halo(T &b, const Matrix<TConfig> &m, int num_rings, cudaStream_t &stream)
{
#ifdef AMGX_WITH_MPI
    typedef typename T::value_type vtyp;
    AddFromHalo1Functor<T_Config, T> ex1(b, m, num_rings, 0, stream);
    AddFromHalo2Functor<T_Config, T> ex2(b, m, num_rings, 0, stream);
    AddFromHalo3Functor<T_Config, T> ex3(b, m, num_rings, 0, stream);
    FSMVisitor<T_Config> fsmV;
    fsmV.get_functors().push_back(&ex1);
    fsmV.get_functors().push_back(&ex2);
    fsmV.get_functors().push_back(&ex3);
    Accept(fsmV);
    ex2.get_send_size() = ex1.get_send_size();//pass relevant info from one state to another
    fsmV.next();//advance FSM
    b.in_transfer = RECEIVING | SENDING;

    Accept(fsmV);
    //pass relevant info from one state to another:
    //
    ex3.get_recv_size() = ex2.get_offset();
    ex3.get_send_size() = ex2.get_send_size();
    fsmV.next();//advance FSM

    // I only wait to receive data, I can start working before all my buffers were sent
    MPI_Waitall(2 * neighbors, &b.requests[0], MPI_STATUSES_IGNORE);
    b.dirtybit = 0;
    Accept(fsmV);
#else
    FatalError("MPI Comms module requires compiling with MPI", AMGX_ERR_NOT_IMPLEMENTED);
#endif
}

template <class T_Config>
template <class T>
void CommsMPIHostBufferStream<T_Config>::do_exchange_halo(T &b, const Matrix<TConfig> &m, int num_rings)
{
#ifdef AMGX_WITH_MPI
    typedef typename T::value_type value_type;
    ExcHalo1Functor<T_Config, T> ex1(b, m, num_rings, 0);
    ExcHalo2Functor<T_Config, T> ex2(b, m, num_rings, 0);
    ExcHalo3Functor<T_Config, T> ex3(b, m, num_rings, 0);
    FSMVisitor<T_Config> fsmV;
    fsmV.get_functors().push_back(&ex1);
    fsmV.get_functors().push_back(&ex2);
    fsmV.get_functors().push_back(&ex3);
    Accept(fsmV);
    fsmV.next();//advance FSM
    b.in_transfer = RECEIVING | SENDING;
    Accept(fsmV);
    ex3.get_offset() = ex2.get_offset();//pass relevant info from one state to another
    fsmV.next();//advance FSM
    MPI_Waitall(2 * neighbors, &b.requests[0],  MPI_STATUSES_IGNORE); //I only wait to receive data, I can start working before all my buffers were sent
    b.dirtybit = 0;
    //FSM step 3:
    Accept(fsmV);
#else
    FatalError("MPI Comms module requires compiling with MPI", AMGX_ERR_NOT_IMPLEMENTED);
#endif
}

template <class T_Config>
template <class T>
void CommsMPIHostBufferStream<T_Config>::do_exchange_halo_async(T &b, const Matrix<TConfig> &m, cudaEvent_t event, int tag, cudaStream_t stream)
{
#ifdef AMGX_WITH_MPI

    if (tag == -1) { FatalError("Untagged vector\n", AMGX_ERR_NOT_IMPLEMENTED); }

    cudaEventSynchronize(event);
    cudaCheckError();
    ExcHalo1Functor<T_Config, T> ex1(b, m, 1, 0, tag, stream);
    ExcHalo2AsyncFunctor<T_Config, T> ex2(b, m, 1, 0, tag);
    ExcHalo3AsyncFunctor<T_Config, T> ex3(b, m, 1, 0, tag, stream);
    FSMVisitor<T_Config> fsmV;
    fsmV.get_functors().push_back(&ex1);
    fsmV.get_functors().push_back(&ex2);
    fsmV.get_functors().push_back(&ex3);
    Accept(fsmV);
    fsmV.next();//advance FSM
    b.in_transfer = RECEIVING | SENDING;
    Accept(fsmV);
    fsmV.next();//advance FSM

    if (min_rows_latency_hiding < 0 || m.get_num_rows() < min_rows_latency_hiding)
    {
        MPI_Waitall(2 * neighbors, &b.requests[0], /*&b.statuses[0]*/ MPI_STATUSES_IGNORE); //I only wait to receive data, I can start working before all my buffers were sent
        b.dirtybit = 0;
        Accept(fsmV);
    }

#else
    FatalError("MPI Comms module requires compiling with MPI", AMGX_ERR_NOT_IMPLEMENTED);
#endif
}

template <class T_Config>
template <class T>
void CommsMPIHostBufferStream<T_Config>::do_exchange_halo_wait(T &b, const Matrix<TConfig> &m, cudaStream_t stream)
{
#ifdef AMGX_WITH_MPI
    int bsize = b.get_block_size();

    if (!(min_rows_latency_hiding < 0 || m.get_num_rows() < min_rows_latency_hiding))
    {
        MPI_Waitall(2 * neighbors, &b.requests[0], MPI_STATUSES_IGNORE);
        b.dirtybit = 0;
        b.in_transfer = IDLE;
        HalloWaitCopyFunctor<T_Config, T> hcpyf(b, m, stream);
        LightVisitor<T_Config> lvcpy(hcpyf);//Visitor branching off: no-op for GPU Direct
        Accept(lvcpy);
    }

#else
    FatalError("MPI Comms module requires compiling with MPI", AMGX_ERR_NOT_IMPLEMENTED);
#endif
}

template <class T_Config>
template <class T>
void CommsMPIHostBufferStream<T_Config>::do_send_receive_wait(T &b, const Matrix<TConfig> &m, cudaEvent_t event, int tag, cudaStream_t &stream)
{
#ifdef AMGX_WITH_MPI

    if (tag == -1)
    {
        FatalError("Untagged vector\n", AMGX_ERR_NOT_IMPLEMENTED);
    }

    SendRecvWait1Functor<T_Config, T> ex1(b, m, 1, 0, tag, stream);
    SendRecvWait2Functor<T_Config, T> ex2(b, m, 1, 0, tag, stream);
    SendRecvWait3Functor<T_Config, T> ex3(b, m, 1, 0, tag, stream);
    FSMVisitor<T_Config> fsmV;
    fsmV.get_functors().push_back(&ex1);
    fsmV.get_functors().push_back(&ex2);
    fsmV.get_functors().push_back(&ex3);
    cudaEventSynchronize(event);
    cudaCheckError();
    Accept(fsmV);
    fsmV.next();//advance FSM
    b.in_transfer = RECEIVING | SENDING;
    Accept(fsmV);
    fsmV.next();//advance FSM

    if (neighbors > 0)
    {
        MPI_Waitall(2 * neighbors, &b.requests[0], MPI_STATUSES_IGNORE);
    }

    Accept(fsmV);
#else
    FatalError("MPI Comms module requires compiling with MPI", AMGX_ERR_NOT_IMPLEMENTED);
#endif
}

template <class T_Config>
template <class T>
bool CommsMPIHostBufferStream<T_Config>::do_exchange_halo_query(T &b, const Matrix<TConfig> &m)
{
#ifdef AMGX_WITH_MPI

    if (b.dirtybit == 0) { return true; }

    if ((b.in_transfer & RECEIVING) == 0) { return false; }

    int finished = 1;
    MPI_Status status;

    for (int i = 0; i < neighbors; i++)
    {
        int res = MPI_Test(&b.requests[neighbors + i], &finished, &status);

        if (finished == 0  || res != MPI_SUCCESS)
        {
            return false;
        }
    }

    b.dirtybit = 0;
    b.in_transfer = SENDING;
    HalloWaitCopyFunctor<T_Config, T> hcpyf(b, m, NULL);
    LightVisitor<T_Config> lvcpy(hcpyf);//Visitor branching off: no-op for GPU Direct
    Accept(lvcpy);
    return true;
#else
    FatalError("MPI Comms module requires compiling with MPI", AMGX_ERR_NOT_IMPLEMENTED);
#endif
}

template <class T_Config>
void CommsMPIHostBufferStream<T_Config>::exchange_matrix_halo(IVector_Array &row_offsets, I64Vector_Array &col_indices, MVector_Array &values, I64Vector_Array &halo_row_ids, IVector_h &neighbors_list, int global_id)
{
#ifdef AMGX_WITH_MPI
    int total = 0;
    MPI_Comm_size( mpi_comm, &total );
    int num_neighbors = neighbors_list.size();
    std::vector<HIVector> local_row_offsets(num_neighbors);
    std::vector<HI64Vector> local_col_indices(num_neighbors);
    std::vector<HMVector> local_values(num_neighbors);
    std::vector<HIVector> send_row_offsets(num_neighbors);
    std::vector<HI64Vector> send_col_indices(num_neighbors);
    std::vector<HMVector> send_values(num_neighbors);

    for (int i = 0; i < num_neighbors; i++)
    {
        send_row_offsets[i] = row_offsets[i];
        send_col_indices[i] = col_indices[i];
        send_values[i] = values[i];
    }

    std::vector<HI64Vector> local_row_ids(0);
    std::vector<HI64Vector> send_row_ids(0);

    if (halo_row_ids.size() != 0)
    {
        local_row_ids.resize(num_neighbors);
        send_row_ids.resize(num_neighbors);

        for (int i = 0; i < num_neighbors; i++)
        {
            send_row_ids[i] = halo_row_ids[i];
        }
    }

    // send metadata
    std::vector<INDEX_TYPE> metadata(num_neighbors * 2); // num_rows+1, num_nz

    for (int i = 0; i < num_neighbors; i++)
    {
        metadata[i * 2 + 0] = row_offsets[i].size();
        metadata[i * 2 + 1] = col_indices[i].size();
        MPI_Isend(&metadata[i * 2 + 0], 2, MPI_INT, neighbors_list[i], 0, mpi_comm, &requests[i]);
    }

    // receive metadata
    std::vector<INDEX_TYPE> metadata_recv(2);

    for (int i = 0; i < num_neighbors; i++)
    {
        MPI_Recv(&metadata_recv[0], 2, MPI_INT, neighbors_list[i], 0, mpi_comm, MPI_STATUSES_IGNORE);
        local_row_offsets[i].resize(metadata_recv[0]);
        local_col_indices[i].resize(metadata_recv[1]);
        local_values[i].resize(metadata_recv[1]);

        if (local_row_ids.size() != 0)
        {
            if (metadata_recv[0] - 1 > 0)
            {
                local_row_ids[i].resize(metadata_recv[0] - 1);    // row_ids is one smaller than row_offsets
            }
        }
    }

    MPI_Waitall(num_neighbors, &requests[0], MPI_STATUSES_IGNORE); // data is already received, just closing the handles
    // receive matrix data
    typedef typename T_Config::MatPrec mvalue;

    for (int i = 0; i < num_neighbors; i++)
    {
        MPI_Irecv(local_row_offsets[i].raw(), local_row_offsets[i].size(), MPI_INT, neighbors_list[i], 10 * neighbors_list[i] + 0, mpi_comm, &requests[3 * num_neighbors + i]);
        MPI_Irecv(local_col_indices[i].raw(), local_col_indices[i].size()*sizeof(int64_t), MPI_BYTE, neighbors_list[i], 10 * neighbors_list[i] + 1, mpi_comm, &requests[4 * num_neighbors + i]);
        MPI_Irecv(local_values[i].raw(), local_values[i].size()*sizeof(mvalue), MPI_BYTE, neighbors_list[i], 10 * neighbors_list[i] + 2, mpi_comm, &requests[5 * num_neighbors + i]);

        if (send_row_ids.size() != 0)
        {
            MPI_Irecv(local_row_ids[i].raw(), local_row_ids[i].size()*sizeof(int64_t), MPI_BYTE, neighbors_list[i], 10 * neighbors_list[i] + 3, mpi_comm, &requests[7 * num_neighbors + i]);
        }
    }
    //Note: GPU Direct should use row_offsets[], col_indices[], values[] directly in here:
    // send matrix: row offsets, col indices, values
    for (int i = 0; i < num_neighbors; i++)
    {
        MPI_Isend(send_row_offsets[i].raw(), send_row_offsets[i].size(), MPI_INT, neighbors_list[i], 10 * global_id + 0, mpi_comm, &requests[i]);
        MPI_Isend(send_col_indices[i].raw(), send_col_indices[i].size()*sizeof(int64_t), MPI_BYTE, neighbors_list[i], 10 * global_id + 1, mpi_comm, &requests[num_neighbors + i]);
        MPI_Isend(send_values[i].raw(), send_values[i].size()*sizeof(mvalue), MPI_BYTE, neighbors_list[i], 10 * global_id + 2, mpi_comm, &requests[2 * num_neighbors + i]);

        if (send_row_ids.size() != 0)
        {
            MPI_Isend(send_row_ids[i].raw(), send_row_ids[i].size()*sizeof(int64_t), MPI_BYTE, neighbors_list[i], 10 * global_id + 3, mpi_comm, &requests[6 * num_neighbors + i]);
        }
    }

    if (halo_row_ids.size() != 0)
    {
        MPI_Waitall(8 * num_neighbors, &requests[0], MPI_STATUSES_IGNORE);    //I have to wait for my stuff to be sent too, because I deallocate those matrices upon exditing this function
    }
    else
    {
        MPI_Waitall(6 * num_neighbors, &requests[0], MPI_STATUSES_IGNORE);    //I have to wait for my stuff to be sent too, because I deallocate those matrices upon exditing this function
    }

    //Note: GPU Direct should swap here
    for (int i = 0; i < num_neighbors; i++)
    {
        row_offsets[i] = local_row_offsets[i];
        col_indices[i] = local_col_indices[i];
        values[i] = local_values[i];

        if (halo_row_ids.size() != 0)
        {
            halo_row_ids[i] = local_row_ids[i];
        }
    }

#else
    FatalError("MPI Comms module requires compiling with MPI", AMGX_ERR_NOT_IMPLEMENTED);
#endif
}

template <class T_Config>
void CommsMPIHostBufferStream<T_Config>::exchange_matrix_halo(Matrix_Array &halo_rows, DistributedManager_Array &halo_btl, const Matrix<TConfig> &m)
{
#ifdef AMGX_WITH_MPI
    int total = 0;
    MPI_Comm_size( mpi_comm, &total );
    int bsize = m.get_block_size();
    int rings = m.manager->B2L_rings[0].size() - 1;
    int diag = m.hasProps(DIAG);
    std::vector<Matrix_h> local_copy(halo_rows.size());
    std::vector<Matrix_h> send(halo_rows.size());
    std::vector<Manager_h> local_copy_manager(halo_rows.size());
    std::vector<Manager_h> send_manager(halo_rows.size());

    //Note: GPU Direct does *NOT* do that:
    for (int i = 0; i < halo_rows.size(); i++)
    {
        send_manager[i] = halo_btl[i];
        send[i] = halo_rows[i];
    }

    {
        // there shouldn't be any uncompleted requests, because we don't want to rewrite them
        int completed;
        MPI_Testall(requests.size(), &requests[0], &completed, MPI_STATUSES_IGNORE);

        if (!completed)
        {
            MPI_Waitall(requests.size(), &requests[0], MPI_STATUSES_IGNORE);
        }
    }

    std::vector<INDEX_TYPE> metadata(neighbors * (rings + 1 + 5)); //ring offsets (rings+1), num_rows, num_nz, base_index, index_range

    for (int i = 0; i < neighbors; i++)
    {
        for (int j = 0; j <= rings; j++) { metadata[i * (rings + 1 + 5) + j] = halo_btl[i].B2L_rings[0][j]; }

        metadata[i * (rings + 1 + 5) + rings + 1] = halo_rows[i].get_num_rows();
        metadata[i * (rings + 1 + 5) + rings + 2] = halo_rows[i].get_num_nz();
        metadata[i * (rings + 1 + 5) + rings + 3] = halo_btl[i].base_index();
        metadata[i * (rings + 1 + 5) + rings + 4] = halo_btl[i].index_range();
        metadata[i * (rings + 1 + 5) + rings + 5] = halo_btl[i].L2H_maps[0].size();
        MPI_Isend(&metadata[i * (rings + 1 + 5)], rings + 6, MPI_INT, m.manager->neighbors[i], 0, mpi_comm, &requests[i]);
    }

    std::vector<INDEX_TYPE> metadata_recv(rings + 1 + 5);

    for (int i = 0; i < neighbors; i++)
    {
        MPI_Recv(&metadata_recv[0], rings + 6, MPI_INT, m.manager->neighbors[i], 0, mpi_comm, /*&status*/MPI_STATUSES_IGNORE);
        local_copy[i].addProps(CSR);

        if (diag) { local_copy[i].addProps(DIAG); }

        local_copy[i].resize(metadata_recv[rings + 1], metadata_recv[rings + 1], metadata_recv[rings + 2], m.get_block_dimy(), m.get_block_dimx(), 1);
        local_copy_manager[i].set_base_index(metadata_recv[rings + 3]);
        local_copy_manager[i].set_index_range(metadata_recv[rings + 4]);
        local_copy_manager[i].B2L_rings.resize(1);
        local_copy_manager[i].B2L_rings[0].resize(rings + 1);
        local_copy_manager[i].B2L_maps.resize(1);
        local_copy_manager[i].B2L_maps[0].resize(local_copy[i].get_num_rows());
        local_copy_manager[i].L2H_maps.resize(1);
        local_copy_manager[i].L2H_maps[0].resize(metadata_recv[rings + 5]);

        for (int j = 0; j <= rings; j++) { local_copy_manager[i].B2L_rings[0][j] = metadata_recv[j]; }
    }

    MPI_Waitall(neighbors, &requests[0], MPI_STATUSES_IGNORE); //I have to wait for my stuff to be sent too, because I deallocate those matrices upon exditing this function
    typedef typename T_Config::MatPrec mvalue;

    for (int i = 0; i < neighbors; i++)
    {
        MPI_Irecv(local_copy[i].row_offsets.raw(), local_copy[i].row_offsets.size(), MPI_INT, m.manager->neighbors[i], 10 * m.manager->neighbors[i] + 0, mpi_comm, &requests[5 * neighbors + 5 * i]);
        MPI_Irecv(local_copy[i].col_indices.raw(), local_copy[i].col_indices.size(), MPI_INT, m.manager->neighbors[i], 10 * m.manager->neighbors[i] + 1, mpi_comm, &requests[5 * neighbors + 5 * i + 1]);
        MPI_Irecv(local_copy_manager[i].B2L_maps[0].raw(), local_copy_manager[i].B2L_maps[0].size(), MPI_INT, m.manager->neighbors[i], 10 * m.manager->neighbors[i] + 2, mpi_comm, &requests[5 * neighbors + 5 * i + 2]);
        MPI_Irecv(local_copy_manager[i].L2H_maps[0].raw(), local_copy_manager[i].L2H_maps[0].size(), MPI_INT, m.manager->neighbors[i], 10 * m.manager->neighbors[i] + 3, mpi_comm, &requests[5 * neighbors + 5 * i + 3]);
        MPI_Irecv(local_copy[i].values.raw(), local_copy[i].values.size()*sizeof(mvalue), MPI_BYTE, m.manager->neighbors[i], 10 * m.manager->neighbors[i] + 4, mpi_comm, &requests[5 * neighbors + 5 * i + 4]);
    }

    //Note: GPU Direct uses halo_rows and halo_btl in here:
    for (int i = 0; i < neighbors; i++)
    {
        MPI_Isend(send[i].row_offsets.raw(), send[i].row_offsets.size(), MPI_INT, m.manager->neighbors[i], 10 * m.manager->global_id() + 0, mpi_comm, &requests[5 * i]);
        MPI_Isend(send[i].col_indices.raw(), send[i].col_indices.size(), MPI_INT, m.manager->neighbors[i], 10 * m.manager->global_id() + 1, mpi_comm, &requests[5 * i + 1]);
        MPI_Isend(send_manager[i].B2L_maps[0].raw(), send_manager[i].B2L_maps[0].size(), MPI_INT, m.manager->neighbors[i], 10 * m.manager->global_id() + 2, mpi_comm, &requests[5 * i + 2]);
        MPI_Isend(send_manager[i].L2H_maps[0].raw(), send_manager[i].L2H_maps[0].size(), MPI_INT, m.manager->neighbors[i], 10 * m.manager->global_id() + 3, mpi_comm, &requests[5 * i + 3]);
        MPI_Isend(send[i].values.raw(), send[i].values.size()*sizeof(mvalue), MPI_BYTE, m.manager->neighbors[i], 10 * m.manager->global_id() + 4, mpi_comm, &requests[5 * i + 4]);
    }

    MPI_Waitall(2 * 5 * neighbors, &requests[0], MPI_STATUSES_IGNORE); //I have to wait for my stuff to be sent too, because I deallocate those matrices upon exditing this function

    //Note: GPU Direct swaps here:
    for (int i = 0; i < halo_rows.size(); i++)
    {
        halo_btl[i] = local_copy_manager[i];
        halo_rows[i] = local_copy[i];
    }

#else
    FatalError("MPI Comms module requires compiling with MPI", AMGX_ERR_NOT_IMPLEMENTED);
#endif
}

template <class T_Config>
void CommsMPIHostBufferStream<T_Config>::send_raw_data(const void *ptr, int size, int destination, int tag )
{
#ifdef AMGX_WITH_MPI
    MPI_Send(const_cast<void *>(ptr), size, MPI_BYTE, destination, tag, mpi_comm);
#else
    FatalError("MPI Comms module requires compiling with MPI", AMGX_ERR_NOT_IMPLEMENTED);
#endif
}

template <class T_Config>
void CommsMPIHostBufferStream<T_Config>::recv_raw_data(void *ptr, int size, int source, int tag)
{
#ifdef AMGX_WITH_MPI
    MPI_Recv(ptr, size, MPI_BYTE, source, tag, mpi_comm, MPI_STATUSES_IGNORE);
#else
    FatalError("MPI Comms module requires compiling with MPI", AMGX_ERR_NOT_IMPLEMENTED);
#endif
}

template <class T_Config>
void CommsMPIHostBufferStream<T_Config>::setup(DVector &b, const Matrix<TConfig> &m, int tag, int num_rings) { do_setup(b, m, num_rings);}
template <class T_Config>
void CommsMPIHostBufferStream<T_Config>::exchange_halo(DVector &b, const Matrix<TConfig> &m, int tag, int num_rings) {           do_exchange_halo(b, m, num_rings);}
template <class T_Config>
void CommsMPIHostBufferStream<T_Config>::exchange_halo_async(DVector &b, const Matrix<TConfig> &m, cudaEvent_t event, int tag, cudaStream_t stream) {     do_exchange_halo_async(b, m, event, tag, stream);}
template <class T_Config>
void CommsMPIHostBufferStream<T_Config>::exchange_halo_wait(DVector &b, const Matrix<TConfig> &m, cudaEvent_t event, int tag, cudaStream_t stream) {      do_exchange_halo_wait(b, m, stream);}

template <class T_Config>
void CommsMPIHostBufferStream<T_Config>::setup(FVector &b, const Matrix<TConfig> &m, int tag, int num_rings) { do_setup(b, m, num_rings);}
template <class T_Config>
void CommsMPIHostBufferStream<T_Config>::exchange_halo(FVector &b, const Matrix<TConfig> &m, int tag, int num_rings) {           do_exchange_halo(b, m, num_rings);}
template <class T_Config>
void CommsMPIHostBufferStream<T_Config>::exchange_halo_async(FVector &b, const Matrix<TConfig> &m, cudaEvent_t event, int tag, cudaStream_t stream) {     do_exchange_halo_async(b, m, event, tag, stream);}
template <class T_Config>
void CommsMPIHostBufferStream<T_Config>::exchange_halo_wait(FVector &b, const Matrix<TConfig> &m, cudaEvent_t event, int tag, cudaStream_t stream) {      do_exchange_halo_wait(b, m, stream);}


template <class T_Config>
void CommsMPIHostBufferStream<T_Config>::setup(ZVector &b, const Matrix<TConfig> &m, int tag, int num_rings) { do_setup(b, m, num_rings);}
template <class T_Config>
void CommsMPIHostBufferStream<T_Config>::exchange_halo(ZVector &b, const Matrix<TConfig> &m, int tag, int num_rings) {           do_exchange_halo(b, m, num_rings);}
template <class T_Config>
void CommsMPIHostBufferStream<T_Config>::exchange_halo_async(ZVector &b, const Matrix<TConfig> &m, cudaEvent_t event, int tag, cudaStream_t stream) {     do_exchange_halo_async(b, m, event, tag, stream);}
template <class T_Config>
void CommsMPIHostBufferStream<T_Config>::exchange_halo_wait(ZVector &b, const Matrix<TConfig> &m, cudaEvent_t event, int tag, cudaStream_t stream) {      do_exchange_halo_wait(b, m, stream);}

template <class T_Config>
void CommsMPIHostBufferStream<T_Config>::setup(CVector &b, const Matrix<TConfig> &m, int tag, int num_rings) { do_setup(b, m, num_rings);}
template <class T_Config>
void CommsMPIHostBufferStream<T_Config>::exchange_halo(CVector &b, const Matrix<TConfig> &m, int tag, int num_rings) {           do_exchange_halo(b, m, num_rings);}
template <class T_Config>
void CommsMPIHostBufferStream<T_Config>::exchange_halo_async(CVector &b, const Matrix<TConfig> &m, cudaEvent_t event, int tag, cudaStream_t stream) {     do_exchange_halo_async(b, m, event, tag, stream);}
template <class T_Config>
void CommsMPIHostBufferStream<T_Config>::exchange_halo_wait(CVector &b, const Matrix<TConfig> &m, cudaEvent_t event, int tag, cudaStream_t stream) {      do_exchange_halo_wait(b, m, stream);}

template <class T_Config>
void CommsMPIHostBufferStream<T_Config>::setup(IVector &b, const Matrix<TConfig> &m, int tag, int num_rings) { do_setup(b, m, num_rings);}
template <class T_Config>
void CommsMPIHostBufferStream<T_Config>::exchange_halo(IVector &b, const Matrix<TConfig> &m, int tag, int num_rings) {           do_exchange_halo(b, m, num_rings);}
template <class T_Config>
void CommsMPIHostBufferStream<T_Config>::exchange_halo_async(IVector &b, const Matrix<TConfig> &m, cudaEvent_t event, int tag, cudaStream_t stream) {     do_exchange_halo_async(b, m, event, tag, stream);}
template <class T_Config>
void CommsMPIHostBufferStream<T_Config>::exchange_halo_wait(IVector &b, const Matrix<TConfig> &m, cudaEvent_t event, int tag, cudaStream_t stream) {      do_exchange_halo_wait(b, m, stream);}

template <class T_Config>
void CommsMPIHostBufferStream<T_Config>::setup(BVector &b, const Matrix<TConfig> &m, int tag, int num_rings) { do_setup(b, m, num_rings);}
template <class T_Config>
void CommsMPIHostBufferStream<T_Config>::exchange_halo(BVector &b, const Matrix<TConfig> &m, int tag, int num_rings) {           FatalError("MPI Comms boolean exchange not implemented", AMGX_ERR_NOT_IMPLEMENTED); /*do_exchange_halo(b, m);*/}
template <class T_Config>
void CommsMPIHostBufferStream<T_Config>::exchange_halo_async(BVector &b, const Matrix<TConfig> &m, cudaEvent_t event, int tag, cudaStream_t stream) {     FatalError("MPI Comms boolean exchange not implemented", AMGX_ERR_NOT_IMPLEMENTED);/*do_exchange_halo_async(b,m, event, tag, stream);*/}
template <class T_Config>
void CommsMPIHostBufferStream<T_Config>::exchange_halo_wait(BVector &b, const Matrix<TConfig> &m, cudaEvent_t event, int tag, cudaStream_t stream) {      FatalError("MPI Comms boolean exchange not implemented", AMGX_ERR_NOT_IMPLEMENTED);/*do_exchange_halo_wait(b, m, stream);*/}

template <class T_Config>
void CommsMPIHostBufferStream<T_Config>::setup(I64Vector &b, const Matrix<TConfig> &m, int tag, int num_rings) { do_setup(b, m, num_rings);}
template <class T_Config>
void CommsMPIHostBufferStream<T_Config>::exchange_halo(I64Vector &b, const Matrix<TConfig> &m, int tag, int num_rings) {           do_exchange_halo(b, m, num_rings);}
template <class T_Config>
void CommsMPIHostBufferStream<T_Config>::exchange_halo_async(I64Vector &b, const Matrix<TConfig> &m, cudaEvent_t event, int tag, cudaStream_t stream) {     do_exchange_halo_async(b, m, event, tag, stream);}
template <class T_Config>
void CommsMPIHostBufferStream<T_Config>::exchange_halo_wait(I64Vector &b, const Matrix<TConfig> &m, cudaEvent_t event, int tag, cudaStream_t stream) {      do_exchange_halo_wait(b, m, stream);}

template <class T_Config>
void CommsMPIHostBufferStream<T_Config>::setup_L2H(DVector &b, Matrix<TConfig> &m, int num_rings) { do_setup_L2H(b, m, num_rings);}
template <class T_Config>
void CommsMPIHostBufferStream<T_Config>::setup_L2H(IVector &b, Matrix<TConfig> &m, int num_rings) { do_setup_L2H(b, m, num_rings);}
template <class T_Config>
void CommsMPIHostBufferStream<T_Config>::setup_L2H(FVector &b, Matrix<TConfig> &m, int num_rings) { do_setup_L2H(b, m, num_rings);}
template <class T_Config>
void CommsMPIHostBufferStream<T_Config>::setup_L2H(CVector &b, Matrix<TConfig> &m, int num_rings) { do_setup_L2H(b, m, num_rings);}
template <class T_Config>
void CommsMPIHostBufferStream<T_Config>::setup_L2H(ZVector &b, Matrix<TConfig> &m, int num_rings) { do_setup_L2H(b, m, num_rings);}
template <class T_Config>
void CommsMPIHostBufferStream<T_Config>::setup_L2H(BVector &b, Matrix<TConfig> &m, int num_rings) { do_setup_L2H(b, m, num_rings);}
template <class T_Config>
void CommsMPIHostBufferStream<T_Config>::setup_L2H(I64Vector &b, Matrix<TConfig> &m, int num_rings) { do_setup_L2H(b, m, num_rings);}

template <class T_Config>
void CommsMPIHostBufferStream<T_Config>::send_receive_wait(DVector &b, const Matrix<TConfig> &m, cudaEvent_t event, int tag, cudaStream_t &stream)
{
    do_send_receive_wait(b, m, event, tag, stream);
}
template <class T_Config>
void CommsMPIHostBufferStream<T_Config>::send_receive_wait(IVector &b, const Matrix<TConfig> &m, cudaEvent_t event, int tag, cudaStream_t &stream)
{
    do_send_receive_wait(b, m, event, tag, stream);
}
template <class T_Config>
void CommsMPIHostBufferStream<T_Config>::send_receive_wait(FVector &b, const Matrix<TConfig> &m, cudaEvent_t event, int tag, cudaStream_t &stream)
{
    do_send_receive_wait(b, m, event, tag, stream);
}
template <class T_Config>
void CommsMPIHostBufferStream<T_Config>::send_receive_wait(CVector &b, const Matrix<TConfig> &m, cudaEvent_t event, int tag, cudaStream_t &stream)
{
    do_send_receive_wait(b, m, event, tag, stream);
}
template <class T_Config>
void CommsMPIHostBufferStream<T_Config>::send_receive_wait(ZVector &b, const Matrix<TConfig> &m, cudaEvent_t event, int tag, cudaStream_t &stream)
{
    do_send_receive_wait(b, m, event, tag, stream);
}
template <class T_Config>
void CommsMPIHostBufferStream<T_Config>::send_receive_wait(BVector &b, const Matrix<TConfig> &m, cudaEvent_t event, int tag, cudaStream_t &stream)
{
    FatalError("MPI Comms send-receive with streams is not implemented", AMGX_ERR_NOT_IMPLEMENTED);
}
template <class T_Config>
void CommsMPIHostBufferStream<T_Config>::send_receive_wait(I64Vector &b, const Matrix<TConfig> &m, cudaEvent_t event, int tag, cudaStream_t &stream)
{
    do_send_receive_wait(b, m, event, tag, stream);
}

template <class T_Config>
void CommsMPIHostBufferStream<T_Config>::add_from_halo(DVector &b, const Matrix<TConfig> &m, int tag, int num_rings, cudaStream_t &stream) {           do_add_from_halo(b, m, num_rings, stream);}
template <class T_Config>
void CommsMPIHostBufferStream<T_Config>::add_from_halo(FVector &b, const Matrix<TConfig> &m, int tag, int num_rings, cudaStream_t &stream) {           do_add_from_halo(b, m, num_rings, stream);}
template <class T_Config>
void CommsMPIHostBufferStream<T_Config>::add_from_halo(CVector &b, const Matrix<TConfig> &m, int tag, int num_rings, cudaStream_t &stream) {           do_add_from_halo(b, m, num_rings, stream);}
template <class T_Config>
void CommsMPIHostBufferStream<T_Config>::add_from_halo(ZVector &b, const Matrix<TConfig> &m, int tag, int num_rings, cudaStream_t &stream) {           do_add_from_halo(b, m, num_rings, stream);}
template <class T_Config>
void CommsMPIHostBufferStream<T_Config>::add_from_halo(IVector &b, const Matrix<TConfig> &m, int tag, int num_rings, cudaStream_t &stream) {           do_add_from_halo(b, m, num_rings, stream);}

template <class T_Config>
void CommsMPIHostBufferStream<T_Config>::gather_L2H(DVector &b, const Matrix<TConfig> &m, int num_rings, cudaStream_t stream)
{
    do_gather_L2H(b, m, num_rings, stream);
}

template <class T_Config>
void CommsMPIHostBufferStream<T_Config>::gather_L2H(FVector &b, const Matrix<TConfig> &m, int num_rings, cudaStream_t stream)
{
    do_gather_L2H(b, m, num_rings, stream);
}

template <class T_Config>
void CommsMPIHostBufferStream<T_Config>::gather_L2H(CVector &b, const Matrix<TConfig> &m, int num_rings, cudaStream_t stream)
{
    do_gather_L2H(b, m, num_rings, stream);
}

template <class T_Config>
void CommsMPIHostBufferStream<T_Config>::gather_L2H(ZVector &b, const Matrix<TConfig> &m, int num_rings, cudaStream_t stream)
{
    do_gather_L2H(b, m, num_rings, stream);
}


template <class T_Config>
void CommsMPIHostBufferStream<T_Config>::gather_L2H(IVector &b, const Matrix<TConfig> &m, int num_rings, cudaStream_t stream)
{
    do_gather_L2H(b, m, num_rings, stream);
}

template <class T_Config>
void CommsMPIHostBufferStream<T_Config>::gather_L2H_v2(DVector &b, const Matrix<TConfig> &m, int num_rings, cudaStream_t stream)
{
    do_gather_L2H_v2(b, m, num_rings, stream);;
}

template <class T_Config>
void CommsMPIHostBufferStream<T_Config>::gather_L2H_v2(FVector &b, const Matrix<TConfig> &m, int num_rings, cudaStream_t stream)
{
    do_gather_L2H_v2(b, m, num_rings, stream);;
}

template <class T_Config>
void CommsMPIHostBufferStream<T_Config>::gather_L2H_v2(CVector &b, const Matrix<TConfig> &m, int num_rings, cudaStream_t stream)
{
    do_gather_L2H_v2(b, m, num_rings, stream);;
}

template <class T_Config>
void CommsMPIHostBufferStream<T_Config>::gather_L2H_v2(ZVector &b, const Matrix<TConfig> &m, int num_rings, cudaStream_t stream)
{
    do_gather_L2H_v2(b, m, num_rings, stream);;
}


template <class T_Config>
void CommsMPIHostBufferStream<T_Config>::gather_L2H_v2(IVector &b, const Matrix<TConfig> &m, int num_rings, cudaStream_t stream)
{
    do_gather_L2H_v2(b, m, num_rings, stream);;
}


template <class T_Config>
bool CommsMPIHostBufferStream<T_Config>::exchange_halo_query(DVector &b, const Matrix<TConfig> &m, cudaEvent_t event) {      return do_exchange_halo_query(b, m); }
template <class T_Config>
bool CommsMPIHostBufferStream<T_Config>::exchange_halo_query(FVector &b, const Matrix<TConfig> &m, cudaEvent_t event) {      return do_exchange_halo_query(b, m); }
template <class T_Config>
bool CommsMPIHostBufferStream<T_Config>::exchange_halo_query(CVector &b, const Matrix<TConfig> &m, cudaEvent_t event) {      return do_exchange_halo_query(b, m); }
template <class T_Config>
bool CommsMPIHostBufferStream<T_Config>::exchange_halo_query(ZVector &b, const Matrix<TConfig> &m, cudaEvent_t event) {      return do_exchange_halo_query(b, m); }
template <class T_Config>
bool CommsMPIHostBufferStream<T_Config>::exchange_halo_query(IVector &b, const Matrix<TConfig> &m, cudaEvent_t event) {      return do_exchange_halo_query(b, m); }
template <class T_Config>
bool CommsMPIHostBufferStream<T_Config>::exchange_halo_query(BVector &b, const Matrix<TConfig> &m, cudaEvent_t event) {      return false;}
template <class T_Config>
bool CommsMPIHostBufferStream<T_Config>::exchange_halo_query(I64Vector &b, const Matrix<TConfig> &m, cudaEvent_t event) {      return false;}

template <class T_Config>
void CommsMPIHostBufferStream<T_Config>::global_reduce(HDVector_Array &a, HDVector &b, const Operator<TConfig> &m, int tag) {   do_reduction(a, b, m);};
template <class T_Config>
void CommsMPIHostBufferStream<T_Config>::global_reduce(HFVector_Array &a, HFVector &b, const Operator<TConfig> &m, int tag) {   do_reduction(a, b, m);};
template <class T_Config>
void CommsMPIHostBufferStream<T_Config>::global_reduce(HCVector_Array &a, HCVector &b, const Operator<TConfig> &m, int tag) {   do_reduction(a, b, m);};
template <class T_Config>
void CommsMPIHostBufferStream<T_Config>::global_reduce(HZVector_Array &a, HZVector &b, const Operator<TConfig> &m, int tag) {   do_reduction(a, b, m);};
template <class T_Config>
void CommsMPIHostBufferStream<T_Config>::global_reduce(HIVector_Array &a, HIVector &b, const Operator<TConfig> &m, int tag) {   do_reduction(a, b, m);};
template <class T_Config>
void CommsMPIHostBufferStream<T_Config>::global_reduce(HBVector_Array &a, HBVector &b, const Operator<TConfig> &m, int tag) {   FatalError("MPI Comms boolean reduction not implemented", AMGX_ERR_NOT_IMPLEMENTED);/*do_reduction(a,b,m);*/};
template <class T_Config>
void CommsMPIHostBufferStream<T_Config>::global_reduce(HI64Vector_Array &a, HI64Vector &b, const Operator<TConfig> &m, int tag) {   do_reduction(a, b, m);};

template <class T_Config>
void CommsMPIHostBufferStream<T_Config>::global_reduce_sum(HDVector &a, HDVector &b, const Matrix<TConfig> &m, int tag)
{
#ifdef AMGX_WITH_MPI
    MPI_Allreduce(&b[0], &a[0], b.size(), MPI_DOUBLE, MPI_SUM, mpi_comm);
#else
    FatalError("MPI Comms module requires compiling with MPI", AMGX_ERR_NOT_IMPLEMENTED);
#endif
};

template <class T_Config>
void CommsMPIHostBufferStream<T_Config>::global_reduce_sum(HFVector &a, HFVector &b, const Matrix<TConfig> &m, int tag)
{
#ifdef AMGX_WITH_MPI
    MPI_Allreduce(&b[0], &a[0], b.size(), MPI_FLOAT, MPI_SUM, mpi_comm);
#else
    FatalError("MPI Comms module requires compiling with MPI", AMGX_ERR_NOT_IMPLEMENTED);
#endif
};

template <class T_Config>
void CommsMPIHostBufferStream<T_Config>::global_reduce_sum(HCVector &a, HCVector &b, const Matrix<TConfig> &m, int tag)
{
    FatalError("MPI sum reduce is now supported with complex types", AMGX_ERR_NOT_IMPLEMENTED);
};

template <class T_Config>
void CommsMPIHostBufferStream<T_Config>::global_reduce_sum(HZVector &a, HZVector &b, const Matrix<TConfig> &m, int tag)
{
    FatalError("MPI sum reduce is now supported with complex types", AMGX_ERR_NOT_IMPLEMENTED);
};

template <class T_Config>
void CommsMPIHostBufferStream<T_Config>::global_reduce_sum(HIVector &a, HIVector &b, const Matrix<TConfig> &m, int tag)
{
#ifdef AMGX_WITH_MPI
    MPI_Allreduce(&b[0], &a[0], b.size(), MPI_INT, MPI_SUM, mpi_comm);
#else
    FatalError("MPI Comms module requires compiling with MPI", AMGX_ERR_NOT_IMPLEMENTED);
#endif
};

template <class T_Config>
void CommsMPIHostBufferStream<T_Config>::global_reduce_sum(HI64Vector &a, HI64Vector &b, const Matrix<TConfig> &m, int tag)
{
#ifdef AMGX_WITH_MPI
    MPI_Allreduce(&b[0], &a[0], b.size(), MPI_LONG_LONG_INT, MPI_SUM, mpi_comm);
#else
    FatalError("MPI Comms module requires compiling with MPI", AMGX_ERR_NOT_IMPLEMENTED);
#endif
};

template <class T_Config>
void CommsMPIHostBufferStream<T_Config>::exchange_vectors(DVector_Array &a, const Matrix<TConfig> &m, int tag) {              do_vec_exchange(a, m);};
template <class T_Config>
void CommsMPIHostBufferStream<T_Config>::exchange_vectors(FVector_Array &a, const Matrix<TConfig> &m, int tag) {              do_vec_exchange(a, m);};
template <class T_Config>
void CommsMPIHostBufferStream<T_Config>::exchange_vectors(CVector_Array &a, const Matrix<TConfig> &m, int tag) {              do_vec_exchange(a, m);};
template <class T_Config>
void CommsMPIHostBufferStream<T_Config>::exchange_vectors(ZVector_Array &a, const Matrix<TConfig> &m, int tag) {              do_vec_exchange(a, m);};
template <class T_Config>
void CommsMPIHostBufferStream<T_Config>::exchange_vectors(IVector_Array &a, const Matrix<TConfig> &m, int tag) {              do_vec_exchange(a, m);};
template <class T_Config>
void CommsMPIHostBufferStream<T_Config>::exchange_vectors(BVector_Array &a, const Matrix<TConfig> &m, int tag) {              do_vec_exchange(a, m);};

template <class T_Config>
void CommsMPIHostBufferStream<T_Config>::send_vector(DIVector &a, int destination, int tag, int offset, int size) { send_vec(a, destination, tag, offset, size);};
template <class T_Config>
void CommsMPIHostBufferStream<T_Config>::send_vector(HIVector &a, int destination, int tag, int offset, int size) { send_vec(a, destination, tag, offset, size);};
template <class T_Config>
void CommsMPIHostBufferStream<T_Config>::send_vector(DDVector &a, int destination, int tag, int offset, int size) { send_vec(a, destination, tag, offset, size);};
template <class T_Config>
void CommsMPIHostBufferStream<T_Config>::send_vector(HDVector &a, int destination, int tag, int offset, int size) { send_vec(a, destination, tag, offset, size);};
template <class T_Config>
void CommsMPIHostBufferStream<T_Config>::send_vector(DFVector &a, int destination, int tag, int offset, int size) { send_vec(a, destination, tag, offset, size);};
template <class T_Config>
void CommsMPIHostBufferStream<T_Config>::send_vector(HFVector &a, int destination, int tag, int offset, int size) { send_vec(a, destination, tag, offset, size);};
template <class T_Config>
void CommsMPIHostBufferStream<T_Config>::send_vector(DCVector &a, int destination, int tag, int offset, int size) { send_vec(a, destination, tag, offset, size);};
template <class T_Config>
void CommsMPIHostBufferStream<T_Config>::send_vector(HCVector &a, int destination, int tag, int offset, int size) { send_vec(a, destination, tag, offset, size);};
template <class T_Config>
void CommsMPIHostBufferStream<T_Config>::send_vector(DZVector &a, int destination, int tag, int offset, int size) { send_vec(a, destination, tag, offset, size);};
template <class T_Config>
void CommsMPIHostBufferStream<T_Config>::send_vector(HZVector &a, int destination, int tag, int offset, int size) { send_vec(a, destination, tag, offset, size);};


template <class T_Config>
void CommsMPIHostBufferStream<T_Config>::send_vector_async(DIVector &a, int destination, int tag, int offset, int size) { send_vec_async(a, destination, tag, offset, size);};
template <class T_Config>
void CommsMPIHostBufferStream<T_Config>::send_vector_async(HIVector &a, int destination, int tag, int offset, int size) { send_vec_async(a, destination, tag, offset, size);};
template <class T_Config>
void CommsMPIHostBufferStream<T_Config>::send_vector_async(DDVector &a, int destination, int tag, int offset, int size) { send_vec_async(a, destination, tag, offset, size);};
template <class T_Config>
void CommsMPIHostBufferStream<T_Config>::send_vector_async(HDVector &a, int destination, int tag, int offset, int size) { send_vec_async(a, destination, tag, offset, size);};
template <class T_Config>
void CommsMPIHostBufferStream<T_Config>::send_vector_async(DFVector &a, int destination, int tag, int offset, int size) { send_vec_async(a, destination, tag, offset, size);};
template <class T_Config>
void CommsMPIHostBufferStream<T_Config>::send_vector_async(HFVector &a, int destination, int tag, int offset, int size) { send_vec_async(a, destination, tag, offset, size);};
template <class T_Config>
void CommsMPIHostBufferStream<T_Config>::send_vector_async(DCVector &a, int destination, int tag, int offset, int size) { send_vec_async(a, destination, tag, offset, size);};
template <class T_Config>
void CommsMPIHostBufferStream<T_Config>::send_vector_async(HCVector &a, int destination, int tag, int offset, int size) { send_vec_async(a, destination, tag, offset, size);};
template <class T_Config>
void CommsMPIHostBufferStream<T_Config>::send_vector_async(DZVector &a, int destination, int tag, int offset, int size) { send_vec_async(a, destination, tag, offset, size);};
template <class T_Config>
void CommsMPIHostBufferStream<T_Config>::send_vector_async(HZVector &a, int destination, int tag, int offset, int size) { send_vec_async(a, destination, tag, offset, size);};


template <class T_Config>
void CommsMPIHostBufferStream<T_Config>::send_vector_wait_all(DIVector &a) { send_vec_wait_all(a);};
template <class T_Config>
void CommsMPIHostBufferStream<T_Config>::send_vector_wait_all(HIVector &a) { send_vec_wait_all(a);};
template <class T_Config>
void CommsMPIHostBufferStream<T_Config>::send_vector_wait_all(DDVector &a) { send_vec_wait_all(a);};
template <class T_Config>
void CommsMPIHostBufferStream<T_Config>::send_vector_wait_all(HDVector &a) { send_vec_wait_all(a);};
template <class T_Config>
void CommsMPIHostBufferStream<T_Config>::send_vector_wait_all(DFVector &a) { send_vec_wait_all(a);};
template <class T_Config>
void CommsMPIHostBufferStream<T_Config>::send_vector_wait_all(HFVector &a) { send_vec_wait_all(a);};
template <class T_Config>
void CommsMPIHostBufferStream<T_Config>::send_vector_wait_all(DCVector &a) { send_vec_wait_all(a);};
template <class T_Config>
void CommsMPIHostBufferStream<T_Config>::send_vector_wait_all(HCVector &a) { send_vec_wait_all(a);};
template <class T_Config>
void CommsMPIHostBufferStream<T_Config>::send_vector_wait_all(DZVector &a) { send_vec_wait_all(a);};
template <class T_Config>
void CommsMPIHostBufferStream<T_Config>::send_vector_wait_all(HZVector &a) { send_vec_wait_all(a);};


template <class T_Config>
void CommsMPIHostBufferStream<T_Config>::recv_vector(DIVector &a, int source, int tag, int offset, int size) { recv_vec(a, source, tag, offset, size);};
template <class T_Config>
void CommsMPIHostBufferStream<T_Config>::recv_vector(HIVector &a, int source, int tag, int offset, int size) { recv_vec(a, source, tag, offset, size);};
template <class T_Config>
void CommsMPIHostBufferStream<T_Config>::recv_vector(DDVector &a, int source, int tag, int offset, int size) { recv_vec(a, source, tag, offset, size);};
template <class T_Config>
void CommsMPIHostBufferStream<T_Config>::recv_vector(HDVector &a, int source, int tag, int offset, int size) { recv_vec(a, source, tag, offset, size);};
template <class T_Config>
void CommsMPIHostBufferStream<T_Config>::recv_vector(DFVector &a, int source, int tag, int offset, int size) { recv_vec(a, source, tag, offset, size);};
template <class T_Config>
void CommsMPIHostBufferStream<T_Config>::recv_vector(HFVector &a, int source, int tag, int offset, int size) { recv_vec(a, source, tag, offset, size);};
template <class T_Config>
void CommsMPIHostBufferStream<T_Config>::recv_vector(DCVector &a, int source, int tag, int offset, int size) { recv_vec(a, source, tag, offset, size);};
template <class T_Config>
void CommsMPIHostBufferStream<T_Config>::recv_vector(HCVector &a, int source, int tag, int offset, int size) { recv_vec(a, source, tag, offset, size);};
template <class T_Config>
void CommsMPIHostBufferStream<T_Config>::recv_vector(DZVector &a, int source, int tag, int offset, int size) { recv_vec(a, source, tag, offset, size);};
template <class T_Config>
void CommsMPIHostBufferStream<T_Config>::recv_vector(HZVector &a, int source, int tag, int offset, int size) { recv_vec(a, source, tag, offset, size);};


template <class T_Config>
void CommsMPIHostBufferStream<T_Config>::recv_vector_async(DIVector &a, int source, int tag, int offset, int size) { recv_vec_async(a, source, tag, offset, size);};
template <class T_Config>
void CommsMPIHostBufferStream<T_Config>::recv_vector_async(HIVector &a, int source, int tag, int offset, int size) { recv_vec_async(a, source, tag, offset, size);};
template <class T_Config>
void CommsMPIHostBufferStream<T_Config>::recv_vector_async(DDVector &a, int source, int tag, int offset, int size) { recv_vec_async(a, source, tag, offset, size);};
template <class T_Config>
void CommsMPIHostBufferStream<T_Config>::recv_vector_async(HDVector &a, int source, int tag, int offset, int size) { recv_vec_async(a, source, tag, offset, size);};
template <class T_Config>
void CommsMPIHostBufferStream<T_Config>::recv_vector_async(DFVector &a, int source, int tag, int offset, int size) { recv_vec_async(a, source, tag, offset, size);};
template <class T_Config>
void CommsMPIHostBufferStream<T_Config>::recv_vector_async(HFVector &a, int source, int tag, int offset, int size) { recv_vec_async(a, source, tag, offset, size);};
template <class T_Config>
void CommsMPIHostBufferStream<T_Config>::recv_vector_async(DCVector &a, int source, int tag, int offset, int size) { recv_vec_async(a, source, tag, offset, size);};
template <class T_Config>
void CommsMPIHostBufferStream<T_Config>::recv_vector_async(HCVector &a, int source, int tag, int offset, int size) { recv_vec_async(a, source, tag, offset, size);};
template <class T_Config>
void CommsMPIHostBufferStream<T_Config>::recv_vector_async(DZVector &a, int source, int tag, int offset, int size) { recv_vec_async(a, source, tag, offset, size);};
template <class T_Config>
void CommsMPIHostBufferStream<T_Config>::recv_vector_async(HZVector &a, int source, int tag, int offset, int size) { recv_vec_async(a, source, tag, offset, size);};

template <class T_Config>
void CommsMPIHostBufferStream<T_Config>::recv_vector_wait_all(DIVector &a) { recv_vec_wait_all(a);};
template <class T_Config>
void CommsMPIHostBufferStream<T_Config>::recv_vector_wait_all(HIVector &a) { recv_vec_wait_all(a);};
template <class T_Config>
void CommsMPIHostBufferStream<T_Config>::recv_vector_wait_all(DDVector &a) { recv_vec_wait_all(a);};
template <class T_Config>
void CommsMPIHostBufferStream<T_Config>::recv_vector_wait_all(HDVector &a) { recv_vec_wait_all(a);};
template <class T_Config>
void CommsMPIHostBufferStream<T_Config>::recv_vector_wait_all(DFVector &a) { recv_vec_wait_all(a);};
template <class T_Config>
void CommsMPIHostBufferStream<T_Config>::recv_vector_wait_all(HFVector &a) { recv_vec_wait_all(a);};
template <class T_Config>
void CommsMPIHostBufferStream<T_Config>::recv_vector_wait_all(DCVector &a) { recv_vec_wait_all(a);};
template <class T_Config>
void CommsMPIHostBufferStream<T_Config>::recv_vector_wait_all(HCVector &a) { recv_vec_wait_all(a);};
template <class T_Config>
void CommsMPIHostBufferStream<T_Config>::recv_vector_wait_all(DZVector &a) { recv_vec_wait_all(a);};
template <class T_Config>
void CommsMPIHostBufferStream<T_Config>::recv_vector_wait_all(HZVector &a) { recv_vec_wait_all(a);};

template <class T_Config>
int CommsMPIHostBufferStream<T_Config>::get_num_partitions()
{
#ifdef AMGX_WITH_MPI
    int total = 0;
    MPI_Comm_size( mpi_comm, &total );
    return total;
#else
    FatalError("MPI Comms module requires compiling with MPI", AMGX_ERR_NOT_IMPLEMENTED);
#endif
}

template <class T_Config>
int CommsMPIHostBufferStream<T_Config>::get_global_id()
{
#ifdef AMGX_WITH_MPI
    int rank = 0;
    MPI_Comm_rank( mpi_comm, &rank);
    return rank;
#else
    FatalError("MPI Comms module requires compiling with MPI", AMGX_ERR_NOT_IMPLEMENTED);
#endif
}


template <class T_Config>
void CommsMPIHostBufferStream<T_Config>::barrier()
{
#ifdef AMGX_WITH_MPI
    MPI_Barrier(mpi_comm);
#else
    FatalError("MPI Comms module requires compiling with MPI", AMGX_ERR_NOT_IMPLEMENTED);
#endif
}

#define MAX_HOSTNAME_LENGTH 256
template <class T_Config>
void CommsMPIHostBufferStream<T_Config>::get_hostname(std::string &my_hostname)
{
#ifdef AMGX_WITH_MPI
    my_hostname.resize(MAX_HOSTNAME_LENGTH);
    int hostname_length;
    MPI_Get_processor_name(&my_hostname[0], &hostname_length);
    my_hostname.resize(hostname_length);
#else
    FatalError("MPI Comms module requires compiling with MPI", AMGX_ERR_NOT_IMPLEMENTED);
#endif
}


template <class T_Config>
void CommsMPIHostBufferStream<T_Config>::exchange_hostnames(std::string &my_hostname, std::vector<std::string> &hostnames, int num_parts)
{
#ifdef AMGX_WITH_MPI

    if (num_parts != this->get_num_partitions())
    {
        FatalError("Inconsistent number of partitions", AMGX_ERR_CORE);
    }

    hostnames.resize(num_parts);
    std::vector<int> hostname_lengths(num_parts);
    std::vector<int> displs(num_parts + 1);
    int my_hostname_length = my_hostname.length();
    MPI_Allgather( &my_hostname_length, 1, MPI_INT, hostname_lengths.data(), 1, MPI_INT, mpi_comm);
    // Compute displacement for each hostname
    displs[0] = 0;
    int totlen = 0;

    for (int i = 0; i < num_parts; i++)
    {
        displs[i + 1] = displs[i] + hostname_lengths[i];
        totlen      += hostname_lengths[i];
    }

    std::string all_names;
    all_names.resize(totlen);
    MPI_Allgatherv( &my_hostname[0], my_hostname_length, MPI_CHAR,
                    (void *)(all_names.data()), hostname_lengths.data(), displs.data(), MPI_CHAR,
                    mpi_comm);

    for (int i = 0; i < num_parts; i++)
    {
        hostnames[i] = std::string(&all_names[displs[i]], hostname_lengths[i]);
    }

#else
    FatalError("MPI Comms module requires compiling with MPI", AMGX_ERR_NOT_IMPLEMENTED);
#endif
}

template <class T_Config>
void CommsMPIHostBufferStream<T_Config>::all_gather(const int &my_data, HIVector &gathered_data, int num_parts) { all_gather_templated(my_data, gathered_data, num_parts); }

template <class T_Config>
void CommsMPIHostBufferStream<T_Config>::all_gather(const int64_t &my_data, HI64Vector &gathered_data, int num_parts) { all_gather_templated(my_data, gathered_data, num_parts); }

template <class T_Config>
void CommsMPIHostBufferStream<T_Config>::all_gather_v(HIVector &my_data, HIVector &gathered_data, int num_parts) { all_gather_v_templated(my_data[0], my_data.size(), gathered_data, num_parts); }

template <class T_Config>
void CommsMPIHostBufferStream<T_Config>::all_reduce_max(IndexType_h &my_data, IndexType_h &result_data)
{
#ifdef AMGX_WITH_MPI
    MPI_Allreduce(&my_data, &result_data, 1, MPI_INT, MPI_MAX, mpi_comm);
#else
    FatalError("MPI Comms module requires compiling with MPI", AMGX_ERR_NOT_IMPLEMENTED);
#endif
}


template <class T_Config>
template <class T, class T2>
void CommsMPIHostBufferStream<T_Config>::all_gather_templated(const T &my_data, T2 &gathered_data, int num_parts)
{
#ifdef AMGX_WITH_MPI
    gathered_data.resize(num_parts);
    MPI_Allgather(&my_data, sizeof(T), MPI_BYTE,
                  &gathered_data[0], sizeof(T), MPI_BYTE,
                  mpi_comm);
#else
    FatalError("MPI Comms module requires compiling with MPI", AMGX_ERR_NOT_IMPLEMENTED);
#endif
}

template <class T_Config>
template <class T, class T2>
void CommsMPIHostBufferStream<T_Config>::all_gather_v_templated(T &my_data, int num_elems, T2 &gathered_data, int num_parts)
{
#ifdef AMGX_WITH_MPI
    gathered_data.resize(num_parts * num_elems);
    MPI_Allgather(&my_data, num_elems * sizeof(T), MPI_BYTE,
                  &gathered_data[0], num_elems * sizeof(T), MPI_BYTE,
                  mpi_comm);
#else
    FatalError("MPI Comms module requires compiling with MPI", AMGX_ERR_NOT_IMPLEMENTED);
#endif
}

template <class T_Config>
void CommsMPIHostBufferStream<T_Config>::all_gather_v(HDVector& data, int num_elems, HDVector& gathered_data, HIVector counts, HIVector displs)
{
#ifdef AMGX_WITH_MPI
    MPI_Allgatherv(data.raw(), num_elems, MPI_DOUBLE, gathered_data.raw(), counts.raw(), displs.raw(), MPI_DOUBLE, mpi_comm);
#else
    FatalError("MPI Comms module requires compiling with MPI", AMGX_ERR_NOT_IMPLEMENTED);
#endif
}

template <class T_Config>
void CommsMPIHostBufferStream<T_Config>::all_gather_v(HFVector& data, int num_elems, HFVector& gathered_data, HIVector counts, HIVector displs)
{
#ifdef AMGX_WITH_MPI
    MPI_Allgatherv(data.raw(), num_elems, MPI_FLOAT, gathered_data.raw(), counts.raw(), displs.raw(), MPI_FLOAT, mpi_comm);
#else
    FatalError("MPI Comms module requires compiling with MPI", AMGX_ERR_NOT_IMPLEMENTED);
#endif
}

template <class T_Config>
void CommsMPIHostBufferStream<T_Config>::all_gather_v(HCVector& data, int num_elems, HCVector& gathered_data, HIVector counts, HIVector displs)
{
#ifdef AMGX_WITH_MPI
    FatalError("AllgatherV with complex data.", AMGX_ERR_NOT_IMPLEMENTED);
#else
    FatalError("MPI Comms module requires compiling with MPI", AMGX_ERR_NOT_IMPLEMENTED);
#endif
}

template <class T_Config>
void CommsMPIHostBufferStream<T_Config>::all_gather_v(HZVector& data, int num_elems, HZVector& gathered_data, HIVector counts, HIVector displs)
{
#ifdef AMGX_WITH_MPI
    FatalError("AllgatherV with complex data.", AMGX_ERR_NOT_IMPLEMENTED);
#else
    FatalError("MPI Comms module requires compiling with MPI", AMGX_ERR_NOT_IMPLEMENTED);
#endif
}

template <class T_Config>
void CommsMPIHostBufferStream<T_Config>::all_gather_v(HIVector& data, int num_elems, HIVector& gathered_data, HIVector counts, HIVector displs)
{
#ifdef AMGX_WITH_MPI
    MPI_Allgatherv(data.raw(), num_elems, MPI_INT, gathered_data.raw(), counts.raw(), displs.raw(), MPI_INT, mpi_comm);
#else
    FatalError("MPI Comms module requires compiling with MPI", AMGX_ERR_NOT_IMPLEMENTED);
#endif
}


/****************************************
 * Explict instantiations
 ***************************************/
#define AMGX_CASE_LINE(CASE) template class CommsMPIHostBufferStream<TemplateMode<CASE>::Type>;
AMGX_FORALL_BUILDS(AMGX_CASE_LINE)
AMGX_FORCOMPLEX_BUILDS(AMGX_CASE_LINE)
#undef AMGX_CASE_LINE

} // namespace amgx
