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

#include <distributed/comms_mpi.h>
#include <distributed/comms_visitors.h>//
#include <vector.h>

namespace amgx
{

template <class T_Config> class CommsMPIHostBufferStream;

template <class T_Config>
class CommsMPIHostBufferStream : public CommsMPI<T_Config>
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
        typedef typename Base::CVector CVector;
        typedef typename Base::CVector_Array CVector_Array;
        typedef typename Base::ZVector ZVector;
        typedef typename Base::ZVector_Array ZVector_Array;
        typedef typename Base::IVector IVector;
        typedef typename Base::IVector_h IVector_h;
        typedef typename Base::IVector_Array IVector_Array;
        typedef typename Base::BVector BVector;
        typedef typename Base::BVector_Array BVector_Array;
        typedef typename Base::I64Vector I64Vector;
        typedef typename Base::I64Vector_Array I64Vector_Array;

        typedef typename Base::Matrix_h Matrix_h;
        typedef typename Base::TConfig_h TConfig_h;
        typedef DistributedManager<TConfig_h> Manager_h;
        typedef typename Base::HMVector HMVector;
        typedef typename Base::HDVector HDVector;
        typedef typename Base::HDVector_Array HDVector_Array;
        typedef typename Base::HFVector HFVector;
        typedef typename Base::HFVector_Array HFVector_Array;
        typedef typename Base::HCVector HCVector;
        typedef typename Base::HCVector_Array HCVector_Array;
        typedef typename Base::HZVector HZVector;
        typedef typename Base::HZVector_Array HZVector_Array;
        typedef typename Base::HIVector HIVector;
        typedef typename Base::HI64Vector HI64Vector;
        typedef typename Base::HI64Vector_Array HI64Vector_Array;
        typedef typename Base::DIVector DIVector;
        typedef typename Base::DFVector DFVector;
        typedef typename Base::DDVector DDVector;
        typedef typename Base::DCVector DCVector;
        typedef typename Base::DZVector DZVector;
        typedef typename Base::HIVector_Array HIVector_Array;
        typedef typename Base::HBVector HBVector;
        typedef typename Base::HBVector_Array HBVector_Array;

        typedef typename Base::Matrix_Array Matrix_Array;
        typedef typename Base::DistributedManager_Array DistributedManager_Array;

        typedef typename TConfig_h::IndPrec IndexType_h;

#ifdef AMGX_WITH_MPI
        CommsMPIHostBufferStream(AMG_Config &cfg, const std::string &cfg_scope, const MPI_Comm *mpi_communicator) : CommsMPI<T_Config>(cfg, cfg_scope)
        {
            min_rows_latency_hiding = cfg.AMG_Config::getParameter<int>("min_rows_latency_hiding", "default");
            this->halo_coloring = cfg.AMG_Config::getParameter<ColoringType>("halo_coloring", "default");
            MPI_Comm_dup(*mpi_communicator, &mpi_comm);
            MPI_Comm_set_errhandler(mpi_comm, glbMPIErrorHandler);
        };
#endif

        CommsMPIHostBufferStream(AMG_Config &cfg, const std::string &cfg_scope) : CommsMPI<T_Config>(cfg, cfg_scope)
        {
#ifdef AMGX_WITH_MPI
            MPI_Comm_dup(MPI_COMM_WORLD, &mpi_comm);
            MPI_Comm_set_errhandler(mpi_comm, glbMPIErrorHandler);
#endif
            min_rows_latency_hiding = cfg.AMG_Config::getParameter<int>("min_rows_latency_hiding", "default");
            this->halo_coloring = cfg.AMG_Config::getParameter<ColoringType>("halo_coloring", "default");
        };

        CommsMPIHostBufferStream(const CommsMPIHostBufferStream &comm) : CommsMPI<T_Config>()
        {
#ifdef AMGX_WITH_MPI
            MPI_Comm_dup(comm.mpi_comm, &this->mpi_comm);
            MPI_Comm_set_errhandler(mpi_comm, glbMPIErrorHandler);
#endif
            this->min_rows_latency_hiding = comm.min_rows_latency_hiding;
            this->halo_coloring = comm.halo_coloring;
        };

#ifdef AMGX_WITH_MPI
        CommsMPIHostBufferStream(const CommsMPIHostBufferStream *comm, MPI_Comm *new_comm) : CommsMPI<T_Config>()
        {
            this->mpi_comm = *new_comm;
            this->min_rows_latency_hiding = comm->min_rows_latency_hiding;
            this->halo_coloring = comm->halo_coloring;
        };
#endif

#ifdef AMGX_WITH_MPI
        MPI_Comm get_mpi_comm()
        {
            return mpi_comm;
        }
        void set_mpi_comm(MPI_Comm &new_comm)
        {
            mpi_comm = new_comm;
        }
#endif

        void printString(const std::string &str);

        CommsMPIHostBufferStream<TConfig> *Clone() const
        {
            CommsMPIHostBufferStream<TConfig> *ret = new CommsMPIHostBufferStream<TConfig>(*this);
            return ret;
        }

        virtual ~CommsMPIHostBufferStream()
        {
#ifdef AMGX_WITH_MPI
            MPI_Comm_free(&mpi_comm);
            MPI_Waitall(requests.size(), &requests[0], &statuses[0]); // complete possible requests in flight
#endif
        }


        void set_neighbors(int num_neighbors)
        {
            neighbors = num_neighbors;
#ifdef AMGX_WITH_MPI
            requests.resize(10 * neighbors);

            for (int i = 0; i < 10 * neighbors; i++)
            {
                requests[i] = MPI_REQUEST_NULL;
            }

            statuses.resize(10 * neighbors);
#endif
        }
    private:
        int min_rows_latency_hiding;

    public:
        void Accept(VisitorBase<T_Config> &v)
        {
            v.VisitCommsHostbuffer(*this);
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

        template <class T>
        void do_setup(T &b, const Matrix<TConfig> &m, int num_rings);

        template <class T>
        void do_setup_L2H(T &b, Matrix<TConfig> &m, int num_rings);

        template <class T, class T2>
        void do_reduction(T &arrays, T2 &data, const Operator<TConfig> &m);

        template <AMGX_VecPrecision t_vecPrec, AMGX_MatPrecision t_matPrec, AMGX_IndPrecision t_indPrec>
        void do_vec_exchange(std::vector<Vector<TemplateConfig<TConfig::memSpace, t_vecPrec, t_matPrec, t_indPrec> > > &b, const Matrix<TConfig> &m);

        template <class T>
        void do_exchange_halo(T &b, const Matrix<TConfig> &m, int num_rings);

        template <class T>
        void do_exchange_halo_async(T &b, const Matrix<TConfig> &m, cudaEvent_t event, int tag, cudaStream_t stream);

        template <class T>
        void do_send_receive_wait(T &b, const Matrix<TConfig> &m, cudaEvent_t event, int tag, cudaStream_t &stream);

        template <class T>
        void do_exchange_halo_wait(T &b, const Matrix<TConfig> &m, cudaStream_t stream);

        template <class T>
        bool do_exchange_halo_query( T &b, const Matrix<TConfig> &m);

        template <class T>
        void do_add_from_halo(T &b, const Matrix<TConfig> &m, int num_rings, cudaStream_t &stream);

        template <class T>
        void do_gather_L2H(T &b, const Matrix<TConfig> &m, int num_rings, cudaStream_t stream);

        template <class T>
        void do_gather_L2H_v2(T &b, const Matrix<TConfig> &m, int num_rings, cudaStream_t stream);

        template <class T>
        void send_vec(T &b, int destination, int tag, int offset, int size);

        template <class T>
        void send_vec_async(T &b, int destination, int tag, int offset, int size);

        template <class T>
        void send_vec_wait_all(T &b);

        template <class T>
        void recv_vec(T &b, int source, int tag, int offset, int size);

        template <class T>
        void recv_vec_async(T &b, int source, int tag, int offset, int size);

        template <class T>
        void recv_vec_wait_all(T &b);

        template <class T, class T2>
        void all_gather_templated(const T &my_data, T2 &gathered_data, int num_parts);

        template <class T, class T2>
        void all_gather_v_templated(T &my_data, int num_elems, T2 &gathered_data, int num_parts);

        int neighbors;
#ifdef AMGX_WITH_MPI
        std::vector<MPI_Request> requests;
        std::vector<MPI_Status> statuses;
    private:
        MPI_Comm mpi_comm;
#endif

    public:
        int get_neighbors(void) const
        {
            return neighbors;
        }

#ifdef AMGX_WITH_MPI
        std::vector<MPI_Request> &get_requests(void)
        {
            return requests;
        }

        const std::vector<MPI_Request> &get_requests(void) const
        {
            return requests;
        }
#endif

        void exchange_matrix_halo(Matrix_Array &halo_rows, DistributedManager_Array &halo_btl, const Matrix<TConfig> &m);
        void exchange_matrix_halo(IVector_Array &row_offsets, I64Vector_Array &col_indices, MVector_Array &values, I64Vector_Array &halo_row_ids, IVector_h &neighbors_list, int global_id);
        void setup(DVector &b, const Matrix<TConfig> &m, int tag, int num_rings = 1);
        void setup_L2H(DVector &b, Matrix<TConfig> &m, int num_rings = 1);
        void exchange_halo(DVector &b, const Matrix<TConfig> &m, int tag, int num_rings = 1);
        void exchange_halo_async(DVector &b, const Matrix<TConfig> &m, cudaEvent_t event, int tag, cudaStream_t stream = NULL);
        void send_receive_wait(DVector &b, const Matrix<TConfig> &m, cudaEvent_t event, int tag, cudaStream_t &stream);
        void exchange_halo_wait(DVector &b, const Matrix<TConfig> &m, cudaEvent_t event, int tag, cudaStream_t stream = NULL);
        void setup(FVector &b, const Matrix<TConfig> &m, int tag, int num_rings = 1);
        void setup_L2H(FVector &b, Matrix<TConfig> &m, int num_rings = 1);
        void exchange_halo(FVector &b, const Matrix<TConfig> &m, int tag, int num_rings = 1);
        void exchange_halo_async(FVector &b, const Matrix<TConfig> &m, cudaEvent_t event, int tag, cudaStream_t stream = NULL);
        void send_receive_wait(FVector &b, const Matrix<TConfig> &m, cudaEvent_t event, int tag, cudaStream_t &stream);
        void exchange_halo_wait(FVector &b, const Matrix<TConfig> &m, cudaEvent_t event, int tag, cudaStream_t stream = NULL);
        void setup(CVector &b, const Matrix<TConfig> &m, int tag, int num_rings = 1);
        void setup_L2H(CVector &b, Matrix<TConfig> &m, int num_rings = 1);
        void exchange_halo(CVector &b, const Matrix<TConfig> &m, int tag, int num_rings = 1);
        void exchange_halo_async(CVector &b, const Matrix<TConfig> &m, cudaEvent_t event, int tag, cudaStream_t stream = NULL);
        void send_receive_wait(CVector &b, const Matrix<TConfig> &m, cudaEvent_t event, int tag, cudaStream_t &stream);
        void exchange_halo_wait(CVector &b, const Matrix<TConfig> &m, cudaEvent_t event, int tag, cudaStream_t stream = NULL);
        void setup(ZVector &b, const Matrix<TConfig> &m, int tag, int num_rings = 1);
        void setup_L2H(ZVector &b, Matrix<TConfig> &m, int num_rings = 1);
        void exchange_halo(ZVector &b, const Matrix<TConfig> &m, int tag, int num_rings = 1);
        void exchange_halo_async(ZVector &b, const Matrix<TConfig> &m, cudaEvent_t event, int tag, cudaStream_t stream = NULL);
        void send_receive_wait(ZVector &b, const Matrix<TConfig> &m, cudaEvent_t event, int tag, cudaStream_t &stream);
        void exchange_halo_wait(ZVector &b, const Matrix<TConfig> &m, cudaEvent_t event, int tag, cudaStream_t stream = NULL);
        void setup(IVector &b, const Matrix<TConfig> &m, int tag, int num_rings = 1);
        void setup_L2H(IVector &b, Matrix<TConfig> &m, int num_rings = 1);
        void exchange_halo(IVector &b, const Matrix<TConfig> &m, int tag, int num_rings = 1);
        void exchange_halo_async(IVector &b, const Matrix<TConfig> &m, cudaEvent_t event, int tag, cudaStream_t stream = NULL);
        void send_receive_wait(IVector &b, const Matrix<TConfig> &m, cudaEvent_t event, int tag, cudaStream_t &stream);
        void exchange_halo_wait(IVector &b, const Matrix<TConfig> &m, cudaEvent_t event, int tag, cudaStream_t stream = NULL);
        void setup(BVector &b, const Matrix<TConfig> &m, int tag, int num_rings = 1);
        void setup_L2H(BVector &b, Matrix<TConfig> &m, int num_rings = 1);
        void exchange_halo(BVector &b, const Matrix<TConfig> &m, int tag, int num_rings = 1);
        void exchange_halo_async(BVector &b, const Matrix<TConfig> &m, cudaEvent_t event, int tag, cudaStream_t stream = NULL);
        void send_receive_wait(BVector &b, const Matrix<TConfig> &m, cudaEvent_t event, int tag, cudaStream_t &stream);
        void exchange_halo_wait(BVector &b, const Matrix<TConfig> &m, cudaEvent_t event, int tag, cudaStream_t stream = NULL);
        void setup(I64Vector &b, const Matrix<TConfig> &m, int tag, int num_rings = 1);
        void setup_L2H(I64Vector &b, Matrix<TConfig> &m, int num_rings = 1);
        void exchange_halo(I64Vector &b, const Matrix<TConfig> &m, int tag, int num_rings = 1);
        void exchange_halo_async(I64Vector &b, const Matrix<TConfig> &m, cudaEvent_t event, int tag, cudaStream_t stream = NULL);
        void send_receive_wait(I64Vector &b, const Matrix<TConfig> &m, cudaEvent_t event, int tag, cudaStream_t &stream);
        void exchange_halo_wait(I64Vector &b, const Matrix<TConfig> &m, cudaEvent_t event, int tag, cudaStream_t stream = NULL);

        void add_from_halo(IVector &b, const Matrix<TConfig> &m, int tag, int num_rings, cudaStream_t &stream);
        void add_from_halo(DVector &b, const Matrix<TConfig> &m, int tag, int num_rings, cudaStream_t &stream);
        void add_from_halo(FVector &b, const Matrix<TConfig> &m, int tag, int num_rings, cudaStream_t &stream);
        void add_from_halo(CVector &b, const Matrix<TConfig> &m, int tag, int num_rings, cudaStream_t &stream);
        void add_from_halo(ZVector &b, const Matrix<TConfig> &m, int tag, int num_rings, cudaStream_t &stream);

        void gather_L2H(IVector &b, const Matrix<TConfig> &m, int num_rings, cudaStream_t stream = NULL);
        void gather_L2H(DVector &b, const Matrix<TConfig> &m, int num_rings, cudaStream_t stream = NULL);
        void gather_L2H(FVector &b, const Matrix<TConfig> &m, int num_rings, cudaStream_t stream = NULL);
        void gather_L2H(CVector &b, const Matrix<TConfig> &m, int num_rings, cudaStream_t stream = NULL);
        void gather_L2H(ZVector &b, const Matrix<TConfig> &m, int num_rings, cudaStream_t stream = NULL);

        void gather_L2H_v2(IVector &b, const Matrix<TConfig> &m, int num_rings, cudaStream_t stream = NULL);
        void gather_L2H_v2(DVector &b, const Matrix<TConfig> &m, int num_rings, cudaStream_t stream = NULL);
        void gather_L2H_v2(FVector &b, const Matrix<TConfig> &m, int num_rings, cudaStream_t stream = NULL);
        void gather_L2H_v2(CVector &b, const Matrix<TConfig> &m, int num_rings, cudaStream_t stream = NULL);
        void gather_L2H_v2(ZVector &b, const Matrix<TConfig> &m, int num_rings, cudaStream_t stream = NULL);

        void global_reduce(HDVector_Array &a, HDVector &b, const Operator<TConfig> &m, int tag);
        void global_reduce(HFVector_Array &a, HFVector &b, const Operator<TConfig> &m, int tag);
        void global_reduce(HCVector_Array &a, HCVector &b, const Operator<TConfig> &m, int tag);
        void global_reduce(HZVector_Array &a, HZVector &b, const Operator<TConfig> &m, int tag);
        void global_reduce(HIVector_Array &a, HIVector &b, const Operator<TConfig> &m, int tag);
        void global_reduce(HBVector_Array &a, HBVector &b, const Operator<TConfig> &m, int tag);
        void global_reduce(HI64Vector_Array &a, HI64Vector &b, const Operator<TConfig> &m, int tag);

        void global_reduce_sum(HDVector &a, HDVector &b, const Matrix<TConfig> &m, int tag);
        void global_reduce_sum(HFVector &a, HFVector &b, const Matrix<TConfig> &m, int tag);
        void global_reduce_sum(HCVector &a, HCVector &b, const Matrix<TConfig> &m, int tag);
        void global_reduce_sum(HZVector &a, HZVector &b, const Matrix<TConfig> &m, int tag);
        void global_reduce_sum(HIVector &a, HIVector &b, const Matrix<TConfig> &m, int tag);
        void global_reduce_sum(HI64Vector &a, HI64Vector &b, const Matrix<TConfig> &m, int tag);

        void exchange_vectors(DVector_Array &a, const Matrix<TConfig> &m, int tag);
        void exchange_vectors(FVector_Array &a, const Matrix<TConfig> &m, int tag);
        void exchange_vectors(CVector_Array &a, const Matrix<TConfig> &m, int tag);
        void exchange_vectors(ZVector_Array &a, const Matrix<TConfig> &m, int tag);
        void exchange_vectors(IVector_Array &a, const Matrix<TConfig> &m, int tag);
        void exchange_vectors(BVector_Array &a, const Matrix<TConfig> &m, int tag);

        bool exchange_halo_query(DVector &b, const Matrix<TConfig> &m, cudaEvent_t event);
        bool exchange_halo_query(FVector &b, const Matrix<TConfig> &m, cudaEvent_t event);
        bool exchange_halo_query(CVector &b, const Matrix<TConfig> &m, cudaEvent_t event);
        bool exchange_halo_query(ZVector &b, const Matrix<TConfig> &m, cudaEvent_t event);
        bool exchange_halo_query(IVector &b, const Matrix<TConfig> &m, cudaEvent_t event);
        bool exchange_halo_query(BVector &b, const Matrix<TConfig> &m, cudaEvent_t event);
        bool exchange_halo_query(I64Vector &b, const Matrix<TConfig> &m, cudaEvent_t event);

        void send_vector(DIVector &a, int destination, int tag, int offset = 0, int size = -1);
        void send_vector(HIVector &a, int destination, int tag, int offset = 0, int size = -1);
        void send_vector(DDVector &a, int destination, int tag, int offset = 0, int size = -1);
        void send_vector(HDVector &a, int destination, int tag, int offset = 0, int size = -1);
        void send_vector(DFVector &a, int destination, int tag, int offset = 0, int size = -1);
        void send_vector(HFVector &a, int destination, int tag, int offset = 0, int size = -1);
        void send_vector(DCVector &a, int destination, int tag, int offset = 0, int size = -1);
        void send_vector(HCVector &a, int destination, int tag, int offset = 0, int size = -1);
        void send_vector(DZVector &a, int destination, int tag, int offset = 0, int size = -1);
        void send_vector(HZVector &a, int destination, int tag, int offset = 0, int size = -1);


        void send_vector_async(DIVector &a, int destination, int tag, int offset = 0, int size = -1);
        void send_vector_async(HIVector &a, int destination, int tag, int offset = 0, int size = -1);
        void send_vector_async(DDVector &a, int destination, int tag, int offset = 0, int size = -1);
        void send_vector_async(HDVector &a, int destination, int tag, int offset = 0, int size = -1);
        void send_vector_async(DFVector &a, int destination, int tag, int offset = 0, int size = -1);
        void send_vector_async(HFVector &a, int destination, int tag, int offset = 0, int size = -1);
        void send_vector_async(DCVector &a, int destination, int tag, int offset = 0, int size = -1);
        void send_vector_async(HCVector &a, int destination, int tag, int offset = 0, int size = -1);
        void send_vector_async(DZVector &a, int destination, int tag, int offset = 0, int size = -1);
        void send_vector_async(HZVector &a, int destination, int tag, int offset = 0, int size = -1);

        void send_vector_wait_all(DIVector &a);
        void send_vector_wait_all(HIVector &a);
        void send_vector_wait_all(DDVector &a);
        void send_vector_wait_all(HDVector &a);
        void send_vector_wait_all(DFVector &a);
        void send_vector_wait_all(HFVector &a);
        void send_vector_wait_all(DCVector &a);
        void send_vector_wait_all(HCVector &a);
        void send_vector_wait_all(DZVector &a);
        void send_vector_wait_all(HZVector &a);

        void recv_vector(DIVector &a, int source, int tag, int offset = 0, int size = -1);
        void recv_vector(HIVector &a, int source, int tag, int offset = 0, int size = -1);
        void recv_vector(DDVector &a, int source, int tag, int offset = 0, int size = -1);
        void recv_vector(HDVector &a, int source, int tag, int offset = 0, int size = -1);
        void recv_vector(DFVector &a, int source, int tag, int offset = 0, int size = -1);
        void recv_vector(HFVector &a, int source, int tag, int offset = 0, int size = -1);
        void recv_vector(DCVector &a, int source, int tag, int offset = 0, int size = -1);
        void recv_vector(HCVector &a, int source, int tag, int offset = 0, int size = -1);
        void recv_vector(DZVector &a, int source, int tag, int offset = 0, int size = -1);
        void recv_vector(HZVector &a, int source, int tag, int offset = 0, int size = -1);


        void recv_vector_async(DIVector &a, int source, int tag, int offset = 0, int size = -1);
        void recv_vector_async(HIVector &a, int source, int tag, int offset = 0, int size = -1);
        void recv_vector_async(DDVector &a, int source, int tag, int offset = 0, int size = -1);
        void recv_vector_async(HDVector &a, int source, int tag, int offset = 0, int size = -1);
        void recv_vector_async(DFVector &a, int source, int tag, int offset = 0, int size = -1);
        void recv_vector_async(HFVector &a, int source, int tag, int offset = 0, int size = -1);
        void recv_vector_async(DCVector &a, int source, int tag, int offset = 0, int size = -1);
        void recv_vector_async(HCVector &a, int source, int tag, int offset = 0, int size = -1);
        void recv_vector_async(DZVector &a, int source, int tag, int offset = 0, int size = -1);
        void recv_vector_async(HZVector &a, int source, int tag, int offset = 0, int size = -1);

        void recv_vector_wait_all(DIVector &a);
        void recv_vector_wait_all(HIVector &a);
        void recv_vector_wait_all(DDVector &a);
        void recv_vector_wait_all(HDVector &a);
        void recv_vector_wait_all(DFVector &a);
        void recv_vector_wait_all(HFVector &a);
        void recv_vector_wait_all(DCVector &a);
        void recv_vector_wait_all(HCVector &a);
        void recv_vector_wait_all(DZVector &a);
        void recv_vector_wait_all(HZVector &a);

        void send_raw_data(const void *ptr, int size, int destination, int tag);
        void recv_raw_data(void *ptr, int size, int source, int tag);

        int get_num_partitions();
        int get_global_id();

        void barrier();

        void get_hostname(std::string &my_hostname);
        void exchange_hostnames(std::string &my_hostname, std::vector<std::string> &hostnames, int num_parts );

        void all_gather(const int &my_data, HIVector &gathered_data, int num_parts);
        void all_gather(const int64_t &my_data, HI64Vector &gathered_data, int num_parts);

        void all_gather_v(HIVector &my_data, HIVector &gathered_data, int num_parts);
        void all_reduce_max(IndexType_h &my_data, IndexType_h &result_data);

        void all_gather_v(HDVector& data, int num_elems, HDVector& gathered_data, HIVector counts, HIVector displs);
        void all_gather_v(HFVector& data, int num_elems, HFVector& gathered_data, HIVector counts, HIVector displs);
        void all_gather_v(HCVector& data, int num_elems, HCVector& gathered_data, HIVector counts, HIVector displs);
        void all_gather_v(HZVector& data, int num_elems, HZVector& gathered_data, HIVector counts, HIVector displs);
        void all_gather_v(HIVector& data, int num_elems, HIVector& gathered_data, HIVector counts, HIVector displs);

#ifdef AMGX_WITH_MPI
        const MPI_Comm &get_mpi_comm() const {return mpi_comm;}
#endif

};

} // namespace amgx
