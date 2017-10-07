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
namespace amgx
{
template <class T_Config> class DistributedComms;
}

#include <getvalue.h>
#include <error.h>
#include <amg_config.h>
#include <map>
#include <string>
#include <distributed/distributed_manager.h>
#include <stacktrace.h>

namespace amgx
{

template <class T_Config> class Operator;
template <class T_Config> class Matrix;

template<class T_Config>
class DistributedComms
{
    public:
        typedef T_Config TConfig;
        typedef typename TConfig::template setMemSpace<AMGX_host>::Type TConfig_h;
        typedef typename TConfig::MatPrec ValueTypeA;
        typedef typename TConfig::VecPrec ValueTypeB;
        typedef typename TConfig::template setVecPrec<(AMGX_VecPrecision)AMGX_GET_MODE_VAL(AMGX_MatPrecision, TConfig::mode)>::Type mvec_value_type;
        typedef Vector<mvec_value_type> MVector;
        typedef std::vector<MVector> MVector_Array;

        typedef TemplateConfig<TConfig::memSpace, AMGX_vecDouble, TConfig::matPrec, TConfig::indPrec> dvec_value_type;
        typedef Vector<dvec_value_type> DVector;
        typedef std::vector<DVector> DVector_Array;

        typedef TemplateConfig<TConfig::memSpace, AMGX_vecFloat, TConfig::matPrec, TConfig::indPrec> fvec_value_type;
        typedef Vector<fvec_value_type> FVector;
        typedef std::vector<FVector> FVector_Array;

        typedef TemplateConfig<TConfig::memSpace, AMGX_vecComplex, TConfig::matPrec, TConfig::indPrec> cvec_value_type;
        typedef Vector<cvec_value_type> CVector;
        typedef std::vector<CVector> CVector_Array;

        typedef TemplateConfig<TConfig::memSpace, AMGX_vecDoubleComplex, TConfig::matPrec, TConfig::indPrec> zvec_value_type;
        typedef Vector<zvec_value_type> ZVector;
        typedef std::vector<ZVector> ZVector_Array;

        typedef TemplateConfig<TConfig::memSpace, AMGX_vecInt, TConfig::matPrec, TConfig::indPrec> ivec_value_type;
        typedef typename TConfig_h::template setVecPrec<AMGX_vecInt>::Type ivec_value_type_h;
        typedef Vector<ivec_value_type> IVector;
        typedef Vector<ivec_value_type_h> IVector_h;
        typedef std::vector<IVector> IVector_Array;

        typedef typename TConfig::template setVecPrec<AMGX_vecInt64>::Type i64vec_value_type;
        typedef Vector<i64vec_value_type> I64Vector;
        typedef std::vector<I64Vector> I64Vector_Array;

        typedef TemplateConfig<TConfig::memSpace, AMGX_vecBool, TConfig::matPrec, TConfig::indPrec> bvec_value_type;
        typedef Vector<bvec_value_type> BVector;
        typedef std::vector<BVector> BVector_Array;

        typedef typename TConfig_h::IndPrec IndexType_h;
        typedef TemplateConfig<AMGX_device, TConfig::vecPrec, TConfig::matPrec, TConfig::indPrec> TConfig_d;
        typedef Matrix<TConfig_h> Matrix_h;
        typedef Matrix<TConfig_d> Matrix_d;
        typedef Matrix<TConfig> Matrix_hd;
        typedef std::vector<Matrix_hd> Matrix_Array;

        typedef typename TConfig_h::template setVecPrec<(AMGX_VecPrecision)AMGX_GET_MODE_VAL(AMGX_MatPrecision, TConfig_h::mode)>::Type hmvec_value_type;
        typedef Vector<hmvec_value_type> HMVector;

        typedef typename TConfig_h::template setVecPrec<AMGX_vecDouble>::Type hdvec_value_type;
        typedef Vector<hdvec_value_type> HDVector;
        typedef std::vector<HDVector> HDVector_Array;

        typedef typename TConfig_h::template setVecPrec<AMGX_vecFloat>::Type hfvec_value_type;
        typedef Vector<hfvec_value_type> HFVector;
        typedef std::vector<HFVector> HFVector_Array;

        typedef typename TConfig_h::template setVecPrec<AMGX_vecComplex>::Type hcvec_value_type;
        typedef Vector<hcvec_value_type> HCVector;
        typedef std::vector<HCVector> HCVector_Array;

        typedef typename TConfig_h::template setVecPrec<AMGX_vecDoubleComplex>::Type hzvec_value_type;
        typedef Vector<hzvec_value_type> HZVector;
        typedef std::vector<HZVector> HZVector_Array;

        typedef typename TConfig_h::template setVecPrec<AMGX_vecInt>::Type hivec_value_type;
        typedef Vector<hivec_value_type> HIVector;
        typedef std::vector<HIVector> HIVector_Array;

        typedef typename TConfig_h::template setVecPrec<AMGX_vecInt64>::Type hi64vec_value_type;
        typedef Vector<hi64vec_value_type> HI64Vector;
        typedef std::vector<HI64Vector> HI64Vector_Array;

        typedef typename TConfig_d::template setVecPrec<AMGX_vecInt>::Type divec_value_type;
        typedef Vector<divec_value_type> DIVector;
        typedef typename TConfig_d::template setVecPrec<AMGX_vecFloat>::Type dfvec_value_type;
        typedef Vector<dfvec_value_type> DFVector;
        typedef typename TConfig_d::template setVecPrec<AMGX_vecDouble>::Type ddvec_value_type;
        typedef Vector<ddvec_value_type> DDVector;
        typedef typename TConfig_d::template setVecPrec<AMGX_vecComplex>::Type dcvec_value_type;
        typedef Vector<dcvec_value_type> DCVector;
        typedef typename TConfig_d::template setVecPrec<AMGX_vecDoubleComplex>::Type dzvec_value_type;
        typedef Vector<dzvec_value_type> DZVector;

        typedef typename TConfig_h::template setVecPrec<AMGX_vecBool>::Type hbvec_value_type;
        typedef Vector<hbvec_value_type> HBVector;
        typedef std::vector<HBVector> HBVector_Array;

        typedef DistributedManager<TConfig> DistributedManager_hd;
        typedef std::vector<DistributedManager_hd> DistributedManager_Array;

        DistributedComms(AMG_Config &cfg, const std::string &cfg_scope) : m_ref_count(1) {  };
        DistributedComms(): m_ref_count(1)  {};
        virtual void set_neighbors(int num_neighbors) = 0;

        virtual DistributedComms<TConfig> *Clone() const = 0;
        virtual DistributedComms<TConfig> *CloneSubComm(HIVector &coarse_part_to_fine_part, bool is_root_partition) = 0;

        virtual void createSubComm( HIVector &coarse_part_to_fine_part, bool is_root_partition ) = 0;

#ifdef AMGX_WITH_MPI
        virtual MPI_Comm get_mpi_comm() = 0;
        virtual void set_mpi_comm(MPI_Comm &new_comm) = 0;
#endif

        virtual void printString(const std::string &str) { };

        virtual void send_raw_data(const void *ptr, int size, int destination, int tag) = 0;
        virtual void recv_raw_data(void *ptr, int size, int source, int tag) = 0;

        virtual void exchange_matrix_halo(Matrix_Array &halo_rows, DistributedManager_Array &halo_btl, const Matrix<TConfig> &m) = 0;
        virtual void exchange_matrix_halo(IVector_Array &row_offsets, I64Vector_Array &col_indices, MVector_Array &values, I64Vector_Array &halo_row_ids, IVector_h &neighbors_list, int global_id) = 0;
        // external double vector
        virtual void setup(DVector &b, const Matrix<TConfig> &m, int tag, int num_rings = 1) = 0;
        virtual void setup_L2H(DVector &b, Matrix<TConfig> &m, int num_rings = 1) = 0;
        virtual void exchange_halo(DVector &b, const Matrix<TConfig> &m, cudaEvent_t event, int tag, int num_rings = 1) = 0;
        virtual void exchange_halo_async(DVector &b, const Matrix<TConfig> &m, cudaEvent_t event, int tag) = 0;
        virtual void send_receive_wait(DVector &b, const Matrix<TConfig> &m, cudaEvent_t event, int tag, cudaStream_t &stream) = 0;
        virtual void exchange_halo_wait(DVector &b, const Matrix<TConfig> &m, cudaEvent_t event, int tag) = 0;
        virtual bool exchange_halo_query(DVector &b, const Matrix<TConfig> &m, cudaEvent_t event) = 0;
        // external flout vector
        virtual void setup(FVector &b, const Matrix<TConfig> &m, int tag, int num_rings = 1) = 0;
        virtual void setup_L2H(FVector &b, Matrix<TConfig> &m, int num_rings = 1) = 0;
        virtual void exchange_halo(FVector &b, const Matrix<TConfig> &m, cudaEvent_t event, int tag, int num_rings = 1) = 0;
        virtual void exchange_halo_async(FVector &b, const Matrix<TConfig> &m, cudaEvent_t event, int tag) = 0;
        virtual void send_receive_wait(FVector &b, const Matrix<TConfig> &m, cudaEvent_t event, int tag, cudaStream_t &stream) = 0;
        virtual void exchange_halo_wait(FVector &b, const Matrix<TConfig> &m, cudaEvent_t event, int tag) = 0;
        virtual bool exchange_halo_query(FVector &b, const Matrix<TConfig> &m, cudaEvent_t event) = 0;
        // external complex vector
        virtual void setup(CVector &b, const Matrix<TConfig> &m, int tag, int num_rings = 1) = 0;
        virtual void setup_L2H(CVector &b, Matrix<TConfig> &m, int num_rings = 1) = 0;
        virtual void exchange_halo(CVector &b, const Matrix<TConfig> &m, cudaEvent_t event, int tag, int num_rings = 1) = 0;
        virtual void exchange_halo_async(CVector &b, const Matrix<TConfig> &m, cudaEvent_t event, int tag) = 0;
        virtual void send_receive_wait(CVector &b, const Matrix<TConfig> &m, cudaEvent_t event, int tag, cudaStream_t &stream) = 0;
        virtual void exchange_halo_wait(CVector &b, const Matrix<TConfig> &m, cudaEvent_t event, int tag) = 0;
        virtual bool exchange_halo_query(CVector &b, const Matrix<TConfig> &m, cudaEvent_t event) = 0;
        // external double complex vector
        virtual void setup(ZVector &b, const Matrix<TConfig> &m, int tag, int num_rings = 1) = 0;
        virtual void setup_L2H(ZVector &b, Matrix<TConfig> &m, int num_rings = 1) = 0;
        virtual void exchange_halo(ZVector &b, const Matrix<TConfig> &m, cudaEvent_t event, int tag, int num_rings = 1) = 0;
        virtual void exchange_halo_async(ZVector &b, const Matrix<TConfig> &m, cudaEvent_t event, int tag) = 0;
        virtual void send_receive_wait(ZVector &b, const Matrix<TConfig> &m, cudaEvent_t event, int tag, cudaStream_t &stream) = 0;
        virtual void exchange_halo_wait(ZVector &b, const Matrix<TConfig> &m, cudaEvent_t event, int tag) = 0;
        virtual bool exchange_halo_query(ZVector &b, const Matrix<TConfig> &m, cudaEvent_t event) = 0;
        // external int vector
        virtual void setup(IVector &b, const Matrix<TConfig> &m, int tag, int num_rings = 1) = 0;
        virtual void setup_L2H(IVector &b, Matrix<TConfig> &m, int num_rings = 1) = 0;
        virtual void exchange_halo(IVector &b, const Matrix<TConfig> &m, cudaEvent_t event, int tag, int num_rings = 1) = 0;
        virtual void exchange_halo_async(IVector &b, const Matrix<TConfig> &m, cudaEvent_t event, int tag) = 0;
        virtual void send_receive_wait(IVector &b, const Matrix<TConfig> &m, cudaEvent_t event, int tag, cudaStream_t &stream) = 0;
        virtual void exchange_halo_wait(IVector &b, const Matrix<TConfig> &m, cudaEvent_t event, int tag) = 0;
        virtual bool exchange_halo_query(IVector &b, const Matrix<TConfig> &m, cudaEvent_t event) = 0;
        // external boolean vector
        virtual void setup(BVector &b, const Matrix<TConfig> &m, int tag, int num_rings = 1) = 0;
        virtual void setup_L2H(BVector &b, Matrix<TConfig> &m, int num_rings = 1) = 0;
        virtual void exchange_halo(BVector &b, const Matrix<TConfig> &m, cudaEvent_t event, int tag, int num_rings = 1) = 0;
        virtual void exchange_halo_async(BVector &b, const Matrix<TConfig> &m, cudaEvent_t event, int tag) = 0;
        virtual void send_receive_wait(BVector &b, const Matrix<TConfig> &m, cudaEvent_t event, int tag, cudaStream_t &stream) = 0;
        virtual void exchange_halo_wait(BVector &b, const Matrix<TConfig> &m, cudaEvent_t event, int tag) = 0;
        virtual bool exchange_halo_query(BVector &b, const Matrix<TConfig> &m, cudaEvent_t event) = 0;
        // external i64 vector
        virtual void setup(I64Vector &b, const Matrix<TConfig> &m, int tag, int num_rings = 1) = 0;
        virtual void setup_L2H(I64Vector &b, Matrix<TConfig> &m, int num_rings = 1) = 0;
        virtual void exchange_halo(I64Vector &b, const Matrix<TConfig> &m, cudaEvent_t event, int tag, int num_rings = 1) = 0;
        virtual void exchange_halo_async(I64Vector &b, const Matrix<TConfig> &m, cudaEvent_t event, int tag) = 0;
        virtual void send_receive_wait(I64Vector &b, const Matrix<TConfig> &m, cudaEvent_t event, int tag, cudaStream_t &stream) = 0;
        virtual void exchange_halo_wait(I64Vector &b, const Matrix<TConfig> &m, cudaEvent_t event, int tag) = 0;
        virtual bool exchange_halo_query(I64Vector &b, const Matrix<TConfig> &m, cudaEvent_t event) = 0;

        virtual void add_from_halo(IVector &b, const Matrix<TConfig> &m, int tag, int num_rings, cudaStream_t &stream) = 0;
        virtual void add_from_halo(DVector &b, const Matrix<TConfig> &m, int tag, int num_rings, cudaStream_t &stream) = 0;
        virtual void add_from_halo(FVector &b, const Matrix<TConfig> &m, int tag, int num_rings, cudaStream_t &stream) = 0;
        virtual void add_from_halo(ZVector &b, const Matrix<TConfig> &m, int tag, int num_rings, cudaStream_t &stream) = 0;
        virtual void add_from_halo(CVector &b, const Matrix<TConfig> &m, int tag, int num_rings, cudaStream_t &stream) = 0;

        virtual void gather_L2H(IVector &b, const Matrix<TConfig> &m, int num_rings) = 0;
        virtual void gather_L2H(DVector &b, const Matrix<TConfig> &m, int num_rings) = 0;
        virtual void gather_L2H(FVector &b, const Matrix<TConfig> &m, int num_rings) = 0;
        virtual void gather_L2H(CVector &b, const Matrix<TConfig> &m, int num_rings) = 0;
        virtual void gather_L2H(ZVector &b, const Matrix<TConfig> &m, int num_rings) = 0;

        virtual void gather_L2H_v2(IVector &b, const Matrix<TConfig> &m, int num_rings) = 0;
        virtual void gather_L2H_v2(DVector &b, const Matrix<TConfig> &m, int num_rings) = 0;
        virtual void gather_L2H_v2(FVector &b, const Matrix<TConfig> &m, int num_rings) = 0;
        virtual void gather_L2H_v2(CVector &b, const Matrix<TConfig> &m, int num_rings) = 0;
        virtual void gather_L2H_v2(ZVector &b, const Matrix<TConfig> &m, int num_rings) = 0;

        virtual void global_reduce(HDVector_Array &a, HDVector &b, const Operator<TConfig> &m, int tag) = 0;
        virtual void global_reduce(HFVector_Array &a, HFVector &b, const Operator<TConfig> &m, int tag) = 0;
        virtual void global_reduce(HCVector_Array &a, HCVector &b, const Operator<TConfig> &m, int tag) = 0;
        virtual void global_reduce(HZVector_Array &a, HZVector &b, const Operator<TConfig> &m, int tag) = 0;
        virtual void global_reduce(HIVector_Array &a, HIVector &b, const Operator<TConfig> &m, int tag) = 0;
        virtual void global_reduce(HBVector_Array &a, HBVector &b, const Operator<TConfig> &m, int tag) = 0;
        virtual void global_reduce(HI64Vector_Array &a, HI64Vector &b, const Operator<TConfig> &m, int tag) = 0;

        virtual void global_reduce_sum(HDVector &a, HDVector &b, const Matrix<TConfig> &m, int tag) = 0;
        virtual void global_reduce_sum(HFVector &a, HFVector &b, const Matrix<TConfig> &m, int tag) = 0;
        virtual void global_reduce_sum(HCVector &a, HCVector &b, const Matrix<TConfig> &m, int tag) = 0;
        virtual void global_reduce_sum(HZVector &a, HZVector &b, const Matrix<TConfig> &m, int tag) = 0;
        virtual void global_reduce_sum(HIVector &a, HIVector &b, const Matrix<TConfig> &m, int tag) = 0;
        virtual void global_reduce_sum(HI64Vector &a, HI64Vector &b, const Matrix<TConfig> &m, int tag) = 0;

        virtual void exchange_vectors(DVector_Array &a, const Matrix<TConfig> &m, int tag) = 0;
        virtual void exchange_vectors(FVector_Array &a, const Matrix<TConfig> &m, int tag) = 0;
        virtual void exchange_vectors(CVector_Array &a, const Matrix<TConfig> &m, int tag) = 0;
        virtual void exchange_vectors(ZVector_Array &a, const Matrix<TConfig> &m, int tag) = 0;
        virtual void exchange_vectors(IVector_Array &a, const Matrix<TConfig> &m, int tag) = 0;
        virtual void exchange_vectors(BVector_Array &a, const Matrix<TConfig> &m, int tag) = 0;

        virtual void send_vector(DIVector &a, int destination, int tag, int offset = 0, int size = -1) = 0;
        virtual void send_vector(HIVector &a, int destination, int tag, int offset = 0, int size = -1) = 0;
        virtual void send_vector(DDVector &a, int destination, int tag, int offset = 0, int size = -1) = 0;
        virtual void send_vector(HDVector &a, int destination, int tag, int offset = 0, int size = -1) = 0;
        virtual void send_vector(DFVector &a, int destination, int tag, int offset = 0, int size = -1) = 0;
        virtual void send_vector(HFVector &a, int destination, int tag, int offset = 0, int size = -1) = 0;
        virtual void send_vector(DCVector &a, int destination, int tag, int offset = 0, int size = -1) = 0;
        virtual void send_vector(HCVector &a, int destination, int tag, int offset = 0, int size = -1) = 0;
        virtual void send_vector(DZVector &a, int destination, int tag, int offset = 0, int size = -1) = 0;
        virtual void send_vector(HZVector &a, int destination, int tag, int offset = 0, int size = -1) = 0;


        virtual void send_vector_async(DIVector &a, int destination, int tag, int offset = 0, int size = -1) = 0;
        virtual void send_vector_async(HIVector &a, int destination, int tag, int offset = 0, int size = -1) = 0;
        virtual void send_vector_async(DDVector &a, int destination, int tag, int offset = 0, int size = -1) = 0;
        virtual void send_vector_async(HDVector &a, int destination, int tag, int offset = 0, int size = -1) = 0;
        virtual void send_vector_async(DFVector &a, int destination, int tag, int offset = 0, int size = -1) = 0;
        virtual void send_vector_async(HFVector &a, int destination, int tag, int offset = 0, int size = -1) = 0;
        virtual void send_vector_async(DCVector &a, int destination, int tag, int offset = 0, int size = -1) = 0;
        virtual void send_vector_async(HCVector &a, int destination, int tag, int offset = 0, int size = -1) = 0;
        virtual void send_vector_async(DZVector &a, int destination, int tag, int offset = 0, int size = -1) = 0;
        virtual void send_vector_async(HZVector &a, int destination, int tag, int offset = 0, int size = -1) = 0;

        virtual void send_vector_wait_all(DIVector &a) = 0;
        virtual void send_vector_wait_all(HIVector &a) = 0;
        virtual void send_vector_wait_all(DDVector &a) = 0;
        virtual void send_vector_wait_all(HDVector &a) = 0;
        virtual void send_vector_wait_all(DFVector &a) = 0;
        virtual void send_vector_wait_all(HFVector &a) = 0;
        virtual void send_vector_wait_all(DCVector &a) = 0;
        virtual void send_vector_wait_all(HCVector &a) = 0;
        virtual void send_vector_wait_all(DZVector &a) = 0;
        virtual void send_vector_wait_all(HZVector &a) = 0;

        virtual void recv_vector(DIVector &a, int source, int tag, int offset = 0, int size = -1) = 0;
        virtual void recv_vector(HIVector &a, int source, int tag, int offset = 0, int size = -1) = 0;
        virtual void recv_vector(DDVector &a, int source, int tag, int offset = 0, int size = -1) = 0;
        virtual void recv_vector(HDVector &a, int source, int tag, int offset = 0, int size = -1) = 0;
        virtual void recv_vector(DFVector &a, int source, int tag, int offset = 0, int size = -1) = 0;
        virtual void recv_vector(HFVector &a, int source, int tag, int offset = 0, int size = -1) = 0;
        virtual void recv_vector(DCVector &a, int source, int tag, int offset = 0, int size = -1) = 0;
        virtual void recv_vector(HCVector &a, int source, int tag, int offset = 0, int size = -1) = 0;
        virtual void recv_vector(DZVector &a, int source, int tag, int offset = 0, int size = -1) = 0;
        virtual void recv_vector(HZVector &a, int source, int tag, int offset = 0, int size = -1) = 0;

        virtual void recv_vector_async(DIVector &a, int source, int tag, int offset = 0, int size = -1) = 0;
        virtual void recv_vector_async(HIVector &a, int source, int tag, int offset = 0, int size = -1) = 0;
        virtual void recv_vector_async(DDVector &a, int source, int tag, int offset = 0, int size = -1) = 0;
        virtual void recv_vector_async(HDVector &a, int source, int tag, int offset = 0, int size = -1) = 0;
        virtual void recv_vector_async(DFVector &a, int source, int tag, int offset = 0, int size = -1) = 0;
        virtual void recv_vector_async(HFVector &a, int source, int tag, int offset = 0, int size = -1) = 0;
        virtual void recv_vector_async(DCVector &a, int source, int tag, int offset = 0, int size = -1) = 0;
        virtual void recv_vector_async(HCVector &a, int source, int tag, int offset = 0, int size = -1) = 0;
        virtual void recv_vector_async(DZVector &a, int source, int tag, int offset = 0, int size = -1) = 0;
        virtual void recv_vector_async(HZVector &a, int source, int tag, int offset = 0, int size = -1) = 0;

        virtual void recv_vector_wait_all(DIVector &a) = 0;
        virtual void recv_vector_wait_all(HIVector &a) = 0;
        virtual void recv_vector_wait_all(DDVector &a) = 0;
        virtual void recv_vector_wait_all(HDVector &a) = 0;
        virtual void recv_vector_wait_all(DFVector &a) = 0;
        virtual void recv_vector_wait_all(HFVector &a) = 0;
        virtual void recv_vector_wait_all(DCVector &a) = 0;
        virtual void recv_vector_wait_all(HCVector &a) = 0;
        virtual void recv_vector_wait_all(DZVector &a) = 0;
        virtual void recv_vector_wait_all(HZVector &a) = 0;

        virtual int get_num_partitions() = 0;
        virtual int get_global_id() = 0;

        virtual void barrier() = 0;

        virtual void get_hostname(std::string &my_hostname) = 0;
        virtual void exchange_hostnames(std::string &my_hostname, std::vector<std::string> &hostnames, int num_parts ) = 0;

        virtual void all_gather(IndexType_h &my_data, HIVector &gathered_data, int num_parts) = 0;
        virtual void all_gather_v(HIVector &my_data, HIVector &gathered_data, int num_parts) = 0;

        virtual void all_reduce_max(IndexType_h &my_data, IndexType_h &result_data) = 0;



        // Increment the reference counter.
        void incr_ref_count() { ++m_ref_count;}

        // Decrement the reference counter.
        bool decr_ref_count() { return --m_ref_count == 0; }

        virtual ~DistributedComms();

        ColoringType halo_coloring;

        int m_ref_count;

};

class DistributedCommsFactory
{
    protected:
        DistributedComms< TemplateConfig<AMGX_host, AMGX_vecDouble, AMGX_matDouble, AMGX_indInt> > *m_hddi;
        DistributedComms< TemplateConfig<AMGX_host, AMGX_vecDouble, AMGX_matFloat, AMGX_indInt> > *m_hdfi;
        DistributedComms< TemplateConfig<AMGX_host, AMGX_vecFloat, AMGX_matFloat, AMGX_indInt> > *m_hffi;
        DistributedComms< TemplateConfig<AMGX_host, AMGX_vecInt, AMGX_matDouble, AMGX_indInt> > *m_hidi;
        DistributedComms< TemplateConfig<AMGX_host, AMGX_vecInt, AMGX_matFloat, AMGX_indInt> > *m_hifi;

        DistributedComms< TemplateConfig<AMGX_host, AMGX_vecDoubleComplex, AMGX_matDoubleComplex, AMGX_indInt> > *m_hzzi;
        DistributedComms< TemplateConfig<AMGX_host, AMGX_vecDoubleComplex, AMGX_matComplex, AMGX_indInt> > *m_hzci;
        DistributedComms< TemplateConfig<AMGX_host, AMGX_vecComplex, AMGX_matComplex, AMGX_indInt> > *m_hcci;
        DistributedComms< TemplateConfig<AMGX_host, AMGX_vecInt, AMGX_matComplex, AMGX_indInt> > *m_hizi;
        DistributedComms< TemplateConfig<AMGX_host, AMGX_vecInt, AMGX_matDoubleComplex, AMGX_indInt> > *m_hici;

        DistributedComms< TemplateConfig<AMGX_device, AMGX_vecDouble, AMGX_matDouble, AMGX_indInt> > *m_dddi;
        DistributedComms< TemplateConfig<AMGX_device, AMGX_vecDouble, AMGX_matFloat, AMGX_indInt> > *m_ddfi;
        DistributedComms< TemplateConfig<AMGX_device, AMGX_vecFloat, AMGX_matFloat, AMGX_indInt> > *m_dffi;
        DistributedComms< TemplateConfig<AMGX_device, AMGX_vecInt, AMGX_matDouble, AMGX_indInt> > *m_didi;
        DistributedComms< TemplateConfig<AMGX_device, AMGX_vecInt, AMGX_matFloat, AMGX_indInt> > *m_difi;

        DistributedComms< TemplateConfig<AMGX_device, AMGX_vecDoubleComplex, AMGX_matDoubleComplex, AMGX_indInt> > *m_dzzi;
        DistributedComms< TemplateConfig<AMGX_device, AMGX_vecDoubleComplex, AMGX_matComplex, AMGX_indInt> > *m_dzci;
        DistributedComms< TemplateConfig<AMGX_device, AMGX_vecComplex, AMGX_matComplex, AMGX_indInt> > *m_dcci;
        DistributedComms< TemplateConfig<AMGX_device, AMGX_vecInt, AMGX_matComplex, AMGX_indInt> > *m_dizi;
        DistributedComms< TemplateConfig<AMGX_device, AMGX_vecInt, AMGX_matDoubleComplex, AMGX_indInt> > *m_dici;

    public:
        DistributedCommsFactory() : m_hddi(NULL), m_hdfi(NULL), m_hffi(NULL), m_hidi(NULL), m_hifi(NULL), m_dddi(NULL), m_ddfi(NULL), m_dffi(NULL), m_didi(NULL), m_difi(NULL),
            m_hzzi(NULL), m_hzci(NULL), m_hcci(NULL), m_hici(NULL), m_hizi(NULL), m_dzzi(NULL), m_dzci(NULL), m_dcci(NULL), m_dizi(NULL), m_dici(NULL) {};

        virtual DistributedComms< TemplateConfig<AMGX_host, AMGX_vecDouble, AMGX_matDouble, AMGX_indInt> > *getInstanceHDDI() = 0;
        virtual DistributedComms< TemplateConfig<AMGX_host, AMGX_vecDouble, AMGX_matFloat, AMGX_indInt> > *getInstanceHDFI() = 0;
        virtual DistributedComms< TemplateConfig<AMGX_host, AMGX_vecFloat, AMGX_matFloat, AMGX_indInt> > *getInstanceHFFI() = 0;
        virtual DistributedComms< TemplateConfig<AMGX_host, AMGX_vecInt, AMGX_matDouble, AMGX_indInt> > *getInstanceHIDI() = 0;
        virtual DistributedComms< TemplateConfig<AMGX_host, AMGX_vecInt, AMGX_matFloat, AMGX_indInt> > *getInstanceHIFI() = 0;

        virtual DistributedComms< TemplateConfig<AMGX_host, AMGX_vecDoubleComplex, AMGX_matDoubleComplex, AMGX_indInt> > *getInstanceHZZI() = 0;
        virtual DistributedComms< TemplateConfig<AMGX_host, AMGX_vecDoubleComplex, AMGX_matComplex, AMGX_indInt> > *getInstanceHZCI() = 0;
        virtual DistributedComms< TemplateConfig<AMGX_host, AMGX_vecComplex, AMGX_matComplex, AMGX_indInt> > *getInstanceHCCI() = 0;
        virtual DistributedComms< TemplateConfig<AMGX_host, AMGX_vecInt, AMGX_matDoubleComplex, AMGX_indInt> > *getInstanceHIZI() = 0;
        virtual DistributedComms< TemplateConfig<AMGX_host, AMGX_vecInt, AMGX_matComplex, AMGX_indInt> > *getInstanceHICI() = 0;

        virtual DistributedComms< TemplateConfig<AMGX_device, AMGX_vecDouble, AMGX_matDoubleComplex, AMGX_indInt> > *getInstanceDDDI() = 0;
        virtual DistributedComms< TemplateConfig<AMGX_device, AMGX_vecDouble, AMGX_matComplex, AMGX_indInt> > *getInstanceDDFI() = 0;
        virtual DistributedComms< TemplateConfig<AMGX_device, AMGX_vecFloat, AMGX_matComplex, AMGX_indInt> > *getInstanceDFFI() = 0;
        virtual DistributedComms< TemplateConfig<AMGX_device, AMGX_vecInt, AMGX_matDoubleComplex, AMGX_indInt> > *getInstanceDIDI() = 0;
        virtual DistributedComms< TemplateConfig<AMGX_device, AMGX_vecInt, AMGX_matComplex, AMGX_indInt> > *getInstanceDIFI() = 0;

        virtual DistributedComms< TemplateConfig<AMGX_device, AMGX_vecDoubleComplex, AMGX_matDoubleComplex, AMGX_indInt> > *getInstanceDZZI() = 0;
        virtual DistributedComms< TemplateConfig<AMGX_device, AMGX_vecDoubleComplex, AMGX_matFloat, AMGX_indInt> > *getInstanceDZCI() = 0;
        virtual DistributedComms< TemplateConfig<AMGX_device, AMGX_vecComplex, AMGX_matFloat, AMGX_indInt> > *getInstanceDCCI() = 0;
        virtual DistributedComms< TemplateConfig<AMGX_device, AMGX_vecInt, AMGX_matDoubleComplex, AMGX_indInt> > *getInstanceDIZI() = 0;
        virtual DistributedComms< TemplateConfig<AMGX_device, AMGX_vecInt, AMGX_matComplex, AMGX_indInt> > *getInstanceDICI() = 0;
};

} // namespace amgx
