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
#include <vector.h>
#include <array>
#include <memory>
#include <stdexcept>
#include <sstream>//

namespace amgx
{

template<typename T, typename VecType, typename VecHost>
struct CommVisitor: VisitorBase<T>
{
        CommVisitor(void)
        {
        }

        CommVisitor(std::vector<VecType> &rhs,
                    std::vector<VecHost> &host_vec,
                    const std::vector<int> &sizes,
                    const Matrix<T> &m):
            m_ptr_rhs(&rhs),
            m_ptr_host(&host_vec),
            m_ptr_sizes(&sizes),
            m_ptr_m(&m)
        {
        }

        void VisitComms(CommsMPI<T> & )
        {
            //purposely empty...
        }

        void set_rhs(std::vector<VecType> &rhs)
        {
            m_ptr_rhs = &rhs;
        }

        void set_host_vec(std::vector<VecHost> &host_vec)
        {
            m_ptr_host = &host_vec;
        }

        void set_sizes(const std::vector<int> &sizes)
        {
            m_ptr_sizes = &sizes;
        }

        void set_matrix(const Matrix<T> &m)
        {
            m_ptr_m = &m;
        }

    protected:
        std::vector<VecType> &get_rhs(void)
        {
            if ( m_ptr_rhs )
            {
                return *m_ptr_rhs;
            }
            else
            {
                std::stringstream ss;
                ss << "ERROR: Attempt to dereference m_ptr_rhs nullptr."
                   << __FILE__
                   << "; "
                   << __LINE__;
                throw std::runtime_error(ss.str());
            }
        }

        std::vector<VecHost> &get_host_vec(void)
        {
            if ( m_ptr_host )
            {
                return *m_ptr_host;
            }
            else
            {
                std::stringstream ss;
                ss << "ERROR: Attempt to dereference m_ptr_host nullptr."
                   << __FILE__
                   << "; "
                   << __LINE__;
                throw std::runtime_error(ss.str());
            }
        }

        const std::vector<int> &get_sizes(void) const
        {
            if ( m_ptr_sizes )
            {
                return *m_ptr_sizes;
            }
            else
            {
                std::stringstream ss;
                ss << "ERROR: Attempt to dereference m_ptr_sizes nullptr."
                   << __FILE__
                   << "; "
                   << __LINE__;
                throw std::runtime_error(ss.str());
            }
        }

        const Matrix<T> &get_matrix(void) const
        {
            if ( m_ptr_m )
            {
                return *m_ptr_m;
            }
            else
            {
                std::stringstream ss;
                ss << "ERROR: Attempt to dereference m_ptr_m nullptr."
                   << __FILE__
                   << "; "
                   << __LINE__;
                throw std::runtime_error(ss.str());
            }
        }
    private:
        //Pointers to avoid copying and
        //so that they can be set later
        //than construction (anti-RAII)
        //
        //(PROBLEM: raw pointers?!)
        //
        std::vector<VecType> *m_ptr_rhs;
        std::vector<VecHost> *m_ptr_host;
        const std::vector<int> *m_ptr_sizes;
        const Matrix<T> *m_ptr_m;
};

//Receiver:
//
template<typename T, typename VecType, typename VecHost>
struct ReceiverVisitor: CommVisitor<T, VecType, VecHost>
{
        using CommVisitor<T, VecType, VecHost>::get_rhs;
        using CommVisitor<T, VecType, VecHost>::get_host_vec;
        using CommVisitor<T, VecType, VecHost>::get_sizes;
        using CommVisitor<T, VecType, VecHost>::get_matrix;

        using CommVisitor<T, VecType, VecHost>::set_rhs;
        using CommVisitor<T, VecType, VecHost>::set_host_vec;
        using CommVisitor<T, VecType, VecHost>::set_sizes;
        using CommVisitor<T, VecType, VecHost>::set_matrix;

        ReceiverVisitor(void) {}

        ReceiverVisitor(std::vector<VecType> &rhs,
                        std::vector<VecHost> &host_vec,
                        const std::vector<int> &sizes,
                        const Matrix<T> &m):
            CommVisitor<T, VecType, VecHost>(rhs, host_vec, sizes, m)
        {
        }

        void VisitCommsHostbuffer(CommsMPIHostBufferStream<T> &comm);
        void VisitCommsMPIDirect(CommsMPIDirect<T> &comm);

        std::vector<VecType> &get_local(void)
        {
            return m_local_copy;
        }
    private:
        std::vector<VecType> m_local_copy;//CUDA-aware version needs it
};

//Sender:
//
template<typename T, typename VecType, typename VecHost>
struct SenderVisitor: CommVisitor<T, VecType, VecHost>
{
    using CommVisitor<T, VecType, VecHost>::get_rhs;
    using CommVisitor<T, VecType, VecHost>::get_host_vec;
    using CommVisitor<T, VecType, VecHost>::get_sizes;
    using CommVisitor<T, VecType, VecHost>::get_matrix;

    using CommVisitor<T, VecType, VecHost>::set_rhs;
    using CommVisitor<T, VecType, VecHost>::set_host_vec;
    using CommVisitor<T, VecType, VecHost>::set_sizes;
    using CommVisitor<T, VecType, VecHost>::set_matrix;

    SenderVisitor(void) {}

    SenderVisitor(std::vector<VecType> &rhs,
                  std::vector<VecHost> &host_vec,
                  const std::vector<int> &sizes,
                  const Matrix<T> &m):
        CommVisitor<T, VecType, VecHost>(rhs, host_vec, sizes, m)
    {
    }

    void VisitCommsHostbuffer(CommsMPIHostBufferStream<T> &comm);
    void VisitCommsMPIDirect(CommsMPIDirect<T> &comm);
};

enum class Direction {None, HostToDevice, DeviceToHost};

//Sync (copying)
//between host and device buffers
//
template<typename T, typename VecType, typename VecHost>
struct SynchronizerVisitor: CommVisitor<T, VecType, VecHost>
{
        using CommVisitor<T, VecType, VecHost>::get_rhs;
        using CommVisitor<T, VecType, VecHost>::get_host_vec;
        using CommVisitor<T, VecType, VecHost>::get_sizes;
        using CommVisitor<T, VecType, VecHost>::get_matrix;

        using CommVisitor<T, VecType, VecHost>::set_rhs;
        using CommVisitor<T, VecType, VecHost>::set_host_vec;
        using CommVisitor<T, VecType, VecHost>::set_sizes;
        using CommVisitor<T, VecType, VecHost>::set_matrix;

        SynchronizerVisitor(void): m_dir(Direction::None)
        {
        }

        explicit SynchronizerVisitor(Direction dir): m_dir(dir)
        {
        }

        SynchronizerVisitor(Direction dir,
                            std::vector<VecType> &rhs,
                            std::vector<VecHost> &host_vec,
                            const std::vector<int> &sizes,
                            const Matrix<T> &m):
            CommVisitor<T, VecType, VecHost>(rhs, host_vec, sizes, m),
            m_dir(dir)
        {
        }

        virtual void switch_direction(void)
        {
            if ( m_dir == Direction::DeviceToHost )
            {
                m_dir = Direction::HostToDevice;
            }
            else if ( m_dir == Direction::HostToDevice )
            {
                m_dir = Direction::DeviceToHost;
            }

            //else nothing;
        }

        void VisitCommsHostbuffer(CommsMPIHostBufferStream<T> &comm);
        void VisitCommsMPIDirect(CommsMPIDirect<T> &comm);

        void set_local(std::vector<VecType> &local_copy)
        {
            m_ptr_local = &local_copy;
        }

        std::vector<VecType> &get_local(void)
        {
            if ( m_ptr_local )
            {
                return *m_ptr_local;
            }
            else
            {
                std::stringstream ss;
                ss << "ERROR: Attempt to dereference m_ptr_local nullptr."
                   << __FILE__
                   << "; "
                   << __LINE__;
                throw std::runtime_error(ss.str());
            }
        }

    private:
        Direction m_dir;
        std::vector<VecType> *m_ptr_local;//CUDA-aware version may need it
};

template<typename TConfig, typename Tb>
struct BaseVecVisitor: VisitorBase<TConfig>
{
        BaseVecVisitor(int addr_rank,
                       int tag,
                       int offset,
                       int size):
            m_rank(addr_rank), m_tag(tag), m_offset(offset), m_sz(size)
        {
        }

        BaseVecVisitor(Tb &b,
                       int addr_rank,
                       int tag,
                       int offset,
                       int size):
            m_ptr_bVec(&b), m_rank(addr_rank), m_tag(tag), m_offset(offset), m_sz(size)
        {
        }

        void VisitComms(CommsMPI<TConfig> & )
        {
            //purposely empty...
        }

        void set_vec(Tb &b)
        {
            m_ptr_bVec = &b;
        }

        Tb &get_vec(void)
        {
            return *m_ptr_bVec;
        }

        const Tb &get_vec(void) const
        {
            return *m_ptr_bVec;
        }

        int get_rank(void) const
        {
            return m_rank;
        }

        int get_tag(void) const
        {
            return m_tag;
        }

        int get_offset(void) const
        {
            return m_offset;
        }

        int get_size(void) const
        {
            return m_sz;
        }

    private:
        Tb *m_ptr_bVec;
        int m_rank;
        int m_tag;
        int m_offset;
        int m_sz;
};

template<typename TConfig, typename Tb>
struct SynchSendVecVisitor: BaseVecVisitor<TConfig, Tb>
{
    using BaseVecVisitor<TConfig, Tb>::get_vec;
    using BaseVecVisitor<TConfig, Tb>::get_offset;
    using BaseVecVisitor<TConfig, Tb>::get_size;
    using BaseVecVisitor<TConfig, Tb>::get_tag;
    using BaseVecVisitor<TConfig, Tb>::get_rank;

    SynchSendVecVisitor(int addr_rank,
                        int tag,
                        int offset,
                        int size):
        BaseVecVisitor<TConfig, Tb>(addr_rank, tag, offset, size)
    {
    }

    SynchSendVecVisitor(Tb &b,
                        int addr_rank,
                        int tag,
                        int offset,
                        int size):
        BaseVecVisitor<TConfig, Tb>(b, addr_rank, tag, offset, size)
    {
    }

    void VisitCommsHostbuffer(CommsMPIHostBufferStream<TConfig> &comm);
    void VisitCommsMPIDirect(CommsMPIDirect<TConfig> &comm);
};

template<typename TConfig, typename Tb>
struct AsynchSendVecVisitor: BaseVecVisitor<TConfig, Tb>
{
    using BaseVecVisitor<TConfig, Tb>::get_vec;
    using BaseVecVisitor<TConfig, Tb>::get_offset;
    using BaseVecVisitor<TConfig, Tb>::get_size;
    using BaseVecVisitor<TConfig, Tb>::get_tag;
    using BaseVecVisitor<TConfig, Tb>::get_rank;

    AsynchSendVecVisitor(int addr_rank,
                         int tag,
                         int offset,
                         int size):
        BaseVecVisitor<TConfig, Tb>(addr_rank, tag, offset, size)
    {
    }

    AsynchSendVecVisitor(Tb &b,
                         int addr_rank,
                         int tag,
                         int offset,
                         int size):
        BaseVecVisitor<TConfig, Tb>(b, addr_rank, tag, offset, size)
    {
    }

    void VisitCommsHostbuffer(CommsMPIHostBufferStream<TConfig> &comm);
    void VisitCommsMPIDirect(CommsMPIDirect<TConfig> &comm);
};

template<typename TConfig, typename Tb>
struct SynchRecvVecVisitor: BaseVecVisitor<TConfig, Tb>
{
    using BaseVecVisitor<TConfig, Tb>::get_vec;
    using BaseVecVisitor<TConfig, Tb>::get_offset;
    using BaseVecVisitor<TConfig, Tb>::get_size;
    using BaseVecVisitor<TConfig, Tb>::get_tag;
    using BaseVecVisitor<TConfig, Tb>::get_rank;

    SynchRecvVecVisitor(int addr_rank,
                        int tag,
                        int offset,
                        int size):
        BaseVecVisitor<TConfig, Tb>(addr_rank, tag, offset, size)
    {
    }

    SynchRecvVecVisitor(Tb &b,
                        int addr_rank,
                        int tag,
                        int offset,
                        int size):
        BaseVecVisitor<TConfig, Tb>(b, addr_rank, tag, offset, size)
    {
    }

    void VisitCommsHostbuffer(CommsMPIHostBufferStream<TConfig> &comm);
    void VisitCommsMPIDirect(CommsMPIDirect<TConfig> &comm);
};

template<typename TConfig, typename Tb>
struct AsynchRecvVecVisitor: BaseVecVisitor<TConfig, Tb>
{
    using BaseVecVisitor<TConfig, Tb>::get_vec;
    using BaseVecVisitor<TConfig, Tb>::get_offset;
    using BaseVecVisitor<TConfig, Tb>::get_size;
    using BaseVecVisitor<TConfig, Tb>::get_tag;
    using BaseVecVisitor<TConfig, Tb>::get_rank;

    AsynchRecvVecVisitor(int addr_rank,
                         int tag,
                         int offset,
                         int size):
        BaseVecVisitor<TConfig, Tb>(addr_rank, tag, offset, size)
    {
    }

    AsynchRecvVecVisitor(Tb &b,
                         int addr_rank,
                         int tag,
                         int offset,
                         int size):
        BaseVecVisitor<TConfig, Tb>(b, addr_rank, tag, offset, size)
    {
    }

    void VisitCommsHostbuffer(CommsMPIHostBufferStream<TConfig> &comm);
    void VisitCommsMPIDirect(CommsMPIDirect<TConfig> &comm);
};

//Functor to be trampolined by a Visitor
//to do something easy (a handfull of operations)
//
template<typename TConfig>
struct BaseFunctor
{
    virtual ~BaseFunctor(void)
    {
    }

    virtual void operator()(CommsMPIHostBufferStream<TConfig> &comm) = 0;
    virtual void operator()(CommsMPIDirect<TConfig> &comm) = 0;
};

template<typename TConfig, typename Tb>
struct CopyHostFunctor: BaseFunctor<TConfig>
{
        explicit CopyHostFunctor(Tb &b): m_b(b)
        {
        }

        void operator()(CommsMPIHostBufferStream<TConfig> &comm)
        {
            m_b = *(m_b.host_send_recv_buffer);
        }

        void operator()(CommsMPIDirect<TConfig> &comm)
        {
            //purposely empty...
        }
    private:
        Tb &m_b;
};

template<typename TConfig, typename Tb>
struct HalloWaitCopyFunctor: BaseFunctor<TConfig>
{
        HalloWaitCopyFunctor(Tb &b, const Matrix<TConfig> &m, cudaStream_t stream):
            m_b(b),
            m_m(m),
            m_stream(stream)
        {
        }

        void operator()(CommsMPIHostBufferStream<TConfig> &comm)
        {
            int bsize = m_b.get_block_size();
            int size = (m_m.manager->halo_offsets[comm.get_neighbors()] - m_m.manager->halo_offsets[0]) * bsize;

            if (size != 0)
            {
                cudaMemcpyAsync(m_b.raw() + m_m.manager->halo_offsets[0]*bsize,
                           &(m_b.explicit_host_buffer[m_b.buffer_size]),
                           size * sizeof(typename Tb::value_type),
                           cudaMemcpyHostToDevice, m_stream);
                cudaStreamSynchronize(m_stream);
            }
        }

        void operator()(CommsMPIDirect<TConfig> &comm)
        {
            //purposely empty...
        }
    private:
        Tb &m_b;
        const Matrix<TConfig> &m_m;
        cudaStream_t m_stream;
};

//multi-purpose trampoline visitor
//for lightweight operations,
//which are delegated to the Functor
//who, in turn, is independent of the Visitor chain of inheritance,
//hence, allowing the Visitor to depend on less template arguments
//and providing a decoupling between Visitor's abstraction and its
//implementation, a la pImpl idiom
//
template<typename TConfig>
struct LightVisitor: VisitorBase<TConfig>
{
        explicit LightVisitor(BaseFunctor<TConfig> &func): m_func(func)
        {
        }
        void VisitComms(CommsMPI<TConfig> & )
        {
            //purposely empty...
        }
        void VisitCommsHostbuffer(CommsMPIHostBufferStream<TConfig> &comm)
        {
            m_func(comm);
        }

        void VisitCommsMPIDirect(CommsMPIDirect<TConfig> &comm)
        {
            m_func(comm);
        }
    private:
        BaseFunctor<TConfig> &m_func;
};

//Finite State Machine (FSM) trampoline visitor
//for finite state step-by-step visits,
//which are delegated to a vector of Functors
//decoupling the Visitor's abstraction from its
//implementation and enabling the FSM;
//the FSM is linear, no branches or loops
//(not a tree, not a graph,
// no need for transition table)
//
template<typename TConfig>
struct FSMVisitor: VisitorBase<TConfig>
{
        FSMVisitor(void): m_func_index(0)
        {
        }
        void VisitComms(CommsMPI<TConfig> & )
        {
            //purposely empty...
        }
        void VisitCommsHostbuffer(CommsMPIHostBufferStream<TConfig> &comm)
        {
            (*(m_v_func.at(m_func_index)))(comm); //why at()? because it throws on trespassing vector bounds.
        }

        void VisitCommsMPIDirect(CommsMPIDirect<TConfig> &comm)
        {
            (*(m_v_func.at(m_func_index)))(comm); //why at()? because it throws on trespassing vector bounds.
        }

        std::vector<BaseFunctor<TConfig>*> &get_functors(void)
        {
            return m_v_func;
        }

        const std::vector<BaseFunctor<TConfig>*> &get_functors(void) const
        {
            return m_v_func;
        }

        //advance the FSM:
        //
        void next(void)
        {
            ++m_func_index;
        }

        size_t get_state(void) const
        {
            return m_func_index;
        }

    private:
        size_t m_func_index;
        std::vector<BaseFunctor<TConfig>*> m_v_func;
};

//FSM step 1 in do_exchange_halo and do_exchange_halo_async
//(instantiate like SynchSendVecVisitor)
//
template<typename TConfig, typename Tb>
struct ExcHalo1Functor: BaseFunctor<TConfig>
{
        ExcHalo1Functor(void):
            m_ptr_b(nullptr),
            m_ptr_m(nullptr),
            m_num_rings(0),
            m_offset(0),
            m_tag(-1),
            m_stream(NULL)
        {
        }

        ExcHalo1Functor(Tb &b,
                        const Matrix<TConfig> &m,
                        int num_rings,
                        int offset,
                        cudaStream_t stream = NULL):
            m_ptr_b(&b),
            m_ptr_m(&m),
            m_num_rings(num_rings),
            m_offset(offset),
            m_tag(m.manager->global_id()),
            m_stream(stream)
        {
        }

        ExcHalo1Functor(Tb &b,
                        const Matrix<TConfig> &m,
                        int num_rings,
                        int offset,
                        int tag,
                        cudaStream_t stream = NULL):
            m_ptr_b(&b),
            m_ptr_m(&m),
            m_num_rings(num_rings),
            m_offset(offset),
            m_tag(tag),
            m_stream(stream)
        {
        }

        Tb &get_b(void)
        {
            if ( m_ptr_b )
            {
                return *m_ptr_b;
            }
            else
            {
                std::stringstream ss;
                ss << "ERROR: Attempt to dereference m_ptr_b nullptr."
                   << __FILE__
                   << "; "
                   << __LINE__;
                throw std::runtime_error(ss.str());
            }
        }

        cudaStream_t &get_stream(void)
        {
            return m_stream;
        }

        const Matrix<TConfig> &get_m(void) const
        {
            if ( m_ptr_m )
            {
                return *m_ptr_m;
            }
            else
            {
                std::stringstream ss;
                ss << "ERROR: Attempt to dereference m_ptr_m nullptr."
                   << __FILE__
                   << "; "
                   << __LINE__;
                throw std::runtime_error(ss.str());
            }
        }

        int get_num_rings(void) const
        {
            return m_num_rings;
        }

        int &get_offset(void)
        {
            return m_offset;
        }

        int get_tag(void) const
        {
            return m_tag;
        }

        void operator()(CommsMPIHostBufferStream<TConfig> &comm);

        void operator()(CommsMPIDirect<TConfig> &comm);

    private:
        Tb *m_ptr_b;
        cudaStream_t m_stream;
        const Matrix<TConfig> *m_ptr_m;
        int m_num_rings;
        int m_offset;
        int m_tag;
};

// FSMVisitor Functors:
//
// CAVEAT:
// These functors need separate instantiation machinery
// excluding t_vecPrec = bool,
// which generates "vector<bool> host_buffer;" in Vector,
// adress of whose elements
// cannot be taken,
// because vector<bool>::operator[](int ) returns a proxy,
// hence a temporary, and taking its address is, at best,
// "running with scissors")
//
//Or,
//
//as an alternative, (partial) template specialization may be used
//to avoid t_vecPrec = bool

//FSM step 2 in do_exchange_halo
//(instantiate like SynchSendVecVisitor)
//
template<typename TConfig, typename Tb>
struct ExcHalo2Functor: ExcHalo1Functor<TConfig, Tb>
{
    using ExcHalo1Functor<TConfig, Tb>::get_b;
    using ExcHalo1Functor<TConfig, Tb>::get_m;
    using ExcHalo1Functor<TConfig, Tb>::get_num_rings;
    using ExcHalo1Functor<TConfig, Tb>::get_offset;

    ExcHalo2Functor(void)
    {
    }

    ExcHalo2Functor(Tb &b,
                    const Matrix<TConfig> &m,
                    int num_rings,
                    int offset):
        ExcHalo1Functor<TConfig, Tb>(b, m, num_rings, offset)
    {
    }

    void operator()(CommsMPIHostBufferStream<TConfig> &comm);

    void operator()(CommsMPIDirect<TConfig> &comm);
};

//FSM step 3 in do_exchange_halo
//(instantiate like SynchSendVecVisitor)
//
template<typename TConfig, typename Tb>
struct ExcHalo3Functor: ExcHalo1Functor<TConfig, Tb>
{
    using ExcHalo1Functor<TConfig, Tb>::get_b;
    using ExcHalo1Functor<TConfig, Tb>::get_m;
    using ExcHalo1Functor<TConfig, Tb>::get_num_rings;
    using ExcHalo1Functor<TConfig, Tb>::get_offset;
    using ExcHalo1Functor<TConfig, Tb>::get_stream;

    ExcHalo3Functor(void)
    {
    }

    ExcHalo3Functor(Tb &b,
                    const Matrix<TConfig> &m,
                    int num_rings,
                    int offset,
                    cudaStream_t stream = NULL):
        ExcHalo1Functor<TConfig, Tb>(b, m, num_rings, offset, stream)
    {
    }

    void operator()(CommsMPIHostBufferStream<TConfig> &comm);

    void operator()(CommsMPIDirect<TConfig> &comm);
};

//FSM step 2 in do_exchange_halo_async
//(instantiate like SynchSendVecVisitor)
//
template<typename TConfig, typename Tb>
struct ExcHalo2AsyncFunctor: ExcHalo1Functor<TConfig, Tb>
{
    using ExcHalo1Functor<TConfig, Tb>::get_b;
    using ExcHalo1Functor<TConfig, Tb>::get_m;
    using ExcHalo1Functor<TConfig, Tb>::get_num_rings;
    using ExcHalo1Functor<TConfig, Tb>::get_offset;
    using ExcHalo1Functor<TConfig, Tb>::get_tag;

    ExcHalo2AsyncFunctor(void)
    {
    }

    ExcHalo2AsyncFunctor(Tb &b,
                         const Matrix<TConfig> &m,
                         int num_rings,
                         int offset):
        ExcHalo1Functor<TConfig, Tb>(b, m, num_rings, offset)
    {
    }

    ExcHalo2AsyncFunctor(Tb &b,
                         const Matrix<TConfig> &m,
                         int num_rings,
                         int offset,
                         int tag):
        ExcHalo1Functor<TConfig, Tb>(b, m, num_rings, offset, tag)
    {
    }

    void operator()(CommsMPIHostBufferStream<TConfig> &comm);

    void operator()(CommsMPIDirect<TConfig> &comm);
};


//FSM step 3 in do_exchange_halo_async
//(instantiate like SynchSendVecVisitor)
//
template<typename TConfig, typename Tb>
struct ExcHalo3AsyncFunctor: ExcHalo1Functor<TConfig, Tb>
{
    using ExcHalo1Functor<TConfig, Tb>::get_b;
    using ExcHalo1Functor<TConfig, Tb>::get_m;
    using ExcHalo1Functor<TConfig, Tb>::get_num_rings;
    using ExcHalo1Functor<TConfig, Tb>::get_offset;
    using ExcHalo1Functor<TConfig, Tb>::get_tag;
    using ExcHalo1Functor<TConfig, Tb>::get_stream;

    ExcHalo3AsyncFunctor(void)
    {
    }

    ExcHalo3AsyncFunctor(Tb &b,
                         const Matrix<TConfig> &m,
                         int num_rings,
                         int offset,
                         cudaStream_t stream):
        ExcHalo1Functor<TConfig, Tb>(b, m, num_rings, offset, stream)
    {
    }

    ExcHalo3AsyncFunctor(Tb &b,
                         const Matrix<TConfig> &m,
                         int num_rings,
                         int offset,
                         int tag,
                         cudaStream_t stream):
        ExcHalo1Functor<TConfig, Tb>(b, m, num_rings, offset, tag, stream)
    {
    }

    void operator()(CommsMPIHostBufferStream<TConfig> &comm);

    void operator()(CommsMPIDirect<TConfig> &comm);
};

//Visitor that specifies whether
//computation should continue
//for GPU DIrect (CUDA-Aware MPI)
//
template<typename T>
struct NoOpDirectVisitor: VisitorBase<T>
{
        NoOpDirectVisitor(void):
            m_continue(false)
        {
        }

        void VisitComms(CommsMPI<T> & )
        {
            //purposely empty...
        }

        void VisitCommsHostbuffer(CommsMPIHostBufferStream<T> &comm)
        {
            m_continue = true;
        }

        void VisitCommsMPIDirect(CommsMPIDirect<T> &comm)
        {
            m_continue = false;
        }

        bool do_continue(void) const
        {
            return m_continue;
        }
    private:
        bool m_continue;
};

template<typename TConfig, typename Tb>
struct AddFromHalo1Functor: ExcHalo1Functor<TConfig, Tb>
{
        using ExcHalo1Functor<TConfig, Tb>::get_b;
        using ExcHalo1Functor<TConfig, Tb>::get_m;
        using ExcHalo1Functor<TConfig, Tb>::get_num_rings;
        using ExcHalo1Functor<TConfig, Tb>::get_offset;
        using ExcHalo1Functor<TConfig, Tb>::get_tag;

        AddFromHalo1Functor(void):
            ExcHalo1Functor<TConfig, Tb>(),
            m_ptr_stream(nullptr)
        {
        }

        AddFromHalo1Functor(Tb &b,
                            const Matrix<TConfig> &m,
                            int num_rings,
                            int offset,
                            cudaStream_t &stream):
            ExcHalo1Functor<TConfig, Tb>(b, m, num_rings, offset),
            m_ptr_stream(&stream)
        {
        }

        AddFromHalo1Functor(Tb &b,
                            const Matrix<TConfig> &m,
                            int num_rings,
                            int offset,
                            int tag,
                            cudaStream_t &stream):
            ExcHalo1Functor<TConfig, Tb>(b, m, num_rings, offset, tag),
            m_ptr_stream(&stream)
        {
        }

        cudaStream_t &get_stream(void)
        {
            if ( m_ptr_stream )
            {
                return *m_ptr_stream;
            }
            else
            {
                std::stringstream ss;
                ss << "ERROR: Attempt to dereference m_ptr_stream nullptr."
                   << __FILE__
                   << "; "
                   << __LINE__;
                throw std::runtime_error(ss.str());
            }
        }

        int &get_send_size(void)
        {
            return m_send_size;
        }

        void operator()(CommsMPIHostBufferStream<TConfig> &comm);

        void operator()(CommsMPIDirect<TConfig> &comm);
    private:
        cudaStream_t *m_ptr_stream;
        int m_send_size;
};

template<typename TConfig, typename Tb>
struct AddFromHalo2Functor: AddFromHalo1Functor<TConfig, Tb>
{
    using ExcHalo1Functor<TConfig, Tb>::get_b;
    using ExcHalo1Functor<TConfig, Tb>::get_m;
    using ExcHalo1Functor<TConfig, Tb>::get_num_rings;
    using ExcHalo1Functor<TConfig, Tb>::get_offset;
    using ExcHalo1Functor<TConfig, Tb>::get_tag;
    using AddFromHalo1Functor<TConfig, Tb>::get_stream;
    using AddFromHalo1Functor<TConfig, Tb>::get_send_size;

    AddFromHalo2Functor(void)
    {
    }

    AddFromHalo2Functor(Tb &b,
                        const Matrix<TConfig> &m,
                        int num_rings,
                        int offset,
                        cudaStream_t &stream):
        AddFromHalo1Functor<TConfig, Tb>(b, m, num_rings, offset, stream)
    {
    }

    AddFromHalo2Functor(Tb &b,
                        const Matrix<TConfig> &m,
                        int num_rings,
                        int offset,
                        int tag,
                        cudaStream_t &stream):
        AddFromHalo1Functor<TConfig, Tb>(b, m, num_rings, offset, tag, stream)
    {
    }

    void operator()(CommsMPIHostBufferStream<TConfig> &comm);

    void operator()(CommsMPIDirect<TConfig> &comm);

};

template<typename TConfig, typename Tb>
struct AddFromHalo3Functor:  AddFromHalo1Functor<TConfig, Tb>
{
        using ExcHalo1Functor<TConfig, Tb>::get_b;
        using ExcHalo1Functor<TConfig, Tb>::get_m;
        using ExcHalo1Functor<TConfig, Tb>::get_num_rings;
        using ExcHalo1Functor<TConfig, Tb>::get_offset;
        using ExcHalo1Functor<TConfig, Tb>::get_tag;
        using AddFromHalo1Functor<TConfig, Tb>::get_stream;
        using AddFromHalo1Functor<TConfig, Tb>::get_send_size;

        AddFromHalo3Functor(void)
        {
        }

        AddFromHalo3Functor(Tb &b,
                            const Matrix<TConfig> &m,
                            int num_rings,
                            int offset,
                            cudaStream_t &stream):
            AddFromHalo1Functor<TConfig, Tb>(b, m, num_rings, offset, stream)
        {
        }

        AddFromHalo3Functor(Tb &b,
                            const Matrix<TConfig> &m,
                            int num_rings,
                            int offset,
                            int tag,
                            cudaStream_t &stream):
            AddFromHalo1Functor<TConfig, Tb>(b, m, num_rings, offset, tag, stream)
        {
        }

        void operator()(CommsMPIHostBufferStream<TConfig> &comm);

        void operator()(CommsMPIDirect<TConfig> &comm);

        int &get_recv_size(void)
        {
            return m_recv_size;
        }

    private:
        int m_recv_size;
};

template<typename TConfig, typename Tb>
struct SendRecvWait1Functor: AddFromHalo1Functor<TConfig, Tb>
{
    using ExcHalo1Functor<TConfig, Tb>::get_b;
    using ExcHalo1Functor<TConfig, Tb>::get_m;
    using ExcHalo1Functor<TConfig, Tb>::get_num_rings;
    using ExcHalo1Functor<TConfig, Tb>::get_offset;
    using ExcHalo1Functor<TConfig, Tb>::get_tag;
    using AddFromHalo1Functor<TConfig, Tb>::get_stream;
    using AddFromHalo1Functor<TConfig, Tb>::get_send_size;

    SendRecvWait1Functor(void)
    {
    }

    SendRecvWait1Functor(Tb &b,
                         const Matrix<TConfig> &m,
                         int num_rings,
                         int offset,
                         cudaStream_t &stream):
        AddFromHalo1Functor<TConfig, Tb>(b, m, num_rings, offset, stream)
    {
    }

    SendRecvWait1Functor(Tb &b,
                         const Matrix<TConfig> &m,
                         int num_rings,
                         int offset,
                         int tag,
                         cudaStream_t &stream):
        AddFromHalo1Functor<TConfig, Tb>(b, m, num_rings, offset, tag, stream)
    {
    }

    void operator()(CommsMPIHostBufferStream<TConfig> &comm);

    void operator()(CommsMPIDirect<TConfig> &comm);

};

template<typename TConfig, typename Tb>
struct SendRecvWait2Functor: AddFromHalo1Functor<TConfig, Tb>
{
    using ExcHalo1Functor<TConfig, Tb>::get_b;
    using ExcHalo1Functor<TConfig, Tb>::get_m;
    using ExcHalo1Functor<TConfig, Tb>::get_num_rings;
    using ExcHalo1Functor<TConfig, Tb>::get_offset;
    using ExcHalo1Functor<TConfig, Tb>::get_tag;
    using AddFromHalo1Functor<TConfig, Tb>::get_stream;
    using AddFromHalo1Functor<TConfig, Tb>::get_send_size;

    SendRecvWait2Functor(void)
    {
    }

    SendRecvWait2Functor(Tb &b,
                         const Matrix<TConfig> &m,
                         int num_rings,
                         int offset,
                         cudaStream_t &stream):
        AddFromHalo1Functor<TConfig, Tb>(b, m, num_rings, offset, stream)
    {
    }

    SendRecvWait2Functor(Tb &b,
                         const Matrix<TConfig> &m,
                         int num_rings,
                         int offset,
                         int tag,
                         cudaStream_t &stream):
        AddFromHalo1Functor<TConfig, Tb>(b, m, num_rings, offset, tag, stream)
    {
    }

    void operator()(CommsMPIHostBufferStream<TConfig> &comm);

    void operator()(CommsMPIDirect<TConfig> &comm);

};

template<typename TConfig, typename Tb>
struct SendRecvWait3Functor: AddFromHalo1Functor<TConfig, Tb>
{
    using ExcHalo1Functor<TConfig, Tb>::get_b;
    using ExcHalo1Functor<TConfig, Tb>::get_m;
    using ExcHalo1Functor<TConfig, Tb>::get_num_rings;
    using ExcHalo1Functor<TConfig, Tb>::get_offset;
    using ExcHalo1Functor<TConfig, Tb>::get_tag;
    using AddFromHalo1Functor<TConfig, Tb>::get_stream;
    using AddFromHalo1Functor<TConfig, Tb>::get_send_size;

    SendRecvWait3Functor(void)
    {
    }

    SendRecvWait3Functor(Tb &b,
                         const Matrix<TConfig> &m,
                         int num_rings,
                         int offset,
                         cudaStream_t &stream):
        AddFromHalo1Functor<TConfig, Tb>(b, m, num_rings, offset, stream)
    {
    }

    SendRecvWait3Functor(Tb &b,
                         const Matrix<TConfig> &m,
                         int num_rings,
                         int offset,
                         int tag,
                         cudaStream_t &stream):
        AddFromHalo1Functor<TConfig, Tb>(b, m, num_rings, offset, tag, stream)
    {
    }

    void operator()(CommsMPIHostBufferStream<TConfig> &comm);

    void operator()(CommsMPIDirect<TConfig> &comm);

};


} // namespace amgx
