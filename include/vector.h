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

/* NOTE: These enums are mostly used in distributed setting,
   where the matrix is partitioned by sets of rows across distributed nodes.
   Each distributed node contains a rectangular partition of the matrix,
   defined by a set of rows belonging to this node.
   The rows are reordered so that
   interior rows - rows with no connections to other partitions - come first
   boundary rows - rows with connections to other partitions - come second
   halo rows - rows from other parttions to which the boundary rows were connected to - come last */
enum ViewType
{
    INTERIOR = 1, /* only the interior rows (with no connections to other partitions) are seen */
    BOUNDARY = 2, /* only the boundary rows (with connections to other partitions) are seen */
    OWNED = 3,  /* both the interior and boundary rows are seen */
    HALO1 = 4,  /* only the (first ring*) halo rows (rows from other partitions that are connected to the local boundary rows and that have been appended locally) are seen */
    FULL = 7,   /* all interior, boundary and halo rows are seen */
    HALO2 = 8,  /* only the (second ring) halo rows (rows from other partitions that are connected to the [first ring] halo rows and that have been appended locally) are seen */
    ALL = 15    /* all interior, boundary and (first and second ring) halo rows are seen */
};
/* *: first and second ring refer to first and second degree neighbors in a graph associated with the current matrix. */

/* NOTE: These enums are mostly used in distributed setting.
   They refer to properties of the vector, and are assigned to variable in_transfer below. */
enum TransferType
{
    IDLE = 0,    /* the vector is neither being send or received (default) */
    SENDING = 1, /* Isend for the vector has been issued */
    RECEIVING = 2 /* Irecv for the vector has been issued (often need to synchronize to make sure the transfer is finished) */
};

/* NOTE: These enums are used during coloring of the matrix (for incomplete factorizations, such as DILU, ILU0, etc.).
         They refer to the order in which colors are assigned to the boundary (boundary_coloring) and halo (halo_coloring)
         with respect to the colors used for the interior nodes. */
enum ColoringType
{
    FIRST = 0,     /* color the boundary/halo nodes first */
    SYNC_COLORS = 1, /* color the boundary/halo and the interior nodes together at the same time */
    LAST = 2       /* color the boundary/halo nodes last */
};

}

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

#include <cusp/array1d.h>
#include <async_event.h>
#include <error.h>
#include <basic_types.h>
#include <resources.h>
#include <distributed/amgx_mpi.h>
#include <auxdata.h>

#include <amgx_types/pod_types.h>

#include "vector_thrust_allocator.h"

namespace amgx
{

const int sleep_us = 20;

// usleep for windows
#ifdef _WIN32
#include <Windows.h>
static void usleep(int waitTime)
{
    LARGE_INTEGER time1;
    LARGE_INTEGER time2;
    LARGE_INTEGER freq;
    QueryPerformanceCounter((LARGE_INTEGER *)&time1);
    QueryPerformanceFrequency((LARGE_INTEGER *)&freq);

    do
    {
        QueryPerformanceCounter((LARGE_INTEGER *)&time2);
    }
    while ( (time2.QuadPart - time1.QuadPart) * 1000000.0 / freq.QuadPart < waitTime);
}
#endif

template <class T_Config> class Vector;
template <class TConfig> class DistributedManager;

//Vector wrapping class which automatically performs indexing algerbra
template <class value_type, class index_type>
class PODVector
{
        typedef index_type IndexType;
        typedef index_type ValueType;
    public:
        __device__ __host__ inline PODVector(value_type *ptr, index_type block_dimy, index_type block_dimx) : ptr(ptr), block_size(block_dimx * block_dimy), block_dimx(block_dimx) {}
        __device__ __host__ inline PODVector(const value_type *ptr, index_type block_dimy, index_type block_dimx) : ptr(ptr), block_size(block_dimx * block_dimy), block_dimx(block_dimx) {}

        //access operators
        __device__ __host__ inline value_type &operator[](index_type idx) { return ptr[idx]; }
        __device__ __host__ inline value_type &operator()(index_type k, index_type i = 0, index_type j = 0) { return ptr[k * block_size + i * block_dimx + j]; }

        __device__ __host__ inline short get_block_dimx() const { return block_dimx; }
        __device__ __host__ inline short get_block_dimy() const { return block_size / block_dimx; }
        __device__ __host__ inline short get_block_size() const { return block_size; }

    private:
        value_type *ptr;
        short block_size;    //=block_dimx*block_dimy
        short block_dimx;
        //const short block_dimy;  //Not stored
};
//Vector wrapping class which automatically performs indexing algerbra
template <class value_type, class index_type>
class constPODVector
{
        typedef index_type IndexType;
        typedef index_type ValueType;
    public:
        __device__ __host__ inline constPODVector(const value_type *ptr, index_type block_dimy, index_type block_dimx) : ptr(ptr), block_size(block_dimx * block_dimy), block_dimx(block_dimx) {}

        //access operators
        __device__ __host__ inline const value_type &operator[](index_type idx) const { return ptr[idx]; }
        __device__ __host__ inline const value_type &operator()(index_type k, index_type i = 0, index_type j = 0) const { return ptr[k * block_size + i * block_dimx + j]; }

        __device__ __host__ inline short get_block_dimx() const { return block_dimx; }
        __device__ __host__ inline short get_block_dimy() const { return block_size / block_dimx; }
        __device__ __host__ inline short get_block_size() const { return block_size; }

    private:
        const value_type *ptr;
        const short block_size;    //=block_dimx*block_dimy;
        const short block_dimx;
        //const short block_dimy;  //Not stored
};

// specialization for host
template <AMGX_VecPrecision t_vecPrec, AMGX_MatPrecision t_matPrec, AMGX_IndPrecision t_indPrec>
class Vector<TemplateConfig<AMGX_host, t_vecPrec, t_matPrec, t_indPrec> > : public thrust::host_vector<typename VecPrecisionMap<t_vecPrec>::Type>
{
        typedef typename MemorySpaceMap<AMGX_host>::Type host_memory;
        typedef typename MemorySpaceMap<AMGX_device>::Type device_memory;
    public:
        typedef TemplateConfig<AMGX_host, t_vecPrec, t_matPrec, t_indPrec> TConfig;
        typedef TemplateConfig<AMGX_host, t_vecPrec, t_matPrec, t_indPrec> TConfig_h;
        typedef TemplateConfig<AMGX_device, t_vecPrec, t_matPrec, t_indPrec> TConfig_d;
        typedef typename TConfig::MemSpace memory_space;
        typedef typename TConfig::VecPrec value_type;
        typedef typename TConfig::IndPrec index_type;
        typedef cusp::array1d_format format;

        Vector() :  block_dimx(1), block_dimy(1), num_rows(0), num_cols(1), lda(0), buffer(NULL), buffer_size(0), dirtybit(1), in_transfer(IDLE), tag(-1), delayed_send(1), cancel(0), manager(NULL), v_is_transformed(false), v_is_read_partitioned(false), host_send_recv_buffer(NULL), linear_buffers_size(0), explicit_host_buffer(NULL), explicit_buffer_size(0), m_resources(0) { };
        inline Vector(unsigned int N) : thrust::host_vector<value_type>(N), block_dimx(1), block_dimy(1), num_rows(0), num_cols(1), lda(0), buffer(NULL), buffer_size(0), dirtybit(1), in_transfer(IDLE), tag(-1), delayed_send(1), cancel(0), manager(NULL), v_is_transformed(false), v_is_read_partitioned(false), host_send_recv_buffer(NULL), linear_buffers_size(0), explicit_buffer_size(0), explicit_host_buffer(NULL), m_resources(0) {}
        inline Vector(unsigned int N, value_type v) : thrust::host_vector<value_type>(N, v), block_dimx(1), block_dimy(1), num_rows(0), num_cols(1), lda(0), buffer(NULL), buffer_size(0), dirtybit(1), in_transfer(IDLE), tag(-1), delayed_send(1), cancel(0), manager(NULL), v_is_transformed(false), v_is_read_partitioned(false), host_send_recv_buffer(NULL), linear_buffers_size(0), explicit_buffer_size(0), explicit_host_buffer(NULL), m_resources(0) {}
        inline Vector(const Vector<TConfig_h> &a) : thrust::host_vector<value_type>(a), block_dimx(a.get_block_dimx()), block_dimy(a.get_block_dimy()), num_rows(a.get_num_rows()), num_cols(a.get_num_cols()), lda(a.get_lda()), buffer(NULL), buffer_size(0), dirtybit(1), in_transfer(IDLE), tag(-1), delayed_send(1), cancel(0), manager(NULL), v_is_transformed(false), v_is_read_partitioned(false), host_send_recv_buffer(NULL), linear_buffers_size(0), explicit_buffer_size(0), explicit_host_buffer(NULL), m_resources(0) {}
        inline Vector(const Vector<TConfig_d> &a) : thrust::host_vector<value_type>(a), block_dimx(a.get_block_dimx()), block_dimy(a.get_block_dimy()), num_rows(a.get_num_rows()), num_cols(a.get_num_cols()), lda(a.get_lda()), buffer(NULL), buffer_size(0), dirtybit(1), in_transfer(IDLE), tag(-1), delayed_send(1), cancel(0), manager(NULL), v_is_transformed(false), v_is_read_partitioned(false), host_send_recv_buffer(NULL), linear_buffers_size(0), explicit_buffer_size(0), explicit_host_buffer(NULL), m_resources(0) {}

        ~Vector()
        {
#ifdef AMGX_WITH_MPI

            if (requests.size() && in_transfer != IDLE)
            {
                //if async host-copy comms module is being used, signal deallocation and wait until communications are finished/canceled
                if (explicit_host_buffer != NULL && (in_transfer & RECEIVING))
                {
                    cancel = 1;

                    while (cancel) {usleep(5);}
                }
                else
                {
                    for (unsigned int i = 0; i < requests.size(); i++)
                    {
                        if ((i < requests.size() / 2 && (in_transfer & SENDING)) || (i >= requests.size() / 2 && (in_transfer & RECEIVING)))
                        {
                            MPI_Cancel(&requests[i]);
                        }
                    }

                    MPI_Waitall((int)requests.size(), &requests[0], &statuses[0]);
                }
            }

#endif

            if (buffer != NULL) { delete buffer; }

            if (linear_buffers_size != 0)
            {
                amgx::memory::cudaFreeHost(&(linear_buffers[0]));
            }

            if (host_send_recv_buffer != NULL) { delete host_send_recv_buffer; }

            if (explicit_host_buffer)
            {
                amgx::memory::cudaFreeHost(explicit_host_buffer);
                cudaEventDestroy(mpi_event);
            }
        }

        operator PODVector<value_type, index_type>() { return PODVector<value_type, index_type>(raw(), block_dimy, block_dimx); }
        operator constPODVector<value_type, index_type>() const { return constPODVector<value_type, index_type>(raw(), block_dimy, block_dimx); }
        PODVector<value_type, index_type> pod() { return PODVector<value_type, index_type>(raw(), block_dimy, block_dimx); }
        constPODVector<value_type, index_type> const_pod() const { return constPODVector<value_type, index_type>(raw(), block_dimy, block_dimx); }

        template< typename OtherVector >
        inline void copy( const OtherVector &a )
        {
            //copy dimensions
            this->set_block_dimx( a.get_block_dimx());
            this->set_block_dimy( a.get_block_dimy());
            this->set_num_rows(a.get_num_rows());
            this->set_num_cols(a.get_num_cols());
            this->set_lda(a.get_lda());
            this->resize(a.size(), value_type());
            //copy data
            this->assign( a.begin( ), a.end( ) );

            //reset distributed flags and buffers
            if (tag == -1) { tag = a.tag; }

            in_transfer = IDLE;
            dirtybit = a.dirtybit;

            if (buffer != NULL)
            {
                delete buffer;
                buffer = NULL;
                buffer_size = 0;
            }

            if (linear_buffers_size != 0)
            {
                amgx::memory::cudaFreeHost(&(linear_buffers[0]));
                linear_buffers_size = 0;
            }

            if (explicit_host_buffer)
            {
                amgx::memory::cudaFreeHost(explicit_host_buffer);
                explicit_host_buffer = NULL;
                explicit_buffer_size = 0;
                cudaEventDestroy(mpi_event);
            }
        }

        //inline void copy_async(const Vector<ScalarType,host_memory> & a) { ... }  No host to host asynchronous copy
        inline void copy_async(const Vector<TConfig_d> &a, cudaStream_t s = 0)
        {
            //copy dimensions
            this->set_block_dimx( a.get_block_dimx());
            this->set_block_dimy( a.get_block_dimy());
            this->set_num_rows(a.get_num_rows());
            this->set_num_cols(a.get_num_cols());
            this->set_lda(a.get_lda());
            this->resize(a.size(), value_type());
            //copy data
            cudaMemcpyAsync(raw(), a.raw(), bytes(), cudaMemcpyDefault, s);
            event.record();

            //reset distributed flags and buffers
            if (tag == -1) { tag = a.tag; }

            in_transfer = IDLE;
            dirtybit = a.dirtybit;

            if (buffer != NULL)
            {
                delete buffer;
                buffer = NULL;
                buffer_size = 0;
            }

            if (linear_buffers_size != 0)
            {
                amgx::memory::cudaFreeHost(&(linear_buffers[0]));
                linear_buffers_size = 0;
            }

            if (explicit_host_buffer)
            {
                amgx::memory::cudaFreeHost(explicit_host_buffer);
                explicit_host_buffer = NULL;
                explicit_buffer_size = 0;
                cudaEventDestroy(mpi_event);
            }
        }

        inline void sync() { event.sync(); }

        inline Vector<TConfig_h> &operator=(const cusp::array1d<value_type, host_memory> &a) { copy(a); return *this; }
        inline Vector<TConfig_h> &operator=(const cusp::array1d<value_type, device_memory> &a) { copy(a); return *this; }
        inline Vector<TConfig_h> &operator=(const Vector<TConfig_h> &a) { copy(a); return *this; }
        inline Vector<TConfig_h> &operator=(const Vector<TConfig_d> &a) { copy(a); return *this; }

        inline value_type *raw()
        {
            if (bytes() > 0) { return thrust::raw_pointer_cast(this->data()); }
            else { return 0; }
        }
        inline const value_type *raw() const
        {
            if (bytes() > 0) { return thrust::raw_pointer_cast(this->data()); }
            else { return 0; }
        }

        inline short get_block_size() const { return block_dimx * block_dimy; }
        inline short get_block_dimx() const { return block_dimx; }
        inline short get_block_dimy() const { return block_dimy; }
        inline void set_block_dimx(short dimx) { block_dimx = dimx; }
        inline void set_block_dimy(short dimy) { block_dimy = dimy; }

        int get_num_cols() const { return num_cols; }
        int get_num_rows() const { return num_rows; }
        int get_lda() const { return lda; }
        void set_num_rows(int size) { num_rows = size; }
        void set_num_cols(int size) { num_cols = size; }
        void set_lda(int size) { lda = size; }

        inline size_t bytes(bool device_only = false) const
        {
            size_t res = 0;

            if (!device_only)
            {
                res = this->size() * sizeof(value_type);
            }

            return res;
        }

        void printConfig()
        {
            printf("Configuration: %s, %s, %s, %s\n",
                   TConfig::MemSpaceInfo::getName(),
                   TConfig::VecPrecInfo::getName(),
                   TConfig::MatPrecInfo::getName(),
                   TConfig::IndPrecInfo::getName());
        }

        void init_host_send_recv_buffer() { host_send_recv_buffer =  new Vector<TConfig_h>(this->size()); }

        value_type **linear_buffers;
        int linear_buffers_size;
        thrust::host_vector<value_type *> linear_buffers_ptrs;

        Vector<TemplateConfig<AMGX_host, t_vecPrec, t_matPrec, t_indPrec> > *buffer;
        Vector<TConfig_h> *host_send_recv_buffer;

        void setManager(DistributedManager<TConfig> &manager) { this->manager = &manager; }
        void unsetManager() { this->manager = NULL;}
        DistributedManager<TConfig> *getManager() const {return manager;}
        void unset_transformed() {v_is_transformed = false;}
        void set_transformed() {v_is_transformed = true;}
        bool is_transformed() { return v_is_transformed;}
        void set_is_vector_read_partitioned(bool is_read_partitioned) {v_is_read_partitioned = is_read_partitioned;}
        inline bool is_vector_read_partitioned() const {return v_is_read_partitioned;}

        inline Resources *getResources() const { return m_resources; }
        inline void setResources(Resources *resources) { m_resources = resources; }

        int tag;
        volatile int dirtybit;
        volatile int cancel; //Signals to the async host-copy comms module that vector is being deallocated
        int delayed_send;
        unsigned int in_transfer;
        std::vector<value_type> host_buffer;
        value_type *explicit_host_buffer;  //A separate pinned memory buffer to be used by async host-copy comms module
        int explicit_buffer_size;
        int buffer_size;
        cudaEvent_t mpi_event;

#ifdef AMGX_WITH_MPI
        std::vector<MPI_Request> requests;
        std::vector<MPI_Status> statuses;

        std::vector<MPI_Request> send_requests;
        std::vector<MPI_Request> recv_requests;

        std::vector<MPI_Status> send_statuses;
        std::vector<MPI_Status> recv_statuses;
#endif
    private:
        AsyncEvent event;
        short block_dimx, block_dimy;
        unsigned int num_rows, num_cols, lda;
        bool v_is_transformed;
        bool v_is_read_partitioned;
        DistributedManager<TConfig> *manager;

        Resources *m_resources;
};

// specialization for device
template <AMGX_VecPrecision t_vecPrec, AMGX_MatPrecision t_matPrec, AMGX_IndPrecision t_indPrec>
class Vector<TemplateConfig<AMGX_device, t_vecPrec, t_matPrec, t_indPrec> > : public device_vector_alloc<typename VecPrecisionMap<t_vecPrec>::Type>
{
        typedef typename MemorySpaceMap<AMGX_host>::Type host_memory;
        typedef typename MemorySpaceMap<AMGX_device>::Type device_memory;
    public:
        typedef TemplateConfig<AMGX_device, t_vecPrec, t_matPrec, t_indPrec> TConfig;
        typedef TemplateConfig<AMGX_host, t_vecPrec, t_matPrec, t_indPrec> TConfig_h;
        typedef TemplateConfig<AMGX_device, t_vecPrec, t_matPrec, t_indPrec> TConfig_d;
        typedef typename TConfig::MemSpace memory_space;
        typedef typename TConfig::VecPrec value_type;
        typedef typename TConfig::IndPrec index_type;
        typedef cusp::array1d_format format;

        Vector() : block_dimx(1), block_dimy(1), num_rows(0), num_cols(1), lda(0), buffer(NULL), dirtybit(1), in_transfer(IDLE), tag(-1), delayed_send(1), cancel(0), manager(NULL), v_is_transformed(false), v_is_read_partitioned(false), host_send_recv_buffer(NULL), linear_buffers_size(0), explicit_buffer_size(0), explicit_host_buffer(NULL), m_resources(0) {}
        inline Vector(unsigned int N) : device_vector_alloc<value_type>(N), block_dimx(1), block_dimy(1), num_rows(0), num_cols(1), lda(0), buffer(NULL), dirtybit(1), in_transfer(IDLE), tag(-1), delayed_send(1), cancel(0), manager(NULL), v_is_transformed(false), v_is_read_partitioned(false), host_send_recv_buffer(NULL), linear_buffers_size(0), explicit_buffer_size(0), explicit_host_buffer(NULL), m_resources(0) {}
        inline Vector(unsigned int N, int dimx, int dimy) : device_vector_alloc<value_type>(N), block_dimx(dimx), block_dimy(dimy), num_rows(0), num_cols(1), lda(0), buffer(NULL), dirtybit(1), in_transfer(IDLE), tag(-1), delayed_send(1), cancel(0), manager(NULL), v_is_transformed(false), v_is_read_partitioned(false), host_send_recv_buffer(NULL), linear_buffers_size(0), explicit_buffer_size(0), explicit_host_buffer(NULL), m_resources(0) {}
        inline Vector(unsigned int N, value_type v) : device_vector_alloc<value_type>(N, v), block_dimx(1), block_dimy(1), num_rows(0), num_cols(1), lda(0), buffer(NULL), buffer_size(0), dirtybit(1), in_transfer(IDLE), tag(-1), delayed_send(1), cancel(0), manager(NULL), v_is_transformed(false), v_is_read_partitioned(false), host_send_recv_buffer(NULL), linear_buffers_size(0), explicit_buffer_size(0), explicit_host_buffer(NULL), m_resources(0) {}
        inline Vector(const Vector<TConfig_h> &a) : device_vector_alloc<value_type>(a), block_dimx(a.get_block_dimx()), block_dimy(a.get_block_dimy()), num_rows(a.get_num_rows()), num_cols(a.get_num_cols()), lda(a.get_lda()), buffer(NULL), buffer_size(0), dirtybit(1), in_transfer(IDLE), tag(-1), delayed_send(1), cancel(0), manager(NULL), v_is_transformed(false), v_is_read_partitioned(false), host_send_recv_buffer(NULL), linear_buffers_size(0), explicit_buffer_size(0), explicit_host_buffer(NULL), m_resources(0) {}
        inline Vector(const Vector<TConfig_d> &a) : device_vector_alloc<value_type>(a), block_dimx(a.get_block_dimx()), block_dimy(a.get_block_dimy()), num_rows(a.get_num_rows()), num_cols(a.get_num_cols()), lda(a.get_lda()), buffer(NULL), buffer_size(0), dirtybit(1), in_transfer(IDLE), tag(-1), delayed_send(1), cancel(0), manager(NULL), v_is_transformed(false), v_is_read_partitioned(false), host_send_recv_buffer(NULL), linear_buffers_size(0), explicit_buffer_size(0), explicit_host_buffer(NULL), m_resources(0) {}

        ~Vector()
        {
#ifdef AMGX_WITH_MPI

            if (requests.size() && in_transfer != IDLE)
            {
                //if async host-copy comms module is being used, signal deallocation and wait until communications are finished/canceled
                if (explicit_host_buffer != NULL && (in_transfer & RECEIVING))
                {
                    cancel = 1;

                    while (cancel) {usleep(5);}
                }
                else
                {
                    for (unsigned int i = 0; i < requests.size(); i++)
                    {
                        if ((i < requests.size() / 2 && (in_transfer & SENDING)) || (i >= requests.size() / 2 && (in_transfer & RECEIVING)))
                        {
                            MPI_Cancel(&requests[i]);
                        }
                    }

                    MPI_Waitall((int)requests.size(), &requests[0], &statuses[0]);
                }
            }

#endif

            if (buffer != NULL) { delete buffer; }

            if (linear_buffers_size != 0)
            {
                amgx::memory::cudaFreeHost(&(linear_buffers[0]));
            }

            if (host_send_recv_buffer != NULL) { delete host_send_recv_buffer; }

            if (explicit_host_buffer)
            {
                amgx::memory::cudaFreeHost(explicit_host_buffer);
                cudaEventDestroy(mpi_event);
            }
        }

        operator PODVector<value_type, index_type>() { return PODVector<value_type, index_type>(raw(), block_dimy, block_dimx); }
        operator constPODVector<value_type, index_type>() const { return constPODVector<value_type, index_type>(raw(), block_dimy, block_dimx); }
        PODVector<value_type, index_type> pod() { return PODVector<value_type, index_type>(raw(), block_dimy, block_dimx); }
        constPODVector<value_type, index_type> const_pod() const { return constPODVector<value_type, index_type>(raw(), block_dimy, block_dimx); }

        template< typename OtherVector >
        inline void copy( const OtherVector &a )
        {
            //copy dimensions
            this->set_block_dimx( a.get_block_dimx());
            this->set_block_dimy( a.get_block_dimy());
            this->set_num_rows(a.get_num_rows());
            this->set_num_cols(a.get_num_cols());
            this->set_lda(a.get_lda());
            this->resize(a.size(), value_type());
            //copy data
            this->assign( a.begin( ), a.end( ) );

            //reset distributed flags and buffers
            if (tag == -1) { tag = a.tag; }

            in_transfer = IDLE;
            dirtybit = a.dirtybit;

            if (buffer != NULL)
            {
                delete buffer;
                buffer = NULL;
                buffer_size = 0;
            }

            if (linear_buffers_size != 0)
            {
                amgx::memory::cudaFreeHost(&(linear_buffers[0]));
                linear_buffers_size = 0;
            }

            if (explicit_host_buffer)
            {
                amgx::memory::cudaFreeHost(explicit_host_buffer);
                explicit_host_buffer = NULL;
                explicit_buffer_size = 0;
                cudaEventDestroy(mpi_event);
            }
        }

        inline void copy_async(const Vector<TConfig_h> &a, cudaStream_t s = 0)
        {
            //copy dimensions
            this->set_block_dimx( a.get_block_dimx());
            this->set_block_dimy( a.get_block_dimy());
            this->set_num_rows(a.get_num_rows());
            this->set_num_cols(a.get_num_cols());
            this->set_lda(a.get_lda());
            this->resize(a.size(), value_type());
            //copy data
            cudaMemcpyAsync(raw(), a.raw(), bytes(), cudaMemcpyDefault, s);
            event.record();

            //reset distributed flags and buffers
            if (tag == -1) { tag = a.tag; }

            in_transfer = IDLE;
            dirtybit = a.dirtybit;

            if (buffer != NULL)
            {
                delete buffer;
                buffer = NULL;
                buffer_size = 0;
            }

            if (linear_buffers_size != 0)
            {
                amgx::memory::cudaFreeHost(&(linear_buffers[0]));
                linear_buffers_size = 0;
            }

            if (explicit_host_buffer)
            {
                amgx::memory::cudaFreeHost(explicit_host_buffer);
                explicit_host_buffer = NULL;
                explicit_buffer_size = 0;
                cudaEventDestroy(mpi_event);
            }
        }

        inline void copy_async(const Vector<TConfig_d> &a, cudaStream_t s = 0)
        {
            //copy dimensions
            this->set_block_dimx( a.get_block_dimx());
            this->set_block_dimy( a.get_block_dimy());
            this->set_num_rows(a.get_num_rows());
            this->set_num_cols(a.get_num_cols());
            this->set_lda(a.get_lda());
            this->resize(a.size(), value_type());
            //copy data
            cudaMemcpyAsync(raw(), a.raw(), bytes(), cudaMemcpyDefault, s);
            event.record();

            //reset distributed flags and buffers
            if (tag == -1) { tag = a.tag; }

            in_transfer = IDLE;
            dirtybit = a.dirtybit;

            if (buffer != NULL)
            {
                delete buffer;
                buffer = NULL;
                buffer_size = 0;
            }

            if (linear_buffers_size != 0)
            {
                amgx::memory::cudaFreeHost(&(linear_buffers[0]));
                linear_buffers_size = 0;
            }

            if (explicit_host_buffer)
            {
                amgx::memory::cudaFreeHost(explicit_host_buffer);
                explicit_host_buffer = NULL;
                explicit_buffer_size = 0;
                cudaEventDestroy(mpi_event);
            }
        }

        inline void sync() { event.sync(); }

        inline Vector<TConfig_d> &operator=(const cusp::array1d<value_type, host_memory> &a) { copy(a); return *this; }
        inline Vector<TConfig_d> &operator=(const cusp::array1d<value_type, device_memory> &a) { copy(a); return *this; }
        inline Vector<TConfig_d> &operator=(const Vector<TConfig_h> &a) { copy(a); return *this; }
        inline Vector<TConfig_d> &operator=(const Vector<TConfig_d> &a) { copy(a); return *this; }

        inline value_type *raw()
        {
            if (bytes() > 0) { return thrust::raw_pointer_cast(this->data()); }
            else { return 0; }
        }
        inline const value_type *raw() const
        {
            if (bytes() > 0) { return thrust::raw_pointer_cast(this->data()); }
            else { return 0; }
        }

        inline short get_block_size() const { return block_dimx * block_dimy; }
        inline short get_block_dimx() const { return block_dimx; }
        inline short get_block_dimy() const { return block_dimy; }
        inline void set_block_dimx(short dimx) { block_dimx = dimx; }
        inline void set_block_dimy(short dimy) { block_dimy = dimy; }

        int get_num_rows() const { return num_rows; }
        int get_num_cols() const { return num_cols; }
        int get_lda() const { return lda; }
        void set_num_rows(int size) { num_rows = size; }
        void set_num_cols(int size) { num_cols = size; }
        void set_lda(int size) { lda = size; }

        inline size_t bytes(bool device_only = false) const
        {
            return this->size() * sizeof(value_type);
        }
        void printConfig()
        {
            printf("Configuration: %s, %s, %s, %s\n",
                   TConfig::MemSpaceInfo::getName(),
                   TConfig::VecPrecInfo::getName(),
                   TConfig::MatPrecInfo::getName(),
                   TConfig::IndPrecInfo::getName());
        }

        void setManager(DistributedManager<TConfig> &manager) { this->manager = &manager; }
        void unsetManager() { this->manager = NULL;}
        DistributedManager<TConfig> *getManager() const {return manager;}
        void set_transformed() {v_is_transformed = true;}
        void unset_transformed() {v_is_transformed = false;}
        bool is_transformed() { return v_is_transformed;}
        void set_is_vector_read_partitioned(bool is_read_partitioned) {v_is_read_partitioned = is_read_partitioned;}
        inline bool is_vector_read_partitioned() const {return v_is_read_partitioned;}

        inline Resources *getResources() const { return m_resources; }
        inline void setResources(Resources *resources) { m_resources = resources; }

        void init_host_send_recv_buffer() {  host_send_recv_buffer =  new Vector<TConfig_h>(this->size()); }

        value_type **linear_buffers;
        int linear_buffers_size;
        device_vector_alloc<value_type *> linear_buffers_ptrs;

        Vector<TemplateConfig<AMGX_device, t_vecPrec, t_matPrec, t_indPrec> > *buffer;
        Vector<TConfig_h > *host_send_recv_buffer;

        int tag;
        volatile int dirtybit;
        volatile int cancel; //Signals to the async host-copy comms module that vector is being deallocated
        int delayed_send;
        unsigned int in_transfer;
        std::vector<value_type> host_buffer;
        value_type *explicit_host_buffer;  //A separate pinned memory buffer to be used by async host-copy comms module
        int explicit_buffer_size;
        int buffer_size;
        cudaEvent_t mpi_event;

#ifdef AMGX_WITH_MPI
        std::vector<MPI_Request> requests;
        std::vector<MPI_Status> statuses;

        std::vector<MPI_Request> send_requests;
        std::vector<MPI_Request> recv_requests;

        std::vector<MPI_Status> send_statuses;
        std::vector<MPI_Status> recv_statuses;
#endif

    private:
        DistributedManager<TConfig> *manager;
        bool v_is_transformed;
        bool v_is_read_partitioned; //distributed: need this to tell the partitioner the vector is coming from distributed reader
        AsyncEvent event;
        short block_dimx, block_dimy;
        unsigned int num_rows, num_cols, lda;

        Resources *m_resources;

};


} //end namespace amgx

extern "C" {
#define DEFINE_VECTOR_TYPES \
\
typedef Vector<TConfig> VVector;\
typedef typename TConfig::template setVecPrec<(AMGX_VecPrecision)AMGX_GET_MODE_VAL(AMGX_MatPrecision, TConfig::mode)>::Type mvec_value_type;\
typedef Vector<mvec_value_type> MVector;\
typedef typename TConfig::template setVecPrec<AMGX_vecInt>::Type ivec_value_type;\
typedef Vector<ivec_value_type> IVector;\
typedef typename TConfig::template setVecPrec<AMGX_vecInt64>::Type i64vec_value_type;\
typedef Vector<i64vec_value_type> I64Vector;\
typedef TemplateConfig<AMGX_host,TConfig::vecPrec,TConfig::matPrec,TConfig::indPrec> TConfig_h;\
typedef TemplateConfig<AMGX_device,TConfig::vecPrec,TConfig::matPrec,TConfig::indPrec> TConfig_d;\
typedef Vector<TConfig_d> VVector_d;\
typedef typename TConfig_d::template setVecPrec<(AMGX_VecPrecision)AMGX_GET_MODE_VAL(AMGX_MatPrecision, TConfig_d::mode)>::Type mvec_value_type_d;\
typedef Vector<mvec_value_type_d> MVector_d;\
typedef typename TConfig_d::template setVecPrec<AMGX_vecInt>::Type ivec_value_type_d;\
typedef Vector<ivec_value_type_d> IVector_d;\
typedef typename TConfig_d::template setVecPrec<AMGX_vecInt64>::Type i64vec_value_type_d;\
typedef Vector<i64vec_value_type_d> I64Vector_d;\
typedef Vector<TConfig_h> VVector_h;\
typedef typename TConfig_h::template setVecPrec<(AMGX_VecPrecision)AMGX_GET_MODE_VAL(AMGX_MatPrecision, TConfig_h::mode)>::Type mvec_value_type_h;\
typedef Vector<mvec_value_type_h> MVector_h;\
typedef typename TConfig_h::template setVecPrec<AMGX_vecInt>::Type ivec_value_type_h;\
typedef Vector<ivec_value_type_h> IVector_h;\
typedef typename TConfig_h::template setVecPrec<AMGX_vecInt64>::Type i64vec_value_type_h;\
typedef Vector<i64vec_value_type_h> I64Vector_h;\
typedef TemplateConfig<TConfig::memSpace, types::PODTypes<typename VVector::value_type>::vec_prec, TConfig::matPrec, TConfig::indPrec> PODConfig;\
typedef Vector<PODConfig> POD_Vector;\



}

