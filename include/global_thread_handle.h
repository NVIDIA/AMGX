/* Copyright (c) 2013-2017, NVIDIA CORPORATION. All rights reserved.
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

#include <map>
#include <list>
#include <vector>

#include <mutex>

// os specific vars
// windows
#ifdef WIN32
#include <windows.h>
#include <process.h>
typedef DWORD _thread_id;
#else // WIN32
#include <pthread.h>
typedef pthread_t _thread_id;
#endif // WIN32

_thread_id getCurrentThreadId();

namespace amgx
{
namespace memory
{

// Implementation class.
class MemoryManager;

// The base class for memory pools.
class MemoryPool
{
        // Give special rights to the memory manager.
        friend class MemoryManager;

    protected:
        // Types.
        // typedef std::pair<void*, size_t>        MemoryBlock;
        struct MemoryBlock
        {
            void *m_begin;
            size_t m_size;
            bool   m_first;
            bool   m_managed;

            MemoryBlock(void *begin, size_t size, bool first = false, bool managed = true)
                : m_begin(begin)
                , m_size(size)
                , m_first(first)
                , m_managed(managed)
            {}
        };
        typedef std::list<MemoryBlock>          MemoryBlockList;
        typedef MemoryBlockList::iterator       MemoryBlockListIterator;
        typedef MemoryBlockList::const_iterator MemoryBlockListConstIterator;

    protected:
        // Constructor. Cannot be created directly.
        MemoryPool(size_t max_block_size, size_t page_size, size_t max_size);
    public:
        // Dtor.
        virtual ~MemoryPool();

        // Add a memory region managed by that pool.
        void add_memory(void *ptr, size_t size, bool managed = true);

        // Allocate memory.
        virtual void *allocate(size_t size, size_t &allocated_size);
        // Free a memory block.
        virtual void free(void *ptr, size_t &freed_size);
        // Has a block been allocated?
        bool is_allocated(void *ptr) ;
        // The amount of free memory.
        inline size_t get_free_mem() const { return m_free_mem; }
        // The amount of used memory.
        inline size_t get_used_mem() const { return m_size - m_free_mem; }
        // Size of the max block.
        inline size_t get_max_block_size() const { return m_max_block_size; }

    protected:
        // Get the address of the beginning of a memory block.
        inline char  *get_block_begin(MemoryBlockListIterator it) { return (char *) it->m_begin; }
        // Get the size of a memory block.
        inline size_t get_block_size(MemoryBlockListIterator it) { return it->m_size; }
        // Is it the 1st block of a memory segment.
        inline size_t is_block_first(MemoryBlockListIterator it) { return it->m_first; }

        // Get the address of the beginning of a memory block.
        inline void set_block_begin(MemoryBlockListIterator it, void *begin) { it->m_begin = begin; }
        // Get the size of a memory block.
        inline void set_block_size(MemoryBlockListIterator it, size_t size) { it->m_size = size; }

    private:
        // Get the first allocated block.
        MemoryBlockList::iterator get_used_begin() { return m_used_blocks.begin(); }
        // Get the last allocated block.
        MemoryBlockList::iterator get_used_end() { return m_used_blocks.end(); }
        // Free all blocks.
        void free_all();

    protected:
        // The addresses of the blocks to free.
        std::vector<MemoryBlock> m_owned_ptrs;

        // Memory alignment enforced by the pool.
        size_t m_size, m_max_size, m_max_block_size, m_page_size;
        // Have we recently ran a merge.
        bool m_recently_merged;
        // Statistics.
        size_t m_free_mem;
        // The lists of blocks: Used blocks and free ones.
        MemoryBlockList m_used_blocks;
        MemoryBlockList m_free_blocks;
        //Mutex added to fix ICE threadsafe issue
        std::mutex m_mutex2;

#ifdef USE_CUDAMALLOCASYNC
        cudaMemPool_t m_mem_pool;
#endif

    private:
        // No copy.
        MemoryPool(const MemoryPool &);
        // No assignment.
        MemoryPool &operator=(const MemoryPool &);
};


// global memory pools
class PinnedMemoryPool : public MemoryPool
{
    public:
        // Ctor.
        PinnedMemoryPool();

        // Dtor.
        ~PinnedMemoryPool();
};

struct DeviceMemoryPool : public MemoryPool
{
    public:
        //adds block to the pool
        void expandPool(size_t size,
                        size_t max_block_size);
    public:
        // Ctor.
        DeviceMemoryPool(size_t size,
                         size_t max_block_size,
                         size_t max_size);

        // Dtor.
        ~DeviceMemoryPool();
};

// Do we have a pinned/device memory pool ?
bool hasPinnedMemoryPool();
bool hasDeviceMemoryPool();

// Set memory pools.
void setPinnedMemoryPool(PinnedMemoryPool *pool);
void setDeviceMemoryPool(DeviceMemoryPool *pool);

void setPinnedMemoryPool(_thread_id thread_id, PinnedMemoryPool *pool);
void setDeviceMemoryPool(_thread_id thread_id, DeviceMemoryPool *pool);

// Destroy memory pools.
void destroyPinnedMemoryPool();
void destroyDeviceMemoryPool();

// Destroy memory pools.
void destroyAllPinnedMemoryPools();
void destroyAllDeviceMemoryPools();

void printInfo();

// adds pre-allocated block to the device pool
void expandDeviceMemoryPool(size_t size, size_t max_block_size);

// Do we use async free.
void setAsyncFreeFlag(bool set);
// Do we use device memory pool.
void setDeviceMemoryPoolFlag(bool set);
// Define the scaling factor.
void setMallocScalingFactor(size_t factor);
// Define the scaling threshold.
void setMallocScalingThreshold(size_t threshold);

// Create an async free pool for a thread.
void createAsyncFreePool(_thread_id thread_id);

// Set the main stream.
void setMainStream(cudaStream_t stream);
// Set the stream associated with the current thread.
void setStream(_thread_id thread_id, cudaStream_t stream);

// Get the stream associated with the current thread.
cudaStream_t getStream();

// Register a host pointer.
void cudaHostRegister(void *ptr, int size);

// Allocate/free pinned memory.
cudaError_t cudaMallocHost(void **ptr, size_t size);
cudaError_t cudaFreeHost(void *ptr);

cudaError_t cudaMallocAsync(void **ptr, size_t size, cudaStream_t stream = 0);
cudaError_t cudaFreeAsync(void *ptr, cudaStream_t stream = 0);

// Wait for the asynchronous frees to complete.
void cudaFreeWait();

// Join threads. ????
void joinPinnedPools();
void joinDevicePools();

// For backward compatibility
inline cudaStream_t get_stream()
{
    return amgx::memory::getStream();
}

} // namespace memory

namespace thrust
{

namespace global_thread_handle = amgx::memory; // Expose global_thread_handle in thrust.

}

} // namespace amgx
