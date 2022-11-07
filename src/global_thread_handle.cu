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

#include <global_thread_handle.h>
#include <iostream>
#include <memory>
#include <error.h>
#include <limits>
#include <vector>
#include <cassert>
#include <amgx_timer.h>
#include <algorithm>
#include <iomanip>

#if defined(_WIN32)
#include <stddef.h>
#else
#include <inttypes.h>
#endif

#define PAGE_SIZE 4096

// threshold to consider using pre-allocated pool
#define PINNED_POOL_SIZE_THRESHOLD  (100*1024*1024)

// 8 MB for pool allocations on host & device
#define PINNED_POOL_SIZE        ( 100 * 1024 * 1024)

// set that macro on if you want to see print info
// #define AMGX_PRINT_MEMORY_INFO 1
// set that macro to print the call stack for each malloc/free (it's extensive).
// #define AMGX_PRINT_MALLOC_CALL_STACK 1
// #define MULTIGPU 1


_thread_id getCurrentThreadId()
{
#ifdef WIN32
    return GetCurrentThreadId();
#else
    return pthread_self();
#endif
}

namespace amgx
{
namespace memory
{

MemoryPool::MemoryPool(size_t max_block_size, size_t page_size, size_t max_size)
    : m_size(0)
    , m_max_size(max_size)
    , m_max_block_size(max_block_size)
    , m_page_size(page_size)
    , m_free_mem(0)
    , m_used_blocks()
    , m_free_blocks()
    , m_recently_merged(false)
{
    //initializeCriticalSection(&m_mutex2);
}

MemoryPool::~MemoryPool()
{
#ifdef MULTIGPU
    int rank;
    MPI_Comm_rank( MPI_COMM_WORLD, &rank );

    if (rank == 0)
    {
#endif

        if ( !m_used_blocks.empty() )
        {
            std::cerr << "!!! detected some memory leaks in the code: trying to free non-empty temporary device pool !!!" << std::endl;

            for ( MemoryBlockListIterator it = m_used_blocks.begin() ; it != m_used_blocks.end() ; ++it )
            {
                std::cerr << "ptr: " << std::setw(18) << (void *) get_block_begin(it) << " size: " << get_block_size(it) << std::endl;
            }
        }

        //deleteCriticalSection(&m_mutex2);
#ifdef MULTIGPU
    }

#endif
}

void MemoryPool::add_memory(void *ptr, size_t size, bool managed)
{
    if (m_max_size != 0 && managed && (size + m_size > m_max_size))
    {
        FatalError("Memory pool limit is reached", AMGX_ERR_NO_MEMORY);
    }

    m_mutex2.lock();
    m_owned_ptrs.push_back(MemoryBlock(ptr, size, true, managed));
    char *aligned_ptr = (char *) ptr;

    if ( (size_t) aligned_ptr % m_page_size )
    {
        aligned_ptr = (char *) ((((size_t) aligned_ptr + m_page_size - 1) / m_page_size) * m_page_size);
    }

    size_t free_size = size - (aligned_ptr - (char *) ptr);
#ifdef AMGX_PRINT_MEMORY_INFO
    // std::cerr << "INFO: Adding memory block " << (void*) aligned_ptr << " " << free_size << std::endl;
#endif
    m_free_blocks.push_back(MemoryBlock(aligned_ptr, free_size, true, managed));
    m_size += free_size;
    m_free_mem += free_size;
    m_mutex2.unlock();
}

void *MemoryPool::allocate(size_t size, size_t &allocated_size)
{
    m_mutex2.lock();
    void *ptr = NULL;

    // Fail if the size is 0.
    if ( size == 0 )
    {
        FatalError("Allocating memory buffer of size 0!!!", AMGX_ERR_BAD_PARAMETERS);
    }

    // The memory size we are actually going to allocate.
    size_t aligned_size = m_page_size * ((size + m_page_size - 1) / m_page_size);
    // The chosen block (if any).
    MemoryBlockListIterator best_it = m_free_blocks.end();
    // The best cost (wasted amount of memory).
    size_t best_cost = std::numeric_limits<size_t>::max();
    // The address of the first correctly aligned region we're interested in.
    char *best_aligned_ptr = NULL;

    // Look for a large enough block.
    for ( MemoryBlockListIterator it = m_free_blocks.begin() ; it != m_free_blocks.end() ; ++it )
    {
#ifdef AMGX_PRINT_MEMORY_INFO
#ifdef MULTIGPU
        int rank;
        MPI_Comm_rank( MPI_COMM_WORLD, &rank );

        if (rank == 0)
#endif
        {
            std::cerr << "INFO: [block " << std::setw(18) << (void *) get_block_begin(it)
                      << " "             << std::setw(12) << get_block_size(it) << std::endl;
        }

#endif
        // Get an aligned pointer.
        char *aligned_ptr = get_block_begin(it);

        // Make sure alignments are fine. It shouldn't be needed but it's actually cheap to test.
        if ( (size_t) aligned_ptr & (m_page_size - 1) )
        {
            FatalError("INTERNAL ERROR: Invalid alignment!!!", AMGX_ERR_UNKNOWN);
        }

        // If the pointer fits in that block, just keep it.
        if ( aligned_size > get_block_size(it) )
        {
            continue;
        }

        // The cost.
        size_t cost = get_block_size(it) - aligned_size;

        // If the cost is better, keep it.
        if ( cost < best_cost )
        {
            best_it = it;
            best_cost = cost;
            best_aligned_ptr = aligned_ptr;
        }
    }

    // No block found??? Fallback to regular malloc treated outside of this function.
    if ( best_it == m_free_blocks.end() )
    {
        allocated_size = 0;
        m_mutex2.unlock();
        return ptr;
    }

    // Our allocation starts at aligned_ptr.
    ptr = best_aligned_ptr;
    // Allocated size.
    allocated_size = aligned_size;
    // Store the used block.
    MemoryBlock used_block(best_aligned_ptr, aligned_size, is_block_first(best_it));
    m_used_blocks.push_back(used_block);
    // Update statistics.
    m_free_mem -= aligned_size;
    // We store the pointer to the beginning of the block.
    char *block_begin = get_block_begin(best_it);
    // ... and its size.
    size_t block_size = get_block_size(best_it);

    // We use all the block. Simply remove it.
    if ( best_aligned_ptr == block_begin && aligned_size == block_size )
    {
        m_free_blocks.erase(best_it);
    }
    else
    {
        set_block_begin(best_it, best_aligned_ptr + aligned_size);
        set_block_size (best_it, block_size  - aligned_size);
        best_it->m_first = false;
    }

    m_mutex2.unlock();
    // fallback to regular malloc treated outside of this function
    return ptr;
}

void MemoryPool::free(void *ptr, size_t &freed_size)
{
    m_mutex2.lock();
    // Find the element to remove.
    MemoryBlockListIterator it = m_used_blocks.begin();

    for ( ; it != m_used_blocks.end() ; ++it )
        if ( get_block_begin(it) == ptr )
        {
            break;
        }

    // Sanity check.
    if ( it == m_used_blocks.end() )
    {
        FatalError("INTERNAL ERROR: Invalid iterator!!!", AMGX_ERR_UNKNOWN);
    }

    // We keep the pointers sorted. So find where to insert the new block.
    MemoryBlockListIterator insert_it = m_free_blocks.begin();

    for ( ; insert_it != m_free_blocks.end() ; ++insert_it )
    {
        // Same pointer in used and free... That's surely a bug.
        if ( get_block_begin(insert_it) == get_block_begin(it) )
        {
            FatalError("INTERNAL ERROR: Invalid memory block iterator!!! Free was called twice on same pointer.", AMGX_ERR_UNKNOWN);
        }

        if ( get_block_begin(insert_it) > get_block_begin(it) )
        {
            break;
        }
    }

    m_free_blocks.insert(insert_it, *it);
    // We merge contiguous blocks.
    MemoryBlockListIterator first = m_free_blocks.begin();
    MemoryBlockListIterator last = m_free_blocks.begin();
    char *last_ptr = get_block_begin(first) + get_block_size(first);
    size_t merged_size = get_block_size(first);
    int num_merged_blocks = 0;

    for ( ++last ; last != m_free_blocks.end() ; ++last )
    {
        if ( last_ptr != get_block_begin(last) || is_block_first(last) ) // We won't merge those two.
        {
            if ( num_merged_blocks != 0 ) // We have found the end of the block.
            {
                break;
            }

            // We have found nothing to merge... Shift the window.
            first = last;
            last_ptr = get_block_begin(first) + get_block_size(first);
            merged_size = get_block_size(first);
        }
        else
        {
            last_ptr = get_block_begin(last) + get_block_size(last);
            merged_size += get_block_size(last);
            num_merged_blocks++;
        }
    }

#ifdef AMGX_PRINT_MEMORY_INFO
#ifdef MULTIGPU
    int rank;
    MPI_Comm_rank( MPI_COMM_WORLD, &rank );

    if (rank == 0)
#endif
    {
        std::cerr << "INFO: Merging " << num_merged_blocks << " blocks" << std::endl;
    }

#endif

    if ( num_merged_blocks != 0 ) // Do the merge.
    {
        set_block_size(first, merged_size);
        first++;
        m_free_blocks.erase(first, last);
    }

    // Remove the used block and update statistics.
    m_free_mem += get_block_size(it);
    m_used_blocks.erase(it);
    //m_recently_merged = true;
    m_mutex2.unlock();
}

void MemoryPool::free_all()
{
    m_mutex2.lock();
    m_used_blocks.clear();
    m_free_blocks.clear();
    std::vector<MemoryBlock> owned_ptrs = m_owned_ptrs;
    m_owned_ptrs.clear();

    for ( size_t i = 0 ; i < owned_ptrs.size() ; ++i )
    {
        add_memory(owned_ptrs[i].m_begin, owned_ptrs[i].m_size, owned_ptrs[i].m_managed);
    }

    m_free_mem = m_size;
    m_mutex2.unlock();
}

bool MemoryPool::is_allocated(void *ptr)
{
    m_mutex2.lock();

    for ( MemoryBlockListConstIterator it = m_used_blocks.begin() ; it != m_used_blocks.end() ; ++it )
        if ( it->m_begin == ptr )
        {
            m_mutex2.unlock();
            return true;
        }

    m_mutex2.unlock();
    return false;
}

PinnedMemoryPool::PinnedMemoryPool()
    : MemoryPool(PINNED_POOL_SIZE_THRESHOLD, 4096, 0)
{
    void *ptr = NULL;
    ::cudaMallocHost(&ptr, PINNED_POOL_SIZE);

    if ( ptr == NULL )
    {
        FatalError("Cannot allocate pinned memory", AMGX_ERR_NO_MEMORY);
    }

    add_memory(ptr, PINNED_POOL_SIZE);
}

PinnedMemoryPool::~PinnedMemoryPool()
{
    for ( size_t i = 0 ; i < m_owned_ptrs.size() ; ++i )
        if (m_owned_ptrs[i].m_managed)
        {
            ::cudaFreeHost(m_owned_ptrs[i].m_begin);
        }

    m_owned_ptrs.clear();
}


DeviceMemoryPool::DeviceMemoryPool(size_t size,
                                   size_t max_block_size,
                                   size_t max_size)
    : MemoryPool(max_block_size, 4096, max_size)
{
    if (max_size > 0 && size > max_size)
    {
        FatalError("Initial size for the memory pool specified is more than memory limit", AMGX_ERR_NO_MEMORY);
    }

    void *ptr = NULL;
    ::cudaMalloc(&ptr, size);

    if ( ptr == NULL )
    {
        FatalError("Cannot allocate device memory", AMGX_ERR_NO_MEMORY);
    }

    add_memory(ptr, size);
}

void DeviceMemoryPool::expandPool(size_t size,
                                  size_t max_block_size)
{
    if (this->m_max_size > 0 && (size + this->m_size) > this->m_max_size)
    {
        FatalError("Pool memory size is exceeded.", AMGX_ERR_NO_MEMORY);
    }

    void *ptr = NULL;
    ::cudaMalloc(&ptr, size);

    if ( ptr == NULL )
    {
        FatalError("Cannot allocate device memory", AMGX_ERR_NO_MEMORY);
    }

    add_memory(ptr, size);
}


DeviceMemoryPool::~DeviceMemoryPool()
{
    for ( size_t i = 0 ; i < m_owned_ptrs.size() ; ++i )
        if (m_owned_ptrs[i].m_managed)
        {
            ::cudaFree(m_owned_ptrs[i].m_begin);
        }

    m_owned_ptrs.clear();
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////

struct MemoryManager
{
    // Get the global instance.
    static MemoryManager &get_instance()
    {
        static MemoryManager s_instance;
        return s_instance;
    }

    // Ctor.
    MemoryManager()
        : m_main_pinned_pool(NULL)
        , m_main_device_pool(NULL)
        , m_use_async_free(false)
        , m_use_device_pool(false)
        , m_alloc_scaling_factor(0)
        , m_alloc_scaling_threshold(16 * 1024 * 1024)
    {
        //initializeCriticalSection(&m_mutex);

        cudaDeviceProp devProp;
        cudaGetDeviceProperties(&devProp, 0);
        this->smCount = devProp.multiProcessorCount;
    }

    // Dtor.
    ~MemoryManager()
    {
        //deleteCriticalSection(&m_mutex);
    }

    // Synchronize a device pool.
    void sync_pinned_pool(PinnedMemoryPool *pool);
    void sync_device_pool(DeviceMemoryPool *pool);

    // Scale a memory size.
    size_t scale(size_t size) const
    {
        size_t new_size = size;

        if ( size >= m_alloc_scaling_threshold )
        {
            new_size += m_alloc_scaling_factor * (size / 100);
        }

        return new_size;
    }

    // Mutex to make functions thread-safe.
    std::recursive_mutex m_mutex;

    // Streams.
    typedef std::map<_thread_id, cudaStream_t> StreamMap;
    StreamMap m_thread_stream;
    cudaStream_t m_main_stream;

    // Items to free (async free).
    // typedef std::map<_thread_id, std::vector<void*> > AsyncFreeMap;
    // AsyncFreeMap m_thread_free;
    // std::vector<void*> m_main_free;

    // Pinned pools.
    typedef std::map<_thread_id, PinnedMemoryPool *> PinnedPoolMap;
    PinnedPoolMap m_thread_pinned_pools;
    PinnedMemoryPool *m_main_pinned_pool;

    // Device pools.
    typedef std::map<_thread_id, DeviceMemoryPool *> DevicePoolMap;
    DevicePoolMap m_thread_device_pools;
    DeviceMemoryPool *m_main_device_pool;

    // Registered memory blocks.
    typedef std::vector<std::pair<void *, void *> > RegisteredBlocks;
    typedef std::map<_thread_id, RegisteredBlocks> RegisteredBlocksMap;
    RegisteredBlocksMap m_thread_registered;
    RegisteredBlocks m_main_registered;

    // We keep a list of allocations that go through cudaMalloc.
    typedef std::map<void *, size_t> MemoryBlockMap;
    MemoryBlockMap m_allocated_blocks;

    // whether we want to use async free/wait or regular free.
    bool m_use_async_free;
    // whether we want to use device pool or simply do regular malloc.
    bool m_use_device_pool;

    // Scaling factor.
    size_t m_alloc_scaling_factor;
    // Scaling threshold.
    size_t m_alloc_scaling_threshold;

    int smCount;
};

void MemoryManager::sync_pinned_pool(PinnedMemoryPool *pool)
{
    MemoryPool *mem_pool = (MemoryPool *) pool;
    assert(mem_pool);
    MemoryPool *main_pool = (MemoryPool *) m_main_pinned_pool;
    main_pool->m_used_blocks.insert(main_pool->m_used_blocks.end(),
                                    mem_pool->get_used_begin(),
                                    mem_pool->get_used_end());
    mem_pool->free_all();
}

void MemoryManager::sync_device_pool(DeviceMemoryPool *pool)
{
    MemoryPool *mem_pool = (MemoryPool *) pool;
    assert(mem_pool);
    MemoryPool *main_pool = (MemoryPool *) m_main_device_pool;
    main_pool->m_used_blocks.insert(main_pool->m_used_blocks.end(),
                                    mem_pool->get_used_begin(),
                                    mem_pool->get_used_end());
    mem_pool->free_all();
}

bool hasPinnedMemoryPool()
{
    MemoryManager &manager = MemoryManager::get_instance();
    return manager.m_main_pinned_pool != NULL;
}

bool hasDeviceMemoryPool()
{
    MemoryManager &manager = MemoryManager::get_instance();
    return manager.m_main_device_pool != NULL;
}

void setPinnedMemoryPool(PinnedMemoryPool *pool)
{
    MemoryManager &manager = MemoryManager::get_instance();
    manager.m_mutex.lock();
    manager.m_main_pinned_pool = pool;
    manager.m_mutex.unlock();
}

void setDeviceMemoryPool(DeviceMemoryPool *pool)
{
    MemoryManager &manager = MemoryManager::get_instance();
    manager.m_mutex.lock();
    manager.m_main_device_pool = pool;
    manager.m_mutex.unlock();
}

void setPinnedMemoryPool(_thread_id thread_id, PinnedMemoryPool *pool)
{
    MemoryManager &manager = MemoryManager::get_instance();
    manager.m_mutex.lock();
    manager.m_thread_pinned_pools[thread_id] = pool;
    manager.m_mutex.unlock();
}

void setDeviceMemoryPool(_thread_id thread_id, DeviceMemoryPool *pool)
{
    MemoryManager &manager = MemoryManager::get_instance();
    manager.m_mutex.lock();
    manager.m_thread_device_pools[thread_id] = pool;
    manager.m_mutex.unlock();
}

void destroyPinnedMemoryPool()
{
    MemoryManager &manager = MemoryManager::get_instance();
    manager.m_mutex.lock();
    delete manager.m_main_pinned_pool;
    manager.m_main_pinned_pool = NULL;
    manager.m_mutex.unlock();
}

void destroyDeviceMemoryPool()
{
    MemoryManager &manager = MemoryManager::get_instance();
    manager.m_mutex.lock();
    delete manager.m_main_device_pool;
    manager.m_main_device_pool = NULL;
    manager.m_mutex.unlock();
}

void destroyPinnedMemoryPool(_thread_id thread_id)
{
    MemoryManager &manager = MemoryManager::get_instance();
    manager.m_mutex.lock();
    MemoryManager::PinnedPoolMap::iterator it = manager.m_thread_pinned_pools.find(thread_id);

    if ( it == manager.m_thread_pinned_pools.end() )
    {
        FatalError("INTERNAL ERROR: Invalid pinned memory pool", AMGX_ERR_UNKNOWN);
    }

    delete it->second;
    manager.m_thread_pinned_pools.erase(it);
    manager.m_mutex.unlock();
}

void destroyDeviceMemoryPool(_thread_id thread_id)
{
    MemoryManager &manager = MemoryManager::get_instance();
    manager.m_mutex.lock();
    MemoryManager::DevicePoolMap::iterator it = manager.m_thread_device_pools.find(thread_id);

    if ( it == manager.m_thread_device_pools.end() )
    {
        FatalError("INTERNAL ERROR: Invalid device memory pool", AMGX_ERR_UNKNOWN);
    }

    delete it->second;
    manager.m_thread_device_pools.erase(it);
    manager.m_mutex.unlock();
}

void destroyAllPinnedMemoryPools()
{
    MemoryManager &manager = MemoryManager::get_instance();
    manager.m_mutex.lock();
    MemoryManager::PinnedPoolMap::iterator it = manager.m_thread_pinned_pools.begin();

    for ( ; it != manager.m_thread_pinned_pools.end() ; ++it )
    {
        delete it->second;
        manager.m_thread_pinned_pools.erase(it);
    }

    destroyPinnedMemoryPool();
    manager.m_mutex.unlock();
}

void destroyAllDeviceMemoryPools()
{
    MemoryManager &manager = MemoryManager::get_instance();
    manager.m_mutex.lock();
    MemoryManager::DevicePoolMap::iterator it = manager.m_thread_device_pools.begin();

    for ( ; it != manager.m_thread_device_pools.end() ; ++it )
    {
        delete it->second;
        manager.m_thread_device_pools.erase(it);
    }

    destroyDeviceMemoryPool();
    manager.m_mutex.unlock();
}

void setAsyncFreeFlag(bool set)
{
    MemoryManager &manager = MemoryManager::get_instance();
    manager.m_use_async_free = set;
}

void setDeviceMemoryPoolFlag(bool set)
{
    MemoryManager &manager = MemoryManager::get_instance();
    manager.m_use_device_pool = set;
}

void setMallocScalingFactor(size_t factor)
{
    MemoryManager &manager = MemoryManager::get_instance();
    manager.m_alloc_scaling_factor = factor;
}

void setMallocScalingThreshold(size_t threshold)
{
    MemoryManager &manager = MemoryManager::get_instance();
    manager.m_alloc_scaling_threshold = threshold;
}

void createAsyncFreePool(_thread_id thread_id)
{
    /*
    MemoryManager &manager = MemoryManager::get_instance();

    manager.m_mutex.lock();
      manager.m_thread_free[thread_id] = std::vector<void*>();
    manager.m_mutex.unlock(); */
}

void setMainStream(cudaStream_t stream)
{
    MemoryManager &manager = MemoryManager::get_instance();
    manager.m_main_stream = stream;
}

void setStream(_thread_id thread_id, cudaStream_t stream)
{
    MemoryManager &manager = MemoryManager::get_instance();
    manager.m_mutex.lock();
    manager.m_thread_stream[thread_id] = stream;
    manager.m_mutex.unlock();
}

cudaStream_t getStream()
{
    MemoryManager &manager = MemoryManager::get_instance();
    _thread_id thread_id = getCurrentThreadId();
    MemoryManager::StreamMap::iterator it = manager.m_thread_stream.find(thread_id);

    if ( it != manager.m_thread_stream.end() )
    {
        return it->second;
    }

    return manager.m_main_stream;
}

int getSMCount()
{
    MemoryManager &manager = MemoryManager::get_instance();
    return manager.smCount;
}

void cudaHostRegister(void *ptr, int size)
{
    MemoryManager &manager = MemoryManager::get_instance();
    _thread_id thread_id = getCurrentThreadId();
    MemoryManager::RegisteredBlocks *blocks = &manager.m_main_registered;
    MemoryManager::RegisteredBlocksMap::iterator it = manager.m_thread_registered.find(thread_id);

    if ( it != manager.m_thread_registered.end() )
    {
        blocks = &it->second;
    }

    bool reg = true;

    for ( size_t i = 0; i < blocks->size() ; ++i )
        if ( blocks->at(i).first <= ptr && blocks->at(i).second >= ptr)
        {
            reg = false;
            break;
        }

    if ( reg )
    {
        ::cudaHostRegister(ptr, size, 0);
        blocks->push_back(std::pair<void *, void *>(ptr, (char *)ptr + size));
    }
}

cudaError_t cudaMallocHost(void **ptr, size_t size)
{
    MemoryManager &manager = MemoryManager::get_instance();
    _thread_id thread_id = getCurrentThreadId();
    PinnedMemoryPool *pool = manager.m_main_pinned_pool;
    MemoryManager::PinnedPoolMap::iterator it = manager.m_thread_pinned_pools.find(thread_id);

    if ( it != manager.m_thread_pinned_pools.end() )
    {
        pool = it->second;
    }

    size_t allocated_size = 0;
    cudaError_t error = cudaSuccess;
    void *new_ptr = NULL;

    if ( pool != NULL && size < PINNED_POOL_SIZE_THRESHOLD )
    {
        new_ptr = pool->allocate(size, allocated_size);
    }

    if ( pool != NULL && new_ptr == NULL && size < PINNED_POOL_SIZE_THRESHOLD ) // retry with size
    {
        new_ptr = pool->allocate(size, allocated_size);
    }

    if ( new_ptr != NULL )
    {
        *ptr = new_ptr;
    }
    else
    {
        //printf("calling cudaMallocHost, size = %lu\n",size);
        error = ::cudaMallocHost(ptr, size);
    }

    return error;
}

cudaError_t cudaFreeHost(void *ptr)
{
    MemoryManager &manager = MemoryManager::get_instance();
    _thread_id thread_id = getCurrentThreadId();
    PinnedMemoryPool *pool = manager.m_main_pinned_pool;
    MemoryManager::PinnedPoolMap::iterator it = manager.m_thread_pinned_pools.find(thread_id);

    if ( it != manager.m_thread_pinned_pools.end() )
    {
        pool = it->second;
    }

    size_t freed_size = 0;
    cudaError_t error = cudaSuccess;

    if ( pool != NULL && pool->is_allocated(ptr) )
    {
        pool->free(ptr, freed_size);
    }
    else
    {
//printf("calling cudaFreeHost\n");
        error = ::cudaFreeHost(ptr);
    }

    return error;
}

cudaError_t cudaMalloc(void **ptr, size_t size)
{
    AMGX_CPU_PROFILER("cudaMalloc");
#ifdef AMGX_PRINT_MALLOC_CALL_STACK
#ifdef MULTIGPU
    int rank;
    MPI_Comm_rank( MPI_COMM_WORLD, &rank );

    if (rank == 0)
#endif
    {
        std::cerr << "----" << std::endl;
        std::cerr << "cudaMalloc call stack:" << std::endl;
        printStackTrace(std::cerr);
    }

#endif
    MemoryManager &manager = MemoryManager::get_instance();
    DeviceMemoryPool *pool = manager.m_main_device_pool;
    _thread_id thread_id = getCurrentThreadId();
    MemoryManager::DevicePoolMap::iterator it = manager.m_thread_device_pools.find(thread_id);

    if ( it != manager.m_thread_device_pools.end() )
    {
        pool = it->second;
    }

    bool use_pool = manager.m_use_device_pool;
#ifdef AMGX_PRINT_MEMORY_INFO
    bool print_fallback = false;
#endif
    size_t allocated_size = 0;
    cudaError_t error = cudaSuccess;
    void *new_ptr = NULL;

    if ( pool != NULL /*&& size < pool->get_max_block_size()*/ && use_pool )
    {
        new_ptr = pool->allocate(size, allocated_size);
    }

    if ( new_ptr != NULL )
    {
        *ptr = new_ptr;
    }
    else
    {
#ifdef AMGX_PRINT_MEMORY_INFO
        print_fallback = true;
#endif
        // We allocate an extra fraction here.
        allocated_size = manager.scale(size);
        // We hack the size to make it a multiple of a page size.
        allocated_size = PAGE_SIZE * ((allocated_size + PAGE_SIZE - 1) / PAGE_SIZE);
        error = ::cudaMalloc(ptr, allocated_size);

        // Very last attempt. Try without over allocation.
        if ( *ptr == NULL )
        {
            allocated_size = size;
            error = ::cudaMalloc(ptr, allocated_size);
        }

        manager.m_mutex.lock();
        manager.m_allocated_blocks[*ptr] = allocated_size;
        manager.m_mutex.unlock();
#ifdef AMGX_PRINT_MEMORY_INFO
#ifdef MULTIGPU

        if (rank == 0)
#endif
        {
            std::cerr << "INFO: Registered [block " << std::setw(18) << *ptr << " size: " << allocated_size << "]" << std::endl;
        }

#endif
    }

#ifdef AMGX_PRINT_MEMORY_INFO
#ifdef MULTIGPU

    if (rank == 0)
#endif
    {
        if ( print_fallback )
        {
            std::cerr << "cudaMalloc    ";
        }
        else
        {
            std::cerr << "pool::allocate";
        }

        std::cerr << ";" << std::setw(18) << *ptr
                  << ";" << std::setw(12) << size
                  << ";" << std::setw(12) << allocated_size
                  << ";" << std::setw(12) << pool->get_used_mem()
                  << ";" << std::setw(12) << pool->get_free_mem();
        size_t gpu_free_mem, gpu_total_mem;
        cudaMemGetInfo(&gpu_free_mem, &gpu_total_mem);
        std::cerr << ";" << std::setw(12) << gpu_free_mem
                  << ";" << std::setw(12) << gpu_total_mem;
        std::cerr << std::endl;
    }

#endif
    return error;
}

cudaError_t cudaFreeAsync(void *ptr)
{
    AMGX_CPU_PROFILER("cudaFreeAsync");
#ifdef AMGX_PRINT_MALLOC_CALL_STACK
#ifdef MULTIGPU
    int rank;
    MPI_Comm_rank( MPI_COMM_WORLD, &rank );

    if (rank == 0)
#endif
    {
        std::cerr << "----" << std::endl;
        std::cerr << "cudaFreeAsync call stack:" << std::endl;
        printStackTrace(std::cerr);
    }

#endif

    // We accept NULL pointers and we do nothing.
    if ( ptr == NULL )
    {
        return cudaSuccess;
    }

    MemoryManager &manager = MemoryManager::get_instance();
    _thread_id thread_id = getCurrentThreadId();
#ifdef AMGX_PRINT_MEMORY_INFO
    bool print_async = false, print_fallback = false;
#endif
    DeviceMemoryPool *pool = manager.m_main_device_pool;
    size_t freed_size = 0;
    cudaError_t status = cudaSuccess;
    MemoryManager::DevicePoolMap::iterator it_pool = manager.m_thread_device_pools.find(thread_id);

    if ( it_pool != manager.m_thread_device_pools.end() )
    {
        pool = manager.m_thread_device_pools[thread_id];
    }

    if ( pool != NULL && pool->is_allocated(ptr) )
    {
        pool->free(ptr, freed_size);
    }
    else if ( pool != NULL && manager.m_use_async_free )
    {
#ifdef AMGX_PRINT_MEMORY_INFO
#ifdef MULTIGPU

        if (rank == 0)
#endif
        {
            print_async = true;
            std::cerr << "INFO: Async free [ptr " << std::setw(18) << ptr << "]" << std::endl;
        }

#endif
        MemoryManager::MemoryBlockMap::iterator ptr_it = manager.m_allocated_blocks.find(ptr);

        if ( ptr_it == manager.m_allocated_blocks.end() )
        {
            FatalError("INTERNAL ERROR: Invalid call to cudaFreeAsync", AMGX_ERR_UNKNOWN);
        }

        pool->add_memory(ptr, ptr_it->second);
        manager.m_mutex.lock();
        manager.m_allocated_blocks.erase(ptr_it);
        manager.m_mutex.unlock();
    }
    else
    {
#ifdef AMGX_PRINT_MEMORY_INFO
        print_fallback = true;
#endif
        status = ::cudaFree(ptr);
    }

#ifdef AMGX_PRINT_MEMORY_INFO
#ifdef MULTIGPU

    if (rank == 0)
#endif
    {
        if ( print_fallback )
        {
            std::cerr << "cudaFree      ";
        }
        else if ( print_async )
        {
            std::cerr << "pool::async   ";
        }
        else
        {
            std::cerr << "pool::free    ";
        }

        std::cerr << ";" << std::setw(18) << ptr
                  << ";" << std::setw(12) << freed_size
                  << ";" << std::setw(12) << pool->get_used_mem()
                  << ";" << std::setw(12) << pool->get_free_mem();
        size_t gpu_free_mem, gpu_total_mem;
        cudaMemGetInfo(&gpu_free_mem, &gpu_total_mem);
        std::cerr << ";" << std::setw(12) << gpu_free_mem
                  << ";" << std::setw(12) << gpu_total_mem;
        std::cerr << std::endl;
    }

#endif
    return status;
}

void cudaFreeWait()
{
}

// Join device pools
void joinPinnedPools()
{
    MemoryManager &manager = MemoryManager::get_instance();
    typedef MemoryManager::PinnedPoolMap::iterator Iterator;
    Iterator it  = manager.m_thread_pinned_pools.begin();
    Iterator end = manager.m_thread_pinned_pools.end();

    for ( ; it != end ; ++it )
    {
        manager.sync_pinned_pool(it->second);
    }
}

void joinDevicePools()
{
    MemoryManager &manager = MemoryManager::get_instance();
    typedef MemoryManager::DevicePoolMap::iterator Iterator;
    Iterator it  = manager.m_thread_device_pools.begin();
    Iterator end = manager.m_thread_device_pools.end();

    for ( ; it != end ; ++it )
    {
        manager.sync_device_pool(it->second);
    }
}

void printInfo()
{
    //
}

void expandDeviceMemoryPool(size_t size, size_t max_block_size)
{
    MemoryManager &manager = MemoryManager::get_instance();

    if (manager.m_main_device_pool)
    {
        manager.m_main_device_pool->expandPool(size, max_block_size);
    }
}

} // namespace memory
} // namespace amgx

