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

#include <hash_index.inl>

#include <amgx_types/util.h>

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

static __constant__ unsigned c_hash_keys[] =
{
    3499211612,  581869302, 3890346734, 3586334585,
    545404204,  4161255391, 3922919429,  949333985,
    2715962298, 1323567403,  418932835, 2350294565,
    1196140740,  809094426, 2348838239, 4264392720
};

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template< typename Key_type, int SMEM_SIZE = 128, int NUM_HASH_FCTS = 4, int WARP_SIZE = 32 >
class Hash_set
{
        // Associated index.
        typedef Hash_index<Key_type, SMEM_SIZE, WARP_SIZE> Index;

    protected:
        // The size of the table (occupancy).
        int m_smem_count, m_gmem_count;
        // The keys stored in the hash table.
        volatile Key_type *m_smem_keys, *m_gmem_keys;
        // The size of the global memory buffer.
        const int m_gmem_size;
        // Is it ok?
        bool m_fail;

    public:
        // Constructor.
        __device__ __forceinline__ Hash_set( volatile Key_type *smem_keys, volatile Key_type *gmem_keys, int gmem_size ) :
            m_smem_count(0),
            m_gmem_count(1),
            m_smem_keys (smem_keys),
            m_gmem_keys (gmem_keys),
            m_gmem_size (gmem_size),
            m_fail      (false)
        {}

        // Clear the table.
        __device__ __forceinline__ void clear( bool skip_gmem = false );
        // Compute the size of the table. Only thread with lane_id==0 gives the correct result (no broadcast of the value).
        __device__ __forceinline__ int compute_size();
        // Compute the size of the table. Only thread with lane_id==0 gives the correct result (no broadcast of the value).
        __device__ __forceinline__ int compute_size_with_duplicates();
        // Does the set contain those values?
        __device__ __forceinline__ bool contains( Key_type key ) const;
        // Find an index.
        __device__ __forceinline__ int find_index( Key_type key, const Index &index, bool print_debug ) const;
        // Has the process failed.
        __device__ __forceinline__ bool has_failed() const { return m_fail; }
        // Insert a key/value inside the hash table.
        __device__ __forceinline__ void insert( Key_type key, int *status );
        // Load the elements of a set.
        __device__ __forceinline__ void load( int count, const Key_type *keys, const int *pos );
        // Load a set and use it as an index.
        __device__ __forceinline__ void load_index( int count, const Key_type *keys, const int *pos, Index &index, bool print_debug );
        // Store a set.
        __device__ __forceinline__ void store( int count, Key_type *keys );
        // Store a set.
        __device__ __forceinline__ int  store_with_positions( Key_type *keys, int *pos );
        // Store a set.
        __device__ __forceinline__ int  store( Key_type *keys );
};

// ====================================================================================================================

template< typename Key_type, int SMEM_SIZE, int NUM_HASH_FCTS, int WARP_SIZE >
__device__ __forceinline__
void Hash_set<Key_type, SMEM_SIZE, NUM_HASH_FCTS, WARP_SIZE>::clear( bool skip_gmem )
{
    int lane_id = utils::lane_id();
    const int NUM_STEPS = SMEM_SIZE / WARP_SIZE;
#pragma unroll

    for ( int i_step = 0 ; i_step < NUM_STEPS ; ++i_step )
    {
        m_smem_keys[i_step * WARP_SIZE + lane_id] = -1;
    }

    m_smem_count = 0;

    if ( skip_gmem || m_gmem_count == 0 )
    {
        m_gmem_count = 0;
        return;
    }

#pragma unroll 4

    for ( int offset = lane_id ; offset < m_gmem_size ; offset += WARP_SIZE )
    {
        m_gmem_keys[offset] = -1;
    }

    m_gmem_count = 0;
}

// ====================================================================================================================

template< typename Key_type, int SMEM_SIZE, int NUM_HASH_FCTS, int WARP_SIZE >
__device__ __forceinline__
int Hash_set<Key_type, SMEM_SIZE, NUM_HASH_FCTS, WARP_SIZE>::compute_size()
{
    m_smem_count += m_gmem_count;
    const int lane_id = utils::lane_id();
    m_smem_keys[lane_id] = m_smem_count;
#pragma unroll

    for ( int offset = WARP_SIZE / 2 ; offset > 0 ; offset >>= 1 )
        if ( lane_id < offset )
        {
            m_smem_keys[lane_id] = m_smem_count += m_smem_keys[lane_id + offset];
        }

    m_gmem_count = utils::any( m_gmem_count > 0 );
    return m_smem_count;
}

// ====================================================================================================================

template< typename Key_type, int SMEM_SIZE, int NUM_HASH_FCTS, int WARP_SIZE >
__device__ __forceinline__
int Hash_set<Key_type, SMEM_SIZE, NUM_HASH_FCTS, WARP_SIZE>::compute_size_with_duplicates()
{
    int lane_id = utils::lane_id();
    // Count the number of keys in SMEM.
    int sum = 0;
    const int NUM_STEPS = SMEM_SIZE / WARP_SIZE;
#pragma unroll

    for ( int i_step = 0 ; i_step < NUM_STEPS ; ++i_step )
    {
        const int offset = i_step * WARP_SIZE + lane_id;
        Key_type key = m_smem_keys[offset];
        sum += __popc( utils::ballot( key != -1 ) );
    }

    // Is there any key in GMEM. If not, just quit.
    m_gmem_count = utils::any(m_gmem_count > 0);

    if ( !m_gmem_count )
    {
        return sum;
    }

    // Count the number of keys in GMEM.
#pragma unroll 4

    for ( int offset = lane_id ; offset < m_gmem_size ; offset += WARP_SIZE )
    {
        Key_type key = m_gmem_keys[offset];
        sum += __popc( utils::ballot( key != -1, utils::activemask() ) );
    }

    return sum;
}

// ====================================================================================================================

template< typename Key_type, int SMEM_SIZE, int NUM_HASH_FCTS, int WARP_SIZE >
__device__ __forceinline__
bool Hash_set<Key_type, SMEM_SIZE, NUM_HASH_FCTS, WARP_SIZE>::contains( Key_type key ) const
{
    bool done = key == -1, found = false;
#pragma unroll

    for ( int i_hash = 0 ; i_hash < NUM_HASH_FCTS ; ++i_hash )
    {
        if ( utils::all(done) )
        {
            return found;
        }

        unsigned ukey = reinterpret_cast<unsigned &>( key );
        int hash = ( (ukey ^ c_hash_keys[i_hash]) + c_hash_keys[NUM_HASH_FCTS + i_hash] ) & (SMEM_SIZE - 1);

        if ( !done )
        {
            Key_type stored_key = m_smem_keys[hash];

            if ( stored_key == key )
            {
                found = true;
            }

            if ( found || stored_key == -1 )
            {
                done = true;
            }
        }
    }

    const int num_bits = utils::bfind( m_gmem_size ); // TODO: move it outside ::insert.
#pragma unroll

    for ( int i_hash = 0 ; i_hash < NUM_HASH_FCTS ; ++i_hash )
    {
        if ( utils::all(done) )
        {
            return found;
        }

        unsigned ukey = reinterpret_cast<unsigned &>( key );
        int hash = utils::bfe( (ukey ^ c_hash_keys[i_hash]) + c_hash_keys[NUM_HASH_FCTS + i_hash], num_bits );

        if ( !done )
        {
            Key_type stored_key = m_gmem_keys[hash];

            if ( stored_key == key )
            {
                found = true;
            }

            if ( found || stored_key == -1 )
            {
                done = true;
            }
        }
    }

    return found;
}

// ====================================================================================================================

template< typename Key_type, int SMEM_SIZE, int NUM_HASH_FCTS, int WARP_SIZE >
__device__ __forceinline__
int Hash_set<Key_type, SMEM_SIZE, NUM_HASH_FCTS, WARP_SIZE>::find_index( Key_type key, const Index &index, bool print_debug ) const
{
    int idx = -1;
    bool done = key == -1, found = false;
#pragma unroll

    for ( int i_hash = 0 ; i_hash < NUM_HASH_FCTS ; ++i_hash )
    {
        if ( utils::all(done) )
        {
            return idx;
        }

        unsigned ukey = reinterpret_cast<unsigned &>( key );
        int hash = ( (ukey ^ c_hash_keys[i_hash]) + c_hash_keys[NUM_HASH_FCTS + i_hash] ) & (SMEM_SIZE - 1);
        int result = index.find_smem(hash);

        if ( !done )
        {
            Key_type stored_key = m_smem_keys[hash];

            if ( stored_key == key )
            {
                idx = result;
                done = true;
            }
        }
    }

    const int num_bits = utils::bfind( m_gmem_size ); // TODO: move it outside ::insert.
#pragma unroll

    for ( int i_hash = 0 ; i_hash < NUM_HASH_FCTS ; ++i_hash )
    {
        if ( utils::all(done) )
        {
            return idx;
        }

        unsigned ukey = reinterpret_cast<unsigned &>( key );
        int hash = utils::bfe( (ukey ^ c_hash_keys[i_hash]) + c_hash_keys[NUM_HASH_FCTS + i_hash], num_bits );

        if ( !done )
        {
            Key_type stored_key = m_gmem_keys[hash];

            if ( stored_key == key )
            {
                idx = index.find_gmem(hash);
                found = true;
            }

            if ( found || stored_key == -1 )
            {
                done = true;
            }
        }
    }

    return idx;
}

// ====================================================================================================================

template< typename Key_type, int SMEM_SIZE, int NUM_HASH_FCTS, int WARP_SIZE >
__device__ __forceinline__
void Hash_set<Key_type, SMEM_SIZE, NUM_HASH_FCTS, WARP_SIZE>::insert( Key_type key, int *status )
{
    bool done = key == -1;
#pragma unroll

    for ( int i_hash = 0 ; i_hash < NUM_HASH_FCTS ; ++i_hash )
    {
        if ( utils::all(done) )
        {
            return;
        }

        bool candidate = false;
        unsigned ukey = reinterpret_cast<unsigned &>( key );
        int hash = ( (ukey ^ c_hash_keys[i_hash]) + c_hash_keys[NUM_HASH_FCTS + i_hash] ) & (SMEM_SIZE - 1);

        if ( !done )
        {
            Key_type stored_key = m_smem_keys[hash];

            if ( stored_key == key )
            {
                done = true;
            }

            candidate = stored_key == -1;

            if ( candidate )
            {
                m_smem_keys[hash] = key;
            }

            if ( candidate && key == m_smem_keys[hash] ) // More than one candidate may have written to that slot.
            {
                m_smem_count++;
                done = true;
            }
        }
    }

    const int num_bits = utils::bfind( m_gmem_size ); // TODO: move it outside ::insert.
#pragma unroll

    for ( int i_hash = 0 ; i_hash < NUM_HASH_FCTS ; ++i_hash )
    {
        if ( utils::all(done) )
        {
            return;
        }

        bool candidate = false;
        unsigned ukey = reinterpret_cast<unsigned &>( key );
        int hash = utils::bfe( (ukey ^ c_hash_keys[i_hash]) + c_hash_keys[NUM_HASH_FCTS + i_hash], num_bits );

        if ( !done )
        {
            Key_type stored_key = m_gmem_keys[hash];

            if ( stored_key == key )
            {
                done = true;
            }

            candidate = stored_key == -1;

            if ( candidate )
            {
                m_gmem_keys[hash] = key;
            }

            if ( candidate && key == m_gmem_keys[hash] ) // More than one candidate may have written to that slot.
            {
                m_gmem_count++;
                done = true;
            }
        }
    }

    if ( utils::all(done) )
    {
        return;
    }

    if ( utils::lane_id() == 0 )
    {
        *status = 1;
    }

    m_fail = true;
}

// ====================================================================================================================

template< typename Key_type, int SMEM_SIZE, int NUM_HASH_FCTS, int WARP_SIZE >
__device__ __forceinline__
void Hash_set<Key_type, SMEM_SIZE, NUM_HASH_FCTS, WARP_SIZE>::load( int count, const Key_type *keys, const int *pos )
{
    int lane_id = utils::lane_id();
#pragma unroll 4

    for ( int offset = lane_id ; offset < count ; offset += WARP_SIZE )
    {
        Key_type key = keys[offset];
        int idx = pos [offset];
        // Where to store the item.
        volatile int *ptr = m_smem_keys;

        if ( idx >= SMEM_SIZE )
        {
            ptr = m_gmem_keys;
            m_gmem_count = 1;
            idx -= SMEM_SIZE;
        }

        // Store the item.
        ptr[idx] = key;
    }

    m_gmem_count = utils::any( m_gmem_count );
}

// ====================================================================================================================

template< typename Key_type, int SMEM_SIZE, int NUM_HASH_FCTS, int WARP_SIZE >
__device__ __forceinline__
void Hash_set<Key_type, SMEM_SIZE, NUM_HASH_FCTS, WARP_SIZE>::load_index( int count, const Key_type *keys, const int *pos, Index &index, bool print_debug )
{
#pragma unroll 4

    for ( int offset = utils::lane_id() ; offset < count ; offset += WARP_SIZE  )
    {
        Key_type key = keys[offset];
        int idx = pos [offset];

        // Store the item.
        if ( idx < SMEM_SIZE )
        {
            m_smem_keys[idx] = key;
        }
        else
        {
            m_gmem_keys[idx - SMEM_SIZE] = key;
            index.set_gmem_index( idx - SMEM_SIZE, offset );
            m_gmem_count = 1;
        }
    }

    // Build the local index.
    index.build_smem_index( m_smem_keys );
    m_gmem_count = utils::any( m_gmem_count );
}

// ====================================================================================================================

template< typename Key_type, int SMEM_SIZE, int NUM_HASH_FCTS, int WARP_SIZE >
__device__ __forceinline__
void Hash_set<Key_type, SMEM_SIZE, NUM_HASH_FCTS, WARP_SIZE>::store( int count, Key_type *keys )
{
    int lane_id = utils::lane_id();
    int lane_mask_lt = utils::lane_mask_lt();
    int warp_offset = 0;
    const int NUM_STEPS = SMEM_SIZE / WARP_SIZE;
#pragma unroll

    for ( int i_step = 0 ; i_step < NUM_STEPS ; ++i_step )
    {
        const int offset = i_step * WARP_SIZE + lane_id;
        Key_type key = m_smem_keys[offset];
        int poll = utils::ballot( key != -1 );

        if ( poll == 0 )
        {
            continue;
        }

        int dst_offset = warp_offset + __popc( poll & lane_mask_lt );

        if ( key != -1 )
        {
            keys[dst_offset] = key;
        }

        warp_offset += __popc( poll );
    }

    m_gmem_count = utils::any( m_gmem_count > 0 );

    if ( !m_gmem_count )
    {
        return;
    }

#pragma unroll 4

    for ( int offset = lane_id ; offset < m_gmem_size ; offset += WARP_SIZE )
    {
        Key_type key = m_gmem_keys[offset];
        int poll = utils::ballot( key != -1, utils::activemask() );

        if ( poll == 0 )
        {
            continue;
        }

        int dst_offset = warp_offset + __popc( poll & lane_mask_lt );

        if ( key != -1 )
        {
            keys[dst_offset] = key;
        }

        warp_offset += __popc( poll );
    }
}

// ====================================================================================================================

template< typename Key_type, int SMEM_SIZE, int NUM_HASH_FCTS, int WARP_SIZE >
__device__ __forceinline__
int Hash_set<Key_type, SMEM_SIZE, NUM_HASH_FCTS, WARP_SIZE>::store_with_positions( Key_type *keys, int *pos )
{
    int lane_id = utils::lane_id();
    int lane_mask_lt = utils::lane_mask_lt();
    int warp_offset = 0;
    const int NUM_STEPS = SMEM_SIZE / WARP_SIZE;
#pragma unroll

    for ( int i_step = 0 ; i_step < NUM_STEPS ; ++i_step )
    {
        const int offset = i_step * WARP_SIZE + lane_id;
        Key_type key = m_smem_keys[offset];
        int poll = utils::ballot( key != -1 );

        if ( poll == 0 )
        {
            continue;
        }

        int dst_offset = warp_offset + __popc( poll & lane_mask_lt );

        if ( key != -1 )
        {
            keys[dst_offset] = key;
            pos [dst_offset] = offset;
        }

        warp_offset += __popc( poll );
    }

    m_gmem_count = utils::any( m_gmem_count > 0 );

    if ( !m_gmem_count )
    {
        return warp_offset;
    }

#pragma unroll 4

    for ( int offset = lane_id ; offset < m_gmem_size ; offset += WARP_SIZE )
    {
        Key_type key = m_gmem_keys[offset];
        int poll = utils::ballot( key != -1, utils::activemask() );

        if ( poll == 0 )
        {
            continue;
        }

        int dst_offset = warp_offset + __popc( poll & lane_mask_lt );

        if ( key != -1 )
        {
            keys[dst_offset] = key;
            pos [dst_offset] = SMEM_SIZE + offset;
        }

        warp_offset += __popc( poll );
    }

    return warp_offset;
}


template< typename Key_type, int SMEM_SIZE, int NUM_HASH_FCTS, int WARP_SIZE >
__device__ __forceinline__
int Hash_set<Key_type, SMEM_SIZE, NUM_HASH_FCTS, WARP_SIZE>::store( Key_type *keys)
{
    int lane_id = utils::lane_id();
    int lane_mask_lt = utils::lane_mask_lt();
    int warp_offset = 0;
    const int NUM_STEPS = SMEM_SIZE / WARP_SIZE;
#pragma unroll

    for ( int i_step = 0 ; i_step < NUM_STEPS ; ++i_step )
    {
        const int offset = i_step * WARP_SIZE + lane_id;
        Key_type key = m_smem_keys[offset];
        int poll = utils::ballot( key != -1 );

        if ( poll == 0 )
        {
            continue;
        }

        int dst_offset = warp_offset + __popc( poll & lane_mask_lt );

        if ( key != -1 )
        {
            keys[dst_offset] = key;
        }

        warp_offset += __popc( poll );
    }

    m_gmem_count = utils::any( m_gmem_count > 0 );

    if ( !m_gmem_count )
    {
        return warp_offset;
    }

#pragma unroll 4

    for ( int offset = lane_id ; offset < m_gmem_size ; offset += WARP_SIZE )
    {
        Key_type key = m_gmem_keys[offset];
        int poll = utils::ballot( key != -1, utils::activemask() );

        if ( poll == 0 )
        {
            continue;
        }

        int dst_offset = warp_offset + __popc( poll & lane_mask_lt );

        if ( key != -1 )
        {
            keys[dst_offset] = key;
        }

        warp_offset += __popc( poll );
    }

    return warp_offset;
}



///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template< typename Key_type, typename T, int SMEM_SIZE = 128, int NUM_HASH_FCTS = 4, int WARP_SIZE = 32 >
class Hash_map
{
    protected:
        // The keys stored in the map.
        volatile Key_type *m_smem_keys, *m_gmem_keys;
        // The values stored in the map.
        T *m_smem_vals, *m_gmem_vals;
        // The size of the global memory buffer.
        const int m_gmem_size;
        // Is there any value in GMEM.
        bool m_any_gmem;

    public:
        // Constructor.
        __device__ __forceinline__
        Hash_map( volatile Key_type *smem_keys, volatile Key_type *gmem_keys, T *smem_vals, T *gmem_vals, int gmem_size ) :
            m_smem_keys(smem_keys),
            m_gmem_keys(gmem_keys),
            m_smem_vals(smem_vals),
            m_gmem_vals(gmem_vals),
            m_gmem_size(gmem_size),
            m_any_gmem (true)
        {}

        // Clear the table. It doesn't clear GMEM values.
        __device__ __forceinline__ void clear();
        // Clear the table. It also clears GMEM values (set them to 0).
        __device__ __forceinline__ void clear_all();
        // Insert a key/value inside the hash table.
        __device__ __forceinline__ void insert( Key_type key, T val, int *status );
        // Insert a key/value inside the hash table.
        __device__ __forceinline__ void insert_with_duplicates( Key_type key, T val, int *status );
        // Load a set.
        __device__ __forceinline__ void load( int count, const Key_type *keys, const int *pos );
        // Store the map.
        __device__ __forceinline__ void store( int count, T *vals );
        // Store the map.
        __device__ __forceinline__ void store( int count, Key_type *keys, T *vals );
        // Store the map.
        __device__ __forceinline__ void store_map_keys_scale_values( int count, const int *map, Key_type *keys, T alpha, T *vals );
        // Store the map.
        __device__ __forceinline__ void store_keys_scale_values( int count, Key_type *keys, T alpha, T *vals );
        // Update a value if it is in the table. Do nothing otherwise.
        __device__ __forceinline__ bool update( Key_type key, T val );
};

// ====================================================================================================================

template< typename Key_type, typename T, int SMEM_SIZE, int NUM_HASH_FCTS, int WARP_SIZE >
__device__ __forceinline__
void Hash_map<Key_type, T, SMEM_SIZE, NUM_HASH_FCTS, WARP_SIZE>::clear()
{
    int lane_id = utils::lane_id();
    const int NUM_STEPS = SMEM_SIZE / WARP_SIZE;
#pragma unroll

    for ( int i_step = 0 ; i_step < NUM_STEPS ; ++i_step )
    {
        m_smem_keys[i_step * WARP_SIZE + lane_id] = -1;
    }

    if ( !m_any_gmem )
    {
        return;
    }

#pragma unroll 4

    for ( int offset = lane_id ; offset < m_gmem_size ; offset += WARP_SIZE )
    {
        m_gmem_keys[offset] = -1;
    }

    m_any_gmem = false;
}

// ====================================================================================================================

template< typename Key_type, typename T, int SMEM_SIZE, int NUM_HASH_FCTS, int WARP_SIZE >
__device__ __forceinline__
void Hash_map<Key_type, T, SMEM_SIZE, NUM_HASH_FCTS, WARP_SIZE>::clear_all()
{
    int lane_id = utils::lane_id();
    const int NUM_STEPS = SMEM_SIZE / WARP_SIZE;
#pragma unroll

    for ( int i_step = 0 ; i_step < NUM_STEPS ; ++i_step )
    {
        m_smem_keys[i_step * WARP_SIZE + lane_id] = -1;
        m_smem_vals[i_step * WARP_SIZE + lane_id] = amgx::types::util<T>::get_zero();
    }

    if ( !m_any_gmem )
    {
        return;
    }

#pragma unroll 4

    for ( int offset = lane_id ; offset < m_gmem_size ; offset += WARP_SIZE )
    {
        m_gmem_keys[offset] = -1;
        m_gmem_vals[offset] = amgx::types::util<T>::get_zero();
    }

    m_any_gmem = false;
}

// ====================================================================================================================

template< typename Key_type, typename T, int SMEM_SIZE, int NUM_HASH_FCTS, int WARP_SIZE >
__device__ __forceinline__
void Hash_map<Key_type, T, SMEM_SIZE, NUM_HASH_FCTS, WARP_SIZE>::insert( Key_type key, T val, int *status )
{
    const int lane_id = utils::lane_id();
    bool done = key == -1;
#pragma unroll

    for ( int i_hash = 0 ; i_hash < NUM_HASH_FCTS ; ++i_hash )
    {
        if ( utils::all(done) )
        {
            break;
        }

        bool candidate = false;
        unsigned ukey = reinterpret_cast<unsigned &>( key );
        int hash = ( (ukey ^ c_hash_keys[i_hash]) + c_hash_keys[NUM_HASH_FCTS + i_hash] ) & (SMEM_SIZE - 1);

        if ( !done )
        {
            Key_type stored_key = m_smem_keys[hash];

            if ( stored_key == key )
            {
                m_smem_vals[hash] = m_smem_vals[hash] + val;
                done = true;
            }

            candidate = stored_key == -1;

            if ( candidate )
            {
                m_smem_keys[hash] = key;
            }

            if ( candidate && key == m_smem_keys[hash] )
            {
                m_smem_vals[hash] = val;
                done = true;
            }
        }
    }

    const int num_bits = utils::bfind( m_gmem_size ); // TODO: move it outside ::insert.
#pragma unroll

    for ( int i_hash = 0 ; i_hash < NUM_HASH_FCTS ; ++i_hash )
    {
        if ( utils::all(done) )
        {
            return;
        }

        bool candidate = false;
        unsigned ukey = reinterpret_cast<unsigned &>( key );
        int hash = utils::bfe( (ukey ^ c_hash_keys[i_hash]) + c_hash_keys[NUM_HASH_FCTS + i_hash], num_bits );

        if ( !done )
        {
            Key_type stored_key = m_gmem_keys[hash];

            if ( stored_key == key )
            {
                m_gmem_vals[hash] = m_gmem_vals[hash] + val;
                done = true;
            }

            candidate = stored_key == -1;

            if ( candidate )
            {
                m_gmem_keys[hash] = key;
            }

            if ( candidate && key == m_gmem_keys[hash] ) // More than one candidate may have written to that slot.
            {
                m_gmem_vals[hash] = val;
                done = true;
            }
        }

        m_any_gmem = true;
    }

    if ( status == NULL || utils::all(done) )
    {
        return;
    }

    if ( lane_id == 0 )
    {
        status[0] = 1;
    }
}

// ====================================================================================================================

template< typename Key_type, typename T, int SMEM_SIZE, int NUM_HASH_FCTS, int WARP_SIZE >
__device__ __forceinline__
void Hash_map<Key_type, T, SMEM_SIZE, NUM_HASH_FCTS, WARP_SIZE>::insert_with_duplicates( Key_type key, T val, int *status )
{
    const int lane_id = utils::lane_id();
    bool done = key == -1;
#pragma unroll

    for ( int i_hash = 0 ; i_hash < NUM_HASH_FCTS ; ++i_hash )
    {
        if ( utils::all(done) )
        {
            break;
        }

        bool candidate = false;
        unsigned ukey = reinterpret_cast<unsigned &>( key );
        int hash = ( (ukey ^ c_hash_keys[i_hash]) + c_hash_keys[NUM_HASH_FCTS + i_hash] ) & (SMEM_SIZE - 1);

        if ( !done )
        {
            Key_type stored_key = m_smem_keys[hash];

            if ( stored_key == key )
            {
                utils::atomic_add( &m_smem_vals[hash], val );
                done = true;
            }

            candidate = stored_key == -1;

            if ( candidate )
            {
                m_smem_keys[hash] = key;
            }

            if ( candidate && key == m_smem_keys[hash] )
            {
                utils::atomic_add( &m_smem_vals[hash], val );
                done = true;
            }
        }
    }

    const int num_bits = utils::bfind( m_gmem_size ); // TODO: move it outside ::insert.
#pragma unroll

    for ( int i_hash = 0 ; i_hash < NUM_HASH_FCTS ; ++i_hash )
    {
        if ( utils::all(done) )
        {
            return;
        }

        m_any_gmem = true;
        bool candidate = false;
        unsigned ukey = reinterpret_cast<unsigned &>( key );
        int hash = utils::bfe( (ukey ^ c_hash_keys[i_hash]) + c_hash_keys[NUM_HASH_FCTS + i_hash], num_bits );

        if ( !done )
        {
            Key_type stored_key = m_gmem_keys[hash];

            if ( stored_key == key )
            {
                utils::atomic_add( &m_gmem_vals[hash], val );
                done = true;
            }

            candidate = stored_key == -1;

            if ( candidate )
            {
                m_gmem_keys[hash] = key;
            }

            if ( candidate && key == m_gmem_keys[hash] ) // More than one candidate may have written to that slot.
            {
                utils::atomic_add( &m_gmem_vals[hash], val );
                done = true;
            }
        }
    }

    if ( status == NULL || utils::all(done) )
    {
        return;
    }

    if ( lane_id == 0 )
    {
        status[0] = 1;
    }
}

// ====================================================================================================================

template< typename Key_type, typename T, int SMEM_SIZE, int NUM_HASH_FCTS, int WARP_SIZE >
__device__ __forceinline__
void Hash_map<Key_type, T, SMEM_SIZE, NUM_HASH_FCTS, WARP_SIZE>::load( int count, const Key_type *keys, const int *pos )
{
    int lane_id = utils::lane_id();
#pragma unroll 4

    for ( int offset = lane_id ; offset < count ; offset += WARP_SIZE )
    {
        Key_type key = keys[offset];
        int idx = pos [offset];
        // Where to store the item.
        volatile Key_type *key_ptr = m_smem_keys;
        volatile T *val_ptr = m_smem_vals;

        if ( idx >= SMEM_SIZE )
        {
            key_ptr = m_gmem_keys;
            val_ptr = m_gmem_vals;
            m_any_gmem = 1;
            idx -= SMEM_SIZE;
        }

        // Store the item.
        key_ptr[idx] = key;
        amgx::types::util<T>::volcast( amgx::types::util<T>::get_zero(), val_ptr + idx);
    }

    m_any_gmem = utils::any( m_any_gmem );
}

// ====================================================================================================================

template< typename Key_type, typename T, int SMEM_SIZE, int NUM_HASH_FCTS, int WARP_SIZE >
__device__ __forceinline__
void Hash_map<Key_type, T, SMEM_SIZE, NUM_HASH_FCTS, WARP_SIZE>::store( int count, T *vals )
{
    int lane_id = utils::lane_id();
    int lane_mask_lt = utils::lane_mask_lt();
    int warp_offset = 0;
    const int NUM_STEPS = SMEM_SIZE / WARP_SIZE;
#pragma unroll

    for ( int i_step = 0 ; i_step < NUM_STEPS ; ++i_step )
    {
        const int offset = i_step * WARP_SIZE + lane_id;
        Key_type key = m_smem_keys[offset];
        int poll = utils::ballot( key != -1 );

        if ( poll == 0 )
        {
            continue;
        }

        int dst_offset = warp_offset + __popc( poll & lane_mask_lt );

        if ( key != -1 )
        {
            vals[dst_offset] = m_smem_vals[offset];
        }

        warp_offset += __popc( poll );
    }

    if ( !m_any_gmem )
    {
        return;
    }

#pragma unroll 4

    for ( int offset = lane_id ; offset < m_gmem_size ; offset += WARP_SIZE )
    {
        Key_type key = m_gmem_keys[offset];
        int poll = utils::ballot( key != -1, utils::activemask() );

        if ( poll == 0 )
        {
            continue;
        }

        int dst_offset = warp_offset + __popc( poll & lane_mask_lt );

        if ( key != -1 )
        {
            vals[dst_offset] = m_gmem_vals[offset];
        }

        warp_offset += __popc( poll );
    }
}

// ====================================================================================================================

template< typename Key_type, typename T, int SMEM_SIZE, int NUM_HASH_FCTS, int WARP_SIZE >
__device__ __forceinline__
void Hash_map<Key_type, T, SMEM_SIZE, NUM_HASH_FCTS, WARP_SIZE>::store( int count, Key_type *keys, T *vals )
{
    int lane_id = utils::lane_id();
    int lane_mask_lt = utils::lane_mask_lt();
    int warp_offset = 0;
    const int NUM_STEPS = SMEM_SIZE / WARP_SIZE;
#pragma unroll

    for ( int i_step = 0 ; i_step < NUM_STEPS ; ++i_step )
    {
        const int offset = i_step * WARP_SIZE + lane_id;
        Key_type key = m_smem_keys[offset];
        int poll = utils::ballot( key != -1 );

        if ( poll == 0 )
        {
            continue;
        }

        int dst_offset = warp_offset + __popc( poll & lane_mask_lt );

        if ( key != -1 )
        {
            keys[dst_offset] = key;
            vals[dst_offset] = m_smem_vals[offset];
        }

        warp_offset += __popc( poll );
    }

    if ( !m_any_gmem )
    {
        return;
    }

#pragma unroll 4

    for ( int offset = lane_id ; offset < m_gmem_size ; offset += WARP_SIZE )
    {
        Key_type key = m_gmem_keys[offset];
        int poll = utils::ballot( key != -1, utils::activemask() );

        if ( poll == 0 )
        {
            continue;
        }

        int dst_offset = warp_offset + __popc( poll & lane_mask_lt );

        if ( key != -1 )
        {
            keys[dst_offset] = key;
            vals[dst_offset] = m_gmem_vals[offset];
        }

        warp_offset += __popc( poll );
    }
}

// ====================================================================================================================

template< typename Key_type, typename T, int SMEM_SIZE, int NUM_HASH_FCTS, int WARP_SIZE >
__device__ __forceinline__
void Hash_map<Key_type, T, SMEM_SIZE, NUM_HASH_FCTS, WARP_SIZE>::store_map_keys_scale_values( int count, const int *map, Key_type *keys, T alpha, T *vals )
{
    int lane_id = utils::lane_id();
    int lane_mask_lt = utils::lane_mask_lt();
    int warp_offset = 0;
    const int NUM_STEPS = SMEM_SIZE / WARP_SIZE;
#pragma unroll

    for ( int i_step = 0 ; i_step < NUM_STEPS ; ++i_step )
    {
        const int offset = i_step * WARP_SIZE + lane_id;
        Key_type key = m_smem_keys[offset];
        int poll = utils::ballot( key != -1 );

        if ( poll == 0 )
        {
            continue;
        }

        int dst_offset = warp_offset + __popc( poll & lane_mask_lt );

        if ( key != -1 )
        {
            keys[dst_offset] = map[key];
            vals[dst_offset] = alpha * m_smem_vals[offset];
        }

        warp_offset += __popc( poll );
    }

    if ( !m_any_gmem )
    {
        return;
    }

#pragma unroll 4

    for ( int offset = lane_id ; offset < m_gmem_size ; offset += WARP_SIZE )
    {
        Key_type key = m_gmem_keys[offset];
        int poll = utils::ballot( key != -1, utils::activemask() );

        if ( poll == 0 )
        {
            continue;
        }

        int dst_offset = warp_offset + __popc( poll & lane_mask_lt );

        if ( key != -1 )
        {
            keys[dst_offset] = map[key];
            vals[dst_offset] = alpha * m_gmem_vals[offset];
        }

        warp_offset += __popc( poll );
    }
}

template< typename Key_type, typename T, int SMEM_SIZE, int NUM_HASH_FCTS, int WARP_SIZE >
__device__ __forceinline__
void Hash_map<Key_type, T, SMEM_SIZE, NUM_HASH_FCTS, WARP_SIZE>::store_keys_scale_values( int count, Key_type *keys, T alpha, T *vals )
{
    int lane_id = utils::lane_id();
    int lane_mask_lt = utils::lane_mask_lt();
    int warp_offset = 0;
    const int NUM_STEPS = SMEM_SIZE / WARP_SIZE;
#pragma unroll

    for ( int i_step = 0 ; i_step < NUM_STEPS ; ++i_step )
    {
        const int offset = i_step * WARP_SIZE + lane_id;
        Key_type key = m_smem_keys[offset];
        int poll = utils::ballot( key != -1 );

        if ( poll == 0 )
        {
            continue;
        }

        int dst_offset = warp_offset + __popc( poll & lane_mask_lt );

        if ( key != -1 )
        {
            keys[dst_offset] = key;
            vals[dst_offset] = alpha * m_smem_vals[offset];
        }

        warp_offset += __popc( poll );
    }

    if ( !m_any_gmem )
    {
        return;
    }

#pragma unroll 4

    for ( int offset = lane_id ; offset < m_gmem_size ; offset += WARP_SIZE )
    {
        Key_type key = m_gmem_keys[offset];
        int poll = utils::ballot( key != -1, utils::activemask() );

        if ( poll == 0 )
        {
            continue;
        }

        int dst_offset = warp_offset + __popc( poll & lane_mask_lt );

        if ( key != -1 )
        {
            keys[dst_offset] = key;
            vals[dst_offset] = alpha * m_gmem_vals[offset];
        }

        warp_offset += __popc( poll );
    }
}

// ====================================================================================================================



// ====================================================================================================================

template< typename Key_type, typename T, int SMEM_SIZE, int NUM_HASH_FCTS, int WARP_SIZE >
__device__ __forceinline__
bool Hash_map<Key_type, T, SMEM_SIZE, NUM_HASH_FCTS, WARP_SIZE>::update( Key_type key, T val )
{
    const int lane_id = utils::lane_id();
    bool done = key == -1, found = false;
#pragma unroll

    for ( int i_hash = 0 ; i_hash < NUM_HASH_FCTS ; ++i_hash )
    {
        if ( utils::all(done) )
        {
            break;
        }

        unsigned ukey = reinterpret_cast<unsigned &>( key );
        int hash = ( (ukey ^ c_hash_keys[i_hash]) + c_hash_keys[NUM_HASH_FCTS + i_hash] ) & (SMEM_SIZE - 1);

        if ( !done )
        {
            Key_type stored_key = m_smem_keys[hash];

            if ( stored_key == key )
            {
                m_smem_vals[hash] = m_smem_vals[hash] + val;
                found = true;
            }

            done = found || stored_key == -1;
        }
    }

    const int num_bits = utils::bfind( m_gmem_size ); // TODO: move it outside ::insert.
#pragma unroll

    for ( int i_hash = 0 ; i_hash < NUM_HASH_FCTS ; ++i_hash )
    {
        if ( utils::all(done) )
        {
            return found;
        }

        unsigned ukey = reinterpret_cast<unsigned &>( key );
        int hash = utils::bfe( (ukey ^ c_hash_keys[i_hash]) + c_hash_keys[NUM_HASH_FCTS + i_hash], num_bits );

        if ( !done )
        {
            Key_type stored_key = m_gmem_keys[hash];

            if ( stored_key == key )
            {
                m_gmem_vals[hash] = m_gmem_vals[hash] + val;
                found = true;
            }

            done = found || stored_key == -1;
        }
    }

    return found;
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

