// SPDX-FileCopyrightText: 2011 - 2025 NVIDIA CORPORATION. All Rights Reserved.
//
// SPDX-License-Identifier: BSD-3-Clause

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template< typename Key_type, int SMEM_SIZE = 128, int WARP_SIZE = 32 >
class Hash_index
{
    public:
        // The number of registers needed to store the index.
        enum { REGS_SIZE = SMEM_SIZE / WARP_SIZE };

//private:
        // The partial sums of the index (stored in registers).
        int m_partial[REGS_SIZE];
        // The index in GMEM.
        int *m_gmem;

    public:
        // Create an index (to be associated with a hash set).
        __device__ __forceinline__ Hash_index( int *gmem ) : m_gmem(gmem) {}

        // Build the index from a SMEM buffer of size SMEM_SIZE.
        __device__ __forceinline__ void build_smem_index( const volatile Key_type *s_buffer );
        // Given an offset in SMEM, it finds the index.
        __device__ __forceinline__ int find_smem( int offset ) const;
        // Given an offset in GMEM, it finds the index.
        __device__ __forceinline__ int find_gmem( int offset ) const;
        // Set an indexed item in GMEM.
        __device__ __forceinline__ void set_gmem_index( int offset, int val ) { m_gmem[offset] = val; }
};

// ====================================================================================================================

template< typename Key_type, int SMEM_SIZE, int WARP_SIZE >
__device__ __forceinline__
void
Hash_index<Key_type, SMEM_SIZE, WARP_SIZE>::build_smem_index( const volatile Key_type *s_buffer )
{
    const int lane_id = utils::lane_id();
#pragma unroll

    for ( int i = 0, offset = lane_id ; i < REGS_SIZE ; ++i, offset += WARP_SIZE )
    {
        m_partial[i] = utils::ballot( s_buffer[offset] != -1 );
    }
}

// ====================================================================================================================

template< typename Key_type, int SMEM_SIZE, int WARP_SIZE >
__device__ __forceinline__
int
Hash_index<Key_type, SMEM_SIZE, WARP_SIZE>::find_smem( int offset ) const
{
    const int offset_div_warp_size = offset / WARP_SIZE;
    const int offset_mod_warp_size = offset % WARP_SIZE;
    int result = 0;
#pragma unroll

    for ( int i = 0 ; i < REGS_SIZE ; ++i )
    {
        int mask = 0xffffffff;

        if ( i == offset_div_warp_size )
        {
            mask = (1 << offset_mod_warp_size) - 1;
        }

        if ( i <= offset_div_warp_size )
        {
            result += __popc( m_partial[i] & mask );
        }
    }

    return result;
}

// ====================================================================================================================

template< typename Key_type, int SMEM_SIZE, int WARP_SIZE >
__device__ __forceinline__
int
Hash_index<Key_type, SMEM_SIZE, WARP_SIZE>::find_gmem( int offset ) const
{
    return m_gmem[offset];
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

