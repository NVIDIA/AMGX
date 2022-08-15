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

