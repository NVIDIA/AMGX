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

#include <hash_workspace.h>
#include <global_thread_handle.h>

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

namespace amgx
{

template< AMGX_VecPrecision V, AMGX_MatPrecision M, AMGX_IndPrecision I, typename Key_type >
Hash_Workspace<TemplateConfig<AMGX_device, V, M, I>, Key_type >::Hash_Workspace( bool allocate_vals, int grid_size, int max_warp_count, int gmem_size ) :
    m_allocate_vals(allocate_vals),
    m_grid_size(grid_size),
    m_max_warp_count(max_warp_count),
    m_num_threads_per_row_count(32),
    m_num_threads_per_row_compute(32),
    m_gmem_size(gmem_size),
    m_status(NULL),
    m_work_queue(NULL),
    m_keys(NULL),
    m_vals(NULL)
{
    amgx::memory::cudaMalloc( (void **) &m_status, sizeof(int) );
    amgx::memory::cudaMalloc( (void **) &m_work_queue, sizeof(int) );
    allocate_workspace();
}

// ====================================================================================================================

template< AMGX_VecPrecision V, AMGX_MatPrecision M, AMGX_IndPrecision I, typename Key_type  >
Hash_Workspace<TemplateConfig<AMGX_device, V, M, I>, Key_type >::~Hash_Workspace()
{
    if ( m_allocate_vals )
    {
        amgx::memory::cudaFreeAsync( m_vals );
    }

    amgx::memory::cudaFreeAsync( m_keys );
    amgx::memory::cudaFreeAsync( m_work_queue );
    amgx::memory::cudaFreeAsync( m_status );
}

// ====================================================================================================================

template< AMGX_VecPrecision V, AMGX_MatPrecision M, AMGX_IndPrecision I, typename Key_type >
void Hash_Workspace<TemplateConfig<AMGX_device, V, M, I>, Key_type >::allocate_workspace()
{
    const int NUM_WARPS_IN_GRID = m_grid_size * m_max_warp_count;

    // Allocate memory to store the keys of the device-based hashtable.
    if ( m_keys != NULL )
    {
        amgx::memory::cudaFreeAsync( m_keys );
    }

    size_t sz = NUM_WARPS_IN_GRID * m_gmem_size * sizeof(Key_type);
    amgx::memory::cudaMalloc( (void **) &m_keys, sz );

    // Skip value allocation if needed.
    if ( !m_allocate_vals )
    {
        return;
    }

    // Allocate memory to store the values of the device-based hashtable.
    if ( m_vals != NULL )
    {
        amgx::memory::cudaFreeAsync( m_vals );
    }

    sz = NUM_WARPS_IN_GRID * m_gmem_size * sizeof(double);
    amgx::memory::cudaMalloc( (void **) &m_vals, sz );
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#define AMGX_CASE_LINE(CASE) template class Hash_Workspace<TemplateMode<CASE>::Type, int>;
AMGX_FORALL_BUILDS(AMGX_CASE_LINE)
AMGX_FORCOMPLEX_BUILDS(AMGX_CASE_LINE)
#undef AMGX_CASE_LINE

#define AMGX_CASE_LINE(CASE) template class Hash_Workspace<TemplateMode<CASE>::Type, int64_t>;
AMGX_FORALL_BUILDS(AMGX_CASE_LINE)
AMGX_FORCOMPLEX_BUILDS(AMGX_CASE_LINE)
#undef AMGX_CASE_LINE

/////////////////////////////////////////////////////////////////////////////////////////

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace amgx

