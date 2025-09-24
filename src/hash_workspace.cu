// SPDX-FileCopyrightText: 2013 - 2024 NVIDIA CORPORATION. All Rights Reserved.
//
// SPDX-License-Identifier: BSD-3-Clause

#include <hash_workspace.h>
#include <global_thread_handle.h>
#include <error.h>

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
    amgx::memory::cudaMallocAsync( (void **) &m_status, sizeof(int) );
    cudaCheckError();
    amgx::memory::cudaMallocAsync( (void **) &m_work_queue, sizeof(int) );
    cudaCheckError();
    allocate_workspace();
    cudaStreamSynchronize(0);
    cudaCheckError();
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
    const size_t NUM_WARPS_IN_GRID = m_grid_size * m_max_warp_count;

    // Allocate memory to store the keys of the device-based hashtable.
    if ( m_keys != NULL )
    {
        amgx::memory::cudaFreeAsync( m_keys );
    }

    size_t sz = NUM_WARPS_IN_GRID * m_gmem_size * sizeof(Key_type);
    amgx::memory::cudaMallocAsync( (void **) &m_keys, sz );

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
    amgx::memory::cudaMallocAsync( (void **) &m_vals, sz );
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

