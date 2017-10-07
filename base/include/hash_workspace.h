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

#include <basic_types.h>

namespace amgx
{

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template< typename T_Config, typename Key_type >
class Hash_Workspace
{};

template< AMGX_VecPrecision V, AMGX_MatPrecision M, AMGX_IndPrecision I, typename Key_type >
class Hash_Workspace<TemplateConfig<AMGX_device, V, M, I>, Key_type >
{
    public:
        typedef TemplateConfig<AMGX_device, V, M, I> TConfig_d;
        typedef typename MatPrecisionMap<M>::Type Value_type;

    protected:
        // Do we need values on the GPU?
        bool m_allocate_vals;
        // Constant parameters.
        const int m_grid_size, m_max_warp_count;
        // The number of threads per row of B.
        int m_num_threads_per_row_count, m_num_threads_per_row_compute;
        // The size of the GMEM buffers (number of elements).
        int m_gmem_size;
        // The status: OK if count_non_zeroes succeeded, FAILED otherwise.
        int *m_status;
        // The work queue for dynamic load balancing in the kernels.
        int *m_work_queue;
        // The buffer to store keys in GMEM.
        Key_type *m_keys;
        // The buffer to store values in GMEM.
        Value_type *m_vals;

    public:
        // Create a workspace.
        Hash_Workspace( bool allocate_vals = true, int grid_size = 128, int max_warp_count = 8, int gmem_size = 2048 );

        // Release memory used by the workspace.
        virtual ~Hash_Workspace();

        // Get the size of GMEM.
        inline int get_gmem_size() const { return m_gmem_size; }
        // Get the status flag.
        inline int *get_status() const { return m_status; }
        // Get the work queue.
        inline int *get_work_queue() const { return m_work_queue; }
        // Get the keys.
        inline Key_type *get_keys() const { return m_keys; }
        // Get the values.
        inline Value_type *get_vals() const { return m_vals; }

        // Expand the workspace.
        inline void expand() { m_gmem_size *= 2; allocate_workspace(); }

        // Define the number of threads per row of B.
        inline void set_num_threads_per_row_count( int val ) { m_num_threads_per_row_count = val; }
        // Define the number of threads per row of B.
        inline void set_num_threads_per_row_compute( int val ) { m_num_threads_per_row_compute = val; }

    protected:
        // Allocate memory to store keys/vals in GMEM.
        virtual void allocate_workspace();
};

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace amgx

