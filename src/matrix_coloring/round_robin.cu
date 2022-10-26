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


#include <basic_types.h>
#include <util.h>
#include <error.h>
#include <types.h>
#include <matrix_coloring/round_robin.h>
#include <cusp/format.h>
#include <cusp/copy.h>
#include <cusp/detail/random.h>
#include <thrust/count.h>
#include <thrust/extrema.h>

namespace amgx
{
// ---------------------------
// Kernels
// ---------------------------

// Kernel to color the rows of the matrix, using min-max approach
template <typename IndexType>
__global__
void colorRowsKernel(IndexType *row_colors, const int num_colors, const int num_rows)
{
    for (int i = threadIdx.x + blockDim.x * blockIdx.x; i < num_rows; i += gridDim.x * blockDim.x)
    {
        row_colors[i] = i % num_colors;
    }
}

// ---------------------------
// Methods
// ---------------------------

template<class T_Config>
RoundRobinMatrixColoringBase<T_Config>::RoundRobinMatrixColoringBase(AMG_Config &cfg, const std::string &cfg_scope) : MatrixColoring<T_Config>(cfg, cfg_scope)
{
    if (cfg.AMG_Config::getParameter<IndexType>("determinism_flag", "default"))
    {
        FatalError("Current implementation of the round-robin coloring does not permit an exact coloring, and therefore cannot lead to deterministic results. Implementation of a deterministic round-robin coloring algorithm still pending", AMGX_ERR_NOT_IMPLEMENTED);
    }

    this->m_num_colors = cfg.AMG_Config::getParameter<int>("num_colors", cfg_scope);
}

template<class TConfig>
void RoundRobinMatrixColoringBase<TConfig>::colorMatrix(Matrix<TConfig> &A)
{
    ViewType oldView = A.currentView();
    this->m_row_colors.resize(A.row_offsets.size() - 1, 0);

    if  (this->m_halo_coloring == SYNC_COLORS) { A.setView(ALL); }
    else { A.setViewExterior(); }

    if (this->m_coloring_level == 0)
    {
        FatalError("Callingn coloring scheme but coloring level==0", AMGX_ERR_NOT_IMPLEMENTED);
    }
    else if (this->m_coloring_level == 1)
    {
        this->colorMatrixOneRing(A);
    }
    else
    {
        FatalError("Round robin coloring algorithm can only do one ring coloring", AMGX_ERR_NOT_IMPLEMENTED);
    }

    A.setView(oldView);
}

// Block version
template <AMGX_VecPrecision t_vecPrec, AMGX_MatPrecision t_matPrec, AMGX_IndPrecision t_indPrec>
void RoundRobinMatrixColoring<TemplateConfig<AMGX_device, t_vecPrec, t_matPrec, t_indPrec> >::colorMatrixOneRing(Matrix_d &A)
{
    // One thread per row
    const int num_rows = A.get_num_rows();
    IndexType *row_colors_ptr = this->m_row_colors.raw();
    const int threads_per_block = 64;
    const int num_blocks = min( AMGX_GRID_MAX_SIZE, (int) (num_rows - 1) / threads_per_block + 1);
    colorRowsKernel<IndexType> <<< num_blocks, threads_per_block>>>(row_colors_ptr, this->m_num_colors, num_rows);
    cudaCheckError();
    /*
    // Sort the vertices based on their color
    A.sorted_rows_by_color.resize(num_owned_rows);

    // Copy row colors
    IVector row_colors(num_owned_rows);
    amgx::thrust::copy(A.row_colors.begin(), A.row_colors.begin()+num_owned_rows, row_colors.begin());

    amgx::thrust::sequence(A.sorted_rows_by_color.begin(),A.sorted_rows_by_color.end());
    amgx::thrust::sort_by_key(row_colors.begin(),row_colors.end(),A.sorted_rows_by_color.begin());
    cudaCheckError();

    // Compute the offset for each color
    IVector d_offsets_rows_per_color(A.num_colors+1);
    A.offsets_rows_per_color.resize(A.num_colors+1);

    amgx::thrust::lower_bound(row_colors.begin(),
                        row_colors.end(),
                        amgx::thrust::counting_iterator<IndexType>(0),
                         amgx::thrust::counting_iterator<IndexType>(d_offsets_rows_per_color.size()),
                         d_offsets_rows_per_color.begin());

    cudaCheckError();

    // Copy from device to host
    A.offsets_rows_per_color = d_offsets_rows_per_color;
    */
}

template <AMGX_VecPrecision t_vecPrec, AMGX_MatPrecision t_matPrec, AMGX_IndPrecision t_indPrec>
void RoundRobinMatrixColoring<TemplateConfig<AMGX_host, t_vecPrec, t_matPrec, t_indPrec> >::colorMatrixOneRing(Matrix_h &A)
{
    FatalError("Haven't implemented RoundRobin matrix coloring for host", AMGX_ERR_NOT_SUPPORTED_TARGET);
}

#define AMGX_CASE_LINE(CASE) template class RoundRobinMatrixColoringBase<TemplateMode<CASE>::Type>;
AMGX_FORALL_BUILDS(AMGX_CASE_LINE)
#undef AMGX_CASE_LINE

#define AMGX_CASE_LINE(CASE) template class RoundRobinMatrixColoring<TemplateMode<CASE>::Type>;
AMGX_FORALL_BUILDS(AMGX_CASE_LINE)
#undef AMGX_CASE_LINE

} // end namespace amgx

