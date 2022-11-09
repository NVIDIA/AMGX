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
#include <matrix_coloring/uniform.h>
#include <thrust/count.h>

namespace amgx
{

// ---------------------------
// Kernels
// ---------------------------

template <typename IndexType>
__global__
void color3d(IndexType *row_colors, const int n, const int n1d)
{
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int n1d2 = n1d * n1d;
    int x, y, z;

    for (; idx < n; idx += gridDim.x * blockDim.x)
    {
        int label = idx % n1d2;
        x = (label % n1d) % 2;
        y = (label / n1d) % 2;
        z = (idx / n1d2) % 2 ;
        row_colors[idx] = z * 4 + y * 2 + x;
    }
}

template <typename IndexType>
__global__
void color2d(IndexType *row_colors, const int n, const int n1d)
{
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int x, y;

    for (; idx < n; idx += gridDim.x * blockDim.x)
    {
        x = (idx % n1d) % 2;
        y = (idx / n1d) % 2;
        row_colors[idx] = 2 * y + x;
    }
}

// ---------------------------
// Methods
// ---------------------------

template<class T_Config>
UniformMatrixColoringBase<T_Config>::UniformMatrixColoringBase(AMG_Config &cfg, const std::string &cfg_scope) : MatrixColoring<T_Config>(cfg, cfg_scope)
{
}


// Block version
template <class T_Config>
void UniformMatrixColoringBase<T_Config>::colorMatrix(Matrix<T_Config> &A)
{
    profileSubphaseMatrixColoring();
    ViewType oldView = A.currentView();
    this->m_row_colors.resize(A.row_offsets.size() - 1, 0);

    if  (this->m_halo_coloring == SYNC_COLORS) { A.setView(ALL); }
    else { A.setViewExterior(); }

    if (this->m_coloring_level == 0)
    {
        FatalError("Calling coloring scheme but coloring level==0", AMGX_ERR_NOT_SUPPORTED_TARGET);
    }
    else
    {
        int num_rows = A.get_num_rows();
        int dim = 2;
        this->m_num_colors = 2 << dim;

        if (A.get_num_rows() < this->m_num_colors)
        {
            thrust_wrapper::sequence<T_Config::memSpace>(this->m_row_colors.begin(), this->m_row_colors.begin() + num_rows);
            cudaCheckError();
            this->m_num_colors = A.get_num_rows();
        }
        else
        {
            IndexType *row_colors_ptr = this->m_row_colors.raw();
            const int threads_per_block = 512;
            const int num_blocks = min( AMGX_GRID_MAX_SIZE, (num_rows - 1) / threads_per_block + 1 );

            if (dim == 3)
            {
                int n1d = round(cbrt((double)num_rows));
                //printf("Uniform coloring: n==%d, dim == %d, n1d == %d\n", num_rows, dim, n1d);
                color3d <<< num_blocks, threads_per_block>>>(row_colors_ptr, num_rows, n1d);
            }
            else
            {
                int n1d = round(sqrt((double)num_rows));
                //printf("Uniform coloring: n==%d, dim == %d, n1d == %d\n", num_rows, dim, n1d);
                color2d <<< num_blocks, threads_per_block>>>(row_colors_ptr, num_rows, n1d);
            }

            cudaCheckError();
        }
    }

    A.setView(oldView);
}


#define AMGX_CASE_LINE(CASE) template class UniformMatrixColoringBase<TemplateMode<CASE>::Type>;
AMGX_FORALL_BUILDS(AMGX_CASE_LINE)
#undef AMGX_CASE_LINE

#define AMGX_CASE_LINE(CASE) template class UniformMatrixColoring<TemplateMode<CASE>::Type>;
AMGX_FORALL_BUILDS(AMGX_CASE_LINE)
#undef AMGX_CASE_LINE


} // end namespace amgx

