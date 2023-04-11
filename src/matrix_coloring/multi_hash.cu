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
#include <matrix_coloring/multi_hash.h>
#include <cusp/format.h>
#include <cusp/copy.h>
#include <cusp/detail/random.h>
#include <thrust/count.h>
#include <thrust/extrema.h>

// Pseudo-random number generator
namespace amgx
{

__host__ __device__ unsigned int hash(unsigned int a, unsigned int seed)
{
    a ^= seed;
    a = (a + 0x7ed55d16) + (a << 12);
    a = (a ^ 0xc761c23c) + (a >> 19);
    a = (a + 0x165667b1) + (a << 5);
    a = (a ^ 0xd3a2646c) + (a << 9);
    a = (a + 0xfd7046c5) + (a << 3);
    a = (a ^ 0xb55a4f09) + (a >> 16);
    return a;
}

struct is_less_than_zero
{
    __host__ __device__
    bool operator()(int x)
    {
        return x < 0;
    }
};

// ---------------------------
// Kernels
// ---------------------------
template <typename IndexType, int num_hash>
__global__
void colorRowsMultiHashKernel_1step(const IndexType *A_offsets, const IndexType *A_column_indices, IndexType *row_colors, const int num_rows, const int next_color)
{
    int hash_j;
    int hash_i;
    int my_colors[num_hash];
    int my_row_color;

    for (int i = threadIdx.x + blockDim.x * blockIdx.x; i < num_rows; i += blockDim.x * gridDim.x)
    {
        int num_possible_colors = 0;
        bool max_i, min_i;
        int row_start = A_offsets[i];
        int row_end = A_offsets[i + 1];

        //#pragma unroll //can't unroll because it is not the innermost loop
        for (int k = 0; k < num_hash; k++)
        {
            max_i = true;
            min_i = true;

            for (int r = row_start; r < row_end; r++)
            {
                int j = A_column_indices[r];

                if (j >= num_rows) { continue; }

                hash_j = hash(j, k);
                hash_i = hash(i, k);

                // There is an uncolored neighbour that is greater
                if ( hash_j > hash_i)
                {
                    max_i = false;
                }

                // There is an uncolored neighbour that is smaller
                if (hash_j < hash_i)
                {
                    min_i = false;
                }
            }

            // If not colored or colored but coin flip decides color should be changed
            if (max_i)
            {
                my_colors[num_possible_colors++] = 2 * k;
            }
            else if (min_i)
            {
                my_colors[num_possible_colors++] = 2 * k + 1;
            }
        }

        if (num_possible_colors)
        {
            int rand_pick = hash(i, 0) % num_possible_colors;
            my_row_color = my_colors[rand_pick];
        }
        else
        {
            my_row_color = 2 * num_hash;
        }

        row_colors[i] = my_row_color;
    }
}

template <typename IndexType, int num_hash>
__global__
void colorRowsMultiHashKernel(const IndexType *A_offsets, const IndexType *A_column_indices, IndexType *row_colors, const int num_rows, const int next_color, const int seed)
{
    unsigned int i_rand[num_hash];

    for (int i = threadIdx.x + blockDim.x * blockIdx.x; i < num_rows; i += blockDim.x * gridDim.x)
    {
        // skip if previously matched
        if (row_colors[i] != -1) { continue; }

        int not_min = 0, not_max = 0;
        int t;

#pragma unroll
        for (t = 0; t < num_hash; t++)
        {
            i_rand[t] = hash(i, seed + 1043 * t);
        }

        // have we been proved to be not min or max
        int row_start = A_offsets[i];
        int row_end = A_offsets[i + 1];
        int possible_colors = 2 * num_hash;

        for (int r = row_start; r < row_end; r++)
        {
            int j = A_column_indices[r];

            // skip diagonal
            if (j == i || j >= num_rows)
            {
                continue;
            }

            int j_color = row_colors[j];

            if (j_color != -1 && j_color < next_color)
            {
                continue;
            }

#pragma unroll
            for (t = 0; t < num_hash; t++)
            {
                unsigned int j_rand = hash(j, seed + 1043 * t);

                // bail if any neighbor is greater
                if (i_rand[t] <= j_rand && !(not_max & (0x1 << t)  ))
                {
                    not_max |= (0x1 << t);
                    possible_colors--;
                }

                if (i_rand[t] >= j_rand && !(not_min & (0x1 << t)  ))
                {
                    not_min |= (0x1 << t);
                    possible_colors--;
                }
            }

            if (possible_colors == 0)
            {
                break;
            }
        }

        if (possible_colors == 0) { continue; }

        // made it here, so possible_colors > 0
        // pick one of the possible colors pseudo-randomly
        int col_id = i % possible_colors;
        int this_col_id = 0;

        for (t = 0; t < num_hash; t++)
        {
            if (!(not_min & (0x1 << t) ) && col_id == this_col_id)
            {
                row_colors[i] = 2 * t + next_color;
                return;
            }

            this_col_id += !(not_min & (0x1 << t));

            if (!(not_max & (0x1 << t) ) && col_id == this_col_id)
            {
                row_colors[i] = 2 * t + 1 + next_color;
                return;
            }

            this_col_id += !(not_max & (0x1 << t));
        }
    }
}

// ---------------------------
// Methods
// ---------------------------

template<class T_Config>
MultiHashMatrixColoringBase<T_Config>::MultiHashMatrixColoringBase(AMG_Config &cfg, const std::string &cfg_scope) : MatrixColoring<T_Config>(cfg, cfg_scope)
{
    if (cfg.AMG_Config::getParameter<IndexType>("determinism_flag", "default"))
    {
        m_uncolored_fraction = 0.;
    }
    else
    {
        m_uncolored_fraction = cfg.AMG_Config::getParameter<double>("max_uncolored_percentage", cfg_scope);
    }

    max_num_hash = cfg.AMG_Config::getParameter<int>("max_num_hash", cfg_scope);
}

template<class TConfig>
void MultiHashMatrixColoringBase<TConfig>::colorMatrix(Matrix<TConfig> &A)
{
    ViewType oldView = A.currentView();
    this->m_row_colors.resize(A.row_offsets.size() - 1, 0);

    if  (this->m_halo_coloring == SYNC_COLORS) { A.setView(ALL); }
    else { A.setViewExterior(); }

    if (this->m_coloring_level == 0)
    {
        FatalError("Calling coloring scheme but coloring level==0", AMGX_ERR_NOT_SUPPORTED_TARGET);
    }
    else if (this->m_coloring_level == 1)
    {
        this->colorMatrixOneRing(A);
    }
    else
    {
        FatalError("Multi-hash coloring algorithm can only do one ring coloring", AMGX_ERR_NOT_IMPLEMENTED);
    }

    A.setView(oldView);
}


// Block version
template <AMGX_VecPrecision t_vecPrec, AMGX_MatPrecision t_matPrec, AMGX_IndPrecision t_indPrec>
void MultiHashMatrixColoring<TemplateConfig<AMGX_device, t_vecPrec, t_matPrec, t_indPrec> >::colorMatrixOneRing(Matrix_d &A)
{
    // One thread per row
    const int num_rows = A.get_num_rows();
    int max_uncolored_rows = (int) (this->m_uncolored_fraction * ((ValueType) num_rows));
    thrust::fill(this->m_row_colors.begin(), this->m_row_colors.begin() + num_rows, -1);
    cudaCheckError();
    const IndexType *A_row_offsets_ptr    = A.row_offsets.raw();
    const IndexType *A_column_indices_ptr = A.col_indices.raw();
    IndexType *row_colors_ptr = this->m_row_colors.raw();
    const int threads_per_block = 256;
    const int num_blocks = min(AMGX_GRID_MAX_SIZE, (int) (num_rows - 1) / threads_per_block + 1);
    this->m_num_colors = 0;
    // Heuristic for setting the number of hash function to use
    int avg_nonzero = 1.5 * A.row_offsets[num_rows] / num_rows;
    this->num_hash = min(avg_nonzero, this->max_num_hash);
    int next_color = 0;
    int it = 0;
    int seed = 1012;

    if (avg_nonzero != 0)
    {
        for ( int num_uncolored = num_rows; num_uncolored > max_uncolored_rows ; )
        {
            it++;

            // Assign all nodes to 0 colors by default
            if (this->num_hash == 1)
            {
                colorRowsMultiHashKernel<IndexType, 1> <<< num_blocks, threads_per_block>>> (A_row_offsets_ptr, A_column_indices_ptr, row_colors_ptr, num_rows, next_color, seed);
            }
            else if (this->num_hash == 2)
            {
                colorRowsMultiHashKernel<IndexType, 2> <<< num_blocks, threads_per_block>>> (A_row_offsets_ptr, A_column_indices_ptr, row_colors_ptr, num_rows, next_color, seed);
            }
            else if (this->num_hash == 3)
            {
                colorRowsMultiHashKernel<IndexType, 3> <<< num_blocks, threads_per_block>>> (A_row_offsets_ptr, A_column_indices_ptr, row_colors_ptr, num_rows, next_color, seed);
            }
            else if (this->num_hash == 4)
            {
                colorRowsMultiHashKernel<IndexType, 4> <<< num_blocks, threads_per_block>>> (A_row_offsets_ptr, A_column_indices_ptr, row_colors_ptr, num_rows, next_color, seed);
            }
            else if (this->num_hash == 5)
            {
                colorRowsMultiHashKernel<IndexType, 5> <<< num_blocks, threads_per_block>>> (A_row_offsets_ptr, A_column_indices_ptr, row_colors_ptr, num_rows, next_color, seed);
            }
            else if (this->num_hash == 6)
            {
                colorRowsMultiHashKernel<IndexType, 6> <<< num_blocks, threads_per_block>>> (A_row_offsets_ptr, A_column_indices_ptr, row_colors_ptr, num_rows, next_color, seed);
            }
            else if (this->num_hash == 7)
            {
                colorRowsMultiHashKernel<IndexType, 7> <<< num_blocks, threads_per_block>>> (A_row_offsets_ptr, A_column_indices_ptr, row_colors_ptr, num_rows, next_color, seed);
            }
            else if (this->num_hash == 8)
            {
                colorRowsMultiHashKernel<IndexType, 8> <<< num_blocks, threads_per_block>>> (A_row_offsets_ptr, A_column_indices_ptr, row_colors_ptr, num_rows, next_color, seed);
            }
            else if (this->num_hash == 9)
            {
                colorRowsMultiHashKernel<IndexType, 9> <<< num_blocks, threads_per_block>>> (A_row_offsets_ptr, A_column_indices_ptr, row_colors_ptr, num_rows, next_color, seed);
            }
            else if (this->num_hash == 10)
            {
                colorRowsMultiHashKernel<IndexType, 10> <<< num_blocks, threads_per_block>>> (A_row_offsets_ptr, A_column_indices_ptr, row_colors_ptr, num_rows, next_color, seed);
            }
            else if (this->num_hash == 11)
            {
                colorRowsMultiHashKernel<IndexType, 11> <<< num_blocks, threads_per_block>>> (A_row_offsets_ptr, A_column_indices_ptr, row_colors_ptr, num_rows, next_color, seed);
            }
            else if (this->num_hash == 12)
            {
                colorRowsMultiHashKernel<IndexType, 12> <<< num_blocks, threads_per_block>>> (A_row_offsets_ptr, A_column_indices_ptr, row_colors_ptr, num_rows, next_color, seed);
            }
            else if (this->num_hash == 13)
            {
                colorRowsMultiHashKernel<IndexType, 13> <<< num_blocks, threads_per_block>>> (A_row_offsets_ptr, A_column_indices_ptr, row_colors_ptr, num_rows, next_color, seed);
            }
            else if (this->num_hash == 14)
            {
                colorRowsMultiHashKernel<IndexType, 14> <<< num_blocks, threads_per_block>>> (A_row_offsets_ptr, A_column_indices_ptr, row_colors_ptr, num_rows, next_color, seed);
            }
            else if (this->num_hash == 15)
            {
                colorRowsMultiHashKernel<IndexType, 15> <<< num_blocks, threads_per_block>>> (A_row_offsets_ptr, A_column_indices_ptr, row_colors_ptr, num_rows, next_color, seed);
            }
            else if (this->num_hash == 16)
            {
                colorRowsMultiHashKernel<IndexType, 16> <<< num_blocks, threads_per_block>>> (A_row_offsets_ptr, A_column_indices_ptr, row_colors_ptr, num_rows, next_color, seed);
            }
            else if (this->num_hash == 17)
            {
                colorRowsMultiHashKernel<IndexType, 17> <<< num_blocks, threads_per_block>>> (A_row_offsets_ptr, A_column_indices_ptr, row_colors_ptr, num_rows, next_color, seed);
            }
            else if (this->num_hash == 18)
            {
                colorRowsMultiHashKernel<IndexType, 18> <<< num_blocks, threads_per_block>>> (A_row_offsets_ptr, A_column_indices_ptr, row_colors_ptr, num_rows, next_color, seed);
            }
            else if (this->num_hash == 19)
            {
                colorRowsMultiHashKernel<IndexType, 19> <<< num_blocks, threads_per_block>>> (A_row_offsets_ptr, A_column_indices_ptr, row_colors_ptr, num_rows, next_color, seed);
            }
            else if (this->num_hash == 20)
            {
                colorRowsMultiHashKernel<IndexType, 20> <<< num_blocks, threads_per_block>>> (A_row_offsets_ptr, A_column_indices_ptr, row_colors_ptr, num_rows, next_color, seed);
            }
            else if (this->num_hash == 21)
            {
                colorRowsMultiHashKernel<IndexType, 21> <<< num_blocks, threads_per_block>>> (A_row_offsets_ptr, A_column_indices_ptr, row_colors_ptr, num_rows, next_color, seed);
            }
            else if (this->num_hash == 22)
            {
                colorRowsMultiHashKernel<IndexType, 22> <<< num_blocks, threads_per_block>>> (A_row_offsets_ptr, A_column_indices_ptr, row_colors_ptr, num_rows, next_color, seed);
            }
            else if (this->num_hash == 23)
            {
                colorRowsMultiHashKernel<IndexType, 23> <<< num_blocks, threads_per_block>>> (A_row_offsets_ptr, A_column_indices_ptr, row_colors_ptr, num_rows, next_color, seed);
            }
            else if (this->num_hash == 24)
            {
                colorRowsMultiHashKernel<IndexType, 24> <<< num_blocks, threads_per_block>>> (A_row_offsets_ptr, A_column_indices_ptr, row_colors_ptr, num_rows, next_color, seed);
            }
            else if (this->num_hash == 25)
            {
                colorRowsMultiHashKernel<IndexType, 25> <<< num_blocks, threads_per_block>>> (A_row_offsets_ptr, A_column_indices_ptr, row_colors_ptr, num_rows, next_color, seed);
            }
            else if (this->num_hash > 25)
            {
                FatalError("Multi-hash coloring algorithm currently can't handle more than 25 hash functions", AMGX_ERR_NOT_IMPLEMENTED);
            }

            cudaCheckError();
            seed = hash(seed, 0);
            next_color += 2 * this->num_hash;
            num_uncolored = (int) thrust::count_if( this->m_row_colors.begin(), this->m_row_colors.begin() + num_rows, is_less_than_zero() );
            cudaCheckError();
        }
    }
    else
    {
        thrust::fill(this->m_row_colors.begin(), this->m_row_colors.end(), 0);
        cudaCheckError();
    }

    this->m_num_colors = *thrust::max_element(this->m_row_colors.begin(), this->m_row_colors.begin() + num_rows) + 1;
    cudaCheckError();
}

template <AMGX_VecPrecision t_vecPrec, AMGX_MatPrecision t_matPrec, AMGX_IndPrecision t_indPrec>
void MultiHashMatrixColoring<TemplateConfig<AMGX_host, t_vecPrec, t_matPrec, t_indPrec> >::colorMatrixOneRing(Matrix_h &A)
{
    FatalError("Haven't implemented MultiHash matrix coloring for host format", AMGX_ERR_NOT_SUPPORTED_TARGET);
}

#define AMGX_CASE_LINE(CASE) template class MultiHashMatrixColoringBase<TemplateMode<CASE>::Type>;
AMGX_FORALL_BUILDS(AMGX_CASE_LINE)
#undef AMGX_CASE_LINE

#define AMGX_CASE_LINE(CASE) template class MultiHashMatrixColoring<TemplateMode<CASE>::Type>;
AMGX_FORALL_BUILDS(AMGX_CASE_LINE)
#undef AMGX_CASE_LINE

} // end namespace amgx

