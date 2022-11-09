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
#include <matrix_coloring/greedy_recolor.h>
//#include <sm_utils.inl>
#include <strided_reduction.h>
#include <math.h> //std::pow
#include <matrix_coloring/coloring_utils.h>
#include <map>
#include <thrust/sort.h>
#include <thrust/count.h>
#include <thrust/sequence.h>
#include <thrust/sort.h>
#include <thrust/binary_search.h>
#include <memory_intrinsics.h>
#include <thrust/replace.h>
#include <thrust_wrapper.h>

#include <algorithm>

#define GET_COLORS_THRUST 0
#define GET_COLORS_TWOPASS 1
#define GET_COLORS_ATOMIC 2
#define GET_COLORS_GATHER 3
#define GETMAX_THRUST  0
#define GETMAX_TWOPASS 1

//CONFIG

#define GET_COLORS GET_COLORS_TWOPASS
//#define GET_COLORS GET_COLORS_THRUST

#define GETMAX_COLOR GETMAX_TWOPASS
//#define GETMAX_COLOR GETMAX_THRUST

//#define DO_SORT 1
#define USE_GTLT 0
#define DISCARD_COLORED (1 && !USE_GTLT)
#define ONLY_FIRSTSTEP 0

namespace amgx
{
//normal hash function
__host__ __device__ __forceinline__ unsigned int hash2(unsigned int a, const unsigned int seed)
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
//rehashes a given hash (h), rotating its bits.
__host__ __device__ __forceinline__ unsigned int rehash(const unsigned int &a, const unsigned int h, const  unsigned int k, const unsigned int seed)
{
    return (h << k) ^ (h >> k);
}

//compute hashes for a column, compare with the row's hashes and the updated bitfield is_max/is_min for the row.
template<int HASHES>
__device__ __forceinline__ void compute_hashes(const int &row_id, const int &col_id, const int &base_row_hash, const int &base_col_hash, const int &seed, unsigned long long int &is_min, unsigned long long int &is_max)
{
    int col_hash_k = base_col_hash;
    int row_hash_k = base_row_hash;
#pragma unroll

    for (int k = 0; k < HASHES; k++)
    {
        col_hash_k = rehash(col_id, base_col_hash, k, seed);
        row_hash_k = rehash(row_id, base_row_hash, k, seed);
        unsigned long long q = (~(1ull << (63 - k))); //bit of hash k in bitfields

        if (col_hash_k > row_hash_k || (col_hash_k == row_hash_k && row_id < col_id))
        {
            is_max &= q;
        }

        //unsigned long long x2 = col_hash_k < row_hash_k;
        if (col_hash_k < row_hash_k || (col_hash_k == row_hash_k && row_id > col_id))
        {
            is_min &= q;
        }
    }
}

template<int COLORING_LEVEL>
struct neigbor_visitor
{

    template<int WARP_SIZE, int HASHES, bool DISCARD_COLORED_T>
    static __device__ __forceinline__ void visit(const int &row_id, const int &base_row_hash, const int &explore_row, const int *A_rows, const int *A_cols, const int *A_colors_in, const int lane_id, const int seed, unsigned long long int &is_min, unsigned long long int &is_max)
    {
        int row_begin = A_rows[explore_row  ];
        int row_end   = A_rows[explore_row + 1];

        for ( ; row_begin < row_end ; row_begin += WARP_SIZE)
        {
            int row_it = row_begin + lane_id;
            int col_id = -1;
            bool do_compute_hashes = true;
            col_id = A_cols[row_it];//__load_global(A_cols+row_it);
#if DISCARD_COLORED

            if (DISCARD_COLORED_T)
            {
                int col_color = __load_nc(A_colors_in + col_id);

                if (col_color != 0) { do_compute_hashes = false; }
            }

#endif

            if (do_compute_hashes)
            {
                //#pragma unroll
                //for(int k=0; k<(HASHES/16>0)?HASHES/16:1; k++)
                //{
                int base_col_hash = hash2(col_id, seed/*+k*12358*/);
                compute_hashes<HASHES>(row_id, col_id, base_row_hash, base_col_hash, seed, is_min, is_max);
                //}
            }

            neigbor_visitor < COLORING_LEVEL - 1 >::visit<WARP_SIZE, HASHES, DISCARD_COLORED_T>(row_id, base_row_hash, col_id, A_rows, A_cols, A_colors_in, lane_id, seed, is_min, is_max);
        }
    }

};
template<>
struct neigbor_visitor<0>
{

    template<int WARP_SIZE, int HASHES, bool DISCARD_COLORED_T>
    static __device__ __forceinline__ void visit(const int &row_id, const int &base_row_hash, const int &explore_row, const int *A_rows, const int *A_cols, const int *A_colors_in, const int lane_id, const int seed, unsigned long long int &is_min, unsigned long long int &is_max)
    {
        //is_min |= 0ull;
    }

};


template< bool DISCARD_COLORED_T, int HASHES, int COLORING_LEVEL, int CTA_SIZE, int WARP_SIZE>
__global__
void fast_multihash_kernel(
    const int A_num_rows, const int *A_rows, const int *A_cols,
    int *A_colors, const int *A_colors_in,
    int c0, int seed,
    int *sets_per_block)
{
    const int NUM_WARPS_PER_CTA  = CTA_SIZE / WARP_SIZE;
    const int NUM_WARPS_PER_GRID = gridDim.x * NUM_WARPS_PER_CTA;
    const int warp_id = subwarp<WARP_SIZE>::subwarp_id();
    const int lane_id = subwarp<WARP_SIZE>::sublane_id();
    int row_id = blockIdx.x * NUM_WARPS_PER_CTA + warp_id;
    int warp_count = 0;

    for ( ; utils::any(row_id < A_num_rows) ; row_id += NUM_WARPS_PER_GRID )
    {
        int row_color = -1;
        int available_color = 0;

        if (row_id < A_num_rows)
        {
#if DISCARD_COLORED

            if (DISCARD_COLORED_T)
            {
                row_color = A_colors_in[row_id];
                A_colors[row_id] = row_color;
            }
            else
#endif
            {
                row_color = A_colors[row_id];
            }
        }

        if (row_color == 0)
        {
            unsigned long long int is_min = ~0ull;
            unsigned long long int is_max = ~0ull;
            int base_row_hash = hash2(row_id, seed);
#if 1 //COLORING_LEVELR>1
            neigbor_visitor<COLORING_LEVEL>::visit<WARP_SIZE, HASHES, DISCARD_COLORED_T>(row_id, base_row_hash, row_id, A_rows, A_cols, A_colors_in, lane_id, seed, is_min, is_max);
#else
            int row_begin = A_rows[row_id  ];
            int row_end   = A_rows[row_id + 1];

            for ( ; row_begin < row_end ; row_begin += WARP_SIZE)
            {
                int row_it = row_begin + lane_id;
                int col_id = -1;
                col_id = A_cols[row_it];//__load_global(A_cols+row_it);
#if DISCARD_COLORED

                if (DISCARD_COLORED_T)
                {
                    int col_color = __load_nc(A_colors_in + col_id);

                    if (col_color != 0) { continue; }
                }

#endif
                int base_col_hash = hash2(col_id, seed);
                compute_hashes<HASHES>(row_id, col_id, base_row_hash, base_col_hash, seed, is_min, is_max);
            }

#endif

            if (WARP_SIZE > 1)
            {
                using namespace amgx::strided_reduction;
                amgx::strided_reduction::op_and op;
                warp_reduce<1, CTA_SIZE, WARP_SIZE>(is_max, op);
                warp_reduce<1, CTA_SIZE, WARP_SIZE>(is_min, op);
            }

#if 1
            int cmax = 63 - utils::bfind(is_max);
            int cmin = 63 - utils::bfind(is_min);

            if (cmin < HASHES)
            {
                available_color = cmin * 2 + 2;
            }

            if (cmax < HASHES)
            {
                available_color = cmax * 2 + 1;
            }

#else
#pragma unroll

            for (int k = 0; k < HASHES; k++)
            {
                if (is_max & (1ull << 63 - k))
                {
                    available_color = 2 * k + 1;
                    break;
                }
                else if (is_min & (1ull << 63 - k))
                {
                    available_color = 2 * k + 2;
                    break;
                }
            }

#endif
        }

        bool has_set_color = (lane_id == 0) && (available_color != 0);

        if (has_set_color)
        {
            A_colors[row_id] = available_color + c0;
        }

        warp_count += amgx::strided_reduction::warp_binary_count(has_set_color);
    }

#if GET_COLORS == GET_COLORS_ATOMIC
    amgx::strided_reduction::block_count_atomic<1, CTA_SIZE, 32, int>(warp_count, sets_per_block);
    //__threadfence_system();
#elif GET_COLORS == GET_COLORS_TWOPASS
    amgx::strided_reduction::block_count<1, CTA_SIZE, 32, int>(warp_count, sets_per_block);
#elif GET_COLORS == GET_COLORS_GATHER
    amgx::strided_reduction::block_count<1, CTA_SIZE, 32, int>(warp_count, sets_per_block);
    amgx::strided_reduction::block_count_gatherer<int, 1> global_gather(sets_per_block);
#endif
}

////////////////////////////

template< int HASHES, int CTA_SIZE, int WARP_SIZE>
__global__
void fast_multihash_kernel_gtlt_kernel(
    const int A_num_rows, const int *A_rows, const int *A_cols,
    int *A_colors, unsigned long long int *gtlt_out,
    int seed)
{
    const int NUM_WARPS_PER_CTA  = CTA_SIZE / WARP_SIZE;
    const int NUM_WARPS_PER_GRID = gridDim.x * NUM_WARPS_PER_CTA;
    const int warp_id = subwarp<WARP_SIZE>::subwarp_id();
    const int lane_id = subwarp<WARP_SIZE>::sublane_id();
    int row_id = blockIdx.x * NUM_WARPS_PER_CTA + warp_id;
    //int warp_count = 0;

    for ( ; utils::any(row_id < A_num_rows) ; row_id += NUM_WARPS_PER_GRID )
    {
        int row_color = -1;

        if (row_id < A_num_rows)
        {
            row_color = A_colors[row_id];
        }

        if (row_color == 0)
        {
            int row_begin = A_rows[row_id  ];
            int row_end   = A_rows[row_id + 1];
            int base_row_hash = hash2(row_id, seed);
            unsigned char count_gt[HASHES];
            unsigned char count_lt[HASHES];
#pragma unroll

            for (int k = 0; k < HASHES; k++)
            {
                count_gt[k] = 0;
                count_lt[k] = 0;
            }

            for ( ; row_begin < row_end ; row_begin += WARP_SIZE)
            {
                int row_it = row_begin + lane_id;
                int col_id = -1;
                col_id = A_cols[row_it];
                int col_color = A_colors[col_id];

                if (col_color != 0) { continue; }

                int base_col_hash = hash2(col_id, seed);
                int col_hash_k = base_col_hash;
                int row_hash_k = base_row_hash;
#pragma unroll

                for (int k = 0; k < HASHES; k++)
                {
                    col_hash_k = rehash(col_id, base_col_hash, k, seed);
                    row_hash_k = rehash(row_id, base_row_hash, k, seed);

                    //unsigned long long q = (~(1ull << 63-k));

                    if (col_hash_k > row_hash_k || (col_hash_k == row_hash_k && row_id < col_id))
                    {
                        if (count_gt[k] < 255) //avoid overflow
                        {
                            ++count_gt[k];
                        }
                    }

                    if (col_hash_k < row_hash_k || (col_hash_k == row_hash_k && row_id > col_id))
                    {
                        if (count_lt[k] < 255)
                        {
                            ++count_lt[k];
                        }
                    }
                }
            }

            //compress count_gt,lt in 2 bits and pack in an integer
            unsigned long long int packed_gtlt = 0;
#pragma unroll

            for (int k = 0; k < HASHES; k++)
            {
                unsigned long long int max3_gt = count_gt[k];
                unsigned long long int max3_lt = count_lt[k];

                if (max3_gt > 3) { max3_gt = 3; } //TODO use only bits above

                if (max3_lt > 3) { max3_lt = 3; }

                //TODO: use BFI
                packed_gtlt |= (max3_gt << (k * 2));
                packed_gtlt |= (max3_lt << (k * 2 + 32));
            }

            if (lane_id == 0)
            {
                gtlt_out[row_id] = packed_gtlt;
            }
        }
    }
}

template< int HASHES, int CTA_SIZE, int WARP_SIZE>
__global__
void fast_multihash_kernel_gtlt_assign_kernel(
    const int A_num_rows, const int *A_rows, const int *A_cols,
    int *A_colors, unsigned long long int *gtlt_in,
    int c0)
{
    const int NUM_WARPS_PER_CTA  = CTA_SIZE / WARP_SIZE;
    const int NUM_WARPS_PER_GRID = gridDim.x * NUM_WARPS_PER_CTA;
    const int warp_id = subwarp<WARP_SIZE>::subwarp_id();
    const int lane_id = subwarp<WARP_SIZE>::sublane_id();
    int row_id = blockIdx.x * NUM_WARPS_PER_CTA + warp_id;

    for ( ; utils::any(row_id < A_num_rows) ; row_id += NUM_WARPS_PER_GRID )
    {
        int row_color = -1;
        unsigned long long int row_gtlt = 0ull;

        if (row_id < A_num_rows)
        {
            row_color = A_colors[row_id];
            row_gtlt = gtlt_in[row_id];
        }

        int available_color = 0;

        if (row_color == 0)
        {
            int row_begin = A_rows[row_id  ];
            int row_end   = A_rows[row_id + 1];
            unsigned long long int is_min = ~0ull;
            unsigned long long int is_max = ~0ull;

            for ( ; row_begin < row_end ; row_begin += WARP_SIZE)
            {
                int row_it = row_begin + lane_id;
                int col_id = -1;
                unsigned long long int col_gtlt = 0ull;

                if (row_it < row_end)
                {
                    col_id   = A_cols[row_it];
                    col_gtlt = gtlt_in[col_id];
#pragma unroll

                    for (int k = 0; k < HASHES; k++)
                    {
                        int row_gt_k = (row_gtlt >> (k * 2 + 0)) & 3ull;
                        int row_lt_k = (row_gtlt >> (k * 2 + 32)) & 3ull;
                        int col_gt_k = (col_gtlt >> (k * 2 + 0)) & 3ull;
                        int col_lt_k = (col_gtlt >> (k * 2 + 32)) & 3ull;
                        unsigned long long q = (~(1ull << 63 - k));

                        if ((row_gt_k > col_gt_k) || (col_gt_k == row_gt_k && row_id < col_id))
                        {
                            is_max &= q; //xor with k bit: if was 1, becomes 0, if was 0 stays 0
                        }

                        if ((row_lt_k > col_lt_k) || (col_lt_k == row_lt_k && row_id > col_id))
                        {
                            is_min &= q;
                        }
                    }
                }
            }

            int cmax = 63 - utils::bfind(is_max);
            int cmin = 63 - utils::bfind(is_min);

            //if(cmin < HASHES)
            //  available_color = cmin*2+2;
            if (cmax < HASHES)
            {
                available_color = cmax * 2 + 1;
            }
        }

        bool has_set_color = (lane_id == 0) && (available_color != 0);

        if (has_set_color)
        {
            A_colors[row_id] = available_color + c0;
        }
    }
}

////////////////////////////

template< bool SORTED_ROWS, int COLORING_LEVEL, int CTA_SIZE, int WARP_SIZE>
__global__
void recolor_greedy_kernel(
    const int A_num_rows, const int *__restrict A_rows,
    const int *__restrict A_cols,
    int color_to_reassign,
    int *__restrict A_colors,
    int *sorted_rows_by_color, int offset_start, int offset_end
    , int *maxs_per_block
)
{
    const int NUM_WARPS_PER_CTA  = CTA_SIZE / WARP_SIZE;
    const int NUM_WARPS_PER_GRID = gridDim.x * NUM_WARPS_PER_CTA;
    const int warp_id = subwarp<WARP_SIZE>::subwarp_id();
    const int lane_id = subwarp<WARP_SIZE>::sublane_id();
    int job_id = blockIdx.x * NUM_WARPS_PER_CTA + warp_id;
    int last_job  = A_num_rows;

    if (SORTED_ROWS)
    {
        job_id = job_id + offset_start;
        last_job  = offset_end;
    }

    int thread_max = 0;

    //const int NUM_THREADS_PER_GRID = gridDim.x * CTA_SIZE;
    //int row_id = blockIdx.x * CTA_SIZE + threadIdx.x;

    // Iterate over rows.
    for ( ; utils::any(job_id < last_job) ; job_id += NUM_WARPS_PER_GRID )
    {
        int row_color = -1;
        int row_id = -1;

        if (job_id < last_job)
        {
            if (SORTED_ROWS)
            {
                row_id    = sorted_rows_by_color[job_id];
            }
            else
            {
                row_id = job_id;
            }

            if (row_id < A_num_rows)
            {
                row_color = A_colors[row_id];
            }
        }

        used_color_structure_64_bit used_colors;
        used_color_structure_64_bit::aux_tmp aux;
        //used_color_structure_64_bit_colorbox<5> used_colors; //still buggy

        if (row_color == color_to_reassign)
        {
            int row_begin = A_rows[row_id  ];
            int row_end   = A_rows[row_id + 1];

            //used_colors.use_color(color_to_reassign);
            //int num_greater = 0;

            for ( ; row_begin < row_end ; row_begin += WARP_SIZE)
            {
                int row_it = row_begin + lane_id;
                int col_id = -1;

                if (row_it < row_end)
                {
                    col_id = A_cols[row_it];
                }

                if (row_it < row_end)// && col_id < A_num_rows)
                {
                    int col_color = A_colors[col_id];

                    //if(col_color>40) printf("%d\n",col_color);
                    if (col_color != 0)
                    {
                        used_colors.use_color(col_color, aux);
                    }
                }

                if (COLORING_LEVEL == 2)
                {
                    int row_begin = A_rows[col_id  ];
                    int row_end   = A_rows[col_id + 1];

                    for ( ; row_begin < row_end ; row_begin += WARP_SIZE)
                    {
                        int row_it = row_begin + lane_id;
                        int col_id = -1;

                        if (row_it < row_end)
                        {
                            col_id = A_cols[row_it];
                        }

                        if (row_it < row_end)// && col_id < A_num_rows)
                        {
                            int col_color = A_colors[col_id];

                            //if(col_color>40) printf("%d\n",col_color);
                            if (col_color != 0)
                            {
                                used_colors.use_color(col_color, aux);
                            }
                        }
                    }
                }
            }
        }

        used_colors.sync_subwarp<CTA_SIZE, WARP_SIZE>(lane_id);

        /*if(row_color != color_to_reassign && row_id != -1)
        {
          printf("ERR %d %d!\n", row_color, color_to_reassign);
        }*/
        if (row_color == color_to_reassign && lane_id == 0)
        {
            int new_color = used_colors.first_available_color(aux);

            if (new_color != row_color && new_color > 0)
            {
                //printf("changing %d from %d to %d\n",row_id,row_color,new_color);
                A_colors[row_id] = new_color;
            }

#if GETMAX_COLOR == GETMAX_TWOPASS

            if (new_color > thread_max)
            {
                thread_max = new_color;
            }

#endif
            /*if(new_color >= 7)
            {
                printf("SEVEN %d %d\n", new_color, thread_max);
            }*/
        }
    }

#if GETMAX_COLOR == GETMAX_TWOPASS
    using namespace amgx::strided_reduction;
    amgx::strided_reduction::op_max op;
    warp_reduce<1, CTA_SIZE, 32>(thread_max, op);
    block_reduce<1, CTA_SIZE, 32, true>(thread_max, maxs_per_block, op);
#endif
}

////////////////////////////



template< bool SORTED_ROWS, int COLORING_LEVEL, int CTA_SIZE, int WARP_SIZE>
__global__
void recolor_greedy_kernel_anyring(
    const int A_num_rows, const int *A_rows, const int *A_cols,
    int color_to_reassign, int *A_colors,
    int *sorted_rows_by_color, int offset_start, int offset_end, int *maxs_per_block,

    used_color_structure_64_bit *used_colors_inout
)
{
    const int NUM_WARPS_PER_CTA  = CTA_SIZE / WARP_SIZE;
    const int NUM_WARPS_PER_GRID = gridDim.x * NUM_WARPS_PER_CTA;
    const int warp_id = subwarp<WARP_SIZE>::subwarp_id();
    const int lane_id = subwarp<WARP_SIZE>::sublane_id();
    int job_id = blockIdx.x * NUM_WARPS_PER_CTA + warp_id;
    int last_job  = A_num_rows;

    if (SORTED_ROWS)
    {
        job_id = job_id + offset_start;
        last_job  = offset_end;
    }

    int thread_max = 0;

    //const int NUM_THREADS_PER_GRID = gridDim.x * CTA_SIZE;
    //int row_id = blockIdx.x * CTA_SIZE + threadIdx.x;

    // Iterate over rows.
    for ( ; utils::any(job_id < last_job) ; job_id += NUM_WARPS_PER_GRID )
    {
        int row_color = -1;
        int row_id = -1;

        if (job_id < last_job)
        {
            if (SORTED_ROWS)
            {
                row_id    = sorted_rows_by_color[job_id];
            }
            else
            {
                row_id = job_id;
            }

            if (row_id < A_num_rows)
            {
                row_color = A_colors[row_id];
            }
        }

        used_color_structure_64_bit used_colors;
        used_color_structure_64_bit::aux_tmp aux;
        used_colors = used_colors_inout[row_id];
        //used_color_structure_64_bit_colorbox<5> used_colors; //still buggy

        if (row_color == color_to_reassign)
        {
            int new_color = used_colors.first_available_color(aux);

            if (lane_id == 0)
            {
                if (new_color != row_color && new_color > 0)
                {
                    //printf("changing %d from %d to %d\n",row_id,row_color,new_color);
                    A_colors[row_id] = new_color;
                }

#if GETMAX_COLOR == GETMAX_TWOPASS

                if (new_color > thread_max)
                {
                    thread_max = new_color;
                }

#endif
            }

            int row_begin = A_rows[row_id  ];
            int row_end   = A_rows[row_id + 1];

            for ( ; row_begin < row_end ; row_begin += WARP_SIZE)
            {
                int row_it = row_begin + lane_id;
                int col_id = -1;

                if (row_it < row_end)
                {
                    col_id = A_cols[row_it];
                }

                if (row_it < row_end)// && col_id < A_num_rows)
                {
                    int col_color = A_colors[col_id];

                    if (col_color == 0)
                    {
                        used_color_structure_64_bit used_colors_col = used_colors_inout[col_id];
                        used_colors_col.use_color(new_color, aux);
                        used_colors_inout[col_id] = used_colors_col;
                    }

                    if (COLORING_LEVEL == 2)
                    {
                        int row_begin = A_rows[col_id  ];
                        int row_end   = A_rows[col_id + 1];

                        for ( ; row_begin < row_end ; row_begin += WARP_SIZE)
                        {
                            int row_it = row_begin + lane_id;
                            int col_id = -1;

                            if (row_it < row_end)
                            {
                                col_id = A_cols[row_it];
                            }

                            if (row_it < row_end)// && col_id < A_num_rows)
                            {
                                int col_color = A_colors[col_id];

                                if (col_color == 0)
                                {
                                    used_color_structure_64_bit used_colors_col = used_colors_inout[col_id];
                                    used_colors_col.use_color(new_color, aux);
                                    used_colors_inout[col_id] = used_colors_col;
                                }
                            }
                        }
                    }
                }
            }
        }
    }

#if GETMAX_COLOR == GETMAX_TWOPASS
    using namespace amgx::strided_reduction;
    amgx::strided_reduction::op_max op;
    warp_reduce<1, CTA_SIZE, 32>(thread_max, op);
    block_reduce<1, CTA_SIZE, 32, true>(thread_max, maxs_per_block, op);
#endif
}

// ---------------------------
// Methods
// ---------------------------

template< class T_Config >
Greedy_Recolor_MatrixColoring_Base<T_Config>::Greedy_Recolor_MatrixColoring_Base( AMG_Config &cfg, const std::string &cfg_scope) : MatrixColoring<T_Config>(cfg, cfg_scope)
{
    first_pass_config = cfg;
    first_pass_config_scope = cfg_scope;
    //MULTI_HASH: it would be faster but it leaves some uncolored vertices: uncolored vertices cannot be processed in parallel
    first_pass_config.setParameter("matrix_coloring_scheme", std::string("MULTI_HASH"), first_pass_config_scope);
    first_pass_config.setParameter("max_uncolored_percentage", 0.0, first_pass_config_scope);
    m_coloring_custom_arg = cfg.AMG_Config::getParameter<std::string>( "coloring_custom_arg", cfg_scope );
    m_coloring_try_remove_last_color_ = cfg.AMG_Config::getParameter<int>( "coloring_try_remove_last_colors", cfg_scope );
}

// Block version
template< AMGX_VecPrecision V, AMGX_MatPrecision M, AMGX_IndPrecision I >
void
Greedy_Recolor_MatrixColoring<TemplateConfig<AMGX_device, V, M, I> >::colorMatrix( Matrix_d &A )
{
    const int num_rows = A.get_num_rows();
    const int num_nz   = A.get_num_nz();
    int avg_nz   = num_nz / num_rows;
    //printf("AVG %d ",avg_nz);
    //P(single) = 1-((avg-1)/avg)^(K)
    //P(all)    = P(single)^(N)
    //          = (1-((avg-1)/avg)^(K))^(N) == 0.99
    //          = (1-((avg-1)/avg)^(K)) == Pow(0.99,1/N)
    //P(single == 0.99) K = Log(1-0.99)/Log(1/avg)
    //P(all    == 0.99) K = Log(1-Pow(0.99,1/N))/Log(1/avg)
    //99% probability
    //table here http://www.wolframalpha.com/input/?i=Log%281-0.99%29%2FLog%28%28a-1%29%2Fa%29+for+a+in+[2%2C10]
    avg_nz = pow(static_cast<double>(avg_nz), static_cast<double>(this->m_coloring_level));

    if (avg_nz > 16) { avg_nz = 16; }

    if (this->m_coloring_custom_arg != "")
    {
        int x = atoi(this->m_coloring_custom_arg.c_str());

        if (x > 0)
        {
            avg_nz = x;
        }
    }

#define CASENZ(X,ORDER,DO_SORT)\
 else if(avg_nz <= X)\
 {\
     color_matrix_specialized_per_numhash<X,ORDER,DO_SORT>(A);\
 }
#define CASEORDER(ORDER,DO_SORT)\
 if(0)\
 {\
\
 }\
 CASENZ(2,ORDER,DO_SORT)\
 CASENZ(4,ORDER,DO_SORT)\
 CASENZ(8,ORDER,DO_SORT)\
 CASENZ(12,ORDER,DO_SORT)\
 CASENZ(16,ORDER,DO_SORT)\
 CASENZ(20,ORDER,DO_SORT)\
 CASENZ(24,ORDER,DO_SORT)\
 CASENZ(32,ORDER,DO_SORT)\
 CASENZ(48,ORDER,DO_SORT)\
 else\
 {\
     color_matrix_specialized_per_numhash<16,ORDER,DO_SORT>(A);\
 }

    if (0)
    {
    }
    else if (this->m_coloring_level == 1)
    {
        CASEORDER(1, 0);
    }
    else if (this->m_coloring_level == 2)
    {
        CASEORDER(2, 1);
    }
    else if (this->m_coloring_level == 3)
    {
        CASEORDER(3, 1);
    }
    else if (this->m_coloring_level == 4)
    {
        CASEORDER(4, 1);
    }
    else if (this->m_coloring_level == 5)
    {
        CASEORDER(5, 1);
    }
}


// Block version
template< AMGX_VecPrecision V, AMGX_MatPrecision M, AMGX_IndPrecision I >
template<int K1, int COLORING_LEVEL, int DO_SORT>
void
Greedy_Recolor_MatrixColoring<TemplateConfig<AMGX_device, V, M, I> >::color_matrix_specialized_per_numhash( Matrix_d &A )
{
    const int num_rows = A.get_num_rows();
    const int n_blocks = 26 * 4; //TODO: choose optimal blocks number
    const int block_size = 256;
    ViewType oldView = A.currentView();
#if 1 //use fast multihash above as precolorer
    this->m_row_colors.resize( A.row_offsets.size() - 1, 0 );
    typedef typename Matrix<TemplateConfig<AMGX_host, V, M, I> > ::IVector IVector_h;
    IVector sorted_rows_by_color;
    IVector_h offsets_rows_per_color;
    {
        device_vector_alloc<int> sets_per_block(n_blocks + 1);
        int *sets_per_block_ptr = amgx::thrust::raw_pointer_cast(sets_per_block.data());

        if  (this->m_halo_coloring == SYNC_COLORS) { A.setView(ALL); }
        else { A.setViewExterior(); }

        int num_uncolored = num_rows;
        int i = 0;
#if DISCARD_COLORED
        device_vector_alloc<int> color_in(num_rows);
#else
        device_vector_alloc<int> color_in;
#endif
#if USE_GTLT
        device_vector_alloc<unsigned long long int> gtlt(num_rows);
        const int K = K1 * 0.5;
#else
        const int K = K1;
#endif
        int prev_num_uncolored = num_uncolored;
        unsigned int seed = 0;

        while (num_uncolored > 0)
        {
            //int mx  = amgx::thrust::reduce( this->m_row_colors.begin(), this->m_row_colors.begin()+num_rows, 0, amgx::thrust::maximum<int>() );
            //if(mx > 2*K*i) printf("ERR UX %d %d %d\n", num_uncolored, mx, 2*K*i);
#if USE_GTLT
            fast_multihash_kernel_gtlt_kernel<K, block_size, 1> <<< n_blocks, block_size>>>
            (num_rows, A.row_offsets.raw(), A.col_indices.raw(),
             this->m_row_colors.raw(), amgx::thrust::raw_pointer_cast(gtlt.data()),
             seed);
            cudaCheckError();
            fast_multihash_kernel_gtlt_assign_kernel<K, block_size, 1> <<< n_blocks, block_size>>>
            (num_rows, A.row_offsets.raw(), A.col_indices.raw(),
             this->m_row_colors.raw(), amgx::thrust::raw_pointer_cast(gtlt.data()),
             2 * K * i);
            cudaCheckError();
#else

            if (num_uncolored == num_rows || (!DISCARD_COLORED))
            {
                fast_multihash_kernel<false, K, COLORING_LEVEL, block_size, 1> <<< n_blocks, block_size>>>(num_rows, A.row_offsets.raw(), A.col_indices.raw(),
                        this->m_row_colors.raw(), amgx::thrust::raw_pointer_cast(color_in.data()),
                        2 * K * i, seed,
                        sets_per_block_ptr);
            }
            else
            {
                fast_multihash_kernel<true, K, COLORING_LEVEL, block_size, 1> <<< n_blocks, block_size>>>(num_rows, A.row_offsets.raw(), A.col_indices.raw(),
                        this->m_row_colors.raw(), amgx::thrust::raw_pointer_cast(color_in.data()),
                        2 * K * i, seed,
                        sets_per_block_ptr);
            }

            cudaCheckError();
#endif
            seed = hash2(seed, 0);
#if GET_COLORS == GET_COLORS_THRUST || USE_GTLT
            num_uncolored = amgx::thrust::count( this->m_row_colors.begin(), this->m_row_colors.begin() + num_rows, 0);
            cudaCheckError();
#elif GET_COLORS == GET_COLORS_TWOPASS
            num_uncolored -= amgx::strided_reduction::count_block_results_1(n_blocks, sets_per_block_ptr, amgx::strided_reduction::op_sum());
#elif (GET_COLORS == GET_COLORS_ATOMIC) || (GET_COLORS == GET_COLORS_GATHER)
            num_uncolored -= sets_per_block[0];
#endif
            //printf("%d -> %d\n", prev_num_uncolored,num_uncolored,prev_num_uncolored-num_uncolored);
            i++;

            if (prev_num_uncolored == num_uncolored)
            {
                //printf("ERR %d %d %d\n", prev_num_uncolored,num_uncolored,num_rows);
                break;
            }

            prev_num_uncolored = num_uncolored;
#if DISCARD_COLORED
            color_in.swap(this->m_row_colors);
#endif
            /*if (iteration++%4==0) {
                cudaEventRecord(throttle_event);
            } else {
                cudaEventSynchronize(throttle_event);
            }*/
        }

#if DISCARD_COLORED
        color_in.swap(this->m_row_colors);
#endif
        int max_color = i * K * 2; //available_color+c0 = 2*K*i + 2*(K-1)+2=2K(i+1), but i++
        //max_color = amgx::thrust::reduce( this->m_row_colors.begin(), this->m_row_colors.begin()+num_rows, 0, amgx::thrust::maximum<int>() );
        //printf("CC %d<%d\n",max_color,i*K*2+2);
        this->m_num_colors = max_color + 1;
#if ONLY_FIRSTSTEP

        if (1)
#else
        if (this->m_coloring_custom_arg == "FIRSTSTEP")
#endif
        {
            A.setView(oldView);
            return;
        }

        if (DO_SORT)
        {
            //return;
            //this->createColorArrays(A)
            IVector row_colors(this->m_row_colors); //useless copy
            sorted_rows_by_color.resize(num_rows);
            //this->m_sorted_rows_by_color.resize(num_rows);
            thrust_wrapper::sequence(sorted_rows_by_color.begin(), sorted_rows_by_color.end()); //useless sequence
            thrust_wrapper::sort_by_key(row_colors.begin(), row_colors.begin() + num_rows, sorted_rows_by_color.begin()); //useless read from sequence
            cudaCheckError();
            IVector offsets_rows_per_color_d(this->m_num_colors + 1);
            amgx::thrust::lower_bound(row_colors.begin(),
                                row_colors.begin() + num_rows,
                                amgx::thrust::counting_iterator<IndexType>(0),
                                amgx::thrust::counting_iterator<IndexType>(offsets_rows_per_color_d.size()),
                                offsets_rows_per_color_d.begin());
            // Copy from device to host
            offsets_rows_per_color = offsets_rows_per_color_d;
            cudaCheckError();
            /*for(int i=0;i<offsets_rows_per_color_d.size(); i++)
            {
                int x= offsets_rows_per_color[i];
            }*/
            /*oldView = A.currentView();
            if  (this->m_halo_coloring == SYNC_COLORS) A.setView(ALL);
            else A.setViewExterior();*/
        }
    }
#else
    //1st pass: color with a fast method
    MatrixColoring<typename Matrix_d::TConfig> *first_pass_matrix_coloring = MatrixColoringFactory<typename Matrix_d::TConfig>::allocate(this->first_pass_config, this->first_pass_config_scope);
    first_pass_matrix_coloring->colorMatrix(A);
#if DO_SORT
    first_pass_matrix_coloring->createColorArrays(A);
#endif
    this->m_row_colors = first_pass_matrix_coloring->getRowColors();
    this->m_num_colors = first_pass_matrix_coloring->getNumColors();
    sorted_rows_by_color = first_pass_matrix_coloring->getSortedRowsByColor();
    offsets_rows_per_color = first_pass_matrix_coloring->getOffsetsRowsPerColor();
    delete first_pass_matrix_coloring;
    //ViewType oldView = A.currentView();
    //if  (this->m_halo_coloring == SYNC_COLORS) A.setView(ALL);
    //else A.setViewExterior();
#endif
    cudaCheckError();
    //2nd pass: optimize using parallel greedy for each color.
    //If 1st pass coloring is valid all elements of same color can be processed in parallel by definition
#if GETMAX_COLOR == GETMAX_TWOPASS
    device_vector_alloc<int> max_color_per_block(n_blocks + 1, 1);
    int *max_color_per_block_ptr = amgx::thrust::raw_pointer_cast(max_color_per_block.data());
#else
    int *max_color_per_block_ptr = 0;
#endif
    float avg_nnz_per_row = A.get_num_nz() / float(A.get_num_rows());
    int WARP_SIZE = 2;

    if ( avg_nnz_per_row < 4.0f )
    {
        WARP_SIZE = 2;
    }
    else if ( avg_nnz_per_row < 8.0f )
    {
        WARP_SIZE = 4;
    }
    else if ( avg_nnz_per_row < 32.0f )
    {
        WARP_SIZE = 8;
    }
    else
    {
        WARP_SIZE = 32;
    }

    if (this->m_coloring_level >= 2)
    {
        WARP_SIZE = 1;
    }

    //;
    //int max_colorx = amgx::thrust::reduce( this->m_row_colors.begin(), this->m_row_colors.begin()+num_rows, 0, amgx::thrust::maximum<int>() );
    //printf("MAX %d",max_colorx);
#define RECOLOR_CALL_CASE(__WSIZE) if(WARP_SIZE==__WSIZE){ \
  recolor_greedy_kernel<DO_SORT,COLORING_LEVEL,block_size,__WSIZE><<<GRID_SIZE, block_size>>>(\
    num_rows,\
    A.row_offsets.raw(),\
    A.col_indices.raw(),\
    color, this->m_row_colors.raw(),\
    sorted_rows_by_color.raw(),start,end,\
    max_color_per_block_ptr\
    ); cudaCheckError(); }
#if 1 //straight order: generates a color histogram similar to parallel greedy
    //start from color=2: 1st color would be assigned to 1, no change

    //iteration = 0;

    for (int color = 2; color <= this->m_num_colors; color++)
    {
        const int GRID_SIZE = n_blocks;//
        int start, end;

        if (DO_SORT)
        {
            start = offsets_rows_per_color[color];
            end  = num_rows;

            if (color < this->m_num_colors) { end = offsets_rows_per_color[color + 1]; }

            if (end >= num_rows) { end = num_rows; }

            //printf("F %d %d %d < num_rows: %d\n",color,start,end,num_rows,(int)end<=start);
            if (end <= start) { continue; }

            //int NUM_WARPS_PER_CTA = block_size/WARP_SIZE;
            //const int GRID_SIZE = std::min( 2048, ((end-start) + NUM_WARPS_PER_CTA-1) / NUM_WARPS_PER_CTA );
            //printf("%d - %d > %d %d %d\n",end-start,color,start,end,num_rows);
        }
        else
        {
            start = 0;
            end = num_rows;
        }

        //printf("%d : %d %d\n", color, start, end);
        RECOLOR_CALL_CASE(1);
        RECOLOR_CALL_CASE(2);
        RECOLOR_CALL_CASE(4);
        RECOLOR_CALL_CASE(8);
        RECOLOR_CALL_CASE(16);
        RECOLOR_CALL_CASE(32);
        cudaCheckError();
        /*if (iteration++%4==0) {
            cudaEventRecord(throttle_event);
        } else {
            cudaEventSynchronize(throttle_event);
        }*/
    }

#else //reverse order: generates a different kind of histogram

    for (int color = this->m_num_colors; color >= 0; color--)
    {
        RECOLOR_CALL_CASE(1);
        RECOLOR_CALL_CASE(2);
        RECOLOR_CALL_CASE(4);
        RECOLOR_CALL_CASE(8);
        RECOLOR_CALL_CASE(16);
        RECOLOR_CALL_CASE(32);
    }

#endif
    cudaCheckError();
#if GETMAX_COLOR == GETMAX_TWOPASS
    int max_color = amgx::strided_reduction::count_block_results_1(n_blocks, max_color_per_block_ptr, amgx::strided_reduction::op_max());
    /*int max_colorg = amgx::thrust::reduce( this->m_row_colors.begin(), this->m_row_colors.begin()+num_rows, 0, amgx::thrust::maximum<int>() );

    max_color=0;
    for(int i=0;i<n_blocks;i++)
    {
      if(max_color_per_block[i] > max_color) max_color = max_color_per_block[i];
    }
    printf("MAXCOLOR ref=%d %d\n",max_colorg,max_color);
    if(max_colorg!=max_color)exit(1);*/
#else
    int max_color = thrust_wrapper::reduce( this->m_row_colors.begin(), this->m_row_colors.begin() + num_rows, 0, amgx::thrust::maximum<int>() );
#endif
    //printf("MAXCOLOR %d %d\n",max_color_gold,max_color);
    cudaCheckError();
    this->m_num_colors = max_color + 1;
    A.setView(oldView);
}



#define AMGX_CASE_LINE(CASE) template class Greedy_Recolor_MatrixColoring_Base<TemplateMode<CASE>::Type>;
AMGX_FORALL_BUILDS(AMGX_CASE_LINE)
#undef AMGX_CASE_LINE

#define AMGX_CASE_LINE(CASE) template class Greedy_Recolor_MatrixColoring<TemplateMode<CASE>::Type>;
AMGX_FORALL_BUILDS(AMGX_CASE_LINE)
#undef AMGX_CASE_LINE

} // end namespace amgx
