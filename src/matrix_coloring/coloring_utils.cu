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

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/sort.h>
#include <thrust/copy.h>
#include <thrust/count.h>
#include <thrust/inner_product.h>
#include <thrust/binary_search.h>
#include <thrust/adjacent_difference.h>
#include <thrust/iterator/constant_iterator.h>
#include <thrust/iterator/counting_iterator.h>
#include <error.h>
#include <vector_thrust_allocator.h>
#include <thrust_wrapper.h>

#include <algorithm>

namespace amgx
{

void coloring_histogram(int *out_hist_host, int num_rows, int max_color, int *row_colors)
{
    amgx::thrust::device_ptr<int> data(row_colors);

    if (false) //TODO: enable, since almost always faster
    {
        device_vector_alloc<int> histogram(num_rows);
        thrust_wrapper::sort(data, data + num_rows);
        cudaCheckError();
        amgx::thrust::counting_iterator<int> search_begin(0);
        amgx::thrust::upper_bound(data, data + num_rows,
                            search_begin, search_begin + num_rows,
                            histogram.begin());
        cudaCheckError();
        amgx::thrust::adjacent_difference(histogram.begin(), histogram.end(), histogram.begin());
        cudaCheckError();
        amgx::thrust::host_vector<int> h(max_color + 1);
        amgx::thrust::copy(histogram.begin(), histogram.begin() + max_color + 1, h.begin());
        cudaCheckError();

        for (int i = 0; i <= max_color; i++)
        {
            out_hist_host[i] = h[i]; //TODO avoid this copy
        }
    }
    else
    {
        for (int i = 0; i <= max_color; i++)
        {
            out_hist_host[i] = amgx::thrust::count(data, data + num_rows, i);
        }

        cudaCheckError();
    }
}


__global__ void permute_colors_kernel(int num_rows, int *row_colors, int *color_permutation)
{
    int row_id = blockIdx.x * blockDim.x + threadIdx.x;

    for ( ; row_id < num_rows ; row_id += blockDim.x * gridDim.x )
    {
        int color = row_colors[row_id];
        color = __ldg(color_permutation + color);
        row_colors[row_id] = color;
    }
}

__global__ void reverse_colors_kernel(int num_rows, int max_color, int *row_colors)
{
    int row_id = blockIdx.x * blockDim.x + threadIdx.x;

    for ( ; row_id < num_rows ; row_id += blockDim.x * gridDim.x )
    {
        int color = row_colors[row_id];

        if (color > 0)
        {
            //1 -> max_color
            //max_color -> 1
            color = max_color - color + 1;
        }

        row_colors[row_id] = color;
    }
}

int reverse_colors(int num_rows, int max_color, int *row_colors)
{
    const int GRID_SIZE_1 = std::min( 2048, (num_rows + 256 - 1) / 256 );
    reverse_colors_kernel <<< GRID_SIZE_1, 256>>>(num_rows, max_color, row_colors);
    cudaCheckError();
    return max_color;
}

int eliminate_null_colors(int num_rows, int max_color, int *row_colors)
{
    amgx::thrust::host_vector<int> hist(max_color + 1, 0);
    amgx::thrust::host_vector<int> perm(max_color + 1, 0);
    device_vector_alloc<int> perm_d(max_color + 1, 0);
    coloring_histogram(amgx::thrust::raw_pointer_cast(hist.data()), num_rows, max_color, row_colors);
    int nonempty_color = 1;

    for (int i = 1; i <= max_color; i++) //0 is blank
    {
        if (hist[i] > 0) //keep it if not empty
        {
            perm[i] = nonempty_color++;
        }
    }

    perm_d = perm;
    const int GRID_SIZE_1 = std::min( 2048, (num_rows + 256 - 1) / 256 );
    permute_colors_kernel <<< GRID_SIZE_1, 256>>>(num_rows, row_colors, amgx::thrust::raw_pointer_cast(perm_d.data()));
    cudaCheckError();
    return nonempty_color - 1;
}
int reorder_colors_by_frequency(int num_rows, int max_color, int *row_colors)
{
    amgx::thrust::host_vector<int> hist(max_color + 1, 0);
    amgx::thrust::host_vector<int> perm(max_color + 1, 0);
    device_vector_alloc<int> perm_d(max_color + 1, 0);
    coloring_histogram(amgx::thrust::raw_pointer_cast(hist.data()), num_rows, max_color, row_colors);
    int nonempty_color = 1;

    for (int i = 1; i <= max_color; i++) //0 is blank
    {
        if (hist[i] > 0) //keep it if not empty
        {
            perm[i] = nonempty_color++;
        }
    }

    perm_d = perm;
    const int GRID_SIZE_1 = std::min( 2048, (num_rows + 256 - 1) / 256 );
    permute_colors_kernel <<< GRID_SIZE_1, 256>>>(num_rows, row_colors, amgx::thrust::raw_pointer_cast(perm_d.data()));
    cudaCheckError();
    return nonempty_color - 1;
    //return 0;
}

}
