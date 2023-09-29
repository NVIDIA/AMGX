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

#include <cusp/detail/device/utils.h>
#include <determinism_checker.h>
#include <error.h>
#include <thrust/reduce.h>
#include <thrust/device_vector.h>

#include <map>

namespace amgx
{
namespace testing_tools
{


struct hash_path_determinism_checker_private
{
    std::map<std::string, int> checkpoint_counts;
};

hash_path_determinism_checker *hash_path_determinism_checker::singleton()
{
    static hash_path_determinism_checker *m = 0;

    if (!m) { m = new hash_path_determinism_checker; }

    return m;
}

hash_path_determinism_checker::hash_path_determinism_checker()
{
    priv = new hash_path_determinism_checker_private();
}
hash_path_determinism_checker::~hash_path_determinism_checker()
{
    if (priv) { delete priv; }

    priv = 0;
}
static __host__ __device__ unsigned int hash_function(unsigned int a, unsigned int seed)
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

__global__ void fast_hash_kernel(unsigned int *din, unsigned int *dout, long long int N, unsigned int seed)
{
    long long int tid = blockIdx.x * blockDim.x + threadIdx.x;

    for (; tid < N; tid += blockDim.x * gridDim.x)
    {
        dout[tid] = hash_function(din[tid], seed);
    }
}

__global__ void fast_hash_kernel_no_permute(unsigned int *din, unsigned int *dout, long long int N, unsigned int seed)
{
    long long int tid = blockIdx.x * blockDim.x + threadIdx.x;

    for (; tid < N; tid += blockDim.x * gridDim.x)
    {
        dout[tid] = hash_function(din[tid], seed) * hash_function((unsigned int)tid + (unsigned int)(tid >> 32), seed);
    }
}


void hash_path_determinism_checker::checkpoint(const std::string &name, void *data, long long int size_in_bytes, bool no_permute)
{
    amgx::thrust::device_vector<unsigned int> hash_buff(size_in_bytes / 4);

    if ( no_permute )
    {
        fast_hash_kernel_no_permute <<< 26, 256>>>((unsigned int *)data, amgx::thrust::raw_pointer_cast(hash_buff.data()), size_in_bytes / 4, 1987);
    }
    else
    {
        fast_hash_kernel <<< 26, 256>>>((unsigned int *)data, amgx::thrust::raw_pointer_cast(hash_buff.data()), size_in_bytes / 4, 1987);
    }

    cudaCheckError();
    unsigned long long int checksum = amgx::thrust::reduce(hash_buff.begin(), hash_buff.end(), 0ull, amgx::thrust::plus<unsigned long long int>());
    cudaCheckError();
    int &checkpoint_count = priv->checkpoint_counts[name];
    printf("checksum %s\t\t%d\t%llx\n", name.c_str(), checkpoint_count++, checksum);
}

unsigned long long int hash_path_determinism_checker::checksum( void *data, long long int size_in_bytes, bool no_permute )
{
    amgx::thrust::device_vector<unsigned int> hash_buff(size_in_bytes / 4);

    if ( no_permute )
    {
        fast_hash_kernel_no_permute <<< 26, 256>>>((unsigned int *)data, amgx::thrust::raw_pointer_cast(hash_buff.data()), size_in_bytes / 4, 1987);
    }
    else
    {
        fast_hash_kernel <<< 26, 256>>>((unsigned int *)data, amgx::thrust::raw_pointer_cast(hash_buff.data()), size_in_bytes / 4, 1987);
    }

    cudaCheckError();
    unsigned long long int checksum = amgx::thrust::reduce(hash_buff.begin(), hash_buff.end(), 0ull, amgx::thrust::plus<unsigned long long int>());
    cudaCheckError();
    return checksum;
}



}
}

