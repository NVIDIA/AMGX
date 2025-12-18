// SPDX-FileCopyrightText: 2013 - 2025 NVIDIA CORPORATION. All Rights Reserved.
//
// SPDX-License-Identifier: BSD-3-Clause

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
        cudaCheckError();
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
        cudaCheckError();
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

