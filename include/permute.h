// SPDX-FileCopyrightText: 2011 - 2025 NVIDIA CORPORATION. All Rights Reserved.
//
// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#include <assert.h>

namespace amgx
{

//has scattered writes & colesced reads
template<class VectorType1, class VectorType2, class VectorType3>
__global__ void permuteVector_kernel(VectorType1 v1, VectorType2 v2, VectorType3 p, int N)
{
    //have one thread take each element in a block.
    short bsize = v1.get_block_size();

    for (int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < N; idx += gridDim.x * blockDim.x)
    {
        int idk = idx / bsize; //index into the block
        int idj = idx % bsize; //index within the block
        //we can colapse i/j indexing to just i indexing by using 0 in the second offset
        v2(p[idk], 0, idj) = v1(idk, 0, idj);
    }
}

//has colesced writes & scattered reads
template<class VectorType1, class VectorType2, class VectorType3>
__global__ void unpermuteVector_kernel(VectorType1 v1, VectorType2 v2, VectorType3 p, int N)
{
    //have one thread take each element in a block.
    short bsize = v2.get_block_size();

    for (int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < N; idx += gridDim.x * blockDim.x)
    {
        int idk = idx / bsize; //index into the block
        int idj = idx % bsize; //index within the block
        //we can colapse i/j indexing to just i indexing by using 0 in the second offset
        v2(idk, 0, idj) = v1(p[idk], 0, idj);
    }
}


template<class VectorType1, class VectorType2, class VectorType3>
void permuteVector(VectorType1 &v1, VectorType2 &v2, VectorType3 &p, int size_v1 )
{
    int threads_per_block = v1.get_block_size(); //1 thread per element
    //int N=v1.size();
    int threads = 256;
    int blocks = (size_v1 + threads - 1) / threads;
    blocks = std::min(4096, blocks);
    assert(v1.get_block_size() == v2.get_block_size());
    permuteVector_kernel <<< blocks, threads>>>(v1.pod(), v2.pod(), p.pod(), size_v1);
    cudaCheckError();
}

template<class VectorType1, class VectorType2, class VectorType3>
void unpermuteVector(VectorType1 &v1, VectorType2 &v2, VectorType3 &p, int size_v2 )
{
    int threads_per_block = v2.get_block_size(); //1 thread per element
    //int N=v2.size();
    int threads = 256;
    int blocks = (size_v2 + threads - 1) / threads;
    blocks = std::min(4096, blocks);
    assert(v1.get_block_size() == v2.get_block_size());
    unpermuteVector_kernel <<< blocks, threads>>>(v1.pod(), v2.pod(), p.pod(), size_v2);
    cudaCheckError();
}
}

