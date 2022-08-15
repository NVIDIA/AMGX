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
    blocks = min(4096, blocks);
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
    blocks = min(4096, blocks);
    assert(v1.get_block_size() == v2.get_block_size());
    unpermuteVector_kernel <<< blocks, threads>>>(v1.pod(), v2.pod(), p.pod(), size_v2);
    cudaCheckError();
}
}

