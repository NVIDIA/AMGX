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

#include <distributed/distributed_manager.h>
#include <vector.h>

namespace amgx
{

template <class T_Config>
class Operator
{
    public:
        typedef T_Config TConfig;
        typedef typename TConfig::MemSpace memory_space;
        typedef typename TConfig::MatPrec  value_type;
        typedef typename TConfig::IndPrec  index_type;

        Operator()
        {
        }

        virtual ~Operator()
        {
        }

        // Apply the operator on vector v and store the result in vector res.
        // Latency hiding must be handled internally by the concrete
        // operator class and cannot be currently propagated to other
        // operators.
        virtual void apply(const Vector<TConfig> &v, Vector<TConfig> &res, ViewType view = OWNED) = 0;

        virtual DistributedManager<TConfig> *getManager() const = 0;

        virtual index_type get_num_rows() const = 0;
        virtual index_type get_num_cols() const = 0;
        virtual index_type get_block_dimx() const = 0;
        virtual index_type get_block_dimy() const = 0;
        virtual index_type get_block_size() const = 0;

        virtual bool is_matrix_singleGPU() const = 0;
        virtual bool is_matrix_distributed() const = 0;

        virtual ViewType currentView() const = 0;
        virtual void setView(ViewType type) = 0;
        virtual void setViewInterior() = 0;
        virtual void setViewExterior() = 0;
        virtual void setInteriorView(ViewType view) = 0;
        virtual void setExteriorView(ViewType view) = 0;
        virtual ViewType getViewInterior() const = 0;
        virtual ViewType getViewExterior() const = 0;

        virtual void getOffsetAndSizeForView(ViewType type, int *offset, int *size) const = 0;
};

}
