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
#include <aggregation/selectors/agg_selector.h>
#include <matrix.h>

namespace amgx
{
namespace aggregation
{

template <class T_Config> class Serial_BFS_Selector;

template <class T_Config>
class Serial_BFS_Selector : public Selector<T_Config>
{
    public:
        typedef T_Config TConfig;
        typedef typename T_Config::MatPrec ValueType;
        typedef typename T_Config::IndPrec IndexType;
        typedef typename T_Config::MemSpace MemorySpace;
        typedef typename Matrix<T_Config>::IVector IVector;

        // Constructor
        Serial_BFS_Selector(AMG_Config &cfg, const std::string &cfg_scope);

        void setAggregates( Matrix<T_Config> &A,
                            IVector &aggregates, IVector &aggregates_global, int &num_aggregates);

    private:
        int aggregate_size;
        AMG_Config coloring_cfg;
        std::string coloring_cfg_scope;
};

template<class T_Config>
class Serial_BFS_SelectorFactory : public SelectorFactory<T_Config>
{
    public:
        Selector<T_Config> *create(AMG_Config &cfg, const std::string &cfg_scope) { return new Serial_BFS_Selector<T_Config>(cfg, cfg_scope); }
};
}
}
