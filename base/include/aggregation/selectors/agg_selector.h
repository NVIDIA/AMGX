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

#include <getvalue.h>
#include <error.h>
#include <basic_types.h>
#include <amg_config.h>
#include <matrix.h>

namespace amgx
{

namespace aggregation
{

template <class T_Config>
class Selector
{
        typedef T_Config TConfig;
        typedef typename T_Config::MatPrec ValueType;
        typedef typename T_Config::IndPrec IndexType;
        typedef typename T_Config::MemSpace MemorySpace;
        typedef typename Matrix<T_Config>::IVector IVector;
        typedef typename Matrix<T_Config>::MVector MVector;

    public:
        virtual void setAggregates(Matrix<T_Config> &A,
                                   IVector &aggregates, IVector &aggregates_global, int &num_aggregates) = 0;
        void printAggregationInfo(const IVector &aggregates, const IVector &aggregates_global, const IndexType num_aggregates) const;

        virtual ~Selector() {}

        void renumberAndCountAggregates(IVector &aggregates, IVector &aggregates_global, const IndexType num_block_rows, IndexType &num_aggregates);
        void assertAggregates( const IVector &aggregates, int numAggregates );
        void assertRestriction( const IVector &R_row_offsets, const IVector &R_col_indices, const IVector &aggregates );
};

template<class T_Config>
class SelectorFactory
{
    public:
        virtual Selector<T_Config> *create(AMG_Config &cfg, const std::string &cfg_scope) = 0;
        virtual ~SelectorFactory() {};

        /********************************************
         * Register a selector class with key "name"
         *******************************************/
        static void registerFactory(std::string name, SelectorFactory<T_Config> *f);

        /********************************************
         * Unregister a selector class with key "name"
         *******************************************/
        static void unregisterFactory(std::string name);

        /********************************************
         * Unregister all the selector classes
         *******************************************/
        static void unregisterFactories( );

        /*********************************************
        * Allocates selector based on cfg
        *********************************************/
        static Selector<T_Config> *allocate(AMG_Config &cfg, const std::string &current_scope);

        typedef typename std::map<std::string, SelectorFactory<T_Config>*>::const_iterator Iterator;

        static Iterator getIterator() { return getFactories().begin(); };
        static bool isIteratorLast(const Iterator &iter) { if ( iter == getFactories().end() ) return true; else return false; };

    private:
        static std::map<std::string, SelectorFactory<T_Config>*> &getFactories( );
};

}

} // namespace amgx


