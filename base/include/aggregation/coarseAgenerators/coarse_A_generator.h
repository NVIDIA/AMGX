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
class CoarseAGenerator
{
        typedef T_Config TConfig;
        typedef typename T_Config::MatPrec ValueType;
        typedef typename T_Config::IndPrec IndexType;
        typedef typename T_Config::MemSpace MemorySpace;

    public:
        virtual void computeAOperator(const Matrix<T_Config> &A, Matrix<T_Config> &Ac, const typename Matrix<T_Config>::IVector &aggregates, const typename Matrix<T_Config>::IVector &R_row_offsets, const typename Matrix<T_Config>::IVector &R_column_indices, const int num_aggregates) = 0;

        virtual ~CoarseAGenerator() {}

    protected:
        void printNonzeroStats(const typename Matrix<T_Config>::IVector &Ac_row_offsets, const int num_aggregates);

};

template<class T_Config>
class CoarseAGeneratorFactory
{
    public:
        virtual CoarseAGenerator<T_Config> *create(AMG_Config &cfg, const std::string &cfg_scope) = 0;
        virtual ~CoarseAGeneratorFactory() {};

        /********************************************
         * Register a generator class with key "name"
         *******************************************/
        static void registerFactory(std::string name, CoarseAGeneratorFactory<T_Config> *f);

        /********************************************
         * Unregister a CoarseAGenerator class with key "name"
         *******************************************/
        static void unregisterFactory(std::string name);

        /********************************************
         * Unregister all the CoarseAGenerator classes
         *******************************************/
        static void unregisterFactories( );

        /*********************************************
        * Allocates generator based on cfg
        *********************************************/
        static CoarseAGenerator<T_Config> *allocate(AMG_Config &cfg, const std::string &cfg_scope );

        typedef typename std::map<std::string, CoarseAGeneratorFactory<T_Config>*>::const_iterator Iterator;

        static Iterator getIterator() { return getFactories().begin(); };
        static bool isIteratorLast(const Iterator &iter) { if ( iter == getFactories().end() ) return true; else return false; };

    private:
        static std::map<std::string, CoarseAGeneratorFactory<T_Config>*> &getFactories( );
};

}

} // namespace amgx


