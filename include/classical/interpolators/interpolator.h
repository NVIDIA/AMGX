/* Copyright (c) 2011-2022, NVIDIA CORPORATION. All rights reserved.
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
#include <amg.h>

namespace amgx
{

template <class TConfig>
class Interpolator
{
        typedef typename TConfig::MatPrec ValueType;
        typedef typename TConfig::IndPrec IndexType;
        typedef Vector<typename TConfig::template setVecPrec<AMGX_vecInt>::Type> IVector;
        typedef Vector<typename TConfig::template setVecPrec<AMGX_vecBool>::Type> BVector;

    public:
        virtual void generateInterpolationMatrix(Matrix<TConfig> &A,
                IVector &cf_map,
                BVector &s_con,
                IVector &scratch,
                Matrix<TConfig> &P) = 0;
        virtual ~Interpolator() {}
};

template<class TConfig>
class InterpolatorFactory
{
    public:
        virtual Interpolator<TConfig> *create(AMG_Config &cfg, const std::string &cfg_scope) = 0;
        virtual ~InterpolatorFactory() {};

        /***********************************************
         * Register a interpolator class with key "name"
         **********************************************/
        static void registerFactory(std::string name, InterpolatorFactory<TConfig> *f);

        /********************************************
         * Unregister a interpolator class with key "name"
         *******************************************/
        static void unregisterFactory(std::string name);

        /********************************************
         * Unregister all the interpolator classes
         *******************************************/
        static void unregisterFactories( );

        /**********************************************
        * Allocates interpolators based on cfg
        **********************************************/
        static Interpolator<TConfig> *allocate(AMG_Config &cfg, const std::string &cfg_scope);

        typedef typename std::map<std::string, InterpolatorFactory<TConfig>*>::const_iterator Iterator;

        static Iterator getIterator() { return getFactories().begin(); };
        static bool isIteratorLast(const Iterator &iter) { if ( iter == getFactories().end() ) return true; else return false; };

    private:
        static std::map<std::string, InterpolatorFactory<TConfig>*> &getFactories( );
};

} // namespace amgx
