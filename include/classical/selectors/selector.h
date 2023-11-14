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

namespace classical
{

template <class T_Config> class Selector;

template <class TConfig>
class Selector_Base
{
        typedef typename TConfig::MatPrec ValueType;
        typedef typename TConfig::IndPrec IndexType;
        typedef Vector<typename TConfig::template setVecPrec<AMGX_vecInt>::Type> IVector;
        typedef Vector<typename TConfig::template setVecPrec<AMGX_vecFloat>::Type> FVector;
        typedef Vector<typename TConfig::template setVecPrec<AMGX_vecBool>::Type> BVector;
        typedef Vector<typename TConfig::template setVecPrec<AMGX_vecInt>::Type> IntVector;

        typedef typename TConfig::template setVecPrec<AMGX_vecInt64>::Type i64vec_value_type;
        typedef Vector<i64vec_value_type> I64Vector;

    protected:
        int m_use_opt_kernels = 0;

    public:
        virtual void markCoarseFinePoints(Matrix<TConfig> &A,
                                          FVector &weights,
                                          const BVector &s_con,
                                          IVector &cf_map,
                                          IVector &scratch,
                                          int cf_map_init = 0) = 0;


        virtual void demoteStrongEdges(const Matrix<TConfig> &A,
                                       const FVector &weights,
                                       BVector &s_con,
                                       const IVector &cf_map,
                                       const IndexType offset) = 0;

        virtual void renumberAndCountCoarsePoints( IVector &cf_map,
                int &num_coarse_points,
                int num_rows) = 0;

        virtual void correctCfMap(IVector &cf_map, IVector &cf_map_scanned, IVector &cf_map_S2) = 0;


        virtual void createS2(Matrix<TConfig> &A,
                              Matrix<TConfig> &S2,
                              const BVector &s_con,
                              IVector &cf_map) = 0;

        virtual ~Selector_Base() {}

};


// ----------------------------
//  specialization for host
// ----------------------------

template< AMGX_VecPrecision V, AMGX_MatPrecision M, AMGX_IndPrecision I >
class Selector<TemplateConfig<AMGX_host, V, M, I> > : public Selector_Base< TemplateConfig<AMGX_host, V, M, I> >
{
    public:
        typedef TemplateConfig<AMGX_host, V, M, I> TConfig_h;
        typedef typename TConfig_h::MatPrec ValueType;
        typedef typename TConfig_h::IndPrec IndexType;
        typedef Vector<typename TConfig_h::template setVecPrec<AMGX_vecInt>::Type> IVector;
        typedef Vector<typename TConfig_h::template setVecPrec<AMGX_vecFloat>::Type> FVector;
        typedef Vector<typename TConfig_h::template setVecPrec<AMGX_vecBool>::Type> BVector;
        typedef Matrix<TConfig_h> Matrix_h;
        typedef Vector<typename TConfig_h::template setVecPrec<AMGX_vecInt>::Type> IntVector;

        typedef typename TConfig_h::template setVecPrec<AMGX_vecInt64>::Type i64vec_value_type_h;
        typedef Vector<i64vec_value_type_h> I64Vector;

        virtual void markCoarseFinePoints(Matrix<TConfig_h> &A,
                                          FVector &weights,
                                          const BVector &s_con,
                                          IVector &cf_map,
                                          IVector &scratch,
                                          int cf_map_init = 0) = 0;


        void demoteStrongEdges(const Matrix<TConfig_h> &A,
                               const FVector &weights,
                               BVector &s_con,
                               const IVector &cf_map,
                               const IndexType offset);



        void renumberAndCountCoarsePoints( IVector &cf_map,
                                           int &num_coarse_points,
                                           int num_rows);

        void correctCfMap(IVector &cf_map, IVector &cf_map_scanned, IVector &cf_map_S2) ;

        void createS2(Matrix_h &A,
                      Matrix_h &S2,
                      const BVector &s_con,
                      IVector &cf_map);

        Selector(AMG_Config &cfg, const std::string &cfg_scope)
        {
          this->m_use_opt_kernels = cfg.getParameter<int>("use_opt_kernels", "default");
        }
};

// ----------------------------
//  specialization for device
// ----------------------------

template< AMGX_VecPrecision V, AMGX_MatPrecision M, AMGX_IndPrecision I >
class Selector<TemplateConfig<AMGX_device, V, M, I> > : public Selector_Base< TemplateConfig<AMGX_device, V, M, I> >
{
    public:
        typedef TemplateConfig<AMGX_device, V, M, I> TConfig_d;
        typedef typename TConfig_d::MatPrec ValueType;
        typedef typename TConfig_d::IndPrec IndexType;
        typedef Vector<typename TConfig_d::template setVecPrec<AMGX_vecInt>::Type> IVector;
        typedef Vector<typename TConfig_d::template setVecPrec<AMGX_vecFloat>::Type> FVector;
        typedef Vector<typename TConfig_d::template setVecPrec<AMGX_vecBool>::Type> BVector;
        typedef Matrix<TConfig_d> Matrix_d;
        typedef Vector<typename TConfig_d::template setVecPrec<AMGX_vecInt>::Type> IntVector;

        typedef typename TConfig_d::template setVecPrec<AMGX_vecInt64>::Type i64vec_value_type_d;
        typedef Vector<i64vec_value_type_d> I64Vector_d;

        virtual void markCoarseFinePoints(Matrix<TConfig_d> &A,
                                          FVector &weights,
                                          const BVector &s_con,
                                          IVector &cf_map,
                                          IVector &scratch,
                                          int cf_map_init = 0) = 0;


        void demoteStrongEdges(const Matrix<TConfig_d> &A,
                               const FVector &weights,
                               BVector &s_con,
                               const IVector &cf_map,
                               const IndexType offset);

        void renumberAndCountCoarsePoints( IVector &cf_map,
                                           int &num_coarse_points,
                                           int num_rows);

        void correctCfMap(IVector &cf_map, IVector &cf_map_scanned, IVector &cf_map_S2) ;

        void createS2(Matrix_d &A,
                      Matrix_d &S2,
                      const BVector &s_con,
                      IVector &cf_map);

        template <int hash_size, int group_size>
            void compute_c_hat_opt_dispatch(
                    const Matrix_d &A,
                    const bool *s_con,
                    const int *C_hat_start,
                    int *C_hat,
                    int *C_hat_end,
                    int *cf_map,
                    IntVector &unfine_set);

        Selector(AMG_Config &cfg, const std::string &cfg_scope)
        {
          this->m_use_opt_kernels = cfg.getParameter<int>("use_opt_kernels", "default");
        }
};


template<class TConfig>
class SelectorFactory
{
    public:
        virtual Selector<TConfig> *create(AMG_Config &cfg, const std::string &cfg_scope) = 0;
        virtual ~SelectorFactory() {};

        /********************************************
         * Register a selector class with key "name"
         *******************************************/
        static void registerFactory(std::string name, SelectorFactory<TConfig> *f);

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
        static Selector<TConfig> *allocate(AMG_Config &cfg, const std::string &cfg_scope);

        typedef typename std::map<std::string, SelectorFactory<TConfig>*>::const_iterator Iterator;

        static Iterator getIterator() { return getFactories().begin(); };
        static bool isIteratorLast(const Iterator &iter) { if ( iter == getFactories().end() ) return true; else return false; };

    private:
        static std::map<std::string, SelectorFactory<TConfig>*> &getFactories( );
};

} // namespace classical

} // namespace amgx


