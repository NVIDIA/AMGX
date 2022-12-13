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

#include <basic_types.h>
#include <error.h>
#include <amg_config.h>
#include <distributed/amgx_mpi.h>
#include <amg_config.h>
#include <matrix_io.h>

namespace amgx
{

template<class T_Config>
struct ReadMatrixMarket
{
    typedef typename T_Config::template setMemSpace<AMGX_host>::Type TConfig_h;
    typedef typename TConfig_h::template setVecPrec<AMGX_vecInt>::Type ivec_value_type_h;
    typedef Vector<ivec_value_type_h> IVector_h;
    static bool readMatrixMarket(std::ifstream &fin, const char *fnamec, Matrix<T_Config> &A
                                 , Vector<T_Config> &b
                                 , Vector<T_Config> &x
                                 , const AMG_Config &cfg
                                 , unsigned int props = io_config::MTX | io_config::RHS | io_config::SOLN
                                 , const IVector_h &rank_rows = IVector_h(0) // row indices for given rank
                                )
    {
        FatalError("readMatrixMarket for specified matrix type is unsupported", AMGX_ERR_IO);
    }
};

template<class T_Config>
struct ReadNVAMGBinary
{
    typedef typename T_Config::template setMemSpace<AMGX_host>::Type TConfig_h;
    typedef typename TConfig_h::template setVecPrec<AMGX_vecInt>::Type ivec_value_type_h;
    typedef Vector<ivec_value_type_h> IVector_h;
    static bool read(std::ifstream &fin, const char *fnamec, Matrix<T_Config> &A
                     , Vector<T_Config> &b
                     , Vector<T_Config> &x
                     , const AMG_Config &cfg
                     , unsigned int props = io_config::MTX | io_config::RHS | io_config::SOLN
                     , const IVector_h &rank_rows = IVector_h(0) // row indices for given rank
                    )
    {
        FatalError("ReadNVAMGBinary for specified matrix type is unsupported", AMGX_ERR_IO);
    }
};

template <AMGX_VecPrecision t_vecPrec, AMGX_MatPrecision t_matPrec, AMGX_IndPrecision t_indPrec>
struct ReadNVAMGBinary<TemplateConfig<AMGX_host, t_vecPrec, t_matPrec, t_indPrec> >
{
    typedef TemplateConfig<AMGX_host, t_vecPrec, t_matPrec, t_indPrec> TConfig_h;
    typedef typename TConfig_h::template setVecPrec<AMGX_vecInt>::Type ivec_value_type_h;
    typedef Vector<ivec_value_type_h> IVector_h;
    typedef Matrix<TConfig_h> Matrix_h;
    typedef Vector<TConfig_h> Vector_h;
    static bool read(std::ifstream &fin, const char *fnamec, Matrix_h &A
                     , Vector_h &b
                     , Vector_h &x
                     , const AMG_Config &cfg
                     , unsigned int props = io_config::MTX | io_config::RHS | io_config::SOLN
                     , const IVector_h &rank_rows = IVector_h(0) // row indices for given rank
                    );
};

// host specialization
template <AMGX_VecPrecision t_vecPrec, AMGX_MatPrecision t_matPrec, AMGX_IndPrecision t_indPrec>
struct ReadMatrixMarket<TemplateConfig<AMGX_host, t_vecPrec, t_matPrec, t_indPrec> >
{
    typedef TemplateConfig<AMGX_host, t_vecPrec, t_matPrec, t_indPrec> TConfig_h;
    typedef typename TConfig_h::template setVecPrec<AMGX_vecInt>::Type ivec_value_type_h;
    typedef Vector<ivec_value_type_h> IVector_h;
    typedef Matrix<TConfig_h> Matrix_h;
    typedef Vector<TConfig_h> Vector_h;
    static bool readMatrixMarket(std::ifstream &fin, const char *fnamec, Matrix_h &A
                                 , Vector_h &b
                                 , Vector_h &x
                                 , const AMG_Config &cfg
                                 , unsigned int props = io_config::MTX | io_config::RHS | io_config::SOLN
                                 , const IVector_h &rank_rows = IVector_h(0) // row indices for given rank
                                );
    static bool readMatrixMarketV2(std::ifstream &fin, const char *fnamec, Matrix_h &A
                                   , Vector_h &b
                                   , Vector_h &x
                                   , const AMG_Config &cfg
                                   , unsigned int props = io_config::MTX | io_config::RHS | io_config::SOLN
                                   , const IVector_h &rank_rows = IVector_h(0) // row indices for given rank
                                  );
};

// device specialization
template <AMGX_VecPrecision t_vecPrec, AMGX_MatPrecision t_matPrec, AMGX_IndPrecision t_indPrec>
struct ReadMatrixMarket<TemplateConfig<AMGX_device, t_vecPrec, t_matPrec, t_indPrec> >
{
    typedef TemplateConfig<AMGX_device, t_vecPrec, t_matPrec, t_indPrec> TConfig_d;
    typedef TemplateConfig<AMGX_host, t_vecPrec, t_matPrec, t_indPrec> TConfig_h;
    typedef typename TConfig_h::template setVecPrec<AMGX_vecInt>::Type ivec_value_type_h;
    typedef Vector<ivec_value_type_h> IVector_h;
    typedef Matrix<TConfig_d> Matrix_d;
    typedef Vector<TConfig_d> Vector_d;
    static bool readMatrixMarket(std::ifstream &fin, const char *fnamec, Matrix_d &A
                                 , Vector_d &b
                                 , Vector_d &x
                                 , const AMG_Config &cfg
                                 , unsigned int props = io_config::MTX | io_config::RHS | io_config::SOLN
                                 , const IVector_h &rank_rows = IVector_h(0) // row indices for given rank
                                )
    {
        FatalError("readMatrixMarket for specified matrix type is unsupported", AMGX_ERR_IO);
    }
    static bool readMatrixMarketV2(std::ifstream &fin, const char *fname, Matrix_d &A
                                   , Vector_d &b
                                   , Vector_d &x
                                   , const AMG_Config &cfg
                                   , unsigned int props = io_config::MTX | io_config::RHS | io_config::SOLN
                                   , const IVector_h &rank_rows = IVector_h(0) // row indices for given rank
                                  )
    {
        FatalError("readMatrixMarket for specified matrix type is unsupported", AMGX_ERR_IO);
    }
};
} // end namespace amgx
