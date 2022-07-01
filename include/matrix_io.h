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

#include <types.h>
#include <iomanip>
#include <map>
#include <vector.h>
#include <fstream>
#include <matrix.h>
#include <amg_solver.h>
#include <amg_config.h>
#include <distributed/amgx_mpi.h>

namespace amgx
{

namespace io_config
{
enum ReaderProps { NONE = 0, MTX = 1, RHS = 2, SOLN = 4, SIZE = 8, PRINT = 16, GEN_RHS = 32};
static inline bool hasProps( unsigned int query, unsigned int props) { return (query | props) == props; }
inline void addProps(const unsigned int new_props, unsigned int &props) { props |= new_props; }
}

template<class T_Config>
class MatrixIO
{
    public:
        typedef Vector<T_Config> VVector;
        typedef typename T_Config::template setMemSpace<AMGX_host>::Type TConfig_h;
        typedef Vector<TConfig_h> Vector_h;
        typedef typename Matrix<T_Config>::MVector MVector;
        typedef typename Matrix<TConfig_h>::MVector MVector_h;
        typedef typename TConfig_h::template setVecPrec<AMGX_vecInt>::Type ivec_value_type_h;
        typedef Vector<ivec_value_type_h> IVector_h;

        typedef bool (*readerFunc) (std::ifstream &fin, const char *fname
                                    , Matrix<T_Config> &A
                                    , VVector &b
                                    , VVector &x
                                    , const AMG_Config &cfg //@TODO: change behaviour of config::get_parameter to be const, and make const here.
                                    , unsigned int props
                                    , const IVector_h &rank_rows // = IVector_h(0) row indices for given rank
                                   );
        typedef std::map<std::string, readerFunc> readerMap;
        static void registerReader(std::string key, readerFunc func);
        static void unregisterReaders();

        // This is what is called from C-API
        static AMGX_ERROR readSystem(const char *fname
                                     , Matrix<T_Config> &A
                                     , VVector &b
                                     , VVector &x
                                     , const AMG_Config &cfg = AMG_Config()
                                     , unsigned int props = io_config::MTX | io_config::RHS | io_config::SOLN
                                     , const IVector_h &rank_rows = IVector_h(0) // row indices for given rank
                                    );


        static AMGX_ERROR readSystem(const char *fname
                                     , Matrix<T_Config> &A
                                     , const AMG_Config &cfg = AMG_Config()
                                     , unsigned int props = io_config::MTX
                                     , const IVector_h &rank_rows = IVector_h(0) // row indices for given rank
                                    );
        static AMGX_ERROR readSystem(const char *fname
                                     , Matrix<T_Config> &A
                                     , VVector &b
                                     , const AMG_Config &cfg = AMG_Config()
                                     , unsigned int props = io_config::MTX | io_config::RHS
                                     , const IVector_h &rank_rows = IVector_h(0) // row indices for given rank
                                    );
        static std::string readSystemFormat(const char *fname);
        //static AMGX_ERROR readColoring(AuxData* obj, const char* fname);
        //static AMGX_ERROR readGeometry(AuxData* obj, const char* fname);
        //static AMGX_ERROR readGeometry(AuxData* obj, int n, int dimension);


        typedef bool (*writerFunc) (const char *filename, const Matrix<T_Config> *A, const VVector *b, const VVector *x);
        typedef std::map<std::string, writerFunc> writerMap;
        static void registerWriter(std::string key, writerFunc func);
        static void unregisterWriters();

        static AMGX_ERROR writeSystem (const char *filename, const Matrix<T_Config> *A, const VVector *b, const VVector *x);
        static AMGX_ERROR writeSystemWithFormat (const char *filename, const char *format, const Matrix<T_Config> *A, const VVector *b, const VVector *x);

        static bool writeSystemMatrixMarket(const char *fname, const Matrix<T_Config> *tA, const VVector *tb, const VVector *tx);
        static bool writeSystemBinary(const char *fname, const Matrix<T_Config> *tA, const VVector *tb, const VVector *tx);


    private:
        static readerMap &getReaderMap();
        static writerMap &getWriterMap();
};

} // end namespace amgx
