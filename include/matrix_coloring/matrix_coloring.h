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
namespace amgx
{
template <class Matrix> class MatrixColoring;
template <class Matrix> class MatrixColoringFactory;
}

#include <getvalue.h>
#include <error.h>
#include <amg_config.h>
#include <map>
#include <string>
#include <vector.h>
#include <matrix.h>

namespace amgx
{

/*************************************
 * MatrixColoring base class
 *************************************/
template<class T_Config>
class MatrixColoring
{
//  friend MatrixColoringFactory<T_Config>;
    public:
        typedef T_Config TConfig;
        static const AMGX_VecPrecision vecPrec = TConfig::vecPrec;
        static const AMGX_MatPrecision matPrec = TConfig::matPrec;
        static const AMGX_IndPrecision indPrec = TConfig::indPrec;

        typedef TemplateConfig<AMGX_host, vecPrec, matPrec, indPrec> TConfig_h;
        typedef TemplateConfig<AMGX_device, vecPrec, matPrec, indPrec> TConfig_d;

        typedef typename TConfig::MatPrec ValueType;
        typedef typename TConfig::IndPrec IndexType;
        //typedef typename TConfig::MemSpace MemorySpace;
        typedef Vector<T_Config> VVector;

        typedef Matrix<TConfig_h> Matrix_h;
        typedef Matrix<TConfig_d> Matrix_d;

        typedef typename Matrix_h::IVector IVector_h;
        typedef typename Matrix_d::IVector IVector_d;
        typedef typename Matrix<T_Config>::IVector IVector;

        // Ctor.
        MatrixColoring( AMG_Config &cfg, const std::string &cfg_scope);

        // Constructing from host matrix coloring
        MatrixColoring( const MatrixColoring<TConfig_h> &a) {  this->copy(a); }

        // Constructing from device matrix coloring
        MatrixColoring( const MatrixColoring<TConfig_d> &a) { this->copy(a); }

        virtual ~MatrixColoring();

        virtual void colorMatrix( Matrix<T_Config> &A ) {}; //TODO: Make an interface, only implementations are able to color the matrix, not the interface
        virtual void colorMatrixUsingAggregates( Matrix<T_Config> &A, IVector &R_row_offsets, IVector &R_col_indices, IVector &aggregates ) { colorMatrix( A ); }

        virtual void createColorArrays(Matrix<T_Config> &A);
        void assertColoring(Matrix<TConfig> &A, IVector &aggregates ); //prints some useful coloring quality info

        inline size_t bytes()
        {
            return m_row_colors.bytes() +
                   m_sorted_rows_by_color.bytes() +
                   m_offsets_rows_per_color.bytes() +
                   m_offsets_rows_per_color_separation.bytes();
        };

        inline int getNumColors() {return m_num_colors;}
        inline int getNumColors() const {return m_num_colors;}

        inline const IVector &getRowColors() {return m_row_colors;}
        inline const IVector &getRowColors() const {return m_row_colors;}

        inline void setRowColors(IVector_h &row_colors) {m_row_colors = row_colors;}
        inline void setRowColors(IVector_d &row_colors) {m_row_colors = row_colors;}

        inline void setNumColors(int num_colors) {m_num_colors = num_colors;}

        inline const IVector &getSortedRowsByColor() {return m_sorted_rows_by_color;}
        inline const IVector &getSortedRowsByColor() const {return m_sorted_rows_by_color;}

        inline const IVector_h &getOffsetsRowsPerColor() {return m_offsets_rows_per_color;}
        inline const IVector_h &getOffsetsRowsPerColor() const {return m_offsets_rows_per_color;}

        inline const IVector_h &getSeparationOffsetsRowsPerColor() {return m_offsets_rows_per_color_separation;}
        inline const IVector_h &getSeparationOffsetsRowsPerColor() const {return m_offsets_rows_per_color_separation;}

        inline unsigned int getColoringLevel() {return m_coloring_level;}
        inline unsigned int getColoringLevel() const {return m_coloring_level;}

        template<class MatrixColoringType>
        inline void copy(const MatrixColoringType &a)
        {
            m_coloring_level = a.getColoringLevel();
            m_num_colors = a.getNumColors();
            m_row_colors = a.getRowColors();
            m_sorted_rows_by_color = a.getSortedRowsByColor();
            m_offsets_rows_per_color = a.getOffsetsRowsPerColor();
            m_ref_count = 1;
        }

        void retain() {++m_ref_count;}
        bool release() {return --m_ref_count == 0;}

    protected:

        int m_coloring_level;
        int m_num_colors;
        int m_ref_count;
        int m_halo_coloring;
        int m_boundary_coloring;

        IVector m_row_colors;
        IVector m_sorted_rows_by_color;
        IVector_h m_offsets_rows_per_color;
        IVector_h m_offsets_rows_per_color_separation;


};

template<class T_Config>
class MatrixColoringFactory
{
    public:
        virtual MatrixColoring<T_Config> *create(AMG_Config &cfg, const std::string &cfg_scope) = 0;
        virtual ~MatrixColoringFactory() {};

        /********************************************
         * Register a MatrixColoring class with key "name"
         *******************************************/
        static void registerFactory(std::string name, MatrixColoringFactory<T_Config> *f);

        /********************************************
          * Unregister a MatrixColoring class with key "name"
          *******************************************/
        static void unregisterFactory(std::string name);

        /********************************************
         * Unregister all the MatrixColoring classes
         *******************************************/
        static void unregisterFactories( );

        /*********************************************
        * Allocates MatrixColoring based on cfg
        *********************************************/
        static MatrixColoring<T_Config> *allocate(AMG_Config &cfg, const std::string &cfg_scope);

        typedef typename std::map<std::string, MatrixColoringFactory<T_Config>*>::const_iterator Iterator;

        static Iterator getIterator() { return getFactories().begin(); };
        static bool isIteratorLast(const Iterator &iter) { if ( iter == getFactories().end() ) return true; else return false; };

    private:
        static std::map<std::string, MatrixColoringFactory<T_Config>*> &getFactories( );
};

} // namespace amgx
