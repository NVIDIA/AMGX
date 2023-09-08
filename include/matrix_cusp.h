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

#include <matrix.h>

#include <cusp/detail/format_utils.h>

#include <thrust/sort.h>
#include <thrust/iterator/zip_iterator.h>
#include <error.h>
#include <permute.h>

namespace amgx
{

template<class TConfig, class Format>
class MatrixCusp
{
        Matrix<TConfig> *pM;
        DEFINE_VECTOR_TYPES
    public:
        typedef typename TConfig::IndPrec  index_type;
        typedef typename TConfig::MatPrec  value_type;
        typedef Format   format;
        typedef typename TConfig::MemSpace memory_space;

        typedef  IVector row_offsets_array_type;
        typedef  IVector row_indices_array_type;
        typedef  IVector column_indices_array_type;
        typedef  MVector values_array_type;

        index_type num_rows;
        index_type num_cols;
        index_type num_entries;

        row_offsets_array_type &row_offsets;
        row_indices_array_type &row_indices;
        column_indices_array_type &column_indices;
        values_array_type &values;

        MatrixCusp() : pM(NULL), num_rows(0), num_cols(0), num_entries(0), row_offsets(row_offsets_array_type()), row_indices(row_indices_array_type()), column_indices(column_indices_array_type()), values(values_array_type()) {}

        MatrixCusp(index_type num_rows_, index_type num_cols_, index_type num_entries_, row_offsets_array_type &row_offsets_, row_indices_array_type &row_indices_, column_indices_array_type &column_indices_, values_array_type &values_)
            : pM(NULL), num_rows(num_rows_), num_cols(num_cols_), num_entries(num_entries_), row_offsets(row_offsets_), row_indices(row_indices_), column_indices(column_indices_), values(values_) {}

        MatrixCusp(Matrix<TConfig> *_pM)  : pM(_pM), num_rows(_pM->get_num_rows()), num_cols(_pM->get_num_cols()), num_entries(_pM->get_num_nz()), row_offsets(_pM->row_offsets), row_indices(_pM->row_indices), column_indices(_pM->col_indices), values(_pM->values)
        {
            //if (_pM->get_block_size() != 1)
            //   FatalError("Cannot create cusp matrix based on a block matrix", AMGX_ERR_NOT_SUPPORTED_BLOCKSIZE);
            /*if (!_pM->is_initialized())
                FatalError("Trying to create cusp matrix over uninitialized matrix", AMGX_ERR_BAD_PARAMETERS);*/
        }

        inline void addProps(unsigned int new_props)
        {
            pM->addProps(new_props);
        }

        inline bool is_initialized() const {return pM->m_initialized();}
        inline void set_initialized(int new_value) { pM->set_initialized(new_value);}

        inline void delProps(unsigned int rem_props)
        {
            pM->delProps(rem_props);
        }

        void resize(size_t _num_rows, size_t _num_cols, size_t _num_entries)
        {
            num_rows = _num_rows;
            num_cols = _num_cols;
            num_entries = _num_entries;
            pM->resize(_num_rows, _num_cols, _num_entries, 0 );
        }
        // sort matrix elements by row index
        void sort_by_row(void)
        {
            cusp::detail::sort_by_row(pM->row_indices, pM->col_indices, pM->values);
        }

        //sort matrix elements by row and column
        void sort_by_row_and_column()
        {
            if ( pM->get_block_dimx() == 1 && pM->get_block_dimy() == 1 )
            {
                cusp::detail::sort_by_row_and_column(pM->row_indices, pM->col_indices, pM->values);
                return;
            }

            column_indices_array_type element_permutation(num_entries);
            thrust_wrapper::sequence(element_permutation.begin(), element_permutation.end());
            cusp::detail::sort_by_row_and_column(pM->row_indices, pM->col_indices, element_permutation);
            cudaCheckError();
            MVector temp_values;
            temp_values.resize(pM->values.size());
            temp_values.set_block_dimx(pM->values.get_block_dimx());
            temp_values.set_block_dimy(pM->values.get_block_dimy());
            amgx::unpermuteVector(pM->values, temp_values, element_permutation, (pM->get_num_nz()) * (pM->get_block_size()));
            pM->values.swap(temp_values);
            temp_values.clear();
            temp_values.shrink_to_fit();
        }

        // determine whether matrix elements are sorted by row index
        bool is_sorted_by_row(void)
        {
            return amgx::thrust::is_sorted(row_indices.begin(), row_indices.end());
        }

        // determine whether matrix elements are sorted by row and column index
        bool is_sorted_by_row_and_column(void)
        {
            return amgx::thrust::is_sorted
                   (amgx::thrust::make_zip_iterator(amgx::thrust::make_tuple(row_indices.begin(), column_indices.begin())),
                    amgx::thrust::make_zip_iterator(amgx::thrust::make_tuple(row_indices.end(),   column_indices.end())));
        }
};
} //end namespace amgx

