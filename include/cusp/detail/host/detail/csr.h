/*
 *  Copyright 2008-2009 NVIDIA Corporation
 *
 *  Licensed under the Apache License, Version 2.0 (the "License");
 *  you may not use this file except in compliance with the License.
 *  You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 *  Unless required by applicable law or agreed to in writing, software
 *  distributed under the License is distributed on an "AS IS" BASIS,
 *  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  See the License for the specific language governing permissions and
 *  limitations under the License.
 */

#pragma once

#include <cusp/array1d.h>

namespace cusp
{
namespace detail
{
namespace host
{
namespace detail
{

template <typename Matrix1,
          typename Matrix2,
          typename Matrix3,
          typename BinaryFunction>
void csr_transform_elementwise(const Matrix1& A,
                               const Matrix2& B,
                                     Matrix3& C,
                                     BinaryFunction op)
{
    //Method that works for duplicate and/or unsorted indices

    typedef typename Matrix3::index_type IndexType;
    typedef typename Matrix3::value_type ValueType;

    cusp::array1d<IndexType,cusp::host_memory>  next(A.num_cols, IndexType(-1));
    cusp::array1d<ValueType,cusp::host_memory> A_row(A.num_cols, ValueType(0));
    cusp::array1d<ValueType,cusp::host_memory> B_row(A.num_cols, ValueType(0));
   
    cusp::csr_matrix<IndexType,ValueType,cusp::host_memory> temp(A.num_rows, A.num_cols, A.num_entries + B.num_entries);

    size_t nnz = 0;

    temp.row_offsets[0] = 0;
    
    for(size_t i = 0; i < A.num_rows; i++)
    {
        IndexType head   = -2;
        IndexType length =  0;
    
        //add a row of A to A_row
        IndexType i_start = A.row_offsets[i];
        IndexType i_end   = A.row_offsets[i + 1];
        for(IndexType jj = i_start; jj < i_end; jj++)
        {
            IndexType j = A.column_indices[jj];
    
            A_row[j] += A.values[jj];
    
            if(next[j] == -1) { next[j] = head; head = j; length++; }
        }
    
        //add a row of B to B_row
        i_start = B.row_offsets[i];
        i_end   = B.row_offsets[i + 1];
        for(IndexType jj = i_start; jj < i_end; jj++)
        {
            IndexType j = B.column_indices[jj];
    
            B_row[j] += B.values[jj];
    
            if(next[j] == -1) { next[j] = head; head = j; length++;  }
        }
   
        // scan through columns where A or B has 
        // contributed a non-zero entry
        for(IndexType jj = 0; jj < length; jj++)
        {
            ValueType result = op(A_row[head], B_row[head]);
    
            if(result != 0)
            {
                temp.column_indices[nnz] = head;
                temp.values[nnz]         = result;
                nnz++;
            }
    
            IndexType prev = head;  head = next[head];  next[prev]  = -1;

            A_row[prev] =  0;                             
            B_row[prev] =  0;
        }

        temp.row_offsets[i + 1] = nnz;
    }

    // TODO replace with destructive assignment?

    temp.resize(A.num_rows, A.num_cols, nnz);
    cusp::copy(temp, C);
} // csr_transform_elementwise


template <typename Array1, typename Array2,
          typename Array3, typename Array4>
size_t spmm_csr_pass1(const size_t num_rows, const size_t num_cols,
                         const Array1& A_row_offsets, const Array2& A_column_indices,
                         const Array3& B_row_offsets, const Array4& B_column_indices)
{
    typedef typename Array1::value_type IndexType1;
    typedef typename Array2::value_type IndexType2;
    
    cusp::array1d<size_t, cusp::host_memory> mask(num_cols, static_cast<size_t>(-1));

    // Compute nnz in C (including explicit zeros)
    size_t num_nonzeros = 0;

    for(size_t i = 0; i < num_rows; i++)
    {
        for(IndexType1 jj = A_row_offsets[i]; jj < A_row_offsets[i+1]; jj++)
        {
            IndexType1 j = A_column_indices[jj];

            for(IndexType2 kk = B_row_offsets[j]; kk < B_row_offsets[j+1]; kk++)
            {
                IndexType2 k = B_column_indices[kk];

                if(mask[k] != i)
                {
                    mask[k] = i;                        
                    num_nonzeros++;
                }
            }
        }
    }

    return num_nonzeros;
}

template <typename Array1, typename Array2, typename Array3,
          typename Array4, typename Array5, typename Array6,
          typename Array7, typename Array8, typename Array9>
size_t spmm_csr_pass2(const size_t num_rows, const size_t num_cols,
                         const Array1& A_row_offsets, const Array2& A_column_indices, const Array3& A_values,
                         const Array4& B_row_offsets, const Array5& B_column_indices, const Array6& B_values,
                               Array7& C_row_offsets,       Array8& C_column_indices,       Array9& C_values)
{
    typedef typename Array7::value_type IndexType;
    typedef typename Array9::value_type ValueType;

    size_t num_nonzeros = 0;

    const IndexType unseen = static_cast<IndexType>(-1);
    const IndexType init   = static_cast<IndexType>(-2);  

    // Compute entries of C
    cusp::array1d<IndexType,cusp::host_memory> next(num_cols, unseen);
    cusp::array1d<ValueType,cusp::host_memory> sums(num_cols, ValueType(0));
    
    num_nonzeros = 0;
    
    C_row_offsets[0] = 0;
    
    for(size_t i = 0; i < num_rows; i++)
    {
        IndexType head   = init;
        IndexType length =    0;
    
        IndexType jj_start = A_row_offsets[i];
        IndexType jj_end   = A_row_offsets[i+1];

        for(IndexType jj = jj_start; jj < jj_end; jj++)
        {
            IndexType j = A_column_indices[jj];
            ValueType v = A_values[jj];
    
            IndexType kk_start = B_row_offsets[j];
            IndexType kk_end   = B_row_offsets[j+1];

            for(IndexType kk = kk_start; kk < kk_end; kk++)
            {
                IndexType k = B_column_indices[kk];
    
                sums[k] += v * B_values[kk];
    
                if(next[k] == unseen)
                {
                    next[k] = head;                        
                    head  = k;
                    length++;
                }
            }
        }
    
        for(IndexType jj = 0; jj < length; jj++)
        {
            if(sums[head] != ValueType(0))
            {
                C_column_indices[num_nonzeros] = head;
                C_values[num_nonzeros]         = sums[head];
                num_nonzeros++;
            }
    
            IndexType temp = head; head = next[head];
    
            // clear arrays
            next[temp] = unseen; 
            sums[temp] = ValueType(0);                              
        }
    
        C_row_offsets[i+1] = num_nonzeros;
    }
    
    // XXX note: entries of C are unsorted within each row

    return num_nonzeros;
}

template <typename Matrix1,
          typename Matrix2,
          typename Matrix3>
void spmm_csr(const Matrix1& A,
              const Matrix2& B,
                    Matrix3& C)
{
    typedef typename Matrix3::index_type IndexType;

    IndexType num_nonzeros = 
        spmm_csr_pass1(A.num_rows, B.num_cols,
                       A.row_offsets, A.column_indices,
                       B.row_offsets, B.column_indices);
                         
    // Resize output
    C.resize(A.num_rows, B.num_cols, num_nonzeros);
    
    num_nonzeros =
      spmm_csr_pass2(A.num_rows, B.num_cols,
                     A.row_offsets, A.column_indices, A.values,
                     B.row_offsets, B.column_indices, B.values,
                     C.row_offsets, C.column_indices, C.values);

    // Resize output again since pass2 omits explict zeros
    C.resize(A.num_rows, B.num_cols, num_nonzeros);
}

} // end namespace detail
} // end namespace host
} // end namespace detail
} // end namespace cusp

