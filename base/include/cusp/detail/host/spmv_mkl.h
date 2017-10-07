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

#include <thrust/functional.h>
#include <cusp/detail/functional.h>
#include <cusp/array1d.h>
#include <cusp/array2d.h>

#include <mkl_spblas.h>
#include <mkl_types.h>

namespace cusp
{
namespace detail
{
namespace host
{

//////////////
// COO SpMV //
//////////////

// TODO Implement spmv_coo method compatible with HYB SpMV
template <typename Matrix,
          typename Vector1,
          typename Vector2,
          typename UnaryFunction,
          typename BinaryFunction1,
          typename BinaryFunction2>
void spmv_coo(const Matrix&  A,
              const Vector1& x,
                    Vector2& y,
              UnaryFunction   initialize,
              BinaryFunction1 combine,
              BinaryFunction2 reduce)
{
    typedef typename Matrix::index_type  IndexType;
    typedef typename Vector2::value_type ValueType;

    for(IndexType i = 0; i < A.num_rows; i++)
        y[i] = initialize(y[i]);

    for(IndexType n = 0; n < A.num_entries; n++)
    {
        const IndexType& i   = A.row_indices[n];
        const IndexType& j   = A.column_indices[n];
        const ValueType& Aij = A.values[n];
        const ValueType& xj  = x[j];

        y[i] = reduce(y[i], combine(Aij, xj));
    }
}

template<typename IndexType>
void spmv_coo( 	char * transa,
		IndexType * num_rows,
		IndexType * num_cols,
		float * A_values,
		IndexType * row_indices,
		IndexType * column_indices,
		IndexType * num_entries,
		float * x,
		float * y) 
{
    mkl_cspblas_scoogemv(transa, num_rows, A_values, row_indices, column_indices, num_entries, x, y); 
}

template<typename IndexType>
void spmv_coo( 	char * transa,
		IndexType * num_rows,
		IndexType * num_cols,
		double * A_values,
		IndexType * row_indices,
		IndexType * column_indices,
		IndexType * num_entries,
		double * x,
		double * y) 
{
    mkl_cspblas_dcoogemv(transa, num_rows, A_values, row_indices, column_indices, num_entries, x, y); 
}

template <typename Matrix,
          typename Vector1,
          typename Vector2>
void spmv_coo(const Matrix&  A,
              const Vector1& x,
                    Vector2& y)
{
    typedef typename Matrix::index_type IndexType;
    typedef typename Matrix::value_type ValueType;

    char transa = 'N';

    MKL_INT * m   = (MKL_INT *) &A.num_rows;
    MKL_INT * n   = (MKL_INT *) &A.num_cols;
    MKL_INT * nnz = (MKL_INT *) &A.num_entries;

    ValueType * V = (ValueType *) thrust::raw_pointer_cast(&A.values[0]);
    ValueType * X = (ValueType *) thrust::raw_pointer_cast(&x[0]);
    ValueType * Y = (ValueType *) thrust::raw_pointer_cast(&y[0]);

    // Intel MKL sparse matrix functions only support 32-bit index types
    // TODO Check if  matrix dimensions allow conversion to 32-bit indexing (i.e. num_cols < 2^32)
    if( sizeof(IndexType) > 4 )
    {
	cusp::array1d<MKL_INT,cusp::host_memory> I_32(A.row_indices);
	cusp::array1d<MKL_INT,cusp::host_memory> J_32(A.column_indices);

    	MKL_INT * pI_32 = (MKL_INT *) thrust::raw_pointer_cast(&I_32[0]);
	MKL_INT * pJ_32 = (MKL_INT *) thrust::raw_pointer_cast(&J_32[0]);

        spmv_coo(&transa,m,n,V,pI_32,pJ_32,nnz,X,Y);
    }
    else
    {
    	MKL_INT * I = (MKL_INT *) thrust::raw_pointer_cast(&A.row_indices[0]);
	MKL_INT * J = (MKL_INT *) thrust::raw_pointer_cast(&A.column_indices[0]);

        spmv_coo(&transa,m,n,V,I,J,nnz,X,Y);
    }
}


//////////////
// CSR SpMV //
//////////////
template<typename IndexType>
void spmv_csr( 	char * transa,
		IndexType * num_rows,
		IndexType * num_cols,
		float * A_values,
		IndexType * column_indices,
		IndexType * row_offsets,
		float * x,
		float * y) 
{
    mkl_cspblas_scsrgemv(transa, num_rows, A_values, row_offsets, column_indices, x, y); 
}

template<typename IndexType>
void spmv_csr( 	char * transa,
		IndexType * num_rows,
		IndexType * num_cols,
		double * A_values,
		IndexType * column_indices,
		IndexType * row_offsets,
		double * x,
		double * y) 
{
    mkl_cspblas_dcsrgemv(transa, num_rows, A_values, row_offsets, column_indices, x, y); 
}

template <typename Matrix,
          typename Vector1,
          typename Vector2>
void spmv_csr(const Matrix&  A,
              const Vector1& x,
                    Vector2& y)
{
    typedef typename Matrix::index_type IndexType;
    typedef typename Matrix::value_type ValueType;

    char transa = 'N';

    MKL_INT * m   = (MKL_INT *) &A.num_rows;
    MKL_INT * n   = (MKL_INT *) &A.num_cols;

    ValueType * V = (ValueType *) thrust::raw_pointer_cast(&A.values[0]);
    ValueType * X = (ValueType *) thrust::raw_pointer_cast(&x[0]);
    ValueType * Y = (ValueType *) thrust::raw_pointer_cast(&y[0]);

    // Intel MKL sparse matrix functions only support 32-bit index types
    // TODO Check if  matrix dimensions allow conversion to 32-bit indexing (i.e. num_cols < 2^32)
    if( sizeof(IndexType) > 4 )
    {
	cusp::array1d<MKL_INT,cusp::host_memory> P_32(A.row_offsets);
	cusp::array1d<MKL_INT,cusp::host_memory> J_32(A.column_indices);

    	MKL_INT * pP_32 = (MKL_INT *) thrust::raw_pointer_cast(&P_32[0]);
	MKL_INT * pJ_32 = (MKL_INT *) thrust::raw_pointer_cast(&J_32[0]);

        spmv_csr(&transa,m,n,V,pJ_32,pP_32,X,Y);
    }
    else
    {
    	MKL_INT * P = (MKL_INT *) thrust::raw_pointer_cast(&A.row_offsets[0]);
	MKL_INT * J = (MKL_INT *) thrust::raw_pointer_cast(&A.column_indices[0]);

        spmv_csr(&transa,m,n,V,J,P,X,Y);
    }
}


//////////////
// DIA SpMV //
//////////////

// TODO Reconcile DIA data layout between CUSP and MKL SpMV kernels
template <typename Matrix,
          typename Vector1,
          typename Vector2,
          typename UnaryFunction,
          typename BinaryFunction1,
          typename BinaryFunction2>
void spmv_dia(const Matrix&  A,
              const Vector1& x,
                    Vector2& y,
              UnaryFunction   initialize,
              BinaryFunction1 combine,
              BinaryFunction2 reduce)
{
    typedef typename Matrix::index_type  IndexType;
    typedef typename Vector2::value_type ValueType;

    const IndexType& num_diagonals = A.values.num_cols;

    for(IndexType i = 0; i < A.num_rows; i++)
        y[i] = initialize(y[i]);

    for(IndexType i = 0; i < num_diagonals; i++)
    {
        const IndexType& k = A.diagonal_offsets[i];

        const IndexType& i_start = std::max<IndexType>(0, -k);
        const IndexType& j_start = std::max<IndexType>(0,  k);

        // number of elements to process in this diagonal
        const IndexType N = std::min(A.num_rows - i_start, A.num_cols - j_start);

        for(IndexType n = 0; n < N; n++)
        {
            const ValueType& Aij = A.values(i_start + n, i);

            const ValueType& xj = x[j_start + n];
                  ValueType& yi = y[i_start + n];
    
            yi = reduce(yi, combine(Aij, xj));
        }
    }
}

template <typename Matrix,
          typename Vector1,
          typename Vector2>
void spmv_dia(const Matrix&  A,
              const Vector1& x,
                    Vector2& y)
{
    typedef typename Vector2::value_type ValueType;

    spmv_dia(A, x, y,
             cusp::detail::zero_function<ValueType>(),
             thrust::multiplies<ValueType>(),
             thrust::plus<ValueType>());
}


//////////////
// ELL SpMV //
//////////////
template <typename Matrix,
          typename Vector1,
          typename Vector2>
void spmv_ell(const Matrix&  A,
              const Vector1& x,
                    Vector2& y)
{
    typedef typename Matrix::index_type IndexType;
    typedef typename Matrix::value_type ValueType;

    char transa = 'N';
    
    MKL_INT m   = (MKL_INT) A.num_rows;
    MKL_INT n   = (MKL_INT) A.num_cols;

    // MKL uses row major data layout
    cusp::array2d<MKL_INT, cusp::host_memory, cusp::row_major> column_indices(A.column_indices);
    cusp::array2d<ValueType, cusp::host_memory, cusp::row_major> values(A.values);

    MKL_INT num_entries_per_row = column_indices.num_cols;

    ValueType * V = (ValueType *) thrust::raw_pointer_cast(&values(0,0));
    ValueType * X = (ValueType *) thrust::raw_pointer_cast(&x[0]);
    ValueType * Y = (ValueType *) thrust::raw_pointer_cast(&y[0]);

    cusp::array1d<MKL_INT,cusp::host_memory> row_offsets(A.num_rows+1);
    for( IndexType index = 0; index < row_offsets.size(); index++ )
	row_offsets[index] = index*num_entries_per_row;

    MKL_INT * P = (MKL_INT *) thrust::raw_pointer_cast(&row_offsets[0]);
    MKL_INT * J = (MKL_INT *) thrust::raw_pointer_cast(&column_indices(0,0));

    spmv_csr(&transa,&m,&n,V,J,P,X,Y);
}

} // end namespace host
} // end namespace detail
} // end namespace cusp

