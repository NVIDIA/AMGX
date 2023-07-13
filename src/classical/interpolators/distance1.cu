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

#include <thrust/transform.h>
#include <thrust/transform_scan.h>
#include <thrust_wrapper.h>
#include <classical/interpolators/distance1.h>
#include <classical/interpolators/common.h>
#include <basic_types.h>
#include <types.h>
#include <cutil.h>
#include <fstream>
#include <set>
#include <vector>
#include <algorithm>
#include <assert.h>

#include <thrust/count.h>

namespace amgx
{

namespace detail
{

template< class CsrMatrixType >
struct find_row_diagonal
{
//    typedef typename CsrMatrixType::values_array_type::value_type value_type;
    typedef typename CsrMatrixType::value_type value_type;
    const CsrMatrixType &my_A;
    find_row_diagonal( const CsrMatrixType &A ) : my_A( A ) {}

    // Operator to extract the diagonal value.
    value_type operator()( int irow ) const
    {
        for ( int i = my_A.row_offsets[irow], end = my_A.row_offsets[irow + 1] ; i < end ; ++i )
            if ( my_A.col_indices[i] == irow )
            {
                return my_A.values[i];
            }

        // assert( false );
        return value_type( 0 );
    }
};

template< class CsrMatrixType >
struct count_nnz_per_row : public thrust::unary_function<int, int>
{
    const CsrMatrixType &my_A;
    static const AMGX_VecPrecision vecPrec = CsrMatrixType::TConfig::vecPrec;
    static const AMGX_MatPrecision matPrec = CsrMatrixType::TConfig::matPrec;
    static const AMGX_IndPrecision indPrec = CsrMatrixType::TConfig::indPrec;
    typedef Vector<TemplateConfig<AMGX_host, AMGX_vecBool, matPrec, indPrec> > BVector;
    typedef Vector<TemplateConfig<AMGX_host, AMGX_vecInt, matPrec, indPrec> > IntVector;

    const  BVector &my_are_sc;
    const  IntVector &my_cf_map;
    count_nnz_per_row( const CsrMatrixType &A,
                       const BVector &are_sc,
                       const IntVector &cf_map ) :
        my_A( A ),
        my_are_sc( are_sc ),
        my_cf_map( cf_map )
    {}

    // Operator to extract the diagonal value.
    int operator()( int i ) const
    {
        if ( my_cf_map[i] >= 0 ) // If it's a coarse node, only one NZ element.
        {
            return 1;
        }

        // Count the number of strongly connected coarse nodes.
        int count = 0;

        for ( int itj = my_A.row_offsets[i] ; itj < my_A.row_offsets[i + 1] ; ++itj )
        {
            int j = my_A.col_indices[itj];

            if ( my_cf_map[j] >= 0 && my_are_sc[itj] )
            {
                count++;
            }
        }

        return count;
    }
};

template< class CsrMatrixType >
struct compute_weights
{
    typedef typename CsrMatrixType::value_type value_type;
    typedef typename CsrMatrixType::MVector VVector;
    static const AMGX_VecPrecision vecPrec = CsrMatrixType::TConfig::vecPrec;
    static const AMGX_MatPrecision matPrec = CsrMatrixType::TConfig::matPrec;
    static const AMGX_IndPrecision indPrec = CsrMatrixType::TConfig::indPrec;
    typedef Vector<TemplateConfig<AMGX_host, AMGX_vecBool, matPrec, indPrec> > BVector;
    typedef Vector<TemplateConfig<AMGX_host, AMGX_vecInt, matPrec, indPrec> > IntVector;
    const CsrMatrixType &my_A;
    const VVector &my_diag;
    const BVector &my_are_sc;
    const IntVector &my_cf_map;
    IntVector *my_edges_markers;
    CsrMatrixType &my_P;

    // Constructor.
    compute_weights( const CsrMatrixType &A,
                     const VVector &diag,
                     const BVector &are_sc,
                     const IntVector &cf_map,
                     IntVector *edges_markers,
                     CsrMatrixType &P ) :
        my_A( A ),
        my_diag( diag ),
        my_are_sc( are_sc ),
        my_cf_map( cf_map ),
        my_edges_markers( edges_markers ),
        my_P( P )
    { }

    // Get the thread index. Always 0 if OpenMP is off.
    inline
    int get_thread_id( ) const
    {
#if( THRUST_DEVICE_BACKEND == THRUST_DEVICE_BACKEND_OMP )
        return omp_get_thread_num( );
#else
        return 0;
#endif
    }

    // Operator to find the weights.
    void operator()( int i )
    {
        // Skip coarse nodes.
        int nnz = my_P.row_offsets[i];
        int id = my_cf_map[i];

        if ( id >= 0 )
        {
            my_P.col_indices[nnz] = id;
            my_P.values[nnz] = value_type( 1 );
            return;
        }

        assert( my_A.row_offsets[i] < my_A.row_offsets[i + 1] );
        const int strong_marker = -2;

        // Initialize the F-C connections to 0 and mark F-F connections.
        for ( int itj = my_A.row_offsets[i] ; itj < my_A.row_offsets[i + 1] ; ++itj )
        {
            if ( !my_are_sc[itj] )
            {
                continue;
            }

            int j = my_A.col_indices[itj];

            if ( my_cf_map[j] == FINE ) // It's a F-F connection, mark it.
            {
                assert( my_cf_map[j] < 0 );
                my_edges_markers[get_thread_id( )][j] = strong_marker;
                continue;
            }

            assert( nnz < my_P.row_offsets[i + 1] );
            my_edges_markers[get_thread_id( )][j] = nnz; // It's a F-C connection, mark the coarse end point.
            my_P.col_indices[nnz] = my_cf_map[j];
            my_P.values[nnz] = value_type( 0 );
            nnz++;
        }

        // Make sure we exhausted all the indices.
        assert( nnz == my_P.row_offsets[i + 1] );
        // Reset NNZ variable.
        nnz = my_P.row_offsets[i];
        // The diagonal element a_ii.
        value_type diag = my_diag[i];

        // Loop over the neighbours of the node n_i to add contributions.
        for ( int itj = my_A.row_offsets[i] ; itj < my_A.row_offsets[i + 1] ; ++itj )
        {
            int j = my_A.col_indices[itj];

            if ( i == j ) // Skip the diagonal.
            {
                continue;
            }

            // 1st case: j is a coarse node and it strongly influences i, so we accumulate a_ij.
            if ( my_edges_markers[get_thread_id( )][j] >= nnz )
            {
                int itk = my_edges_markers[get_thread_id( )][j];
                my_P.values[itk] += my_A.values[itj];
                continue;
            }
            // 2nd case: j is a fine node and strongly influences i. We distribute a_ij to the coarse
            // nodes that strongly influence i (not the diagonal).
            else if ( my_edges_markers[get_thread_id( )][j] == strong_marker )
            {
                value_type sum( 0 );
                int diag_sign = std::signbit( my_diag[i] );

                for ( int itk = my_A.row_offsets[j] ; itk < my_A.row_offsets[j + 1] ; ++itk )
                {
                    int k = my_A.col_indices[itk];

                    if ( my_edges_markers[get_thread_id( )][k] >= nnz && std::signbit( my_A.values[itk] ) != diag_sign )
                    {
                        sum += my_A.values[itk];
                    }
                }

                if ( sum == value_type( 0 ) )
                {
                    diag += my_A.values[itj];
                    continue;
                }

                value_type d = my_A.values[itj] / sum;

                for ( int itk = my_A.row_offsets[j] ; itk < my_A.row_offsets[j + 1] ; ++itk )
                {
                    int it = my_edges_markers[get_thread_id( )][my_A.col_indices[itk]];

                    if ( it >= nnz && std::signbit( my_A.values[itk] ) != diag_sign )
                    {
                        my_P.values[it] += d * my_A.values[itk];
                    }
                }
            }
            else if ( my_cf_map[j] == FINE )
            {
                diag += my_A.values[itj];
            }
        }

        if ( diag == value_type( 0 ) )
        {
            diag = my_diag[i];

            if ( diag == value_type( 0 ) ) // REALLY ?
            {
                diag = value_type( 1 );
            }
        }

        assert( diag != value_type( 0 ) );
        diag = value_type( -1.0 ) / diag;

        for ( ; nnz < my_P.row_offsets[i + 1] ; ++nnz )
        {
            my_P.values[nnz] *= diag;
        }

        // strong_marker--;
    }
};

}

/*************************************************************************
* create the interpolation matrix P
************************************************************************/
template <AMGX_VecPrecision t_vecPrec, AMGX_MatPrecision t_matPrec, AMGX_IndPrecision t_indPrec>
void Distance1_Interpolator<TemplateConfig<AMGX_host, t_vecPrec, t_matPrec, t_indPrec> >::generateInterpolationMatrix_1x1(
    Matrix_h &A,
    IntVector &cf_map,
    BVector &are_sc,
    IntVector &wk,
    Matrix_h &P)
{
    // The diagonal of A.
    VVector diag( A.get_num_rows() );
    thrust::transform( thrust::make_counting_iterator<int>( 0 ),
                       thrust::make_counting_iterator<int>( A.get_num_rows() ),
                       diag.begin( ),
                       detail::find_row_diagonal<Matrix_h>( A ) );
    cudaCheckError();
    // Count the number of coarse points.
    int n_coarse = 0;

    for ( int i = 0 ; i < A.get_num_rows() ; ++i )
        if ( cf_map[i] >= 0) { n_coarse++; }

    // Count the number of non-zero elements per row of P.
    IntVector nnz_per_row( A.get_num_rows() + 1 );
    thrust::transform( thrust::make_counting_iterator<int>( 0 ),
                       thrust::make_counting_iterator<int>( A.get_num_rows() ),
                       nnz_per_row.begin( ),
                       detail::count_nnz_per_row<Matrix_h>( A, are_sc, cf_map ) );
    cudaCheckError();
    nnz_per_row[A.get_num_rows()] = 0;
    // Find the total number of non-zeroes. 
    // TODO: merge with the exclusive scan.
    int nnz = thrust::reduce( nnz_per_row.begin( ), nnz_per_row.end( ) );
    cudaCheckError();
    // Resize P so that there is enough space to store the non-zero elements.
    P.addProps(CSR);
    P.resize( A.get_num_rows(), n_coarse, nnz );
    // Compute row offsets of P.
    thrust::exclusive_scan( nnz_per_row.begin( ),
                            nnz_per_row.end( ),
                            P.row_offsets.begin( ) );
    cudaCheckError();
    // For each row we compute the weights.
#if( THRUST_DEVICE_BACKEND == THRUST_DEVICE_BACKEND_OMP )
    typedef IntVector IntArray1d;
    int nthreads = omp_get_max_threads( );
    size_t sz = nthreads * sizeof( IntArray1d );
    IntArray1d *edges_markers = reinterpret_cast<IntArray1d *>( ::operator new ( sz ) );

    for ( int i = 0 ; i < nthreads ; ++i )
    {
        new (edges_markers + i) IntArray1d( A.get_num_rows(), -1 );
    }

#else
    IntVector edges_markers_on_stack( A.get_num_rows(), -1 );
    IntVector *edges_markers = &edges_markers_on_stack;
#endif
    detail::compute_weights<Matrix_h> compute_fct( A, diag, are_sc, cf_map, edges_markers, P );

    thrust::for_each( thrust::host,
                      thrust::make_counting_iterator<int>( 0 ),
                      thrust::make_counting_iterator<int>( A.get_num_rows() ),
                      compute_fct );
    cudaCheckError();
#if( THRUST_DEVICE_BACKEND == THRUST_DEVICE_BACKEND_OMP )

    for ( int i = 0 ; i < nthreads ; ++i )
    {
        (edges_markers + i)->~IntArray1d( );
    }

    ::operator delete ( edges_markers );
#endif
}

/*************************************************************************
* create the interpolation matrix P on GPU
************************************************************************/

enum SETS {SET_STRONG_COARSE = 1,
           SET_WEAK_COARSE = 2,
           SET_STRONG_FINE = 4,
           SET_STRONG_FINE_NO_COMMON = 8,
           SET_WEAK_FINE = 16,
           SET_DIAGONAL = 32
          };

/*
 * grab the diagonal of the matrix
 */
/*template <typename IndexType, typename ValueType>
__global__
void getDiagonalKernel(const IndexType *offsets, const IndexType *column_indices,
                       const ValueType *values, const IndexType numRows, ValueType *diagonal)
{
  const int tIdx = threadIdx.x + blockDim.x*blockIdx.x;

  if (tIdx >= numRows) return;

  const int offset = offsets[tIdx];
  const int numj = offsets[tIdx+1]-offset;

  for (int j=offset; j < offset+numj; j++)
  {
    int jcol = column_indices[j];
    if (tIdx == jcol)
    {
      diagonal[tIdx] = values[j];
    }
  }
}*/

/*
 * Count the non-zeros in final output
 */
template <typename IndexType>
__global__
void numNonZerosVecKernel(const IndexType *offsets, const IndexType *column_indices,
                          const IndexType numRows, int *cf_map, const bool *s_con,
                          int *nonZerosVec)
{
    for (int tIdx = threadIdx.x + blockDim.x * blockIdx.x; tIdx < numRows; tIdx += blockDim.x * gridDim.x)
    {
        const int offset = offsets[tIdx];
        const int numj = offsets[tIdx + 1] - offset;
        int numOnRow = 0;

        // count strong coarse influences
        if (cf_map[tIdx] == FINE)
        {
            for (int j = offset; j < offset + numj; j++)
            {
                if (!s_con[j]) { continue; } // if weakly connected, ignore

                int jcol = column_indices[j];

                if (cf_map[jcol] >= 0) // if coarse
                {
                    numOnRow++;
                }
            }
        }
        else
        {
            numOnRow++;
        }

        nonZerosVec[tIdx] = numOnRow;
    }
}

/*
 * Calculate D_i vector
 * - Weakly influencing and strong fine with no common C-points to be added to diagonal term
 */
template <typename IndexType, typename ValueType>
__global__
void calculateDKernel(const IndexType *offsets, const IndexType *column_indices,
                      const ValueType *values, int num_rows, ValueType *diagonal,
                      int *set_fields, ValueType *D)
{
    for (int tIdx = threadIdx.x + blockDim.x * blockIdx.x; tIdx < num_rows; tIdx += blockDim.x * gridDim.x)
    {
        const int offset = offsets[tIdx];
        const int numk = offsets[tIdx + 1] - offset;
        ValueType sum = 0;
        int mask = SET_WEAK_FINE;

        for (int k = offset; k < offset + numk; k++)
        {
            if (set_fields[k] & mask)
            {
                sum += values[k];
            }
        }

        D[tIdx] += sum;
    }
}

/*
 * Calculate the k dependent term in w_i,j
 */
template <typename IndexType, typename ValueType>
__global__
void calculateBKernel(const IndexType *offsets, const IndexType *column_indices,
                      const ValueType *values, int num_rows, ValueType *diagonal, ValueType *diag_inc,
                      int *set_fields, int *cf_map, int *nonZerosVec, ValueType *B)
{
    for (int tIdx = threadIdx.x + blockDim.x * blockIdx.x; tIdx < num_rows; tIdx += blockDim.x * gridDim.x)
    {
        const int offset = offsets[tIdx];
        const int numj = offsets[tIdx + 1] - offset;
        int localNonZeros = 0;
        const int nonZerosOffset = nonZerosVec[tIdx];
        ValueType sum = 0.0, top = 0.0, bottom_sum = 0.0, a_ik;

        if (cf_map[tIdx] >= 0)   // If I am a coarse row
        {
            B[nonZerosOffset] = 1;
            return;
        }

        diag_inc[tIdx] = 0;
        int flag = 0;
        int first_j_loop = 0;
        ValueType tol = 1e-10;

        // otherwise
        for (int j = offset; j < offset + numj; j++)
        {
            sum = 0.0;
            bottom_sum = 0.0;

            if (set_fields[j] & SET_STRONG_COARSE) // we actually care about this j
            {
                int jcol = column_indices[j]; // this is the only part of j I need / care about.

                if (flag == 0)
                {
                    first_j_loop = 1;
                    flag = 1;
                }
                else
                {
                    first_j_loop = 0;
                }

                // run through the columns of row i, summing over all k
                for (int k = offset; k < offset + numj; k++)
                {
                    top = 0.0;
                    bottom_sum = 0.0;

                    if ( (set_fields[k] & SET_STRONG_FINE) || (set_fields[k] & SET_STRONG_FINE_NO_COMMON))   // k we're interested in
                    {
                        // part of the desired set - get column, diagonal and a_ik
                        int kcol = column_indices[k];
                        a_ik = values[k]; // get a_i,k term
                        // now find a_k,j
                        ValueType diag = diagonal[kcol];
                        int sgn = diag < 0.0 ? -1 : 1;

                        // now I need to find the j-th column on the k-th row. another loop (sigh)
                        for (int n = offsets[kcol]; n < offsets[kcol + 1]; n++)
                        {
                            int ncol = column_indices[n];

                            if (ncol == jcol)   // this is the entry I need
                            {
                                if (sgn * values[n] < 0)
                                {
                                    // add to the sum
                                    top = a_ik * values[n]; //  discard diagonal
                                }
                            }
                        }

                        // calculate the bottom sum - loop through columns of i
                        for (int m = offset; m < offset + numj; m++)
                        {
                            if (set_fields[m] & SET_STRONG_COARSE)
                            {
                                // this is a point I care about
                                int mcol = column_indices[m];

                                // now find the m-th column of the k-th row and add to sum
                                for (int n = offsets[kcol]; n < offsets[kcol + 1]; n++)
                                {
                                    int ncol = column_indices[n];

                                    if (ncol == mcol)
                                    {
                                        if (sgn * values[n] < 0)
                                        {
                                            bottom_sum += values[n];
                                        }
                                    }
                                }
                            }
                        }

                        if (abs(bottom_sum) < tol)
                        {
                            if (first_j_loop == 1)
                            {
                                //add it to the diagonal....
                                diag_inc[tIdx] += values[k];
                            }
                        }
                        else
                        {
                            sum += top / bottom_sum;
                        }
                    }
                }

                B[nonZerosOffset + localNonZeros] = sum;
                localNonZeros++;
            }
        }
    }
}

/*
 * count the number of strong coarse points per row
 * - was used for degugging, might be useful in the future
 */
template <typename IndexType>
__global__
void coarsePerRow(const IndexType *offsets, int num_rows, int *set_fields, int *output)
{
    for (int tIdx = threadIdx.x + blockDim.x * blockIdx.x; tIdx < num_rows; tIdx += blockDim.x * gridDim.x)
    {
        const int offset = offsets[tIdx];
        const int numj = offsets[tIdx + 1] - offset;
        int localCoarse = 0;

        for (int j = offset; j < offset + numj; j++)
        {
            if (set_fields[j] & SET_STRONG_COARSE)
            {
                localCoarse++;
            }
        }

        output[tIdx] = localCoarse;
    }
}

/*
 * Assemble final w_ij terms
 */
template <typename IndexType, typename ValueType>
__global__
void calculateWKernel(const IndexType *offsets, const IndexType *column_indices,
                      const ValueType *values, int num_rows, ValueType *diagonal,
                      int *set_fields, int *nonZerosVec, ValueType *D, ValueType *B,
                      ValueType *w, IndexType *Pcolumn_indices, int *cf_map)
{
    for (int tIdx = threadIdx.x + blockDim.x * blockIdx.x; tIdx < num_rows; tIdx += blockDim.x * gridDim.x)
    {
        const int offset = offsets[tIdx];
        const int numj = offsets[tIdx + 1] - offset;
        int localNonZeros = 0;
        const int nonZerosOffset = nonZerosVec[tIdx];
        ValueType diag = diagonal[tIdx];

        if (cf_map[tIdx] >= 0)
        {
            w[nonZerosOffset] = 1.0;
            Pcolumn_indices[nonZerosOffset] = cf_map[tIdx];
            continue;
        }

        for (int j = offset; j < offset + numj; j++)
        {
            if (set_fields[j] & SET_STRONG_COARSE)
            {
                ValueType aij = values[j];
                ValueType Bij = B[nonZerosOffset + localNonZeros];
                ValueType Di  = D[tIdx];
                Pcolumn_indices[nonZerosOffset + localNonZeros] = cf_map[column_indices[j]];
                double bottom_sum = (fabs(diag + Di) < 1e-10) ? 1. : diag + Di;
                w[nonZerosOffset + localNonZeros] = -1.0 / bottom_sum * (aij + Bij);
                localNonZeros++;
            }
        }
    }
}

template <typename IndexType>
__global__
void markCoarseEdgesKernel(const IndexType *offsets, const IndexType *column_indices,
                           const size_t numRows, const int *cf_map, const bool *s_con,
                           bool *coarseMark)
{
    for (int tIdx = threadIdx.x + blockDim.x * blockIdx.x; tIdx < numRows; tIdx += blockDim.x * gridDim.x)
    {
        const int rowBegin = offsets[tIdx];
        const int rowEnd = offsets[tIdx + 1];

        // loop over columns
        for (int j = rowBegin; j < rowEnd; j++)
        {
            int jcol = column_indices[j];

            if (tIdx == jcol) { continue; } // skip the diagonal

            if (cf_map[jcol] >= 0)
            {
                // mark this point
                coarseMark[j] = true;
            }
        }
    }
}

template <typename IndexType, typename ValueType>
__device__
bool sparseIntersectDevice(const IndexType *column_indices, ValueType *values,
                           int begin1, int end1, int begin2, int end2)
{
    int i1 = begin1;
    int i2 = begin2;

    // check arrays non zero length
    if (begin1 >= end1 || begin2 >= end2) { return false; }

    bool r = false;

    while ( true )
    {
        int col1 = column_indices[i1];
        int col2 = column_indices[i2];

        if (col1 == col2) // this is a match
        {
            if (values[i1] && values[i2])
            {
                r = true;
                break;
            }

            i1++;
            i2++;

            if ( i1 >= end1 || i2 >= end2 ) { break; }
        }
        else if (col1 > col2)
        {
            i2++;

            if (i2 >= end2) { break; }
        }
        else
        {
            i1++;

            if (i1 >= end1) { break; }
        }
    }

    return r;
}

template <typename IndexType>
__global__
void categoriseEdgesKernel(const IndexType *offsets, const IndexType *column_indices,
                           const size_t numRows, const int *cf_map, const bool *s_con,
                           bool *coarse_mark, int *set_fields)
{
    for (int tIdx = threadIdx.x + blockDim.x * blockIdx.x; tIdx < numRows; tIdx += blockDim.x * gridDim.x)
    {
        // set_fields[tIdx] = 1;
        if (cf_map[tIdx] >= 0) { continue; } // we don't need to categorise C-? edges

        const int rowBegin = offsets[tIdx];
        const int rowEnd = offsets[tIdx + 1];

        // we know I'm a fine row now - loop over non-zeros
        for (int j = rowBegin; j < rowEnd; j++)
        {
            int jcol = column_indices[j];

            if (jcol == tIdx) { continue; }

            // check F-C edges
            if (cf_map[jcol] >= 0)
            {
                if (s_con[j])
                {
                    set_fields[j] |= SET_STRONG_COARSE;
                }
                else
                {
                    set_fields[j] |= SET_WEAK_COARSE;
                }
            }
            else // F-F edges
            {
                if (!s_con[j])
                {
                    set_fields[j] |= SET_WEAK_FINE;
                }
                else // check if there exists a common C-point
                {
                    // intersect i and jcol
                    bool common_C = true;
                    common_C = sparseIntersectDevice(column_indices, coarse_mark, rowBegin, rowEnd,
                                                     offsets[jcol], offsets[jcol + 1]);

                    if (common_C)
                    {
                        set_fields[j] |= SET_STRONG_FINE;
                    }
                    else
                    {
                        set_fields[j] |= SET_STRONG_FINE_NO_COMMON;
                    }
                }
            }
        }
    }
}

template <AMGX_VecPrecision t_vecPrec, AMGX_MatPrecision t_matPrec, AMGX_IndPrecision t_indPrec>
void Distance1_Interpolator<TemplateConfig<AMGX_device, t_vecPrec, t_matPrec, t_indPrec> >::setSetsGPU(const Matrix_d &A, const BVector &s_con,
        const IntVector &cf_map,
        IntVector &set_fields)
{
    typedef typename Matrix_d::index_type IndexType;
    typedef typename Matrix_d::value_type ValueType;
    const int blocksize = 64;
    const int numBlocks = std::min( AMGX_GRID_MAX_SIZE, (int) (A.get_num_rows() / blocksize + 1) );
    // raw pointers for matrix structure
    const IndexType *offsets = A.row_offsets.raw();
    const IndexType *column_indices = A.col_indices.raw();
    // raw pointers for coarse/fine map and strength of connection array
    const bool *s_con_ptr = s_con.raw();
    const int *cf_map_ptr = cf_map.raw();
    int *set_fields_ptr = set_fields.raw();
    // memory for CoarseMark array
    BVector coarseMark(A.get_num_nz(), false);
    bool *coarseMark_ptr = coarseMark.raw();
    markCoarseEdgesKernel <<< numBlocks, blocksize>>>(offsets, column_indices, A.get_num_rows(),
            cf_map_ptr, s_con_ptr, coarseMark_ptr);
    cudaCheckError();
    // now we have the strong coarse edges marked, categorise all edges
    categoriseEdgesKernel <<< numBlocks, blocksize>>>(offsets, column_indices, A.get_num_rows(),
            cf_map_ptr, s_con_ptr, coarseMark_ptr,
            set_fields_ptr);
    cudaCheckError();
}

template <AMGX_VecPrecision t_vecPrec, AMGX_MatPrecision t_matPrec, AMGX_IndPrecision t_indPrec>
void Distance1_Interpolator<TemplateConfig<AMGX_device, t_vecPrec, t_matPrec, t_indPrec> >::generateInterpolationMatrix_1x1(Matrix_d &A,
        IntVector &cf_map,
        BVector &s_con,
        IntVector &scratch,
        Matrix_d &P)
{
    typedef typename Matrix_d::index_type IndexType;
    typedef typename Matrix_d::value_type ValueType;
    // Assemble all the raw pointers needed
    // Matrix A
    const IndexType *Aoffsets_ptr = A.row_offsets.raw();
    const IndexType *Acolumn_indices_ptr = A.col_indices.raw();
    const ValueType *Avalues_ptr = A.values.raw();
    // Coarse/Fine map, strength of connection, scratch
    int *cf_map_ptr = cf_map.raw();
    const bool *s_con_ptr = s_con.raw();
    int *scratch_ptr = scratch.raw();
    // temporary vectors - diagonal and non-zero offsets
    VVector diag(A.get_num_rows(), 0.0);
    IntVector nonZerosVec(A.get_num_rows() + 1, 0);
    ValueType *diag_ptr = diag.raw();
    int *nonZerosVec_ptr = nonZerosVec.raw();
    // NB - will get pointers for P *after* it has been initialised
    // choose blocksize. Using 1 thread / row for now
    const int blocksize = 64;
    const int numBlocks = std::min (AMGX_GRID_MAX_SIZE, (int) (A.get_num_rows() / blocksize + 1));
    const IndexType numRows = (IndexType) A.get_num_rows();
    // extract the diagonal of A
    find_diag_kernel_indexed_dia <<< numBlocks, blocksize>>>(A.get_num_rows(),
            A.diag.raw(),
            A.values.raw(),
            diag_ptr);
    /*  getDiagonalKernel<<<numBlocks,blocksize>>>(Aoffsets_ptr,Acolumn_indices_ptr,Avalues_ptr,
                                                 numRows,diag_ptr);*/
    cudaCheckError();
    // count the number of non-zeros in final output per row
    numNonZerosVecKernel <<< numBlocks, blocksize>>>(Aoffsets_ptr, Acolumn_indices_ptr, numRows,
            cf_map_ptr, s_con_ptr, nonZerosVec_ptr);
    cudaCheckError();
    // now I have the non-zeros per row for matrix P, count the non-zeros on each row
    // to get total NNZ
    // sum non-zeros per row
    int NNZidx = thrust_wrapper::reduce(nonZerosVec.begin(), nonZerosVec.end());
    cudaCheckError();
    thrust_wrapper::exclusive_scan(nonZerosVec.begin(), nonZerosVec.end(), nonZerosVec.begin());
    cudaCheckError();
    nonZerosVec[A.get_num_rows()] = NNZidx;
    // generate sets on the device
    IntVector set_fields(A.get_num_nz(), 0);
    setSetsGPU(A, s_con, cf_map, set_fields);
    int *set_field_ptr = set_fields.raw();
    // count the coarse points first
    // (no need to worry about >4B coarse points)
    int coarsePoints = (int) thrust::count_if(cf_map.begin(), cf_map.end(), is_non_neg());
    cudaCheckError();
    // I have all the information needed - create P and get the pointers to its data
    P.addProps(CSR);
    P.resize(A.get_num_rows(), coarsePoints, NNZidx, 1);
    IndexType *Pcolumn_indices_ptr = P.col_indices.raw();
    ValueType *Pvalues_ptr = P.values.raw();
    // now P is initialised, get row offsets from nonZerosVec with an exclusive prefix sum
    // set final value
    thrust::copy(nonZerosVec.begin(), nonZerosVec.end(), P.row_offsets.begin());
    cudaCheckError();
    IndexType *Poffsets_ptr = P.row_offsets.raw();
    // assign temp memory
    VVector B(NNZidx, 0.0);
    VVector D(A.get_num_rows(), 0.0);
    VVector W(NNZidx, 0.0);
    // get the raw pointers for temp memory
    ValueType *D_ptr = D.raw();
    ValueType *B_ptr = B.raw();
    ValueType *W_ptr = W.raw();
    // call each kernel in turn
    calculateBKernel <<< numBlocks, blocksize>>>(Aoffsets_ptr, Acolumn_indices_ptr, Avalues_ptr,
            (int) A.get_num_rows(), diag_ptr, D_ptr, set_field_ptr, cf_map_ptr,
            nonZerosVec_ptr, B_ptr);
    cudaCheckError();
    calculateDKernel <<< numBlocks, blocksize>>>(Aoffsets_ptr, Acolumn_indices_ptr, Avalues_ptr,
            (int) A.get_num_rows(), diag_ptr, set_field_ptr, D_ptr);
    cudaCheckError();
    calculateWKernel <<< numBlocks, blocksize>>>(Aoffsets_ptr, Acolumn_indices_ptr, Avalues_ptr,
            (int) A.get_num_rows(), diag_ptr, set_field_ptr, nonZerosVec_ptr,
            D_ptr, B_ptr, Pvalues_ptr, Pcolumn_indices_ptr,
            cf_map_ptr);
    cudaCheckError();
}

template< class T_Config>
void Distance1_InterpolatorBase<T_Config>::generateInterpolationMatrix(Matrix<T_Config> &A,
        IntVector &cf_map,
        BVector &s_con,
        IntVector &scratch,
        Matrix<T_Config> &P)
{
    P.set_initialized(0);

    if (A.get_block_size() == 1)
    {
        generateInterpolationMatrix_1x1(A, cf_map, s_con, scratch, P);
    }
    else
    {
        FatalError("Unsupported block size for distance1 interpolator", AMGX_ERR_NOT_SUPPORTED_BLOCKSIZE);
    }

    P.computeDiagonal();
    P.set_initialized(1);
}

#define AMGX_CASE_LINE(CASE) template class Distance1_InterpolatorBase<TemplateMode<CASE>::Type>;
AMGX_FORALL_BUILDS(AMGX_CASE_LINE)
#undef AMGX_CASE_LINE

#define AMGX_CASE_LINE(CASE) template class Distance1_Interpolator<TemplateMode<CASE>::Type>;
AMGX_FORALL_BUILDS(AMGX_CASE_LINE)
#undef AMGX_CASE_LINE

} // namespace amgx

