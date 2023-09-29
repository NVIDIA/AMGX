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

#include "unit_test.h"
#include "amg_solver.h"
#include "aggregation/coarseAgenerators/low_deg_coarse_A_generator.h"
#include <matrix_io.h>
#include <basic_types.h>
#include "test_utils.h"
#include "util.h"
#include "time.h"
#include <cfloat>
#include <thrust/sequence.h>

namespace amgx
{


struct Not_empty_row
{
    typedef bool result_type;
    template< typename Tuple >
    inline __device__ __host__ bool operator()( const Tuple &t ) const
    {
        return amgx::thrust::get<0>(t) != amgx::thrust::get<1>(t);
    }
};

template< typename Matrix, typename Vector >
static
void build_sort_permutation( const Matrix &M, Vector &permutation )
{
    int num_nz = M.get_num_nz();
    Vector row_indices( num_nz, 0 );
    amgx::thrust::scatter_if( amgx::thrust::counting_iterator<int>(0),
                        amgx::thrust::counting_iterator<int>(M.row_offsets.size() - 1),
                        M.row_offsets.begin(),
                        amgx::thrust::make_transform_iterator(
                            amgx::thrust::make_zip_iterator( amgx::thrust::make_tuple( M.row_offsets.begin(), M.row_offsets.begin() + 1 ) ),
                            Not_empty_row()),
                        row_indices.begin());
    cudaCheckError();
    amgx::thrust::inclusive_scan( row_indices.begin(), row_indices.begin() + M.get_num_nz(), row_indices.begin(), amgx::thrust::maximum<int>() );
    cudaCheckError();
    permutation.resize( num_nz );
    thrust_wrapper::sequence<Vector::TConfig::memSpace>( permutation.begin(), permutation.end() );
    cudaCheckError();
    Vector tmp( M.col_indices );
    amgx::thrust::stable_sort_by_key( tmp.begin(), tmp.end(), permutation.begin() );
    cudaCheckError();
    tmp = row_indices;
    amgx::thrust::gather( permutation.begin(), permutation.end(), tmp.begin(), row_indices.begin() );
    cudaCheckError();
    amgx::thrust::stable_sort_by_key( row_indices.begin(), row_indices.end(), permutation.begin() );
    cudaCheckError();
}

template< typename Value_type >
static __device__ __inline__
bool equal( Value_type x, Value_type y, Value_type epsilon, Value_type max_abs_error )
{
    if ( x == y )
    {
        return true;
    }

    if ( abs(x - y) < max_abs_error )
    {
        return true;
    }

    if ( abs(x - y) <= epsilon )
    {
        return true;
    }

    return false;
}

static __device__ __inline__
bool equal( float x, float y )
{
    return equal( x, y, 1.0e-6f, 1.0e10f * FLT_MIN );
}

static __device__ __inline__
bool equal( double x, double y )
{
    return equal( x, y, 1.0e-12, 1.0e10 * DBL_MIN );
}

template< typename Value_type >
__global__
void compare_matrices_kernel( const int  num_rows,
                              const int  block_size_sq,
                              const int  has_diag,
                              const int *A_rows,
                              const int *A_cols,
                              const int *A_diag,
                              const Value_type *A_vals,
                              const int *A_perm,
                              const int *B_rows,
                              const int *B_cols,
                              const int *B_diag,
                              const Value_type *B_vals,
                              const int *B_perm,
                              int *ok )
{
    int row_id = blockIdx.x * blockDim.x + threadIdx.x;

    for ( ; row_id < num_rows ; row_id += gridDim.x * blockDim.x )
    {
        if ( has_diag )
        {
            int a_it = A_diag[row_id];
            int b_it = B_diag[row_id];
            Value_type a_val = A_vals[a_it];
            Value_type b_val = A_vals[b_it];

            if ( !equal( a_val, b_val ) )
            {
                printf( "ERROR: row=%d: Diag A[%d]=%f and B[%d]=%f are different!!!\n", row_id, a_it, a_val, b_it, b_val );
                ok[0] = 0;
                return;
            }
        }

        int row_it  = A_rows[row_id + 0];
        int row_end = A_rows[row_id + 1];

        if ( row_it != B_rows[row_id + 0] || row_end != B_rows[row_id + 1] )
        {
            printf( "ERROR: Rows A[%d] and B[%d] have different lenghts!!!\n", row_id, row_id );
            ok[0] = 0;
            return;
        }

        for ( ; row_it < row_end ; ++row_it )
        {
            const int a_it = A_perm[row_it];
            const int b_it = B_perm[row_it];
            const int a_col = A_cols[a_it];
            const int b_col = B_cols[b_it];

            if ( a_col != b_col )
            {
                printf( "ERROR: row=%d row_it=%d: Cols A[%d]=%d and B[%d]=%d are different!!!\n", row_id, row_it, a_it, a_col, b_it, b_col );
                ok[0] = 0;
                return;
            }

            for ( int k = 0 ; k < block_size_sq ; ++k )
            {
                Value_type a_val = A_vals[block_size_sq * a_it + k];
                Value_type b_val = B_vals[block_size_sq * b_it + k];

                if ( !equal( a_val, b_val ) )
                {
                    printf( "ERROR: row=%d row_it=%d: Vals A[%d]=%f and B[%d]=%f are different!!!\n", row_id, row_it, a_it, a_val, b_it, b_val );
                    ok[0] = 0;
                    return;
                }
            }
        }
    }
}






// parameter is used as test name
DECLARE_UNITTEST_BEGIN(AggregatesCoarseGeneratorTest);

// setup restriction on HOST
void fillRowOffsetsAndColIndices(const int num_aggregates,
                                 typename Matrix_h::IVector aggregates,
                                 const int R_num_cols,
                                 typename Matrix_h::IVector &R_row_offsets,
                                 typename Matrix_h::IVector &R_col_indices)
{
    for (int i = 0; i < num_aggregates + 1; i++)
    {
        R_row_offsets[i] = 0;
    }

    // Count number of neighbors for each row
    for (int i = 0; i < R_num_cols; i++)
    {
        int I = aggregates[i];
        R_row_offsets[I]++;
    } 

    R_row_offsets[num_aggregates] = R_num_cols;

    for (int i = num_aggregates - 1; i >= 0; i--)
    {
        R_row_offsets[i] = R_row_offsets[i + 1] - R_row_offsets[i];
    }

    /* Set column indices. */
    for (int i = 0; i < R_num_cols; i++)
    {
        int I = aggregates[i];
        int Ip = R_row_offsets[I]++;
        R_col_indices[Ip] = i;
    }

    /* Reset r[i] to start of row memory. */
    for (int i = num_aggregates - 1; i > 0; i--)
    {
        R_row_offsets[i] = R_row_offsets[i - 1];
    }

    R_row_offsets[0] = 0;
}

template< typename Matrix_type >
void compare_matrices( const Matrix_type &A, const Matrix_type &B )
{
    typedef typename Matrix_type::IVector IVector;
    IVector A_perm;
    build_sort_permutation( A, A_perm );
    IVector B_perm;
    build_sort_permutation( B, B_perm );
    IVector ok( 1, 1 );
    compare_matrices_kernel <<< 2048, 256>>>( A.get_num_rows(),
            A.get_block_dimx() * A.get_block_dimy(),
            A.hasProps(DIAG),
            A.row_offsets.raw(),
            A.col_indices.raw(),
            A.diag.raw(),
            A.values.raw(),
            A_perm.raw(),
            B.row_offsets.raw(),
            B.col_indices.raw(),
            B.diag.raw(),
            B.values.raw(),
            B_perm.raw(),
            ok.raw() );
    cudaCheckError();
    UNITTEST_ASSERT_TRUE( ok[0] == 1 );
}

void run()
{
    cudaCheckError();
    this->randomize( 375139 );
    AMG_Config cfg;
    const std::string &cfg_scope = "default";
    // setup generators
    cudaCheckError();
    aggregation::CoarseAGenerator<TConfig> *d_generator = new aggregation::LowDegCoarseAGenerator<TConfig>(cfg, cfg_scope);
    this->PrintOnFail("Device generator creation");
    UNITTEST_ASSERT_TRUE(d_generator != NULL);
    cudaCheckError();
    aggregation::CoarseAGenerator<TConfig_h> *h_generator = new aggregation::LowDegCoarseAGenerator<TConfig_h>(cfg, cfg_scope);
    this->PrintOnFail("Host generator creation");
    UNITTEST_ASSERT_TRUE(h_generator != NULL);
    cudaCheckError();
    MatrixA A, Ac;
    int bsizes[] = {1, 2, 3, 4, 5, 8, 10};

    for (int diag_prop = 0; diag_prop < 2; diag_prop++)
    {
        for (int bs = 0; bs < sizeof(bsizes)/sizeof(bsizes[0]); bs++)
        {
            int b = bsizes[bs];
            // setup matrix A
            cudaCheckError();
            generateMatrixRandomStruct<TConfig>::generate(A, 128, diag_prop, b, false);
            cudaCheckError();
            random_fill(A);
            cudaCheckError();
            Matrix_h h_A = A;
            cudaCheckError();
            Matrix_h h_Ac;
            cudaCheckError();
            // setup aggregates on CPU
            int num_aggregates = (A.get_num_rows() - 1) / 2 + 1; //A.get_num_rows();
            typename Matrix_h::IVector h_aggregates;
            h_aggregates.resize( A.get_num_rows() );

            for ( int i = 0; i < h_aggregates.size(); i++ )
            {
                h_aggregates[i] = i / 2;    //i;
            }

            // setup R matrix on CPU
            typename Matrix_h::IVector h_R_row_offsets;
            typename Matrix_h::IVector h_R_col_indices;
            h_R_row_offsets.resize( num_aggregates + 1 );
            h_R_col_indices.resize( A.get_num_rows() );
            fillRowOffsetsAndColIndices( num_aggregates, h_aggregates, A.get_num_rows(), h_R_row_offsets, h_R_col_indices );
            // assign GPU vectors
            IVector aggregates = h_aggregates;
            IVector R_row_offsets = h_R_row_offsets;
            IVector R_col_indices = h_R_col_indices;
            // compute Galerkin product on CPU and GPU
            h_generator->computeAOperator(h_A, h_Ac, h_aggregates, h_R_row_offsets, h_R_col_indices, num_aggregates);
            d_generator->computeAOperator(A, Ac, aggregates, R_row_offsets, R_col_indices, num_aggregates);
            // simple check on matrix size
            this->PrintOnFail("Coarse matrix has wrong size %i != num aggregates %i", Ac.get_num_rows(), num_aggregates);
            UNITTEST_ASSERT_TRUE( Ac.get_num_rows() == num_aggregates );
            // dump matrix to file
            VVector v;
            v.resize(Ac.get_num_rows() * Ac.get_block_dimy());
            random_fill(v);
            // compare structure
            this->PrintOnFail("Coarse matrix has incorrect structure, diag prop %i, block size %i, num rows %i, num aggregates %i", diag_prop, b, A.get_num_rows(), num_aggregates);
            compare_matrices(Ac, MatrixA(h_Ac) /*, b==1 && !diag_prop */ );
        }
    }

    delete d_generator;
    delete h_generator;
}

DECLARE_UNITTEST_END(AggregatesCoarseGeneratorTest);

// run for all device configs
#define AMGX_CASE_LINE(CASE) AggregatesCoarseGeneratorTest<TemplateMode<CASE>::Type> AggregatesCoarseGeneratorTest_##CASE;
AMGX_FORALL_BUILDS_DEVICE(AMGX_CASE_LINE)
#undef AMGX_CASE_LINE

//AggregatesCoarseGeneratorTest<TemplateMode<AMGX_mode_dDDI>::Type> AggregatesCoarseGeneratorTest_dDDI;


} //namespace amgx
