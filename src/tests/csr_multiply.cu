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

#ifdef USE_CPP_TR1
#include <unordered_set>
#include <unordered_map>
#else
#include <set>
#include <map>
#endif
#include <thrust/scan.h>
#include <cusp/gallery/poisson.h>

#include "unit_test.h"
#include "matrix.h"
#include "csr_multiply.h"

namespace amgx
{

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template< typename T >
struct Epsilon
{};

template<>
struct Epsilon<float>
{
    static __device__ __host__ __forceinline__ float value( ) { return 1.0e-4f; }
};

template<>
struct Epsilon<double>
{
    static __device__ __host__ __forceinline__ double value( ) { return 1.0e-8; }
};

DECLARE_UNITTEST_BEGIN(CsrMultiplyTests_Base);

std::string base_keywords()
{
    return "csr";
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template< typename Index_vector >
void
count_non_zeroes( const Index_vector &A_rows, const Index_vector &A_cols,
                  const Index_vector &B_rows, const Index_vector &B_cols,
                  Index_vector &C_rows )
{
    typedef typename Index_vector::value_type Index_type;
    int nRows = static_cast<int>( A_rows.size( ) - 1 );
    #pragma omp parallel for shared(nRows)

    for ( int aRowId = 0 ; aRowId < nRows ; ++aRowId )
    {
#ifdef USE_CPP_TR1
        std::tr1::unordered_set<Index_type> cCols;
#else
        std::set<Index_type> cCols;
#endif

        for ( Index_type aColIt = A_rows[aRowId], aColEnd = A_rows[aRowId + 1] ; aColIt < aColEnd ; ++aColIt )
        {
            Index_type bRowId = A_cols[aColIt];

            for ( Index_type bColIt = B_rows[bRowId], bColEnd = B_rows[bRowId + 1] ; bColIt < bColEnd ; ++bColIt )
            {
                cCols.insert( B_cols[bColIt] );
            }
        }

        C_rows[aRowId] = static_cast<Index_type>( cCols.size( ) );
    }
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template< typename Index_vector, typename Value_vector >
void
compute_values( const Index_vector &A_rows, const Index_vector &A_cols, const Value_vector &A_vals,
                const Index_vector &B_rows, const Index_vector &B_cols, const Value_vector &B_vals,
                const Index_vector &C_rows,       Index_vector &C_cols,       Value_vector &C_vals )
{
    typedef typename Index_vector::value_type Index_type;
    typedef typename Value_vector::value_type Value_type;
#ifdef USE_CPP_TR1
    typedef std::tr1::unordered_map<Index_type, Value_type> Map;
    typedef typename Map::iterator MapIterator;
    typedef std::pair<MapIterator, bool> MapQuery;
    typedef typename Map::const_iterator MapConstIterator;
#else
    typedef std::map<Index_type, Value_type> Map;
    typedef typename Map::iterator MapIterator;
    typedef std::pair<MapIterator, bool> MapQuery;
    typedef typename Map::const_iterator MapConstIterator;
#endif
    int nRows = static_cast<int>( A_rows.size( ) - 1 );
    #pragma omp parallel for shared(nRows)

    for ( int aRowId = 0 ; aRowId < nRows ; ++aRowId )
    {
        Map cVals;

        for ( unsigned aColIt = A_rows[aRowId], aColEnd = A_rows[aRowId + 1] ; aColIt < aColEnd ; ++aColIt )
        {
            Index_type bRowId = A_cols[aColIt];
            Value_type aValue = A_vals[aColIt];

            for ( Index_type bColIt = B_rows[bRowId], bColEnd = B_rows[bRowId + 1] ; bColIt < bColEnd ; ++bColIt )
            {
                Index_type bColId = B_cols[bColIt];
                Value_type bValue = B_vals[bColIt];
                Value_type value = aValue * bValue;
                MapQuery q = cVals.insert( typename Map::value_type( bColId, value ) );

                if ( !q.second )
                {
                    q.first->second += value;
                }
            }
        }

        Index_type cRowIt = C_rows[aRowId];

        for ( MapConstIterator it = cVals.begin( ) ; it != cVals.end( ) ; ++it, ++cRowIt )
        {
            assert( cRowIt < C_rows[aRowId + 1] );
            C_cols[cRowIt] = it->first;
            C_vals[cRowIt] = it->second;
        }
    }
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template< typename Config >
void
compare_matrices( Matrix<Config> &A, Matrix<Config> &B )
{
    A.sortByRowAndColumn();
    B.sortByRowAndColumn();
    UNITTEST_ASSERT_EQUAL_DESC( "Rows", A.row_offsets, B.row_offsets );
    UNITTEST_ASSERT_EQUAL_DESC( "Cols", A.col_indices, B.col_indices );
    typedef typename MatPrecisionMap<Config::matPrec>::Type Value_type;
    const Value_type epsilon = Epsilon<Value_type>::value();
    UNITTEST_ASSERT_EQUAL_TOL_DESC( "Vals", A.values, B.values, epsilon );
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template< AMGX_VecPrecision VecPrecision, AMGX_MatPrecision MatPrecision >
void
check_csr_square( const Matrix<TemplateConfig<AMGX_host, VecPrecision, MatPrecision, AMGX_indInt> > &A_h, void *wk )
{
    typedef TemplateConfig<AMGX_host,   VecPrecision, MatPrecision, AMGX_indInt> Config_h;
    typedef TemplateConfig<AMGX_device, VecPrecision, MatPrecision, AMGX_indInt> Config_d;
    typedef Matrix<Config_h> Matrix_h;
    typedef Matrix<Config_d> Matrix_d;
    Matrix_h B_h( A_h ), C_h;
    C_h.set_num_rows( A_h.get_num_rows() );
    C_h.set_num_cols( B_h.get_num_rows() );
    C_h.row_offsets.resize( A_h.get_num_rows() + 1 );
    std::ostringstream buffer;
    {
        count_non_zeroes( A_h.row_offsets, A_h.col_indices, B_h.row_offsets, B_h.col_indices, C_h.row_offsets );
        amgx::thrust::exclusive_scan( C_h.row_offsets.begin( ), C_h.row_offsets.end( ), C_h.row_offsets.begin( ) );
        cudaCheckError();
        int nVals = C_h.row_offsets[A_h.get_num_rows()];
        C_h.col_indices.resize( nVals );
        C_h.values.resize( nVals );
        C_h.set_num_nz( nVals );
        compute_values( A_h.row_offsets, A_h.col_indices, A_h.values, B_h.row_offsets, B_h.col_indices, B_h.values, C_h.row_offsets, C_h.col_indices, C_h.values );
    }
    Matrix_d C_d;
    {
        Matrix_d A_d( A_h ), B_d( B_h );
        CSR_Multiply<Config_d>::csr_multiply( A_d, B_d, C_d, wk );
    }
    Matrix_d C_d_ref( C_h );
    compare_matrices( C_d, C_d_ref );
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template< AMGX_VecPrecision VecPrecision, AMGX_MatPrecision MatPrecision >
void
check_csr_multiply_poisson( int points, int nx, int ny, int nz, AMG_Config &cfg )
{
    typedef TemplateConfig<AMGX_host,   VecPrecision, MatPrecision, AMGX_indInt> Config_h;
    typedef TemplateConfig<AMGX_device, VecPrecision, MatPrecision, AMGX_indInt> Config_d;
    typedef Matrix<Config_h> Matrix_h;
    Matrix_h A_h;
    A_h.set_initialized(0);

    switch (points)
    {
        case 5:
        case 7:
        case 9:
        case 27:
            generatePoissonForTest(A_h, 1, 0, points, nx, ny, nz);
            break;

        default:
            printf("Error invalid number of poisson points specified, valid numbers are 5, 7, 9, 27\n");
    }

    A_h.set_initialized(1);
    void *wk = CSR_Multiply<Config_d>::csr_workspace_create( cfg, "default" );
    check_csr_square( A_h, wk );
    CSR_Multiply<Config_d>::csr_workspace_delete( wk );
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template< AMGX_VecPrecision VecPrecision, AMGX_MatPrecision MatPrecision >
void
check_csr_multiply_file( const std::string &filename, AMG_Config &cfg, bool one_value = false )
{
    typedef TemplateConfig<AMGX_host,   VecPrecision, MatPrecision, AMGX_indInt> Config_h;
    typedef TemplateConfig<AMGX_device, VecPrecision, MatPrecision, AMGX_indInt> Config_d;
    typedef Matrix<Config_h> Matrix_h;
    typedef Vector<Config_h> Vector_h;
    Matrix_h A_h;
    Vector_h x_h, b_h;
    A_h.set_initialized(0);
    A_h.addProps(CSR);
    UNITTEST_ASSERT_TRUE(MatrixIO<Config_h>::readSystem( filename.c_str(), A_h, b_h, x_h ) == AMGX_OK);

    // set all values to 1 to avoid numerical problems
    if (one_value)
        for (int i = 0; i < A_h.values.size(); i++)
        {
            A_h.values[i] = 1.0;
        }

    A_h.set_initialized(1);
    void *wk = CSR_Multiply<Config_d>::csr_workspace_create( cfg, "default" );
    check_csr_square( A_h, wk );
    CSR_Multiply<Config_d>::csr_workspace_delete( wk );
}

DECLARE_UNITTEST_END(CsrMultiplyTests_Base);

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

DECLARE_UNITTEST_BEGIN_EXTD(CsrMultiplyTests_Poisson5_10_10, CsrMultiplyTests_Base<T_Config>);

void run()
{

  AMG_Config cfg;
  CsrMultiplyTests_Base<T_Config>::template check_csr_multiply_poisson<T_Config::vecPrec, T_Config::matPrec>( 5, 10, 10, 10, cfg );
}

DECLARE_UNITTEST_END(CsrMultiplyTests_Poisson5_10_10)

CsrMultiplyTests_Poisson5_10_10<TemplateMode<AMGX_mode_dDDI>::Type> CsrMultiplyTests_Poisson5_10_10_dDDI;
CsrMultiplyTests_Poisson5_10_10<TemplateMode<AMGX_mode_dFFI>::Type> CsrMultiplyTests_Poisson5_10_10_dFFI;

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

DECLARE_UNITTEST_BEGIN_EXTD(CsrMultiplyTests_Poisson5_100_100, CsrMultiplyTests_Base<T_Config>);

void run()
{
  AMG_Config cfg;
  CsrMultiplyTests_Base<T_Config>::template check_csr_multiply_poisson<T_Config::vecPrec, T_Config::matPrec>( 5, 100, 100, 100, cfg );
}

DECLARE_UNITTEST_END(CsrMultiplyTests_Poisson5_100_100)

CsrMultiplyTests_Poisson5_100_100<TemplateMode<AMGX_mode_dDDI>::Type> CsrMultiplyTests_Poisson5_100_100_dDDI;
CsrMultiplyTests_Poisson5_100_100<TemplateMode<AMGX_mode_dFFI>::Type> CsrMultiplyTests_Poisson5_100_100_dFFI;

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

DECLARE_UNITTEST_BEGIN_EXTD(CsrMultiplyTests_Poisson7_10_10, CsrMultiplyTests_Base<T_Config>);

void run()
{
  AMG_Config cfg;
  CsrMultiplyTests_Base<T_Config>::template check_csr_multiply_poisson<T_Config::vecPrec, T_Config::matPrec>( 5, 10, 10, 10, cfg );
}

DECLARE_UNITTEST_END(CsrMultiplyTests_Poisson7_10_10)

CsrMultiplyTests_Poisson7_10_10<TemplateMode<AMGX_mode_dDDI>::Type> CsrMultiplyTests_Poisson7_10_10_dDDI;
CsrMultiplyTests_Poisson7_10_10<TemplateMode<AMGX_mode_dFFI>::Type> CsrMultiplyTests_Poisson7_10_10_dFFI;

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

DECLARE_UNITTEST_BEGIN_EXTD(CsrMultiplyTests_Poisson7_100_100, CsrMultiplyTests_Base<T_Config>);

void run()
{
  AMG_Config cfg;
  CsrMultiplyTests_Base<T_Config>::template check_csr_multiply_poisson<T_Config::vecPrec, T_Config::matPrec>( 5, 100, 100, 100, cfg );
}

DECLARE_UNITTEST_END(CsrMultiplyTests_Poisson7_100_100)

CsrMultiplyTests_Poisson7_100_100<TemplateMode<AMGX_mode_dDDI>::Type> CsrMultiplyTests_Poisson7_100_100_dDDI;
CsrMultiplyTests_Poisson7_100_100<TemplateMode<AMGX_mode_dFFI>::Type> CsrMultiplyTests_Poisson7_100_100_dFFI;

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

DECLARE_UNITTEST_BEGIN_EXTD(CsrMultiplyTests_Poisson9_10_10, CsrMultiplyTests_Base<T_Config>);

void run()
{
  AMG_Config cfg;
  CsrMultiplyTests_Base<T_Config>::template check_csr_multiply_poisson<T_Config::vecPrec, T_Config::matPrec>( 5, 10, 10, 10, cfg );
}

DECLARE_UNITTEST_END(CsrMultiplyTests_Poisson9_10_10)

CsrMultiplyTests_Poisson9_10_10<TemplateMode<AMGX_mode_dDDI>::Type> CsrMultiplyTests_Poisson9_10_10_dDDI;
CsrMultiplyTests_Poisson9_10_10<TemplateMode<AMGX_mode_dFFI>::Type> CsrMultiplyTests_Poisson9_10_10_dFFI;

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

DECLARE_UNITTEST_BEGIN_EXTD(CsrMultiplyTests_Poisson9_100_100, CsrMultiplyTests_Base<T_Config>);

void run()
{
  AMG_Config cfg;
  CsrMultiplyTests_Base<T_Config>::template check_csr_multiply_poisson<T_Config::vecPrec, T_Config::matPrec>( 5, 100, 100, 100, cfg );
}

DECLARE_UNITTEST_END(CsrMultiplyTests_Poisson9_100_100)

CsrMultiplyTests_Poisson9_100_100<TemplateMode<AMGX_mode_dDDI>::Type> CsrMultiplyTests_Poisson9_100_100_dDDI;
CsrMultiplyTests_Poisson9_100_100<TemplateMode<AMGX_mode_dFFI>::Type> CsrMultiplyTests_Poisson9_100_100_dFFI;

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

DECLARE_UNITTEST_BEGIN_EXTD(CsrMultiplyTests_Poisson27_10_10, CsrMultiplyTests_Base<T_Config>);

void run()
{
  AMG_Config cfg;
  CsrMultiplyTests_Base<T_Config>::template check_csr_multiply_poisson<T_Config::vecPrec, T_Config::matPrec>( 5, 10, 10, 10, cfg );
}

DECLARE_UNITTEST_END(CsrMultiplyTests_Poisson27_10_10)

CsrMultiplyTests_Poisson27_10_10<TemplateMode<AMGX_mode_dDDI>::Type> CsrMultiplyTests_Poisson27_10_10_dDDI;
CsrMultiplyTests_Poisson27_10_10<TemplateMode<AMGX_mode_dFFI>::Type> CsrMultiplyTests_Poisson27_10_10_dFFI;

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

DECLARE_UNITTEST_BEGIN_EXTD(CsrMultiplyTests_Poisson27_100_100, CsrMultiplyTests_Base<T_Config>);

void run()
{
  AMG_Config cfg;
  CsrMultiplyTests_Base<T_Config>::template check_csr_multiply_poisson<T_Config::vecPrec, T_Config::matPrec>( 5, 100, 100, 100, cfg );
}

DECLARE_UNITTEST_END(CsrMultiplyTests_Poisson27_100_100)

CsrMultiplyTests_Poisson27_100_100<TemplateMode<AMGX_mode_dDDI>::Type> CsrMultiplyTests_Poisson27_100_100_dDDI;
CsrMultiplyTests_Poisson27_100_100<TemplateMode<AMGX_mode_dFFI>::Type> CsrMultiplyTests_Poisson27_100_100_dFFI;

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace amgx
