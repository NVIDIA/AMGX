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
#include "matrix_coloring/matrix_coloring.h"

namespace amgx
{

DECLARE_UNITTEST_BEGIN(CsrSparsityILU1Tests_Base);

std::string base_keywords()
{
    return "csr";
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template< typename Index_vector >
void
count_non_zeroes( const Index_vector &A_rows, const Index_vector &A_cols, const Index_vector &A_coloring, Index_vector &B_rows, Index_vector &B_cols, bool countOnly )
{
    typedef typename Index_vector::value_type Index_type;
#ifdef USE_CPP_TR1
    typedef std::tr1::unordered_set<Index_type> Set;
#else
    typedef std::set<Index_type> Set;
#endif
    int nRows = static_cast<int>( A_rows.size( ) - 1 );
    #pragma omp parallel for shared(nRows)

    for ( int aRowId = 0 ; aRowId < nRows ; ++aRowId )
    {
        Set bCols;
        Index_type aRowColor = A_coloring[aRowId];

        // Insert the column of A inside the set.
        for ( Index_type aColIt = A_rows[aRowId], aColEnd = A_rows[aRowId + 1] ; aColIt < aColEnd ; ++aColIt )
        {
            bCols.insert( A_cols[aColIt] );
        }

        // If the color of the row is != 0.
        if ( aRowColor != 0 )
        {
            for ( Index_type aColIt = A_rows[aRowId], aColEnd = A_rows[aRowId + 1] ; aColIt < aColEnd ; ++aColIt )
            {
                Index_type bRowId = A_cols[aColIt];

                if ( A_coloring[bRowId] < aRowColor )
                {
                    for ( Index_type bColIt = A_rows[bRowId], bColEnd = A_rows[bRowId + 1] ; bColIt < bColEnd ; ++bColIt )
                    {
                        Index_type bColId = A_cols[bColIt];

                        if ( A_coloring[bColId] >= A_coloring[bRowId] && A_coloring[bColId] != aRowColor )
                        {
                            bCols.insert( bColId );
                        }
                    }
                }
            }
        }

        if ( countOnly )
        {
            B_rows[aRowId] = static_cast<Index_type>( bCols.size( ) );
        }
        else
        {
            Index_type cRowIt = B_rows[aRowId];

            for ( typename Set::const_iterator it = bCols.begin() ; it != bCols.end() ; ++it, ++cRowIt )
            {
                assert( cRowIt < B_rows[aRowId + 1] );
                B_cols[cRowIt] = *it;
            }
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
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template< AMGX_VecPrecision VecPrecision, AMGX_MatPrecision MatPrecision >
void
check_csr_sparsity_ilu1( const Matrix<TemplateConfig<AMGX_host, VecPrecision, MatPrecision, AMGX_indInt> > &A_h, void *wk )
{
    typedef TemplateConfig<AMGX_host,   VecPrecision, MatPrecision, AMGX_indInt> Config_h;
    typedef TemplateConfig<AMGX_device, VecPrecision, MatPrecision, AMGX_indInt> Config_d;
    typedef Matrix<Config_h> Matrix_h;
    typedef Matrix<Config_d> Matrix_d;
    typedef Vector<typename Config_h::template setVecPrec<AMGX_vecInt>::Type> IVector_h;
    typedef Vector<typename Config_d::template setVecPrec<AMGX_vecInt>::Type> IVector_d;
    Matrix_h B_h;
    B_h.set_num_rows( A_h.get_num_rows() );
    B_h.set_num_cols( A_h.get_num_rows() );
    B_h.row_offsets.resize( A_h.get_num_rows() + 1 );
    std::ostringstream buffer;
    {
        count_non_zeroes( A_h.row_offsets, A_h.col_indices, A_h.getMatrixColoring().getRowColors(), B_h.row_offsets, B_h.col_indices, true );
        thrust_wrapper::exclusive_scan( B_h.row_offsets.begin( ), B_h.row_offsets.end( ), B_h.row_offsets.begin( ) );
        cudaCheckError();
        int nVals = B_h.row_offsets[A_h.get_num_rows()];
        B_h.col_indices.resize( nVals );
        B_h.values.resize( nVals );
        B_h.set_num_nz( nVals );
        count_non_zeroes( A_h.row_offsets, A_h.col_indices, A_h.getMatrixColoring().getRowColors(), B_h.row_offsets, B_h.col_indices, false );
    }
    Matrix_d B_d;
    {
        Matrix_d A_d( A_h );
        IVector_d row_colors_d = A_h.getMatrixColoring().getRowColors();;
        A_d.getMatrixColoring().setRowColors(row_colors_d);;
        CSR_Multiply<Config_d>::csr_sparsity_ilu1( A_d, B_d, wk );
    }
    Matrix_d B_d_ref( B_h );
    compare_matrices( B_d, B_d_ref );
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template< AMGX_VecPrecision VecPrecision, AMGX_MatPrecision MatPrecision >
void
check_csr_sparsity_poisson( int points, int nx, int ny, int nz, AMG_Config &cfg )
{
    typedef TemplateConfig<AMGX_host,   VecPrecision, MatPrecision, AMGX_indInt> Config_h;
    typedef TemplateConfig<AMGX_device, VecPrecision, MatPrecision, AMGX_indInt> Config_d;
    typedef Matrix<Config_h> Matrix_h;
    typedef Matrix<Config_d> Matrix_d;
    typedef Vector<typename Config_h::template setVecPrec<AMGX_vecInt>::Type> IVector_h;
    typedef Vector<typename Config_d::template setVecPrec<AMGX_vecInt>::Type> IVector_d;
    typedef AMG<VecPrecision, MatPrecision, AMGX_indInt> AMG_Class;
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
    Matrix_d A_d( A_h );
    UNITTEST_ASSERT_TRUE ( cfg.parseParameterString("coloring_level=1") == AMGX_OK);
    A_d.set_initialized(0);
    A_d.colorMatrix(cfg, "default");
    A_h.set_initialized(0);
    A_h.copyMatrixColoring(A_d.getMatrixColoring());
    A_h.set_initialized(1);
    void *wk = CSR_Multiply<Config_d>::csr_workspace_create( cfg, "default" );
    check_csr_sparsity_ilu1( A_h, wk );
    CSR_Multiply<Config_d>::csr_workspace_delete( wk );
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template< AMGX_VecPrecision VecPrecision, AMGX_MatPrecision MatPrecision >
void
check_csr_sparsity_square_file( const std::string &filename, AMG_Config &cfg )
{
    typedef TemplateConfig<AMGX_host,   VecPrecision, MatPrecision, AMGX_indInt> Config_h;
    typedef TemplateConfig<AMGX_device, VecPrecision, MatPrecision, AMGX_indInt> Config_d;
    typedef Matrix<Config_h> Matrix_h;
    typedef Matrix<Config_d> Matrix_d;
    typedef Vector<typename Config_h::template setVecPrec<AMGX_vecInt>::Type> IVector_h;
    typedef Vector<typename Config_d::template setVecPrec<AMGX_vecInt>::Type> IVector_d;
    typedef AMG<T_Config::vecPrec, T_Config::matPrec, AMGX_indInt> AMG_Class;
    Matrix_h A_h;
    Vector_h x_h, b_h;
    A_h.set_initialized(0);
    A_h.addProps(CSR);
    UNITTEST_ASSERT_TRUE(MatrixIO<Config_h>::readSystem( filename.c_str(), A_h, b_h, x_h ) == AMGX_OK);
    A_h.set_initialized(1);
    Matrix_d A_d( A_h );
    UNITTEST_ASSERT_TRUE ( cfg.parseParameterString("coloring_level=1") == AMGX_OK);
    A_d.set_initialized(0);
    A_h.set_initialized(0);
    A_d.colorMatrix(cfg, "default");
    A_h.copyMatrixColoring(A_d.getMatrixColoring());
    A_h.set_initialized(1);
    void *wk = CSR_Multiply<Config_d>::csr_workspace_create( cfg, "default" );
    check_csr_sparsity_ilu1( A_h, wk );
    CSR_Multiply<Config_d>::csr_workspace_delete( wk );
}

DECLARE_UNITTEST_END(CsrSparsityILU1Tests_Base);

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

DECLARE_UNITTEST_BEGIN_EXTD(CsrSparsityILU1Tests_Poisson5_10_10, CsrSparsityILU1Tests_Base<T_Config>);

void run()
{
  AMG_Config cfg;
  CsrSparsityILU1Tests_Base<T_Config>::template check_csr_sparsity_poisson<T_Config::vecPrec, T_Config::matPrec>( 5, 10, 10, 10, cfg );
}

DECLARE_UNITTEST_END(CsrSparsityILU1Tests_Poisson5_10_10)

CsrSparsityILU1Tests_Poisson5_10_10<TemplateMode<AMGX_mode_dDDI>::Type> CsrSparsityILU1Tests_Poisson5_10_10_dDDI;

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

DECLARE_UNITTEST_BEGIN_EXTD(CsrSparsityILU1Tests_Poisson5_100_100, CsrSparsityILU1Tests_Base<T_Config>);

void run()
{
  AMG_Config cfg;
  CsrSparsityILU1Tests_Base<T_Config>::template check_csr_sparsity_poisson<T_Config::vecPrec, T_Config::matPrec>( 5, 100, 100, 100, cfg );
}

DECLARE_UNITTEST_END(CsrSparsityILU1Tests_Poisson5_100_100)

CsrSparsityILU1Tests_Poisson5_100_100<TemplateMode<AMGX_mode_dDDI>::Type> CsrSparsityILU1Tests_Poisson5_100_100_dDDI;

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

DECLARE_UNITTEST_BEGIN_EXTD(CsrSparsityILU1Tests_Poisson7_10_10, CsrSparsityILU1Tests_Base<T_Config>);

void run()
{
  AMG_Config cfg;
  CsrSparsityILU1Tests_Base<T_Config>::template check_csr_sparsity_poisson<T_Config::vecPrec, T_Config::matPrec>( 5, 10, 10, 10, cfg );
}

DECLARE_UNITTEST_END(CsrSparsityILU1Tests_Poisson7_10_10)

CsrSparsityILU1Tests_Poisson7_10_10<TemplateMode<AMGX_mode_dDDI>::Type> CsrSparsityILU1Tests_Poisson7_10_10_dDDI;

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

DECLARE_UNITTEST_BEGIN_EXTD(CsrSparsityILU1Tests_Poisson7_100_100, CsrSparsityILU1Tests_Base<T_Config>);

void run()
{
  AMG_Config cfg;
  CsrSparsityILU1Tests_Base<T_Config>::template check_csr_sparsity_poisson<T_Config::vecPrec, T_Config::matPrec>( 5, 100, 100, 100, cfg );
}

DECLARE_UNITTEST_END(CsrSparsityILU1Tests_Poisson7_100_100)

CsrSparsityILU1Tests_Poisson7_100_100<TemplateMode<AMGX_mode_dDDI>::Type> CsrSparsityILU1Tests_Poisson7_100_100_dDDI;

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

DECLARE_UNITTEST_BEGIN_EXTD(CsrSparsityILU1Tests_Poisson9_10_10, CsrSparsityILU1Tests_Base<T_Config>);

void run()
{
  AMG_Config cfg;
  CsrSparsityILU1Tests_Base<T_Config>::template check_csr_sparsity_poisson<T_Config::vecPrec, T_Config::matPrec>( 5, 10, 10, 10, cfg );
}

DECLARE_UNITTEST_END(CsrSparsityILU1Tests_Poisson9_10_10)

CsrSparsityILU1Tests_Poisson9_10_10<TemplateMode<AMGX_mode_dDDI>::Type> CsrSparsityILU1Tests_Poisson9_10_10_dDDI;

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

DECLARE_UNITTEST_BEGIN_EXTD(CsrSparsityILU1Tests_Poisson9_100_100, CsrSparsityILU1Tests_Base<T_Config>);

void run()
{
  AMG_Config cfg;
  CsrSparsityILU1Tests_Base<T_Config>::template check_csr_sparsity_poisson<T_Config::vecPrec, T_Config::matPrec>( 5, 100, 100, 100, cfg );
}

DECLARE_UNITTEST_END(CsrSparsityILU1Tests_Poisson9_100_100)

CsrSparsityILU1Tests_Poisson9_100_100<TemplateMode<AMGX_mode_dDDI>::Type> CsrSparsityILU1Tests_Poisson9_100_100_dDDI;

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

DECLARE_UNITTEST_BEGIN_EXTD(CsrSparsityILU1Tests_Poisson27_10_10, CsrSparsityILU1Tests_Base<T_Config>);

void run()
{
  AMG_Config cfg;
  CsrSparsityILU1Tests_Base<T_Config>::template check_csr_sparsity_poisson<T_Config::vecPrec, T_Config::matPrec>( 5, 10, 10, 10, cfg );
}

DECLARE_UNITTEST_END(CsrSparsityILU1Tests_Poisson27_10_10)

CsrSparsityILU1Tests_Poisson27_10_10<TemplateMode<AMGX_mode_dDDI>::Type> CsrSparsityILU1Tests_Poisson27_10_10_dDDI;

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

DECLARE_UNITTEST_BEGIN_EXTD(CsrSparsityILU1Tests_Poisson27_100_100, CsrSparsityILU1Tests_Base<T_Config>);

void run()
{
  AMG_Config cfg;
  CsrSparsityILU1Tests_Base<T_Config>::template check_csr_sparsity_poisson<T_Config::vecPrec, T_Config::matPrec>( 5, 100, 100, 100, cfg );
}

DECLARE_UNITTEST_END(CsrSparsityILU1Tests_Poisson27_100_100)

CsrSparsityILU1Tests_Poisson27_100_100<TemplateMode<AMGX_mode_dDDI>::Type> CsrSparsityILU1Tests_Poisson27_100_100_dDDI;

} // namespace amgx
