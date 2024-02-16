// SPDX-FileCopyrightText: 2011 - 2024 NVIDIA CORPORATION. All Rights Reserved.
//
// SPDX-License-Identifier: BSD-3-Clause

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

DECLARE_UNITTEST_BEGIN(CsrSparsityTests_Base);

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
    #pragma omp parallel  for  shared(nRows)

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

template< typename Index_vector >
void
compute_sparsity( const Index_vector &A_rows, const Index_vector &A_cols,
                  const Index_vector &B_rows, const Index_vector &B_cols,
                  const Index_vector &C_rows,       Index_vector &C_cols )
{
    typedef typename Index_vector::value_type Index_type;
#ifdef USE_CPP_TR1
    typedef std::tr1::unordered_set<Index_type> Set;
#else
    typedef std::set<Index_type> Set;
#endif
    int nRows = static_cast<int>( A_rows.size( ) - 1 );
    #pragma omp parallel  for  shared(nRows)

    for ( int aRowId = 0 ; aRowId < nRows ; ++aRowId )
    {
        Set cCols;

        for ( Index_type aColIt = A_rows[aRowId], aColEnd = A_rows[aRowId + 1] ; aColIt < aColEnd ; ++aColIt )
        {
            Index_type bRowId = A_cols[aColIt];

            for ( Index_type bColIt = B_rows[bRowId], bColEnd = B_rows[bRowId + 1] ; bColIt < bColEnd ; ++bColIt )
            {
                cCols.insert( B_cols[bColIt] );
            }
        }

        Index_type cRowIt = C_rows[aRowId];

        for ( typename Set::const_iterator it = cCols.begin() ; it != cCols.end() ; ++it, ++cRowIt )
        {
            assert( cRowIt < C_rows[aRowId + 1] );
            C_cols[cRowIt] = *it;
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
check_csr_sparsity( const Matrix<TemplateConfig<AMGX_host, VecPrecision, MatPrecision, AMGX_indInt> > &A_h, void *wk )
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
        thrust_wrapper::exclusive_scan<AMGX_host>( C_h.row_offsets.begin( ), C_h.row_offsets.end( ), C_h.row_offsets.begin( ) );
        cudaCheckError();
        int nVals = C_h.row_offsets[A_h.get_num_rows()];
        C_h.col_indices.resize( nVals );
        C_h.values.resize( nVals );
        C_h.set_num_nz( nVals );
        compute_sparsity( A_h.row_offsets, A_h.col_indices, B_h.row_offsets, B_h.col_indices, C_h.row_offsets, C_h.col_indices );
    }
    Matrix_d C_d;
    {
        Matrix_d A_d( A_h ), B_d( B_h );
        CSR_Multiply<Config_d>::csr_sparsity( A_d, B_d, C_d, wk );
    }
    Matrix_d C_d_ref( C_h );
    compare_matrices( C_d, C_d_ref );
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template< AMGX_VecPrecision VecPrecision, AMGX_MatPrecision MatPrecision >
void
check_csr_sparsity_square( const Matrix<TemplateConfig<AMGX_host, VecPrecision, MatPrecision, AMGX_indInt> > &A_h, void *wk )
{
    typedef TemplateConfig<AMGX_host,   VecPrecision, MatPrecision, AMGX_indInt> Config_h;
    typedef TemplateConfig<AMGX_device, VecPrecision, MatPrecision, AMGX_indInt> Config_d;
    typedef Matrix<Config_h> Matrix_h;
    typedef Matrix<Config_d> Matrix_d;
    Matrix_h C_h;
    C_h.set_num_rows( A_h.get_num_rows() );
    C_h.set_num_cols( A_h.get_num_rows() );
    C_h.row_offsets.resize( A_h.get_num_rows() + 1 );
    std::ostringstream buffer;
    {
        count_non_zeroes( A_h.row_offsets, A_h.col_indices, A_h.row_offsets, A_h.col_indices, C_h.row_offsets );
        thrust_wrapper::exclusive_scan<AMGX_host>( C_h.row_offsets.begin( ), C_h.row_offsets.end( ), C_h.row_offsets.begin( ) );
        cudaCheckError();
        int nVals = C_h.row_offsets[A_h.get_num_rows()];
        C_h.col_indices.resize( nVals );
        C_h.values.resize( nVals );
        C_h.set_num_nz( nVals );
        compute_sparsity( A_h.row_offsets, A_h.col_indices, A_h.row_offsets, A_h.col_indices, C_h.row_offsets, C_h.col_indices );
    }
    Matrix_d C_d;
    {
        Matrix_d A_d( A_h );
        CSR_Multiply<Config_d>::csr_sparsity( A_d, C_d, wk );
    }
    Matrix_d C_d_ref( C_h );
    compare_matrices( C_d, C_d_ref );
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template< AMGX_VecPrecision VecPrecision, AMGX_MatPrecision MatPrecision >
void
check_csr_sparsity_poisson( int points, int nx, int ny, int nz, AMG_Config &cfg )
{
    typedef TemplateConfig<AMGX_host,   VecPrecision, MatPrecision, AMGX_indInt> Config_h;
    typedef TemplateConfig<AMGX_device, VecPrecision, MatPrecision, AMGX_indInt> Config_d;
    typedef Matrix<Config_h> Matrix_h;
    Matrix_h A_h;
    A_h.set_initialized(0);
    A_h.addProps(CSR);
    MatrixCusp<Config_h, cusp::csr_format> wA(&A_h);

    switch (points)
    {
        case 5:
            cusp::gallery::poisson5pt(wA, nx, ny);
            break;

        case 7:
            cusp::gallery::poisson7pt(wA, nx, ny, nz);
            break;

        case 9:
            cusp::gallery::poisson9pt(wA, nx, ny);
            break;

        case 27:
            cusp::gallery::poisson27pt(wA, nx, ny, nz);
            break;

        default:
            printf("Error invalid number of poisson points specified, valid numbers are 5, 7, 9, 27\n");
    }

    A_h.set_initialized(1);
    void *wk = CSR_Multiply<Config_d>::csr_workspace_create( cfg, "default" );
    check_csr_sparsity( A_h, wk );
    CSR_Multiply<Config_d>::csr_workspace_delete( wk );
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template< AMGX_VecPrecision VecPrecision, AMGX_MatPrecision MatPrecision >
void
check_csr_sparsity_square_poisson( int points, int nx, int ny, int nz, AMG_Config &cfg )
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
    check_csr_sparsity_square( A_h, wk );
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
    typedef Vector<Config_h> Vector_h;
    Matrix_h A_h;
    Vector_h x_h, b_h;
    A_h.set_initialized(0);
    A_h.addProps(CSR);
    UNITTEST_ASSERT_TRUE(MatrixIO<Config_h>::readSystem( filename.c_str(), A_h, b_h, x_h ) == AMGX_OK);
    A_h.set_initialized(1);
    void *wk = CSR_Multiply<Config_d>::csr_workspace_create( cfg, "default" );
    check_csr_sparsity_square( A_h, wk );
    CSR_Multiply<Config_d>::csr_workspace_delete( wk );
}

DECLARE_UNITTEST_END(CsrSparsityTests_Base);

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

DECLARE_UNITTEST_BEGIN_EXTD(CsrSparsityTests_Poisson5_10_10, CsrSparsityTests_Base<T_Config>);

void run()
{
  AMG_Config cfg;
  CsrSparsityTests_Base<T_Config>::template check_csr_sparsity_poisson<T_Config::vecPrec, T_Config::matPrec>( 5, 10, 10, 10, cfg );
  CsrSparsityTests_Base<T_Config>::template check_csr_sparsity_square_poisson<T_Config::vecPrec, T_Config::matPrec>( 5, 10, 10, 10, cfg );
}

DECLARE_UNITTEST_END(CsrSparsityTests_Poisson5_10_10)

CsrSparsityTests_Poisson5_10_10<TemplateMode<AMGX_mode_dDDI>::Type> CsrSparsityTests_Poisson5_10_10_dDDI;

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

DECLARE_UNITTEST_BEGIN_EXTD(CsrSparsityTests_Poisson5_100_100, CsrSparsityTests_Base<T_Config>);

void run()
{
  AMG_Config cfg;
  CsrSparsityTests_Base<T_Config>::template check_csr_sparsity_poisson<T_Config::vecPrec, T_Config::matPrec>( 5, 100, 100, 100, cfg );
  CsrSparsityTests_Base<T_Config>::template check_csr_sparsity_square_poisson<T_Config::vecPrec, T_Config::matPrec>( 5, 100, 100, 100, cfg );
}

DECLARE_UNITTEST_END(CsrSparsityTests_Poisson5_100_100)

CsrSparsityTests_Poisson5_100_100<TemplateMode<AMGX_mode_dDDI>::Type> CsrSparsityTests_Poisson5_100_100_dDDI;

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

DECLARE_UNITTEST_BEGIN_EXTD(CsrSparsityTests_Poisson7_10_10, CsrSparsityTests_Base<T_Config>);

void run()
{
  AMG_Config cfg;
  CsrSparsityTests_Base<T_Config>::template check_csr_sparsity_poisson<T_Config::vecPrec, T_Config::matPrec>( 5, 10, 10, 10, cfg );
  CsrSparsityTests_Base<T_Config>::template check_csr_sparsity_square_poisson<T_Config::vecPrec, T_Config::matPrec>( 5, 10, 10, 10, cfg );
}

DECLARE_UNITTEST_END(CsrSparsityTests_Poisson7_10_10)

CsrSparsityTests_Poisson7_10_10<TemplateMode<AMGX_mode_dDDI>::Type> CsrSparsityTests_Poisson7_10_10_dDDI;

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

DECLARE_UNITTEST_BEGIN_EXTD(CsrSparsityTests_Poisson7_100_100, CsrSparsityTests_Base<T_Config>);

void run()
{
  AMG_Config cfg;
  CsrSparsityTests_Base<T_Config>::template check_csr_sparsity_poisson<T_Config::vecPrec, T_Config::matPrec>( 5, 100, 100, 100, cfg );
  CsrSparsityTests_Base<T_Config>::template check_csr_sparsity_square_poisson<T_Config::vecPrec, T_Config::matPrec>( 5, 100, 100, 100, cfg );
}

DECLARE_UNITTEST_END(CsrSparsityTests_Poisson7_100_100)

CsrSparsityTests_Poisson7_100_100<TemplateMode<AMGX_mode_dDDI>::Type> CsrSparsityTests_Poisson7_100_100_dDDI;

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

DECLARE_UNITTEST_BEGIN_EXTD(CsrSparsityTests_Poisson9_10_10, CsrSparsityTests_Base<T_Config>);

void run()
{
  AMG_Config cfg;
  CsrSparsityTests_Base<T_Config>::template check_csr_sparsity_poisson<T_Config::vecPrec, T_Config::matPrec>( 5, 10, 10, 10, cfg );
  CsrSparsityTests_Base<T_Config>::template check_csr_sparsity_square_poisson<T_Config::vecPrec, T_Config::matPrec>( 5, 10, 10, 10, cfg );
}

DECLARE_UNITTEST_END(CsrSparsityTests_Poisson9_10_10)

CsrSparsityTests_Poisson9_10_10<TemplateMode<AMGX_mode_dDDI>::Type> CsrSparsityTests_Poisson9_10_10_dDDI;

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

DECLARE_UNITTEST_BEGIN_EXTD(CsrSparsityTests_Poisson9_100_100, CsrSparsityTests_Base<T_Config>);

void run()
{
  AMG_Config cfg;
  CsrSparsityTests_Base<T_Config>::template check_csr_sparsity_poisson<T_Config::vecPrec, T_Config::matPrec>( 5, 100, 100, 100, cfg );
  CsrSparsityTests_Base<T_Config>::template check_csr_sparsity_square_poisson<T_Config::vecPrec, T_Config::matPrec>( 5, 100, 100, 100, cfg );
}

DECLARE_UNITTEST_END(CsrSparsityTests_Poisson9_100_100)

CsrSparsityTests_Poisson9_100_100<TemplateMode<AMGX_mode_dDDI>::Type> CsrSparsityTests_Poisson9_100_100_dDDI;

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

DECLARE_UNITTEST_BEGIN_EXTD(CsrSparsityTests_Poisson27_10_10, CsrSparsityTests_Base<T_Config>);

void run()
{
  AMG_Config cfg;
  CsrSparsityTests_Base<T_Config>::template check_csr_sparsity_poisson<T_Config::vecPrec, T_Config::matPrec>( 5, 10, 10, 10, cfg );
  CsrSparsityTests_Base<T_Config>::template check_csr_sparsity_square_poisson<T_Config::vecPrec, T_Config::matPrec>( 5, 10, 10, 10, cfg );
}

DECLARE_UNITTEST_END(CsrSparsityTests_Poisson27_10_10)

CsrSparsityTests_Poisson27_10_10<TemplateMode<AMGX_mode_dDDI>::Type> CsrSparsityTests_Poisson27_10_10_dDDI;

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

DECLARE_UNITTEST_BEGIN_EXTD(CsrSparsityTests_Poisson27_100_100, CsrSparsityTests_Base<T_Config>);

void run()
{
  AMG_Config cfg;
  CsrSparsityTests_Base<T_Config>::template check_csr_sparsity_poisson<T_Config::vecPrec, T_Config::matPrec>( 5, 100, 100, 100, cfg );
  CsrSparsityTests_Base<T_Config>::template check_csr_sparsity_square_poisson<T_Config::vecPrec, T_Config::matPrec>( 5, 100, 100, 100, cfg );
}

DECLARE_UNITTEST_END(CsrSparsityTests_Poisson27_100_100)

CsrSparsityTests_Poisson27_100_100<TemplateMode<AMGX_mode_dDDI>::Type> CsrSparsityTests_Poisson27_100_100_dDDI;


} // namespace amgx
