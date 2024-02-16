// SPDX-FileCopyrightText: 2011 - 2024 NVIDIA CORPORATION. All Rights Reserved.
//
// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#include "matrix.h"
#include "matrix_cusp.h"
#include "matrix_io.h"
#include <algorithm>
#include "cusp/gallery/poisson.h"
#include "determinism_checker.h"
#include "util.h"

using namespace amgx::testing_tools;

namespace amgx

{

std::vector<std::string> &split(const std::string &s, char delim, std::vector<std::string> &elems);
std::vector<std::string> split(const std::string &s, char delim);
AMGX_Mode getModeFromString(const char *strmode);
AMGX_Mode getModeFromString(const std::string &strmode);


template <class ValueType1, class ValueType2>
struct write_values
{
    static void write(const ValueType1 &a, const ValueType2 &b, std::ostream &str = std::cout)
    {
        str << "Values: first = " << a << ", second = " << b << std::endl;
    }
};

template<class TConfig1, class TConfig2>
struct write_values< Vector<TConfig1>, Vector<TConfig2> >
{
    static void write(const Vector<TConfig1> &a, const Vector<TConfig2> &b, std::ostream &str = std::cout)
    {
    }
};

template<class TConfig1, class TConfig2>
struct write_values< Matrix<TConfig1>, Matrix<TConfig2> >
{
    static void write(const Matrix<TConfig1> &a, const Matrix<TConfig2> &b, std::ostream &str = std::cout)
    {
    }
};


//
// This is the relative tolerance for the linear iterative solvers.
// This should be looser than getTolerance<>, which is the absolute
// tolerance used for check_equal
//
template <class ValueType>
struct IterativeRelTol
{
    static float get() { return 0; }
};

template <>
struct IterativeRelTol<float>
{
    static float get() { return 1e-3; }
};

template <>
struct IterativeRelTol<double>
{
    static double get() { return 1e-6; }
};

//////////////

template <class ValueType>
struct getTolerance
{
    static ValueType get()
    {
        return 0;
    }
};

template <>
struct getTolerance<float>
{
    static float get()
    {
        return 1e-6;
    }
};

template <>
struct getTolerance<double>
{
    static double get()
    {
        return 1e-12;
    }
};

template <>
struct getTolerance<bool>
{
    static bool get()
    {
        return true;
    }
};

////////////// check equality with tolerance

template <class ValueType1, class ValueType2, class ValueTypeTol>
struct check_equal_tolerance
{
    static bool check(const ValueType1 a, const ValueType2 b, const ValueTypeTol tol, std::ostream &str = std::cout)
    {
        if (abs(a - static_cast<ValueType1>(b)) <= static_cast<ValueType1>(tol)) { return true; }

        str  << "    _val1 = " << a << "    val2 = " << b << "    tol = " << tol;
        return false;
    }
};


template <class ValueType2, class ValueTypeTol>
struct check_equal_tolerance<double, ValueType2, ValueTypeTol>
{
    static bool check(const double a, const ValueType2 b, const ValueTypeTol tol, std::ostream &str = std::cout)
    {
        //double maxAbsoluteError = 1e+10*DBL_MIN;
        double db = static_cast<double>(b);

        if (a == db)
        {
            return true;
        }

        //if (fabs(a - db) < 10*maxAbsoluteError)
        //    return true;
        if ((fabs(a - db) / (fabs(db) + fabs(a))) <= tol)
        {
            return true;
        }

        if (fabs(a - db) <= tol)
        {
            return true;
        }

        str  << std::setprecision(14) << "    dval1 = " << a << "    val2 = " << b << "    tol = " << tol;
        return false;
        /*
        int64_t aInt = *(int64_t*)&a;
        // Make aInt lexicographically ordered as a twos-complement int
        if (aInt < 0)
            aInt = 0x8000000000000000 - aInt;
        // Make bInt lexicographically ordered as a twos-complement int
        int64_t bInt = *(int64_t*)&db;
        if (bInt < 0)
            bInt = 0x8000000000000000 - bInt;
        int64_t intDiff = abs(aInt - bInt);
        if (intDiff <= 2)
            return true;
        return false; */
        /*
        if (a == db)
          return true;

          if (fabs(a - static_cast<double>(b)) <= static_cast<double>(tol * (fabs(a) + fabs(b)) / 2.0)) return true;
          return false;*/
    }
};

template <class ValueType2, class ValueTypeTol>
struct check_equal_tolerance<float, ValueType2, ValueTypeTol>
{
    static bool check(const float a, const ValueType2 b, const ValueTypeTol tol, std::ostream &str = std::cout)
    {
        //float maxAbsoluteError = 1e+10*FLT_MIN;
        float db = static_cast<float>(b);

        if (a == db)
        {
            return true;
        }

        //if (fabsf(a - db) < maxAbsoluteError)
        //    return true;
        if ((fabsf(a - db) / (fabsf(db) + fabsf(a))) <= tol)
        {
            return true;
        }

        if (fabs(a - db) <= tol)
        {
            return true;
        }

        str  << std::setprecision(7) << "    fval1 = " << a << "    val2 = " << b << "    tol = " << tol;
        return false;
        /*float db = static_cast<float>(b);
        int aInt = *(int*)&a;
        // Make aInt lexicographically ordered as a twos-complement int
        if (aInt < 0)
            aInt = 0x8000000000000000 - aInt;
        // Make bInt lexicographically ordered as a twos-complement int
        int bInt = *(int*)&db;
        if (bInt < 0)
            bInt = 0x8000000000000000 - bInt;
        int intDiff = abs(aInt - bInt);
        if (intDiff <= 2)
            return true;
        return false; */
        /*if (fabsf(a - static_cast<float>(b)) <= static_cast<float>(tol * (fabs(a) + fabs(b)) / 2.0f)) return true;
        return false;*/
    }
};

template <class ValueType2, class ValueTypeTol>
struct  check_equal_tolerance<int, ValueType2, ValueTypeTol>
{
    static bool check(const int a, const ValueType2 b, const ValueTypeTol tol, std::ostream &str = std::cout)
    {
        if (abs(a - static_cast<int>(b)) <= static_cast<int>(tol)) { return true; }

        str  << "    ival1 = " << a << "    val2 = " << b << "    tol = " << tol;
        return false;
    }
};

template <>
struct check_equal_tolerance<bool, bool, bool>
{
    static bool check(const bool a, const bool b, const bool tol, std::ostream &str = std::cout)
    {
        if (tol == (a == b)) { return true; }

        str  << "    bval1 = " << a << "    val2 = " << b << "    tol = " << tol;
        return false;
    }
};


template<class TConfig1, class TConfig2, class ValueTypeTol>
struct check_equal_tolerance< Vector<TConfig1>, Vector<TConfig2>, ValueTypeTol >
{
    typedef typename TConfig1::VecPrec ValueType1;
    typedef typename TConfig2::VecPrec ValueType2;
    typedef Vector< TemplateConfig<AMGX_host, TConfig1::vecPrec, TConfig1::matPrec, TConfig1::indPrec> > Vector_h1;
    typedef Vector< TemplateConfig<AMGX_host, TConfig2::vecPrec, TConfig2::matPrec, TConfig2::indPrec> > Vector_h2;
    static bool check(const Vector<TConfig1> &a, const Vector<TConfig2> &b, const ValueTypeTol tol, std::ostream &str = std::cout)
    {
        Vector_h1 tV1 = a;
        Vector_h2 tV2 = b;

        if (tV1.size() != tV2.size()) { return false; }

        for (unsigned int i = 0; i < tV1.size(); i++)
        {
            if ( !check_equal_tolerance<ValueType1, ValueType2, ValueTypeTol>::check(tV1[i], tV2[i], tol, str) ) { str << "   index = " << i; return false; }
        }

        return true;
    }
};

////////////// check strict equality
template <class ValueType1, class ValueType2>
struct check_equal
{
    static bool check(ValueType1 a, ValueType2 b, std::ostream &str = std::cout)
    {
        if (a == static_cast<ValueType1>(b)) { return true; }

        str  << "    _val1 = " << a << "    val2 = " << b;
        return false;
    }
};

template <class ValueType2>
struct check_equal<double, ValueType2>
{
    static bool check(double a, ValueType2 b, std::ostream &str = std::cout)
    {
        return check_equal_tolerance<double, ValueType2, double>::check(a, b, getTolerance<double>::get(), str);
    }
};



template <class ValueType2>
struct check_equal<float, ValueType2>
{
    static bool check(float a, ValueType2 b, std::ostream &str = std::cout)
    {
        return check_equal_tolerance<float, ValueType2, float>::check(a, b, getTolerance<float>::get(), str);
    }
};


template<class TConfig1, class TConfig2>
struct check_equal< Vector<TConfig1>, Vector<TConfig2> >
{
    typedef typename TConfig1::VecPrec ValueType1;
    typedef typename TConfig2::VecPrec ValueType2;
    typedef Vector< TemplateConfig<AMGX_host, TConfig1::vecPrec, TConfig1::matPrec, TConfig1::indPrec> > Vector_h1;
    typedef Vector< TemplateConfig<AMGX_host, TConfig2::vecPrec, TConfig2::matPrec, TConfig2::indPrec> > Vector_h2;
    static bool check(const Vector<TConfig1> &a, const Vector<TConfig2> &b, std::ostream &str = std::cout)
    {
        return check_equal_tolerance< Vector<TConfig1>, Vector<TConfig2>, ValueType1 >::check(a, b, getTolerance<ValueType1>::get(), str);
    }
};

////////////// check equality for matrices


template<class TConfig1, class TConfig2, class ValueTypeTol>
struct check_equal_tolerance< Matrix<TConfig1>, Matrix<TConfig2>, ValueTypeTol >
{
    typedef typename TConfig1::IndPrec IndexType;

    typedef typename TConfig1::MatPrec ValueType1;
    typedef Matrix< TemplateConfig<AMGX_host, TConfig1::vecPrec, TConfig1::matPrec, TConfig1::indPrec> > Matrix_h1;
    typedef TemplateConfig<TConfig1::memSpace, AMGX_vecInt, TConfig1::matPrec, TConfig1::indPrec> IVec1;
    typedef typename TConfig1::template setVecPrec<(AMGX_VecPrecision)AMGX_GET_MODE_VAL(AMGX_MatPrecision, TConfig1::mode)>::Type VVec1;

    typedef typename TConfig2::MatPrec ValueType2;
    typedef Matrix< TemplateConfig<AMGX_host, TConfig2::vecPrec, TConfig2::matPrec, TConfig2::indPrec> > Matrix_h2;
    typedef TemplateConfig<TConfig2::memSpace, AMGX_vecInt, TConfig2::matPrec, TConfig2::indPrec> IVec2;
    typedef typename TConfig2::template setVecPrec<(AMGX_VecPrecision)AMGX_GET_MODE_VAL(AMGX_MatPrecision, TConfig2::mode)>::Type VVec2;
    static bool check(const Matrix<TConfig1> &a, const Matrix<TConfig2> &b, ValueTypeTol tol, std::ostream &str = std::cout)
    {
        Matrix_h1 B1 = a;
        Matrix_h2 B2 = b;

        if    (B1.get_num_rows() != B2.get_num_rows()) {str << "Matrix comparison: Different number of rows: " << B1.get_num_rows() << " and " << B2.get_num_rows(); return false;}

        if    (B1.get_num_cols() != B2.get_num_cols()) {str << "Matrix comparison: Different number of cols: " << B1.get_num_cols() << " and " << B2.get_num_cols(); return false;}

        if    (B1.get_num_nz() != B2.get_num_nz()) {str << "Matrix comparison: Different number of nnz: " << B1.get_num_nz() << " and " << B2.get_num_nz(); return false;}

        if    (B1.get_block_dimy() != B2.get_block_dimy()) {str << "Matrix comparison: Different blockdimy: " << B1.get_block_dimy() << " and " << B2.get_block_dimy(); return false;}

        if    (B1.get_block_dimx() != B2.get_block_dimx()) {str << "Matrix comparison: Different blockdimx: " << B1.get_block_dimx() << " and " << B2.get_block_dimx(); return false;}

        if    (B1.is_initialized() != B2.is_initialized()) {str << "Matrix comparison: Different initialization flag: " << B1.is_initialized() << " and " << B2.is_initialized(); return false;}

        if    (!check_equal_tolerance<Vector<VVec1>, Vector<VVec2>, ValueTypeTol>::check(B1.values, B2.values, tol, str)) {str << " in matrix values array"; return false;}

        if    (!check_equal_tolerance<Vector<IVec1>, Vector<IVec2>, IndexType>::check(B1.col_indices, B2.col_indices, 0, str)) {str << " in matrix col_indices array"; return false;}

        if    (!check_equal_tolerance<Vector<IVec1>, Vector<IVec2>, IndexType>::check(B1.diag, B2.diag, 0, str))  {str << " in matrix diag array"; return false;}

        if ( B1.hasProps(CSR) && B2.hasProps(CSR))
            if (!check_equal_tolerance<Vector<IVec1>, Vector<IVec2>, IndexType>::check(B1.row_offsets, B2.row_offsets, 0, str))
            {str << " in matrix row_offsets"; return false;}

        if ( B1.hasProps(COO) && B2.hasProps(COO))
            if (!check_equal_tolerance<Vector<IVec1>, Vector<IVec2>, IndexType>::check(B1.row_indices, B2.row_indices, 0, str))
            {str << " in matrix row_indices"; return false;}

        return true;
    }
};



template<class TConfig1, class TConfig2>
struct check_equal< Matrix<TConfig1>, Matrix<TConfig2> >
{
    typedef typename TConfig1::MatPrec ValueType1;
    typedef Matrix< TemplateConfig<AMGX_host, TConfig1::vecPrec, TConfig1::matPrec, TConfig1::indPrec> > Matrix_h1;
    typedef TemplateConfig<TConfig1::memSpace, AMGX_vecInt, TConfig1::matPrec, TConfig1::indPrec> IVec1;
    typedef typename TConfig1::template setVecPrec<(AMGX_VecPrecision)AMGX_GET_MODE_VAL(AMGX_MatPrecision, TConfig1::mode)>::Type VVec1;

    typedef typename TConfig2::MatPrec ValueType2;
    typedef Matrix< TemplateConfig<AMGX_host, TConfig2::vecPrec, TConfig2::matPrec, TConfig2::indPrec> > Matrix_h2;
    typedef TemplateConfig<TConfig2::memSpace, AMGX_vecInt, TConfig2::matPrec, TConfig2::indPrec> IVec2;
    typedef typename TConfig2::template setVecPrec<(AMGX_VecPrecision)AMGX_GET_MODE_VAL(AMGX_MatPrecision, TConfig2::mode)>::Type VVec2;
    static bool check(const Matrix<TConfig1> &a, const Matrix<TConfig2> &b, std::ostream &str = std::cout)
    {
        return check_equal_tolerance< Matrix<TConfig1>, Matrix<TConfig2>, ValueType1 >::check(a, b, getTolerance<ValueType1>::get(), str);
    }
};

template<class TConfig1, class TConfig2>
struct equalVectors
{
    typedef typename TConfig1::VecPrec ValueType;
    typedef Vector< TemplateConfig<AMGX_host, TConfig1::vecPrec, TConfig1::matPrec, TConfig1::indPrec> > Vector_h;
    static bool check(const Vector<TConfig1> &b1, const Vector<TConfig2> &b2, const ValueType tol, std::ostream &str = std::cout)
    {
        return check_equal_tolerance<Vector<TConfig1>, Vector<TConfig2>, ValueType>::check(b1, b2, tol, str);
    }
};

/*
template<class ClassName>
struct getChecksum {
  typedef typename TConfig1::VecPrec ValueType;
  typedef Vector< TemplateConfig<AMGX_host, TConfig1::vecPrec, TConfig1::matPrec, TConfig1::indPrec> > Vector_h;
  static bool check(const Vector<TConfig1>& b1, const Vector<TConfig2>& b2, const ValueType tol, std::ostream& str = std::cout)
  {
      return check_equal_tolerance<Vector<TConfig1>, Vector<TConfig2>, ValueType>::check(b1, b2, tol, str);
  }
};
*/


template<class TConfig1, class TConfig2>
struct equalMatrices
{
    typedef typename TConfig1::MatPrec ValueType;
    typedef typename TConfig1::IndPrec IndexType;
    typedef Matrix< TemplateConfig<AMGX_host, TConfig1::vecPrec, TConfig1::matPrec, TConfig1::indPrec> > Matrix_h;
    typedef TemplateConfig<TConfig1::memSpace, AMGX_vecInt, TConfig1::matPrec, TConfig1::indPrec> IVec;
    typedef typename TConfig1::template setVecPrec<(AMGX_VecPrecision)AMGX_GET_MODE_VAL(AMGX_MatPrecision, TConfig1::mode)>::Type VVec;
    static bool check(const Matrix<TConfig1> &A1, const Matrix<TConfig2> &A2, bool check_props = true, std::ostream &str = std::cout)
    {
        Matrix_h B1 = A1;
        Matrix_h B2 = A2;

        if (check_props)
            if (B1.getProps() != B2.getProps()) {str << "Matrix comparison: Different Props: " << B1.getProps() << " and " << B2.getProps(); return false;}

        if    (B1.get_num_rows() != B2.get_num_rows()) {str << "Matrix comparison: Different number of rows: " << B1.get_num_rows() << " and " << B2.get_num_rows(); return false;}

        if    (B1.get_num_cols() != B2.get_num_cols()) {str << "Matrix comparison: Different number of cols: " << B1.get_num_cols() << " and " << B2.get_num_cols(); return false;}

        if    (B1.get_num_nz() != B2.get_num_nz()) {str << "Matrix comparison: Different number of nnz: " << B1.get_num_nz() << " and " << B2.get_num_nz(); return false;}

        if    (B1.get_block_dimy() != B2.get_block_dimy()) {str << "Matrix comparison: Different blockdimy: " << B1.get_block_dimy() << " and " << B2.get_block_dimy(); return false;}

        if    (B1.get_block_dimx() != B2.get_block_dimx()) {str << "Matrix comparison: Different blockdimx: " << B1.get_block_dimx() << " and " << B2.get_block_dimx(); return false;}

        if    (B1.is_initialized() != B2.is_initialized()) {str << "Matrix comparison: Different initialization flag: " << B1.is_initialized() << " and " << B2.is_initialized(); return false;}

        if    (!check_equal_tolerance<Vector<VVec>, Vector<VVec>, ValueType>::check(B1.values, B2.values, getTolerance<ValueType>::get(), str)) {str << " in matrix values array"; return false;}

        if    (!check_equal_tolerance<Vector<IVec>, Vector<IVec>, IndexType>::check(B1.col_indices, B2.col_indices, 0, str)) {str << " in matrix col_indices array"; return false;}

        if    (!check_equal_tolerance<Vector<IVec>, Vector<IVec>, IndexType>::check(B1.diag, B2.diag, 0, str))  {str << " in matrix diag array"; return false;}

        if ( B1.hasProps(CSR) && B2.hasProps(CSR))
            if (!check_equal_tolerance<Vector<IVec>, Vector<IVec>, IndexType>::check(B1.row_offsets, B2.row_offsets, 0, str))
            {str << " in matrix row_offsets"; return false;}

        if ( B1.hasProps(COO) && B2.hasProps(COO))
            if (!check_equal_tolerance<Vector<IVec>, Vector<IVec>, IndexType>::check(B1.row_indices, B2.row_indices, 0, str))
            {str << " in matrix row_indices"; return false;}

        return true;
    }
};

template<class Type1, class Type2>
struct check_hashsum_equal;


template<class TConfig1, class TConfig2>
struct check_hashsum_equal< Vector<TConfig1>, Vector<TConfig2> >
{
    typedef typename TConfig1::VecPrec ValueType1;
    typedef typename TConfig2::VecPrec ValueType2;
    typedef Vector< TemplateConfig<AMGX_device, TConfig1::vecPrec, TConfig1::matPrec, TConfig1::indPrec> > Vector_h1;
    typedef Vector< TemplateConfig<AMGX_device, TConfig2::vecPrec, TConfig2::matPrec, TConfig2::indPrec> > Vector_h2;
    static bool check(const Vector<TConfig1> &a, const Vector<TConfig2> &b, std::ostream &str = std::cout)
    {
        Vector_h1 tV1 = a;
        Vector_h2 tV2 = b;

        if (tV1.size() != tV2.size()) { return false; }

        if (sizeof(TConfig1::vecPrec) != sizeof(TConfig2::vecPrec)) { return false; }

        hash_path_determinism_checker *checker = hash_path_determinism_checker::singleton();

        if (checker->checksum(tV1.raw(), (long long int)(sizeof(TConfig1::vecPrec)*tV1.size())) !=  checker->checksum(tV2.raw(), (long long int)(sizeof(TConfig1::vecPrec)*tV1.size())))
        {
            return false;
        }

        return true;
    }
};

template<class TConfig1, class TConfig2>
struct check_hashsum_equal< Matrix<TConfig1>, Matrix<TConfig2> >
{
    typedef typename TConfig1::MatPrec ValueType;
    typedef typename TConfig1::IndPrec IndexType;
    typedef Matrix< TemplateConfig<AMGX_host, TConfig1::vecPrec, TConfig1::matPrec, TConfig1::indPrec> > Matrix_h;
    typedef TemplateConfig<TConfig1::memSpace, AMGX_vecInt, TConfig1::matPrec, TConfig1::indPrec> IVec;
    typedef typename TConfig1::template setVecPrec<(AMGX_VecPrecision)AMGX_GET_MODE_VAL(AMGX_MatPrecision, TConfig1::mode)>::Type VVec;
    static bool check(const Matrix<TConfig1> &A1, const Matrix<TConfig2> &A2, std::ostream &str = std::cout)
    {
        Matrix_h B1 = A1;
        Matrix_h B2 = A2;

        if    (B1.getProps() != B2.getProps()) {str << "Matrix comparison: Different Props: " << B1.getProps() << " and " << B2.getProps(); return false;}

        if    (B1.get_num_rows() != B2.get_num_rows()) {str << "Matrix comparison: Different number of rows: " << B1.get_num_rows() << " and " << B2.get_num_rows(); return false;}

        if    (B1.get_num_cols() != B2.get_num_cols()) {str << "Matrix comparison: Different number of cols: " << B1.get_num_cols() << " and " << B2.get_num_cols(); return false;}

        if    (B1.get_num_nz() != B2.get_num_nz()) {str << "Matrix comparison: Different number of nnz: " << B1.get_num_nz() << " and " << B2.get_num_nz(); return false;}

        if    (B1.get_block_dimy() != B2.get_block_dimy()) {str << "Matrix comparison: Different blockdimy: " << B1.get_block_dimy() << " and " << B2.get_block_dimy(); return false;}

        if    (B1.get_block_dimx() != B2.get_block_dimx()) {str << "Matrix comparison: Different blockdimx: " << B1.get_block_dimx() << " and " << B2.get_block_dimx(); return false;}

        if    (B1.is_initialized() != B2.is_initialized()) {str << "Matrix comparison: Different initialization flag: " << B1.is_initialized() << " and " << B2.is_initialized(); return false;}

        if    (!check_hashsum_equal< Vector<VVec>, Vector<VVec> >::check(B1.values, B2.values, str)) {str << " in matrix values array"; return false;}

        if    (!check_hashsum_equal< Vector<IVec>, Vector<IVec> >::check(B1.col_indices, B2.col_indices, str)) {str << " in matrix col_indices array"; return false;}

        if    (!check_hashsum_equal< Vector<IVec>, Vector<IVec> >::check(B1.diag, B2.diag, str))  {str << " in matrix diag array"; return false;}

        if ( B1.hasProps(CSR) && B2.hasProps(CSR))
            if (!check_hashsum_equal< Vector<IVec>, Vector<IVec> >::check(B1.row_offsets, B2.row_offsets, str))
            {str << " in matrix row_offsets"; return false;}

        if ( B1.hasProps(COO) && B2.hasProps(COO))
            if (!check_hashsum_equal< Vector<IVec>, Vector<IVec> >::check(B1.row_indices, B2.row_indices, str))
            {str << " in matrix row_indices"; return false;}

        return true;
    }
};




/////////////////////////////////////////////
template<class TConfig>
struct generateMatrixRandomStruct
{
    static void generate(Matrix<TConfig> &A, int max_rows, bool diag_prop, int bsize, bool symmetric, int max_nnz_per_row = 10);
    static void generateExact(Matrix<TConfig> &A, int num_rows, bool diag_prop, int bsize, bool symmetric, int max_nnz_per_row = 10);
};

template <AMGX_VecPrecision t_vecPrec, AMGX_MatPrecision t_matPrec, AMGX_IndPrecision t_indPrec>
struct generateMatrixRandomStruct<TemplateConfig<AMGX_host, t_vecPrec, t_matPrec, t_indPrec> >
{
    typedef Matrix<TemplateConfig<AMGX_host, t_vecPrec, t_matPrec, t_indPrec> > Matrix_h;
    typedef typename Matrix_h::index_type IndexType;

    static void generate (Matrix_h &A, int max_rows, bool diag_prop, int bsize, bool symmetric, int max_nnz_per_row = 10)
    {
        int new_rows = std::max((int)(((float)rand() / RAND_MAX) * max_rows), 1);
        generateExact(A, new_rows, diag_prop, bsize, symmetric, max_nnz_per_row);
    }

    static void generateExact (Matrix_h &A, int new_rows, bool diag_prop, int bsize, bool symmetric, int max_nnz_per_row = 10)
    {
        Matrix_h newA;
        newA.set_initialized(0);
        //int new_rows = std::max((int)(((float)rand() / RAND_MAX)*max_rows), 1);
        //printf("generating matrix with %d rows...\n", new_rows);
        int props = CSR | (diag_prop ? DIAG : 0);
        int bsize_sq = bsize * bsize;
        newA.addProps(props);
        //printf("nr = %d, bsize = %d, diag = %d\n", new_rows, bsize, (diag_prop ? 1 : 0));
        newA.resize(new_rows, new_rows, 0, bsize, bsize, 0);
        newA.values.resize(0);
        newA.col_indices.resize(0);
        std::vector<IndexType> row_col_idx;
        int cur_nnz = 0;

        if (!symmetric)
        {
            IndexType cur_ro = 0;

            for (int i = 0; i < newA.get_num_rows(); i++)
            {
                row_col_idx.clear();
                newA.row_offsets[i] = cur_ro;

                if (!diag_prop) { row_col_idx.push_back(i); }

                if (row_col_idx.size() + (diag_prop ? 1 : 0)  < max_nnz_per_row)
                {
                    int add_nnz = min (new_rows, std::max( 1, (int)( ((float)rand() / RAND_MAX) * (max_nnz_per_row - row_col_idx.size()) )) );

                    while (add_nnz > 0)
                    {
                        IndexType new_col = rand() % new_rows;

                        if ( row_col_idx.end() == std::find(row_col_idx.begin(), row_col_idx.end(), new_col) )
                        {
                            if (!diag_prop || (new_col != i))
                            {
                                row_col_idx.push_back(new_col);
                            }
                        }

                        --add_nnz;
                    }
                }

                std::sort(row_col_idx.begin(), row_col_idx.end());

                for (unsigned int j = 0; j < row_col_idx.size(); j++)
                {
                    for (int k = 0; k < bsize_sq; k++)
                    {
                        newA.values.push_back(1.0);
                    }

                    newA.col_indices.push_back(row_col_idx[j]);
                    cur_nnz++;
                }

                cur_ro = cur_nnz;
            }

            newA.row_offsets[new_rows] = cur_ro;
        }
        else
        {
            std::vector< std::vector<IndexType> > col_idxs;
            col_idxs.resize(new_rows);

            if (!diag_prop)
                for (int i = 0; i < new_rows; ++i)
                {
                    col_idxs[i].push_back(i);
                }

            IndexType x, y;

            for (int i = 0; i < new_rows * max_nnz_per_row / 2; ++i)
            {
                x = rand() % new_rows;
                y = rand() % new_rows;

                if ( (x == y) || (col_idxs[x].size() == max_nnz_per_row) || (col_idxs[y].size() == max_nnz_per_row) || (std::find(col_idxs[x].begin(), col_idxs[x].end(), y) != col_idxs[x].end()) )
                {
                    continue;
                }

                col_idxs[y].push_back(x);
                col_idxs[x].push_back(y);
            }

            for (int i = 0; i < new_rows; ++i)
            {
                std::sort(col_idxs[i].begin(), col_idxs[i].end());
                newA.row_offsets[i] = cur_nnz;

                for (unsigned int j = 0; j < col_idxs[i].size(); ++j)
                {
                    newA.col_indices.push_back(col_idxs[i][j]);

                    for (int k = 0; k < bsize_sq; k++)
                    {
                        newA.values.push_back(1.0);
                    }

                    cur_nnz++;
                }
            }

            newA.row_offsets[new_rows] = cur_nnz;
        }

        newA.set_num_nz(cur_nnz);

        if (diag_prop)
        {
            int new_vals = (newA.get_num_nz() + new_rows) * bsize_sq;
            newA.values.resize(new_vals);
        }
        else
        {
            int new_vals = (newA.get_num_nz() + 1) * bsize_sq;
            newA.values.resize(new_vals);
            thrust_wrapper::fill<AMGX_host>(newA.values.begin() + (newA.diagOffset()*newA.get_block_size()), newA.values.end(), 0.0);
        }

        newA.computeDiagonal();
        A = newA;
        newA = A;
        A.set_initialized(1);
    }
};

template <AMGX_VecPrecision t_vecPrec, AMGX_MatPrecision t_matPrec, AMGX_IndPrecision t_indPrec>
struct generateMatrixRandomStruct<TemplateConfig<AMGX_device, t_vecPrec, t_matPrec, t_indPrec> >
{
    typedef Matrix<TemplateConfig<AMGX_host, t_vecPrec, t_matPrec, t_indPrec> > Matrix_h;
    typedef Matrix<TemplateConfig<AMGX_device, t_vecPrec, t_matPrec, t_indPrec> > Matrix_d;
    static void generate (Matrix_d &A, int max_rows, bool diag_prop, int bsize, bool symmetric, int max_nnz_per_row = 10)
    {
        Matrix_h tA;
        generateMatrixRandomStruct< TemplateConfig<AMGX_host, t_vecPrec, t_matPrec, t_indPrec> >::generate(tA, max_rows, diag_prop, bsize, symmetric, max_nnz_per_row);
        A = tA;
    }
    static void generateExact (Matrix_d &A, int num_rows, bool diag_prop, int bsize, bool symmetric, int max_nnz_per_row = 10)
    {
        Matrix_h tA;
        generateMatrixRandomStruct< TemplateConfig<AMGX_host, t_vecPrec, t_matPrec, t_indPrec> >::generateExact(tA, num_rows, diag_prop, bsize, symmetric, max_nnz_per_row);
        A = tA;
    }
};


template<class Container>
struct fillRandom
{
    static void fill(Container &A)
    {
        for (unsigned int i = 0; i < A.size(); i++)
        {
            A[i] = rand()/RAND_MAX;
        }
    }
};


template<class TConfig>
struct fillRandom < Matrix<TConfig> >
{
    typedef typename TConfig::MatPrec ValueType;
    typedef Matrix< TemplateConfig<AMGX_host, TConfig::vecPrec, TConfig::matPrec, TConfig::indPrec> > Matrix_h;
    static void fill(Matrix<TConfig> &A)
    {
        typedef Vector< TemplateConfig<AMGX_host, TConfig::vecPrec, TConfig::matPrec, TConfig::indPrec> > Vector_h;
        Vector_h tV( (A.get_num_nz() + (A.hasProps(DIAG) ? A.get_num_rows() : 1)) * A.get_block_size() );

        if (A.hasProps(DIAG))
        {
            for (unsigned int i = 0; i < tV.size(); i++)
            {
                tV[i] = ( ((float)(rand()) / RAND_MAX) - 0.5 ) * 2 * (i >= A.get_num_nz()*A.get_block_size() ? 10 : 1);
            }
        }
        else
        {
            for (int r = 0; r < A.get_num_rows(); r++)
            {
                for (int cidx = A.row_offsets[r]; cidx < A.row_offsets[r+1]; cidx++)
                {
                    tV[cidx] = ( ((float)(rand()) / RAND_MAX) - 0.5 ) * 2 * (r == A.col_indices[cidx] ? 10 : 1);
                }
            }

            for (unsigned int i = A.get_num_nz() * A.get_block_size(); i < tV.size(); i++)
            {
                tV[i] = 0;
            }
        }

        bool was_initialized = A.is_initialized();
        A.set_initialized(0);
        A.values.copy(tV);
        A.set_initialized(was_initialized);
    }
};

template<class TConfig>
struct fillRandom < Vector<TConfig> >
{
    typedef typename TConfig::VecPrec ValueType;
    typedef Vector< TemplateConfig<AMGX_host, TConfig::vecPrec, TConfig::matPrec, TConfig::indPrec> > Vector_h;
    static void fill(Vector<TConfig> &b)
    {
        //amgx::thrust::generate(b.begin(), b.end(), rand);
        Vector_h tV(b.size());

        for (unsigned int i = 0; i < tV.size(); i++)
        {
            tV[i] = ( ((float)(rand()) / RAND_MAX) - 0.5 ) * 2;
        }

        b.copy(tV);
    }
};

template <class TConfig>
void generatePoissonForTest(Matrix<TConfig > &Aout, int block_size, bool diag_prop, int points, int x, int y, int z = 1)
{
    typedef TemplateConfig<AMGX_host,TConfig::vecPrec,TConfig::matPrec,TConfig::indPrec> TConfig_h;
    
    Matrix<TConfig_h > A;
    {
        A.set_initialized(0);
        A.addProps(CSR);
        MatrixCusp<TConfig_h, cusp::csr_format> wA(&A);

        switch (points)
        {
            case 5:
                cusp::gallery::poisson5pt(wA, x, y);
                break;

            case 7:
                cusp::gallery::poisson7pt(wA, x, y, z);
                break;

            case 9:
                cusp::gallery::poisson9pt(wA, x, y);
                break;

            case 27:
                cusp::gallery::poisson27pt(wA, x, y, z);
                break;
        }

        A.set_initialized(1);
    }
    Aout = A;
}

}; // namespace amgx

