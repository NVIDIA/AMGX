// SPDX-FileCopyrightText: 2011 - 2025 NVIDIA CORPORATION. All Rights Reserved.
//
// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#include <basic_types.h>
#include "vector.h"

#include <amgx_types/util.h>

namespace amgx
{

//computes out=a*x+b*y+c*z
template<class Vector, class Scalar>
void axpbypcz(const Vector &x, const Vector &y, const Vector &z, Vector &out, Scalar a, Scalar b, Scalar c, int offset = 0, int size = -1);

//computes out=a*x+b*y
template<class Vector, class Scalar>
void axpby(const Vector &x, const Vector &y, Vector &out, Scalar a, Scalar b, int offset = 0, int size = -1);

//computes y+=a*x
template<class Vector, class Scalar>
void axpy(const Vector &x, Vector &y, Scalar a, int offset = 0, int size = -1);


template<class Vector, class Scalar>
void axpy(Vector &x, Vector &y, Scalar a, int offsetx, int offsety, int size);

//computes x=a*x
template<class Vector, class Scalar> void scal(Vector &x, Scalar a, int offset = 0, int size = -1);

//computes r=Ax-b
template <class Matrix, class Vector>
void axmb(Matrix &A, Vector &x, Vector &b, Vector &r, int offset = 0, int size = -1);

//computes dense_matrix_vector_product
template <class Vector, class Scalar>
void gemv_extnd(bool trans, const Vector &A, const Vector &x, Vector &y, int m, int n,
                Scalar alpha, Scalar beta, int incx, int incy, int lda,
                int offsetA, int offsetx, int offsety);
template <class Vector>
void trsv_extnd(bool trans, const Vector &A, int lda, Vector &x, int n, int incx, int offsetA);
//computes the dot product
template <class Vector>
typename Vector::value_type dotc(const Vector &a, const Vector &b, int offset = 0, int size = -1);

template <class Vector>
typename Vector::value_type dotc(const Vector &a, const Vector &b, int offseta, int offsetb, int size);

template <class Matrix, class Vector>
typename Vector::value_type dot(const Matrix &A, const Vector &x, const Vector &y);

template <class Matrix, class Vector>
typename Vector::value_type dot(const Matrix &A, const Vector &x, const Vector &y, int offsetx, int offsety);

template <class Vector>
void copy(const Vector &a, Vector &b, int offset = 0, int size = -1);

template <class Vector>
void copy_ext(Vector &a, Vector &b, int offseta, int offsetb, int size);

template <class Vector>
void fill(Vector &x, typename Vector::value_type val, int offset = 0, int size = -1);

template <class Vector>
typename types::PODTypes<typename Vector::value_type>::type
nrm1(const Vector &x, int offset = 0, int size = -1);

template <AMGX_VecPrecision t_vecPrec, AMGX_MatPrecision t_matPrec, AMGX_IndPrecision t_indPrec>
typename types::PODTypes<typename Vector<TemplateConfig<AMGX_host, t_vecPrec, t_matPrec, t_indPrec> >::value_type>::type
nrm1(const Vector<TemplateConfig<AMGX_host, t_vecPrec, t_matPrec, t_indPrec> > &x, int offset, int size);

template <class Vector>
typename types::PODTypes<typename Vector::value_type>::type
nrm2(const Vector &x, int offset = 0, int size = -1);

template <AMGX_VecPrecision t_vecPrec, AMGX_MatPrecision t_matPrec, AMGX_IndPrecision t_indPrec>
typename types::PODTypes<typename Vector<TemplateConfig<AMGX_host, t_vecPrec, t_matPrec, t_indPrec> >::value_type>::type
nrm2(const Vector<TemplateConfig<AMGX_host, t_vecPrec, t_matPrec, t_indPrec> > &x, int offset, int size);

template <class Vector>
typename types::PODTypes<typename Vector::value_type>::type
nrmmax(const Vector &x, int offset = 0, int size = -1);

} // namespace amgx
