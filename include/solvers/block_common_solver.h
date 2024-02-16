// SPDX-FileCopyrightText: 2011 - 2024 NVIDIA CORPORATION. All Rights Reserved.
//
// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#include "amgx_types/util.h"
#include "amgx_types/math.h"


namespace amgx
{

#define double_epsilon  1e-12   //* AMGX_NUMERICAL_DZERO
#define float_epsilon   1e-7    //* AMGX_NUMERICAL_SZERO
#define s_A(ROW,COL)   s_Amat[s_offset+ROW*bsize+COL]
#define s_At(ROW,COL)   s_Amat[s_offset+COL*bsize+ROW]
#define s_A_rval(ROW,COL)   types::util<ValueType>::volcast(s_Amat[s_offset+ROW*bsize+COL])
#define s_A_lval(val, ROW,COL)   types::util<ValueType>::volcast(val, s_Amat + s_offset+ROW*bsize+COL)


static inline __host__  __device__ bool isCloseToZero( double a)
{
    return  fabs(a) < double_epsilon ? true : false;
}

static inline __host__ __device__ bool isCloseToZero( float a)
{
    return  fabs(a) < float_epsilon ? true : false;
}

static inline __host__ __device__ bool isCloseToZero( const cuComplex &a)
{
    return fabs(types::get_re(a)) < float_epsilon && fabs(types::get_im(a)) < float_epsilon;
}

static inline __host__ __device__ bool isCloseToZero( const cuDoubleComplex &a)
{
    return fabs(types::get_re(a)) < double_epsilon && fabs(types::get_im(a)) < double_epsilon;
}

static inline __host__ __device__ bool isCloseToZero( volatile const cuComplex &a)
{
    return fabs(a.x) < float_epsilon && fabs(a.y) < float_epsilon;
}

static inline __host__ __device__ bool isCloseToZero( volatile const cuDoubleComplex &a)
{
    return fabs(a.x) < double_epsilon && fabs(a.y) < double_epsilon;
}

static inline __host__ __device__ bool isNotCloseToZero( const double &a)
{
    return !isCloseToZero(a);
}

static inline __host__ __device__ bool isNotCloseToZero( const float &a)
{
    return !isCloseToZero(a);
}

static inline __host__ __device__ bool isNotCloseToZero( const cuComplex &a)
{
    return !isCloseToZero(a);
}

static inline __host__ __device__ bool isNotCloseToZero( const cuDoubleComplex &a)
{
    return !isCloseToZero(a);
}

static inline __host__ __device__ bool isNotCloseToZero( volatile const cuComplex &a)
{
    return !isCloseToZero(a);
}

static inline __host__ __device__ bool isNotCloseToZero( volatile const cuDoubleComplex &a)
{
    return !isCloseToZero(a);
}

static inline __host__ __device__ float epsilon( float dummy)
{
    // return a small value with the same sign as the example value
    return copysign(float_epsilon, dummy);
}

static inline __host__ __device__ double epsilon( double dummy)
{
    // return a small value with the same sign as the example value
    return copysign(double_epsilon, dummy);
}

static inline __host__ __device__ cuComplex epsilon( cuComplex dummy)
{
    // return a small value with the same sign as the example value
    return make_cuComplex(copysign(float_epsilon, types::get_re(dummy)), copysign(float_epsilon, types::get_im(dummy)) );
}

static inline __host__ __device__ cuDoubleComplex epsilon( cuDoubleComplex dummy)
{
    // return a small value with the same sign as the example value
    return make_cuDoubleComplex( copysign(double_epsilon, types::get_re(dummy)), copysign(double_epsilon, types::get_im(dummy)) );
}

template<typename IndexType, typename ValueType, int blockrows_per_cta, int bsize, int bsize_sq>
__device__
void compute_block_inverse_row_major( volatile ValueType *s_Amat, const int s_offset, const int e_offset, const int i_ind, const int j_ind, ValueType *Einv )
{
//ROW MAJOR
    ValueType diag;
    ValueType tmp;

    for (int row = 0; row < bsize; row++)
    {
        diag = isNotCloseToZero( s_A_rval(row, row) ) ? types::util<ValueType>::get_one() / s_A_rval(row, row) : types::util<ValueType>::get_one() / epsilon(s_A_rval(row, row));

        if ((i_ind == 0) && !(j_ind == row))
        {
            tmp = s_A_rval(row, j_ind) * diag;
            s_A_lval(tmp, row, j_ind);
        }

        if ((i_ind != row) && !(j_ind == row))
        {
            tmp = types::util<ValueType>::invert(s_A_rval(i_ind, row) * s_A_rval(row, j_ind)) + s_A_rval(i_ind, j_ind);
            s_A_lval(tmp, i_ind, j_ind);
        }

        if (i_ind == 0)
        {
            tmp = (j_ind == row) ? diag : types::util<ValueType>::invert(s_A_rval(j_ind, row) * diag);
            s_A_lval(tmp, j_ind, row);
        }
    }

    Einv[e_offset] = s_A_rval(i_ind, j_ind);
}

template<typename IndexType, typename ValueType, int bsize>
__device__
void compute_block_inverse_row_major4x4_formula( volatile ValueType *s_Amat, const int s_offset, const int e_offset, const int i_ind, const int j_ind, ValueType *Einv )
{
    short I[3];
    short J[3];
    short icount = 0;
    short jcount = 0;

    if (j_ind != 0) { I[icount++] = 0; }

    if (j_ind != 1) { I[icount++] = 1; }

    if (j_ind != 2) { I[icount++] = 2; }

    if (j_ind != 3) { I[icount++] = 3; }

    if (i_ind != 0) { J[jcount++] = 0; }

    if (i_ind != 1) { J[jcount++] = 1; }

    if (i_ind != 2) { J[jcount++] = 2; }

    if (i_ind != 3) { J[jcount++] = 3; }

    // Each thread computes its co-factors
    ValueType cofactor = types::util<ValueType>::get_zero();
    cofactor = cofactor + s_A(I[0], J[0]) * s_A(I[1], J[1]) * s_A(I[2], J[2]);
    cofactor = cofactor + s_A(I[0], J[1]) * s_A(I[1], J[2]) * s_A(I[2], J[0]);
    cofactor = cofactor + s_A(I[0], J[2]) * s_A(I[1], J[0]) * s_A(I[2], J[1]);
    cofactor = cofactor - s_A(I[0], J[0]) * s_A(I[1], J[2]) * s_A(I[2], J[1]);
    cofactor = cofactor - s_A(I[0], J[1]) * s_A(I[1], J[0]) * s_A(I[2], J[2]);
    cofactor = cofactor - s_A(I[0], J[2]) * s_A(I[1], J[1]) * s_A(I[2], J[0]);

    if ((i_ind + j_ind) % 2) { cofactor = types::util<ValueType>::invert(cofactor); }

    ValueType my_A = s_A(j_ind, i_ind);
    // Each thread stores its result in shared memory to compute determinant
    s_A(i_ind, j_ind) = my_A * cofactor;
    // Each thread computes det
    ValueType det = s_A(0, 0) + s_A(0, 1) + s_A(0, 2) + s_A(0, 3);

    if (isNotCloseToZero(det) )
    {
        Einv[e_offset] = cofactor / det;
    }
    else
    {
        Einv[e_offset] = (i_ind == j_ind) ? ( isNotCloseToZero(my_A) ? ValueType(1) / my_A : ValueType(1) / epsilon(my_A) ) : ValueType(0.);
    }

    //Einv[e_offset] = cofactor/epsilon(det) ;
}

template<typename IndexType, typename ValueType, int bsize, bool store_result>
__device__
void compute_block_inverse_row_major4x4_formula2( volatile ValueType *s_Amat, const int s_offset, const int e_offset, const int i_ind, const int j_ind, ValueType *Einv )
{
    short I0, I1, I2;
    short J0, J1, J2;
    I0 = !j_ind;
    I1 = 1 + (j_ind < 2);
    I2 = 2 + (j_ind < 3);
    J0 = !i_ind;
    J1 = 1 + (i_ind < 2);
    J2 = 2 + (i_ind < 3);
    // Each thread computes its co-factors
    ValueType cofactor = 0.;
    cofactor += s_A(I0, J0) * s_A(I1, J1) * s_A(I2, J2);
    cofactor += s_A(I0, J1) * s_A(I1, J2) * s_A(I2, J0);
    cofactor += s_A(I0, J2) * s_A(I1, J0) * s_A(I2, J1);
    cofactor -= s_A(I0, J0) * s_A(I1, J2) * s_A(I2, J1);
    cofactor -= s_A(I0, J1) * s_A(I1, J0) * s_A(I2, J2);
    cofactor -= s_A(I0, J2) * s_A(I1, J1) * s_A(I2, J0);

    if ((i_ind + j_ind) % 2) { cofactor *= -1; }

    ValueType my_A = s_A(j_ind, i_ind);
    // Each thread stores its result in shared memory to compute determinant
    s_A(i_ind, j_ind) = my_A * cofactor;
    // Each thread computes det
    ValueType det = s_A(0, 0) + s_A(0, 1) + s_A(0, 2) + s_A(0, 3);

    if (store_result)
    {
        if (isNotCloseToZero(det) )
        {
            Einv[e_offset] = cofactor / det;
        }
        else
        {
            Einv[e_offset] = (i_ind == j_ind) ? ( isNotCloseToZero(my_A) ? ValueType(1) / my_A : ValueType(1) / epsilon(my_A)) : ValueType(0.);
        }

        //Einv[e_offset] = cofactor/epsilon(det);
    }
    else
    {
        if (isNotCloseToZero(det) )
        {
            s_A(i_ind, j_ind) = cofactor / det;
        }
        else
            //s_A(i_ind,j_ind) = cofactor/epsilon(det);
        {
            s_A(i_ind, j_ind) = (i_ind == j_ind) ? ( isNotCloseToZero(my_A) ? ValueType(1) / my_A : ValueType(1) / epsilon(my_A)) : ValueType(0.);
        }
    }
}

//COLUMN MAJOR
template<typename IndexType, typename ValueType, int blockrows_per_cta, int bsize, int bsize_sq>
__device__
void compute_block_inverse_col_major( volatile ValueType *s_Amat, const int s_offset, const int e_offset, const int i_ind, const int j_ind, ValueType *Einv )
{
    ValueType diag;

    for (int row = 0; row < bsize; row++)
    {
        diag = isNotCloseToZero(s_At(row, row)) ? ValueType(1) / s_At(row, row) : ValueType(1) / epsilon(s_At(row, row));

        if ((i_ind == 0) && !(j_ind == row))
        {
            s_At(row, j_ind) = s_At(row, j_ind) * diag;
        }

        if ((i_ind != row) && !(j_ind == row))
        {
            s_At(i_ind, j_ind) = -(s_At(i_ind, row) * s_At(row, j_ind)) + s_At(i_ind, j_ind);
        }

        if (i_ind == 0)
        {
            s_At(j_ind, row) = (j_ind == row) ? diag : -(s_At(j_ind, row) * diag);
        }
    }

    Einv[e_offset] = s_At(j_ind, i_ind);
}


template<typename IndexType, typename ValueType, int bsize, bool store_result>
__device__
void compute_block_inverse_col_major4x4_formula2( volatile ValueType *s_Amat, const int s_offset, const int e_offset, const int i_ind, const int j_ind, ValueType *Einv )
{
    short I0, I1, I2;
    short J0, J1, J2;
    I0 = !j_ind;
    I1 = 1 + (j_ind < 2);
    I2 = 2 + (j_ind < 3);
    J0 = !i_ind;
    J1 = 1 + (i_ind < 2);
    J2 = 2 + (i_ind < 3);
    // Each thread computes its co-factors
    ValueType cofactor = 0.;
    cofactor += s_A(I0, J0) * s_A(I1, J1) * s_A(I2, J2);
    cofactor += s_A(I0, J1) * s_A(I1, J2) * s_A(I2, J0);
    cofactor += s_A(I0, J2) * s_A(I1, J0) * s_A(I2, J1);
    cofactor -= s_A(I0, J0) * s_A(I1, J2) * s_A(I2, J1);
    cofactor -= s_A(I0, J1) * s_A(I1, J0) * s_A(I2, J2);
    cofactor -= s_A(I0, J2) * s_A(I1, J1) * s_A(I2, J0);

    if ((i_ind + j_ind) % 2) { cofactor *= -1; }

    ValueType my_A = s_A(j_ind, i_ind);
    // Each thread stores its result in shared memory to compute determinant
    s_A(i_ind, j_ind) = my_A * cofactor;
    // Each thread computes det
    ValueType det = s_A(0, 0) + s_A(0, 1) + s_A(0, 2) + s_A(0, 3);

    if (store_result)
    {
        if (isNotCloseToZero(det) )
        {
            Einv[e_offset] = cofactor / det;
        }
        else
            //Einv[e_offset] =  cofactor/epsilon(det);
        {
            Einv[e_offset] = (i_ind == j_ind) ? ( isNotCloseToZero(my_A) ? ValueType(1) / my_A : ValueType(1) / epsilon(my_A)) : ValueType(0.);
        }
    }
    else
    {
        if (isNotCloseToZero(det) )
        {
            s_A(i_ind, j_ind) = cofactor / det;
        }
        else
            //s_A(i_ind,j_ind) =  cofactor/epsilon(det);
        {
            s_A(i_ind, j_ind) = (i_ind == j_ind) ? ( isNotCloseToZero(my_A) ? ValueType(1) / my_A : ValueType(1) / epsilon(my_A)) : ValueType(0.);
        }
    }
}




template<typename IndexType, typename ValueType, int blockrows_per_cta>
__device__
void compute_block_inverse2( volatile ValueType *s_Amat, const int s_offset, const int e_offset, const int i_ind, const int j_ind, ValueType *Einv, int tile_num, const int bsize, const int bsize_sq )
{
#define s_A2_lval(val, ROW,COL)   types::util<ValueType>::volcast(val, s_Amat + s_offset+(ROW)*bsize+COL);
#define s_A2_rval(ROW,COL)   types::util<ValueType>::volcast(s_Amat[s_offset+(ROW)*bsize+(COL)])
ValueType diag;
ValueType tmp;

    for (int row = 0; row < bsize; row++)
    {
        diag = isNotCloseToZero(s_A(row, row)) ? types::util<ValueType>::get_one() / s_A2_rval(row, row) : types::util<ValueType>::get_one() / epsilon(s_A2_rval(row, row));

        for (int t2 = 0; t2 < tile_num; t2++)
            if ((i_ind == 0) && !(j_ind + t2 * 4 == row) && ((t2 * 4 + j_ind) < bsize))
            {
                tmp = s_A2_rval(row, j_ind + t2 * 4) * diag;
                s_A2_lval(tmp, row, j_ind + t2 * 4);
            }

        for (int t1 = 0; t1 < tile_num; t1++)
            for (int t2 = 0; t2 < tile_num; t2++)
                if ((i_ind + t1 * 4 != row) && !(j_ind + t2 * 4 == row) && ((t1 * 4 + i_ind) < bsize) && ((t2 * 4 + j_ind) < bsize))
                {
                    tmp = types::util<ValueType>::invert((s_A2_rval(i_ind + t1 * 4, row) * s_A2_rval(row, j_ind + t2 * 4)) + s_A2_rval(i_ind + t1 * 4, j_ind + t2 * 4));
                    s_A2_lval(tmp, i_ind + t1 * 4, j_ind + t2 * 4);
                }

        for (int t2 = 0; t2 < tile_num; t2++)
            if (i_ind == 0 && (t2 * 4 + j_ind) < bsize)
            {
                tmp = ((j_ind + t2 * 4) == row) ? diag : types::util<ValueType>::invert(s_A2_rval(j_ind + t2 * 4, row) * diag);
                s_A2_lval(tmp, j_ind + t2 * 4, row)
            }
    }

    for (int t1 = 0; t1 < tile_num; t1++)
        for (int t2 = 0; t2 < tile_num; t2++)
            if ((t1 * 4 + i_ind) < bsize && (t2 * 4 + j_ind) < bsize)
            {
                Einv[e_offset + t1 * 4 * bsize + t2 * 4] = s_A2_rval(i_ind + t1 * 4, j_ind + t2 * 4);
            }
}

} // namespace amgx
