// SPDX-FileCopyrightText: 2011 - 2025 NVIDIA CORPORATION. All Rights Reserved.
//
// SPDX-License-Identifier: BSD-3-Clause

#include <gaussian_elimination.h>
#include <basic_types.h>
#include <texture.h>
#include <util.h>

#ifdef _WIN32
#pragma warning (push)
#pragma warning (disable : 4244 4267 4521)
#endif
#ifdef _WIN32
#pragma warning (pop)
#endif

namespace amgx
{

// Kernel to invert 4by4 matrix on device using one thread
// TODO: possible optimizations using multiple threads, possible generalisation for NxN blocks
template <typename IndexType, typename ValueTypeA, typename ValueTypeB>
__global__
void gaussianElimination4by4Kernel(const IndexType *dia_indices, const ValueTypeA *values, ValueTypeB *x, const ValueTypeB *b)
{
    const int tid = blockDim.x * blockIdx.x + threadIdx.x;
    ValueTypeB Atemp[BSIZE][BSIZE];
    ValueTypeB xtemp[BSIZE];
    ValueTypeB btemp[BSIZE];
    int offset;

    if (tid == 0)
    {
#pragma unroll

        for (int m = 0; m < BSIZE; m++)
        {
            btemp[m]  = b[m];
        }

        offset = 0;
#pragma unroll

        for (int m = 0; m < BSIZE; m++)
#pragma unroll
            for (int n = 0; n < BSIZE; n++)
            {
                types::util<ValueTypeA>::to_uptype(values[offset++], Atemp[m][n]);    // only one nonzero block - diagonal
            }

        gaussianElimination4by4(Atemp, xtemp, btemp);
        // Store x
#pragma unroll

        for (int m = 0; m < BSIZE; m++)
        {
            x[m] = xtemp[m];
        }
    }
}

// Method to perform gaussian elimination for a matrix stored in row major format
template<typename IndexType, typename ValueTypeA, typename ValueTypeB>
void gaussianEliminationRowMajor(ValueTypeA **e, ValueTypeB *x, ValueTypeB *b, const IndexType bsize)
{
    if (bsize > 4) { FatalError("Warning, Gaussian Elimination code doesn't pivot and bsize > 4", AMGX_ERR_BAD_PARAMETERS); }

    ValueTypeB pivot, ratio, tmp;

    for (int j = 0; j < bsize; j++)
    {
        types::util<ValueTypeA>::to_uptype(e[j][j], pivot);

        for (int k = j + 1; k < bsize; k++)
        {
            ValueTypeB temp;
            types::util<ValueTypeA>::to_uptype(e[k][j], temp);
            ratio = temp / pivot;

            for (int m = j + 1; m < bsize; m++)
            {
                e[k][m] = e[k][m] - e[j][m] * ratio;
            }

            b[k] = b[k] - b[j] * ratio;
        }
    }

    // back substitution
    for (int j = bsize - 1; j >= 0; j--)
    {
        tmp = types::util<ValueTypeB>::get_zero();

        for (int k = j + 1; k < bsize; k++)
        {
            tmp = tmp + e[j][k] * x[k];
        }

        x[j] = (b[j] - tmp) / e[j][j];
    }
}

// Method to perform gaussian elimination on matrix stored in block_dia_csr_matrix_h
template <AMGX_VecPrecision t_vecPrec, AMGX_MatPrecision t_matPrec, AMGX_IndPrecision t_indPrec>
void GaussianElimination<TemplateConfig<AMGX_host, t_vecPrec, t_matPrec, t_indPrec> >::gaussianElimination_4x4_host(const Matrix_h &A, Vector_h &x, const Vector_h &b)
{
    typedef typename Matrix_h::value_type ValueTypeA;
    typedef typename Vector_h::value_type ValueTypeB;

    if (A.get_num_rows() != 1)
    {
        FatalError("Haven't implemented gaussian elimination solver for num_blocks != 1", AMGX_ERR_BAD_PARAMETERS);
    }
    else
    {
        int bsize = A.get_block_dimy();
        // Allocate space for block_matrix
        ValueTypeA **Atemp = new ValueTypeA* [bsize];

        for ( int i = 0; i < bsize; i++)
        {
            Atemp[i] = new ValueTypeA[bsize];
        }

        ValueTypeB *btemp = new ValueTypeB[bsize];
        ValueTypeB *xtemp = new ValueTypeB[bsize];

        // Copy matrix and rhs
        for (int m = 0; m < bsize; m++)
        {
            for (int n = 0; n < bsize; n++)
            {
                Atemp[m][n] = A.values[bsize * m + n]; //diag[bsize*m + n]; since we have 1 nonzero only
            }

            btemp[m] = b[m];
        }

        ValueTypeB *x_ptr = x.raw();
        gaussianEliminationRowMajor(Atemp, x_ptr, btemp, bsize);
    }
}

// Method to perform gaussian elimination on matrix stored in block_dia_csr_matrix_d
template <AMGX_VecPrecision t_vecPrec, AMGX_MatPrecision t_matPrec, AMGX_IndPrecision t_indPrec>
void GaussianElimination<TemplateConfig<AMGX_device, t_vecPrec, t_matPrec, t_indPrec> >::gaussianElimination_4x4_device(const Matrix_d &A, Vector_d &x, const Vector_d &b)
{
    typedef typename Matrix_d::value_type ValueTypeA;
    typedef typename Vector_d::value_type ValueTypeB;
    typedef typename Matrix_d::index_type IndexType;

    if (A.get_num_rows() != 1)
    {
        FatalError("Haven't implemented gaussian elimination solver for num_blocks != 1 or block_size != 4", AMGX_ERR_BAD_PARAMETERS);
    }
    else
    {
        const IndexType *dia_ptr = A.diag.raw();
        const ValueTypeA *values_ptr = A.values.raw();
        const ValueTypeB *b_ptr = b.raw();
        ValueTypeB *x_ptr = x.raw();
        gaussianElimination4by4Kernel <<< 1, 1>>>(dia_ptr, values_ptr, x_ptr, b_ptr);
        cudaCheckError();
    }
}

template <class TConfig>
void GaussianElimination<TConfig>::gaussianElimination(const Matrix<TConfig> &A, Vector<TConfig> &x, const Vector<TConfig> &b)
{
    FatalError("Matrix format not supported in gaussian_elimination", AMGX_ERR_NOT_IMPLEMENTED);
}

template <AMGX_VecPrecision t_vecPrec, AMGX_MatPrecision t_matPrec, AMGX_IndPrecision t_indPrec>
void GaussianElimination<TemplateConfig<AMGX_host, t_vecPrec, t_matPrec, t_indPrec> >::gaussianElimination(const Matrix_h &A, Vector_h &x, const Vector_h &b)
{
    if (A.get_block_dimx() == 4 && A.get_block_dimy() == 4)
    {
        gaussianElimination_4x4_host(A, x, b);
    }
    else
    {
        FatalError("gaussElimination: Blocksize is unsupported", AMGX_ERR_NOT_SUPPORTED_BLOCKSIZE);
    }
}

template <AMGX_VecPrecision t_vecPrec, AMGX_MatPrecision t_matPrec, AMGX_IndPrecision t_indPrec>
void GaussianElimination<TemplateConfig<AMGX_device, t_vecPrec, t_matPrec, t_indPrec> >::gaussianElimination(const Matrix_d &A, Vector_d &x, const Vector_d &b)
{
    if (A.get_block_dimx() == 4 && A.get_block_dimy() == 4)
    {
        gaussianElimination_4x4_device(A, x, b);
    }
    else
    {
        FatalError("gaussElimination: Blocksize is unsupported", AMGX_ERR_NOT_SUPPORTED_BLOCKSIZE);
    }
}
// -------------------------------
//  Explict instantiations
// -------------------------------
#define AMGX_CASE_LINE(CASE) template class GaussianElimination<TemplateMode<CASE>::Type>;
AMGX_FORALL_BUILDS(AMGX_CASE_LINE)
AMGX_FORCOMPLEX_BUILDS(AMGX_CASE_LINE)
#undef AMGX_CASE_LINE

} // namespace amgx
