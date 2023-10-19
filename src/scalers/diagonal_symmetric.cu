/* Copyright (c) 2013-2017, NVIDIA CORPORATION. All rights reserved.
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

#include <scalers/diagonal_symmetric.h>
#include <sm_utils.inl>
#include <thrust/inner_product.h>

namespace amgx
{

/**********************************************************************
 * DEVICE FUNCTIONS
 *********************************************************************/
template <typename IndexType, typename MatrixType, typename VectorType>
__global__
void ScaleMatrix(int rows, IndexType *offsets, IndexType *indices, MatrixType *values,
                 VectorType *scale, ScaleDirection scaleOrUnscale)
{
    if ( scaleOrUnscale == SCALE)
        for (int i = threadIdx.x + blockIdx.x * blockDim.x; i < rows; i += gridDim.x * blockDim.x)
        {
            VectorType si = scale[i];

            for (int jj = offsets[i]; jj < offsets[i + 1]; jj++)
            {
                IndexType j = indices[jj];
                VectorType sj = scale[j];
                // scale in-place
                values[jj] *= si * sj;
            }
        }

    if ( scaleOrUnscale == UNSCALE)
        for (int i = threadIdx.x + blockIdx.x * blockDim.x; i < rows; i += gridDim.x * blockDim.x)
        {
            VectorType si = scale[i];

            for (int jj = offsets[i]; jj < offsets[i + 1]; jj++)
            {
                IndexType j = indices[jj];
                VectorType sj = scale[j];
                // scale in-place
                values[jj] /= si * sj;
            }
        }
}

template <typename VectorType>
__global__
void scaleVectorKernel(int rows, VectorType *values, VectorType *diag, ScaleDirection scaleOrUnscale, ScaleSide leftOrRight)
{
    //Ignore the scaleside, it is the same for both in the diagonal symmetric case
    if (scaleOrUnscale == SCALE)
        for (int i = threadIdx.x + blockIdx.x * blockDim.x; i < rows; i += gridDim.x * blockDim.x)
        {
            values[i] *= diag[i];
        }

    if (scaleOrUnscale == UNSCALE)
        for (int i = threadIdx.x + blockIdx.x * blockDim.x; i < rows; i += gridDim.x * blockDim.x)
        {
            values[i] /= diag[i];
        }
}

template <typename T>
__global__
void checkPositiveVector(int N, T *v, bool *positive)
{
    for (int i = threadIdx.x + blockIdx.x * blockDim.x; i < N; i += gridDim.x * blockDim.x)
    {
        if (v[i] < T(0)) { positive[0] = false; }
    }
}

template <typename T>
__global__
void sqrtVector(int N, T *v)
{
    for (int i = threadIdx.x + blockIdx.x * blockDim.x; i < N; i += gridDim.x * blockDim.x)
    {
        // we know by this point that the diagonal is +ve
        v[i] = 1. / sqrt(v[i]);
    }
}

template <typename IndexType, typename MatrixType, typename VectorType>
__global__
void grabDiagonalVector(int rows, IndexType *offsets, IndexType *indices, MatrixType *values, VectorType *diag)
{
    for (int i = threadIdx.x + blockIdx.x * blockDim.x; i < rows; i += gridDim.x * blockDim.x)
    {
        for (int jj = offsets[i]; jj < offsets[i + 1]; jj++)
        {
            IndexType j = indices[jj];

            if (i == j) { diag[i] = values[jj]; }
        }
    }
}

// Setup scaling vector
// Device version
template <AMGX_VecPrecision t_vecPrec, AMGX_MatPrecision t_matPrec, AMGX_IndPrecision t_indPrec>
void DiagonalSymmetricScaler<TemplateConfig<AMGX_device, t_vecPrec, t_matPrec, t_indPrec> >::setup(Matrix_d &A)
{
    if (A.is_matrix_distributed())
    {
        FatalError("Diagonal Symmetric scaling not supported for distributed matrices", AMGX_ERR_NOT_IMPLEMENTED);
    }

    // grab diagonal
    diag = VVector(A.get_num_rows());
    grabDiagonalVector <<< 4096, 128>>>(A.get_num_rows(), A.row_offsets.raw(), A.col_indices.raw(), A.values.raw(), diag.raw());
    // check diagonal +ve
    bool positive = true, *d_positive;
    amgx::memory::cudaMallocAsync((void**)&d_positive, sizeof(bool));
    cudaMemcpy(d_positive, &positive, sizeof(bool), cudaMemcpyHostToDevice);
    checkPositiveVector <<< 4096, 256>>>(diag.size(), diag.raw(), d_positive);
    cudaMemcpy(&positive, d_positive, sizeof(bool), cudaMemcpyDeviceToHost);

    // fail out if necessary
    if (!positive)
    {
        FatalError("Diagonal symmetric scaling only applies to SPD systems with positive diagonal entries", AMGX_ERR_NOT_IMPLEMENTED);
    }

    // set scale = 1./sqrt(diag) , reciprocal square root
    sqrtVector <<< 4096, 256>>>(diag.size(), diag.raw());
}

//Host version
template <AMGX_VecPrecision t_vecPrec, AMGX_MatPrecision t_matPrec, AMGX_IndPrecision t_indPrec>
void DiagonalSymmetricScaler<TemplateConfig<AMGX_host, t_vecPrec, t_matPrec, t_indPrec> >::setup(Matrix_h &A)
{
    if (A.is_matrix_distributed())
    {
        FatalError("Diagonal Symmetric scaling not supported for distributed matrices", AMGX_ERR_NOT_IMPLEMENTED);
    }

    // grab diagonal
    diag = VVector(A.get_num_rows());
    //grabDiagonalVector<<<4096, 128>>>(A.get_num_rows(), A.row_offsets.raw(), A.col_indices.raw(), A.values.raw(), diag.raw());
    IndexType *offsets =  A.row_offsets.raw();
    IndexType *indices = A.col_indices.raw();
    ValueTypeA *values = A.values.raw();

    for (int i = 0; i < A.get_num_rows(); i++)
    {
        for (int jj = offsets[i]; jj < offsets[i + 1]; jj++)
        {
            IndexType j = indices[jj];

            if (i == j) { diag[i] = values[jj]; }
        }
    }

    // check diagonal +ve
    bool positive = true;

    //checkPositiveVector<<<4096, 256>>>(diag.size(), diag.raw(), d_positive);
    for (int i = 0; i < diag.size(); i++)
    {
        if (diag[i] < ValueTypeB(0)) { positive = false; }
    }

    // fail out if necessary
    if (!positive)
    {
        FatalError("Diagonal symmetric scaling only applies to SPD systems with positive diagonal entries", AMGX_ERR_NOT_IMPLEMENTED);
    }

    // set scale = 1./sqrt(diag) , reciprocal square root
    // sqrtVector<<<4096, 256>>>(diag.size(), diag.raw());
    for (int i = 0; i < diag.size(); i++)
    {
        // we know by this point that the diagonal is +ve
        diag[i] = 1. / sqrt(diag[i]);
    }
}

// 1x1 Device
template <AMGX_VecPrecision t_vecPrec, AMGX_MatPrecision t_matPrec, AMGX_IndPrecision t_indPrec>
void DiagonalSymmetricScaler<TemplateConfig<AMGX_device, t_vecPrec, t_matPrec, t_indPrec> >::scaleMatrix(Matrix_d &A, ScaleDirection scaleOrUnscale)
{
    // A_ij *= scale[i] * scale[j];
    ScaleMatrix <<< 4096, 256>>>(A.get_num_rows(), A.row_offsets.raw(), A.col_indices.raw(),
                                 A.values.raw(), diag.raw(), scaleOrUnscale);
}

// 1x1 Host
template <AMGX_VecPrecision t_vecPrec, AMGX_MatPrecision t_matPrec, AMGX_IndPrecision t_indPrec>
void DiagonalSymmetricScaler<TemplateConfig<AMGX_host, t_vecPrec, t_matPrec, t_indPrec> >::scaleMatrix(Matrix_h &A, ScaleDirection scaleOrUnscale)
{
    IndexType *offsets =  A.row_offsets.raw();
    IndexType *indices = A.col_indices.raw();
    ValueTypeA *values = A.values.raw();

    for (int i = 0; i < A.get_num_rows(); i++)
    {
        if ( scaleOrUnscale == SCALE)
            for (int jj = offsets[i]; jj < offsets[i + 1]; jj++)
            {
                IndexType j = indices[jj];
                values[jj] *= diag[i] * diag[j];
            }

        if ( scaleOrUnscale == UNSCALE)
            for (int jj = offsets[i]; jj < offsets[i + 1]; jj++)
            {
                IndexType j = indices[jj];
                values[jj] /= diag[i] * diag[j];
            }
    }
}


template <AMGX_VecPrecision t_vecPrec, AMGX_MatPrecision t_matPrec, AMGX_IndPrecision t_indPrec>
void DiagonalSymmetricScaler<TemplateConfig<AMGX_device, t_vecPrec, t_matPrec, t_indPrec> >::scaleVector(VVector &v, ScaleDirection scaleOrUnscale, ScaleSide leftOrRight)
{
    scaleVectorKernel <<< 4096, 256>>>(v.size(), v.raw(), diag.raw(), scaleOrUnscale, leftOrRight);
}

// 4x4 Host
template <AMGX_VecPrecision t_vecPrec, AMGX_MatPrecision t_matPrec, AMGX_IndPrecision t_indPrec>
void DiagonalSymmetricScaler<TemplateConfig<AMGX_host, t_vecPrec, t_matPrec, t_indPrec> >::scaleVector(VVector &v, ScaleDirection scaleOrUnscale, ScaleSide leftOrRight)
{
    //ignore leftOrRIght for diagonal scale, they are the same
    if ( scaleOrUnscale == SCALE)
        for (int i = 0; i < diag.size(); i++)
        {
            v[i] *= diag[i];
        }

    if ( scaleOrUnscale == UNSCALE)
        for (int i = 0; i < diag.size(); i++)
        {
            v[i] /= diag[i];
        }
}


/****************************************
 * Explict instantiations
 ***************************************/
#define AMGX_CASE_LINE(CASE) template class DiagonalSymmetricScaler_Base<TemplateMode<CASE>::Type>;
AMGX_FORALL_BUILDS(AMGX_CASE_LINE)
#undef AMGX_CASE_LINE

#define AMGX_CASE_LINE(CASE) template class DiagonalSymmetricScaler<TemplateMode<CASE>::Type>;
AMGX_FORALL_BUILDS(AMGX_CASE_LINE)
#undef AMGX_CASE_LINE

} // namespace amgx

