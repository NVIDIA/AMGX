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

#include <scalers/binormalization.h>
#include <sm_utils.inl>
#include <thrust/inner_product.h>
#include <solvers/block_common_solver.h>
#include <thrust_wrapper.h>

namespace amgx
{

template<class TConfig> class BinormalizationScaler;

/**********************************************************************
 * HOST FUNCTIONS
 *********************************************************************/
template <typename IndexType, typename MatrixType, typename VectorType>
void computeBetaGammaHost(int rows, IndexType *offsets, IndexType *indices, MatrixType *vals,
                          VectorType *x, VectorType *y, VectorType *beta, VectorType *gamma)
{
    for (int i = 0; i < rows; i++) { gamma[i] = 0.; }

    for (int i = 0; i < rows; i++)
    {
        VectorType bi = 0.;

        for (int jj = offsets[i]; jj < offsets[i + 1]; jj++)
        {
            int col = indices[jj];
            VectorType val = vals[jj];
            bi += (val * val) * y[col];
            gamma[col] += (val * val) * x[i];
        }

        beta[i] = bi;
    }
}

// compute Gamma on its own
template <typename IndexType, typename MatrixType, typename VectorType>
void computeGammaHost(int rows, IndexType *offsets, IndexType *indices, MatrixType *vals,
                      VectorType *x, VectorType *gamma)
{
    for (int i = 0; i < rows; i++) { gamma[i] = 0.; }

    for (int i = 0; i < rows; i++)
    {
        for (int jj = offsets[i]; jj < offsets[i + 1]; jj++)
        {
            int col = indices[jj];
            VectorType val = vals[jj];
            gamma[col] += (val * val) * x[i];
        }
    }
}

// compute Beta on its own
template <typename IndexType, typename MatrixType, typename VectorType>
void computeBetaHost(int nrows, IndexType *offsets, IndexType *indices, MatrixType *vals,
                     VectorType *y, VectorType *beta)
{
    for (int i = 0; i < nrows; i++)
    {
        VectorType bi = 0.;

        for (int jj = offsets[i]; jj < offsets[i + 1]; jj++)
        {
            int col = indices[jj];
            VectorType val = vals[jj];
            bi += (val * val) * y[col];
        }

        beta[i] = bi;
    }
}

template <typename IndexType, typename MatrixType, typename VectorType>
void scaleMatrixHost(int nrows, IndexType *offsets, IndexType *indices, MatrixType *values,
                     VectorType *x, VectorType *y)
{
    for (int i = 0; i < nrows; i++)
    {
        VectorType fi = sqrt(fabs(x[i]));

        for (int jj = offsets[i]; jj < offsets[i + 1]; jj++)
        {
            int j = indices[jj];
            VectorType gj = sqrt(fabs(y[j]));
            values[jj] *= fi * gj;
        }
    }
}

/**********************************************************************
 * DEVICE FUNCTIONS
 *********************************************************************/

// compute initial beta, which is B*[1,...,1]'
template <typename IndexType, typename MatrixValue, typename VectorValue>
__global__
void computeBetaIniDevice(int nrows, IndexType *offsets, IndexType *indices, MatrixValue *values, VectorValue *beta)
{
    for (int i = threadIdx.x + blockIdx.x * blockDim.x; i < nrows; i += gridDim.x * blockDim.x)
    {
        VectorValue rowsum = 0.0;

        for (int jj = offsets[i]; jj < offsets[i + 1]; jj++)
        {
            rowsum += values[jj] * values[jj];
        }

        beta[i] = rowsum;
    }
}

template <typename IndexType, typename MatrixType, typename VectorType>
__global__
void grabDiagonalVector(int nrows, IndexType *offsets, IndexType *indices, MatrixType *values, VectorType *diag)
{
    for (int i = threadIdx.x + blockIdx.x * blockDim.x; i < nrows; i += gridDim.x * blockDim.x)
    {
        for (int jj = offsets[i]; jj < offsets[i + 1]; jj++)
        {
            IndexType j = indices[jj];

            if (i == j) { diag[i] = values[jj]; }
        }
    }
}

// functor to generate stddev of vectors
template <typename T>
struct std_f
{
    std_f(T x) : v(x) {};
    T v;

    __host__ __device__
    T operator()(const T &x1, const T &x2) const
    {
        return (x1 * x2 - v) * (x1 * x2 - v);
    }
};

// scaled the matrix using diag(F)*A*diag(G), f = sqrt(fabs(x)), g = sqrt(fabs(y))
template <typename IndexType, typename MatrixType, typename VectorType>
__global__
void scaleMatrixDevice(int rows, IndexType *offsets, IndexType *indices, MatrixType *values,
                       VectorType *x)
{
    for (int i = threadIdx.x + blockIdx.x * blockDim.x; i < rows; i += gridDim.x * blockDim.x)
    {
        VectorType fi = fabs(x[i]);

        for (int jj = offsets[i]; jj < offsets[i + 1]; jj++)
        {
            int j = indices[jj];
            VectorType fj = fabs(x[j]);
            values[jj] *= sqrt(fabs(fi * fj));
        }
    }
}


template <typename IndexType, typename ValueTypeA, typename ValueTypeB>
__global__
void getColRowNorms(int rows, IndexType *offsets, IndexType *indices, ValueTypeA *values,
                    ValueTypeB *rownorms, ValueTypeB *colnorms)
{
    for (int i = threadIdx.x + blockIdx.x * blockDim.x; i < rows; i += gridDim.x * blockDim.x)
    {
        ValueTypeB rownorm = 0.;

        for (int jj = offsets[i]; jj < offsets[i + 1]; jj++)
        {
            int j = indices[jj];
            ValueTypeB curval = values[jj] * values[jj];
            rownorm += curval;
            utils::atomic_add(colnorms + j, curval);
        }

        rownorms[i] = rownorm;
    }
}

// these warp reductions should be able to be replaced with amgx:: functions
template <int warpSize, typename T>
__device__ __inline__ T warpReduceSum(T val)
{
    if (warpSize > 16) { val += utils::shfl_down(val, 16, warpSize); }

    if (warpSize > 8) { val += utils::shfl_down(val, 8, warpSize); }

    if (warpSize > 4) { val += utils::shfl_down(val, 4, warpSize); }

    if (warpSize > 2) { val += utils::shfl_down(val, 2, warpSize); }

    if (warpSize > 1) { val += utils::shfl_down(val, 1, warpSize); }

    return val;
}

template <int warpSize, typename T>
__device__ T warpReduceSumShared(volatile T *vals, const int lane_id)
{
    if (warpSize > 16) { vals[lane_id] += vals[lane_id + 16]; }

    if (warpSize > 8) { vals[lane_id] += vals[lane_id + 8]; }

    if (warpSize > 4) { vals[lane_id] += vals[lane_id + 4]; }

    if (warpSize > 2) { vals[lane_id] += vals[lane_id + 2]; }

    if (warpSize > 1) { vals[lane_id] += vals[lane_id + 1]; }

    return vals[lane_id];
}

// compute gamma = B^T*x (B = A.^2)
template <typename IndexType, typename ValueTypeA, typename ValueTypeB>
__global__
void computeBetaDevice(const int nrows, IndexType *offsets, IndexType *indices, ValueTypeA *values,
                       ValueTypeB *diag, ValueTypeB *x, ValueTypeB *xn, ValueTypeB *beta, const ValueTypeB avg, ValueTypeB *avg_vec)
{
    for (int i = threadIdx.x + blockIdx.x * blockDim.x; i < nrows; i += gridDim.x * blockDim.x)
    {
        ValueTypeB xi = x[i];
        ValueTypeB bi = beta[i];
        ValueTypeB di = diag[i];
        ValueTypeB c0 = -di * xi * xi + 2 * bi * xi - nrows * avg;
        ValueTypeB c1 = (nrows - 2) * (bi - di * xi);
        ValueTypeB c2 = (nrows - 1) * di;
        assert(c0 > epsilon(c0)); //
        // delta = xi - x(i)
        ValueTypeB dx = (2 * c0) / (-c1 - sqrt(c1 * c1 - 4 * c2 * c0)) - x[i];
        ValueTypeB davg = 0.;
        ValueTypeB dbeta = 0.;

        for (int jj = offsets[i]; jj < offsets[i + 1]; jj++)
        {
            int j = indices[jj];
            ValueTypeA Bij = values[jj] * values[jj];
            davg += Bij * x[j]; // += x' * B(:, i) == B(i, :)*x, because B is symmetric
            dbeta += dx * Bij;
        }

        beta[i] = bi + dbeta;
        avg_vec[i] = dx * (davg + bi + di * dx) / nrows;
        //utils::atomic_add(avg, davg);
    }
}


template<typename T>
struct square_value : public unary_function<T, T>
{
    __host__ __device__ T operator()(const T &x) const
    {
        return x * x;
    }
};

// vector constant scale operand
template <typename T>
struct vmul_scale_const
{
    T _alpha;

    vmul_scale_const(T alpha): _alpha(alpha) {};

    __host__ __device__
    T operator()(const T &vec) const
    {
        return vec * _alpha;
    }
};

// vector scale operand
template <typename T>
struct vmul_scale
{

    vmul_scale() {};

    __host__ __device__
    T operator()(const T &vec, const T &alpha) const
    {
        return (vec * sqrt(fabs(alpha)));
    }
};

// vector unscale operand
template <typename T>
struct vmul_unscale
{

    vmul_unscale() {};

    __host__ __device__
    T operator()(const T &vec, const T &alpha) const
    {
        return (vec / sqrt(fabs(alpha)));
    }
};


// Setup on  Device
template <AMGX_VecPrecision t_vecPrec, AMGX_MatPrecision t_matPrec, AMGX_IndPrecision t_indPrec>
void BinormalizationScaler<TemplateConfig<AMGX_device, t_vecPrec, t_matPrec, t_indPrec> >::setup(Matrix_d &A)
{
    if (A.is_matrix_distributed())
    {
        FatalError("Binormalization scaling not supported for distributed matrices", AMGX_ERR_NOT_IMPLEMENTED);
    }

    // move these out to config parameters?
    const int max_iters = 10;
    const ValueTypeB tolerance = 1e-10;
    VVector diag(A.get_num_rows());
    grabDiagonalVector <<< 4096, 128>>>(A.get_num_rows(), A.row_offsets.raw(), A.col_indices.raw(), A.values.raw(), diag.raw());
    int nrows = A.get_num_rows();
    // temporary vectors
    VVector x(nrows, 1), xn(nrows), davg(nrows), beta(nrows, 0);
    computeBetaIniDevice <<< 4096, 256>>>(nrows, A.row_offsets.raw(), A.col_indices.raw(), A.values.raw(), beta.raw());
    cudaCheckError();
    ValueTypeB avg = thrust_wrapper::reduce(beta.begin(), beta.end()) / nrows;
    // calculate initial std1 and std2
    amgx::thrust::device_ptr<ValueTypeB> x_ptr(x.raw()), beta_ptr(beta.raw());
    ValueTypeB stdx = sqrt(amgx::thrust::inner_product(x_ptr, x_ptr + nrows, beta_ptr, ValueTypeB(0.), amgx::thrust::plus<ValueTypeB>(), std_f<ValueTypeB>(avg)) / nrows) / avg;

    for (int t = 0; t < max_iters; t++)
    {
        if (fabs(stdx) < tolerance) { break; } // finished

        computeBetaDevice <<< 4096, 256>>>(nrows, A.row_offsets.raw(), A.col_indices.raw(), A.values.raw(),
                                           diag.raw(), x.raw(), xn.raw(), beta.raw(), avg, davg.raw());
        avg += thrust_wrapper::reduce(davg.begin(), davg.end());
        // ValueTypeB stdx_old = stdx;
        stdx = sqrt(amgx::thrust::inner_product(x_ptr, x_ptr + nrows, beta_ptr, ValueTypeB(0.), amgx::thrust::plus<ValueTypeB>(), std_f<ValueTypeB>(avg)) / nrows) / avg;
        // print it #, current error, convergence rate
        // printf("ITER: %d     %.3e  %.3e   %.4lg\n",t, stdx, stdx_old, stdx / stdx_old);
    }

    //Save scaling vectors for later user, setup complete
    scale_vector = VVector(x);
}


// Matrix Scaling on Device
template <AMGX_VecPrecision t_vecPrec, AMGX_MatPrecision t_matPrec, AMGX_IndPrecision t_indPrec>
void BinormalizationScaler<TemplateConfig<AMGX_device, t_vecPrec, t_matPrec, t_indPrec> >::scaleMatrix(Matrix_d &A, ScaleDirection scaleOrUnscale)
{
    if (scale_vector.size() != A.get_num_rows())
    {
        FatalError("Must call setup(A) before binormalization scaling can scale matrix", AMGX_ERR_NOT_IMPLEMENTED);
    }

    if (A.is_matrix_distributed())
    {
        FatalError("Binormalization scaling not supported for distributed matrices", AMGX_ERR_NOT_IMPLEMENTED);
    }

    int nrows = A.get_num_rows();
    /*VVector rownorms(nrows, 0.0);
    VVector colnorms(nrows, 0.0);
    getColRowNorms<<<4096,256>>>(nrows, A.row_offsets.raw(), A.col_indices.raw(), A.values.raw(), rownorms.raw(), colnorms.raw());
    cudaCheckError();
    ValueTypeB row_max = *(amgx::thrust::max_element(rownorms.begin(), rownorms.end()));
    ValueTypeB row_min = *(amgx::thrust::min_element(rownorms.begin(), rownorms.end()));
    ValueTypeB col_max = *(amgx::thrust::max_element(colnorms.begin(), colnorms.end()));
    ValueTypeB col_min = *(amgx::thrust::min_element(colnorms.begin(), colnorms.end()));
    cudaCheckError();
    printf("Original Matrix: rowmax: %e, rowmin: %e, colmax: %e, colmin: %e\n", row_max, row_min, col_max, col_min);fflush(stdout);*/
    scaleMatrixDevice <<< 4096, 256>>>(nrows, A.row_offsets.raw(), A.col_indices.raw(), A.values.raw(), scale_vector.raw());
    cudaCheckError();
    ValueTypeB C_norm = sqrt(amgx::thrust::transform_reduce(A.values.begin(), A.values.begin() + A.get_num_nz() * A.get_block_size(), square_value<ValueTypeB>(), 0., amgx::thrust::plus<ValueTypeB>()) / nrows);
    amgx::thrust::transform(A.values.begin(), A.values.begin() + A.get_num_nz()*A.get_block_size(), A.values.begin(), vmul_scale_const<ValueTypeB>(1. / C_norm) );
    amgx::thrust::transform(scale_vector.begin(), scale_vector.end(), scale_vector.begin(), vmul_scale_const<ValueTypeB>(sqrt(1. / C_norm)) );
    cudaCheckError();
    /*amgx::thrust::fill(rownorms.begin(), rownorms.end(), 0.);
      amgx::thrust::fill(colnorms.begin(), colnorms.end(), 0.);
    getColRowNorms<<<4096,256>>>(nrows, A.row_offsets.raw(), A.col_indices.raw(), A.values.raw(), rownorms.raw(), colnorms.raw());
    cudaCheckError();
    row_max = *(amgx::thrust::max_element(rownorms.begin(), rownorms.end()));
    row_min = *(amgx::thrust::min_element(rownorms.begin(), rownorms.end()));
    col_max = *(amgx::thrust::max_element(colnorms.begin(), colnorms.end()));
    col_min = *(amgx::thrust::min_element(colnorms.begin(), colnorms.end()));
    cudaCheckError();
    printf("Scaled Matrix: rowmax: %e, rowmin: %e, colmax: %e, colmin: %e\n", row_max, row_min, col_max, col_min);fflush(stdout);*/
    exit(0);
}

// Setup on Host
template <AMGX_VecPrecision t_vecPrec, AMGX_MatPrecision t_matPrec, AMGX_IndPrecision t_indPrec>
void BinormalizationScaler<TemplateConfig<AMGX_host, t_vecPrec, t_matPrec, t_indPrec> >::setup(Matrix_h &A)
{
    FatalError("Host not supported", AMGX_ERR_NOT_IMPLEMENTED);
    /*if (A.is_matrix_distributed()) {
      FatalError("Binormalization scaling not supported for distributed matrices", AMGX_ERR_NOT_IMPLEMENTED);
    }
    // move these out to config parameters?
    const int max_iters = 10;
    const ValueTypeB tolerance = 1e-10;

    int rows = A.get_num_rows(), cols = A.get_num_cols();
    // temporary vectors
    VVector x(rows, 1), y(cols, 1), beta(rows, 0), gamma(cols, 0);

    // perform matvecs to get beta and gamma (spmv for beta, spmvT for gamma)
    computeBetaGammaHost(rows, A.row_offsets.raw(), A.col_indices.raw(), A.values.raw(),
                         x.raw(), y.raw(), beta.raw(), gamma.raw());

    double std1 = 0., std2 = 0., sum1 = cols, sum2 = rows;
    // calculate initial std1 and std2
    for (int i=0; i<rows; i++) {
      std1 += pow(x[i]*beta[i] - sum1, 2.0);
    }
    std1 = sqrt(std1 / rows) / sum1;

    for (int i=0; i<cols; i++) {
      std2 += pow(y[i]*gamma[i] - sum2, 2.0);
    }
    std2 = sqrt(std2 / cols) / sum2;

    //printf("std1: %lg, std2: %lg\n",std1, std2);
    double std_initial = sqrt((std1*std1)+(std2*std2));
    double std = std_initial;

    for (int t=0; t<max_iters; t++) {

      if (std < tolerance) break; // finished

      // x = sum1 ./ beta
      for (int i=0; i<rows; i++) x[i] = ( isNotCloseToZero(beta[i]) ? sum1 / beta[i] : sum1 / epsilon(beta[i]) );

      // gamma = C*x
      computeGammaHost(rows, A.row_offsets.raw(), A.col_indices.raw(), A.values.raw(), x.raw(), gamma.raw());

      // gamma = 1 ./ beta
      for (int i=0; i<cols; i++) y[i] = ( isNotCloseToZero(gamma[i]) ? sum2/gamma[i] : sum2 / epsilon(gamma[i]) );

      // beta = B*y
      computeBetaHost(rows, A.row_offsets.raw(), A.col_indices.raw(), A.values.raw(), y.raw(), beta.raw());

      //ValueTypeB std_old = std;
      std = 0.;
      for (int i=0; i<rows; i++) {
        std += pow(x[i]*beta[i] - sum1, 2.0);
      }
      std = sqrt(std / rows) / sum1;

      // print it #, current error, convergence rate
      //printf("ITER: %d     %.3e     %.4lg\n",t, std, std / std_old);
    }


    //Save scaling vectors for later user, setup complete
    left_scale = VVector(beta);
    right_scale= VVector(gamma);
    */
}


template <AMGX_VecPrecision t_vecPrec, AMGX_MatPrecision t_matPrec, AMGX_IndPrecision t_indPrec>
void BinormalizationScaler<TemplateConfig<AMGX_host, t_vecPrec, t_matPrec, t_indPrec> >::scaleMatrix(Matrix_h &A, ScaleDirection scaleOrUnscale)
{
    FatalError("Host not supported", AMGX_ERR_NOT_IMPLEMENTED);
    /*if (A.is_matrix_distributed()) {
      FatalError("Binormalization scaling not supported for distributed matrices", AMGX_ERR_NOT_IMPLEMENTED);
    }
    if (left_scale.size() != A.get_num_rows()) {
      FatalError("Must call setup(A) before binormalization scaling can scale matrix", AMGX_ERR_NOT_IMPLEMENTED);
    }

    // A_scaled = F*A*G (f = diag(F) = sqrt(fabs(x)), g = diag(G) = sqrt(fabs(y))
    // A_ij = f_i * A_ij * g_j
    scaleMatrixHost(A.get_num_rows(), A.row_offsets.raw(), A.col_indices.raw(), A.values.raw(), left_scale.raw(), right_scale.raw());*/
}


template <AMGX_VecPrecision t_vecPrec, AMGX_MatPrecision t_matPrec, AMGX_IndPrecision t_indPrec>
void BinormalizationScaler<TemplateConfig<AMGX_device, t_vecPrec, t_matPrec, t_indPrec> >::scaleVector(VVector &v, ScaleDirection scaleOrUnscale, ScaleSide leftOrRight)
{
    if (scaleOrUnscale == amgx::SCALE)
    {
        amgx::thrust::transform(v.begin(), v.end(), this->scale_vector.begin(), v.begin(), vmul_scale<ValueTypeB>() );
    }
    else
    {
        amgx::thrust::transform(v.begin(), v.end(), this->scale_vector.begin(), v.begin(), vmul_unscale<ValueTypeB>() );
    }
}

template <AMGX_VecPrecision t_vecPrec, AMGX_MatPrecision t_matPrec, AMGX_IndPrecision t_indPrec>
void BinormalizationScaler<TemplateConfig<AMGX_host, t_vecPrec, t_matPrec, t_indPrec> >::scaleVector(VVector &v, ScaleDirection scaleOrUnscale, ScaleSide leftOrRight)
{
    FatalError("4x4 block size not supported", AMGX_ERR_NOT_SUPPORTED_BLOCKSIZE);
}


/****************************************
 * Explict instantiations
 ***************************************/
#define AMGX_CASE_LINE(CASE) template class BinormalizationScaler_Base<TemplateMode<CASE>::Type>;
AMGX_FORALL_BUILDS(AMGX_CASE_LINE)
#undef AMGX_CASE_LINE

#define AMGX_CASE_LINE(CASE) template class BinormalizationScaler<TemplateMode<CASE>::Type>;
AMGX_FORALL_BUILDS(AMGX_CASE_LINE)
#undef AMGX_CASE_LINE


} // namespace amgx

