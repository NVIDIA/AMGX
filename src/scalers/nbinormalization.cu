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

#include <scalers/nbinormalization.h>
#include <sm_utils.inl>
#include <thrust/inner_product.h>
#include <solvers/block_common_solver.h>

namespace amgx
{

template<class TConfig> class NBinormalizationScaler;

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
void computeBetaHost(int rows, IndexType *offsets, IndexType *indices, MatrixType *vals,
                     VectorType *y, VectorType *beta)
{
    for (int i = 0; i < rows; i++)
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
void scaleMatrixHost(int rows, IndexType *offsets, IndexType *indices, MatrixType *values,
                     VectorType *x, VectorType *y)
{
    for (int i = 0; i < rows; i++)
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

// these warp reductions should be able to be replaced with amgx:: functions
template <int warpSize, typename T>
__device__ __inline__ T warpReduceSum(T val)
{
    if (warpSize > 16) { val += utils::shfl_down(val, 16, warpSize); }

    utils::syncwarp();

    if (warpSize > 8) { val += utils::shfl_down(val, 8, warpSize); }

    utils::syncwarp();

    if (warpSize > 4) { val += utils::shfl_down(val, 4, warpSize); }

    utils::syncwarp();

    if (warpSize > 2) { val += utils::shfl_down(val, 2, warpSize); }

    utils::syncwarp();

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

// compute beta = B*y, gamma = C*x (B = A.^2, C = B^T)
template <int CTASize, int VectorSize, int VectorsPerCTA, typename IndexType, typename MatrixValue, typename VectorValue>
__global__
void computeBetaGammaDevice(IndexType rows, IndexType *offsets, IndexType *indices, MatrixValue *values,
                            VectorValue *x, VectorValue *y, VectorValue *beta, VectorValue *gamma)
{
    const int vectors_per_block = VectorsPerCTA;
    const int vector_id = threadIdx.x / VectorSize;
    const int lane_id = threadIdx.x % VectorSize;

    for (int i = vectors_per_block * blockIdx.x + vector_id; i < rows; i += vectors_per_block * gridDim.x)
    {
        // load start + end pointers
        int row_tmp;

        if (lane_id < 2)
        {
            row_tmp = offsets[i + lane_id];
        }

        // distribute to all other threads in warp
        int row_begin = utils::shfl(row_tmp, vector_id * VectorSize, warpSize, utils::activemask());
        int row_end = utils::shfl(row_tmp, vector_id * VectorSize + 1, warpSize, utils::activemask());
        VectorValue bi(0.);

        for (int jj = row_begin + lane_id; utils::any(jj < row_end, utils::activemask()); jj += VectorSize)
        {
            int col = -1;
            VectorValue val(0.);

            if (jj < row_end)
            {
                col = indices[jj];
                val = values[jj];
                bi += (val * val) * y[col];
                utils::atomic_add(&gamma[col], (val * val) * x[i]);
            }
        }

        // reduce over bi
        VectorValue bi_s = warpReduceSum<VectorSize>(bi);

        if (lane_id == 0)
        {
            beta[i] = bi_s;
        }
    }
}

// compute gamma = B^T*x (B = A.^2)
template <int CTASize, int VectorSize, typename IndexType, typename MatrixValue, typename VectorValue>
__global__
void computeGammaDevice(int rows, IndexType *offsets, IndexType *indices, MatrixValue *values,
                        VectorValue *x, VectorValue *gamma)
{
    const int vectors_per_block = CTASize / VectorSize;
    const int vector_id = threadIdx.x / VectorSize;
    const int lane_id = threadIdx.x % VectorSize;

    for (int i = vectors_per_block * blockIdx.x + vector_id; i < rows; i += vectors_per_block * gridDim.x)
    {
        // load start + end pointers
        int row_tmp;

        if (lane_id < 2)
        {
            row_tmp = offsets[i + lane_id];
        }

        // distribute to all other threads in warp
        int row_begin = utils::shfl(row_tmp, vector_id * VectorSize, warpSize, utils::activemask());
        int row_end = utils::shfl(row_tmp, vector_id * VectorSize + 1, warpSize, utils::activemask());

        for (int jj = row_begin + lane_id; utils::any(jj < row_end, utils::activemask()); jj += VectorSize)
        {
            int col = -1;
            VectorValue val = 0.;

            if (jj < row_end)
            {
                col = indices[jj];
                val = values[jj];
                utils::atomic_add(&gamma[col], (val * val) * x[i]);
            }
        }
    }
}

// compute beta = B*y (B = A.^2)
template <int CTASize, int VectorSize, typename IndexType, typename MatrixValue, typename VectorValue>
__global__
void computeBetaDevice(int rows, IndexType *offsets, IndexType *indices, MatrixValue *values,
                       VectorValue *y, VectorValue *beta)
{
    const int vectors_per_block = CTASize / VectorSize;
    const int vector_id = threadIdx.x / VectorSize;
    const int lane_id = threadIdx.x % VectorSize;

    for (int i = vectors_per_block * blockIdx.x + vector_id; i < rows; i += vectors_per_block * gridDim.x)
    {
        // load start + end pointers
        int row_tmp;

        if (lane_id < 2)
        {
            row_tmp = offsets[i + lane_id];
        }

        // distribute to all other threads in warp
        int row_begin = utils::shfl(row_tmp, vector_id * VectorSize, warpSize, utils::activemask());
        int row_end = utils::shfl(row_tmp, vector_id * VectorSize + 1, warpSize, utils::activemask());
        VectorValue bi = 0.;

        for (int jj = row_begin + lane_id; utils::any(jj < row_end, utils::activemask()); jj += VectorSize)
        {
            int col = -1;
            VectorValue val = 0.;

            if (jj < row_end)
            {
                col = indices[jj];
                val = values[jj];
                bi += (val * val) * y[col];
            }
        }

        // reduce over bi
        VectorValue bi_s = warpReduceSum<VectorSize>(bi);

        if (lane_id == 0)
        {
            beta[i] = bi_s;
        }
    }
}

template <typename ValueType>
__global__
void setOneOverVector(int N, ValueType *x, ValueType sum1, ValueType *beta)
{
    for (int i = threadIdx.x + blockIdx.x * blockDim.x; i < N; i += gridDim.x * blockDim.x)
    {
        //x[i] = (  isNotCloseToZero(beta[i]) ? sum1 / beta[i] : sum1 /  epsilon(beta[i]) );
        x[i] = (  isNotCloseToZero(beta[i]) ? (sum1 / beta[i]) : (ValueType)1. );
    }
}

template<typename T>
struct square_value : public thrust::unary_function<T, T>
{
    __host__ __device__ T operator()(const T &x) const
    {
        return x * x;
    }
};

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
template <amgx::ScaleDirection direction, typename IndexType, typename MatrixType, typename VectorType>
__global__
void scaleMatrixDevice(int rows, IndexType *offsets, IndexType *indices, MatrixType *values,
                       VectorType *x, VectorType *y)
{
    for (int i = threadIdx.x + blockIdx.x * blockDim.x; i < rows; i += gridDim.x * blockDim.x)
    {
        VectorType fi = sqrt(fabs(x[i]));

        for (int jj = offsets[i]; jj < offsets[i + 1]; jj++)
        {
            int j = indices[jj];
            VectorType gj = sqrt(fabs(y[j]));

            // scale matrix value in place
            if (direction == amgx::SCALE)
            {
                values[jj] *= fi * gj;
            }
            else
            {
                values[jj] /= fi * gj;
            }
        }
    }
}


template <typename IndexType, typename MatrixType, typename VectorType>
__global__
void getColRowNorms(int rows, IndexType *offsets, IndexType *indices, MatrixType *values,
                    VectorType *rownorms, VectorType *colnorms)
{
    for (int i = threadIdx.x + blockIdx.x * blockDim.x; i < rows; i += gridDim.x * blockDim.x)
    {
        for (int jj = offsets[i]; jj < offsets[i + 1]; jj++)
        {
            int j = indices[jj];
            VectorType curval = values[jj] * values[jj];
            rownorms[i] += curval;
            utils::atomic_add(colnorms + j, curval);
        }
    }
}

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
void NBinormalizationScaler<TemplateConfig<AMGX_device, t_vecPrec, t_matPrec, t_indPrec> >::setup(Matrix_d &A)
{
    if (A.is_matrix_distributed())
    {
        FatalError("Binormalization scaling not supported for distributed matrices", AMGX_ERR_NOT_IMPLEMENTED);
    }

    // move these out to config parameters?
    const int max_iters = 50;
    const ValueTypeB tolerance = 1e-10;
    int rows = A.get_num_rows(), cols = A.get_num_cols();
    // temporary vectors
    VVector x(rows, 1), y(cols, 1), beta(rows, 0), gamma(cols, 0);
    // perform matvecs to get beta and gamma (spmv for beta, spmvT for gamma)
    computeBetaGammaDevice<256, 8, 32> <<< 4096, 256>>>(rows, A.row_offsets.raw(), A.col_indices.raw(), A.values.raw(),
            x.raw(), y.raw(), beta.raw(), gamma.raw());
    ValueTypeB sum1 = cols, sum2 = rows, std1, std2;
    // calculate initial std1 and std2
    thrust::device_ptr<ValueTypeB> x_ptr(x.raw()), y_ptr(y.raw()), beta_ptr(beta.raw()), gamma_ptr(gamma.raw());
    std1 = sqrt(thrust::inner_product(x_ptr, x_ptr + rows, beta_ptr, ValueTypeB(0.), thrust::plus<ValueTypeB>(), std_f<ValueTypeB>(sum1)) / rows) / sum1;
    std2 = sqrt(thrust::inner_product(y_ptr, y_ptr + cols, gamma_ptr, ValueTypeB(0.), thrust::plus<ValueTypeB>(), std_f<ValueTypeB>(sum2)) / cols) / sum2;
    ValueTypeB std = sqrt(std1 * std1 + std2 * std2);

    for (int t = 0; t < max_iters; t++)
    {
        if (std < tolerance) { break; } // finished

        // x = sum1 ./ beta
        setOneOverVector <<< 4096, 256>>>(rows, x.raw(), sum1, beta.raw());
        // gamma = C*x := B'*x
        thrust::fill(gamma.begin(), gamma.end(), ValueTypeB(0.));
        computeGammaDevice<256, 8> <<< 4096, 256>>>(rows, A.row_offsets.raw(), A.col_indices.raw(), A.values.raw(), x.raw(), gamma.raw());
        // gamma = 1 ./ beta
        setOneOverVector <<< 4096, 256>>>(cols, y.raw(), sum2, gamma.raw());
        // beta = B*y
        computeBetaDevice<256, 8> <<< 4096, 256>>>(rows, A.row_offsets.raw(), A.col_indices.raw(), A.values.raw(), y.raw(), beta.raw());
        //ValueTypeB std_old = std;
        std = sqrt(thrust::inner_product(x_ptr, x_ptr + rows, beta_ptr, ValueTypeB(0.), thrust::plus<ValueTypeB>(), std_f<ValueTypeB>(sum1)) / rows) / sum1;
        // print it #, current error, convergence rate
        //printf("ITER: %d     %.3e     %.4lg\n",t, std, std / std_old);
    }

    //Save scaling vectors for later user, setup complete
    left_scale = VVector(beta);
    right_scale = VVector(gamma);
    this->scaled_before = false;
}


// Matrix Scaling on Device
template <AMGX_VecPrecision t_vecPrec, AMGX_MatPrecision t_matPrec, AMGX_IndPrecision t_indPrec>
void NBinormalizationScaler<TemplateConfig<AMGX_device, t_vecPrec, t_matPrec, t_indPrec> >::scaleMatrix(Matrix_d &A, ScaleDirection scaleOrUnscale)
{
    if (left_scale.size() != A.get_num_rows())
    {
        FatalError("Must call setup(A) before binormalization scaling can scale matrix", AMGX_ERR_NOT_IMPLEMENTED);
    }

    if (A.is_matrix_distributed())
    {
        FatalError("Binormalization scaling not supported for distributed matrices", AMGX_ERR_NOT_IMPLEMENTED);
    }

    int nrows = A.get_num_rows();

    if (scaleOrUnscale == amgx::SCALE)
    {
        /*VVector rownorms(nrows, 0.0);
        VVector colnorms(nrows, 0.0);
        getColRowNorms<<<4096,256>>>(nrows, A.row_offsets.raw(), A.col_indices.raw(), A.values.raw(), rownorms.raw(), colnorms.raw());
        cudaCheckError();
        ValueTypeB row_max = *(thrust::max_element(rownorms.begin(), rownorms.end()));
        ValueTypeB row_min = *(thrust::min_element(rownorms.begin(), rownorms.end()));
        ValueTypeB col_max = *(thrust::max_element(colnorms.begin(), colnorms.end()));
        ValueTypeB col_min = *(thrust::min_element(colnorms.begin(), colnorms.end()));
        cudaCheckError();
        printf("Original Matrix: rowmax: %e, rowmin: %e, colmax: %e, colmin: %e\n", row_max, row_min, col_max, col_min);fflush(stdout);*/
        // A_scaled = F*A*G (f = diag(F) = sqrt(fabs(x)), g = diag(G) = sqrt(fabs(y))
        // A_ij = f_i * A_ij * g_j
        scaleMatrixDevice<amgx::SCALE> <<< 4096, 256>>>(A.get_num_rows(), A.row_offsets.raw(), A.col_indices.raw(), A.values.raw(), left_scale.raw(), right_scale.raw());
        cudaCheckError();

        if (!scaled_before)
        {
            this->norm_coef = sqrt(thrust::transform_reduce(A.values.begin(), A.values.begin() + A.get_num_nz() * A.get_block_size(), square_value<ValueTypeB>(), 0., thrust::plus<ValueTypeB>()) / A.get_num_rows());
            cudaCheckError();
            thrust::transform(A.values.begin(), A.values.begin() + A.get_num_nz()*A.get_block_size(), A.values.begin(), vmul_scale_const<ValueTypeB>(1. / this->norm_coef) );
            thrust::transform(left_scale.begin(), left_scale.end(), left_scale.begin(), vmul_scale_const<ValueTypeB>(sqrt(1. / this->norm_coef)) );
            thrust::transform(right_scale.begin(), right_scale.end(), right_scale.begin(), vmul_scale_const<ValueTypeB>(sqrt(1. / this->norm_coef)) );
            cudaCheckError();
            /*thrust::fill(rownorms.begin(), rownorms.end(), 0.);
            thrust::fill(colnorms.begin(), colnorms.end(), 0.);
            getColRowNorms<<<4096,256>>>(nrows, A.row_offsets.raw(), A.col_indices.raw(), A.values.raw(), rownorms.raw(), colnorms.raw());
            cudaCheckError();
            row_max = *(thrust::max_element(rownorms.begin(), rownorms.end()));
            row_min = *(thrust::min_element(rownorms.begin(), rownorms.end()));
            col_max = *(thrust::max_element(colnorms.begin(), colnorms.end()));
            col_min = *(thrust::min_element(colnorms.begin(), colnorms.end()));
            cudaCheckError();
            printf("Scaled Matrix: rowmax: %e, rowmin: %e, colmax: %e, colmin: %e\n", row_max, row_min, col_max, col_min);fflush(stdout);*/
        }

        this->scaled_before = true;
    }
    else
    {
        scaleMatrixDevice<amgx::UNSCALE> <<< 4096, 256>>>(A.get_num_rows(), A.row_offsets.raw(), A.col_indices.raw(), A.values.raw(), left_scale.raw(), right_scale.raw());
        cudaCheckError();
    }
}

// Setup on Host
template <AMGX_VecPrecision t_vecPrec, AMGX_MatPrecision t_matPrec, AMGX_IndPrecision t_indPrec>
void NBinormalizationScaler<TemplateConfig<AMGX_host, t_vecPrec, t_matPrec, t_indPrec> >::setup(Matrix_h &A)
{
    if (A.is_matrix_distributed())
    {
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
    for (int i = 0; i < rows; i++)
    {
        std1 += pow(x[i] * beta[i] - sum1, 2.0);
    }

    std1 = sqrt(std1 / rows) / sum1;

    for (int i = 0; i < cols; i++)
    {
        std2 += pow(y[i] * gamma[i] - sum2, 2.0);
    }

    std2 = sqrt(std2 / cols) / sum2;
    //printf("std1: %lg, std2: %lg\n",std1, std2);
    double std_initial = sqrt((std1 * std1) + (std2 * std2));
    double std = std_initial;

    for (int t = 0; t < max_iters; t++)
    {
        if (std < tolerance) { break; } // finished

        // x = sum1 ./ beta
        for (int i = 0; i < rows; i++) { x[i] = ( isNotCloseToZero(beta[i]) ? sum1 / beta[i] : sum1 / epsilon(beta[i]) ); }

        // gamma = C*x
        computeGammaHost(rows, A.row_offsets.raw(), A.col_indices.raw(), A.values.raw(), x.raw(), gamma.raw());

        // gamma = 1 ./ beta
        for (int i = 0; i < cols; i++) { y[i] = ( isNotCloseToZero(gamma[i]) ? sum2 / gamma[i] : sum2 / epsilon(gamma[i]) ); }

        // beta = B*y
        computeBetaHost(rows, A.row_offsets.raw(), A.col_indices.raw(), A.values.raw(), y.raw(), beta.raw());
        //ValueTypeB std_old = std;
        std = 0.;

        for (int i = 0; i < rows; i++)
        {
            std += pow(x[i] * beta[i] - sum1, 2.0);
        }

        std = sqrt(std / rows) / sum1;
        // print it #, current error, convergence rate
        //printf("ITER: %d     %.3e     %.4lg\n",t, std, std / std_old);
    }

    //Save scaling vectors for later user, setup complete
    left_scale = VVector(beta);
    right_scale = VVector(gamma);
}


template <AMGX_VecPrecision t_vecPrec, AMGX_MatPrecision t_matPrec, AMGX_IndPrecision t_indPrec>
void NBinormalizationScaler<TemplateConfig<AMGX_host, t_vecPrec, t_matPrec, t_indPrec> >::scaleMatrix(Matrix_h &A, ScaleDirection scaleOrUnscale)
{
    if (A.is_matrix_distributed())
    {
        FatalError("Binormalization scaling not supported for distributed matrices", AMGX_ERR_NOT_IMPLEMENTED);
    }

    if (left_scale.size() != A.get_num_rows())
    {
        FatalError("Must call setup(A) before binormalization scaling can scale matrix", AMGX_ERR_NOT_IMPLEMENTED);
    }

    // A_scaled = F*A*G (f = diag(F) = sqrt(fabs(x)), g = diag(G) = sqrt(fabs(y))
    // A_ij = f_i * A_ij * g_j
    scaleMatrixHost(A.get_num_rows(), A.row_offsets.raw(), A.col_indices.raw(), A.values.raw(), left_scale.raw(), right_scale.raw());
}


template <AMGX_VecPrecision t_vecPrec, AMGX_MatPrecision t_matPrec, AMGX_IndPrecision t_indPrec>
void NBinormalizationScaler<TemplateConfig<AMGX_device, t_vecPrec, t_matPrec, t_indPrec> >::scaleVector(VVector &v, ScaleDirection scaleOrUnscale, ScaleSide leftOrRight)
{
    VVector *scale_vector = (leftOrRight == amgx::LEFT) ? &this->left_scale : &this->right_scale;

    //thrust::transform(v.begin(), v.end(), scale_vector->begin(), v.begin(), (scaleOrUnscale == amgx::SCALE) ? vmul_scale<ValueTypeB>() : vmul_unscale<ValueTypeB>() );
    if (scaleOrUnscale == amgx::SCALE)
    {
        thrust::transform(v.begin(), v.end(), scale_vector->begin(), v.begin(), vmul_scale<ValueTypeB>() );
    }
    else
    {
        thrust::transform(v.begin(), v.end(), scale_vector->begin(), v.begin(), vmul_unscale<ValueTypeB>() );
    }
}

template <AMGX_VecPrecision t_vecPrec, AMGX_MatPrecision t_matPrec, AMGX_IndPrecision t_indPrec>
void NBinormalizationScaler<TemplateConfig<AMGX_host, t_vecPrec, t_matPrec, t_indPrec> >::scaleVector(VVector &v, ScaleDirection scaleOrUnscale, ScaleSide leftOrRight)
{
    FatalError("4x4 block size not supported", AMGX_ERR_NOT_SUPPORTED_BLOCKSIZE);
}


/****************************************
 * Explict instantiations
 ***************************************/
#define AMGX_CASE_LINE(CASE) template class NBinormalizationScaler_Base<TemplateMode<CASE>::Type>;
AMGX_FORALL_BUILDS(AMGX_CASE_LINE)
#undef AMGX_CASE_LINE

#define AMGX_CASE_LINE(CASE) template class NBinormalizationScaler<TemplateMode<CASE>::Type>;
AMGX_FORALL_BUILDS(AMGX_CASE_LINE)
#undef AMGX_CASE_LINE


} // namespace amgx

