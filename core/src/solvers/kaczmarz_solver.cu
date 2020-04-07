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

#include <solvers/kaczmarz_solver.h>
#include <solvers/block_common_solver.h>
#include <thrust/transform.h>
#include <basic_types.h>
#include <string.h>
#include <cutil.h>
#include <util.h>
#include <miscmath.h>
#include <sm_utils.inl>

namespace amgx
{

// -----------
// Kernels
// -----------

/*************************************************************************
* "random" hash function for both device and host
************************************************************************/
__host__ __device__ static int ourHash(const int i, const int max)
{
    unsigned int a = i;
    a = (a + 0x7ed55d16) + (a << 12);
    a = (a ^ 0xc761c23c) + (a >> 19);
    a = (a + 0x165667b1) + (a << 5);
    a = (a ^ 0xd3a2646c) + (a << 9);
    a = (a + 0xfd7046c5) + (a << 3);
    a = (a ^ 0xb55a4f09) + (a >> 16);
    return int(((a ^ 0x4a51e590) / (float)UINT_MAX) * max);
}

struct prg
{
    float a, b;
    int max_int;

    __host__ __device__
    prg(int _max_int, float _a = 0.f, float _b = 1.f) : a(_a), b(_b), max_int(_max_int) {};

    __host__ __device__
    int operator()(const unsigned int n) const
    {
        int  ru = ourHash(n, max_int);
        return (ru);
    }
};

template <class Vector>
void initRandom(Vector &vec, int size, int max_int)
{
    vec.resize(size);
    thrust::counting_iterator<unsigned int> index_sequence_begin(0);
    thrust::transform(index_sequence_begin,
                      index_sequence_begin + size,
                      vec.begin(),
                      prg(max_int));
}

template<typename IndexType, typename ValueTypeA, typename ValueTypeB>
__global__ void compute_anorm_kernel(const IndexType num_rows,
                                     const IndexType *Ap,
                                     const IndexType *Aj,
                                     const ValueTypeA *Ax,
                                     ValueTypeA *d)
{
    IndexType tidx = blockDim.x * blockIdx.x + threadIdx.x;

    for (int ridx = tidx; ridx < num_rows; ridx += blockDim.x * gridDim.x)
    {
        ValueTypeB d_ = 0;
        IndexType row_start = Ap[ridx];
        IndexType row_end   = Ap[ridx + 1];

        for (int j = row_start; j < row_end; j++)
        {
            ValueTypeB Aij = Ax[j];
            d_ += Aij * Aij;
        }

        // Store L2-norm
        d[ridx] = d_;
    }
}

template<typename IndexType, typename ValueTypeA, typename ValueTypeB>
__global__ void compute_multicolor_anorm_kernel(const IndexType num_rows,
        const IndexType *Ap,
        const IndexType *Aj,
        const ValueTypeA *Ax,
        const ValueTypeA *d,
        const int *sorted_rows_by_color,
        const int  num_rows_per_color)
{
    IndexType tidx = blockDim.x * blockIdx.x + threadIdx.x;

    for (int ridx = tidx; ridx < num_rows_per_color; ridx += blockDim.x * gridDim.x)
    {
        int i = sorted_rows_by_color[ridx];
        ValueTypeB d_ = 0;
        IndexType row_start = Ap[i];
        IndexType row_end   = Ap[i + 1];

        for (int j = row_start; j < row_end; j++)
        {
            ValueTypeB Aij = Ax[j];
            d_ += Aij * Aij;
        }

        // Store L2-norm
        d[ridx] = d_;
    }
}

template<typename IndexType, typename ValueTypeA, typename ValueTypeB>
__global__ void compute_cumul_inv_kernel(const IndexType a_cum_num_rows,
        ValueTypeA *a_cum,
        ValueTypeB d_inv,
        int c_inv_sz,
        IndexType *c_inv)
{
    const int tidx = blockDim.x * blockIdx.x + threadIdx.x;
    int idx, idx1, idx2;

    for (int ridx = tidx; ridx < a_cum_num_rows; ridx += blockDim.x * gridDim.x)
    {
        // printf("%d %f %f\n", ridx, d_inv, a_cum[ridx]);
        //printf("%f\n", a_cum[ridx]);
        double a = a_cum[ridx];
        // if (ridx < 0 || ridx >= a_cum_num_rows)
        //   printf("!! %d %d\n", ridx, idx);
        idx1 = int(a / d_inv) - 1; // get index in inverse table (floor - 1)

        if (ridx < a_cum_num_rows - 1)
        {
            idx2 = a_cum[ridx + 1] / d_inv - 1; // get index in inverse table (floor - 1)
        }
        else
        {
            idx2 = c_inv_sz;
        }

        // printf("%d %d\n", idx1, idx2);
        for ( idx = idx1; idx < idx2; idx++)
        {
            if (idx >= c_inv_sz || idx < 0)
            {
                printf("Ai! %d %d\n", idx, ridx);
            }

            c_inv[idx] = ridx;
        }
    }
}

template<typename IndexType, typename ValueTypeA, typename ValueTypeB>
__global__ void compute_amax_kernel(const IndexType num_rows,
                                    const IndexType *Ap,
                                    const IndexType *Aj,
                                    const ValueTypeA *Ax,
                                    IndexType *amax_idx)
{
    ValueTypeA maxVal(0), avalue;
    IndexType jmax;
    IndexType tidx = blockDim.x * blockIdx.x + threadIdx.x;

    for (int ridx = tidx; ridx < num_rows; ridx += blockDim.x * gridDim.x)
    {
        IndexType row_start = Ap[ridx];
        IndexType row_end   = Ap[ridx + 1];
        jmax = row_start;

        for (int j = row_start; j < row_end; j++)
        {
            avalue = Ax[j];

            if (avalue > maxVal)
            {
                maxVal = avalue;
                jmax = j;
            }
        }

        // Store position of maxvalue
        amax_idx[ridx] = jmax;
    }
}

template<typename IndexType, typename ValueTypeA, typename ValueTypeB>
__global__ void kaczmarz_smooth_kernel_naive_atomics(const IndexType num_rows,
        const IndexType *Ap,
        const IndexType *Aj,
        const ValueTypeA *Ax,
        const ValueTypeA *d,
        const ValueTypeB *b,
        const ValueTypeB *x,
        ValueTypeB *xout,
        const IndexType row_offset)

{
    // Naive implementation, needs x copy in xout at the very beginning
    IndexType tidx = blockDim.x * blockIdx.x + threadIdx.x;

    for (int ridx = row_offset + tidx; ridx < num_rows; ridx += blockDim.x * gridDim.x)
    {
        IndexType row_start = Ap[ridx];
        IndexType row_end   = Ap[ridx + 1];
        ValueTypeB Axi = 0.0;
        ValueTypeB r;

        for (int j = row_start; j < row_end; j++)
        {
            Axi += Ax[j] * xout[Aj[j]];
        }

        r = (b[ridx] - Axi) / ( isNotCloseToZero( d[ridx]) ? d[ridx] : epsilon(d[ridx]) );

        for (int j = row_start; j < row_end; j++)
        {
            //xout[Aj[j]] += r*Ax[j];
            utils::atomic_add(&xout[Aj[j]], r * Ax[j]);
        }
    }
}

template<typename IndexType, typename ValueTypeA, typename ValueTypeB, int kCtaSize>
__global__ void kaczmarz_smooth_kernel_warp_atomics(const IndexType num_rows,
        const IndexType *Ap,
        const IndexType *Aj,
        const ValueTypeA *Ax,
        const ValueTypeA *d,
        const ValueTypeB *b,
        const ValueTypeB *x,
        ValueTypeB *xout,
        const IndexType row_offset)

{
    const int num_warps = kCtaSize / 32;
    const int num_rows_per_iter = num_warps * gridDim.x;
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 300
    __shared__ volatile ValueTypeB smem[kCtaSize];
#endif
    const int warpId = threadIdx.x / 32;
    const int laneId = threadIdx.x % 32;

    for ( int ridx = blockIdx.x * num_warps + warpId ; ridx < num_rows ;
            ridx += num_rows_per_iter )
    {
        IndexType row_start = Ap[ridx];
        IndexType row_end   = Ap[ridx + 1];
        ValueTypeB Axi = 0.0;
        ValueTypeB r;

        for (int j = row_start + laneId; utils::any( j < row_end) ; j += 32)
        {
            ValueTypeB aValue = j < row_end ? Ax[j] : ValueTypeB(0);
            ValueTypeB xValue = j < row_end ? xout[Aj[j]] : ValueTypeB(0);
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 300
            r = utils::warp_reduce<1, utils::Add>(aValue * xValue);
#endif
            //#else
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 300
            r = utils::warp_reduce<1, utils::Add>(smem, aValue * xValue);
#endif
            Axi += r;
        }

        r = (b[ridx] - Axi) / ( isNotCloseToZero( d[ridx]) ? d[ridx] : epsilon(d[ridx]) );

        for (int j = row_start + laneId; utils::any( j < row_end) ; j += 32)
        {
            //ValueTypeB dx =  j < row_end ? r*Ax[j] : ValueTypeB(0);
            //int aj = j < row_end ? r*Ax[j] : ValueTypeB(0);
            if (j < row_end)
            {
                utils::atomic_add(&xout[Aj[j]], r * Ax[j]);
            }
        }
    }
}

template<typename IndexType, typename ValueTypeA, typename ValueTypeB, int kCtaSize>
__global__ void randomized_kaczmarz_smooth_kernel_warp_atomics(const IndexType num_rows,
        const IndexType *Ap,
        const IndexType *Aj,
        const ValueTypeA *Ax,
        const IndexType *c_inv,
        const IndexType *rnd_rows,
        const ValueTypeB *b,
        const ValueTypeB *x,
        ValueTypeB *xout,
        const IndexType row_offset)

{
    const int num_warps = kCtaSize / 32;
    const int num_rows_per_iter = num_warps * gridDim.x;
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 300
    __shared__ volatile ValueTypeB smem[kCtaSize];
#endif
    const int warpId = threadIdx.x / 32;
    const int laneId = threadIdx.x % 32;

    for ( int ridx = blockIdx.x * num_warps + warpId ; ridx < num_rows ;
            ridx += num_rows_per_iter )
    {
        int irow = c_inv[rnd_rows[ridx]];
        IndexType row_start = Ap[irow];
        IndexType row_end   = Ap[irow + 1];
        ValueTypeB Axi = 0.0;
        ValueTypeB r;
        ValueTypeA aa;
        ValueTypeA AA = 0.0;

        for (int j = row_start + laneId; utils::any( j < row_end) ; j += 32)
        {
            ValueTypeB aValue = j < row_end ? Ax[j] : ValueTypeB(0);
            ValueTypeB xValue = j < row_end ? xout[Aj[j]] : ValueTypeB(0);
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 300
            r = utils::warp_reduce<1, utils::Add>(aValue * xValue);
            aa = utils::warp_reduce<1, utils::Add>(aValue * aValue);
#endif
            //#else
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 300
            r = utils::warp_reduce<1, utils::Add>(smem, aValue * xValue);
            aa = utils::warp_reduce<1, utils::Add>(smem, aValue * aValue);
#endif
            Axi += r;
            AA += aa;
        }

        r = (b[ridx] - Axi) / ( isNotCloseToZero( AA) ? AA : epsilon(AA) );

        for (int j = row_start + laneId; utils::any( j < row_end) ; j += 32)
        {
            //ValueTypeB dx =  j < row_end ? r*Ax[j] : ValueTypeB(0);
            //int aj = j < row_end ? r*Ax[j] : ValueTypeB(0);
            if (j < row_end)
            {
                utils::atomic_add(&xout[Aj[j]], r * Ax[j]);
            }
        }
    }
}

template<typename IndexType, typename ValueTypeA, typename ValueTypeB, int kCtaSize>
__global__ void kaczmarz_smooth_kernel(const IndexType num_rows,
                                       const IndexType *Ap,
                                       const IndexType *Aj,
                                       const ValueTypeA *Ax,
                                       const IndexType *amax,
                                       const ValueTypeB *b,
                                       const ValueTypeB *x,
                                       ValueTypeB *xout,
                                       const IndexType row_offset)

{
    // Naive implementation, needs x copy in xout at the very beginning
    //IndexType tidx = blockDim.x*blockIdx.x + threadIdx.x;
    IndexType i, t;
    const int num_warps = kCtaSize / 32;
    const int num_rows_per_iter = num_warps * gridDim.x;
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 300
    __shared__ volatile ValueTypeB smem[kCtaSize];
#endif
    const int warpId = threadIdx.x / 32;
    const int laneId = threadIdx.x % 32;

    for ( int ridx = blockIdx.x * num_warps + warpId ; ridx < num_rows ;
            ridx += num_rows_per_iter )
    {
        ValueTypeB Axi = 0.0;
        ValueTypeB r;
        i = ourHash(ridx, num_rows);
        IndexType row_start = Ap[i];
        IndexType row_end   = Ap[i + 1];

        for (int j = row_start + laneId; utils::any( j < row_end) ; j += 32)
        {
            ValueTypeB aValue = j < row_end ? Ax[j] : ValueTypeB(0);
            ValueTypeB xValue = j < row_end ? xout[Aj[j]] : ValueTypeB(0);
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 300
            r = utils::warp_reduce<1, utils::Add>(aValue * xValue);
#endif
            //#else
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 300
            r = utils::warp_reduce<1, utils::Add>(smem, aValue * xValue);
#endif
            Axi += r;
            //Axi += utils::Warp_reduce_linear<1,32>::execute<utils::Add,ValueTypeB>(aValue * xValue);
            //Axi += Ax[j] * xout[Aj[j]];
            printf("j = %d, r = %f\n", j, r);
        }

        if (laneId == 0)
        {
            r = (b[i] - Axi);// / ( isNotCloseToZero( d[ridx]) ? d[ridx] : epsilon(d[ridx]) );
            t = row_start + ourHash(ridx, row_end - row_start);
            printf("ridx=%d, i=%d, t=%d, Aj[t]=%d, r=%f\n", ridx, i, t, Aj[t], r);
            xout[Aj[t]] += r * ((row_end - row_start) * Ax[t]) * 0.5;
        }
    }
}

template<typename IndexType, typename ValueTypeA, typename ValueTypeB>
__global__ void multicolor_kaczmarz_smooth_kernel_naive(const IndexType num_rows,
        const IndexType *Ap,
        const IndexType *Aj,
        const ValueTypeA *Ax,
        const ValueTypeA *d,
        const ValueTypeB *b,
        const ValueTypeB *x,
        ValueTypeB weight,
        const int *sorted_rows_by_color,
        const int     num_rows_per_color,
        ValueTypeB *xout)

{
    int i;
    // Naive implementation, needs x copy in xout at the very beginning
    IndexType tidx = blockDim.x * blockIdx.x + threadIdx.x;

    for (int ridx = tidx; ridx < num_rows_per_color; ridx += blockDim.x * gridDim.x)
    {
        i = sorted_rows_by_color[ridx];
        IndexType row_start = Ap[i];
        IndexType row_end   = Ap[i + 1];
        ValueTypeB Axi = 0.0;
        ValueTypeB r;

        for (int j = row_start; j < row_end; j++)
        {
            Axi += Ax[j] * xout[Aj[j]];
        }

        r = (b[i] - Axi) / ( isNotCloseToZero( d[i]) ? d[i] : epsilon(d[i]) );

        for (int j = row_start; j < row_end; j++)
        {
            utils::atomic_add(&xout[Aj[j]], r * Ax[j]);
        }
    }
}

template<typename IndexType, typename ValueTypeA, typename ValueTypeB, int kCtaSize>
__global__ void multicolor_kaczmarz_smooth_kernel(const IndexType num_rows,
        const IndexType *Ap,
        const IndexType *Aj,
        const ValueTypeA *Ax,
        const ValueTypeA *d,
        const ValueTypeB *b,
        const ValueTypeB *x,
        ValueTypeB weight,
        const int *sorted_rows_by_color,
        const int     num_rows_per_color,
        ValueTypeB *xout)

{
    const int num_warps = kCtaSize / 32;
    const int num_rows_per_iter = num_warps * gridDim.x;
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 300
    __shared__ volatile ValueTypeB smem[kCtaSize];
#endif
    const int warpId = threadIdx.x / 32;
    const int laneId = threadIdx.x % 32;
    int i;

    for ( int ridx = blockIdx.x * num_warps + warpId ; ridx < num_rows_per_color ;
            ridx += num_rows_per_iter )
    {
        i = sorted_rows_by_color[ridx];
        IndexType row_start = Ap[i];
        IndexType row_end   = Ap[i + 1];
        ValueTypeB Axi = 0.0;
        ValueTypeB r;

        for (int j = row_start + laneId; utils::any( j < row_end) ; j += 32)
        {
            ValueTypeB aValue = j < row_end ? Ax[j] : ValueTypeB(0);
            ValueTypeB xValue = j < row_end ? xout[Aj[j]] : ValueTypeB(0);
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 300
            r = utils::warp_reduce<1, utils::Add>(aValue * xValue);
            //#else
#endif
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 300
            r = utils::warp_reduce<1, utils::Add>(smem, aValue * xValue);
#endif
            Axi += r;
        }

        r = (b[i] - Axi) / ( isNotCloseToZero( d[i]) ? d[i] : epsilon(d[i]) );

        for (int j = row_start + laneId; utils::any( j < row_end) ; j += 32)
        {
            if (j < row_end)
            {
                utils::atomic_add(&xout[Aj[j]], r * Ax[j]);
            }
        }
    }
}

template<typename IndexType, typename ValueTypeA, typename ValueTypeB>
__global__ void jacobi_smooth_with_0_initial_guess_kernel(const IndexType num_rows,
        const ValueTypeA *d,
        const ValueTypeB *b,
        ValueTypeB *x,
        ValueTypeB weight,
        const IndexType row_offset)

{
    IndexType tidx = blockDim.x * blockIdx.x + threadIdx.x;

    for (int ridx = row_offset + tidx; ridx < num_rows; ridx += blockDim.x * gridDim.x)
    {
        x[ridx] = weight * b[ridx] /  ( isNotCloseToZero( d[ridx]) ? d[ridx] : epsilon(d[ridx]) );
    }
}

// -----------------
// Methods
// -----------------

// Constructor
template<class T_Config>
KaczmarzSolver_Base<T_Config>::KaczmarzSolver_Base( AMG_Config &cfg, const std::string &cfg_scope) : Solver<T_Config>( cfg, cfg_scope), m_an(0), m_amax(0), m_c_inv(0)
{
    weight = cfg.AMG_Config::getParameter<double>("relaxation_factor", cfg_scope);
    this->m_coloring_needed = (cfg.AMG_Config::getParameter<int>("kaczmarz_coloring_needed", cfg_scope) != 0);
    this->m_reorder_cols_by_color_desired = (cfg.AMG_Config::getParameter<int>("reorder_cols_by_color", cfg_scope) != 0);
    this->m_randomized = true;

    if (weight == 0)
    {
        weight = 1.;
        amgx_printf("Warning, setting weight to 1 instead of estimating largest_eigen_value in Block Jacobi smoother\n");;
    }
}

// Destructor
template<class T_Config>
KaczmarzSolver_Base<T_Config>::~KaczmarzSolver_Base()
{
}

// Solver setup
template<class T_Config>
void
KaczmarzSolver_Base<T_Config>::solver_setup(bool reuse_matrix_structure)
{
    m_explicit_A = dynamic_cast<Matrix<T_Config>*>(Base::m_A);

    if (!m_explicit_A)
    {
        FatalError("Kaczmarz solver only works with explicit matrices", AMGX_ERR_INTERNAL);
    }

    compute_anorm( *this->m_explicit_A );

    if (m_randomized) // MC RK is not supported here
    {
        if (m_coloring_needed)
        {
            //FatalError("Randomized Kaczmarz solver does not support coloring", AMGX_ERR_INTERNAL);
            m_coloring_needed = false;
        }

        double d_inv = this->m_an[0];
        int c_sz = this->m_an.size();
        d_inv = thrust::reduce(this->m_an.begin(), this->m_an.end(), d_inv, thrust::minimum<ValueTypeA>());
        thrust::inclusive_scan(this->m_an.begin(), this->m_an.end(), this->m_an.begin()); // in-place scan
        int c_inv_sz = (this->m_an[c_sz - 1] + d_inv - 1 ) / d_inv;
        this->m_c_inv.resize(c_inv_sz, -1);
        const size_t THREADS_PER_BLOCK  = 128;
        const size_t NUM_BLOCKS = min(AMGX_GRID_MAX_SIZE, (int)ceil((ValueTypeB)c_sz / (ValueTypeB)THREADS_PER_BLOCK));

        if (c_sz > 0)
        {
            device_vector_alloc<ValueTypeA> aa(c_sz, 1);
            compute_cumul_inv_kernel<IndexType, ValueTypeA, ValueTypeB> <<< (unsigned int)NUM_BLOCKS, (unsigned int)THREADS_PER_BLOCK>>>
            (c_sz,
             this->m_an.raw(),
             d_inv,
             c_inv_sz,
             this->m_c_inv.raw());
        }

        cudaDeviceSynchronize();
        cudaCheckError();
    }
}

template<class T_Config>
void KaczmarzSolver_Base<T_Config>::compute_anorm( Matrix<T_Config> &A)
{
    this->m_an.resize(A.get_num_rows()*A.get_block_dimx());
    ViewType oldView = A.currentView();
    A.setView(this->m_explicit_A->getViewExterior());

    if (A.get_block_dimx() == 1 && A.get_block_dimy() == 1)
    {
        compute_anorm_1x1(A);
    }
    else
    {
        FatalError("Unsupported block size for KaczmarzSolver", AMGX_ERR_NOT_SUPPORTED_BLOCKSIZE);
    }

    A.setView(oldView);
}

template<class T_Config>
void KaczmarzSolver_Base<T_Config>::compute_amax( Matrix<T_Config> &A)
{
    this->m_amax.resize(A.get_num_rows()*A.get_block_dimx());
    ViewType oldView = A.currentView();
    A.setView(this->m_explicit_A->getViewExterior());

    if (A.get_block_dimx() == 1 && A.get_block_dimy() == 1)
    {
        compute_amax_1x1(A);
    }
    else
    {
        FatalError("Unsupported block size for KaczmarzSolver", AMGX_ERR_NOT_SUPPORTED_BLOCKSIZE);
    }

    A.setView(oldView);
}

template <AMGX_VecPrecision t_vecPrec, AMGX_MatPrecision t_matPrec, AMGX_IndPrecision t_indPrec>
void KaczmarzSolver<TemplateConfig<AMGX_device, t_vecPrec, t_matPrec, t_indPrec> >::compute_anorm_1x1(const Matrix_d &A)
{
//DIAG: starnge issues trying to add DIAG property handling
// now leaving !DIAG only
    if (A.hasProps(DIAG))
    {
        FatalError("Unsupported separate diag", AMGX_ERR_NOT_IMPLEMENTED);
    }

    typedef typename Matrix_d::index_type IndexType;
    typedef typename Matrix_d::value_type ValueTypeA;
    const size_t THREADS_PER_BLOCK  = 128;
    const size_t NUM_BLOCKS = min(AMGX_GRID_MAX_SIZE, (int)ceil((ValueTypeB)A.get_num_rows() / (ValueTypeB)THREADS_PER_BLOCK));

    if (A.get_num_rows() > 0)
    {
        compute_anorm_kernel<IndexType, ValueTypeA, ValueTypeB> <<< (unsigned int)NUM_BLOCKS, (unsigned int)THREADS_PER_BLOCK>>>
        ((int)A.get_num_rows(),
         A.row_offsets.raw(),
         A.col_indices.raw(),
         A.values.raw(),
         this->m_an.raw());
    }

    cudaCheckError();
}

template <AMGX_VecPrecision t_vecPrec, AMGX_MatPrecision t_matPrec, AMGX_IndPrecision t_indPrec>
void KaczmarzSolver<TemplateConfig<AMGX_device, t_vecPrec, t_matPrec, t_indPrec> >::compute_amax_1x1(const Matrix_d &A)
{
//DIAG: starnge issues trying to add DIAG property handling
// now leaving !DIAG only
    if (A.hasProps(DIAG))
    {
        FatalError("Unsupported separate diag", AMGX_ERR_NOT_IMPLEMENTED);
    }

    typedef typename Matrix_d::index_type IndexType;
    typedef typename Matrix_d::value_type ValueTypeA;
    const size_t THREADS_PER_BLOCK  = 128;
    const size_t NUM_BLOCKS = min(AMGX_GRID_MAX_SIZE, (int)ceil((ValueTypeB)A.get_num_rows() / (ValueTypeB)THREADS_PER_BLOCK));

    if (A.get_num_rows() > 0)
    {
        compute_amax_kernel<IndexType, ValueTypeA, ValueTypeB> <<< (unsigned int)NUM_BLOCKS, (unsigned int)THREADS_PER_BLOCK>>>
        ((int)A.get_num_rows(),
         A.row_offsets.raw(),
         A.col_indices.raw(),
         A.values.raw(),
         this->m_amax.raw());
    }

    cudaCheckError();
}

template<class T_Config>
void
KaczmarzSolver_Base<T_Config>::solve_init( VVector &b, VVector &x, bool xIsZero )
{
}

// Solve one iteration
template<class T_Config>
bool
KaczmarzSolver_Base<T_Config>::solve_iteration( VVector &b, VVector &x, bool xIsZero )
{
    if (xIsZero) { x.dirtybit = 0; }

    if (!this->m_explicit_A->is_matrix_singleGPU())
    {
        this->m_explicit_A->manager->exchange_halo_async(x, x.tag);

        if (this->m_explicit_A->getViewExterior() == this->m_explicit_A->getViewInterior())
        {
            this->m_explicit_A->manager->exchange_halo_wait(x, x.tag);
        }
    }

    ViewType oldView = this->m_explicit_A->currentView();
    ViewType flags;
    bool latencyHiding = true;

    if (this->m_explicit_A->is_matrix_singleGPU() || (x.dirtybit == 0))
    {
        latencyHiding = false;
        this->m_explicit_A->setViewExterior();
        flags = this->m_explicit_A->getViewExterior();
    }
    else
    {
        flags = this->m_explicit_A->getViewInterior();
        this->m_explicit_A->setViewInterior();
    }

    if (this->m_explicit_A->get_block_dimx() == 1 && this->m_explicit_A->get_block_dimy() == 1)
    {
        if (xIsZero)
        {
            //smooth_with_0_initial_guess_1x1(*this->m_explicit_A, b, x, flags);
            smooth_1x1(*this->m_explicit_A, b, x, flags, latencyHiding);
        }
        else
        {
            smooth_1x1(*this->m_explicit_A, b, x, flags, latencyHiding);
        }
    }
    else
    {
        FatalError("Unsupported block size for Kaczmarz_Solver", AMGX_ERR_NOT_SUPPORTED_BLOCKSIZE);
    }

    x.dirtybit = 1;
    this->m_explicit_A->setView(oldView);
    return this->converged( b, x );
}

template<class T_Config>
void
KaczmarzSolver_Base<T_Config>::solve_finalize( VVector &b, VVector &x )
{
}

// Multicolor version
template <AMGX_VecPrecision t_vecPrec, AMGX_MatPrecision t_matPrec, AMGX_IndPrecision t_indPrec>
void KaczmarzSolver<TemplateConfig<AMGX_device, t_vecPrec, t_matPrec, t_indPrec> >::smooth_1x1_MC(Matrix_d &A, VVector &b, VVector &x, ViewType separation_flags, bool latency_hiding)
{
    typedef typename Matrix_d::index_type IndexType;
    typedef typename Matrix_d::value_type ValueTypeA;
    ValueTypeB *x_ptr = x.raw();
    IndexType num_rows = A.get_num_rows();
    const int num_colors = this->m_explicit_A->getMatrixColoring().getNumColors();
    const IndexType *A_sorted_rows_by_color_ptr = A.getMatrixColoring().getSortedRowsByColor().raw();

    for (int i = 0; i < num_colors; i++)
    {
        const IndexType color_offset = ((separation_flags & INTERIOR) == 0) ? A.getMatrixColoring().getSeparationOffsetsRowsPerColor()[i] : A.getMatrixColoring().getOffsetsRowsPerColor()[i];
        const IndexType num_rows_per_color = ((separation_flags == this->m_explicit_A->getViewInterior()) ? A.getMatrixColoring().getSeparationOffsetsRowsPerColor()[i] : A.getMatrixColoring().getOffsetsRowsPerColor()[i + 1]) - color_offset;

        if (num_rows_per_color == 0) { continue; }

        const int threads_per_block = 128;
        const int blockrows_per_warp = 1;
        const int blockrows_per_cta = (threads_per_block / 32) * blockrows_per_warp;
        const int num_blocks = min( AMGX_GRID_MAX_SIZE, (int) (num_rows_per_color / blockrows_per_cta + 1));
        multicolor_kaczmarz_smooth_kernel<IndexType, ValueTypeA, ValueTypeB, threads_per_block> <<< num_blocks, threads_per_block >>>
        (A.get_num_rows(),
         A.row_offsets.raw(),
         A.col_indices.raw(),
         A.values.raw(),
         this->m_an.raw(),
         b.raw(),
         x_ptr,
         this->weight,
         A_sorted_rows_by_color_ptr + color_offset, num_rows_per_color,
         x_ptr);
        cudaCheckError();
    }
}

template <AMGX_VecPrecision t_vecPrec, AMGX_MatPrecision t_matPrec, AMGX_IndPrecision t_indPrec>
void KaczmarzSolver<TemplateConfig<AMGX_device, t_vecPrec, t_matPrec, t_indPrec> >::smooth_1x1_naive(Matrix_d &A, VVector &b, VVector &x, ViewType separation_flags, bool latency_hiding)
{
    typedef typename Matrix_d::index_type IndexType;
    typedef typename Matrix_d::value_type ValueTypeA;
    ValueTypeB *x_ptr = x.raw();
    IndexType num_rows = A.get_num_rows();
    IndexType offset = 0;
    // Skipping Multi-GPU logic for now
    // Current. Will be exact only with one warp per grid
    const int threads_per_block = 32;
    const int num_blocks = 1;

    if (this->m_randomized)
    {
        IVector rnd_rows;
        int c_inv_sz = this->m_c_inv.size();
        initRandom(rnd_rows, A.get_num_rows(), c_inv_sz);
        randomized_kaczmarz_smooth_kernel_warp_atomics<IndexType, ValueTypeA, ValueTypeB, threads_per_block> <<< num_blocks, threads_per_block >>>
        (A.get_num_rows(),
         A.row_offsets.raw(),
         A.col_indices.raw(),
         A.values.raw(),
         this->m_c_inv.raw(),
         rnd_rows.raw(),
         b.raw(),
         x_ptr,
         x_ptr,
         offset);
    }
    else
    {
        kaczmarz_smooth_kernel_warp_atomics<IndexType, ValueTypeA, ValueTypeB, threads_per_block> <<< num_blocks, threads_per_block >>>
        (A.get_num_rows(),
         A.row_offsets.raw(),
         A.col_indices.raw(),
         A.values.raw(),
         this->m_an.raw(),
         b.raw(),
         x_ptr,
         x_ptr,
         offset);
    }

    cudaCheckError();
}

template <AMGX_VecPrecision t_vecPrec, AMGX_MatPrecision t_matPrec, AMGX_IndPrecision t_indPrec>
void KaczmarzSolver<TemplateConfig<AMGX_device, t_vecPrec, t_matPrec, t_indPrec> >::smooth_1x1(Matrix_d &A, VVector &b, VVector &x, ViewType separation_flags, bool latency_hiding)
{
    if (this->m_coloring_needed)
    {
        smooth_1x1_MC(A, b, x, separation_flags, latency_hiding);
    }
    else
    {
        smooth_1x1_naive(A, b, x, separation_flags, latency_hiding);
    }
}

template <AMGX_VecPrecision t_vecPrec, AMGX_MatPrecision t_matPrec, AMGX_IndPrecision t_indPrec>
void KaczmarzSolver<TemplateConfig<AMGX_device, t_vecPrec, t_matPrec, t_indPrec> >::smooth_with_0_initial_guess_1x1(Matrix_d &A, VVector &b, VVector &x, ViewType separation_flags)
{
    ViewType oldView = A.currentView();
    // Process all rows
    A.setViewExterior();
    ViewType flags = A.getViewExterior();
    int offset, num_rows;
    A.getOffsetAndSizeForView(flags, &offset, &num_rows);
    A.setView(oldView);
    cudaCheckError();
}

/****************************************
 * Explict instantiations
 ***************************************/
#define AMGX_CASE_LINE(CASE) template class KaczmarzSolver_Base<TemplateMode<CASE>::Type>;
AMGX_FORALL_BUILDS(AMGX_CASE_LINE)
#undef AMGX_CASE_LINE

#define AMGX_CASE_LINE(CASE) template class KaczmarzSolver<TemplateMode<CASE>::Type>;
AMGX_FORALL_BUILDS(AMGX_CASE_LINE)
#undef AMGX_CASE_LINE

} // namespace amgx
