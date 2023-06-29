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

#include <aggregation/aggregation_amg_level.h>
#include <matrix_analysis.h>

#ifdef _WIN32
#pragma warning (push)
#pragma warning (disable : 4244 4267 4521)
#endif
#ifdef _WIN32
#pragma warning (pop)
#endif

#include <basic_types.h>
#include <util.h>
#include <fstream>
#include <cutil.h>
#include <multiply.h>
#include <transpose.h>
#include <blas.h>
#include <string>
#include <string.h>
#include <iostream>
#include <algorithm>
#include <amgx_timer.h>

#include <amgx_types/util.h>

#include <thrust/sort.h>
#include <thrust/remove.h>
#include <thrust/transform.h>
#include <thrust/binary_search.h>
#include <thrust/unique.h>
#include <thrust/inner_product.h>
#include <thrust/iterator/constant_iterator.h>
#include <thrust_wrapper.h>

namespace amgx
{

namespace aggregation
{


// ----------------------
// Kernels
// ----------------------

template <typename IndexType, typename ValueType>
__global__
void set_to_one_kernel(IndexType start, IndexType end, IndexType *ind, ValueType *v)
{
    for (int tid = start + blockDim.x * blockIdx.x + threadIdx.x; tid < end; tid += gridDim.x * blockDim.x)
    {
        v[ind[tid]] = types::util<ValueType>::get_one();
    }
}

template <typename IndexType>
__global__
void renumberAggregatesKernel(const IndexType *renumbering, const int interior_offset, const int bdy_offset, IndexType *aggregates, const int num_aggregates, const int n_interior, const int renumbering_size)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    while (tid < num_aggregates)
    {
        IndexType new_agg_id;

        if (renumbering_size == 0)
        {
            new_agg_id = aggregates[tid];
        }
        else
        {
            new_agg_id = renumbering[aggregates[tid]];
        }

        //if (aggregates[tid] > num_aggregates)
        //{
        //printf("ID %d old %d + %d = %d\n", tid, new_agg_id, ((new_agg_id >= n_interior) ? bdy_offset : interior_offset), new_agg_id + ((new_agg_id >= n_interior) ? bdy_offset : interior_offset));
        //}
        new_agg_id +=  ((new_agg_id >= n_interior) ? bdy_offset : interior_offset);
        aggregates[tid] = new_agg_id;
        tid += gridDim.x * blockDim.x;
    }
}

// Kernel to restrict residual using csr_format
template <typename IndexType, typename ValueType>
__global__
void restrictResidualKernel(const IndexType *row_offsets, const IndexType *column_indices, const ValueType *r, ValueType *rr, const int num_aggregates)
{
    int jmin, jmax;

    for (int tid = blockDim.x * blockIdx.x + threadIdx.x; tid < num_aggregates; tid += gridDim.x * blockDim.x)
    {
        ValueType temp(types::util<ValueType>::get_zero());
        jmin = row_offsets[tid];
        jmax = row_offsets[tid + 1];

        for (int j = jmin; j < jmax; j++)
        {
            int j_col = column_indices[j];
            temp = temp + r[j_col];
        }

        rr[tid] = temp;
    }
}

// Kernel to restrict residual using block_dia_csr_format
template <typename IndexType, typename ValueType, int bsize>
__global__
void restrictResidualBlockDiaCsrKernel(const IndexType *row_offsets, const IndexType *column_indices, const ValueType *r, ValueType *rr, const int num_aggregates)
{
    ValueType rr_temp[bsize];
    int offset, jmin, jmax;

    for (int tid = blockDim.x * blockIdx.x + threadIdx.x; tid < num_aggregates; tid += gridDim.x * blockDim.x)
    {
        // Initialize to zero
#pragma unroll
        for (int m = 0; m < bsize; m++)
        {
            rr_temp[m] = types::util<ValueType>::get_zero();
        }

        jmin = row_offsets[tid];
        jmax = row_offsets[tid + 1];

        for (int j = jmin; j < jmax; j++)
        {
            int jcol = column_indices[j];
            offset = jcol * bsize;
#pragma unroll

            for (int m = 0; m < bsize; m++)
            {
                rr_temp[m] = rr_temp[m] + r[offset + m];
            }
        }

        offset = tid * bsize;
#pragma unroll

        for (int m = 0; m < bsize; m++)
        {
            rr[offset + m] = rr_temp[m];
        };
    }
}

// Kernel to prolongate and apply the correction for csr format
template <typename IndexType, typename ValueType>
__global__
void prolongateAndApplyCorrectionKernel(const ValueType alpha, const int num_rows, ValueType *x, const ValueType *e, const IndexType *aggregates, IndexType num_aggregates)
{
    for (int tid = blockDim.x * blockIdx.x + threadIdx.x; tid < num_rows; tid += gridDim.x * blockDim.x)
    {
        IndexType I = aggregates[tid];
        x[tid] = x[tid] + alpha * e[I];
    }
}

// Kernel to prolongate and apply the correction for block-dia-csr format
template <typename IndexType, typename ValueType>
__global__
void prolongateAndApplyCorrectionBlockDiaCsrKernel(const ValueType alpha, const int num_block_rows, ValueType *x, const ValueType *e, const IndexType *aggregates, IndexType num_aggregates, const int bsize)
{
    for (int tid = blockDim.x * blockIdx.x + threadIdx.x; tid < num_block_rows; tid += gridDim.x * blockDim.x)
    {
        IndexType I = aggregates[tid];

        for (int  m = 0; m < bsize; m++)
        {
            x[tid * bsize + m] = x[tid * bsize + m] + alpha * e[I * bsize + m];
        }
    }
}

template <typename IndexType, typename ValueType>
__global__
void prolongateVector(const IndexType *aggregates, const ValueType *in, ValueType *out, IndexType fine_rows, IndexType coarse_rows, int blocksize)
{
    int tid = threadIdx.x + blockDim.x * blockIdx.x;

    while ( tid < fine_rows * blocksize )
    {
        int i = tid / blocksize;
        int e = tid % blocksize;
        IndexType I = aggregates[i];
        out[tid] = in[ I * blocksize + e ];
        tid += gridDim.x * blockDim.x;
    }
}

template <typename IndexType, typename ValueType>
__global__
void applyCorrection(ValueType lambda, const ValueType *e, ValueType *x, IndexType numRows )
{
    int tid = threadIdx.x + blockDim.x * blockIdx.x;

    while ( tid < numRows )
    {
        x[tid] = x[tid] + lambda * e[tid];
        tid += gridDim.x * blockDim.x;
    }
}

// -------------------------------
//  Methods
// ------------------------------

template <class T_Config>
void Aggregation_AMG_Level_Base<T_Config>::transfer_level(AMG_Level<TConfig1> *ref_lvl)
{
    Aggregation_AMG_Level_Base<TConfig1> *ref_agg_lvl = dynamic_cast<Aggregation_AMG_Level_Base<TConfig1>*>(ref_lvl);
    this->scale_counter = ref_agg_lvl->scale_counter;
    this->scale = ref_agg_lvl->scale;
    this->m_R_row_offsets.copy(ref_agg_lvl->m_R_row_offsets);
    this->m_R_column_indices.copy(ref_agg_lvl->m_R_column_indices);
    this->m_aggregates.copy(ref_agg_lvl->m_aggregates);
    this->m_aggregates_fine_idx.copy(ref_agg_lvl->m_aggregates_fine_idx);
    this->m_num_aggregates = ref_agg_lvl->m_num_aggregates;
    this->m_num_all_aggregates = ref_agg_lvl->m_num_all_aggregates;
}


typedef std::pair<int, int> mypair;
bool comparator ( const mypair &l, const mypair &r) { return l.first < r.first; }

// Method to compute R
// General path
// TODO: this could be merged with selector to save some computations
template <typename T_Config>
void Aggregation_AMG_Level_Base<T_Config>::computeRestrictionOperator_common()
{
    m_R_row_offsets.resize(m_num_all_aggregates + 1); //create one more row for the pseudo aggregate
    IVector R_row_indices(m_aggregates);
#if AMGX_ASYNCCPU_PROOF_OF_CONCEPT
    bool use_cpu = m_aggregates.size() < 4096;

    if (use_cpu)
    {
        struct computeRestrictionTask : public task
        {
            Aggregation_AMG_Level_Base<T_Config> *self;
            IVector *R_row_indices;

            void run()
            {
                int N = self->m_aggregates.size();
                IVector_h R_row_indices_host(self->m_aggregates);
                std::vector<mypair> pairs(N);

                for (int i = 0; i < N; i++)
                {
                    pairs[i].first = R_row_indices_host[i];
                    pairs[i].second = i;
                }

                std::stable_sort(pairs.begin(), pairs.end(), comparator);
                IVector_h R_column_indices(self->A->get_num_rows());

                for (int i = 0; i < N; i++)
                {
                    R_column_indices[i] = pairs[i].second;
                    R_row_indices_host[i] = pairs[i].first;
                }

                self->m_R_column_indices = R_column_indices;
                *R_row_indices = R_row_indices_host;
            }
        };
        computeRestrictionTask *t = new computeRestrictionTask();
        t->self = this;
        t->R_row_indices = &R_row_indices;
        t->run();
        delete t;
    }
    else
#endif
    {
        m_R_column_indices.resize(this->A->get_num_rows());
        thrust_wrapper::sequence<TConfig::memSpace>(m_R_column_indices.begin(), m_R_column_indices.end());
        cudaCheckError();
        amgx::thrust::sort_by_key(R_row_indices.begin(), R_row_indices.end(), m_R_column_indices.begin());
        cudaCheckError();
    }

    amgx::thrust::lower_bound(R_row_indices.begin(),
                        R_row_indices.end(),
                        amgx::thrust::counting_iterator<typename IVector::value_type>(0),
                        amgx::thrust::counting_iterator<typename IVector::value_type>(m_R_row_offsets.size()),
                        m_R_row_offsets.begin());
    cudaCheckError();
}


// two methods below could be merged
// Method to compute R on HOST using csr format
template <AMGX_VecPrecision t_vecPrec, AMGX_MatPrecision t_matPrec, AMGX_IndPrecision t_indPrec>
void Aggregation_AMG_Level<TemplateConfig<AMGX_host, t_vecPrec, t_matPrec, t_indPrec> >::computeRestrictionOperator_1x1()
{
    this->m_R_row_offsets.resize(this->m_num_all_aggregates + 1);
    this->m_R_column_indices.resize(this->A->get_num_rows());
    this->fillRowOffsetsAndColIndices(this->A->get_num_rows());
}

// Method to compute R on HOST using block dia-csr format
template <AMGX_VecPrecision t_vecPrec, AMGX_MatPrecision t_matPrec, AMGX_IndPrecision t_indPrec>
void Aggregation_AMG_Level<TemplateConfig<AMGX_host, t_vecPrec, t_matPrec, t_indPrec> >::computeRestrictionOperator_4x4()
{
    this->m_R_row_offsets.resize(this->m_num_all_aggregates + 1);
    this->m_R_column_indices.resize(this->A->get_num_rows());
    this->fillRowOffsetsAndColIndices(this->A->get_num_rows());
}

// Method to create R_row_offsest and R_column_indices array on HOST using csr or block dia-csr format
template <typename T_Config>
void Aggregation_AMG_Level_Base<T_Config>::fillRowOffsetsAndColIndices(const int R_num_cols)
{
    for (int i = 0; i < m_num_all_aggregates + 1; i++)
    {
        m_R_row_offsets[i] = 0;
    }

    // Count number of neighbors for each row
    for (int i = 0; i < R_num_cols; i++)
    {
        int I = m_aggregates[i];
        m_R_row_offsets[I]++;
    }

    m_R_row_offsets[m_num_all_aggregates] = R_num_cols;

    for (int i = m_num_all_aggregates - 1; i >= 0; i--)
    {
        m_R_row_offsets[i] = m_R_row_offsets[i + 1] - m_R_row_offsets[i];
    }

    /* Set column indices. */
    for (int i = 0; i < R_num_cols; i++)
    {
        int I = m_aggregates[i];
        int Ip = m_R_row_offsets[I]++;
        m_R_column_indices[Ip] = i;
    }

    /* Reset r[i] to start of row memory. */
    for (int i = m_num_all_aggregates - 1; i > 0; i--)
    {
        m_R_row_offsets[i] = m_R_row_offsets[i - 1];
    }

    m_R_row_offsets[0] = 0;
}

// Method to compute R on DEVICE using block dia-csr format
template <AMGX_VecPrecision t_vecPrec, AMGX_MatPrecision t_matPrec, AMGX_IndPrecision t_indPrec>
void Aggregation_AMG_Level<TemplateConfig<AMGX_device, t_vecPrec, t_matPrec, t_indPrec> >::computeRestrictionOperator_4x4()
{
    this->computeRestrictionOperator_common();
}

template <AMGX_VecPrecision t_vecPrec, AMGX_MatPrecision t_matPrec, AMGX_IndPrecision t_indPrec>
void Aggregation_AMG_Level<TemplateConfig<AMGX_device, t_vecPrec, t_matPrec, t_indPrec> >::computeRestrictionOperator_1x1()
{
    this->computeRestrictionOperator_common();
}

// Method to restrict Residual on host using csr_matrix format
template <AMGX_VecPrecision t_vecPrec, AMGX_MatPrecision t_matPrec, AMGX_IndPrecision t_indPrec>
void Aggregation_AMG_Level<TemplateConfig<AMGX_host, t_vecPrec, t_matPrec, t_indPrec> >::restrictResidual_1x1(const VVector &r, VVector &rr)
{
    ValueTypeB temp;

    for (int i = 0; i < this->m_num_aggregates; i++)
    {
        temp = types::util<ValueTypeB>::get_zero();

        for (int j = this->m_R_row_offsets[i]; j < this->m_R_row_offsets[i + 1]; j++)
        {
            int j_col = this->m_R_column_indices[j];
            temp = temp + r[j_col];
        }

        rr[i] = temp;
    }
}

// Method to restrict Residual on host using block_dia_csr_matrix format
template <AMGX_VecPrecision t_vecPrec, AMGX_MatPrecision t_matPrec, AMGX_IndPrecision t_indPrec>
void Aggregation_AMG_Level<TemplateConfig<AMGX_host, t_vecPrec, t_matPrec, t_indPrec> >::restrictResidual_4x4(const VVector &r, VVector &rr)
{
    IndexType bsize = this->A->get_block_dimy();
    ValueTypeB *temp = new ValueTypeB[bsize];

    for (int i = 0; i < this->m_num_aggregates; i++)
    {
        // Initialize temp to 0
        for (int k = 0; k < bsize; k++)
        {
            temp[k]  =  types::util<ValueTypeB>::get_zero();
        }

        // Add contributions from each fine point
        for (int j = this->m_R_row_offsets[i]; j < this->m_R_row_offsets[i + 1]; j++)
        {
            int j_col = this->m_R_column_indices[j];

            for (int k = 0; k < bsize; k++)
            {
                temp[k] = temp[k] + r[j_col * bsize + k];
            }
        }

        // Store result
        for (int k = 0; k < bsize; k++)
        {
            rr[i * bsize + k] = temp[k];
        }
    }
}

// Method to restrict Residual on device using csr_matrix format
template <AMGX_VecPrecision t_vecPrec, AMGX_MatPrecision t_matPrec, AMGX_IndPrecision t_indPrec>
void Aggregation_AMG_Level<TemplateConfig<AMGX_device, t_vecPrec, t_matPrec, t_indPrec> >::restrictResidual_1x1(const VVector &r, VVector &rr)
{
    int block_size = 128;
    int max_threads;

    if (!this->isConsolidationLevel())
    {
        max_threads = this->m_num_aggregates;
    }
    else
    {
        max_threads = this->m_num_all_aggregates;
    }

    int num_blocks = max_threads / block_size + 1;
    const IndexType *R_row_offsets_ptr = this->m_R_row_offsets.raw();
    const IndexType *R_column_indices_ptr = this->m_R_column_indices.raw();
    const ValueTypeB *r_ptr = r.raw();
    ValueTypeB *rr_ptr = rr.raw();
    restrictResidualKernel <<< num_blocks, block_size>>>(R_row_offsets_ptr, R_column_indices_ptr, r_ptr, rr_ptr, max_threads);
    cudaCheckError();
}

// Method to restrict Residual on device using block_dia_csr_matrix format
template <AMGX_VecPrecision t_vecPrec, AMGX_MatPrecision t_matPrec, AMGX_IndPrecision t_indPrec>
void Aggregation_AMG_Level<TemplateConfig<AMGX_device, t_vecPrec, t_matPrec, t_indPrec> >::restrictResidual_4x4(const VVector &r, VVector &rr)
{
    int block_size = 128;
    int max_threads;

    if (!this->isConsolidationLevel())
    {
        max_threads = this->m_num_aggregates;
    }
    else
    {
        max_threads = this->m_num_all_aggregates;
    };

    const int num_blocks = max_threads / block_size + 1;
    const IndexType *R_row_offsets_ptr = this->m_R_row_offsets.raw();

    const IndexType *R_column_indices_ptr = this->m_R_column_indices.raw();

    const ValueTypeB *r_ptr = r.raw();

    ValueTypeB *rr_ptr = rr.raw();

    cudaCheckError();

    switch ( this->getA().get_block_dimy() )
    {
        case 2:
            restrictResidualBlockDiaCsrKernel<IndexType, ValueTypeB, 2> <<< num_blocks, block_size>>>(R_row_offsets_ptr, R_column_indices_ptr, r_ptr, rr_ptr, max_threads);
            break;

        case 3:
            restrictResidualBlockDiaCsrKernel<IndexType, ValueTypeB, 3> <<< num_blocks, block_size>>>(R_row_offsets_ptr, R_column_indices_ptr, r_ptr, rr_ptr, max_threads);
            break;

        case 4:
            restrictResidualBlockDiaCsrKernel<IndexType, ValueTypeB, 4> <<< num_blocks, block_size>>>(R_row_offsets_ptr, R_column_indices_ptr, r_ptr, rr_ptr, max_threads);
            break;

        case 5:
            restrictResidualBlockDiaCsrKernel<IndexType, ValueTypeB, 5> <<< num_blocks, block_size>>>(R_row_offsets_ptr, R_column_indices_ptr, r_ptr, rr_ptr, max_threads);
            break;

        case 8:
            restrictResidualBlockDiaCsrKernel<IndexType, ValueTypeB, 8> <<< num_blocks, block_size>>>(R_row_offsets_ptr, R_column_indices_ptr, r_ptr, rr_ptr, max_threads);
            break;

        case 10:
            restrictResidualBlockDiaCsrKernel<IndexType, ValueTypeB, 10> <<< num_blocks, block_size>>>(R_row_offsets_ptr, R_column_indices_ptr, r_ptr, rr_ptr, max_threads);
            break;

        default:
            FatalError( "Unsupported block size in restrictResidual_4x4!!!", AMGX_ERR_NOT_SUPPORTED_BLOCKSIZE );
    }

    cudaCheckError();
}

__inline__ float getAlpha(float &nom, float &denom)
{
    float alpha;

    if (nom * denom <= 0. || std::abs(nom) < std::abs(denom))
    {
        alpha = 1.;
    }
    else if (std::abs(nom) > 2.*std::abs(denom))
    {
        alpha = 2.;
    }
    else
    {
        alpha = nom / denom;
    }

    return alpha;
}

__inline__ double getAlpha(double &nom, double &denom)
{
    double alpha;

    if (nom * denom <= 0. || std::abs(nom) < std::abs(denom))
    {
        alpha = 1.;
    }
    else if (std::abs(nom) > 2.*std::abs(denom))
    {
        alpha = 2.;
    }
    else
    {
        alpha = nom / denom;
    }

    return alpha;
}

__inline__ cuComplex getAlpha(cuComplex &nom, cuComplex &denom)
{
    cuComplex alpha;

    if (types::util<cuComplex>::abs(nom) < types::util<cuComplex>::abs(denom))
    {
        alpha = make_cuComplex(1.f, 0.f);
    }
    else if (types::util<cuComplex>::abs(nom) > 2.*types::util<cuComplex>::abs(denom))
    {
        alpha = make_cuComplex(2.f, 0.f);
    }
    else
    {
        alpha = nom / denom;
    }

    return alpha;
}

__inline__ cuDoubleComplex getAlpha(cuDoubleComplex &nom, cuDoubleComplex &denom)
{
    cuDoubleComplex alpha;

    if (types::util<cuDoubleComplex>::abs(nom) < types::util<cuDoubleComplex>::abs(denom))
    {
        alpha = make_cuDoubleComplex(1., 0.);
    }
    else if (types::util<cuDoubleComplex>::abs(nom) > 2.*types::util<cuDoubleComplex>::abs(denom))
    {
        alpha = make_cuDoubleComplex(2., 0.);
    }
    else
    {
        alpha = nom / denom;
    }

    return alpha;
}

template< class T_Config>
typename T_Config::VecPrec Aggregation_AMG_Level_Base<T_Config>::computeAlpha(const Vector<T_Config> &e, const Vector<T_Config> &bc, const Vector<T_Config> &tmp)
{
    typename T_Config::VecPrec alpha =  types::util<ValueTypeB>::get_one();
    Matrix<TConfig> &Ac = this->getNextLevel( MemorySpace( ) )->getA();
    int size = Ac.get_num_rows();
    VVector v(2,  types::util<ValueTypeB>::get_zero());
    v[0] = amgx::thrust::inner_product(e.begin(), e.begin() + size, bc.begin(),  types::util<ValueTypeB>::get_zero());
    v[1] = amgx::thrust::inner_product(e.begin(), e.begin() + size, tmp.begin(),  types::util<ValueTypeB>::get_zero());
    cudaCheckError();
    return getAlpha(v[0], v[1]);
}

// Method to prolongate the error on HOST using csr format
template <AMGX_VecPrecision t_vecPrec, AMGX_MatPrecision t_matPrec, AMGX_IndPrecision t_indPrec>
void Aggregation_AMG_Level<TemplateConfig<AMGX_host, t_vecPrec, t_matPrec, t_indPrec>  >::prolongateAndApplyCorrection_1x1(Vector<TemplateConfig<AMGX_host, t_vecPrec, t_matPrec, t_indPrec> > &e, Vector<TemplateConfig<AMGX_host, t_vecPrec, t_matPrec, t_indPrec> > &bc, Vector<TemplateConfig<AMGX_host, t_vecPrec, t_matPrec, t_indPrec> > &x, Vector<TemplateConfig<AMGX_host, t_vecPrec, t_matPrec, t_indPrec> > &tmp)
{
    Matrix<TemplateConfig<AMGX_host, t_vecPrec, t_matPrec, t_indPrec> > &A = this->getA();
    Matrix<TemplateConfig<AMGX_host, t_vecPrec, t_matPrec, t_indPrec> > &C = this->next_h->getA();

    if ( this->m_error_scaling >= 2 )
    {
        FatalError("error_scaling=2,3 is not implemented on host", AMGX_ERR_NOT_IMPLEMENTED );
    }

    ValueTypeB alpha = types::util<ValueTypeB>::get_one();

    if (this->m_error_scaling)
    {
        multiply(this->next_h->getA(), e, tmp);
        alpha = this->computeAlpha (e, bc, tmp);
    }

    // Apply correction on all (interior and exterior) equations.
    for (int i = 0; i < A.get_num_cols(); i++)
    {
        int I = this->m_aggregates[i];
        x[i] = x[i] + alpha * e[I];
    }
}

// Method to prolongate the error on HOST using block_dia_csr format
template <AMGX_VecPrecision t_vecPrec, AMGX_MatPrecision t_matPrec, AMGX_IndPrecision t_indPrec>
void Aggregation_AMG_Level<TemplateConfig<AMGX_host, t_vecPrec, t_matPrec, t_indPrec> >::prolongateAndApplyCorrection_4x4(Vector<TemplateConfig<AMGX_host, t_vecPrec, t_matPrec, t_indPrec> > &e, Vector<TemplateConfig<AMGX_host, t_vecPrec, t_matPrec, t_indPrec> > &bc, Vector<TemplateConfig<AMGX_host, t_vecPrec, t_matPrec, t_indPrec> > &x, Vector<TemplateConfig<AMGX_host, t_vecPrec, t_matPrec, t_indPrec> > &tmp)
{
    if (this->A->get_block_dimy() != this->A->get_block_dimx())
    {
        FatalError("Aggregation_AMG_Level not implemented for non square blocks, exiting", AMGX_ERR_NOT_SUPPORTED_BLOCKSIZE);
    }

    if ( this->m_error_scaling >= 2 )
    {
        FatalError("error_scaling=2,3 is not implemented on host", AMGX_ERR_NOT_IMPLEMENTED );
    }

    Matrix<TConfig> &C = this->next_h->getA();
    ValueTypeB alpha = types::util<ValueTypeB>::get_one();

    if (this->m_error_scaling)
    {
        multiply(this->next_h->getA(), e, tmp);
        alpha = this->computeAlpha (e, bc, tmp);
    }

    // Apply correction on all equations.
    for (int i = 0; i < this->A->get_num_rows(); i++)
    {
        int I = this->m_aggregates[i];

        for (int k = 0; k < this->A->get_block_dimy(); k++)
        {
            x[i * this->A->get_block_dimy() + k] =  x[i * this->A->get_block_dimy() + k] + alpha * e[I * this->A->get_block_dimy() + k];
        }
    }
}

// Prolongate the error on DEVICE using csr format
template <AMGX_VecPrecision t_vecPrec, AMGX_MatPrecision t_matPrec, AMGX_IndPrecision t_indPrec>
void Aggregation_AMG_Level<TemplateConfig<AMGX_device, t_vecPrec, t_matPrec, t_indPrec> >::prolongateAndApplyCorrection_1x1(Vector<TemplateConfig<AMGX_device, t_vecPrec, t_matPrec, t_indPrec> > &e, Vector<TemplateConfig<AMGX_device, t_vecPrec, t_matPrec, t_indPrec> > &bc, Vector<TemplateConfig<AMGX_device, t_vecPrec, t_matPrec, t_indPrec> > &x, Vector<TemplateConfig<AMGX_device, t_vecPrec, t_matPrec, t_indPrec> > &tmp)
{
    ValueTypeB alpha = types::util<ValueTypeB>::get_one();
    const int block_size = 128;
    const int num_blocks = this->A->get_num_rows() / block_size + 1;
    const IndexType *aggregates_ptr = this->m_aggregates.raw();
    ValueTypeB *x_ptr = x.raw();
    const ValueTypeB *e_ptr = e.raw();

    if (this->m_error_scaling)
    {
        FatalError("error_scaling=1 is deprecated", AMGX_ERR_NOT_IMPLEMENTED );
    }

    prolongateAndApplyCorrectionKernel <<< num_blocks, block_size>>>(alpha, (int)this->A->get_num_rows(), x_ptr, e_ptr, aggregates_ptr, this->m_num_aggregates);
    cudaCheckError();
}

// Prolongate the error on DEVICE using block dia-csr format
template <AMGX_VecPrecision t_vecPrec, AMGX_MatPrecision t_matPrec, AMGX_IndPrecision t_indPrec>
void Aggregation_AMG_Level<TemplateConfig<AMGX_device, t_vecPrec, t_matPrec, t_indPrec> >::prolongateAndApplyCorrection_4x4(Vector<TemplateConfig<AMGX_device, t_vecPrec, t_matPrec, t_indPrec> > &ec,
        Vector<TemplateConfig<AMGX_device, t_vecPrec, t_matPrec, t_indPrec> > &bf,
        Vector<TemplateConfig<AMGX_device, t_vecPrec, t_matPrec, t_indPrec> > &xf,
        Vector<TemplateConfig<AMGX_device, t_vecPrec, t_matPrec, t_indPrec> > &rf)
{
    if ( this->m_error_scaling >= 2 )
    {
        if ( this->scale_counter > 0 )
        {
            const IndexType *aggregates_ptr = this->m_aggregates.raw();
            ValueTypeB *x_ptr = xf.raw();
            const ValueTypeB *e_ptr = ec.raw();
            const int block_size = 128;
            const int num_blocks = (this->A->get_num_rows() - 1) / block_size + 1;
            prolongateAndApplyCorrectionBlockDiaCsrKernel <<< num_blocks, block_size>>>(this->scale, (int)this->getA().get_num_rows(), x_ptr, e_ptr, aggregates_ptr, this->m_num_aggregates, this->getA().get_block_dimy());
            cudaCheckError();
            this->scale_counter--;
            return;
        }

        bool vanek_scaling = this->m_error_scaling > 3;
        IndexType numRowsCoarse = this->next_d->getA().get_num_rows();
        IndexType numRowsFine = this->A->get_num_rows();
        IndexType blockdim = this->A->get_block_dimx();

        if ( blockdim != this->A->get_block_dimy() )
        {
            FatalError("Unsupported dimension for aggregation amg level", AMGX_ERR_NOT_SUPPORTED_BLOCKSIZE);
        }

        VVector ef( rf.size() );
        VVector Aef( rf.size() );
        ef.set_block_dimy( blockdim );
        Aef.set_block_dimy( blockdim );
        // prolongate e
        const int threads_per_block = 256;
        const int num_block_values = std::min( AMGX_GRID_MAX_SIZE, (numRowsFine * blockdim - 1) / threads_per_block + 1);
        const cudaStream_t stream = nullptr;
        prolongateVector <<< num_block_values, threads_per_block, 0, stream>>>( this->m_aggregates.raw(), ec.raw(), ef.raw(), numRowsFine, numRowsCoarse, blockdim );
        ef.dirtybit = 1;
        cudaStreamSynchronize(stream);
        cudaCheckError();
        int preSmooth;

        if ( vanek_scaling )
        {
            preSmooth = this->amg->getNumPostsweeps();
        }
        else
        {
            preSmooth = this->scaling_smoother_steps;
        }

        //smooth error
        this->smoother->setTolerance( 0.0 );
        this->smoother->set_max_iters( preSmooth );

        if ( vanek_scaling )
        {
            thrust_wrapper::fill<TConfig::memSpace>( Aef.begin(), Aef.end(), types::util<ValueTypeB>::get_zero() );
            cudaCheckError();
            this->smoother->solve( Aef, ef, false ); //smooth correction with rhs 0
            this->smoother->solve( bf, xf, false ); // smooth x with rhs residual
            //recompute residual
            int offset, size;
            this->getA().getOffsetAndSizeForView(OWNED, &offset, &size);
            axmb( this->getA(), xf, bf, rf, offset, size );
        }
        else
        {
            this->smoother->solve( rf, ef, false ); //smooth correction with rhs residual
        }

        // multiply for lambda computation
        multiply(this->getA(), ef, Aef, OWNED);
        ValueTypeB nominator, denominator;
        int offset = 0, size = 0;
        this->A->getOffsetAndSizeForView(OWNED, &offset, &size);

        if ( this->m_error_scaling == 2 || this->m_error_scaling == 4 )
        {
            // compute lambda=<rf,Aef>/<Aef,Aef>
            nominator = amgx::thrust::inner_product( rf.begin(), rf.end(), Aef.begin(), types::util<ValueTypeB>::get_zero() );
            denominator = amgx::thrust::inner_product( Aef.begin(), Aef.end(), Aef.begin(), types::util<ValueTypeB>::get_zero() );
            cudaCheckError();
        }

        if ( this->m_error_scaling == 3 || this->m_error_scaling == 5)
        {
            // compute lambda=<rf,ef>/<ef,Aef>
            nominator = amgx::thrust::inner_product( rf.begin(), rf.begin() + size * blockdim, ef.begin(), types::util<ValueTypeB>::get_zero() );
            denominator = amgx::thrust::inner_product( ef.begin(), ef.begin() + size * blockdim, Aef.begin(), types::util<ValueTypeB>::get_zero() );

            if (!this->A->is_matrix_singleGPU())
            {
                this->A->getManager()->global_reduce_sum(&nominator);
                this->A->getManager()->global_reduce_sum(&denominator);
            }

            cudaCheckError();
        }

        if (types::util<ValueTypeB>::abs(denominator) == 0.0)
        {
            nominator = denominator = types::util<ValueTypeB>::get_one();
        }

        // apply correction x <- x + lambda*e
        const int num_block_fine = std::min( AMGX_GRID_MAX_SIZE, (numRowsFine * blockdim - 1) / threads_per_block + 1 );
        ValueTypeB alpha = nominator / denominator;

        if ( types::util<ValueTypeB>::abs(alpha) < .3 )
        {
            alpha = (alpha / types::util<ValueTypeB>::abs(alpha)) * .3;    // it was this before: alpha = .3, which is not 100% equal
        }

        if ( types::util<ValueTypeB>::abs(alpha) > 10 )
        {
            alpha = (alpha / types::util<ValueTypeB>::abs(alpha)) * 10.;    // it was this before: alpha = 10., which is not 100% equal
        }

        applyCorrection <<< num_block_fine, threads_per_block, 0, stream>>>( alpha, ef.raw(), xf.raw(), numRowsFine * blockdim );
        cudaCheckError();
        this->scale_counter = this->reuse_scale; //reuse this scale scale_counter times
        this->scale = alpha;
        return;
    }

    ValueTypeB alpha = types::util<ValueTypeB>::get_one();
    const int block_size = 128;
    const int num_blocks = this->A->get_num_rows() / block_size + 1;
    const IndexType *aggregates_ptr = this->m_aggregates.raw();
    ValueTypeB *x_ptr = xf.raw();
    const ValueTypeB *e_ptr = ec.raw();

    if (this->m_error_scaling == 1)
    {
        FatalError("error_scaling=1 is deprecated", AMGX_ERR_NOT_IMPLEMENTED );
    }

    prolongateAndApplyCorrectionBlockDiaCsrKernel <<< num_blocks, block_size>>>(alpha, (int)this->A->get_num_rows(), x_ptr, e_ptr, aggregates_ptr, this->m_num_aggregates, this->A->get_block_dimy());
    cudaCheckError();
}

template <class T_Config>
void Aggregation_AMG_Level_Base<T_Config >::prolongateAndApplyCorrection(VVector &e, VVector &bf, VVector &x, VVector &tmp)
{
    Matrix<TConfig> &Ac = this->getNextLevel( MemorySpace( ) )->getA();

    //this is dirty, but error scaling 2 and 3 do not have a specialized version. Instead, the general version sits in the 4x4 function
    if ( this->m_error_scaling >= 2 )
    {
        prolongateAndApplyCorrection_4x4(e, bf, x, tmp);
    }
    else if (this->A->get_block_size() == 1)
    {
        prolongateAndApplyCorrection_1x1(e, bf, x, tmp);
    }
    else if (this->A->get_block_dimx() == this->A->get_block_dimy() )
    {
        prolongateAndApplyCorrection_4x4(e, bf, x, tmp);
    }
    else
    {
        FatalError("Unsupported dimension for aggregation amg level", AMGX_ERR_NOT_SUPPORTED_BLOCKSIZE);
    }

    x.dirtybit = 1;

    if (!this->A->is_matrix_singleGPU() && x.delayed_send == 0)
    {
        if (x.in_transfer & RECEIVING) { this->A->manager->exchange_halo_wait(x, x.tag); }

        this->A->manager->exchange_halo_async(x, x.tag);
    }
}


template <class T_Config>
void Aggregation_AMG_Level_Base<T_Config>::restrictResidual(VVector &r, VVector &rr)
{
    if (this->A->get_block_size() == 1)
    {
        restrictResidual_1x1(r, rr);
    }
    else if (this->A->get_block_dimx() == this->A->get_block_dimy() )
    {
        restrictResidual_4x4(r, rr);
    }
    else
    {
        FatalError("Unsupported dimension for aggregation amg level", AMGX_ERR_NOT_SUPPORTED_BLOCKSIZE);
    }

    //TODO: check level transfer between host and device for multiGPU
    if (!this->A->is_matrix_singleGPU())
    {
        Matrix<TConfig> &Ac = this->getNextLevel( MemorySpace( ) )->getA();
        rr.dirtybit = 1;

        if (!Ac.is_matrix_singleGPU() && !this->isConsolidationLevel() && rr.delayed_send == 0)
        {
            Matrix<TConfig> &Ac = this->getNextLevel( MemorySpace( ) )->getA(); //TODO problem in memoryspace transfer is here

            if (rr.in_transfer & RECEIVING) { Ac.manager->exchange_halo_wait(rr, rr.tag); }

            Ac.manager->exchange_halo_async(rr, rr.tag);
        }
    }
}

template <class T_Config>
void Aggregation_AMG_Level_Base<T_Config>::computeRestrictionOperator()
{
    if (this->A->get_block_size() == 1)
    {
        computeRestrictionOperator_1x1();
    }
    else if (this->A->get_block_dimx() == 4 && this->A->get_block_dimy() == 4)
    {
        computeRestrictionOperator_4x4();
    }
    else
    {
        this->computeRestrictionOperator_common();
    }
}

template <typename IndexType>
__global__ void coarse_to_global(IndexType *aggregates, IndexType *aggregates_global, IndexType *renumbering, IndexType num_elements, int64_t offset)
{
    int element = blockIdx.x * blockDim.x + threadIdx.x;

    while (element < num_elements)
    {
        renumbering[aggregates[element]] = aggregates_global[element] + offset; //this won't be a problem, because we are overwriting the same thing
        element += blockDim.x * gridDim.x;
    }
}

template <typename T, typename IndexType>
__global__ void export_matrix_elements(IndexType *row_offsets, IndexType *col_indices, T *values, IndexType *maps, IndexType *renumbering, IndexType *new_row_offsets, IndexType *new_col_indices, T *new_values, IndexType bsize, IndexType size)
{
    int idx = blockIdx.x * blockDim.x / 32 + threadIdx.x / 32;
    int coopIdx = threadIdx.x % 32;

    while (idx < size)
    {
        int row = maps[idx];
        INDEX_TYPE src_base = row_offsets[row];
        INDEX_TYPE dst_base = new_row_offsets[idx];

        for (int m = coopIdx; m < row_offsets[row + 1]*bsize - src_base * bsize; m += 32)
        {
            new_values[dst_base * bsize + m] = values[src_base * bsize + m];
        }

        for (int m = coopIdx; m < row_offsets[row + 1] - src_base; m += 32)
        {
            new_col_indices[dst_base + m] = renumbering[col_indices[src_base + m]];
        }

        idx += gridDim.x * blockDim.x / 32;
    }
}

template <class T>
__global__ void export_matrix_diagonal(T *values, INDEX_TYPE bsize, INDEX_TYPE *maps, T *output, INDEX_TYPE size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    while (idx < size)
    {
        int row = maps[idx];
        INDEX_TYPE src_base = row;
        INDEX_TYPE dst_base = idx;

        for (int m = 0; m < bsize; m++)
        {
            output[dst_base * bsize + m] = values[src_base * bsize + m];
        }

        idx += gridDim.x * blockDim.x;
    }
}

__global__ void remove_boundary(INDEX_TYPE *flags, INDEX_TYPE *maps, INDEX_TYPE size)
{
    int element = blockIdx.x * blockDim.x + threadIdx.x;

    while (element < size)
    {
        flags[maps[element]] = 0; //this won't be a problem, because we are overwriting the same thing
        element += blockDim.x * gridDim.x;
    }
}

__global__ void calc_inverse_renumbering(INDEX_TYPE *renum, INDEX_TYPE *irenum, INDEX_TYPE *renum_gbl, INDEX_TYPE base_index, INDEX_TYPE max_element)
{
    int idx = blockDim.x * blockIdx.x + threadIdx.x;

    while (idx < max_element)
    {
        irenum[renum[idx]] = renum_gbl[idx] - base_index;
        idx += blockDim.x * gridDim.x;
    }
}

__global__ void create_halo_mapping(INDEX_TYPE *mapping, INDEX_TYPE *node_list, INDEX_TYPE base_index, INDEX_TYPE map_offset, INDEX_TYPE size)
{
    int row = blockIdx.x * blockDim.x + threadIdx.x;

    while (row < size)
    {
        int idx = node_list[row] - base_index;
        mapping[idx] = map_offset + row;
        row += blockDim.x * gridDim.x;
    }
}

__global__ void map_col_indices_and_count_rowlen(INDEX_TYPE *row_offsets, INDEX_TYPE *col_indices, INDEX_TYPE *row_length, INDEX_TYPE *renumbering, INDEX_TYPE *mapping, INDEX_TYPE *map_offsets, int64_t *index_ranges, INDEX_TYPE part_id, INDEX_TYPE my_id, INDEX_TYPE base_index, INDEX_TYPE my_range, INDEX_TYPE num_neighbors, INDEX_TYPE num_rows)
{
    extern __shared__ volatile int reduction[];
    int row = blockIdx.x * blockDim.x / 4 + threadIdx.x / 4;
    int coopIdx = threadIdx.x % 4;

    while (row < num_rows)
    {
        int valid = 0;

        for (int idx = row_offsets[row] + coopIdx; idx < row_offsets[row + 1]; idx += 4) //this may look horrible, but I expect low branch divergence, because col indices in a row usually belong to the same partition (or at most one more)
        {
            int colIdx = col_indices[idx];
            int part = -2;

            if (colIdx >= index_ranges[2 * part_id] && colIdx < index_ranges[2 * part_id + 1]) //the col index probably belongs to the partition I am working on
            {
                part = part_id;
            }
            else if (colIdx >= base_index && colIdx < base_index + my_range)     //or points back to the owned partition
            {
                part = -1;
            }
            else        //or else it points to a third partition
            {
                for (int i = 0; i < num_neighbors; i++)
                {
                    if (colIdx >= index_ranges[2 * i] && colIdx < index_ranges[2 * i + 1])
                    {
                        part = i;
                    }
                }
            }

            if (part == -2)
            {
                col_indices[idx] = -1;
#ifdef DEBUG
                printf("Column index encountered that does not belong to any of my neighbors!! %d\n", colIdx);
#endif
            }
            else
            {
                if (part == -1)
                {
                    col_indices[idx] = renumbering[colIdx - base_index];
                    valid++;
                }
                else
                {
                    int new_col_idx = mapping[map_offsets[part] + colIdx - index_ranges[2 * part]];

                    if (new_col_idx >= 0)
                    {
                        valid++;
                        col_indices[idx] = new_col_idx;
                    }
                    else
                    {
                        col_indices[idx] = -1;
                    }
                }
            }
        }

        reduction[threadIdx.x] = valid;

        for (int s = 2; s > 0; s >>= 1)
        {
            if (coopIdx < s)
            {
                reduction[threadIdx.x] += reduction[threadIdx.x + s];
            }

            __syncthreads();
        }

        if (coopIdx == 0)
        {
            row_length[row] = reduction[threadIdx.x];
        }

        row += gridDim.x * blockDim.x / 4;
    }
}

__global__ void map_col_indices(INDEX_TYPE *row_offsets, INDEX_TYPE *col_indices, int64_t *halo_ranges, INDEX_TYPE *halo_renumbering, INDEX_TYPE *halo_rows, INDEX_TYPE *global_renumbering, INDEX_TYPE num_neighbors, INDEX_TYPE num_rows, INDEX_TYPE num_rows_processed)
{
    int row = blockIdx.x * blockDim.x / 4 + threadIdx.x / 4;
    int coopIdx = threadIdx.x % 4;

    while (row < num_rows_processed)
    {
        for (int idx = row_offsets[row] + coopIdx; idx < row_offsets[row + 1]; idx += 4)
        {
            int colIdx = col_indices[idx];
            int part = 0;

            if (colIdx < num_rows)
            {
                part = -1;
            }
            else
            {
                colIdx = global_renumbering[colIdx];

                for (int i = 0; i < num_neighbors; i++)
                {
                    if (colIdx >= halo_ranges[2 * i] && colIdx < halo_ranges[2 * i + 1])
                    {
                        part = i;
                        break;
                    }
                }
            }

            if (part == -1)
            {
                col_indices[idx] = colIdx;
            }
            else
            {
                col_indices[idx] = halo_renumbering[halo_rows[part] + colIdx - halo_ranges[2 * part]];
            }
        }

        row += gridDim.x * blockDim.x / 4;
    }
}

template <class T>
__global__ void reorder_whole_matrix(INDEX_TYPE *old_rows, INDEX_TYPE *old_cols, T *old_vals, INDEX_TYPE *rows, INDEX_TYPE *cols, T *vals, INDEX_TYPE bsize, INDEX_TYPE num_rows)
{
    int row = blockIdx.x * blockDim.x + threadIdx.x;

    while (row < num_rows)
    {
        INDEX_TYPE dst_row = row;
        INDEX_TYPE src_base = old_rows[row];
        INDEX_TYPE dst = rows[dst_row];

        for (int i = 0; i < old_rows[row + 1] - src_base; i++)
        {
            INDEX_TYPE colIdx = old_cols[src_base + i];

            if (colIdx >= 0)
            {
                cols[dst] = colIdx;

                for (int j = 0; j < bsize; j++) { vals[dst * bsize + j] = old_vals[(src_base + i) * bsize + j]; }

                dst++;
            }
        }

        row += blockDim.x * gridDim.x;
    }
}

__global__ void calc_gbl_renumbering(INDEX_TYPE *inv_renum, INDEX_TYPE *gbl_renum, INDEX_TYPE size)
{
    int idx = blockDim.x * blockIdx.x + threadIdx.x;

    while (idx < size)
    {
        gbl_renum[inv_renum[idx]] = idx;
        idx += blockDim.x * gridDim.x;
    }
}

template <typename ValueType>
__global__ void write_diagonals(ValueType *values, INDEX_TYPE *diag, INDEX_TYPE *map, ValueType *output, INDEX_TYPE bsize, INDEX_TYPE size)
{
    int nzPerBlock = blockDim.x / bsize;
    int row = blockIdx.x * nzPerBlock + threadIdx.x / bsize;
    int vecIdx = threadIdx.x % bsize;

    if (threadIdx.x >= (blockDim.x / bsize)*bsize) { return; }

    while (row < size)
    {
        output[row * bsize + vecIdx] = values[diag[map[row]] * bsize + vecIdx];
        row += gridDim.x * nzPerBlock;
    }
}

template <typename ValueType>
__global__ void write_diagonals_back(ValueType *values, INDEX_TYPE *diag, ValueType *source, INDEX_TYPE bsize, INDEX_TYPE size)
{
    int nzPerBlock = blockDim.x / bsize;
    int row = blockIdx.x * nzPerBlock + threadIdx.x / bsize;
    int vecIdx = threadIdx.x % bsize;

    if (threadIdx.x >= (blockDim.x / bsize)*bsize) { return; }

    while (row < size)
    {
        values[diag[row]*bsize + vecIdx] = source[row * bsize + vecIdx];
        row += gridDim.x * nzPerBlock;
    }
}

template <class T_Config>
void Aggregation_AMG_Level_Base<T_Config>::prepareNextLevelMatrix_full(const Matrix<TConfig> &A, Matrix<TConfig> &Ac)
{
    if (A.is_matrix_singleGPU()) { return; }

    int num_neighbors = A.manager->neighbors.size();

    if (TConfig::memSpace == AMGX_host)
    {
        FatalError("Aggregation AMG Not implemented for host", AMGX_ERR_NOT_IMPLEMENTED);
    }
    else
    {
        int c_size = Ac.get_num_rows();
        int f_size = A.get_num_rows();
        int diag = Ac.hasProps(DIAG);

        if (A.manager->B2L_rings[0].size() > 2) { FatalError("Aggregation_AMG_Level prepareNextLevelMatrix not implemented >1 halo rings", AMGX_ERR_NOT_SUPPORTED_BLOCKSIZE); }

        //get coarse -> fine global renumbering
        IVector renumbering(c_size);
        int num_blocks = std::min(4096, (c_size + 127) / 128);
        coarse_to_global <<< num_blocks, 128>>>(this->m_aggregates.raw(), this->m_aggregates_fine_idx.raw(), renumbering.raw(), f_size, 0);
        cudaCheckError();
        //
        // Step 0 - form halo matrices that are exported to neighbors
        //
        std::vector<Matrix<TConfig> > halo_rows(num_neighbors);
        std::vector<DistributedManager<TConfig> > halo_btl(num_neighbors);

        for (int i = 0; i < num_neighbors; i++ )
        {
            int num_unique = Ac.manager->B2L_rings[i][1];
            //prepare export halo matrices
            halo_btl[i].resize(1, 1);
            halo_btl[i].set_global_id(Ac.manager->global_id());
            halo_btl[i].B2L_maps[0].resize(num_unique);
            halo_btl[i].B2L_rings[0].resize(2);
            halo_btl[i].B2L_rings[0][0] = 0;
            halo_btl[i].B2L_rings[0][1] = num_unique;
            halo_btl[i].set_index_range(A.manager->index_range());
            halo_btl[i].set_base_index(A.manager->base_index());
            //global indices of rows of the halo matrix
            amgx::thrust::copy(amgx::thrust::make_permutation_iterator( renumbering.begin(), Ac.manager->B2L_maps[i].begin()),
                         amgx::thrust::make_permutation_iterator( renumbering.begin(), Ac.manager->B2L_maps[i].begin() + num_unique),
                         halo_btl[i].B2L_maps[0].begin());
            cudaCheckError();
            halo_rows[i].addProps(CSR);

            if (diag) { halo_rows[i].addProps(DIAG); }

            //calculate row length and row_offsets
            halo_rows[i].row_offsets.resize(num_unique + 1);
            thrust_wrapper::transform<TConfig::memSpace>(amgx::thrust::make_permutation_iterator(Ac.row_offsets.begin() + 1, Ac.manager->B2L_maps[i].begin()),
                              amgx::thrust::make_permutation_iterator(Ac.row_offsets.begin() + 1, Ac.manager->B2L_maps[i].end()),
                              amgx::thrust::make_permutation_iterator(Ac.row_offsets.begin(), Ac.manager->B2L_maps[i].begin()),
                              halo_rows[i].row_offsets.begin(),
                              amgx::thrust::minus<IndexType>());
            cudaCheckError();
            thrust_wrapper::exclusive_scan<TConfig::memSpace>(halo_rows[i].row_offsets.begin(), halo_rows[i].row_offsets.end(), halo_rows[i].row_offsets.begin());
            cudaCheckError();
            //resize halo matrix
            IndexType num_nz = halo_rows[i].row_offsets[num_unique];
            halo_rows[i].resize(num_unique, num_unique, num_nz, Ac.get_block_dimy(), Ac.get_block_dimx(), 1);
            //copy relevant rows and renumber their column indices
            num_blocks = std::min(4096, (num_unique + 127) / 128);
            export_matrix_elements <<< num_blocks, 128>>>(Ac.row_offsets.raw(), Ac.col_indices.raw(), Ac.values.raw(), Ac.manager->B2L_maps[i].raw(), renumbering.raw(), halo_rows[i].row_offsets.raw(), halo_rows[i].col_indices.raw(), halo_rows[i].values.raw(), A.get_block_size(), num_unique);
            cudaCheckError();

            if (diag)
            {
                export_matrix_diagonal <<< num_blocks, 128>>>(Ac.values.raw() + Ac.row_offsets[Ac.get_num_rows()]*Ac.get_block_size(), Ac.get_block_size(), Ac.manager->B2L_maps[i].raw(), halo_rows[i].values.raw() + halo_rows[i].row_offsets[halo_rows[i].get_num_rows()]*Ac.get_block_size(), num_unique);
                cudaCheckError();
            }
        }

        Ac.manager->getComms()->exchange_matrix_halo(halo_rows, halo_btl, Ac);
        //--------------------- renumbering/reordering matrix, integrating halo -----------------------------
        Ac.set_initialized(0);
        //number of owned rows
        c_size = Ac.manager->halo_offsets[0];
        f_size = A.manager->halo_offsets[0];
        num_blocks = std::min(4096, (c_size + 511) / 512);
        int rings = 1;
        //
        // Step 1 - calculate inverse renumbering (to global indices - base_index)
        //
        Ac.manager->inverse_renumbering.resize(c_size);
        thrust_wrapper::transform<TConfig::memSpace>(renumbering.begin(),
                          renumbering.begin() + c_size,
                          amgx::thrust::constant_iterator<IndexType>(A.manager->base_index()),
                          Ac.manager->inverse_renumbering.begin(),
                          amgx::thrust::minus<IndexType>());
        cudaCheckError();
        //big renumbering table for going from global index to owned local index
        IVector global_to_coarse_local(Ac.manager->index_range());
        thrust_wrapper::fill<TConfig::memSpace>(global_to_coarse_local.begin(), global_to_coarse_local.begin() + Ac.manager->index_range(), -1);
        cudaCheckError();
        calc_gbl_renumbering <<< num_blocks, 512>>>(Ac.manager->inverse_renumbering.raw(), global_to_coarse_local.raw(), c_size);
        cudaCheckError();
        Ac.manager->set_num_halo_rows(Ac.manager->halo_offsets[Ac.manager->halo_offsets.size() - 1] - c_size);
        cudaCheckError();
        //
        // Step 2 - create big mapping table of all halo indices we received (this may use a little too much memory sum(fine nodes per neighbor)
        //
        amgx::thrust::host_vector<INDEX_TYPE> neighbor_rows(num_neighbors + 1);
        int max_num_rows = 0;

        for (int i = 0; i < num_neighbors; i++)
        {
            neighbor_rows[i] = halo_rows[i].manager->index_range();
            max_num_rows = max_num_rows > halo_rows[i].get_num_rows() ? max_num_rows : halo_rows[i].get_num_rows();
        }

        thrust_wrapper::exclusive_scan<TConfig::memSpace>(neighbor_rows.begin(), neighbor_rows.end(), neighbor_rows.begin());
        cudaCheckError();
        int total_rows_of_neighbors = neighbor_rows[num_neighbors];
        IVector halo_mapping(total_rows_of_neighbors);
        thrust_wrapper::fill<TConfig::memSpace>(halo_mapping.begin(), halo_mapping.end(), -1);
        cudaCheckError();

        for (int ring = 0; ring < rings; ring++)
        {
            for (int i = 0; i < num_neighbors; i++)
            {
                int size = halo_btl[i].B2L_rings[0][ring + 1] - halo_btl[i].B2L_rings[0][ring];
                int num_blocks = std::min(4096, (size + 127) / 128);
                create_halo_mapping <<< num_blocks, 128>>>(halo_mapping.raw() + neighbor_rows[i],
                        halo_btl[i].B2L_maps[0].raw() + halo_btl[i].B2L_rings[0][ring],
                        halo_btl[i].base_index(),
                        Ac.manager->halo_offsets[ring * num_neighbors + i], size);
            }
        }

        cudaCheckError();
        //
        // Step 3 - renumber halo matrices and calculate row length (to eventually append to the big matrix)
        //
        INDEX_TYPE owned_nnz = Ac.row_offsets[c_size];
        IVector neighbor_rows_d(num_neighbors + 1);
        amgx::thrust::copy(neighbor_rows.begin(), neighbor_rows.end(), neighbor_rows_d.begin());
        cudaCheckError();
        //map column indices of my own matrix (the ones that point outward)
        map_col_indices <<< num_blocks, 512>>>(Ac.row_offsets.raw() + Ac.manager->num_interior_nodes(),
                                               Ac.col_indices.raw(),
                                               Ac.manager->halo_ranges.raw(),
                                               halo_mapping.raw(),
                                               neighbor_rows_d.raw(),
                                               renumbering.raw(),
                                               num_neighbors, c_size, c_size - Ac.manager->num_interior_nodes());
        cudaCheckError();
        IVector temp_row_len(max_num_rows);

        for (int i = 0; i < num_neighbors; i++)
        {
            //map column indices of halo matrices
            int size = halo_rows[i].get_num_rows();
            int num_blocks = std::min(4096, (size + 127) / 128);
            map_col_indices_and_count_rowlen <<< num_blocks, 128, 128 * sizeof(INDEX_TYPE)>>>(
                halo_rows[i].row_offsets.raw(),
                halo_rows[i].col_indices.raw(),
                temp_row_len.raw(),
                global_to_coarse_local.raw(),
                halo_mapping.raw(),
                neighbor_rows_d.raw(),
                Ac.manager->halo_ranges.raw(),
                i,
                Ac.manager->global_id(),
                Ac.manager->base_index(),
                Ac.manager->index_range(),
                num_neighbors,
                size);

            for (int ring = 0; ring < rings; ring++)
            {
                amgx::thrust::copy(temp_row_len.begin() + halo_btl[i].B2L_rings[0][ring], temp_row_len.begin() + halo_btl[i].B2L_rings[0][ring + 1], Ac.row_offsets.begin() + Ac.manager->halo_offsets[ring * num_neighbors + i]);
            }
        }

        cudaCheckError();
        INDEX_TYPE old_nnz = Ac.row_offsets[Ac.row_offsets.size() - 1];
        thrust_wrapper::exclusive_scan<TConfig::memSpace>(Ac.row_offsets.begin() + c_size, Ac.row_offsets.end(), Ac.row_offsets.begin() + c_size, owned_nnz);
        cudaCheckError();
        //
        // Step 4 - consolidate column indices and values
        //
        int new_nnz = Ac.row_offsets[Ac.row_offsets.size() - 1];

        Ac.col_indices.resize(new_nnz);
        Ac.values.resize((new_nnz + 1 + diag * (Ac.row_offsets.size() - 2)) * A.get_block_size());

        if (diag)
        {
            MVector diags(c_size * Ac.get_block_size());
            amgx::thrust::copy(Ac.values.begin() + old_nnz * Ac.get_block_size(),
                         Ac.values.begin() + old_nnz * Ac.get_block_size() + c_size * Ac.get_block_size(),
                         diags.begin());
            amgx::thrust::copy(diags.begin(), diags.begin() + c_size * Ac.get_block_size(),
                         Ac.values.begin() + Ac.row_offsets[Ac.get_num_rows()]*Ac.get_block_size());
            cudaCheckError();
        }

        int cumulative_num_rows = c_size;

        for (int i = 0; i < num_neighbors; i++)
        {
            for (int ring = 0; ring < rings; ring++)
            {
                int num_rows = halo_btl[i].B2L_rings[0][ring + 1] - halo_btl[i].B2L_rings[0][ring];
                int num_blocks = std::min(4096, (num_rows + 127) / 128);
                reorder_whole_matrix <<< num_blocks, 128>>>(halo_rows[i].row_offsets.raw() + halo_btl[i].B2L_rings[0][ring], halo_rows[i].col_indices.raw(), halo_rows[i].values.raw(), Ac.row_offsets.raw() + Ac.manager->halo_offsets[ring * num_neighbors + i], Ac.col_indices.raw(), Ac.values.raw(), Ac.get_block_size(), num_rows);

                if (diag)
                {
                    amgx::thrust::copy(halo_rows[i].values.begin() + (halo_rows[i].row_offsets[halo_rows[i].get_num_rows()] + halo_btl[i].B2L_rings[0][ring])*Ac.get_block_size(),
                                 halo_rows[i].values.begin() + (halo_rows[i].row_offsets[halo_rows[i].get_num_rows()] + halo_btl[i].B2L_rings[0][ring + 1])*Ac.get_block_size(),
                                 Ac.values.begin() + (Ac.row_offsets[Ac.get_num_rows()] + cumulative_num_rows)*Ac.get_block_size());
                    cumulative_num_rows += num_rows;
                }
            }
        }

        cudaCheckError();
        Ac.set_num_cols(Ac.manager->halo_offsets[Ac.manager->halo_offsets.size() - 1]);
        Ac.set_num_rows(Ac.get_num_cols());
        Ac.set_num_nz(new_nnz);
        Ac.delProps(COO);
        Ac.set_initialized(1);
        Ac.computeDiagonal();
    }
}

template <class T_Config>
void Aggregation_AMG_Level_Base<T_Config>::prepareNextLevelMatrix_diag(const Matrix<TConfig> &A, Matrix<TConfig> &Ac)
{
    if (A.is_matrix_singleGPU()) { return; }

    int num_neighbors = A.manager->neighbors.size();

    if (TConfig::memSpace == AMGX_host)
    {
        FatalError("Aggregation AMG Not implemented for host", AMGX_ERR_NOT_IMPLEMENTED);
    }
    else
    {
        int c_size = Ac.manager->halo_offsets[0];
        int f_size = A.manager->halo_offsets[0];
        int diag = Ac.hasProps(DIAG);
        Ac.manager->inverse_renumbering.resize(c_size);
        //get coarse -> fine renumbering
        int num_blocks = std::min(4096, (c_size + 127) / 128);
        coarse_to_global <<< num_blocks, 128>>>(this->m_aggregates.raw(), this->m_aggregates_fine_idx.raw(), Ac.manager->inverse_renumbering.raw(), f_size, -1 * A.manager->base_index());
        cudaCheckError();
        Ac.manager->set_num_halo_rows(Ac.manager->halo_offsets[Ac.manager->halo_offsets.size() - 1] - c_size);

        if (!diag) { Ac.computeDiagonal(); }

        Ac.set_initialized(1);
        std::vector<MVector> diagonals(num_neighbors);

        for (int i = 0; i < num_neighbors; i++)
        {
            int size = Ac.manager->B2L_rings[i][Ac.manager->B2L_rings.size() - 1];
            diagonals[i].resize(Ac.get_block_size()*size);
            int num_blocks = std::min(4096, (size + 127) / 128);
            write_diagonals <<< num_blocks, 128>>>(Ac.values.raw(), Ac.diag.raw(), Ac.manager->B2L_maps[i].raw(), diagonals[i].raw(), Ac.get_block_size(), size);
        }

        cudaCheckError();
        Ac.manager->getComms()->exchange_vectors(diagonals, Ac, this->tag * 100 + 10 + 2);

        for (int i = 0; i < num_neighbors; i++)
        {
            int size = Ac.manager->halo_offsets[i + 1] - Ac.manager->halo_offsets[i];

            if (Ac.hasProps(DIAG)) { amgx::thrust::copy(diagonals[i].begin(), diagonals[i].begin() + Ac.get_block_size()*size, Ac.values.begin() + Ac.get_block_size() * (Ac.diagOffset() + Ac.manager->halo_offsets[i])); }
            else
            {
                int num_blocks = std::min(4096, (size + 127) / 128);
                write_diagonals_back <<< num_blocks, 128>>>(Ac.values.raw(), Ac.diag.raw() + Ac.manager->halo_offsets[i], diagonals[i].raw(), Ac.get_block_size(), size);
            }
        }

        cudaCheckError();
    }
}

template <class T_Config>
void Aggregation_AMG_Level_Base<T_Config>::prepareNextLevelMatrix_none(const Matrix<TConfig> &A, Matrix<TConfig> &Ac)
{
    if (A.is_matrix_singleGPU()) { return; }

    int num_neighbors = A.manager->neighbors.size();

    if (TConfig::memSpace == AMGX_host)
    {
        FatalError("Aggregation AMG Not implemented for host", AMGX_ERR_NOT_IMPLEMENTED);
    }
    else
    {
        int c_size = Ac.manager->halo_offsets[0];
        int f_size = A.manager->halo_offsets[0];
        int diag = Ac.hasProps(DIAG);
        Ac.manager->inverse_renumbering.resize(c_size);
        //get coarse -> fine renumbering
        int num_blocks = std::min(4096, (c_size + 127) / 128);
        coarse_to_global <<< num_blocks, 128>>>(this->m_aggregates.raw(), this->m_aggregates_fine_idx.raw(), Ac.manager->inverse_renumbering.raw(), f_size, 0);
        cudaCheckError();
        Ac.manager->set_num_halo_rows(Ac.manager->halo_offsets[Ac.manager->halo_offsets.size() - 1] - c_size);
        Ac.set_initialized(1);

        if (!diag) { Ac.computeDiagonal(); }
    }
}

template <class T_Config>
void Aggregation_AMG_Level_Base<T_Config>::prepareNextLevelMatrix(const Matrix<TConfig> &A, Matrix<TConfig> &Ac)
{
    if (m_matrix_halo_exchange == 0)
    {
        this->prepareNextLevelMatrix_none(A, Ac);
    }
    else if (m_matrix_halo_exchange == 1)
    {
        this->prepareNextLevelMatrix_diag(A, Ac);
    }
    else if (m_matrix_halo_exchange == 2)
    {
        this->prepareNextLevelMatrix_full(A, Ac);
    }
    else
    {
        FatalError("Invalid Aggregation matrix_halo_exchange parameter", AMGX_ERR_NOT_IMPLEMENTED);
    }
}


__global__ void set_halo_rowlen(INDEX_TYPE *work, INDEX_TYPE *output, INDEX_TYPE  size, INDEX_TYPE diag)
{
    int idx = blockDim.x * blockIdx.x + threadIdx.x;

    while (idx < size)
    {
        if (work[idx + 1] - work[idx] > 0)
        {
            output[idx] += work[idx + 1] - work[idx] - (1 - diag);
        }

        idx += blockDim.x * gridDim.x;
    }
}

template <typename T>
__global__ void append_halo_nz(INDEX_TYPE *row_offsets, INDEX_TYPE *new_row_offsets, INDEX_TYPE *col_indices, INDEX_TYPE *new_col_indices, T *values, T *new_values, INDEX_TYPE size, INDEX_TYPE diag, INDEX_TYPE halo_offset, INDEX_TYPE block_size)
{
    int idx = blockDim.x * blockIdx.x + threadIdx.x;

    while (idx < size)
    {
        int add_diag = !diag;

        if (!diag && new_col_indices[new_row_offsets[idx]] != -1) { add_diag = 0; } //if diag or there is already soimething in the row, then don't add diagonal nonzero (inside diag)

        int append_offset = -1;

        for (int i = new_row_offsets[idx]; i < new_row_offsets[idx + 1]; i++)
        {
            if (new_col_indices[i] == -1) {append_offset = i; break;}
        }

        for (int i = row_offsets[idx]; i < row_offsets[idx + 1]; i++)
        {
            if (diag && i == row_offsets[idx])   //if outside diag and this is the first nonzero in a non-empty row, overwrite diagonal value
            {
                for (int j = 0; j < block_size; j++)
                {
                    new_values[(new_row_offsets[size] + halo_offset + idx)*block_size + j] = values[(row_offsets[size] + halo_offset + idx) * block_size + j];
                }
            }

            int col_idx = col_indices[i];

            if (append_offset == -1 && (col_idx != halo_offset + idx)) {printf("ERROR: append offset is -1 but row has nonzeros in it old %d to %d new %d to %d\n", row_offsets[idx], row_offsets[idx + 1], new_row_offsets[idx], new_row_offsets[idx + 1]); append_offset = 0;}

            if (col_idx != halo_offset + idx || add_diag)
            {
                new_col_indices[append_offset] = col_idx;

                for (int j = 0; j < block_size; j++)
                {
                    new_values[append_offset * block_size + j] = values[i * block_size + j];
                }

                append_offset++;
            }
        }

        idx += blockDim.x * gridDim.x;
    }
}

template <class T_Config>
void Aggregation_AMG_Level_Base<T_Config>::createCoarseB2LMaps(std::vector<IVector> &in_coarse_B2L_maps)
{
    Matrix<TConfig> &A = this->getA();
    m_num_all_aggregates = m_num_aggregates;
    int num_neighbors = A.manager->neighbors.size();
    IndexType max_b2l = 0;

    for (int i = 0; i < num_neighbors; i++ ) { max_b2l = max_b2l > A.manager->B2L_rings[i][1] ? max_b2l : A.manager->B2L_rings[i][1]; }

    IVector B2L_aggregates(max_b2l);
    IVector indices(max_b2l);

    for (int i = 0; i < num_neighbors; i++ )
    {
        int size = A.manager->B2L_rings[i][1];
        thrust_wrapper::fill<TConfig::memSpace>(B2L_aggregates.begin(), B2L_aggregates.begin() + size, 0);
        thrust_wrapper::sequence<TConfig::memSpace>(indices.begin(), indices.begin() + size);
        //substitute coarse aggregate indices for fine boundary nodes
        amgx::thrust::copy(amgx::thrust::make_permutation_iterator(this->m_aggregates.begin(), A.manager->B2L_maps[i].begin()),
                     amgx::thrust::make_permutation_iterator(this->m_aggregates.begin(), A.manager->B2L_maps[i].begin() + size),
                     B2L_aggregates.begin());
        //find the unique ones
        amgx::thrust::sort_by_key(B2L_aggregates.begin(), B2L_aggregates.begin() + size, indices.begin());
        IndexType num_unique = amgx::thrust::unique_by_key(B2L_aggregates.begin(), B2L_aggregates.begin() + size, indices.begin()).first - B2L_aggregates.begin();
        in_coarse_B2L_maps[i].resize(num_unique);
        //sort it back so we have the original ordering
        amgx::thrust::sort_by_key(indices.begin(), indices.begin() + num_unique, B2L_aggregates.begin());
        amgx::thrust::copy(B2L_aggregates.begin(), B2L_aggregates.begin() + num_unique, in_coarse_B2L_maps[i].begin());
    }

    cudaCheckError();
}


__global__ void populate_coarse_boundary(INDEX_TYPE *flags, INDEX_TYPE *indices, INDEX_TYPE *maps, INDEX_TYPE *output, INDEX_TYPE  size)
{
    int idx = blockDim.x * blockIdx.x + threadIdx.x;

    while (idx < size)
    {
        output[flags[maps[indices[idx]]]] = maps[indices[idx]];
        idx += blockDim.x * gridDim.x;
    }
}

__global__ void flag_coarse_boundary(INDEX_TYPE *flags, INDEX_TYPE *indices, INDEX_TYPE *maps, INDEX_TYPE  size)
{
    int idx = blockDim.x * blockIdx.x + threadIdx.x;

    while (idx < size)
    {
        flags[maps[indices[idx]]] = 1;
        idx += blockDim.x * gridDim.x;
    }
}

__global__ void flag_halo_indices(INDEX_TYPE *flags, INDEX_TYPE *indices, INDEX_TYPE  offset, INDEX_TYPE  size)
{
    int idx = blockDim.x * blockIdx.x + threadIdx.x;

    while (idx < size)
    {
        flags[indices[idx] - offset] = 1;
        idx += blockDim.x * gridDim.x;
    }
}

__global__ void apply_halo_aggregate_indices(INDEX_TYPE *flags, INDEX_TYPE *indices, INDEX_TYPE *output, INDEX_TYPE offset, INDEX_TYPE aggregates_offset, INDEX_TYPE  size)
{
    int idx = blockDim.x * blockIdx.x + threadIdx.x;

    while (idx < size)
    {
        output[idx] = flags[indices[idx] - offset] + aggregates_offset;
        idx += blockDim.x * gridDim.x;
    }
}

// renumbering the aggregates/communicationg with neighbors
template <class T_Config>
void Aggregation_AMG_Level_Base<T_Config>::setNeighborAggregates()
{
    Matrix<TConfig> &A = this->getA();
    Matrix<TConfig> &Ac = this->getNextLevel( MemorySpace( ) )->getA();
    m_num_all_aggregates = m_num_aggregates;

    /* WARNING: the matrix reordering always happens inside createRenumbering routine. There are three ways to get to this routine
       1. matrix_upload_all -> uploadMatrix -> initializeUploadReorderAll -> reorder_matrix -> createRenumbering
       2. read_system_distributed -> renumberMatrixOneRing -> reorder_matrix_owned -> createRenumbering
       3. solver_setup -> ... -> AMG_Level::setup -> createCoarseMatrices -> setNeighborAggregates -> createRenumbering
       If you are reading the renumbering from file you might need to add intercept code in if statement below,
       otherwise this routine will exit before calling createRenumbering routine (in case of single or disjoint partitions).
    */
    if (this->getA().is_matrix_singleGPU()) { return; }

    int num_neighbors = A.manager->neighbors.size();

    //
    // Step 0 - set up coarse matrix metadata
    //
    if (Ac.manager == NULL) { Ac.manager = new DistributedManager<T_Config>(); }

    Ac.manager->resize(A.manager->neighbors.size(), 1);
    Ac.manager->A = &Ac;
    int f_size = A.get_num_rows();
    Ac.manager->setComms(A.manager->getComms());
    Ac.manager->set_global_id(A.manager->global_id());
    Ac.manager->neighbors = A.manager->neighbors;
    Ac.manager->set_base_index(A.manager->base_index());
    Ac.manager->halo_ranges = A.manager->halo_ranges;
    Ac.manager->set_index_range(A.manager->index_range());
    //-------------------------------------- Section 1 - renumbering -----------------------------------------------------------
    //
    // Step 1 - calculate coarse level B2L maps - any aggregate that has a fine boundary node, becomes a coarse boundary node
    //
    m_num_all_aggregates = m_num_aggregates;
    int vec_size = m_num_aggregates + 1; //A.manager->num_boundary_nodes()+1;
    IVector B2L_aggregates(vec_size);

    for (int i = 0; i < A.manager->neighbors.size(); i++)
    {
        thrust_wrapper::fill<TConfig::memSpace>(B2L_aggregates.begin(), B2L_aggregates.begin() + vec_size, 0);
        int size = A.manager->B2L_rings[i][1];
        int block_size = 128;
        int grid_size = std::min( 4096, ( size + block_size - 1 ) / block_size);
        flag_coarse_boundary <<< grid_size, block_size>>>(B2L_aggregates.raw(), A.manager->B2L_maps[i].raw(), this->m_aggregates.raw(), size);
        thrust_wrapper::exclusive_scan<TConfig::memSpace>(B2L_aggregates.begin(), B2L_aggregates.begin() + vec_size, B2L_aggregates.begin());
        (Ac.manager->B2L_maps)[i].resize(B2L_aggregates[vec_size - 1]);
        populate_coarse_boundary <<< grid_size, block_size>>>(B2L_aggregates.raw(), A.manager->B2L_maps[i].raw(), this->m_aggregates.raw(), Ac.manager->B2L_maps[i].raw(), size);
    }

    cudaCheckError();

    for (int i = 0; i < num_neighbors; i++)
    {
        Ac.manager->B2L_rings[i].resize(2);
        Ac.manager->B2L_rings[i][0] = 0;
        Ac.manager->B2L_rings[i][1] = Ac.manager->B2L_maps[i].size();
    }

    DistributedArranger<T_Config> *prep = new DistributedArranger<T_Config>;
    prep->initialize_B2L_maps_offsets(Ac, 1);
    delete prep;
    Ac.set_num_rows(m_num_aggregates);
    IVector renumbering(m_num_aggregates + 1); /* +1 is actually not needed, it will be resized in createRenumbering */
    Ac.manager->createRenumbering(renumbering);
    //
    // Step 2 - renumber aggregates, so boundary nodes will have higher index than interior ones (based on the renumberiong we have been calculating)
    //
    /* WARNING: 1. Thrust scatter and gather routines seem more appropriate here, but they implicitly assume that the input
                and output have certain size correlation, which is not matched by vectors in our case. The only remaining option
                is to use make_permutation as is done below. Example of Thrust scatter and gather calls
                IVector ttt(f_size,-1);
                amgx::thrust::scatter(this->m_aggregates.begin(), this->m_aggregates.begin()+f_size, renumbering.begin(), ttt.begin());
                amgx::thrust::gather(renumbering.begin(), renumbering.end(), this->m_aggregates.begin(), ttt.begin());
                amgx::thrust::copy(ttt.begin(), ttt.end(), this->m_aggregates.begin());

                2. The original thrust composite call is illegal because it uses the same array (m_aggregates) for input and output.
                amgx::thrust::copy(amgx::thrust::make_permutation_iterator(renumbering.begin(), this->m_aggregates.begin()),
                amgx::thrust::make_permutation_iterator(renumbering.begin(), this->m_aggregates.begin()+f_size),
                             this->m_aggregates.begin());
                Although it somehow still works, it is much safer to use explicit temporary storage for the intermediate result.
    */
    /* WARNING: must save unreordered aggregates for later use before reordering them. */
    IVector unreordered_aggregates(this->m_aggregates);
    /* WARNING: change Thrust call to explicitly use temporary storage for the intermediate result. The earlier version is illegal, but somehow still works. */
    IVector ttt(f_size, -1);
    amgx::thrust::copy(amgx::thrust::make_permutation_iterator(renumbering.begin(), this->m_aggregates.begin()),
                 amgx::thrust::make_permutation_iterator(renumbering.begin(), this->m_aggregates.begin() + f_size),
                 ttt.begin());
    amgx::thrust::copy(ttt.begin(), ttt.end(), this->m_aggregates.begin());
    cudaCheckError();

    //we don't need renumbering anymore, it will be identity on the coarse level

    //-------------------------------------- Section 2 - communication -----------------------------------------------------------

    //
    // Step 3 - populate aggregates_fine_idx, which stores for every fine node the original global index of the aggregate (which is lowest global index of nodes aggregated together)
    //

    //
    // These are different when we do /don't do matrix halo exchanges - when we do we need global indices to match nodes,
    // and in this case Ac after computeA will not have the same ordering of halo nodes as after prepareNextLevel_full.
    // However when we do not do matrix halo exchange we are only interested in the ordering of halo nodes on the coarse level,
    // and we can get that by exchanging the (already renumbered) aggregates vector.
    //
    if (m_matrix_halo_exchange == 2)
    {
        //Find original global indices of nodes that have the minimum id in the aggregates.
        amgx::thrust::copy(amgx::thrust::make_permutation_iterator(A.manager->inverse_renumbering.begin(), this->m_aggregates_fine_idx.begin()),
                     amgx::thrust::make_permutation_iterator(A.manager->inverse_renumbering.begin(), this->m_aggregates_fine_idx.begin() + f_size),
                     this->m_aggregates_fine_idx.begin());
        thrust_wrapper::transform<TConfig::memSpace>(this->m_aggregates_fine_idx.begin(),
                          this->m_aggregates_fine_idx.begin() + f_size,
                          amgx::thrust::constant_iterator<IndexType>(A.manager->base_index()),
                          this->m_aggregates_fine_idx.begin(),
                          amgx::thrust::plus<IndexType>());
        //communicate
        this->m_aggregates_fine_idx.set_block_dimx(1);
        this->m_aggregates_fine_idx.set_block_dimy(1);
        m_aggregates_fine_idx.dirtybit = 1;
        A.manager->exchange_halo(m_aggregates_fine_idx, this->tag * 100 + 1 * 10 + 0);
    }
    else
    {
        //communicate
        this->m_aggregates.set_block_dimx(1);
        this->m_aggregates.set_block_dimy(1);
        m_aggregates.dirtybit = 1;
        /* WARNING: you should exchange unreordered aggregates, and append them to your own reordered aggregates, to conform to asusmptions done by distributed_mamanger. */
        //A.manager->exchange_halo(m_aggregates, this->tag*100+1*10+0); //wrong
        A.manager->exchange_halo(unreordered_aggregates, this->tag * 100 + 1 * 10 + 0);
        amgx::thrust::copy(unreordered_aggregates.begin() + f_size, unreordered_aggregates.end(), this->m_aggregates.begin() + f_size);
    }

    cudaCheckError();
    //
    // Step 4 - consolidate neighbors' aggregates into own list to be able to perform Galerkin product with the n-ring halo
    //
    IVector &exchanged_aggregates = m_matrix_halo_exchange == 2 ? this->m_aggregates_fine_idx : this->m_aggregates;
    int min_index = amgx::thrust::reduce(exchanged_aggregates.begin() + A.manager->halo_offsets[0], exchanged_aggregates.begin() + A.manager->halo_offsets[num_neighbors], (int)0xFFFFFFF, amgx::thrust::minimum<int>());
    int max_index = amgx::thrust::reduce(exchanged_aggregates.begin() + A.manager->halo_offsets[0], exchanged_aggregates.begin() + A.manager->halo_offsets[num_neighbors], (int)0, amgx::thrust::maximum<int>());
    cudaCheckError();
    int s_size = max_index - min_index + 2;
    IVector scratch(s_size);

    for (int i = 0; i < num_neighbors; i++)
    {
        int size = A.manager->halo_offsets[i + 1] - A.manager->halo_offsets[i];
        //Could also use local minimums to perform the same operation. The results are the same.
        //int min_local = amgx::thrust::reduce(exchanged_aggregates.begin()+A.manager->halo_offsets[i], exchanged_aggregates.begin()+A.manager->halo_offsets[i+1], (int)0xFFFFFFF, amgx::thrust::minimum<int>());
        thrust_wrapper::fill<TConfig::memSpace>(scratch.begin(), scratch.begin() + s_size, 0);
        int block_size = 128;
        int grid_size = std::min( 4096, ( size + block_size - 1 ) / block_size);
        flag_halo_indices <<< grid_size, block_size>>>(scratch.raw(), exchanged_aggregates.raw() + A.manager->halo_offsets[i], min_index /*min_local*/, size);
        thrust_wrapper::exclusive_scan<TConfig::memSpace>(scratch.begin(), scratch.begin() + s_size, scratch.begin());
        apply_halo_aggregate_indices <<< grid_size, block_size>>>(scratch.raw(), exchanged_aggregates.raw() + A.manager->halo_offsets[i], this->m_aggregates.raw() + A.manager->halo_offsets[i], min_index /*min_local*/, m_num_all_aggregates, size);
        Ac.manager->halo_offsets[i] = m_num_all_aggregates;
        m_num_all_aggregates += scratch[s_size - 1];
    }

    cudaCheckError();
    Ac.manager->halo_offsets[num_neighbors] = m_num_all_aggregates;
}

//TODO: The consolidate and unconsolidate parts could be made more efficient by only sending the
//      nonzero values
template <class T_Config>
void Aggregation_AMG_Level_Base<T_Config>::consolidateVector(VVector &x)
{
    int my_id = this->getA().manager->global_id();

    if (this->getA().manager->isRootPartition())
    {
        // Here all partitions being consolidated should have same vector size, see TODO above
        INDEX_TYPE num_parts = this->getA().manager->getNumPartsToConsolidate();

        for (int i = 0; i < num_parts; i++)
        {
            int current_part = this->getA().manager->getPartsToConsolidate()[i];

            // Vector has been set to correct size
            if (current_part != my_id)
            {
                //printf("Root partition %d receiving %d -> %d and %d -> %d (total %d)\n", this->getA().manager->global_id(), this->getA().manager->getConsolidationArrayOffsets()[i], this->getA().manager->getConsolidationArrayOffsets()[i+1], this->getA().manager->getConsolidationArrayOffsets()[num_parts+i], this->getA().manager->getConsolidationArrayOffsets()[num_parts+i+1], (int)x.size()/x.get_block_size());
                this->getA().manager->getComms()->recv_vector(x, current_part, 10000 + current_part, x.get_block_size()*this->getA().manager->getConsolidationArrayOffsets()[i], x.get_block_size() * (this->getA().manager->getConsolidationArrayOffsets()[i + 1] - this->getA().manager->getConsolidationArrayOffsets()[i]));
                this->getA().manager->getComms()->recv_vector(x, current_part, 20000 + current_part, x.get_block_size()*this->getA().manager->getConsolidationArrayOffsets()[num_parts + i], x.get_block_size() * (this->getA().manager->getConsolidationArrayOffsets()[num_parts + i + 1] - this->getA().manager->getConsolidationArrayOffsets()[num_parts + i]));
            }
        }
    }
    else
    {
        int my_destination_part = this->getA().manager->getMyDestinationPartition();
        int i_off, i_size, b_off, b_size;
        this->getA().manager->getConsolidationOffsets(&i_off, &i_size, &b_off, &b_size);
        // Here all partitions being consolidated should have same vector size, see TODO above
        this->getA().manager->getComms()->send_vector_async(x, my_destination_part, 10000 + my_id, i_off * x.get_block_size(), i_size * x.get_block_size());
        this->getA().manager->getComms()->send_vector_async(x, my_destination_part, 20000 + my_id, b_off * x.get_block_size(), b_size * x.get_block_size());
    }
}

//TODO: The consolidate and unconsolidate parts could be made more efficient by only sending the
//      nonzero values
template <class T_Config>
void Aggregation_AMG_Level_Base<T_Config>::unconsolidateVector(VVector &x)
{
    if (this->getA().manager->isRootPartition())
    {
        INDEX_TYPE num_parts = this->getA().manager->getNumPartsToConsolidate();

        for (int i = 0; i < num_parts; i++)
        {
            int current_part = this->getA().manager->getPartsToConsolidate()[i];

            // Vector has been set to correct size
            if (current_part != this->getA().manager->global_id())
            {
                this->getA().manager->getComms()->send_vector_async(x, current_part, 30000 + current_part, x.get_block_size()*this->getA().manager->getConsolidationArrayOffsets()[i], x.get_block_size() * (this->getA().manager->getConsolidationArrayOffsets()[i + 1] - this->getA().manager->getConsolidationArrayOffsets()[i]));
                this->getA().manager->getComms()->send_vector_async(x, current_part, 40000 + current_part, x.get_block_size()*this->getA().manager->getConsolidationArrayOffsets()[num_parts + i], x.get_block_size() * (this->getA().manager->getConsolidationArrayOffsets()[num_parts + i + 1] - this->getA().manager->getConsolidationArrayOffsets()[num_parts + i]));
            }
        }
    }
    else
    {
        int my_destination_part = this->getA().manager->getMyDestinationPartition();
        // Vector x is of unknown size
        int i_off, i_size, b_off, b_size;
        this->getA().manager->getConsolidationOffsets(&i_off, &i_size, &b_off, &b_size);
        this->getA().manager->getComms()->recv_vector(x, my_destination_part, 30000 + this->getA().manager->global_id(), i_off * x.get_block_size(), i_size * x.get_block_size());
        this->getA().manager->getComms()->recv_vector(x, my_destination_part, 40000 + this->getA().manager->global_id(), b_off * x.get_block_size(), b_size * x.get_block_size());
    }
}


template <class T_Config>
void Aggregation_AMG_Level_Base<T_Config>::createCoarseVertices()
{
    //Set the aggregates
    this->m_selector->setAggregates(this->getA(), this->m_aggregates, this->m_aggregates_fine_idx, this->m_num_aggregates);

    if ( this->m_print_aggregation_info )
    {
        this->m_selector->printAggregationInfo( this->m_aggregates, this->m_aggregates_fine_idx, this->m_num_aggregates );
    }

    this->getA().template setParameter< int > ("aggregates_num", this->m_num_aggregates); // ptr to aaggregates
}

//  Creating the next level
template <class T_Config>
void Aggregation_AMG_Level_Base<T_Config>::createCoarseMatrices()
{
    Matrix<TConfig> &A = this->getA();
    Matrix<TConfig> &Ac = this->getNextLevel( MemorySpace( ) )->getA();

    /* WARNING: do not recompute prolongation (P) and restriction (R) when you
                are reusing the level structure (structure_reuse_levels > 0).
                Notice that in aggregation path, prolongation P is implicit,
                and is used through the aggregates array. */

    bool const consolidation_level = !A.is_matrix_singleGPU() && this->isConsolidationLevel();

    // bookkeeping for the coarse grid: renumber aggregates,
    // if consolidation compute consolidated halo-offsets, etc
    if (!this->isReuseLevel())
    {
        if (consolidation_level)
        {
            // Consolidation-path steps 1-9
            this->consolidationBookKeeping();
        }
        else
        {
            this->setNeighborAggregates();
        }
    }

    this->getA().setView(ALL);

    // Compute restriction operator
    // TODO: computing the restriction operator could be merged with the selector to save some work
    // If we reuse the level we keep the previous restriction operator
    if (this->isReuseLevel() == false)
    {
        computeRestrictionOperator();
    }

    Ac.set_initialized(0);
    Ac.copyAuxData(&A);
    this->m_coarseAGenerator->computeAOperator(A, Ac, this->m_aggregates, this->m_R_row_offsets, this->m_R_column_indices, this->m_num_all_aggregates);
    Ac.setColsReorderedByColor(false);
    Ac.setView(FULL);

    if (consolidation_level)
    {
        // Consolidation-path Steps 11-12, send matrices to root, consolidate, final bookkeeping
        this->consolidateCoarseGridMatrix();
    }
    else
    {
        this->prepareNextLevelMatrix(A, Ac);
    }

    A.setView(OWNED);
    Ac.setView(OWNED);

    this->m_next_level_size = this->m_num_all_aggregates * Ac.get_block_dimy();

    if (this->m_print_aggregation_info)
    {
        MatrixAnalysis<TConfig> ana(&Ac);
        ana.aggregatesQuality2(this->m_aggregates, this->m_num_aggregates, A);
    }
}

template <class T_Config>
void Aggregation_AMG_Level_Base<T_Config>::consolidationBookKeeping()
{
    Matrix<TConfig> &A = this->getA();
    Matrix<TConfig> &Ac = this->getNextLevel( MemorySpace( ) )->getA();


    int num_parts, num_fine_neighbors, my_id;

    if (!A.is_matrix_singleGPU())
    {
        num_parts = A.manager->getComms()->get_num_partitions();
        num_fine_neighbors = A.manager->neighbors.size();
        my_id = A.manager->global_id();
    }
    else
    {
        num_parts = 1;
        num_fine_neighbors = 0;
        my_id = 0;
    }

    // ----------------------------------------------------
    // Consolidate multiple fine matrices into one coarse matrix
    // ----------------------------------------------------
    // ----------------
    // Step 1
    // Decide which partitions should be merged together, store in destination_partitions vector
    // ---------------
    IVector_h &destination_part = A.manager->getDestinationPartitions();
    int my_destination_part = A.manager->getMyDestinationPartition();

    if (my_destination_part >= num_parts)
    {
        FatalError("During consolidation, sending data to partition that doesn't exist", AMGX_ERR_NOT_IMPLEMENTED);
    }

    // Create mapping from coarse partition indices (ranks on the coarse consolidated level) to partition indices on the fine level (ranks on the fine level)
    IVector_h coarse_part_to_fine_part = destination_part;
    amgx::thrust::sort(coarse_part_to_fine_part.begin(), coarse_part_to_fine_part.end());
    cudaCheckError();
    coarse_part_to_fine_part.erase(thrust::unique(coarse_part_to_fine_part.begin(), coarse_part_to_fine_part.end()), coarse_part_to_fine_part.end());
    cudaCheckError();
    //Then, the number of coarse partitions is simply the size of this vector
    int num_coarse_partitions = coarse_part_to_fine_part.size();
    // Create mapping from fine partition indices to coarse partition indices, with fine partitions that are merging together having the same coarse indices
    IVector_h fine_part_to_coarse_part(num_parts);
    amgx::thrust::lower_bound(coarse_part_to_fine_part.begin(), coarse_part_to_fine_part.end(), destination_part.begin(), destination_part.end(), fine_part_to_coarse_part.begin());
    cudaCheckError();
    // Create mapping from this specific partition's neighbors to consolidated coarse neighbors, but using their fine index (aka. destination partition indices for my neighbors)
    IVector_h fine_neigh_to_fine_part;
    A.manager->createNeighToDestPartMap(fine_neigh_to_fine_part, A.manager->neighbors, destination_part, num_fine_neighbors);
    // Create mapping from consolidated coarse neighbors to fine partition indices (even if the current partition is not going to be a root)
    IVector_h coarse_neigh_to_fine_part;
    int num_coarse_neighbors;
    A.manager->createConsolidatedNeighToPartMap(coarse_neigh_to_fine_part, fine_neigh_to_fine_part, my_destination_part, destination_part, num_coarse_neighbors);
    // Create mapping from fine neighbors to coarse neighbors, with fine neighbors this partition is merging with labeled with -1
    IVector_h fine_neigh_to_coarse_neigh;
    A.manager->createNeighToConsNeigh(fine_neigh_to_coarse_neigh, coarse_neigh_to_fine_part, fine_neigh_to_fine_part, my_destination_part, num_fine_neighbors);
    /*
        EXAMPLE
        Take the following partition graph (that describes connections between partitions, vertices are the partitions themselves), this is the same graph that is used in the setup example
        number of partitions num_parts=12
        CSR row_offsets [0 4 8 13 21 25 32 36 41 46 50 57 61]
        CSR col_indices [0 1 3 8
                    0 1 2 3
                    1 2 3 4 5
                    0 1 2 3 4 5 8 10
                    2 4 5 6
                    2 3 4 5 6 7 10
                    4 5 6 7
                    5 6 7 9 10
                    0 3 8 10 11
                    7 9 10 11
                    3 5 7 8 9 10 11
                    8 9 10 11]
        destination_part = [0 0 0 0 4 4 4 4 8 8 8 8]
        coarse_part_to_fine_part = [0 4 8] num_coarse_partitions = 3
        fine_part_to_coarse_part = [0 0 0 0 1 1 1 1 2 2 2 2]
        original neighbor lists correspond to the rows of the matrix, minus the diagonal elements: (part 0)[1 3 8] (part 3)[0 1 2 4 5 8 10] (part 10)[3 5 7 8 9 11]
        fine_neigh_to_fine_part (part 0)[0 0 2] (part 3)[0 0 0 0 1 2 2] (part 10)[0 1 1 2 2 2]
        coarse_neigh_to_fine_part (part 0)[8] (part 3)[4 8] (part 10)[0 4]
        fine_neigh_to_coarse_neigh (part 0)[-1 -1 0] (part 3)[-1 -1 -1 0 0 1 1] (part 10)[0 1 1 -1 -1 -1]
        */
    // --------------------------
    // Step 2
    // Create coarse B2L_maps, by mapping fine B2L maps to coarse indices using this->m_aggregates and eliminating duplicates
    // --------------------------
    std::vector<IVector> coarse_B2L_maps(num_fine_neighbors);
    m_num_all_aggregates = m_num_aggregates;
    int num_neighbors_temp = A.manager->neighbors.size();
    int num_rings = A.manager->B2L_rings[0].size() - 1;

    if (num_rings != 1)
    {
        FatalError("num_rings > 1 not supported in consolidation\n", AMGX_ERR_NOT_IMPLEMENTED);
    }

    IndexType max_b2l = 0;


    for (int i = 0; i < num_neighbors_temp; i++ ) { max_b2l = max_b2l > A.manager->B2L_rings[i][1] ? max_b2l : A.manager->B2L_rings[i][1]; }

    IVector B2L_aggregates(max_b2l);
    IVector indices(max_b2l);

    //TODO: use the algorithm from setNeighborAggregates()
    for (int i = 0; i < num_neighbors_temp; i++ )
    {
        int size = A.manager->B2L_rings[i][1];
        thrust_wrapper::fill<TConfig::memSpace>(B2L_aggregates.begin(), B2L_aggregates.begin() + size, 0);
        thrust_wrapper::sequence<TConfig::memSpace>(indices.begin(), indices.begin() + size);
        //substitute coarse aggregate indices for fine boundary nodes
        amgx::thrust::copy(amgx::thrust::make_permutation_iterator(this->m_aggregates.begin(), A.manager->B2L_maps[i].begin()),
                        amgx::thrust::make_permutation_iterator(this->m_aggregates.begin(), A.manager->B2L_maps[i].begin() + size),
                        B2L_aggregates.begin());
        //find the unique ones
        amgx::thrust::sort_by_key(B2L_aggregates.begin(), B2L_aggregates.begin() + size, indices.begin());
        IndexType num_unique = amgx::thrust::unique_by_key(B2L_aggregates.begin(), B2L_aggregates.begin() + size, indices.begin()).first - B2L_aggregates.begin();
        coarse_B2L_maps[i].resize(num_unique);
        //sort it back so we have the original ordering
        amgx::thrust::sort_by_key(indices.begin(), indices.begin() + num_unique, B2L_aggregates.begin());
        amgx::thrust::copy(B2L_aggregates.begin(), B2L_aggregates.begin() + num_unique, coarse_B2L_maps[i].begin());
    }

    cudaCheckError();
    /*
        * EXAMPLE
        say, partition 3 has the following coarse B2L_maps:
        neighbors [0 1 2 4 5 8 10]
        B2L_maps[0(=0)] = [6 7 8]
        B2L_maps[1(=1)] = [8 9 10]
        B2L_maps[2(=2)] = [10 11 12 13]
        B2L_maps[3(=4)] = [13 14 15]
        B2L_maps[4(=5)] = [15 16 17]
        B2L_maps[5(=8)] = [6 18 19]
        B2L_maps[6(=10)] = [17 20 19]
        */
    // ---------------------------------------------------
    // Step 3
    // create new B2L maps for each merged destination neighbor and drop B2L maps to neighbors we are merging with
    // ---------------------------------------------------
    std::vector<IVector> dest_coarse_B2L_maps;
    A.manager->consolidateB2Lmaps(dest_coarse_B2L_maps, coarse_B2L_maps, fine_neigh_to_coarse_neigh, num_coarse_neighbors, num_fine_neighbors);
    /*
        * EXAMPLE
        Then, merging the coarse B2L maps on partition 3, we get:
        coarse_neigh_to_fine_part [4 8]
        dest_coarse_B2L_maps[0(=4)] = [13 14 15 16 17]
        dest_coarse_B2L_maps[1(=8)] = [6 17 18 19 20]
        */
    // -----------------------
    // Step 4
    // Create interior-boundary renumbering of aggregates according to dest_coarse_B2L_maps
    // -----------------------
    // Now renumber the aggregates with all interior aggregates first, boundary aggregates second
    int num_interior_aggregates; //returned by createAggregatesRenumbering
    int num_boundary_aggregates; //returned by createAggregatesRenumbering
    IVector renumbering; //returned by createAggregatesRenumbering
    // Following calls create renumbering array and modifies B2L_maps
    A.manager->createAggregatesRenumbering(renumbering, dest_coarse_B2L_maps, this->m_num_aggregates, num_coarse_neighbors, num_interior_aggregates, num_boundary_aggregates, num_rings);
    /*
        * EXAMPLE
        Partition 3 will get a renumbering vector of size 21, for the 21 owned agggregates:
        [0 1 2 3 4 5 17 6 7 8 9 10 11 12 13 14 15 16 18 19 20]
        num_interior_aggregates = 12
        num_boundary_aggregates = 9
        */
    // -------------------------------------------------
    // Step 5
    // Determine whether root partition, make list of partitions merged into one
    // ------------------------------------------------
    // Check if I'm root partition and how fine partitions (including myself) are merging into me
    // bool is_root_partition = false;
    bool &is_root_partition = this->m_is_root_partition;
    is_root_partition = false; 
    int num_fine_parts_to_consolidate = 0;
    // IVector_h fine_parts_to_consolidate;
    IVector_h &fine_parts_to_consolidate = this->m_fine_parts_to_consolidate;

    for (int i = 0; i < num_parts; i++)
    {
        if (destination_part[i] == my_id)
        {
            is_root_partition = true;
            num_fine_parts_to_consolidate++;
        }
    }

    fine_parts_to_consolidate.resize(num_fine_parts_to_consolidate);
    int count = 0;

    for (int i = 0; i < num_parts; i++)
    {
        if (destination_part[i] == my_id)
        {
            fine_parts_to_consolidate[count] = i;
            count++;
        }
    }

    //save this information as state, as this will also be required during solve for restriction/prolongation
    A.manager->setIsRootPartition(is_root_partition);
    A.manager->setNumPartsToConsolidate(num_fine_parts_to_consolidate);
    A.manager->setPartsToConsolidate(fine_parts_to_consolidate);

    // Create a new distributed communicator for coarse levels that only contains active partitions
    if (Ac.manager == NULL)
    {
        Ac.manager = new DistributedManager<TConfig>();
    }

    Ac.manager->setComms(A.manager->getComms()->Clone());
    Ac.manager->getComms()->createSubComm(coarse_part_to_fine_part, is_root_partition);


    /*
        * EXAMPLE
        isRootPartition is true for partitions 0,4,8 false for others
        num_fine_parts_to_consolidate = 4 for partitions 0,4,8
        fine_parts_to_consolidate (part 0)[0 1 2 3] (part 4)[4 5 6 7] (part 8)[8 9 10 11]
        */
    // ----------------------
    // Step 6
    // Compute number of interior, boundary and total nodes in the consolidated coarse matrix. Create offsets so that partitions being merged together will have their aggregate indices ordered like this:
    // [num_interior(fine_parts_to_consolidate[0]] num_interior(fine_parts_to_consolidate[1]] ... num_interior(fine_parts_to_consolidate[num_fine_parts_to_consolidate]
    //        num_boundary(fine_parts_to_consolidate[0]] num_boundary(fine_parts_to_consolidate[1]] ... num_boundary(fine_parts_to_consolidate[num_fine_parts_to_consolidate] ]
    // ----------------------
    // Gather to get number of interior/boundary aggregates of neighbors I will merge with
    // std::vector<IVector_h> vertex_counts;
    std::vector<IVector_h> &vertex_counts = this->m_vertex_counts;
    // int interior_offset, boundary_offset, total_interior_rows_in_merged, total_boundary_rows_in_merged;
    int interior_offset, boundary_offset;
    int &total_interior_rows_in_merged = this->m_total_interior_rows_in_merged;
    int &total_boundary_rows_in_merged = this->m_total_boundary_rows_in_merged;
    int total_rows_in_merged;
    //Computes these offsets on the root, sends them back
    A.manager->computeConsolidatedOffsets(my_id, my_destination_part, is_root_partition, num_interior_aggregates, num_boundary_aggregates, vertex_counts, fine_parts_to_consolidate, num_fine_parts_to_consolidate, interior_offset, boundary_offset, total_interior_rows_in_merged, total_boundary_rows_in_merged, total_rows_in_merged, A.manager->getComms());
    //Partitions save these offsets, as it will be required during solve restriction/prolongation
    A.manager->setConsolidationOffsets(interior_offset, num_interior_aggregates, boundary_offset + num_interior_aggregates, num_boundary_aggregates);
    /*
        * EXAMPLE
        For root partition 0, say we have the following interior/boundary counts (note that partition 1 has 0 boundary, as it is only connected to partitions it is merging with)
        part 0 - interior: 10 boundary 3
        part 1 - interior: 18
        part 2 - interior: 10 boundary 16
        part 3 - interior: 12 boundary 9
        interior_offset for partitions 0,1,2,3: 0 10 28 38 (total_interior_rows_in_merged 50)
        boundary_offset for partitions 0,1,2,3: 0 3 3 19 (total_boundary_rows_in_merged 28)
        */
    // ----------------------
    // Step 7
    // Each partition renumbers its aggregates and dest_coarse_B2L_maps using offsets computed in Step 6 and permutation in Step 4
    // ----------------------
    // Kernel to renumber the aggregates
    int block_size = 128;
    int grid_size = std::min( 4096, ( A.manager->halo_offsets[0] + block_size - 1 ) / block_size);
    renumberAggregatesKernel <<< grid_size, block_size >>>(renumbering.raw(), interior_offset, boundary_offset, this->m_aggregates.raw(), A.manager->halo_offsets[0], num_interior_aggregates, renumbering.size());
    cudaCheckError();

    for (int i = 0; i < num_coarse_neighbors; i++)
    {
        thrust_wrapper::transform<TConfig::memSpace>(dest_coarse_B2L_maps[i].begin(),
                            dest_coarse_B2L_maps[i].end(),
                            amgx::thrust::constant_iterator<IndexType>(boundary_offset),
                            dest_coarse_B2L_maps[i].begin(),
                            amgx::thrust::plus<IndexType>());
    }

    cudaCheckError();
    /*
        * EXAMPLE
        Partition 3 had a renumbering vector:
        [0 1 2 3 4 5 17 6 7 8 9 10 11 12 13 14 15 16 18 19 20]
        which is now adjusted to account for the consolidated coarse matrices' indices:
        [38 39 40 41 42 43 74 44 45 46 47 48 49 69 70 71 72 73 75 76 77]
        And the dest_coarse_B2L_maps, which looked like:
        dest_coarse_B2L_maps[0(=4)] = [13 14 15 16 17]
        dest_coarse_B2L_maps[1(=8)] = [6 17 18 19 20]
        is now:
        dest_coarse_B2L_maps[0(=4)] = [69 70 71 72 73]
        dest_coarse_B2L_maps[1(=8)] = [74 73 75 76 77]
        */
    // -------------------------------------------------
    // Step 8
    // Send dest_coarse_B2L_maps to root partitions
    // ------------------------------------------------
    // Each fine partition sends to its root the number of coarse neighbors it has, their ids, and the number of boundary nodes for each coarse neighbor
    IVector_h num_bdy_per_coarse_neigh(num_coarse_neighbors);

    for (int i = 0; i < num_coarse_neighbors; i++)
    {
        num_bdy_per_coarse_neigh[i] = dest_coarse_B2L_maps[i].size();
    }

    IVector_h consolidated_coarse_neigh_to_fine_part; //consolidated list of coarse neighbors for the root partition, using fine partition indices
    int num_consolidated_neighbors = 0;
    // std::vector<IVector> consolidated_B2L_maps; //concatenates dest_coarse_B2L_maps received from partitions that are merging into the same root and pointing to the same destination coarse neighbor
    std::vector<IVector> &consolidated_B2L_maps = this->m_consolidated_B2L_maps;
    A.manager->consolidateB2LmapsOnRoot(num_consolidated_neighbors, consolidated_B2L_maps, consolidated_coarse_neigh_to_fine_part, dest_coarse_B2L_maps, coarse_neigh_to_fine_part, num_bdy_per_coarse_neigh, fine_parts_to_consolidate, num_fine_parts_to_consolidate, my_id, my_destination_part, is_root_partition, num_coarse_neighbors, A.manager->getComms());
    //
    // Step 9 - figuring out halo aggregate IDs
    //
    //Now we need to update halo aggregate IDs - this is just a halo exchange on this->m_aggregates between partitions
    //that are being merged together, but we need to send other halos to the root to come up with the halo renumbering
    //TODO: separate transactions, send "real halo" to the root nodes (coarse neighbors) immediately
    //Step 9.1: takes care of synchronizing the aggregate IDs between partitions we are merging together and got consistent halo aggregate IDs for neighbor we are not merging with (which are going to be sent to the root in 9.2)
    A.manager->exchange_halo(this->m_aggregates, 6666);
    /*
        * EXAMPLE 2
        This example is independent from the previous ones.
        Say partition 0 and 1 are merging (into 0) partition 0 is neighbors with 1,2,3 and partition 1 is neighbors with 0,3,4
        Partitions 3 and 4 are merging (into partition 3) and partition 2 is not merging with anyone.
        This example details the renumbering of halo indices on partition 0 and partition 1.
        After the exchange halo, we have:
        this->m_aggregates on partition 0:
        [(fine interior nodes) (fine boundary nodes) (fine halo from part 1) (fine halo from part 2) (fine halo from part 3)]
        [(fine interior nodes) (fine boundary nodes) (13 13 15) (12 15 17) (14 16 18)]
        aggregates on partition 1:
        [(fine interior nodes) (fine boundary nodes) (fine halo from part 0) (fine halo from part 3) (fine halo from part 4)]
        [(fine interior nodes) (fine boundary nodes) (14 16 17) (18 19 19) (15 15 17)]
        indices in  (fine halo from part 0) and (fine halo from part 1) actually contain interior aggregate indices (if they are not connected to partitions 2,3 or 4), because the boundary is disappearing there.
        Indices in halo regions contain remote-local indices.

        This example is used throughout consolidateAndRenumberHalos
        */
    //Step 9.2 - 9.5
    // IVector_h halo_offsets(num_consolidated_neighbors + 1, 0);
    IVector_h &halo_offsets = this->m_consolidated_halo_offsets;
    halo_offsets = IVector_h(num_consolidated_neighbors + 1, 0);
    A.manager->consolidateAndRenumberHalos(this->m_aggregates, A.manager->halo_offsets, halo_offsets, A.manager->neighbors, num_fine_neighbors, consolidated_coarse_neigh_to_fine_part, num_consolidated_neighbors, destination_part, my_destination_part, is_root_partition, fine_parts_to_consolidate, num_fine_parts_to_consolidate, num_parts, my_id, total_rows_in_merged, this->m_num_all_aggregates, A.manager->getComms());

    if (is_root_partition)
    {
        for (int i = 0; i < consolidated_B2L_maps.size(); i++)
        {
            amgx::thrust::sort(consolidated_B2L_maps[i].begin(), consolidated_B2L_maps[i].end());
        }

        this->m_consolidated_neighbors.resize(num_consolidated_neighbors);
        for (int i = 0; i < num_consolidated_neighbors; i++)
        {
            this->m_consolidated_neighbors[i] = fine_part_to_coarse_part[consolidated_coarse_neigh_to_fine_part[i]];
        }
            
        cudaCheckError();
    }
}

template <class T_Config>
void Aggregation_AMG_Level_Base<T_Config>::consolidateCoarseGridMatrix()
{
    Matrix<TConfig> &A = this->getA();
    Matrix<TConfig> &Ac = this->getNextLevel( MemorySpace( ) )->getA();

    int num_parts, num_fine_neighbors, my_id;

    num_parts = A.manager->getComms()->get_num_partitions();
    num_fine_neighbors = A.manager->neighbors.size();
    my_id = A.manager->global_id();
        
    IVector_h &destination_part = A.manager->getDestinationPartitions();
    int my_destination_part = A.manager->getMyDestinationPartition();

    // bookkeeping stored in AMG_Level_Base
    std::vector<IVector_h> &vertex_counts = this->m_vertex_counts;
    IVector_h &fine_parts_to_consolidate  = this->m_fine_parts_to_consolidate;

    // bookkeeping stored in either AMG_Level_Base or Acs' DistributedManager
    IVector_h &halo_offsets                     = this->isReuseLevel() ? Ac.manager->getHaloOffsets() : this->m_consolidated_halo_offsets;
    std::vector<IVector> &consolidated_B2L_maps = this->isReuseLevel() ? Ac.manager->getB2Lmaps()     : this->m_consolidated_B2L_maps;

    int num_consolidated_neighbors = this->isRootPartition() ? this->m_consolidated_neighbors.size() : 0;

    if (!this->isRootPartition())
    {
        A.manager->getComms()->send_vector_async(Ac.row_offsets, my_destination_part, 1111);
        A.manager->getComms()->send_vector_async(Ac.col_indices, my_destination_part, 1112);
        A.manager->getComms()->send_vector_async(Ac.values, my_destination_part, 1113);
    }
    else
    {
        int num_fine_parts_to_consolidate = fine_parts_to_consolidate.size();

        int total_num_rows = this->m_num_all_aggregates;
        IVector new_row_offsets(total_num_rows + 1, 0);

        //if diags are inside then we won't be counting those twice when computing halo row length
        if (!Ac.hasProps(DIAG))
        {
            thrust_wrapper::fill<TConfig::memSpace>(new_row_offsets.begin() + halo_offsets[0], new_row_offsets.begin() + halo_offsets[num_consolidated_neighbors], 1);
            cudaCheckError();
        }

        std::vector<IVector> recv_row_offsets(num_fine_parts_to_consolidate);
        std::vector<VecInt_t> num_nz(num_fine_parts_to_consolidate);
        IVector *work_row_offsets;
        std::vector<VecInt_t> index_offset_array(2 * num_fine_parts_to_consolidate + 1);
        int interior_offset = 0;
        int boundary_offset = 0;

        for (int i = 0; i < num_fine_parts_to_consolidate; i++)
        {
            boundary_offset += vertex_counts[i][0];
        }

        int max_num_nz = 0;

        for (int i = 0; i < num_fine_parts_to_consolidate; i++)
        {
            int current_part = fine_parts_to_consolidate[i];

            //receive row offsets
            if (current_part != my_id)
            {
                recv_row_offsets[i].resize(total_num_rows + 1);
                A.manager->getComms()->recv_vector(recv_row_offsets[i], current_part, 1111);
                work_row_offsets = &(recv_row_offsets[i]);
                num_nz[i] = (*work_row_offsets)[work_row_offsets->size() - 1];
                max_num_nz = max_num_nz > num_nz[i] ? max_num_nz : num_nz[i];
            }
            else
            {
                work_row_offsets = &(Ac.row_offsets);
                num_nz[i] = Ac.get_num_nz();
            }

            //Get interior row length
            thrust_wrapper::transform<TConfig::memSpace>(work_row_offsets->begin() + interior_offset + 1,
                                work_row_offsets->begin() + interior_offset + vertex_counts[i][0] + 1,
                                work_row_offsets->begin() + interior_offset,
                                new_row_offsets.begin() + interior_offset,
                                amgx::thrust::minus<IndexType>());
            cudaCheckError();
            //Get boundary row length
            thrust_wrapper::transform<TConfig::memSpace>(work_row_offsets->begin() + boundary_offset + 1,
                                work_row_offsets->begin() + boundary_offset + vertex_counts[i][1] + 1,
                                work_row_offsets->begin() + boundary_offset,
                                new_row_offsets.begin() + boundary_offset,
                                amgx::thrust::minus<IndexType>());
            cudaCheckError();
            //Increment halo row length by one for every nonzero that is an edge from the halo into this partition
            int size = halo_offsets[num_consolidated_neighbors] - halo_offsets[0];
            const int block_size = 128;
            const int num_blocks = std::min( AMGX_GRID_MAX_SIZE, (size - 1) / block_size + 1);
            set_halo_rowlen <<< num_blocks, block_size>>>(work_row_offsets->raw() + halo_offsets[0], new_row_offsets.raw() + halo_offsets[0], size, Ac.hasProps(DIAG));
            cudaCheckError();
            index_offset_array[i] = interior_offset;
            index_offset_array[num_fine_parts_to_consolidate + i] = boundary_offset;
            interior_offset += vertex_counts[i][0];
            boundary_offset += vertex_counts[i][1];
            index_offset_array[i + 1] = interior_offset;
            index_offset_array[num_fine_parts_to_consolidate + i + 1] = boundary_offset;
        }

        A.manager->setConsolidationArrayOffsets(index_offset_array);
        //Exclusive scan row length array to get row offsets
        thrust_wrapper::exclusive_scan<TConfig::memSpace>(new_row_offsets.begin(), new_row_offsets.end(), new_row_offsets.begin());
        cudaCheckError();
        //Prepare to receive column indices and values
        int num_nz_consolidated = new_row_offsets[new_row_offsets.size() - 1];
        IVector recv_col_indices(max_num_nz);
        IVector new_col_indices(num_nz_consolidated);
        MVector recv_values((max_num_nz + 1 + Ac.hasProps(DIAG) * (halo_offsets[num_consolidated_neighbors] - 1))*Ac.get_block_size());
        MVector new_values((num_nz_consolidated + 1 + Ac.hasProps(DIAG) * (halo_offsets[num_consolidated_neighbors] - 1))*Ac.get_block_size());
        thrust_wrapper::fill<TConfig::memSpace>(new_col_indices.begin() + new_row_offsets[halo_offsets[0]], new_col_indices.end(), -1); //Set all the halo col indices to -1


        if (!Ac.hasProps(DIAG)) { thrust_wrapper::fill<TConfig::memSpace>(new_values.begin() + num_nz_consolidated * Ac.get_block_size(), new_values.end(), types::util<ValueTypeA>::get_zero()); }

        cudaCheckError();
        IVector *work_col_indices;
        MVector *work_values;
        interior_offset = 0;
        boundary_offset = 0;

        for (int i = 0; i < num_fine_parts_to_consolidate; i++)
        {
            int current_part = fine_parts_to_consolidate[i];
            boundary_offset += vertex_counts[i][0];
        }

        for (int i = 0; i < num_fine_parts_to_consolidate; i++)
        {
            int current_part = fine_parts_to_consolidate[i];

            if (current_part != my_id)
            {
                A.manager->getComms()->recv_vector(recv_col_indices, current_part, 1112, 0, num_nz[i]);
                A.manager->getComms()->recv_vector(recv_values, current_part, 1113, 0, (num_nz[i] + 1 + Ac.hasProps(DIAG) * (halo_offsets[num_consolidated_neighbors] - 1))*Ac.get_block_size());
                work_col_indices = &(recv_col_indices);
                work_row_offsets = &(recv_row_offsets[i]);
                work_values = &(recv_values);
            }
            else
            {
                work_row_offsets = &(Ac.row_offsets);
                work_col_indices = &(Ac.col_indices);
                work_values = &(Ac.values);
            }

            //Put interior rows in place
            amgx::thrust::copy(work_col_indices->begin() + (*work_row_offsets)[interior_offset],
                            work_col_indices->begin() + (*work_row_offsets)[interior_offset + vertex_counts[i][0]],
                            new_col_indices.begin() + new_row_offsets[interior_offset]);
            cudaCheckError();
            amgx::thrust::copy(work_values->begin() + (*work_row_offsets)[interior_offset]*Ac.get_block_size(),
                            work_values->begin() + ((*work_row_offsets)[interior_offset + vertex_counts[i][0]])*Ac.get_block_size(),
                            new_values.begin() + new_row_offsets[interior_offset]*Ac.get_block_size());
            cudaCheckError();
            //Put boundary rows in place
            amgx::thrust::copy(work_col_indices->begin() + (*work_row_offsets)[boundary_offset],
                            work_col_indices->begin() + (*work_row_offsets)[boundary_offset + vertex_counts[i][1]],
                            new_col_indices.begin() + new_row_offsets[boundary_offset]);
            cudaCheckError();
            amgx::thrust::copy(work_values->begin() + (*work_row_offsets)[boundary_offset]*Ac.get_block_size(),
                            work_values->begin() + ((*work_row_offsets)[boundary_offset + vertex_counts[i][1]])*Ac.get_block_size(),
                            new_values.begin() + new_row_offsets[boundary_offset]*Ac.get_block_size());
            cudaCheckError();
            //Process halo rows (merge)
            int size = halo_offsets[num_consolidated_neighbors] - halo_offsets[0];
            const int block_size = 128;
            const int num_blocks = std::min( AMGX_GRID_MAX_SIZE, (size - 1) / block_size + 1);
            //TODO: vectorise this kernel, will be inefficient for larger block sizes
            append_halo_nz <<< num_blocks, block_size>>>(work_row_offsets->raw() + halo_offsets[0],
                    new_row_offsets.raw() + halo_offsets[0],
                    work_col_indices->raw(),
                    new_col_indices.raw(),
                    work_values->raw(),
                    new_values.raw(),
                    size, Ac.hasProps(DIAG), halo_offsets[0], Ac.get_block_size());
            cudaCheckError();

            // Diagonals
            if (Ac.hasProps(DIAG))
            {
                // Diagonal corresponding to interior rows
                amgx::thrust::copy(work_values->begin() + (num_nz[i] + interior_offset)*Ac.get_block_size(),
                                work_values->begin() + (num_nz[i] + interior_offset + vertex_counts[i][0])*Ac.get_block_size(),
                                new_values.begin() + (new_row_offsets[halo_offsets[halo_offsets.size() - 1]] + interior_offset)*Ac.get_block_size());
                // Diagonal corresponding to boundary rows
                amgx::thrust::copy(work_values->begin() + (num_nz[i] + boundary_offset)*Ac.get_block_size(),
                                work_values->begin() + (num_nz[i] + boundary_offset + vertex_counts[i][1])*Ac.get_block_size(),
                                new_values.begin() + (new_row_offsets[halo_offsets[halo_offsets.size() - 1]] + boundary_offset)*Ac.get_block_size());
                cudaCheckError();
            }

            interior_offset += vertex_counts[i][0];
            boundary_offset += vertex_counts[i][1];
        }

        Ac.set_initialized(0);
        Ac.row_offsets = new_row_offsets;
        Ac.col_indices = new_col_indices;
        Ac.values = new_values;
    }

    // A new distributed communicator for coarse levels that only contains active partitions
    // has already been created in consolidatedBookKeeping!

    //
    // Step 12 - finalizing, bookkeping
    //
    if (this->isRootPartition())
    {
        // int my_consolidated_id = fine_part_to_coarse_part[my_id];
        int my_consolidated_id = Ac.manager->getComms()->get_global_id();

        if (!this->isReuseLevel())
        {
             Ac.manager->initializeAfterConsolidation(
                 my_consolidated_id,
                 Ac,
                this->m_consolidated_neighbors,
                this->m_total_interior_rows_in_merged,
                this->m_total_boundary_rows_in_merged,
                this->m_num_all_aggregates,
                this->m_consolidated_halo_offsets,
                this->m_consolidated_B2L_maps,
                1,
                true);

            // this is now stored in Acs DistributedManager
            this->m_consolidated_neighbors.resize(0);
            this->m_consolidated_halo_offsets.resize(0);
            this->m_consolidated_B2L_maps.resize(0);

            Ac.manager->B2L_rings.resize(num_consolidated_neighbors + 1);

            for (int i = 0; i < num_consolidated_neighbors; i++)
            {
                Ac.manager->B2L_rings[i].resize(2);
                Ac.manager->B2L_rings[i][0] = 0;
                Ac.manager->B2L_rings[i][1] = consolidated_B2L_maps[i].size();
            }
        }

        Ac.manager->set_initialized(Ac.row_offsets);
        Ac.manager->getComms()->set_neighbors(num_consolidated_neighbors);
        int new_nnz = Ac.row_offsets[Ac.row_offsets.size() - 1];

        Ac.set_num_nz(new_nnz);
        Ac.set_num_cols(Ac.manager->halo_offsets[Ac.manager->halo_offsets.size() - 1]);
        Ac.set_num_rows(Ac.get_num_cols());

        if (A.hasProps(DIAG)) { Ac.addProps(DIAG); }

        Ac.computeDiagonal();
        Ac.set_initialized(1);
    }
    else
    {
        this->getA().manager->getComms()->send_vector_wait_all(Ac.row_offsets);
        this->getA().manager->getComms()->send_vector_wait_all(Ac.col_indices);
        this->getA().manager->getComms()->send_vector_wait_all(Ac.values);

        Ac.set_initialized(0);
        // set size of Ac to be zero
        Ac.resize(0, 0, 0, 1);
        Ac.set_initialized(1);
    }
}

// -------------------------------------------------------------
// Explicit instantiations
// -------------------------------------------------------------

#define AMGX_CASE_LINE(CASE) template class Aggregation_AMG_Level<TemplateMode<CASE>::Type>;
AMGX_FORALL_BUILDS(AMGX_CASE_LINE)
AMGX_FORCOMPLEX_BUILDS(AMGX_CASE_LINE)
#undef AMGX_CASE_LINE
}

}
