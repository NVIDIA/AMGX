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

#include <aggregation/selectors/size4_selector.h>
#include <cutil.h>
#include <util.h>
#include <types.h>
#include <basic_types.h>
#include <texture.h>

#include <thrust/count.h> //count
#include <thrust/sort.h> //sort
#include <thrust/unique.h> //unique
#include <cusp/detail/format_utils.h> //offsets_to_indices

namespace amgx
{
namespace aggregation
{
namespace size4_selector
{

// include common routines for all selectors
#include <aggregation/selectors/common_selector.h>

// ------------------------
//  Kernels
// ------------------------

#ifndef DELETE
// Kernel to compute the weight of the edges with block_dia_csr format
template <typename IndexType, typename ValueType>
__global__
void computeEdgeWeightsBlockDiaCsr(const IndexType *row_offsets, const IndexType *column_indices,
                                   const IndexType *dia_idx, const ValueType *nonzero_values, const IndexType num_block_rows, float *edge_weights, int bsize)
{
    int tid = threadIdx.x + blockDim.x * blockIdx.x;
    int jcol;
    int bsize_sq = bsize * bsize;

    while (tid < num_block_rows)
    {
        for (int j = row_offsets[tid]; j < row_offsets[tid + 1]; j++)
        {
            jcol = column_indices[j];

            if (jcol >= num_block_rows) { continue; }

            // Compute edge weight
            for (int k = row_offsets[jcol]; k < row_offsets[jcol + 1]; k++)
            {
                if (column_indices[k] == tid)
                {
                    edge_weights[j] = 0.5 * (fabs(nonzero_values[j * bsize_sq]) + fabs(nonzero_values[k * bsize_sq]))
                                      / max( fabs(nonzero_values[bsize_sq * dia_idx[tid]]), fabs(nonzero_values[bsize_sq * dia_idx[jcol]]));
                    break;
                }
            }
        }

        tid += gridDim.x * blockDim.x;
    }
}
#endif

// -----------------
//  Methods
// ----------------

// Constructor
template<class T_Config>
Size4SelectorBase<T_Config>::Size4SelectorBase(AMG_Config &cfg, const std::string &cfg_scope)
{
    deterministic = cfg.AMG_Config::template getParameter<IndexType>("determinism_flag", "default");
    max_iterations = cfg.AMG_Config::template getParameter<IndexType>("max_matching_iterations", cfg_scope);
    numUnassigned_tol = cfg.AMG_Config::template getParameter<double>("max_unassigned_percentage", cfg_scope);
    m_aggregation_edge_weight_component = cfg.AMG_Config::template getParameter<int>("aggregation_edge_weight_component", cfg_scope);
    weight_formula = cfg.AMG_Config::template getParameter<int>("weight_formula", cfg_scope);
}

template <AMGX_VecPrecision t_vecPrec, AMGX_MatPrecision t_matPrec, AMGX_IndPrecision t_indPrec>
void Size4Selector<TemplateConfig<AMGX_host, t_vecPrec, t_matPrec, t_indPrec> >::setAggregates_common_sqblock(const Matrix_h &A,
        IVector &aggregates, IVector &aggregates_global, int &num_aggregates)
{
    FatalError("OnePhaseHandshaking selector: setAggregates not implemented on CPU, exiting", AMGX_ERR_NOT_SUPPORTED_TARGET);
}


template <AMGX_VecPrecision t_vecPrec, AMGX_MatPrecision t_matPrec, AMGX_IndPrecision t_indPrec>
void Size4Selector<TemplateConfig<AMGX_device, t_vecPrec, t_matPrec, t_indPrec> >::setAggregates_common_sqblock(const Matrix_d &A,
        IVector &aggregates, IVector &aggregates_global, int &num_aggregates)
{
    // both ways are supported
    const IndexType num_block_rows = A.get_num_rows();
    const IndexType num_nonzero_blocks = A.get_num_nz();

    if (!A.is_matrix_singleGPU())
    {
        aggregates.resize(A.manager->halo_offset(A.manager->num_neighbors()));
    }
    else
    {
        aggregates.resize(num_block_rows);
    }

    // Initially, put each vertex in its own aggregate
    thrust_wrapper::sequence<AMGX_device>(aggregates.begin(), aggregates.begin() + num_block_rows);
    cudaCheckError();
    IndexType *aggregates_ptr = aggregates.raw();
    // Create row_indices array
    IndexType total_nz = A.is_matrix_singleGPU() ? num_nonzero_blocks : A.manager->num_nz_all();
    typename Matrix_d::IVector row_indices(total_nz);

    if (total_nz > 0)
    {
        cusp::detail::offsets_to_indices(A.row_offsets, row_indices);
    }

    const IndexType *A_row_offsets_ptr = A.row_offsets.raw();
    const IndexType *A_row_indices_ptr = row_indices.raw();
    const IndexType *A_column_indices_ptr = A.col_indices.raw();
    const IndexType *A_dia_idx_ptr = A.diag.raw();
    const ValueType *A_nonzero_values_ptr = A.values.raw();
    typename Matrix_d::IVector strongest_neighbour(num_block_rows, -1);
    typename Matrix_d::IVector partner_index(num_block_rows * 3, -1);
    IndexType *strongest_neighbour_ptr = strongest_neighbour.raw();
    IndexType *partner_index_ptr = partner_index.raw();
    const int threads_per_block = 256;
    const int num_blocks = std::min( AMGX_GRID_MAX_SIZE, (num_block_rows - 1) / threads_per_block + 1);
    int numUnassigned = num_block_rows;
    int numUnassigned_previous = numUnassigned;
    Vector<TemplateConfig<AMGX_device, AMGX_vecFloat, t_matPrec, t_indPrec> > edge_weights(num_nonzero_blocks, -1);
    float *edge_weights_ptr = edge_weights.raw();
    float *rand_edge_weights_ptr = NULL;
    // Compute the edge weights
    const int num_blocks_V2 = std::min( AMGX_GRID_MAX_SIZE, (num_nonzero_blocks - 1) / threads_per_block + 1);

    if (num_nonzero_blocks > 0)
    {
        cudaFuncSetCacheConfig(computeEdgeWeightsBlockDiaCsr_V2<IndexType, ValueType, float>, cudaFuncCachePreferL1);
        computeEdgeWeightsBlockDiaCsr_V2 <<< num_blocks_V2, threads_per_block>>>(A_row_offsets_ptr, A_row_indices_ptr, A_column_indices_ptr, A_dia_idx_ptr, A_nonzero_values_ptr, num_nonzero_blocks, edge_weights_ptr, rand_edge_weights_ptr, num_block_rows, A.get_block_dimy(), this->m_aggregation_edge_weight_component, this->weight_formula);
    }

    // -------------------------------------------------
    // First create aggregates of size 2
    // -------------------------------------------------
    int icount = 0;

    do
    {
        // For each block_row, find the strongest neighbour who hasn't been assigned
        findStrongestNeighbourBlockDiaCsr_NoMerge <<< num_blocks, threads_per_block>>>(A_row_offsets_ptr, A_column_indices_ptr, edge_weights_ptr, num_block_rows, partner_index_ptr, strongest_neighbour_ptr, this->deterministic);
        cudaCheckError();
        // Look for perfect matches
        matchEdges <<< num_blocks, threads_per_block>>>(num_block_rows, partner_index_ptr, aggregates_ptr, strongest_neighbour_ptr);
        cudaCheckError();
        numUnassigned_previous = numUnassigned;
        numUnassigned = (int)thrust_wrapper::count<AMGX_device>(partner_index.begin(), partner_index.end(), -1);
        cudaCheckError();
        icount++;
    }
    while (!(numUnassigned == 0 || icount > this->max_iterations || 1.0 * numUnassigned / num_block_rows < this->numUnassigned_tol || numUnassigned_previous == numUnassigned));

    assignUnassignedVertices <<< num_blocks, threads_per_block>>>(partner_index_ptr, num_block_rows);
    cudaCheckError();
    // -------------------------------------------------
    // Merge aggregates to create aggregates of size 4
    // -------------------------------------------------
    Vector<TemplateConfig<AMGX_device, AMGX_vecFloat, t_matPrec, t_indPrec> > weight_strongest_neighbour(num_block_rows, -1);
    float *weight_strongest_neighbour_ptr = weight_strongest_neighbour.raw();
    // now used as flag to check if aggregated or not
    typename Matrix_d::IVector aggregated(num_block_rows, -1);
    IndexType *aggregated_ptr = aggregated.raw();
    icount = 0;
    numUnassigned = num_block_rows;
    numUnassigned_previous = numUnassigned;

    do
    {
        // Each vertex stores in strongest_neighbour the aggregates number of strongest neighbour and the weight of connection
        findStrongestNeighbourBlockDiaCsr_StoreWeight <<< num_blocks, threads_per_block>>>(A_row_offsets_ptr, A_column_indices_ptr, edge_weights_ptr, num_block_rows, aggregated_ptr, aggregates_ptr, strongest_neighbour_ptr, partner_index_ptr, weight_strongest_neighbour_ptr, this->deterministic);
        cudaCheckError();
        // Each vertex in same aggregates will agree on aggregates to propose too, and both store the aggregate number they want to match with
        agreeOnProposal <<< num_blocks, threads_per_block>>>(A_row_offsets_ptr, A_column_indices_ptr, num_block_rows, aggregated_ptr, strongest_neighbour_ptr, weight_strongest_neighbour_ptr, partner_index_ptr, aggregates_ptr);
        cudaCheckError();
        matchAggregates <IndexType> <<< num_blocks, threads_per_block>>>(aggregates_ptr, aggregated_ptr, strongest_neighbour_ptr, num_block_rows);
        cudaCheckError();
        numUnassigned_previous = numUnassigned;
        numUnassigned = thrust_wrapper::count<AMGX_device>(aggregated.begin(), aggregated.end(), -1);
        cudaCheckError();
        icount++;
    }
    while (!(numUnassigned == 0 || icount > this->max_iterations || 1.0 * numUnassigned / num_block_rows < this->numUnassigned_tol || numUnassigned_previous == numUnassigned));

    // Merge remaining vertices with current aggregates

    if (!this->deterministic)
    {
        while (numUnassigned != 0)
        {
            mergeWithExistingAggregatesBlockDiaCsr <<< num_blocks, threads_per_block>>>(A_row_offsets_ptr, A_column_indices_ptr, edge_weights_ptr, num_block_rows, aggregates_ptr, aggregated_ptr, this->deterministic, (IndexType *) NULL);
            cudaCheckError();
            numUnassigned = (int)thrust_wrapper::count<AMGX_device>(aggregated.begin(), aggregated.end(), -1);
            cudaCheckError();
        };
    }
    else
    {
        typename Matrix_d::IVector aggregates_candidate(num_block_rows, -1);

        while (numUnassigned != 0)
        {
            mergeWithExistingAggregatesBlockDiaCsr <<< num_blocks, threads_per_block>>>(A_row_offsets_ptr, A_column_indices_ptr, edge_weights_ptr, num_block_rows, aggregates_ptr, aggregated_ptr, this->deterministic, aggregates_candidate.raw());
            cudaCheckError();
            joinExistingAggregates <<< num_blocks, threads_per_block>>>(num_block_rows, aggregates_ptr, aggregated_ptr, aggregates_candidate.raw());
            cudaCheckError();
            numUnassigned = (int)thrust_wrapper::count<AMGX_device>(aggregated.begin(), aggregated.end(), -1);
            cudaCheckError();
        };
    }

    this->renumberAndCountAggregates(aggregates, aggregates_global, num_block_rows, num_aggregates);
}

template <class T_Config>
void Size4SelectorBase<T_Config>::setAggregates(Matrix<T_Config> &A,
        IVector &aggregates, IVector &aggregates_global, int &num_aggregates)
{
    if (A.get_block_dimx() == A.get_block_dimy())
    {
        setAggregates_common_sqblock( A, aggregates, aggregates_global, num_aggregates );
    }
    else
    {
        FatalError("Unsupported block size for Size4Selector", AMGX_ERR_NOT_SUPPORTED_BLOCKSIZE);
    }
}

// -------------------------
//  Explict instantiations
// -------------------------
#define AMGX_CASE_LINE(CASE) template class Size4SelectorBase<TemplateMode<CASE>::Type>;
AMGX_FORALL_BUILDS(AMGX_CASE_LINE)
#undef AMGX_CASE_LINE
#define AMGX_CASE_LINE(CASE) template class Size4Selector<TemplateMode<CASE>::Type>;
AMGX_FORALL_BUILDS(AMGX_CASE_LINE)
#undef AMGX_CASE_LINE
}
}
}
