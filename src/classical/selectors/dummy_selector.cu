// SPDX-FileCopyrightText: 2011 - 2024 NVIDIA CORPORATION. All Rights Reserved.
//
// SPDX-License-Identifier: BSD-3-Clause

#include <classical/selectors/dummy_selector.h>
#include <classical/interpolators/common.h>
#include <cutil.h>
#include <types.h>

#include<thrust/count.h>

namespace amgx
{

namespace classical
{
/*************************************************************************
 * Perform the initial marking for each row
 ************************************************************************/
template <typename IndexType>
__global__
void initialMarkingKernel(IndexType *cf_map, const IndexType numRows)
{
    for ( int tIdx = threadIdx.x + blockDim.x * blockIdx.x; tIdx < numRows; tIdx += blockDim.x * gridDim.x )
    {
        cf_map[tIdx] = (tIdx % 2) ? COARSE : FINE;
    }
}



/*************************************************************************
 * mark coarse and fine points
 ************************************************************************/
template <AMGX_VecPrecision t_vecPrec, AMGX_MatPrecision t_matPrec, AMGX_IndPrecision t_indPrec>
void Dummy_Selector<TemplateConfig<AMGX_host, t_vecPrec, t_matPrec, t_indPrec> >
::markCoarseFinePoints_1x1(Matrix_h &A,
                           FVector &weights,
                           const BVector &s_con,
                           IVector &cf_map,
                           IVector &scratch,
                           int cf_map_init)
{
    FatalError("DummySelector not implemented on host", AMGX_ERR_NOT_IMPLEMENTED);
}

template <AMGX_VecPrecision t_vecPrec, AMGX_MatPrecision t_matPrec, AMGX_IndPrecision t_indPrec>
void Dummy_Selector<TemplateConfig<AMGX_device, t_vecPrec, t_matPrec, t_indPrec> >::markCoarseFinePoints_1x1(Matrix_d &A,
        FVector &weights,
        const BVector &s_con,
        IVector &cf_map,
        IVector &scratch,
        int cf_map_init)
{
    if (A.hasProps(DIAG))
    {
        FatalError("Unsupported separate diag", AMGX_ERR_NOT_IMPLEMENTED);
    }

    typedef typename Matrix_d::index_type IndexType;
    typedef typename Matrix_d::value_type ValueType;
    // Choose blocksize. Using 1 thread / row for now
    const int blockSize = 256;
    const int numBlocks = std::min (AMGX_GRID_MAX_SIZE, (int) ((A.get_num_rows() + blockSize - 1) / blockSize));
    const int numRows = (int) A.get_num_rows();
    // raw pointers from the cusp arrays
    const IndexType *offsets_ptr = A.row_offsets.raw();
    const IndexType *column_indices_ptr = A.col_indices.raw();
    const ValueType *values_ptr = A.values.raw();
    float *weights_ptr = weights.raw();
    const bool *s_con_ptr = s_con.raw();
    int *scratch_ptr = scratch.raw();
    // do the initial marking of points
    initialMarkingKernel <<< numBlocks, blockSize>>>(cf_map.raw(), A.get_num_rows());
    cudaCheckError();
    cf_map.dirtybit = 1;

    // Do a two ring exchange before
    // TODO: This should query the interpolator to figure out what is needed in more general case
    // By default we do 2 rings in classical
    if (!A.is_matrix_singleGPU())
    {
        A.manager->exchange_halo_2ring(cf_map, cf_map.tag);
    }
}

template <class T_Config>
void Dummy_SelectorBase< T_Config>::markCoarseFinePoints(Matrix< T_Config> &A,
        FVector &weights,
        const BVector &s_con,
        IVector &cf_map,
        IVector &scratch,
        int cf_map_init)
{
    ViewType oldView = A.currentView();
    A.setView(OWNED);

    if (A.get_block_size() == 1)
    {
        markCoarseFinePoints_1x1(A, weights, s_con, cf_map, scratch);
    }
    else
    {
        FatalError("Unsupported block size Dummy selector", AMGX_ERR_NOT_SUPPORTED_BLOCKSIZE);
    }

    A.setView(oldView);
    // Mark cf_map as dirty
    cf_map.dirtybit = 1;
}

/****************************************
 * Explict instantiations
 ***************************************/
#define AMGX_CASE_LINE(CASE) template class Dummy_SelectorBase<TemplateMode<CASE>::Type>;
AMGX_FORALL_BUILDS(AMGX_CASE_LINE)
#undef AMGX_CASE_LINE

#define AMGX_CASE_LINE(CASE) template class Dummy_Selector<TemplateMode<CASE>::Type>;
AMGX_FORALL_BUILDS(AMGX_CASE_LINE)
#undef AMGX_CASE_LINE

} // namespace classical

} // namespace amgx
