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
#define COARSE_CLA_CONSO 0

#include <classical/classical_amg_level.h>
#include <amg_level.h>

#include <basic_types.h>
#include <cutil.h>
#include <multiply.h>
#include <transpose.h>
#include <truncate.h>
#include <blas.h>
#include <util.h>
#include <thrust/logical.h>
#include <thrust/remove.h>
#include <thrust/adjacent_difference.h>
#include <thrust_wrapper.h>

#include <thrust/extrema.h> // for minmax_element

#include <algorithm>
#include <assert.h>
#include <matrix_io.h>

#include <csr_multiply.h>

#include <thrust/logical.h>
#include <thrust/count.h>
#include <thrust/sort.h>

#include <profile.h>
#include <distributed/glue.h>
namespace amgx
{

namespace classical
{

void __global__ profiler_tag_1() {}
void __global__ profiler_tag_2() {}
void __global__ profiler_tag_3() {}

struct is_zero
{
    __host__ __device__
    bool operator()(const double &v)
    {
        return fabs(v) < 1e-10;
    }
};

#define AMGX_CAL_BLOCK_SIZE 256

/* There might be a situation where not all local_to_global_map columns are present in the matrix (because some rows were removed
   and the columns in these rows are therefore no longer present. This kernel creates the flags array that marks existing columns. */
template<typename ind_t>
__global__ __launch_bounds__( AMGX_CAL_BLOCK_SIZE )
void flag_existing_local_to_global_columns(ind_t n, ind_t *row_offsets, ind_t *col_indices, ind_t *flags)
{
    ind_t i, j, s, e, col;

    //go through the matrix
    for (i = threadIdx.x + blockIdx.x * blockDim.x; i < n; i += blockDim.x * gridDim.x)
    {
        s = row_offsets[i];
        e = row_offsets[i + 1];

        for (j = s; j < e; j++)
        {
            col = col_indices[j];

            //flag columns outside of the square part (which correspond to local_to_global_map)
            if (col >= n)
            {
                flags[col - n] = 1;
            }
        }
    }
}

/* Renumber the indices based on the prefix-scan/sum of the flags array */
template<typename ind_t>
__global__ __launch_bounds__( AMGX_CAL_BLOCK_SIZE )
void compress_existing_local_columns(ind_t n, ind_t *row_offsets, ind_t *col_indices, ind_t *flags)
{
    ind_t i, j, s, e, col;

    //go through the matrix
    for (i = threadIdx.x + blockIdx.x * blockDim.x; i < n; i += blockDim.x * gridDim.x)
    {
        s = row_offsets[i];
        e = row_offsets[i + 1];

        for (j = s; j < e; j++)
        {
            col = col_indices[j];

            //flag columns outside of the square part (which correspond to local_to_global_map)
            if (col >= n)
            {
                col_indices[j] = n + flags[col - n];
            }
        }
    }
}

/* compress the local to global columns indices based on the prefix-scan/sum of the flags array */
template<typename ind_t, typename ind64_t>
__global__ __launch_bounds__( AMGX_CAL_BLOCK_SIZE )
void compress_existing_local_to_global_columns(ind_t n, ind64_t *l2g_in, ind64_t *l2g_out, ind_t *flags)
{
    ind_t i;

    //go through the arrays (and copy the updated indices when needed)
    for (i = threadIdx.x + blockIdx.x * blockDim.x; i < n; i += blockDim.x * gridDim.x)
    {
        if (flags[i] != flags[i + 1])
        {
            l2g_out[flags[i]] = l2g_in[i];
        }
    }
}


template <class T_Config>
Selector<T_Config> *chooseAggressiveSelector(AMG_Config *m_cfg, std::string std_scope)
{
    AMG_Config cfg;
    std::string cfg_string("");
    cfg_string += "default:";
    // if necessary, allocate aggressive selector + interpolator
    bool use_pmis = false, use_hmis = false;
    // default argument - use the same selector as normal coarsening
    std::string agg_selector = m_cfg->AMG_Config::getParameter<std::string>("aggressive_selector", std_scope);

    if (agg_selector == "DEFAULT")
    {
        std::string std_selector = m_cfg->AMG_Config::getParameter<std::string>("selector", std_scope);

        if      (std_selector == "PMIS") { cfg_string += "selector=AGGRESSIVE_PMIS"; use_pmis = true; }
        else if (std_selector == "HMIS") { cfg_string += "selector=AGGRESSIVE_HMIS"; use_hmis = true; }
        else
        {
            FatalError("Must use either PMIS or HMIS algorithms with aggressive coarsening", AMGX_ERR_NOT_IMPLEMENTED);
        }
    }
    // otherwise use specified selector
    else if (agg_selector == "PMIS") { cfg_string += "selector=AGGRESSIVE_PMIS"; use_pmis = true; }
    else if (agg_selector == "HMIS") { cfg_string += "selector=AGGRESSIVE_HMIS"; use_hmis = true; }
    else
    {
        FatalError("Invalid aggressive coarsener selected", AMGX_ERR_NOT_IMPLEMENTED);
    }

    // check a selector has been selected
    if (!use_pmis && !use_hmis)
    {
        FatalError("No aggressive selector chosen", AMGX_ERR_NOT_IMPLEMENTED);
    }

    cfg.parseParameterString(cfg_string.c_str());
    // now allocate the selector and interpolator
    return classical::SelectorFactory<T_Config>::allocate(cfg, "default" /*std_scope*/);
}

template <class T_Config>
Interpolator<T_Config> *chooseAggressiveInterpolator(AMG_Config *m_cfg, std::string std_scope)
{
    // temporary config and pointer to main config
    AMG_Config cfg;
    std::string cfg_string("");
    cfg_string += "default:";
    // Set the interpolator
    cfg_string += "interpolator=";
    cfg_string += m_cfg->AMG_Config::getParameter<std::string>("aggressive_interpolator", std_scope);
    cfg.parseParameterString(cfg_string.c_str());
    // now allocate the selector and interpolator
    return InterpolatorFactory<T_Config>::allocate(cfg, "default" /*std_scope*/);
}

template <class T_Config>
Classical_AMG_Level_Base<T_Config>::Classical_AMG_Level_Base(AMG_Class *amg) : AMG_Level<T_Config>(amg)
{
    strength = StrengthFactory<T_Config>::allocate(*(amg->m_cfg), amg->m_cfg_scope);
    selector = classical::SelectorFactory<T_Config>::allocate(*(amg->m_cfg), amg->m_cfg_scope);
    interpolator = InterpolatorFactory<T_Config>::allocate(*(amg->m_cfg), amg->m_cfg_scope);
    trunc_factor = amg->m_cfg->AMG_Config::getParameter<double>("interp_truncation_factor", amg->m_cfg_scope);
    max_elmts = amg->m_cfg->AMG_Config::getParameter<int>("interp_max_elements", amg->m_cfg_scope);
    max_row_sum = amg->m_cfg->AMG_Config::getParameter<double>("max_row_sum", amg->m_cfg_scope);
    num_aggressive_levels = amg->m_cfg->AMG_Config::getParameter<int>("aggressive_levels", amg->m_cfg_scope);
}

template <class T_Config>
Classical_AMG_Level_Base<T_Config>::~Classical_AMG_Level_Base()
{
    delete strength;
    delete selector;
    delete interpolator;
}

template <class T_Config>
void Classical_AMG_Level_Base<T_Config>::transfer_level(AMG_Level<TConfig1> *ref_lvl)
{
    Classical_AMG_Level_Base<TConfig1> *ref_cla_lvl = dynamic_cast<Classical_AMG_Level_Base<TConfig1>*>(ref_lvl);
    this->P.copy(ref_cla_lvl->P);
    this->R.copy(ref_cla_lvl->R);
    this->m_s_con.copy(ref_cla_lvl->m_s_con);
    this->m_scratch.copy(ref_cla_lvl->m_scratch);
    this->m_cf_map.copy(ref_cla_lvl->m_cf_map);
}

/****************************************
 * Computes the A, P, and R operators
 ***************************************/
template <class T_Config>
void Classical_AMG_Level_Base<T_Config>::createCoarseVertices()
{
    if (AMG_Level<T_Config>::getLevelIndex() < this->num_aggressive_levels)
    {
        if (selector) { delete selector; }

        selector = chooseAggressiveSelector<T_Config>(AMG_Level<T_Config>::amg->m_cfg, AMG_Level<T_Config>::amg->m_cfg_scope);
    }

    Matrix<T_Config> &RAP = this->getNextLevel( typename Matrix<T_Config>::memory_space( ) )->getA( );
    Matrix<T_Config> &A = this->getA();
    int size_all, size_full, nnz_full;

    if (!A.is_matrix_singleGPU())
    {
        int offset;
        // Need to get number of 2-ring rows
        A.getOffsetAndSizeForView(ALL, &offset, &size_all);
        A.getOffsetAndSizeForView(FULL, &offset, &size_full);
        A.getNnzForView(FULL, &nnz_full);
    }
    else
    {
        size_all = A.get_num_rows();
        size_full = A.get_num_rows();
        nnz_full = A.get_num_nz();
    }

    this->m_cf_map.resize(size_all);
    this->m_s_con.resize(nnz_full);
    this->m_scratch.resize(size_full);
    thrust::fill(this->m_cf_map.begin(), this->m_cf_map.end(), 0);
    cudaCheckError();
    thrust::fill(this->m_s_con.begin(), this->m_s_con.end(), false);
    cudaCheckError();
    thrust::fill(this->m_scratch.begin(), this->m_scratch.end(), 0);
    cudaCheckError();
    markCoarseFinePoints();
}

template <class T_Config>
void Classical_AMG_Level_Base<T_Config>::createCoarseMatrices()
{
    // allocate aggressive interpolator if needed
    if (AMG_Level<T_Config>::getLevelIndex() < this->num_aggressive_levels)
    {
        if (interpolator) { delete interpolator; }

        interpolator = chooseAggressiveInterpolator<T_Config>(AMG_Level<T_Config>::amg->m_cfg, AMG_Level<T_Config>::amg->m_cfg_scope);
    }

    Matrix<T_Config> &RAP = this->getNextLevel( typename Matrix<T_Config>::memory_space( ) )->getA( );
    Matrix<T_Config> &A = this->getA();
    /* WARNING: exit if D1 interpolator is selected in distributed setting */
    std::string s("");
    s += AMG_Level<T_Config>::amg->m_cfg->AMG_Config::getParameter<std::string>("interpolator", AMG_Level<T_Config>::amg->m_cfg_scope);

    if (A.is_matrix_distributed() && (s.compare("D1") == 0))
    {
        FatalError("D1 interpolation is not supported in distributed settings", AMGX_ERR_NOT_IMPLEMENTED);
    }

    /* WARNING: do not recompute prolongation (P) and restriction (R) when you
                are reusing the level structure (structure_reuse_levels > 0) */
    if (this->isReuseLevel() == false)
    {
        computeProlongationOperator();
    }

    // Compute Restriction operator and coarse matrix Ac
    if (!this->A->is_matrix_distributed() || this->A->manager->get_num_partitions() == 1)
    {
        /* WARNING: see above warning. */
        if (this->isReuseLevel() == false)
        {
            computeRestrictionOperator();
        }

        computeAOperator();
    }
    else
    {
        /* WARNING: notice that in this case the computeRestructionOperator() is called
                    inside computeAOperator_distributed() routine. */
        computeAOperator_distributed();
    }

// we also need to renumber columns of P and rows or R correspondingly since we changed RAP halo columns
// for R we just keep track of renumbering in and exchange proper vectors in restriction
// for P we actually need to renumber columns for prolongation:
    if (A.is_matrix_distributed() && this->A->manager->get_num_partitions() > 1)
    {
        RAP.set_initialized(0);
        // Renumber the owned nodes as interior and boundary (renumber rows and columns)
        // We are passing reuse flag to not create neighbours list from scratch, but rather update based on new halos
        RAP.manager->renumberMatrixOneRing(this->isReuseLevel());
        // Renumber the column indices of P and shuffle rows of P
        RAP.manager->renumber_P_R(this->P, this->R, A);
        // Create the B2L_maps for RAP
        RAP.manager->createOneRingHaloRows();
        RAP.manager->getComms()->set_neighbors(RAP.manager->num_neighbors());
        RAP.setView(OWNED);
        RAP.set_initialized(1);
        // update # of columns in P - this is necessary for correct CSR multiply
        P.set_initialized(0);
        int new_num_cols = thrust_wrapper::reduce(P.col_indices.begin(), P.col_indices.end(), int(0), thrust::maximum<int>()) + 1;
        cudaCheckError();
        P.set_num_cols(new_num_cols);
        P.set_initialized(1);
    }

    RAP.copyAuxData(&A);

    if (!A.is_matrix_singleGPU() && RAP.manager == NULL)
    {
        RAP.manager = new DistributedManager<TConfig>();
    }

    if (this->getA().is_matrix_singleGPU())
    {
        this->m_next_level_size = this->getNextLevel(typename Matrix<TConfig>::memory_space() )->getA().get_num_rows() * this->getNextLevel(typename Matrix<TConfig>::memory_space() )->getA().get_block_dimy();
    }
    else
    {
        // m_next_level_size is the size that will be used to allocate xc, bc vectors
        int size, offset;
        this->getNextLevel(typename Matrix<TConfig>::memory_space())->getA().getOffsetAndSizeForView(FULL, &offset, &size);
        this->m_next_level_size = size * this->getNextLevel(typename Matrix<TConfig>::memory_space() )->getA().get_block_dimy();
    }
}

template <class T_Config>
void Classical_AMG_Level_Base<T_Config>::markCoarseFinePoints()
{
    Matrix<T_Config> &A = this->getA();
    //allocate necessary memory
    typedef Vector<typename TConfig::template setVecPrec<AMGX_vecInt>::Type> IVector;
    typedef Vector<typename TConfig::template setVecPrec<AMGX_vecBool>::Type> BVector;
    typedef Vector<typename TConfig::template setVecPrec<AMGX_vecFloat>::Type> FVector;
    FVector weights;

    if (!A.is_matrix_singleGPU())
    {
        int size, offset;
        A.getOffsetAndSizeForView(FULL, &offset, &size);
        // size should now contain the number of 1-ring rows
        weights.resize(size);
    }
    else
    {
        weights.resize(A.get_num_rows());
    }

    thrust::fill(weights.begin(), weights.end(), 0.0);
    cudaCheckError();

    // extend A to include 1st ring nodes
    // compute strong connections and weights
    if (!A.is_matrix_singleGPU())
    {
        ViewType oldView = A.currentView();
        A.setView(FULL);
        strength->computeStrongConnectionsAndWeights(A, this->m_s_con, weights, this->max_row_sum);
        A.setView(oldView);
    }
    else
    {
        strength->computeStrongConnectionsAndWeights(A, this->m_s_con, weights, this->max_row_sum);
    }

    // Exchange the one-ring of the weights
    if (!A.is_matrix_singleGPU())
    {
        A.manager->exchange_halo(weights, weights.tag);
    }

    //mark coarse and fine points
    selector->markCoarseFinePoints(A, weights, this->m_s_con, this->m_cf_map, this->m_scratch);
    // we do resize cf_map to zero later, so we are saving separate copy
    this->m_cf_map.dirtybit = 1;

    // Do a two ring exchange of cf_map
    if (!A.is_matrix_singleGPU())
    {
        A.manager->exchange_halo_2ring(this->m_cf_map, m_cf_map.tag);
    }

    // Modify cf_map array such that coarse points are assigned a local index, while fine points entries are not touched
    selector->renumberAndCountCoarsePoints(this->m_cf_map, this->m_num_coarse_vertices, A.get_num_rows());
}


template <class T_Config>
void Classical_AMG_Level_Base<T_Config>::computeProlongationOperator()
{
    this->Profile.tic("computeP");
    Matrix<T_Config> &A = this->getA();
    //allocate necessary memory
    typedef Vector<typename TConfig::template setVecPrec<AMGX_vecInt>::Type> IVector;
    typedef Vector<typename TConfig::template setVecPrec<AMGX_vecBool>::Type> BVector;
    typedef Vector<typename TConfig::template setVecPrec<AMGX_vecFloat>::Type> FVector;
    //generate the interpolation matrix
    interpolator->generateInterpolationMatrix(A, this->m_cf_map, this->m_s_con, this->m_scratch, P, AMG_Level<TConfig>::amg);
    this->m_cf_map.clear();
    this->m_cf_map.shrink_to_fit();
    this->m_scratch.clear();
    this->m_scratch.shrink_to_fit();
    this->m_s_con.clear();
    this->m_s_con.shrink_to_fit();
    profileSubphaseTruncateP();

    // truncate based on max # of elements if desired
    if (this->max_elmts > 0 && P.get_num_rows() > 0)
    {
        Truncate<TConfig>::truncateByMaxElements(P, this->max_elmts);
    }

    if (!P.isLatencyHidingEnabled(*this->amg->m_cfg))
    {
        // This will cause bsrmv_with_mask to not do latency hiding
        P.setInteriorView(OWNED);
        P.setExteriorView(OWNED);
    }

    profileSubphaseNone();
    this->Profile.toc("computeP");
}

/**********************************************
 * computes R=P^T
 **********************************************/
template <class T_Config>
void Classical_AMG_Level_Base<T_Config>::computeRestrictionOperator()
{
    this->Profile.tic("computeR");
    R.set_initialized(0);
    P.setView(OWNED);
    transpose(P, R, P.get_num_rows());

    if (!R.isLatencyHidingEnabled(*this->amg->m_cfg))
    {
        // This will cause bsrmv_with_mask_restriction to not do latency hiding
        R.setInteriorView(OWNED);
        R.setExteriorView(OWNED);
    }

    if(P.is_matrix_distributed())
    {
        // Setup the number of non-zeros in R using stub DistributedManager
        R.manager = new DistributedManager<T_Config>();
        int nrows_owned = P.manager->halo_offsets[0];
        int nrows_full = P.manager->halo_offsets[P.manager->neighbors.size()];
        int nz_full = R.row_offsets[nrows_full];
        int nz_owned = R.row_offsets[nrows_owned];
        R.manager->setViewSizes(nrows_owned, nz_owned, nrows_owned, nz_owned, nrows_full, nz_full, R.get_num_rows(), R.get_num_nz());
    }

    R.set_initialized(1);
    this->Profile.toc("computeR");
}

/**********************************************
 * computes the Galerkin product: A_c=R*A*P
 **********************************************/

template <AMGX_VecPrecision t_vecPrec, AMGX_MatPrecision t_matPrec, AMGX_IndPrecision t_indPrec>
void Classical_AMG_Level<TemplateConfig<AMGX_host, t_vecPrec, t_matPrec, t_indPrec> >::computeAOperator_1x1()
{
    this->Profile.tic("computeA");
    Matrix<TConfig_h> RA;
    RA.addProps(CSR);
    RA.set_block_dimx(this->getA().get_block_dimx());
    RA.set_block_dimy(this->getA().get_block_dimy());
    Matrix<TConfig_h> &RAP = this->getNextLevel( typename Matrix<TConfig_h>::memory_space( ) )->getA( );
    RAP.addProps(CSR);
    RAP.set_block_dimx(this->getA().get_block_dimx());
    RAP.set_block_dimy(this->getA().get_block_dimy());
    Matrix<TConfig_h> &Atmp = this->getA();
    multiplyMM(this->R, this->getA(), RA);
    multiplyMM(RA, this->P, RAP);
    RAP.sortByRowAndColumn();
    RAP.set_initialized(1);
    this->Profile.toc("computeA");
}

template <AMGX_VecPrecision t_vecPrec, AMGX_MatPrecision t_matPrec, AMGX_IndPrecision t_indPrec>
void Classical_AMG_Level<TemplateConfig<AMGX_host, t_vecPrec, t_matPrec, t_indPrec> >::computeAOperator_1x1_distributed()
{
    FatalError("Distributed classical AMG not implemented for host\n", AMGX_ERR_NOT_IMPLEMENTED);
}


template <AMGX_VecPrecision t_vecPrec, AMGX_MatPrecision t_matPrec, AMGX_IndPrecision t_indPrec>
void Classical_AMG_Level<TemplateConfig<AMGX_device, t_vecPrec, t_matPrec, t_indPrec> >::computeAOperator_1x1()
{
    this->Profile.tic("computeA");
    Matrix<TConfig_d> &RAP = this->getNextLevel( device_memory( ) )->getA( );
    RAP.addProps(CSR);
    RAP.set_block_dimx(this->getA().get_block_dimx());
    RAP.set_block_dimy(this->getA().get_block_dimy());
    this->R.set_initialized( 0 );
    this->R.addProps( CSR );
    this->R.set_initialized( 1 );
    this->P.set_initialized( 0 );
    this->P.addProps( CSR );
    this->P.set_initialized( 1 );
    void *wk = AMG_Level<TConfig_d>::amg->getCsrWorkspace();

    if ( wk == NULL )
    {
        wk = CSR_Multiply<TConfig_d>::csr_workspace_create( *(AMG_Level<TConfig_d>::amg->m_cfg), AMG_Level<TConfig_d>::amg->m_cfg_scope );
        AMG_Level<TConfig_d>::amg->setCsrWorkspace( wk );
    }

    int spmm_verbose = this->amg->m_cfg->AMG_Config::getParameter<int>("spmm_verbose", this->amg->m_cfg_scope);

    if ( spmm_verbose )
    {
        typedef typename Matrix<TConfig_d>::IVector::const_iterator Iterator;
        typedef thrust::pair<Iterator, Iterator> Result;
        std::ostringstream buffer;
        buffer << "SPMM: Level " << this->getLevelIndex() << std::endl;

        if ( this->getLevelIndex() == 0 )
        {
            device_vector_alloc<int> num_nz( this->getA().row_offsets.size() );
            thrust::adjacent_difference( this->getA().row_offsets.begin(), this->getA().row_offsets.end(), num_nz.begin() );
            cudaCheckError();
            Result result = thrust::minmax_element( num_nz.begin() + 1, num_nz.end() );
            cudaCheckError();
            int min_size = *result.first;
            int max_size = *result.second;
            int sum = thrust_wrapper::reduce( num_nz.begin() + 1, num_nz.end() );
            cudaCheckError();
            double avg_size = double(sum) / this->getA().get_num_rows();
            buffer << "SPMM: A: " << std::endl;
            buffer << "SPMM: Matrix avg row size: " << avg_size << std::endl;
            buffer << "SPMM: Matrix min row size: " << min_size << std::endl;
            buffer << "SPMM: Matrix max row size: " << max_size << std::endl;
        }

        device_vector_alloc<int> num_nz( this->P.row_offsets.size() );
        thrust::adjacent_difference( this->P.row_offsets.begin(), this->P.row_offsets.end(), num_nz.begin() );
        cudaCheckError();
        Result result = thrust::minmax_element( num_nz.begin() + 1, num_nz.end() );
        cudaCheckError();
        int min_size = *result.first;
        int max_size = *result.second;
        int sum = thrust_wrapper::reduce( num_nz.begin() + 1, num_nz.end() );
        cudaCheckError();
        double avg_size = double(sum) / this->P.get_num_rows();
        buffer << "SPMM: P: " << std::endl;
        buffer << "SPMM: Matrix avg row size: " << avg_size << std::endl;
        buffer << "SPMM: Matrix min row size: " << min_size << std::endl;
        buffer << "SPMM: Matrix max row size: " << max_size << std::endl;
        num_nz.resize( this->R.row_offsets.size() );
        thrust::adjacent_difference( this->R.row_offsets.begin(), this->R.row_offsets.end(), num_nz.begin() );
        cudaCheckError();
        result = thrust::minmax_element( num_nz.begin() + 1, num_nz.end() );
        cudaCheckError();
        min_size = *result.first;
        max_size = *result.second;
        sum = thrust_wrapper::reduce( num_nz.begin() + 1, num_nz.end() );
        cudaCheckError();
        avg_size = double(sum) / this->R.get_num_rows();
        buffer << "SPMM: R: " << std::endl;
        buffer << "SPMM: Matrix avg row size: " << avg_size << std::endl;
        buffer << "SPMM: Matrix min row size: " << min_size << std::endl;
        buffer << "SPMM: Matrix max row size: " << max_size << std::endl;
        amgx_output( buffer.str().c_str(), static_cast<int>( buffer.str().length() ) );
    }

    RAP.set_initialized( 0 );
    CSR_Multiply<TConfig_d>::csr_galerkin_product( this->R, this->getA(), this->P, RAP, NULL, NULL, NULL, NULL, NULL, NULL, wk );
    RAP.set_initialized( 1 );
    int spmm_no_sort = this->amg->m_cfg->AMG_Config::getParameter<int>("spmm_no_sort", this->amg->m_cfg_scope);
    this->Profile.toc("computeA");
}
/**********************************************
 * computes the restriction: rr=R*r
 **********************************************/
template <class T_Config>
void Classical_AMG_Level_Base<T_Config>::restrictResidual(VVector &r, VVector &rr)
{
// we need to resize residual vector to make sure it can store halo rows to be sent
    if (!P.is_matrix_singleGPU())
    {
        typedef typename TConfig::MemSpace MemorySpace;
        Matrix<TConfig> &Ac = this->getNextLevel( MemorySpace( ) )->getA();
#if COARSE_CLA_CONSO
        int desired_size ;

        if (this->getNextLevel(MemorySpace())->isConsolidationLevel())
        {
            desired_size = std::max(P.manager->halo_offsets[P.manager->neighbors.size()], Ac.manager->halo_offsets_before_glue[Ac.manager->neighbors_before_glue.size()] * rr.get_block_size());
        }
        else
        {
            desired_size = std::max(P.manager->halo_offsets[P.manager->neighbors.size()], Ac.manager->halo_offsets[Ac.manager->neighbors.size()] * rr.get_block_size());
        }

#else
        int desired_size = std::max(P.manager->halo_offsets[P.manager->neighbors.size()], Ac.manager->halo_offsets[Ac.manager->neighbors.size()] * rr.get_block_size());
#endif
        rr.resize(desired_size);
    }

#if 1
    this->Profile.tic("restrictRes");

    // Disable speculative send of rr
    if (P.is_matrix_singleGPU())
    {
        multiply( R, r, rr);
    }
    else
    {
        multiply_with_mask_restriction( R, r, rr, P);
    }

#endif
    // exchange halo residuals & add residual contribution from neighbors
    rr.dirtybit = 1;

    if (!P.is_matrix_singleGPU())
    {
        int desired_size = P.manager->halo_offsets[P.manager->neighbors.size()] * rr.get_block_size();

        if (rr.size() < desired_size)
        {
            rr.resize(desired_size);
        }
    }

    this->Profile.toc("restrictRes");
}

struct is_minus_one
{
    __host__ __device__
    bool operator()(const int &x)
    {
        return x == -1;
    }
};


template <AMGX_VecPrecision t_vecPrec, AMGX_MatPrecision t_matPrec, AMGX_IndPrecision t_indPrec>
void Classical_AMG_Level<TemplateConfig<AMGX_device, t_vecPrec, t_matPrec, t_indPrec> >::computeAOperator_1x1_distributed()
{
    Matrix<TConfig_d> &A = this->getA();
    Matrix<TConfig_d> &P = this->P;
    Matrix<TConfig_d> &RAP = this->getNextLevel( device_memory( ) )->getA( );
    RAP.addProps(CSR);
    RAP.set_block_dimx(this->getA().get_block_dimx());
    RAP.set_block_dimy(this->getA().get_block_dimy());
    IndexType num_parts = A.manager->get_num_partitions();
    IndexType num_neighbors = A.manager->num_neighbors();
    IndexType my_rank = A.manager->global_id();
    // OWNED includes interior and boundary
    A.setView(OWNED);
    int num_owned_coarse_pts = P.manager->halo_offsets[0];
    int num_owned_fine_pts = A.manager->halo_offsets[0];

    // Initialize RAP.manager
    if (RAP.manager == NULL)
    {
        RAP.manager = new DistributedManager<TConfig_d>();
    }

    RAP.manager->A = &RAP;
    RAP.manager->setComms(A.manager->getComms());
    RAP.manager->set_global_id(my_rank);
    RAP.manager->set_num_partitions(num_parts);
    RAP.manager->part_offsets_h = P.manager->part_offsets_h;
    RAP.manager->part_offsets = P.manager->part_offsets;
    RAP.manager->set_base_index(RAP.manager->part_offsets_h[my_rank]);
    RAP.manager->set_index_range(num_owned_coarse_pts);
    RAP.manager->num_rows_global = RAP.manager->part_offsets_h[num_parts];
    // --------------------------------------------------------------------
    // Using the B2L_maps of matrix A, identify the rows of P that need to be sent to neighbors,
    // so that they can compute A*P
    // Once rows of P are identified, convert the column indices to global indices, and send them to neighbors
    //  ---------------------------------------------------------------------------
    // Copy some information about the manager of P, since we don't want to modify those
    IVector_h P_neighbors = P.manager->neighbors;
    I64Vector_h P_halo_ranges_h = P.manager->halo_ranges_h;
    I64Vector_d P_halo_ranges = P.manager->halo_ranges;
    RAP.manager->local_to_global_map = P.manager->local_to_global_map;
    IVector_h P_halo_offsets = P.manager->halo_offsets;
    // Create a temporary distributed arranger
    DistributedArranger<TConfig_d> *prep = new DistributedArranger<TConfig_d>;
    prep->exchange_halo_rows_P(A, this->P, RAP.manager->local_to_global_map, P_neighbors, P_halo_ranges_h, P_halo_ranges, P_halo_offsets, RAP.manager->part_offsets_h, RAP.manager->part_offsets, num_owned_coarse_pts, RAP.manager->part_offsets_h[my_rank]);
    cudaCheckError();
    // At this point, we can compute RAP_full which contains some rows that will need to be sent to neighbors
    // i.e. RAP_full = [ RAP_int ]
    //                 [ RAP_ext ]
    // RAP is [ RAP_int ] + [RAP_ext_received_from_neighbors]
    // We can reuse the serial galerkin product since R, A and P use local indices
    // TODO: latency hiding (i.e. compute RAP_ext, exchange_matrix_halo, then do RAP_int)
    /* WARNING: do not recompute prolongation (P) and restriction (R) when you
                are reusing the level structure (structure_reuse_levels > 0) */
    /* We force for matrix P to have only owned rows to be seen for the correct galerkin product computation*/
    this->P.set_initialized(0);
    this->P.set_num_rows(num_owned_fine_pts);
    this->P.addProps( CSR );
    this->P.set_initialized(1);

    if (this->isReuseLevel() == false)
    {
        this->R.set_initialized( 0 );
        this->R.addProps( CSR );
        // Take the tranpose of P to get R
        // Single-GPU transpose, no mpi exchange
        this->computeRestrictionOperator();
        this->R.set_initialized( 1 );
    }

    this->Profile.tic("computeA");
    Matrix<TConfig_d> RAP_full;
    // Initialize the workspace needed for galerkin product
    void *wk = AMG_Level<TConfig_d>::amg->getCsrWorkspace();

    if ( wk == NULL )
    {
        wk = CSR_Multiply<TConfig_d>::csr_workspace_create( *(AMG_Level<TConfig_d>::amg->m_cfg), AMG_Level<TConfig_d>::amg->m_cfg_scope );
        AMG_Level<TConfig_d>::amg->setCsrWorkspace( wk );
    }

    // Single-GPU RAP, no mpi exchange
    RAP_full.set_initialized( 0 );
    /* WARNING: Since A is reordered (into interior and boundary nodes), while R and P are not reordered,
                you must unreorder A when performing R*A*P product in ordre to obtain the correct result. */
    CSR_Multiply<TConfig_d>::csr_galerkin_product( this->R, this->getA(), this->P, RAP_full,
            /* permutation for rows of R, A and P */       NULL, NULL /*&(this->getA().manager->renumbering)*/,        NULL,
            /* permutation for cols of R, A and P */       NULL, NULL /*&(this->getA().manager->inverse_renumbering)*/, NULL,
            wk );
    RAP_full.set_initialized( 1 );
    this->Profile.toc("computeA");
    // ----------------------------------------------------------------------------------------------
    // Now, send rows of RAP_full requireq by neighbors, received rows from neighbors and create RAP
    // ----------------------------------------------------------------------------------------------
    prep->exchange_RAP_ext(RAP, RAP_full, A, this->P, P_halo_offsets, RAP.manager->local_to_global_map, P_neighbors, P_halo_ranges_h, P_halo_ranges, RAP.manager->part_offsets_h, RAP.manager->part_offsets, num_owned_coarse_pts, RAP.manager->part_offsets_h[my_rank], wk);
    // Delete temporary distributed arranger
    delete prep;
    /* WARNING: The RAP matrix generated at this point contains extra rows (that correspond to rows of R,
       that was obtained by locally transposing P). This rows are ignored by setting the # of matrix rows
       to be smaller, so that they correspond to number of owned coarse nodes. This should be fine, but
       it leaves holes in the matrix as there might be columns that belong to the extra rows that now do not
       belong to the smaller matrix with number of owned coarse nodes rows. The same is trued about the
       local_to_global_map. These two data structures match at this point. However, in the next calls
       local_to_global (exclusively) will be used to geberate B2L_maps (wihtout going through column indices)
       which creates extra elements in the B2L that simply do not exist in the new matrices. I strongly suspect
       this is the reason fore the bug. The below fix simply compresses the matrix so that there are no holes
       in it, or in the local_2_global_map. */
    //mark local_to_global_columns that exist in the owned coarse nodes rows.
    IndexType nrow = RAP.get_num_rows();
    IndexType ncol = RAP.get_num_cols();
    IndexType nl2g = ncol - nrow;

    if (nl2g > 0)
    {
        IVector   l2g_p(nl2g + 1, 0); //+1 is needed for prefix_sum/exclusive_scan
        I64Vector l2g_t(nl2g, 0);
        IndexType nblocks = (nrow + AMGX_CAL_BLOCK_SIZE - 1) / AMGX_CAL_BLOCK_SIZE;

        if (nblocks > 0)
            flag_existing_local_to_global_columns<int> <<< nblocks, AMGX_CAL_BLOCK_SIZE>>>
            (nrow, RAP.row_offsets.raw(), RAP.col_indices.raw(), l2g_p.raw());

        cudaCheckError();
        /*
        //slow version of the above kernel
        for(int ii=0; ii<nrow; ii++){
            int s = RAP.row_offsets[ii];
            int e = RAP.row_offsets[ii+1];
            for (int jj=s; jj<e; jj++) {
                int col = RAP.col_indices[jj];
                if (col>=nrow){
                    int kk = col-RAP.get_num_rows();
                    l2g_p[kk] = 1;
                }
            }
        }
        cudaCheckError();
        */
        //create a pointer map for their location using prefix sum
        thrust_wrapper::exclusive_scan(l2g_p.begin(), l2g_p.end(), l2g_p.begin());
        int new_nl2g = l2g_p[nl2g];

        //compress the columns using the pointer map
        if (nblocks > 0)
            compress_existing_local_columns<int> <<< nblocks, AMGX_CAL_BLOCK_SIZE>>>
            (nrow, RAP.row_offsets.raw(), RAP.col_indices.raw(), l2g_p.raw());

        cudaCheckError();
        /*
        //slow version of the above kernel
        for(int ii=0; ii<nrow; ii++){
            int s = RAP.row_offsets[ii];
            int e = RAP.row_offsets[ii+1];
            for (int jj=s; jj<e; jj++) {
                int col = RAP.col_indices[jj];
                if (col>=nrow){
                    int kk = col-RAP.get_num_rows();
                    RAP.col_indices[jj] = nrow+l2g_p[kk];
                }
            }
        }
        cudaCheckError();
        */
        //adjust matrix size (number of columns) accordingly
        RAP.set_initialized(0);
        RAP.set_num_cols(nrow + new_nl2g);
        RAP.set_initialized(1);
        //compress local_to_global_map using the pointer map
        nblocks = (nl2g + AMGX_CAL_BLOCK_SIZE - 1) / AMGX_CAL_BLOCK_SIZE;

        if (nblocks > 0)
            compress_existing_local_to_global_columns<int, int64_t> <<< nblocks, AMGX_CAL_BLOCK_SIZE>>>
            (nl2g, RAP.manager->local_to_global_map.raw(), l2g_t.raw(), l2g_p.raw());

        cudaCheckError();
        thrust::copy(l2g_t.begin(), l2g_t.begin() + new_nl2g, RAP.manager->local_to_global_map.begin());
        cudaCheckError();
        /*
        //slow version of the above kernel (through Thrust)
        for(int ii=0; ii<(l2g_p.size()-1); ii++){
            if (l2g_p[ii] != l2g_p[ii+1]){
                RAP.manager->local_to_global_map[l2g_p[ii]] = RAP.manager->local_to_global_map[ii];
            }
        }
        cudaCheckError();
        */
        //adjust local_to_global_map size accordingly
        RAP.manager->local_to_global_map.resize(new_nl2g);
    }
}

/**********************************************
 * prolongates the error: x+=P*e
 **********************************************/
template <class T_Config>
void Classical_AMG_Level_Base<T_Config>::prolongateAndApplyCorrection(VVector &e, VVector &bc, VVector &x, VVector &tmp)
{
    this->Profile.tic("proCorr");
    // Use P.manager to exchange halo of e before doing P
    // (since P has columns belonging to one of P.neighbors)
    e.dirtybit = 1;

    if (!P.is_matrix_singleGPU())
    {
        // get coarse matrix
        typedef typename TConfig::MemSpace MemorySpace;
        Matrix<TConfig> &Ac = this->getNextLevel( MemorySpace( ) )->getA();
#if COARSE_CLA_CONSO
        int e_size;

        if (this->getNextLevel(MemorySpace())->isConsolidationLevel())
        {
            e_size = std::max(P.manager->halo_offsets[P.manager->neighbors.size()], Ac.manager->halo_offsets_before_glue[Ac.manager->neighbors_before_glue.size()]) * e.get_block_size();
        }
        else
        {
            e_size = std::max(P.manager->halo_offsets[P.manager->neighbors.size()], Ac.manager->halo_offsets[Ac.manager->neighbors.size()]) * e.get_block_size();
        }

        if (e.size() < e_size) { e.resize(e_size); }

#else
        int e_size = std::max(P.manager->halo_offsets[P.manager->neighbors.size()], Ac.manager->halo_offsets[Ac.manager->neighbors.size()]) * e.get_block_size();
        e.resize(e_size);
#endif
    }

    if (P.is_matrix_singleGPU())
    {
        if (e.size() > 0)
        {
            multiply( P, e, tmp);
        }
    }
    else
    {
        multiply_with_mask( P, e, tmp);
    }

    // get owned num rows for fine matrix
    int owned_size;

    if (this->A->is_matrix_distributed())
    {
        int owned_offset;
        P.manager->getOffsetAndSizeForView(OWNED, &owned_offset, &owned_size);
    }
    else
    {
        owned_size = x.size();
    }

    //apply
    axpby(x, tmp, x, ValueType(1), ValueType(1), 0, owned_size);
    this->Profile.toc("proCorr");
    x.dirtybit = 1;
}

template <class T_Config>
void Classical_AMG_Level_Base<T_Config>::computeAOperator()
{
    if (this->A->get_block_size() == 1)
    {
        computeAOperator_1x1();
    }
    else
    {
        FatalError("Classical AMG not implemented for block_size != 1", AMGX_ERR_NOT_IMPLEMENTED);
    }
}

template <class T_Config>
void Classical_AMG_Level_Base<T_Config>::computeAOperator_distributed()
{
    if (this->A->get_block_size() == 1)
    {
        computeAOperator_1x1_distributed();
    }
    else
    {
        FatalError("Classical AMG not implemented for block_size != 1", AMGX_ERR_NOT_IMPLEMENTED);
    }
}


template <class T_Config>
void Classical_AMG_Level_Base<T_Config>::consolidateVector(VVector &x)
{
#ifdef AMGX_WITH_MPI
#if COARSE_CLA_CONSO
    typedef typename TConfig::MemSpace MemorySpace;
    Matrix<TConfig> &A = this->getA();
    Matrix<TConfig> &Ac = this->getNextLevel( MemorySpace( ) )->getA();
    MPI_Comm comm, temp_com;
    comm = Ac.manager->getComms()->get_mpi_comm();
    temp_com = compute_glue_matrices_communicator(Ac);
    glue_vector(Ac, comm, x, temp_com);
#endif
#endif
}

template <class T_Config>
void Classical_AMG_Level_Base<T_Config>::unconsolidateVector(VVector &x)
{
#ifdef AMGX_WITH_MPI
#if COARSE_CLA_CONSO
    typedef typename TConfig::MemSpace MemorySpace;
    Matrix<TConfig> &A = this->getA();
    Matrix<TConfig> &Ac = this->getNextLevel( MemorySpace( ) )->getA();
    MPI_Comm comm, temp_com;
    comm = Ac.manager->getComms()->get_mpi_comm();
    temp_com = compute_glue_matrices_communicator(Ac);
    unglue_vector(Ac, comm, x, temp_com, x);
#endif
#endif
}

/****************************************
 * Explict instantiations
 ***************************************/
template class Classical_AMG_Level_Base<TConfigGeneric_d>;
template class Classical_AMG_Level_Base<TConfigGeneric_h>;

template class Classical_AMG_Level<TConfigGeneric_d>;
template class Classical_AMG_Level<TConfigGeneric_h>;

} // namespace classical

} // namespace amgx
