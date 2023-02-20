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


#include <energymin/energymin_amg_level.h>
#include <amg_level.h>

#include <basic_types.h>
#include <cutil.h>
#include <multiply.h>
#include <transpose.h>
#include <blas.h>
#include <util.h>
#include <thrust/logical.h>
#include <thrust/remove.h>
#include <thrust/adjacent_difference.h>

#include <assert.h>
#include <matrix_io.h>

#include <csr_multiply.h>

#include <thrust/logical.h>
#include <thrust/count.h>
#include <thrust/sort.h>

#include <profile.h>
#include <string>
#include <algorithm>

namespace amgx
{

namespace energymin
{


// --------------------------- Begin Base Class Public methods ------------------------------------

template <class T_Config>
Energymin_AMG_Level_Base<T_Config>
::Energymin_AMG_Level_Base(AMG_Class *amg) : AMG_Level<T_Config>(amg)
{
    selector     = amgx::classical::SelectorFactory<T_Config>::allocate(*(amg->m_cfg), amg->m_cfg_scope);
    interpolator = InterpolatorFactory<T_Config>::allocate(*(amg->m_cfg), amg->m_cfg_scope);
    strength = NULL;
    std::string selector_val = amg->m_cfg->template getParameter<std::string>("selector", amg->m_cfg_scope);

    if (selector_val == "PMIS") //or any other classical selector
    {
        strength = StrengthFactory<T_Config>::allocate(*(amg->m_cfg), amg->m_cfg_scope); //using default strength
        max_row_sum = amg->m_cfg->AMG_Config::getParameter<double>("max_row_sum", amg->m_cfg_scope);
    }
}

template <class T_Config>
Energymin_AMG_Level_Base<T_Config>::~Energymin_AMG_Level_Base()
{
    delete selector;
    delete interpolator;

    if (strength != NULL) { delete strength; }
}


// Compute A, P, and R operators
template <class T_Config>
void Energymin_AMG_Level_Base<T_Config>
::createCoarseVertices()
{
    Matrix<T_Config> &RAP = this->getNextLevel( typename Matrix<T_Config>::memory_space() )->getA();
    Matrix<T_Config> &A   = this->getA();
    int size_all;
    size_all  = A.get_num_rows();
    this->m_cf_map.resize(size_all);
    thrust::fill(this->m_cf_map.begin(), this->m_cf_map.end(), 0);
    cudaCheckError();
    markCoarseFinePoints();
}

template <class T_Config>
void Energymin_AMG_Level_Base<T_Config>
::createCoarseMatrices()
{
    Matrix<T_Config> &RAP = this->getNextLevel( typename Matrix<T_Config>::memory_space() )->getA();
    Matrix<T_Config> &A   = this->getA();
    /* WARNING: exit if D1 interpolator is selected in distributed setting */
    std::string s("");
    s += AMG_Level<T_Config>::amg->m_cfg->AMG_Config
         ::getParameter<std::string>("energymin_interpolator",
                                     AMG_Level<T_Config>::amg->m_cfg_scope);
    // Compute Restriction operator
    computeRestrictionOperator();

    // Compute Prolongation operator and coarse matrix Ac
    if (!this->A->is_matrix_distributed() || this->A->manager->get_num_partitions() == 1)
    {
        // Create Prolongation operator
        computeProlongationOperator();
        computeAOperator();
    }
    else
    {
        computeAOperator_distributed();
    }

    RAP.copyAuxData(&A);

    if (this->getA().is_matrix_singleGPU())
    {
        this->m_next_level_size = this->getNextLevel(typename Matrix<TConfig>::memory_space())->getA().get_num_rows()
                                  * this->getNextLevel(typename Matrix<TConfig>::memory_space())->getA().get_block_dimy();
    }
    else
    {
        // m_next_level_size is the size that will be used to allocate xc, bc vectors
        int size, offset;
        this->getNextLevel(typename Matrix<TConfig>::memory_space())->getA().getOffsetAndSizeForView(FULL, &offset, &size);
        this->m_next_level_size = size
                                  * this->getNextLevel(typename Matrix<TConfig>::memory_space())->getA().get_block_dimy();
    }
}


template <class T_Config>
void Energymin_AMG_Level_Base<T_Config>
::markCoarseFinePoints()
{
    Matrix<T_Config> &A = this->getA();
    // Allocate necessary memory
    typedef Vector<typename TConfig::template setVecPrec<AMGX_vecInt>::Type>   IVector;
    typedef Vector<typename TConfig::template setVecPrec<AMGX_vecBool>::Type>  BVector;
    typedef Vector<typename TConfig::template setVecPrec<AMGX_vecFloat>::Type> FVector;
    int size_all, size_full, nnz_full;
    BVector m_s_con;
    IVector m_scratch;
    FVector weights;

    if (!A.is_matrix_singleGPU())
    {
        int offset;
        // Need to get number of 2-ring rows
        A.getOffsetAndSizeForView(ALL, &offset, &size_all);
        A.getOffsetAndSizeForView(FULL, &offset, &size_full);
        A.getNnzForView(FULL, &nnz_full);
        weights.resize(size_full);
    }
    else
    {
        size_all = A.get_num_rows();
        size_full = A.get_num_rows();
        nnz_full = A.get_num_nz();
        weights.resize(A.get_num_rows());
    }

    this->m_cf_map.resize(size_all);
    m_s_con.resize(nnz_full);
    m_scratch.resize(size_full);
    thrust::fill(weights.begin(), weights.end(), 0.0);
    cudaCheckError();
    thrust::fill(this->m_cf_map.begin(), this->m_cf_map.end(), 0);
    cudaCheckError();
    thrust::fill(m_s_con.begin(), m_s_con.end(), false);
    cudaCheckError();
    thrust::fill(m_scratch.begin(), m_scratch.end(), 0);
    cudaCheckError();

    if (strength != NULL)
    {
        if (!A.is_matrix_singleGPU())
        {
            ViewType oldView = A.currentView();
            A.setView(FULL);
            strength->computeStrongConnectionsAndWeights(A, m_s_con, weights, this->max_row_sum);
            A.setView(oldView);
            A.manager->exchange_halo(weights, weights.tag);
        }
        else
        {
            strength->computeStrongConnectionsAndWeights(A, m_s_con, weights, this->max_row_sum);
        }
    }

    // Mark coarse and fine points
    selector->markCoarseFinePoints(A, weights, m_s_con, this->m_cf_map, m_scratch);
    this->m_cf_map.dirtybit = 1;
}


template <class T_Config>
void Energymin_AMG_Level_Base<T_Config>
::computeRestrictionOperator()
{
    this->Profile.tic("computeR");
    Matrix<T_Config> &A = this->getA();
    //allocate necessary memory
    typedef Vector<typename TConfig::template setVecPrec<AMGX_vecInt>::Type>   IVector;
    typedef Vector<typename TConfig::template setVecPrec<AMGX_vecBool>::Type>  BVector;
    typedef Vector<typename TConfig::template setVecPrec<AMGX_vecFloat>::Type> FVector;
    // WARNING: Since energymin P is in computed in CSC format and AMGX does not support
    // CSC format, we are actually computing P^T (=R) in generateInterpolationMatrix!!
    //generate the interpolation matrix
    interpolator->generateInterpolationMatrix(A, this->m_cf_map, R,
            AMG_Level<TConfig>::amg);
    this->m_cf_map.clear();
    this->m_cf_map.shrink_to_fit();
    this->Profile.toc("computeR");
}


// Compute R=P^T
template <class T_Config>
void Energymin_AMG_Level_Base<T_Config>
::computeProlongationOperator()
{
    this->Profile.tic("computeP");
    P.set_initialized(0);
    R.setView(OWNED);
    transpose(R, P, R.get_num_rows());

    if (!P.isLatencyHidingEnabled(*this->amg->m_cfg))
    {
        // This will cause bsrmv to not do latency hiding
        P.setInteriorView(OWNED);
        P.setExteriorView(OWNED);
    }

    P.set_initialized(1);
    this->Profile.toc("computeP");
}


// Compute the Galerkin product: A_c=R*A*P
template <AMGX_VecPrecision t_vecPrec, AMGX_MatPrecision t_matPrec, AMGX_IndPrecision t_indPrec>
void Energymin_AMG_Level<TemplateConfig<AMGX_host, t_vecPrec, t_matPrec, t_indPrec> >
::computeAOperator_1x1()
{
    FatalError("Energymin AMG computeAOperator_1x1 not implemented on host\n",
               AMGX_ERR_NOT_IMPLEMENTED);
}


template <AMGX_VecPrecision t_vecPrec, AMGX_MatPrecision t_matPrec, AMGX_IndPrecision t_indPrec>
void Energymin_AMG_Level<TemplateConfig<AMGX_host, t_vecPrec, t_matPrec, t_indPrec> >
::computeAOperator_1x1_distributed()
{
    FatalError("Distributed energymin AMG not implemented for host\n",
               AMGX_ERR_NOT_IMPLEMENTED);
}


template <AMGX_VecPrecision t_vecPrec, AMGX_MatPrecision t_matPrec, AMGX_IndPrecision t_indPrec>
void Energymin_AMG_Level<TemplateConfig<AMGX_device, t_vecPrec, t_matPrec, t_indPrec> >
::computeAOperator_1x1()
{
    this->Profile.tic("computeA");
    Matrix<TConfig_d> &RAP = this->getNextLevel( device_memory() )->getA();
    RAP.addProps(CSR);
    RAP.set_block_dimx(this->getA().get_block_dimx());
    RAP.set_block_dimy(this->getA().get_block_dimy());
    this->R.set_initialized(0);
    this->R.addProps(CSR);
    this->R.set_initialized(1);
    this->P.set_initialized(0);
    this->P.addProps(CSR);
    this->P.set_initialized(1);
    void *wk = AMG_Level<TConfig_d>::amg->getCsrWorkspace();

    if ( wk == NULL )
    {
        wk = CSR_Multiply<TConfig_d>::csr_workspace_create( *(AMG_Level<TConfig_d>::amg->m_cfg),
                AMG_Level<TConfig_d>::amg->m_cfg_scope );
        AMG_Level<TConfig_d>::amg->setCsrWorkspace( wk );
    }

    RAP.set_initialized(0);
    CSR_Multiply<TConfig_d>::csr_galerkin_product(this->R, this->getA(), this->P, RAP,
            NULL, NULL, NULL, NULL, NULL, NULL, wk);
    RAP.set_initialized(1);
    this->Profile.toc("computeA");
}


// Compute the restriction: rr=R*r
template <class T_Config>
void Energymin_AMG_Level_Base<T_Config>
::restrictResidual(VVector &r, VVector &rr)
{
    typedef typename TConfig::MemSpace MemorySpace;
    Matrix<TConfig> &Ac = this->getNextLevel( MemorySpace( ) )->getA();

// we need to resize residual vector to make sure it can store halo rows to be sent
    if (!P.is_matrix_singleGPU())
    {
        int desired_size = std::max(P.manager->halo_offsets[P.manager->neighbors.size()],
                                    Ac.manager->halo_offsets[Ac.manager->neighbors.size()] * rr.get_block_size());
        rr.resize(desired_size);
    }

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

    rr.dirtybit = 1;

    // Do I need this?
    if (!P.is_matrix_singleGPU())
    {
        int desired_size = P.manager->halo_offsets[P.manager->neighbors.size()] * rr.get_block_size();

        // P.manager->transformVector(rr); //This is just to make sure size is right
        if (rr.size() < desired_size)
        {
            rr.resize(P.manager->halo_offsets[P.manager->neighbors.size()]*rr.get_block_size());
        }

        // P.manager->exchange_halo(rr, rr.tag);
    }

    this->Profile.toc("restrictRes");
}


template <AMGX_VecPrecision t_vecPrec, AMGX_MatPrecision t_matPrec, AMGX_IndPrecision t_indPrec>
void Energymin_AMG_Level<TemplateConfig<AMGX_device, t_vecPrec, t_matPrec, t_indPrec> >
::computeAOperator_1x1_distributed()
{
    FatalError("Energymin AMG Level computeAOperator_1x1_distributed() not implemented",
               AMGX_ERR_NOT_IMPLEMENTED);
}


// Prolongate the error: x+=P*e
template <class T_Config>
void Energymin_AMG_Level_Base<T_Config>
::prolongateAndApplyCorrection(VVector &e, VVector &bc, VVector &x, VVector &tmp)
{
    this->Profile.tic("proCorr");
    // get coarse matrix
    typedef typename TConfig::MemSpace MemorySpace;
    Matrix<TConfig> &Ac = this->getNextLevel( MemorySpace( ) )->getA();
    // Use P.manager to exchange halo of e before doing P
    // (since P has columns belonging to one of P.neighbors)
    e.dirtybit = 1;

    if (!P.is_matrix_singleGPU())
    {
        int e_size = std::max(P.manager->halo_offsets[P.manager->neighbors.size()],
                              Ac.manager->halo_offsets[Ac.manager->neighbors.size()])
                     * e.get_block_size();
        e.resize(e_size);
    }

    if (P.is_matrix_singleGPU())
    {
        if (e.size() > 0) {multiply( P, e, tmp);}
    }
    else
    {
        multiply_with_mask( P, e, tmp);
    }

    // get owned num rows for fine matrix
    int owned_size;

    if (Ac.is_matrix_distributed())
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
void Energymin_AMG_Level_Base<T_Config>
::computeAOperator()
{
    if (this->A->get_block_size() == 1)
    {
        computeAOperator_1x1();
    }
    else
    {
        FatalError("Energymin AMG not implemented for block_size != 1", AMGX_ERR_NOT_IMPLEMENTED);
    }
}

template <class T_Config>
void Energymin_AMG_Level_Base<T_Config>
::computeAOperator_distributed()
{
    if (this->A->get_block_size() == 1)
    {
        computeAOperator_1x1_distributed();
    }
    else
    {
        FatalError("Energymin AMG not implemented for block_size != 1", AMGX_ERR_NOT_IMPLEMENTED);
    }
}



/****************************************
 * Explicit instantiations
 ***************************************/
template class Energymin_AMG_Level_Base<TConfigGeneric_d>;
template class Energymin_AMG_Level_Base<TConfigGeneric_h>;

template class Energymin_AMG_Level<TConfigGeneric_d>;
template class Energymin_AMG_Level<TConfigGeneric_h>;

} // namespace energymin
} // namespace amgx
