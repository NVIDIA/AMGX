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

#include <classical/selectors/aggressive_hmis.h>
#include <classical/selectors/hmis.h>
#include <classical/strength/all.h>
#include <classical/interpolators/distance2.h>
#include <hash_workspace.h>
#include <cutil.h>
#include <util.h>
#include <types.h>

#include<thrust/count.h>

namespace amgx
{

namespace classical
{

#include <sm_utils.inl>

template <class T_Config>
void Aggressive_HMIS_SelectorBase< T_Config>::markCoarseFinePoints(Matrix< T_Config> &A,
        FVector &weights,
        const BVector &s_con,
        IVector &cf_map,
        IVector &scratch,
        int cf_map_init)
{
    ViewType oldView = A.currentView();
    A.setView(OWNED);
    // Implementation based on paper "On long range interpolation operators for aggressive coarsening" by Ulrike Meier Yang, section 3
    // Step 1:
    // In the first pass, use HMIS to flag coarse and fine points
    HMIS_Selector<T_Config> *hmis_selector = new HMIS_Selector<T_Config>;
    hmis_selector->markCoarseFinePoints(A, weights, s_con, cf_map, scratch);
    // Step 2:
    // Do a two ring exchange to know if nodes in two-ring halos are coarse or fine
    cf_map.dirtybit = 1;

    if (!A.is_matrix_singleGPU())
    {
        A.manager->exchange_halo_2ring(cf_map, cf_map.tag);
    }

    // Step 3:
    // Create an array cf_map_scanned, which assigns a local coarse index
    // for all points in the first two rings. If point is not a coarse pt, assign -1
    // here, num_coarse_points contains the number of OWNED coarse points
    IVector cf_map_scanned = cf_map;
    int num_coarse_points;
    this->renumberAndCountCoarsePoints(cf_map_scanned, num_coarse_points, A.get_num_rows());
    // Step 4:
    // Create a matrix S2 with "num_coarse_points" rows, where two coarse pts are connected
    // if there's a path of length 2 with strong connections connecting those two coarse pts
    // Function createS2 needs the one ring halo rows of A, two rings of cf_map
    Matrix<TConfig> S2;
    this->createS2(A, S2, s_con, cf_map_scanned);
    cudaCheckError();

    if (S2.get_num_rows() != 0)
    {
        // Step 5:
        // Create the weights_S2 array,  which assigns a weight to each row in S2,
        // based on how many other rows in S2 are connected to that row
        FVector weights_S2;

        if (!S2.is_matrix_singleGPU())
        {
            int size, offset;
            S2.getOffsetAndSizeForView(FULL, &offset, &size);
            weights_S2.resize(size);
        }
        else
        {
            weights_S2.resize(S2.get_num_rows());
        }

        thrust_wrapper::fill(weights_S2.begin(), weights_S2.end(), 0.0);
        cudaCheckError();
        AMG_Config cfg;
        cfg.parseParameterString("");
        Strength_All<TConfig> *strength = new Strength_All<TConfig>(cfg, "default");
        strength->computeWeights(S2, weights_S2);
        delete strength;

        // Exchange the weights with one ring neighbors
        if (!S2.is_matrix_singleGPU())
        {
            weights_S2.dirtybit = 1;
            S2.manager->exchange_halo(weights_S2, weights_S2.tag);
        }

        // Fill s_con with 1, (all connections of S are strong)
        // Initial scratch and cf_map arrays
        int size_full, nnz_owned;

        if (!S2.is_matrix_singleGPU())
        {
            int offset;
            // Need to get number of 2-ring rows
            S2.getOffsetAndSizeForView(FULL, &offset, &size_full);
            S2.getNnzForView(OWNED, &nnz_owned);
        }
        else
        {
            size_full = S2.get_num_rows();
            nnz_owned = S2.get_num_nz();
        }

        BVector s_con_S2;
        s_con_S2.resize(nnz_owned);
        thrust_wrapper::fill(s_con_S2.begin(), s_con_S2.end(), true);
        cudaCheckError();
        // Allocate some scratch space required by HMIS selector
        IVector scratch_S2;
        scratch_S2.resize(size_full);
        thrust_wrapper::fill(scratch_S2.begin(), scratch_S2.end(), 0);
        cudaCheckError();
        // Initialize cf_map_S2
        IVector cf_map_S2;
        cf_map_S2.resize(size_full);
        thrust_wrapper::fill(cf_map_S2.begin(), cf_map_S2.end(), 0);
        cudaCheckError();
        // Step 6:
        // Apply PMIS algorithm on S2 matrix to identify new set of coarse points
        // Array cf_map_S2 now marks each row in S2 as COARSE, FINE, etc...
        hmis_selector->markCoarseFinePoints(S2, weights_S2, s_con_S2, cf_map_S2, scratch_S2, 3);
        delete hmis_selector;
        // Step 7:
        // Modify cf_map based on new info from cf_map_S2
        // i.e remove COARSE points in cf_map which have been marked as FINE in cf_map_S2
        this->correctCfMap(cf_map, cf_map_scanned, cf_map_S2);
        A.setView(oldView);
    }
}

/****************************************
 * Explict instantiations
 ***************************************/
#define AMGX_CASE_LINE(CASE) template class Aggressive_HMIS_SelectorBase<TemplateMode<CASE>::Type>;
AMGX_FORALL_BUILDS(AMGX_CASE_LINE)
#undef AMGX_CASE_LINE

#define AMGX_CASE_LINE(CASE) template class Aggressive_HMIS_Selector<TemplateMode<CASE>::Type>;
AMGX_FORALL_BUILDS(AMGX_CASE_LINE)
#undef AMGX_CASE_LINE

} // namespace classical

} // namespace amgx
