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

#include "unit_test.h"
#include <matrix_io.h>
#include "test_utils.h"
#include "util.h"
#include "time.h"
#include "classical/strength/strength_base.h"
#include <classical/selectors/selector.h>
#include <thrust/iterator/counting_iterator.h>

#define HOST_ONLY_CHECKER

namespace amgx
{

// check selection is valid
template <typename IndexType, typename BoolType>
struct CheckSelection
{
    const IndexType *_A_offsets;
    const IndexType *_A_cols;
    const BoolType *_s_con;
    const int *_cf_map;
    const int _num_rows;

    CheckSelection(const IndexType *A_offsets, const IndexType *A_cols, const BoolType *s_con,
                   const IndexType *cf_map, const int num_rows) : _A_offsets(A_offsets), _A_cols(A_cols),
        _s_con(s_con), _cf_map(cf_map),
        _num_rows(num_rows) {};

    __host__ __device__
    IndexType operator()(IndexType i)
    {
        int CF = _cf_map[i];

        // ignore Fine Points
        if (CF == FINE) { return 0; }

        IndexType start = _A_offsets[i];
        IndexType end   = _A_offsets[i + 1];

        for (IndexType jp = start; jp < end; jp++)
        {
            IndexType j = _A_cols[jp];

            // check column in bounds
            if (j >= _num_rows)
            {
                printf("j OOB: %d\n", j);
            }
            else
            {
                // skip diag
                if (i == j) { continue; }

                // if strongly connected to coarse point, return 1
                if (_cf_map[j] == COARSE && _s_con[jp])
                {
                    return 1;
                }
            }
        }

        // no conflicts, return 0
        return 0;
    }
};

// new host based checker
template <class Matrix, class bVector, class iVector>
int checkCorrectness(const Matrix &A, const bVector &s_con, const iVector &cf_map)
{
    int wrong_count = 0;

    for (int i = 0; i < A.get_num_rows(); i++)
    {
        if (cf_map[i] == COARSE) // don't care about fine points
        {
            // I'm a coarse point, look at entries in my row
            for (int jp = A.row_offsets[i]; jp < A.row_offsets[i + 1]; jp++)
            {
                const int j = A.col_indices[jp];

                // if strongly connected to another coarse point, count.
                if (cf_map[j] == COARSE && s_con[jp])
                {
                    printf("wrong connection on row: %d: %d\n", i, j);
                    wrong_count++;
                }
            }
        }
    }

    return wrong_count;
}

DECLARE_UNITTEST_BEGIN(ClassicalPMISTest);

void run()
{
    typedef Vector<typename TConfig::template setVecPrec<AMGX_vecBool>::Type> BVector;
    typedef Vector<typename TConfig::template setVecPrec<AMGX_vecFloat>::Type> FVector;
    typedef Vector<typename TConfig::template setVecPrec<AMGX_vecInt>::Type> IVector;
    AMG_Config cfg;
    cfg.parseParameterString("");
    const std::string cfg_scope = "default";
    this->randomize( 45 );
    // allocate strength
    Strength<TConfig> *strength;
    strength = StrengthFactory<TConfig>::allocate(cfg, cfg_scope);
    // allocate selector
    classical::Selector<TConfig> *selector;
    selector = classical::SelectorFactory<TConfig>::allocate(cfg, cfg_scope);
    // check selector was generated
    this->PrintOnFail("PMIS: selector not created");
    UNITTEST_ASSERT_TRUE(selector != NULL);
    // we need strength of connection & weights to generate a selection
    Matrix<TConfig> A;
    generateMatrixRandomStruct<TConfig>::generate(A, 45, false, 1, true);
    //generateMatrixRandomStruct<TConfig>::generate(A,4500,false, 1, true);
    cudaCheckError();
    random_fill(A);
    cudaCheckError();
    BVector s_con(A.get_num_nz(), false);
    FVector weights(A.get_num_rows(), 0.0f);
    strength->computeStrongConnectionsAndWeights(A, s_con, weights, 1.1);
    cudaCheckError();
    // now we can create a selection
    IVector cf_map(A.get_num_rows(), -4);
    IVector scratch(A.get_num_rows(), 0);
    // mark C/F points
    selector->markCoarseFinePoints(A, weights, s_con, cf_map, scratch);
    cudaCheckError();
    // check no points unassigned
    int num_unassigned = thrust::count(cf_map.begin(), cf_map.end(), UNASSIGNED);
    int num_coarse     = thrust::count(cf_map.begin(), cf_map.end(), COARSE);
    this->PrintOnFail("PMIS: Some points unassigned");
    UNITTEST_ASSERT_TRUE(num_unassigned == 0);
    this->PrintOnFail("PMIS: No coarse points assigned");
    UNITTEST_ASSERT_TRUE(num_coarse > 0);
#ifdef HOST_ONLY_CHECKER
    int num_wrong = checkCorrectness(A, s_con, cf_map);
    printf("%d wrong connections\n", num_wrong);
#endif
    // check selection - no strong connections between 2 coarse points
    CheckSelection<typename TConfig::IndPrec, bool> checker(
        A.row_offsets.raw(),
        A.col_indices.raw(),
        s_con.raw(),
        cf_map.raw(),
        A.get_num_rows()
    );
    cudaCheckError();
    typedef thrust::counting_iterator<IndexType> c_iter;
    IVector rows(A.get_num_rows(), 0);

    for (int i = 0; i < A.get_num_rows(); i++)
    {
        rows[i] = i;
    }

    IndexType num_bad =
        thrust::transform_reduce(rows.begin(),
                                 rows.end(),
                                 checker, (IndexType)0, thrust::plus<IndexType>());
    cudaCheckError();
    this->PrintOnFail("PMIS: Invalid selection on %d rows", num_bad);
    UNITTEST_ASSERT_TRUE(num_bad == 0);
}

DECLARE_UNITTEST_END(ClassicalPMISTest);

/*
ClassicalPMISTest <TemplateMode<AMGX_mode_hDDI>::Type>  ClassicalPMISTest_instance_mode_hDDI;
ClassicalPMISTest <TemplateMode<AMGX_mode_dDDI>::Type>  ClassicalPMISTest_instance_mode_dDDI;
ClassicalPMISTest <TemplateMode<AMGX_mode_hDFI>::Type>  ClassicalPMISTest_instance_mode_hDFI;
ClassicalPMISTest <TemplateMode<AMGX_mode_hFFI>::Type>  ClassicalPMISTest_instance_mode_hFFI;
ClassicalPMISTest <TemplateMode<AMGX_mode_dDFI>::Type>  ClassicalPMISTest_instance_mode_dDFI;
ClassicalPMISTest <TemplateMode<AMGX_mode_dFFI>::Type>  ClassicalPMISTest_instance_mode_dFFI;
*/

} // namespace amgx
