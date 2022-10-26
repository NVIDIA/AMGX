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
#include "amg_solver.h"
#include "test_utils.h"
#include "matrix_coloring/min_max.h"
#include "matrix_coloring/multi_hash.h"
#include "matrix_coloring/min_max_2ring.h"
#include <thrust/iterator/counting_iterator.h>

namespace amgx

{

template<typename IndexType>
struct CheckColoringVC
{
    const IndexType *_A_offsets;
    const IndexType *_A_cols;
    const IndexType *_colors;
    const int _coloring_level;

    CheckColoringVC(const IndexType *A_offsets, const IndexType *A_cols, const IndexType *colors, const int coloring_level) : _A_offsets(A_offsets), _A_cols(A_cols), _colors(colors), _coloring_level(coloring_level) { }

    __host__ __device__ IndexType operator()(IndexType i)
    {
        IndexType col = _colors[i];

        // don't count uncolored nodes
        if (col == 0) { return 0; }

        IndexType row_start = _A_offsets[i];
        IndexType row_end = _A_offsets[i + 1];

        for (IndexType r = row_start; r < row_end; r++)
        {
            IndexType j = _A_cols[r];

            // skip diagonal
            if (j == i) { continue; }

            IndexType jcol = _colors[j];

            if (jcol == 0) { continue; }

            // if 2 colors are adjacent, return 1
            if (jcol == col)
            {
                return 1;
            }

            if (_coloring_level == 2)
            {
                // Check neighbours neighbours
                IndexType row_start_nei = _A_offsets[j];
                IndexType row_end_nei = _A_offsets[j + 1];

                for (IndexType rr = row_start_nei; rr < row_end_nei; rr++)
                {
                    IndexType k = _A_cols[rr];

                    // skip diagonal
                    if (k == i) { continue; }

                    IndexType kcol = _colors[k];

                    if (kcol == 0) { continue; }

                    // if 2 colors are 2-ring adjacent, return 1
                    if (kcol == col)
                    {
                        return 1;
                    }
                } // loop over two ring neighbours
            } // if (coloring_level==2)
        } // loop over first ring neighbours

        // no conflict => return 0
        return 0;
    }
};

// parameter is used as test name
DECLARE_UNITTEST_BEGIN(MinMaxColoringTest);

void run()
{
    randomize(10125);
    const std::string cfg_scope = "default";
    const int max_iters = 10;
    std::vector<std::string> coloring_configs;
    coloring_configs.push_back("max_uncolored_percentage=0, coloring_level=1");
    coloring_configs.push_back("max_uncolored_percentage=0, coloring_level=2");
    coloring_configs.push_back("max_uncolored_percentage=0, coloring_level=1, matrix_coloring_scheme=MIN_MAX_2RING");
    coloring_configs.push_back("max_uncolored_percentage=0, coloring_level=1, matrix_coloring_scheme=GREEDY_MIN_MAX_2RING");
    coloring_configs.push_back("max_uncolored_percentage=0, coloring_level=2, matrix_coloring_scheme=GREEDY_MIN_MAX_2RING");
    std::vector<AMG_Config> cfg(coloring_configs.size());

    for (int k = 0; k < coloring_configs.size(); k++)
    {
        UNITTEST_ASSERT_TRUE ( (cfg[k]).parseParameterString(coloring_configs[k].c_str()) == AMGX_OK);
    }

    for (int i = 0; i < max_iters; i++)
    {
        MatrixA  A;
        generateMatrixRandomStruct<TConfig>::generate(A, 10000, (rand() % 2) == 0, max(rand() % 10, 1), true);
        A.set_initialized(0);

        // Check one and two ring coloring
        for (int k = 0; k < cfg.size(); k++)
        {
            //MinMaxMatrixColoring<TConfig>* coloring = new MinMaxMatrixColoring<T_Config>(cfg[k], cfg_scope,A);
            A.colorMatrix(cfg[k], cfg_scope);
            //UNITTEST_ASSERT_TRUE_DESC("Coloring creation", &A.getMatrixColoring() != NULL);
            // check coloring
            const IndexType *A_row_offsets_ptr    = A.row_offsets.raw();
            const IndexType *A_column_indices_ptr = A.col_indices.raw();
            const IndexType *row_colors_ptr = A.getMatrixColoring().getRowColors().raw();
            CheckColoringVC<IndexType> checker(A_row_offsets_ptr, A_column_indices_ptr, row_colors_ptr, A.getMatrixColoring().getColoringLevel());
            IndexType num_bad =
                amgx::thrust::transform_reduce(amgx::thrust::counting_iterator<IndexType>(0), amgx::thrust::counting_iterator<IndexType>(A.get_num_rows()),
                                         checker, (IndexType)0, amgx::thrust::plus<IndexType>());
            //printf("Num bad %s-%d = %d\n", coloring_configs[k].c_str(), i, num_bad);
            UNITTEST_ASSERT_EQUAL_DESC("No invalid rows", num_bad, 0);
            //delete coloring;
        }
    }
}

DECLARE_UNITTEST_END(MinMaxColoringTest);


// if you want to be able run this test for all available configs you can write this:
//#define AMGX_CASE_LINE(CASE) TemplateTest <TemplateMode<CASE>::Type>  TemplateTest_instance_mode##CASE;
//  AMGX_FORALL_BUILDS(AMGX_CASE_LINE)
//#undef AMGX_CASE_LINE

// or you can specify several desired configs
MinMaxColoringTest <TemplateMode<AMGX_mode_dDDI>::Type>  MinMaxColoringTest_instance_mode_dDDI;
MinMaxColoringTest <TemplateMode<AMGX_mode_dFFI>::Type>  MinMaxColoringTest_instance_mode_dFFI;

} //namespace amgx
