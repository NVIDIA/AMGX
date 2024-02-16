// SPDX-FileCopyrightText: 2011 - 2024 NVIDIA CORPORATION. All Rights Reserved.
//
// SPDX-License-Identifier: BSD-3-Clause

#include "unit_test.h"
#include "amg_solver.h"
#include "aggregation/selectors/size2_selector.h"
#include "aggregation/selectors/size4_selector.h"
#include "aggregation/selectors/size8_selector.h"
#include "aggregation/selectors/agg_selector.h"
#include <matrix_io.h>
#include "test_utils.h"
#include "util.h"
#include "time.h"

namespace amgx

{

// parameter is used as test name
DECLARE_UNITTEST_BEGIN(AggregatesDiagonalOutside);

void run()
{
    set_forge_ahead(true);
    AMG_Config cfg;
    cfg.parseParameterString("determinism_flag=1");
    randomize( 30 );
    std::vector< aggregation::Selector<TConfig>* > selectors;
    std::vector<std::string> selector_names;
    selectors.push_back(new aggregation::size2_selector::Size2Selector<T_Config>(cfg, "default"));
    selectors.push_back(new aggregation::size8_selector::Size8Selector<T_Config>(cfg, "default"));
    selectors.push_back(new aggregation::size4_selector::Size4Selector<T_Config>(cfg, "default"));
    selector_names.push_back("Size2");
    selector_names.push_back("Size8");
    selector_names.push_back("Size4");

    for (unsigned int i = 0; i < selectors.size(); i++)
    {
        PrintOnFail("Selector creation\n");
        UNITTEST_ASSERT_TRUE(selectors[i] != NULL);
    }

    Matrix_h A_diag_h;
    IVector vec1, vec2, vec3;
    int num1 = 0, num2 = -1;

    // check different block sizes
    for (int bsize = 1; bsize < 5; bsize++)
    {
        // Create random scalar matrix with diagonal inside
        generateMatrixRandomStruct<TConfig_h>::generate(A_diag_h, 1000, true, bsize, false);
        random_fill(A_diag_h);
        // Create a new matrix with diagonal outside
        Matrix_h A_h;
        A_h.convert(A_diag_h, CSR, bsize, bsize);
        MatrixA A, A_diag;
        A = A_diag_h;
        A_diag = A_h;

        for (unsigned int i = 0; i < selectors.size(); i++)
        {
            selectors[i]->setAggregates(A, vec1, vec3, num1);
            selectors[i]->setAggregates(A_diag, vec2, vec3, num2);
            PrintOnFail(selector_names[i].c_str());
            PrintOnFail(": Deterministic aggregates: got %d and %d for one matrix\n", num1, num2);
            UNITTEST_ASSERT_EQUAL(vec1, vec2);
        }
    }

    for (unsigned s = 0; s < selectors.size(); ++s)
    {
        delete selectors[s];
    }
}

DECLARE_UNITTEST_END(AggregatesDiagonalOutside);


// if you want to be able run this test for all available configs you can write this:
//#define AMGX_CASE_LINE(CASE) TemplateTest <TemplateMode<CASE>::Type>  TemplateTest_##CASE;
//  AMGX_FORALL_BUILDS(AMGX_CASE_LINE)
//#undef AMGX_CASE_LINE

// or run for all device configs
#define AMGX_CASE_LINE(CASE) AggregatesDiagonalOutside <TemplateMode<CASE>::Type>  AggregatesDiagonalOutside ##CASE;
AMGX_FORALL_BUILDS_DEVICE(AMGX_CASE_LINE)
#undef AMGX_CASE_LINE

// or you can specify several desired configs
//TemplateTest <TemplateMode<AMGX_mode_hDFI>::Type>  TemplateTest_hDFI;
//TemplateTest <TemplateMode<AMGX_mode_dDFI>::Type>  TemplateTest_dDFI;


} //namespace amgx
