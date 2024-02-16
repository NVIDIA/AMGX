// SPDX-FileCopyrightText: 2011 - 2024 NVIDIA CORPORATION. All Rights Reserved.
//
// SPDX-License-Identifier: BSD-3-Clause

#include "unit_test.h"
#include "amg_solver.h"
#include "aggregation/selectors/size2_selector.h"
#include "aggregation/selectors/size4_selector.h"
#include "aggregation/selectors/size8_selector.h"
#include "aggregation/selectors/agg_selector.h"
#include "matrix_coloring/min_max.h"
#include <matrix_io.h>
#include "test_utils.h"
#include <multiply.h>
#include <blas.h>
#include <csr_multiply.h>
#include "util.h"
#include "time.h"

namespace amgx
{

// parameter is used as test name
DECLARE_UNITTEST_BEGIN(AggregatesCoarseningFactor);

void run()
{
    double tol = 1.1;
    set_forge_ahead(true);
    AMG_Config cfg;
    // unassigned tolerance
    UNITTEST_ASSERT_TRUE( cfg.parseParameterString("config_version=2, determinism_flag=1, max_unassigned_percentage=0.1") == AMGX_OK);
    randomize( 30 );
    std::vector< aggregation::Selector<TConfig>* > selectors;
    std::vector<std::string> selector_names;
    std::vector<float> expected_factors;
    selectors.push_back(new aggregation::size2_selector::Size2Selector<T_Config>(cfg, "default"));
    selectors.push_back(new aggregation::size4_selector::Size4Selector<T_Config>(cfg, "default"));
    selectors.push_back(new aggregation::size8_selector::Size8Selector<T_Config>(cfg, "default"));
    selector_names.push_back("Size2");
    selector_names.push_back("Size4");
    selector_names.push_back("Size8");
    expected_factors.push_back(1. / 2.*tol);
    expected_factors.push_back(1. / 4.*tol);
    expected_factors.push_back(1. / 8.*tol);
    // Create poisson matrix
    Matrix<TConfig_h> tA;
    generatePoissonForTest(tA, 1, 0, 27, 20, 20, 20);
    // perturb
    for (auto& val: tA.values)
        val += (double)rand()/((double)RAND_MAX*50);
    MatrixA A_hd = tA;
    //MatrixIO<TConfig_h>::writeSystemMatrixMarket(".temp.mtx", &tA, NULL, NULL);
    int n_rows = A_hd.get_num_rows();

    for (unsigned int i = 0; i < selectors.size(); i++)
    {
        PrintOnFail("Selector creation\n");
        UNITTEST_ASSERT_TRUE(selectors[i] != NULL);
    }

    IVector vec1, vec2;
    int num1 = 0;

    for (unsigned int i = 0; i < selectors.size(); i++)
    {
        selectors[i]->setAggregates(A_hd, vec1, vec2, num1);
        PrintOnFail(selector_names[i].c_str());
        PrintOnFail(": Aggregates factor: got %f expecting %f \n", 1.0 * num1 / n_rows, expected_factors[i]);
        UNITTEST_ASSERT_TRUE(1.0 * num1 / n_rows < expected_factors[i]);
        PrintOnFail(selector_names[i].c_str());
        PrintOnFail(": Aggregator generates NaNs" );

        for (int i = 0; i < vec1.size(); i++)
        {
            UNITTEST_ASSERT_FINITE(vec1[i]);
        }
    }

    for (unsigned s = 0; s < selectors.size(); ++s)
    {
        delete selectors[s];
    }
}

DECLARE_UNITTEST_END(AggregatesCoarseningFactor);


// if you want to be able run this test for all available configs you can write this:
//#define AMGX_CASE_LINE(CASE) TemplateTest <TemplateMode<CASE>::Type>  TemplateTest_##CASE;
//  AMGX_FORALL_BUILDS(AMGX_CASE_LINE)
//#undef AMGX_CASE_LINE

// or run for all device configs
//#define AMGX_CASE_LINE(CASE) TemplateTest <TemplateMode<CASE>::Type>  TemplateTest_##CASE;
//  AMGX_FORALL_BUILDS_DEVICE(AMGX_CASE_LINE)
//#undef AMGX_CASE_LINE

#define AMGX_CASE_LINE(CASE) AggregatesCoarseningFactor <TemplateMode<CASE>::Type>  AggregatesCoarseningFactor_##CASE;
AMGX_FORALL_BUILDS_DEVICE(AMGX_CASE_LINE)
#undef AMGX_CASE_LINE

// or you can specify several desired configs
//TemplateTest <TemplateMode<AMGX_mode_hDFI>::Type>  TemplateTest_hDFI;
//TemplateTest <TemplateMode<AMGX_mode_dDFI>::Type>  TemplateTest_dDFI;


} //namespace amgx
