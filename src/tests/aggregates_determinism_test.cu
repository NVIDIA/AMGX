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
#include <determinism_checker.h>

namespace amgx

{

// parameter is used as test name
DECLARE_UNITTEST_BEGIN(AggregatesDeterminism);

void run()
{
    set_forge_ahead(true);
    
    AMG_Config cfg;
    cfg.parseParameterString("determinism_flag=1, reorder_cols_by_color=1, matrix_coloring_scheme=MIN_MAX, coloring_level=2, aggregation_passes=1, full_ghost_level=1");
    randomize( 30 );
    const int runs_per_selector = 8;

    MatrixA A;
    IVector vecs[runs_per_selector + 1];
    IVector vecs_b[runs_per_selector + 1];
    int nums[runs_per_selector + 1];

    Matrix<TConfig_h> tA;
    generatePoissonForTest(tA, 1, 0, 27, 20, 20, 20);
    // perturb
    for (auto& val: tA.values)
        val += (double)rand()/((double)RAND_MAX*50);
    A = tA;
    
    typename aggregation::SelectorFactory<T_Config>::Iterator iter = aggregation::SelectorFactory<T_Config>::getIterator();
    aggregation::Selector<TConfig> *selector;
    testing_tools::hash_path_determinism_checker *dc = testing_tools::hash_path_determinism_checker::singleton();

    while (!aggregation::SelectorFactory<T_Config>::isIteratorLast(iter))
    {
        std::string m_name = iter->first.c_str();

        if ((m_name.compare("GEO") == 0) || // skip GEO selector
                (m_name.compare("PARALLEL_GREEDY_SELECTOR") == 0) ||  // non-deterministic
                (m_name.compare("DUMMY") == 0) ) // dummy is not a real selector
        {
            ++iter;
            continue;
        }

        for (int i = 0; i < runs_per_selector; i++)
        {
            PrintOnFail("processing : %s\n", iter->first.c_str());
            selector  = iter->second->create(cfg, "default");
            PrintOnFail("Selector creation\n");
            UNITTEST_ASSERT_TRUE(selector != NULL);
            selector ->setAggregates(A, vecs[i], vecs_b[i], nums[i]);

            if (selector != NULL) { delete selector; }
        }

        //check across runs
        for (int i = 1; i < runs_per_selector; i++)
        {
            PrintOnFail(": %s Deterministic aggregates: got %d and %d aggregates number for different runs\n", iter->first.c_str(), nums[0], nums[i]);
            UNITTEST_ASSERT_EQUAL(nums[0], nums[i]);
            PrintOnFail(": %s Deterministic aggregates: different aggregates vector for different runs\n", iter->first.c_str());
            UNITTEST_ASSERT_EQUAL(vecs[0], vecs[i]);
            PrintOnFail(": %s Deterministic aggregates: different global aggregates vector for different runs\n", iter->first.c_str());
            UNITTEST_ASSERT_EQUAL(vecs_b[0], vecs_b[i]);
        }

        ++iter;
    }
}

DECLARE_UNITTEST_END(AggregatesDeterminism);

// or run for all device configs
#define AMGX_CASE_LINE(CASE) AggregatesDeterminism <TemplateMode<CASE>::Type>  AggregatesDeterminism_##CASE;
AMGX_FORALL_BUILDS_DEVICE(AMGX_CASE_LINE)
#undef AMGX_CASE_LINE


} //namespace amgx
