// SPDX-FileCopyrightText: 2011 - 2025 NVIDIA CORPORATION. All Rights Reserved.
//
// SPDX-License-Identifier: BSD-3-Clause

#include "unit_test.h"
#include "amg_solver.h"
#include "aggregation/selectors/size2_selector.h"
#include <aggregation/coarseAgenerators/low_deg_coarse_A_generator.h>
#include <matrix_io.h>
#include "test_utils.h"
#include "util.h"
#include "time.h"
#include <determinism_checker.h>

namespace amgx

{

// parameter is used as test name
DECLARE_UNITTEST_BEGIN(LowDegDeterminism);

void run()
{
    AMG_Config cfg;
    cfg.parseParameterString("determinism_flag=1,reorder_cols_by_color=1,matrix_coloring_scheme=MIN_MAX,coloring_level=2,aggregation_passes=1,full_ghost_level=1");
    const std::string cfg_scope = "default";
    randomize( 30 );
    testing_tools::hash_path_determinism_checker *dc = testing_tools::hash_path_determinism_checker::singleton();
    const int iterations = 10;
    const int matrix_size = 20000;

    for (int matrixi = 0; matrixi < 10; matrixi++)
    {
        MatrixA A;
        MatrixA A2; //matrix with reorderered columns: are the algorithms giving the same aggregates?

        generateMatrixRandomStruct<TConfig>::generate(A, matrix_size, (matrixi % 2) == 0, (matrixi % 5) + 1, false);
        random_fill(A);

        //tests low_deg with the same matrix with reordered columns
        A2 = A;
        A2.set_initialized(0);
        A2.addProps(CSR);
        A2.colorMatrix(cfg, cfg_scope);
        A2.reorderColumnsByColor(false);
        A2.permuteValues();
        A2.set_initialized(1);
        long long unsigned int last_hash_in, cur_hash_in, last_hash_out, cur_hash_out;
        IndexType numRows = A.get_num_rows();
        //create aggregates
        aggregation::size2_selector::Size2Selector<TConfig> selector( cfg, cfg_scope );
        IVector aggregates;
        IVector aggregates_global;
        IndexType numAggregates;
        selector.setAggregates( A, aggregates, aggregates_global, numAggregates );
        //create restriction
        IVector R_row_offsets, R_col_indices;
        IVector R_row_indices(aggregates);
        R_row_offsets.resize(numAggregates + 1);
        R_col_indices.resize(numRows);
        thrust_wrapper::sequence<TConfig::memSpace>(R_col_indices.begin(), R_col_indices.end());
        amgx::thrust::sort_by_key(R_row_indices.begin(), R_row_indices.end(), R_col_indices.begin());
        cusp::detail::indices_to_offsets(R_row_indices, R_row_offsets);
        //restriction ready

        for (int i = 0; i < iterations; i++)
        {
            //recreate the objects each run to avoid internal state changes
            aggregation::LowDegCoarseAGenerator<TConfig> low_deg( cfg, cfg_scope );
            Matrix<TConfig> nextA;
            //compute hash for low_deg input. this ensures that the data is not accidently modified
            cur_hash_in = 8638 * dc->checksum( R_row_offsets.raw(), R_row_offsets.size() * sizeof(IndexType) ) +
                          3986 * dc->checksum( R_col_indices.raw(), R_col_indices.size() * sizeof(IndexType) ) +
                          3507 * dc->checksum( aggregates.raw(), aggregates.size() * sizeof(IndexType) ) +
                          1861 * dc->checksum( A.row_offsets.raw(), A.row_offsets.size() * sizeof(IndexType) ) +
                          7742 * dc->checksum( A.col_indices.raw(), A.col_indices.size() * sizeof(IndexType) ) +
                          1117 * dc->checksum( A.values.raw(), A.values.size() * sizeof(ValueTypeA) ) +
                          6753 * numAggregates;

            if ( i > 0 )
            {
                std::stringstream fail_msg;
                fail_msg << "Input has been modified in run " << i << ". That renders the test meaningless.";
                this->PrintOnFail( fail_msg.str().c_str() );
                UNITTEST_ASSERT_EQUAL( cur_hash_in, last_hash_in );
            }

            last_hash_in = cur_hash_in;
            low_deg.computeAOperator( A,
                                      nextA, // <- output
                                      aggregates,
                                      R_row_offsets,
                                      R_col_indices,
                                      numAggregates);
            cur_hash_out = 2469 * dc->checksum( nextA.row_offsets.raw(), nextA.row_offsets.size() * sizeof(IndexType) ) +
                           7896 * dc->checksum( nextA.col_indices.raw(), nextA.col_indices.size() * sizeof(IndexType) ) +
                           9214 * dc->checksum( nextA.values.raw(), nextA.values.size() * sizeof(ValueTypeA) );

            if ( i > 0 )
            {
                std::stringstream fail_msg;
                fail_msg << "Test failed: Bitwise comparision of output to last run has failed in run " << i;
                this->PrintOnFail( fail_msg.str().c_str() );
                UNITTEST_ASSERT_EQUAL( cur_hash_out, last_hash_out );
            }

            last_hash_out = cur_hash_out;
        }
    }
}

DECLARE_UNITTEST_END(LowDegDeterminism);

// or run for all device configs
#define AMGX_CASE_LINE(CASE) LowDegDeterminism <TemplateMode<CASE>::Type>  LowDegDeterminism_##CASE;
AMGX_FORALL_BUILDS_DEVICE(AMGX_CASE_LINE)
#undef AMGX_CASE_LINE


} //namespace amgx
