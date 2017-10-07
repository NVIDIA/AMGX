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
#include "aggregation/selectors/size2_selector.h"
#include "aggregation/selectors/size4_selector.h"
#include "aggregation/selectors/size8_selector.h"
#include "aggregation/selectors/multi_pairwise.h"
#include "aggregation/selectors/agg_selector.h"
#include "aggregation/aggregation_amg_level.h"
#include <matrix_io.h>
#include "test_utils.h"
#include "util.h"
#include "time.h"
#include <determinism_checker.h>

//#define DEBUG_MULTIPAIRWISE_TEST
//#define CHK_MULTIPAIRWISE_TEST

#ifdef DEBUG_MULTIPAIRWISE_TEST
#define DEBUG_OUT(str) std::cout << str << std::endl
#else
#define DEBUG_OUT(str)
#endif

#ifdef CHK_MULTIPAIRWISE_TEST
#define CHK_VECTOR( vector, type ) testing_tools::hash_path_determinism_checker::singleton()->checkpoint( #vector, vector.raw(), vector.size()*sizeof( type ), true );
#else
#define CHK_VECTOR( vector, type )
#endif

#ifdef CHK_MULTIPAIRWISE_TEST
#define CHK_MATRIX( matrix ) CHK_VECTOR( matrix.row_offsets, IndexType );\
                             CHK_VECTOR( matrix.col_indices, IndexType );\
                             CHK_VECTOR( matrix.values, ValueTypeA )
#else
#define CHK_MATRIX( matrix )
#endif



namespace amgx

{

// parameter is used as test name
DECLARE_UNITTEST_BEGIN(MultiPairwise);

void run()
{
    //setup multi pairwise aggregator
    AMG_Configuration cfg_mp;
    cfg_mp.parseParameterString("config_version=2,\
                                 determinism_flag=1,\
                                 solver(main)=AMG,\
                                 main:algorithm=AGGREGATION,\
                                 main:selector=MULTI_PAIRWISE,\
                                 main:full_ghost_level=1,\
                                 main:merge_singletons=1,\
                                 main:weight_formula=0,\
                                 main:aggregation_passes=3,\
                                 main:filter_weights=0,\
                                 main:serial_matching=0,\
                                 main:max_levels=2,\
                                 main:presweeps=0,\
                                 main:postsweeps=3,\
                                 main:coarsest_sweeps=6,\
                                 main:tolerance=1e-3,\
                                 main:smoother=MULTICOLOR_DILU,\
                                 main:matrix_coloring_scheme=MULTI_HASH,\
                                 main:coarseAgenerator=LOW_DEG,\
                                 main:print_aggregation_info=0,\
                                 main:print_grid_stats=1,\
                                 main:print_solve_stats=1,\
                                 main:monitor_residual=1,\
                                 main:max_iters=100");
    //setup size 2 aggregator
    AMG_Configuration cfg_size2;
    cfg_size2.parseParameterString("config_version=2,\
                                    determinism_flag=1,\
                                    solver(main)=AMG,\
                                    main:algorithm=AGGREGATION,\
                                    main:selector=SIZE_2,\
                                    main:merge_singletons=1,\
                                    main:max_levels=2,\
                                    main:presweeps=0,\
                                    main:postsweeps=3,\
                                    main:coarse_solver(main2)=AMG,\
                                    main:coarseAgenerator=LOW_DEG,\
                                    main:print_grid_stats=1,\
                                    main:print_aggregation_info=0,\
                                    main2:print_aggregation_info=0,\
                                    main2:print_grid_stats=1,\
                                    main2:presweeps=0,\
                                    main2:postsweeps=0,\
                                    main2:algorithm=AGGREGATION,\
                                    main2:selector=SIZE_2,\
                                    main2:max_iters=1,\
                                    main2:tolerance=0.0,\
                                    main2:coarsest_sweeps=6,\
                                    main2:max_levels=3,\
                                    main2:smoother=MULTICOLOR_DILU,\
                                    main2:matrix_coloring_scheme=MULTI_HASH,\
                                    main2:coarseAgenerator=LOW_DEG,\
                                    main:tolerance=1e-3,\
                                    main:print_solve_stats=1,\
                                    main:smoother=MULTICOLOR_DILU,\
                                    main:matrix_coloring_scheme=MULTI_HASH,\
                                    main:monitor_residual=1,\
                                    main:max_iters=100");
    const int passes = 1;
    std::string filename;

    for (int current_pass = 0; current_pass < passes; current_pass++ )
    {
        filename = UnitTest::get_configuration().data_folder + "Public/florida/";

        switch (current_pass)
        {
            case 0:
                filename += "atmosdd.mtx";
                break;
        }

        Resources res;        // default resources
        {
            MatrixA A;
            Vector<T_Config> x_mp, x_size2, b;
            Vector_h x_h, b_h;
            Matrix_h A_h;
            typedef TemplateConfig<AMGX_host,   T_Config::vecPrec, T_Config::matPrec, AMGX_indInt> Config_h;
            //read matrix
            A_h.set_initialized(0);
            A_h.addProps(CSR);
            std::string fail_msg = "Cannot open " + filename;
            this->PrintOnFail(fail_msg.c_str());
            UNITTEST_ASSERT_TRUE(MatrixIO<Config_h>::readSystem( filename.c_str(), A_h, b_h, x_h ) == AMGX_OK);
            A_h.set_initialized(1);
            A = A_h;
            x_mp = x_h;
            x_size2 = x_h;
            b = b_h;
            cudaDeviceSynchronize();
            cudaCheckError();
            b.set_block_dimx(1);
            b.set_block_dimy(A.get_block_dimy());
            x_mp.set_block_dimx(1);
            x_mp.set_block_dimy(A.get_block_dimx());
            x_mp.resize( b.size(), 1.0 );
            x_size2.set_block_dimx(1);
            x_size2.set_block_dimy(A.get_block_dimx());
            x_size2.resize( b.size(), 1.0 );
            AMG_Solver<T_Config> amg_mp(&res, cfg_mp);
            AMG_Solver<T_Config> amg_size2(&res, cfg_size2);
            cudaDeviceSynchronize();
            cudaCheckError();
            this->PrintOnFail( "failed to run setup");
            UNITTEST_ASSERT_TRUE( AMGX_OK == amg_mp.setup( A ) );
            UNITTEST_ASSERT_TRUE( AMGX_OK == amg_size2.setup( A ) );
            cudaDeviceSynchronize();
            cudaCheckError();
            this->PrintOnFail( "failed to run solve" );
            AMGX_STATUS solve_status = AMGX_ST_CONVERGED;
            UNITTEST_ASSERT_TRUE( AMGX_OK == amg_mp.solve( b, x_mp, solve_status, false ) );
            UNITTEST_ASSERT_TRUE( AMGX_OK == amg_size2.solve( b, x_size2, solve_status, false ) );
            cudaDeviceSynchronize();
            cudaCheckError();
            this->PrintOnFail( "number of iterations must be the same for both setup but is different" );
            UNITTEST_ASSERT_EQUAL( amg_mp.get_num_iters(), amg_size2.get_num_iters() );
        }
    }
}

DECLARE_UNITTEST_END(MultiPairwise);

// or run for all device configs
#define AMGX_CASE_LINE(CASE) MultiPairwise <TemplateMode<CASE>::Type>  MultiPairwise_##CASE;
AMGX_FORALL_BUILDS_DEVICE(AMGX_CASE_LINE)
#undef AMGX_CASE_LINE


} //namespace amgx


