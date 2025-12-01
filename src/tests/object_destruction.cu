// SPDX-FileCopyrightText: 2011 - 2025 NVIDIA CORPORATION. All Rights Reserved.
//
// SPDX-License-Identifier: BSD-3-Clause

#include "unit_test.h"
#include "matrix_io.h"
#include "amg_solver.h"
#include "amgx_c.h"
#include "test_utils.h"
#include "util.h"
#include "time.h"

namespace amgx
{

DECLARE_UNITTEST_BEGIN(ObjectDestructionSequence);

void run()
{
    int testNum = 10;
    Matrix_h Ah;
    generatePoissonForTest(Ah, 1, 0, 27, 40, 40, 40);
    // objects
    AMGX_config_handle cfg;
    AMGX_matrix_handle A;
    AMGX_vector_handle b;
    AMGX_vector_handle x;
    AMGX_solver_handle solver;
    AMGX_config_handle rsrc_cfg = NULL;
    AMGX_resources_handle rsrc = NULL;
    AMGX_Mode mode = (AMGX_Mode)TConfig::mode;


    // repeat creation and destruction
    for (int t = 0; t < testNum; t++)
    {
        // create
        UNITTEST_ASSERT_EQUAL(AMGX_RC_OK, AMGX_config_create(&cfg, "cycle=V, spmm_gmem_size=16384"));
        UNITTEST_ASSERT_EQUAL(AMGX_config_create(&rsrc_cfg, ""), AMGX_OK);
        int device = 0;
        UNITTEST_ASSERT_EQUAL(AMGX_resources_create(&rsrc, rsrc_cfg, NULL, 1, &device), AMGX_OK);
        UNITTEST_ASSERT_EQUAL(AMGX_RC_OK, AMGX_matrix_create(&A, rsrc, mode));
        UNITTEST_ASSERT_EQUAL(AMGX_RC_OK, AMGX_vector_create(&x, rsrc, mode));
        UNITTEST_ASSERT_EQUAL(AMGX_RC_OK, AMGX_vector_create(&b, rsrc, mode));
        UNITTEST_ASSERT_EQUAL(AMGX_RC_OK, AMGX_solver_create(&solver, rsrc, mode, cfg));
        UNITTEST_ASSERT_EQUAL(AMGX_RC_OK, AMGX_matrix_upload_all(A, Ah.get_num_rows(), Ah.get_num_nz(), Ah.get_block_dimx(), Ah.get_block_dimy(), Ah.row_offsets.raw(), Ah.col_indices.raw(), Ah.values.raw(), NULL));
        UNITTEST_ASSERT_EQUAL(AMGX_RC_OK, AMGX_solver_setup(solver, A));
        // destroy
        UNITTEST_ASSERT_EQUAL(AMGX_RC_OK, AMGX_matrix_destroy(A));
        UNITTEST_ASSERT_EQUAL(AMGX_RC_OK, AMGX_vector_destroy(x));
        UNITTEST_ASSERT_EQUAL(AMGX_RC_OK, AMGX_vector_destroy(b));
        UNITTEST_ASSERT_EQUAL(AMGX_RC_OK, AMGX_solver_destroy(solver));
        UNITTEST_ASSERT_EQUAL(AMGX_RC_OK, AMGX_config_destroy(cfg));
        UNITTEST_ASSERT_EQUAL(AMGX_config_destroy( rsrc_cfg ), AMGX_OK);
        UNITTEST_ASSERT_EQUAL(AMGX_resources_destroy( rsrc ), AMGX_OK);
    }
}

DECLARE_UNITTEST_END(ObjectDestructionSequence);

#define AMGX_CASE_LINE(CASE) ObjectDestructionSequence<TemplateMode<CASE>::Type> ObjectDestructionSequence_##CASE;
AMGX_FORALL_BUILDS_DEVICE(AMGX_CASE_LINE)
#undef AMGX_CASE_LINE

} //namespace amgx
