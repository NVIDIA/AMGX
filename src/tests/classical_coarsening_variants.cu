// SPDX-FileCopyrightText: 2011 - 2025 NVIDIA CORPORATION. All Rights Reserved.
//
// SPDX-License-Identifier: BSD-3-Clause

#include "unit_test.h"
#include <classical/interpolators/common.h>
#include <classical/selectors/selector.h>
#include <classical/strength/strength_base.h>
#include <matrix_io.h>

namespace amgx
{

DECLARE_UNITTEST_BEGIN(ClassicalCoarseningVariants);

typedef Vector<typename TConfig::template setVecPrec<AMGX_vecBool>::Type> BVector;
typedef Vector<typename TConfig::template setVecPrec<AMGX_vecFloat>::Type> FVector;
typedef Vector<typename TConfig_h::template setVecPrec<AMGX_vecBool>::Type> BVector_h;

void check_selector(const char *selector_name, const Matrix_h &host_matrix)
{
    AMG_Config cfg;
    std::string parameters = std::string("selector=") + selector_name +
                             ", strength=AHAT, strength_threshold=0.25, "
                             "determinism_flag=1, use_opt_kernels=0";
    UNITTEST_ASSERT_EQUAL(cfg.parseParameterString(parameters.c_str()), AMGX_OK);

    Matrix<TConfig> A = host_matrix;
    BVector strong_connections(A.get_num_nz(), false);
    FVector weights(A.get_num_rows(), 0.0f);
    IVector cf_map(A.get_num_rows(), UNASSIGNED);
    IVector scratch(A.get_num_rows(), 0);

    Strength<TConfig> *strength = StrengthFactory<TConfig>::allocate(cfg, "default");
    UNITTEST_ASSERT_TRUE(strength != NULL);
    strength->computeStrongConnectionsAndWeights(A, strong_connections, weights, 1.1);
    delete strength;

    classical::Selector<TConfig> *selector =
        classical::SelectorFactory<TConfig>::allocate(cfg, "default");
    UNITTEST_ASSERT_TRUE(selector != NULL);
    selector->markCoarseFinePoints(A, weights, strong_connections, cf_map, scratch);
    delete selector;

    IVector_h host_cf_map = cf_map;
    BVector_h host_connections = strong_connections;
    int num_coarse = 0;

    for (int row = 0; row < host_matrix.get_num_rows(); ++row)
    {
        const int state = host_cf_map[row];
        this->PrintOnFail("%s left invalid state %d at row %d", selector_name, state, row);
        UNITTEST_ASSERT_TRUE(state == COARSE || state == FINE || state == STRONG_FINE);
        num_coarse += state == COARSE;
    }

    this->PrintOnFail("%s selected no coarse points", selector_name);
    UNITTEST_ASSERT_TRUE(num_coarse > 0);
    UNITTEST_ASSERT_TRUE(num_coarse < host_matrix.get_num_rows());

    for (int row = 0; row < host_matrix.get_num_rows(); ++row)
    {
        if (host_cf_map[row] != COARSE) { continue; }

        for (int jj = host_matrix.row_offsets[row]; jj < host_matrix.row_offsets[row + 1]; ++jj)
        {
            const int col = host_matrix.col_indices[jj];
            if (col != row && host_connections[jj])
            {
                this->PrintOnFail("%s selected strongly connected coarse rows %d and %d",
                                  selector_name, row, col);
                UNITTEST_ASSERT_TRUE(host_cf_map[col] != COARSE);
            }
        }
    }
}

void run()
{
    Matrix_h A;
    A.set_initialized(0);
    A.addProps(CSR);
    MatrixCusp<TConfig_h, cusp::csr_format> wrapped_A(&A);
    cusp::gallery::poisson5pt(wrapped_A, 8, 8);
    A.computeDiagonal();
    A.set_initialized(1);

    const char *selectors[] = {"PMIS", "HMIS"};
    for (const char *selector : selectors) { check_selector(selector, A); }

    if (TConfig::memSpace == AMGX_device)
    {
        check_selector("AGGRESSIVE_PMIS", A);
        check_selector("AGGRESSIVE_HMIS", A);
    }
}

DECLARE_UNITTEST_END(ClassicalCoarseningVariants);

ClassicalCoarseningVariants<TemplateMode<AMGX_mode_hDDI>::Type> ClassicalCoarseningVariants_hDDI;
ClassicalCoarseningVariants<TemplateMode<AMGX_mode_hDFI>::Type> ClassicalCoarseningVariants_hDFI;
ClassicalCoarseningVariants<TemplateMode<AMGX_mode_hFFI>::Type> ClassicalCoarseningVariants_hFFI;
ClassicalCoarseningVariants<TemplateMode<AMGX_mode_dDDI>::Type> ClassicalCoarseningVariants_dDDI;
ClassicalCoarseningVariants<TemplateMode<AMGX_mode_dDFI>::Type> ClassicalCoarseningVariants_dDFI;
ClassicalCoarseningVariants<TemplateMode<AMGX_mode_dFFI>::Type> ClassicalCoarseningVariants_dFFI;

} // namespace amgx
