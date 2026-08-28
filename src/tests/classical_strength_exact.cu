// SPDX-FileCopyrightText: 2011 - 2025 NVIDIA CORPORATION. All Rights Reserved.
//
// SPDX-License-Identifier: BSD-3-Clause

#include "unit_test.h"
#include <classical/strength/strength_base.h>

namespace amgx
{

DECLARE_UNITTEST_BEGIN(ClassicalStrengthExact);

typedef Vector<typename TConfig::template setVecPrec<AMGX_vecBool>::Type> BVector;
typedef Vector<typename TConfig::template setVecPrec<AMGX_vecFloat>::Type> FVector;
typedef Vector<typename TConfig_h::template setVecPrec<AMGX_vecFloat>::Type> FVector_h;

void build_matrix(Matrix<TConfig> &A)
{
    typedef typename TConfig::MatPrec ValueType;
    const int num_rows = 3;
    const int num_nz = 9;
    const int row_offsets[] = {0, 3, 6, 9};
    const int columns[] = {0, 1, 2, 0, 1, 2, 0, 1, 2};
    const double values[] = {4.0, -2.0, -0.5,
                             -1.0, 4.0, -4.0,
                             -0.25, -1.0, 4.0};

    A.set_initialized(0);
    A.addProps(CSR);
    A.resize(num_rows, num_rows, num_nz);

    for (int i = 0; i <= num_rows; ++i) { A.row_offsets[i] = row_offsets[i]; }
    for (int i = 0; i < num_nz; ++i)
    {
        A.col_indices[i] = columns[i];
        A.values[i] = ValueType(values[i]);
    }

    A.computeDiagonal();
    A.set_initialized(1);
}

void check_strength(const char *strength_name, bool use_opt_kernels,
                    const bool expected_connections[9], const int expected_incoming[3])
{
    AMG_Config cfg;
    std::string parameters = std::string("strength=") + strength_name +
                             ", strength_threshold=0.5, determinism_flag=1, use_opt_kernels=" +
                             (use_opt_kernels ? "1" : "0");
    UNITTEST_ASSERT_EQUAL(cfg.parseParameterString(parameters.c_str()), AMGX_OK);

    Matrix<TConfig> A;
    build_matrix(A);
    BVector connections(A.get_num_nz(), false);
    FVector weights(A.get_num_rows(), 0.0f);
    Strength<TConfig> *strength = StrengthFactory<TConfig>::allocate(cfg, "default");
    UNITTEST_ASSERT_TRUE(strength != NULL);
    strength->computeStrongConnectionsAndWeights(A, connections, weights, 1.1);
    delete strength;

    BVector expected(A.get_num_nz(), false);
    for (int i = 0; i < A.get_num_nz(); ++i) { expected[i] = expected_connections[i]; }
    UNITTEST_ASSERT_EQUAL(connections, expected);

    FVector_h host_weights = weights;
    for (int i = 0; i < A.get_num_rows(); ++i)
    {
        this->PrintOnFail("Unexpected incoming strength weight at row %d: %f", i,
                          host_weights[i]);
        UNITTEST_ASSERT_TRUE(host_weights[i] >= expected_incoming[i]);
        UNITTEST_ASSERT_TRUE(host_weights[i] < expected_incoming[i] + 1.0f);
    }
}

void run()
{
    const bool ahat_connections[] = {
        false, true,  false,
        false, false, true,
        false, true,  false
    };
    const int ahat_incoming[] = {0, 2, 1};
    check_strength("AHAT", false, ahat_connections, ahat_incoming);
    check_strength("AHAT", true, ahat_connections, ahat_incoming);

    const bool all_connections[] = {
        false, true, true,
        true, false, true,
        true, true, false
    };
    const int all_incoming[] = {2, 2, 2};
    check_strength("ALL", false, all_connections, all_incoming);
    check_strength("ALL", true, all_connections, all_incoming);
}

DECLARE_UNITTEST_END(ClassicalStrengthExact);

ClassicalStrengthExact<TemplateMode<AMGX_mode_hDDI>::Type> ClassicalStrengthExact_hDDI;
ClassicalStrengthExact<TemplateMode<AMGX_mode_hDFI>::Type> ClassicalStrengthExact_hDFI;
ClassicalStrengthExact<TemplateMode<AMGX_mode_hFFI>::Type> ClassicalStrengthExact_hFFI;
ClassicalStrengthExact<TemplateMode<AMGX_mode_dDDI>::Type> ClassicalStrengthExact_dDDI;
ClassicalStrengthExact<TemplateMode<AMGX_mode_dDFI>::Type> ClassicalStrengthExact_dDFI;
ClassicalStrengthExact<TemplateMode<AMGX_mode_dFFI>::Type> ClassicalStrengthExact_dFFI;

} // namespace amgx
