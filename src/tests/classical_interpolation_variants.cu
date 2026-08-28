// SPDX-FileCopyrightText: 2011 - 2025 NVIDIA CORPORATION. All Rights Reserved.
//
// SPDX-License-Identifier: BSD-3-Clause

#include "unit_test.h"
#include <classical/interpolators/interpolator.h>
#include <classical/selectors/selector.h>
#include <classical/strength/strength_base.h>
#include <matrix_io.h>
#include <cmath>

namespace amgx
{

DECLARE_UNITTEST_BEGIN(ClassicalInterpolationVariants);

typedef Vector<typename TConfig::template setVecPrec<AMGX_vecBool>::Type> BVector;
typedef Vector<typename TConfig::template setVecPrec<AMGX_vecFloat>::Type> FVector;
typedef typename Matrix_h::value_type HostValueType;

void check_interpolator(const char *interpolator_name, const Matrix_h &host_matrix)
{
    AMG_Config cfg;
    std::string parameters = std::string("selector=PMIS, strength=AHAT, interpolator=") +
                             interpolator_name +
                             ", strength_threshold=0.25, determinism_flag=1, use_opt_kernels=0";
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
    int num_coarse = 0;
    selector->renumberAndCountCoarsePoints(cf_map, num_coarse, A.get_num_rows());
    delete selector;
    UNITTEST_ASSERT_TRUE(num_coarse > 0);

    Interpolator<TConfig> *interpolator = InterpolatorFactory<TConfig>::allocate(cfg, "default");
    UNITTEST_ASSERT_TRUE(interpolator != NULL);
    Matrix<TConfig> P;
    interpolator->generateInterpolationMatrix(A, cf_map, strong_connections, scratch, P);
    delete interpolator;

    Matrix_h host_P = P;
    IVector_h host_cf_map = cf_map;
    this->PrintOnFail("%s generated interpolation matrix with wrong dimensions",
                      interpolator_name);
    UNITTEST_ASSERT_EQUAL(host_P.get_num_rows(), host_matrix.get_num_rows());
    UNITTEST_ASSERT_EQUAL(host_P.get_num_cols(), num_coarse);
    UNITTEST_ASSERT_EQUAL(host_P.row_offsets[host_P.get_num_rows()], host_P.get_num_nz());

    for (int row = 0; row < host_P.get_num_rows(); ++row)
    {
        const int begin = host_P.row_offsets[row];
        const int end = host_P.row_offsets[row + 1];
        this->PrintOnFail("%s generated an empty interpolation row %d",
                          interpolator_name, row);
        UNITTEST_ASSERT_TRUE(end > begin);

        if (host_cf_map[row] >= 0)
        {
            UNITTEST_ASSERT_EQUAL(end - begin, 1);
            UNITTEST_ASSERT_EQUAL(host_P.col_indices[begin], host_cf_map[row]);
            UNITTEST_ASSERT_EQUAL_TOL(host_P.values[begin], HostValueType(1), 1e-7);
        }
        else
        {
            HostValueType row_sum = HostValueType(0);
            for (int jj = begin; jj < end; ++jj)
            {
                const HostValueType value = host_P.values[jj];
                this->PrintOnFail("%s generated invalid weight at row %d", interpolator_name, row);
                UNITTEST_ASSERT_TRUE(std::isfinite(static_cast<double>(value)));
                UNITTEST_ASSERT_TRUE(value >= HostValueType(0));
                UNITTEST_ASSERT_TRUE(host_P.col_indices[jj] >= 0);
                UNITTEST_ASSERT_TRUE(host_P.col_indices[jj] < num_coarse);
                row_sum += value;
            }

            UNITTEST_ASSERT_TRUE(row_sum > HostValueType(0));
            UNITTEST_ASSERT_TRUE(row_sum <= HostValueType(1.00001));
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

    const char *interpolators[] = {"D1", "D2"};
    for (const char *interpolator : interpolators) { check_interpolator(interpolator, A); }

    if (TConfig::memSpace == AMGX_device) { check_interpolator("MULTIPASS", A); }
}

DECLARE_UNITTEST_END(ClassicalInterpolationVariants);

ClassicalInterpolationVariants<TemplateMode<AMGX_mode_hDDI>::Type> ClassicalInterpolationVariants_hDDI;
ClassicalInterpolationVariants<TemplateMode<AMGX_mode_hDFI>::Type> ClassicalInterpolationVariants_hDFI;
ClassicalInterpolationVariants<TemplateMode<AMGX_mode_hFFI>::Type> ClassicalInterpolationVariants_hFFI;
ClassicalInterpolationVariants<TemplateMode<AMGX_mode_dDDI>::Type> ClassicalInterpolationVariants_dDDI;
ClassicalInterpolationVariants<TemplateMode<AMGX_mode_dDFI>::Type> ClassicalInterpolationVariants_dDFI;
ClassicalInterpolationVariants<TemplateMode<AMGX_mode_dFFI>::Type> ClassicalInterpolationVariants_dFFI;

DECLARE_UNITTEST_BEGIN(MultipassLongRangeInterpolation);

typedef Vector<typename TConfig::template setVecPrec<AMGX_vecBool>::Type> BVector;

void run()
{
    AMG_Config cfg;
    UNITTEST_ASSERT_EQUAL(
        cfg.parseParameterString("interpolator=MULTIPASS, use_opt_kernels=0"), AMGX_OK);

    Matrix<TConfig> A;
    const int num_rows = 5;
    const int num_nz = 13;
    const int offsets[] = {0, 2, 5, 8, 11, 13};
    const int columns[] = {0, 1, 0, 1, 2, 1, 2, 3, 2, 3, 4, 3, 4};
    const double values[] = {2, -1, -1, 2, -1, -1, 2, -1, -1, 2, -1, -1, 2};
    A.set_initialized(0);
    A.addProps(CSR);
    A.resize(num_rows, num_rows, num_nz);

    for (int i = 0; i <= num_rows; ++i) { A.row_offsets[i] = offsets[i]; }
    for (int i = 0; i < num_nz; ++i)
    {
        A.col_indices[i] = columns[i];
        A.values[i] = ValueTypeA(values[i]);
    }

    A.computeDiagonal();
    A.set_initialized(1);
    IVector cf_map(num_rows, FINE);
    cf_map[0] = 0;
    cf_map[4] = 1;
    BVector strong_connections(num_nz, false);

    for (int row = 0; row < num_rows; ++row)
    {
        for (int jj = offsets[row]; jj < offsets[row + 1]; ++jj)
        {
            strong_connections[jj] = columns[jj] != row;
        }
    }

    IVector scratch(num_rows, 0);
    Interpolator<TConfig> *interpolator = InterpolatorFactory<TConfig>::allocate(cfg, "default");
    UNITTEST_ASSERT_TRUE(interpolator != NULL);
    Matrix<TConfig> P;
    interpolator->generateInterpolationMatrix(A, cf_map, strong_connections, scratch, P);
    delete interpolator;

    Matrix_h host_P = P;
    UNITTEST_ASSERT_EQUAL(host_P.get_num_rows(), num_rows);
    UNITTEST_ASSERT_EQUAL(host_P.get_num_cols(), 2);

    for (int row = 0; row < num_rows; ++row)
    {
        const int begin = host_P.row_offsets[row];
        const int end = host_P.row_offsets[row + 1];
        this->PrintOnFail("MULTIPASS did not assign long-range row %d", row);
        UNITTEST_ASSERT_TRUE(end > begin);

        for (int jj = begin; jj < end; ++jj)
        {
            UNITTEST_ASSERT_TRUE(host_P.col_indices[jj] == 0 || host_P.col_indices[jj] == 1);
            UNITTEST_ASSERT_TRUE(std::isfinite(static_cast<double>(host_P.values[jj])));
            UNITTEST_ASSERT_TRUE(host_P.values[jj] >= ValueTypeA(0));
        }
    }

    UNITTEST_ASSERT_TRUE(host_P.row_offsets[3] > host_P.row_offsets[2]);
}

DECLARE_UNITTEST_END(MultipassLongRangeInterpolation);

MultipassLongRangeInterpolation<TemplateMode<AMGX_mode_dDDI>::Type> MultipassLongRangeInterpolation_dDDI;
MultipassLongRangeInterpolation<TemplateMode<AMGX_mode_dDFI>::Type> MultipassLongRangeInterpolation_dDFI;
MultipassLongRangeInterpolation<TemplateMode<AMGX_mode_dFFI>::Type> MultipassLongRangeInterpolation_dFFI;

} // namespace amgx
