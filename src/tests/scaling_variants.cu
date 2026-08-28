// SPDX-FileCopyrightText: 2011 - 2025 NVIDIA CORPORATION. All Rights Reserved.
//
// SPDX-License-Identifier: BSD-3-Clause

#include "unit_test.h"
#include "scalers/scaler.h"
#include <matrix_io.h>

#include <algorithm>
#include <cmath>
#include <string>

namespace amgx
{

DECLARE_UNITTEST_BEGIN(ScalingVariants);

void check_scaler(const std::string &scaler_name)
{
    typedef TemplateConfig<AMGX_host, TConfig::vecPrec, TConfig::matPrec, TConfig::indPrec> TConfig_h;
    typedef Matrix<TConfig_h> Matrix_h;
    typedef Vector<TConfig_h> Vector_h;
    typedef Matrix<TConfig> Matrix_d;
    typedef Vector<TConfig> Vector_d;

    Matrix_h original;
    original.addProps(CSR);
    original.set_initialized(0);
    MatrixCusp<TConfig_h, cusp::csr_format> cusp_A(&original);
    cusp::gallery::poisson5pt(cusp_A, 9, 9);

    // Preserve symmetry and positive definiteness while making the diagonal
    // nonuniform enough that every scaler has observable work to do.
    for (int row = 0; row < original.get_num_rows(); ++row)
    {
        const ValueTypeA row_scale = ValueTypeA(1) + ValueTypeA(row % 5) / ValueTypeA(4);

        for (int jj = original.row_offsets[row]; jj < original.row_offsets[row + 1]; ++jj)
        {
            const int col = original.col_indices[jj];
            const ValueTypeA col_scale = ValueTypeA(1) + ValueTypeA(col % 5) / ValueTypeA(4);
            original.values[jj] *= row_scale * col_scale;
        }
    }

    original.computeDiagonal();
    original.set_initialized(1);
    Matrix_d A = original;

    const int rows = original.get_num_rows();
    Vector_h vector_original(rows);
    vector_original.set_block_dimx(1);
    vector_original.set_block_dimy(1);

    for (int i = 0; i < rows; ++i)
    {
        vector_original[i] = ValueTypeB(1) + ValueTypeB(i % 7) / ValueTypeB(3);
    }

    AMG_Config cfg;
    cfg.parseParameterString(("scaling=" + scaler_name).c_str());
    Scaler<TConfig> *scaler = ScalerFactory<TConfig>::allocate(cfg, "default");
    UNITTEST_ASSERT_TRUE_DESC((scaler_name + " factory allocation failed").c_str(), scaler != NULL);

    scaler->setup(A);
    scaler->scaleMatrix(A, SCALE);
    Matrix_h scaled = A;

    bool matrix_changed = false;

    for (int jj = 0; jj < original.get_num_nz(); ++jj)
    {
        const double value = static_cast<double>(scaled.values[jj]);
        UNITTEST_ASSERT_TRUE_DESC((scaler_name + " produced a non-finite matrix value").c_str(),
                                  std::isfinite(value));
        matrix_changed = matrix_changed || scaled.values[jj] != original.values[jj];
    }

    UNITTEST_ASSERT_TRUE_DESC((scaler_name + " left the nonuniform matrix unchanged").c_str(), matrix_changed);

    Vector_d left = vector_original;
    Vector_d right = vector_original;
    scaler->scaleVector(left, SCALE, LEFT);
    scaler->scaleVector(left, UNSCALE, LEFT);
    scaler->scaleVector(right, SCALE, RIGHT);
    scaler->scaleVector(right, UNSCALE, RIGHT);

    scaler->scaleMatrix(A, UNSCALE);
    Matrix_h restored = A;
    Vector_h left_restored = left;
    Vector_h right_restored = right;
    const double tolerance = sizeof(ValueTypeA) == sizeof(float) ? 2e-4 : 2e-10;

    for (int jj = 0; jj < original.get_num_nz(); ++jj)
    {
        const double expected = static_cast<double>(original.values[jj]);
        const double error = std::fabs(static_cast<double>(restored.values[jj]) - expected);
        const double scale = std::max(1.0, std::fabs(expected));
        UNITTEST_ASSERT_TRUE_DESC((scaler_name + " matrix scale/unscale was not reversible").c_str(),
                                  error <= tolerance * scale);
    }

    for (int i = 0; i < rows; ++i)
    {
        const double expected = static_cast<double>(vector_original[i]);
        const double left_error = std::fabs(static_cast<double>(left_restored[i]) - expected);
        const double right_error = std::fabs(static_cast<double>(right_restored[i]) - expected);
        UNITTEST_ASSERT_TRUE_DESC((scaler_name + " left vector scale/unscale was not reversible").c_str(),
                                  left_error <= tolerance * std::max(1.0, std::fabs(expected)));
        UNITTEST_ASSERT_TRUE_DESC((scaler_name + " right vector scale/unscale was not reversible").c_str(),
                                  right_error <= tolerance * std::max(1.0, std::fabs(expected)));
    }

    delete scaler;
}

void run()
{
    check_scaler("DIAGONAL_SYMMETRIC");
    check_scaler("BINORMALIZATION");
    check_scaler("NBINORMALIZATION");
}

DECLARE_UNITTEST_END(ScalingVariants);

ScalingVariants<TemplateMode<AMGX_mode_dDDI>::Type> ScalingVariants_instance_mode_dDDI;
ScalingVariants<TemplateMode<AMGX_mode_dFFI>::Type> ScalingVariants_instance_mode_dFFI;

} // namespace amgx
