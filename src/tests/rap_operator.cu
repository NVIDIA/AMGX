// SPDX-FileCopyrightText: 2011 - 2025 NVIDIA CORPORATION. All Rights Reserved.
//
// SPDX-License-Identifier: BSD-3-Clause

#include "unit_test.h"
#include "csr_multiply.h"

#include <algorithm>
#include <cmath>
#include <string>
#include <vector>

namespace amgx
{

DECLARE_UNITTEST_BEGIN(RAPOperator);

template <class MatrixType>
void dense_to_csr(MatrixType &matrix, int rows, int cols, const std::vector<double> &dense)
{
    typedef typename MatrixType::value_type ValueType;
    matrix.addProps(CSR);
    matrix.set_initialized(0);
    matrix.set_num_rows(rows);
    matrix.set_num_cols(cols);
    matrix.set_block_dimx(1);
    matrix.set_block_dimy(1);
    matrix.row_offsets.resize(rows + 1);
    int nnz = 0;

    for (int row = 0; row < rows; ++row)
    {
        matrix.row_offsets[row] = nnz;

        for (int col = 0; col < cols; ++col)
        {
            if (dense[row * cols + col] != 0.0)
            {
                ++nnz;
            }
        }
    }

    matrix.row_offsets[rows] = nnz;
    matrix.col_indices.resize(nnz);
    matrix.values.resize(nnz);
    matrix.set_num_nz(nnz);
    int offset = 0;

    for (int row = 0; row < rows; ++row)
    {
        for (int col = 0; col < cols; ++col)
        {
            const double value = dense[row * cols + col];

            if (value != 0.0)
            {
                matrix.col_indices[offset] = col;
                matrix.values[offset] = ValueType(value);
                ++offset;
            }
        }
    }

    matrix.computeDiagonal();
    matrix.set_initialized(1);
}

void check_rap(bool use_cusparse)
{
    typedef TemplateConfig<AMGX_host, TConfig::vecPrec, TConfig::matPrec, TConfig::indPrec> TConfig_h;
    typedef Matrix<TConfig_h> Matrix_h;
    typedef Matrix<TConfig> Matrix_d;

    // A is a four-point SPD operator. P contains injection and weighted
    // interpolation rows; R=P^T, as in a Galerkin coarse-grid construction.
    const std::vector<double> A_dense = {
         4, -1,  0,  0,
        -1,  4, -1,  0,
         0, -1,  4, -1,
         0,  0, -1,  3
    };
    const std::vector<double> P_dense = {
        1.00, 0.00,
        0.50, 0.50,
        0.00, 1.00,
        0.25, 0.75
    };
    std::vector<double> R_dense(2 * 4);

    for (int row = 0; row < 2; ++row)
    {
        for (int col = 0; col < 4; ++col)
        {
            R_dense[row * 4 + col] = P_dense[col * 2 + row];
        }
    }

    // Independent dense reference for R*A*P.
    std::vector<double> AP_dense(4 * 2, 0.0);
    std::vector<double> expected(2 * 2, 0.0);

    for (int i = 0; i < 4; ++i)
        for (int k = 0; k < 4; ++k)
            for (int j = 0; j < 2; ++j)
                AP_dense[i * 2 + j] += A_dense[i * 4 + k] * P_dense[k * 2 + j];

    for (int i = 0; i < 2; ++i)
        for (int k = 0; k < 4; ++k)
            for (int j = 0; j < 2; ++j)
                expected[i * 2 + j] += R_dense[i * 4 + k] * AP_dense[k * 2 + j];

    Matrix_h A_h, P_h, R_h;
    dense_to_csr(A_h, 4, 4, A_dense);
    dense_to_csr(P_h, 4, 2, P_dense);
    dense_to_csr(R_h, 2, 4, R_dense);
    Matrix_d A = A_h;
    Matrix_d P = P_h;
    Matrix_d R = R_h;
    Matrix_d RAP;

    AMG_Config cfg;
    cfg.parseParameterString(use_cusparse ? "use_cusparse_spgemm=1" : "use_cusparse_spgemm=0");
    void *workspace = CSR_Multiply<TConfig>::csr_workspace_create(cfg, "default");
    CSR_Multiply<TConfig>::csr_galerkin_product(R, A, P, RAP,
                                                 NULL, NULL, NULL, NULL, NULL, NULL,
                                                 workspace);
    CSR_Multiply<TConfig>::csr_workspace_delete(workspace);

    UNITTEST_ASSERT_EQUAL_DESC("RAP row count", RAP.get_num_rows(), 2);
    UNITTEST_ASSERT_EQUAL_DESC("RAP column count", RAP.get_num_cols(), 2);
    Matrix_h result = RAP;
    std::vector<double> actual(4, 0.0);

    for (int row = 0; row < result.get_num_rows(); ++row)
    {
        for (int jj = result.row_offsets[row]; jj < result.row_offsets[row + 1]; ++jj)
        {
            const int col = result.col_indices[jj];
            UNITTEST_ASSERT_TRUE_DESC("RAP produced an out-of-range column", col >= 0 && col < 2);
            actual[row * 2 + col] += static_cast<double>(result.values[jj]);
        }
    }

    const double tolerance = sizeof(ValueTypeA) == sizeof(float) ? 2e-5 : 2e-12;
    const std::string backend = use_cusparse ? "cuSPARSE" : "AMGX";

    for (int i = 0; i < 4; ++i)
    {
        const double error = std::fabs(actual[i] - expected[i]);
        const double scale = std::max(1.0, std::fabs(expected[i]));
        UNITTEST_ASSERT_TRUE_DESC((backend + " RAP differs from the dense Galerkin reference").c_str(),
                                  std::isfinite(actual[i]) && error <= tolerance * scale);
    }

    // R=P^T and A is SPD, so the coarse operator must remain symmetric with
    // positive diagonal entries.
    UNITTEST_ASSERT_TRUE_DESC((backend + " RAP lost symmetry").c_str(),
                              std::fabs(actual[1] - actual[2]) <= tolerance);
    UNITTEST_ASSERT_TRUE_DESC((backend + " RAP has a non-positive diagonal").c_str(),
                              actual[0] > 0.0 && actual[3] > 0.0);
}

void run()
{
    check_rap(false);
    check_rap(true);
}

DECLARE_UNITTEST_END(RAPOperator);

RAPOperator<TemplateMode<AMGX_mode_dDDI>::Type> RAPOperator_instance_mode_dDDI;
RAPOperator<TemplateMode<AMGX_mode_dFFI>::Type> RAPOperator_instance_mode_dFFI;

} // namespace amgx
