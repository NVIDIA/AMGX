// SPDX-FileCopyrightText: 2026 NVIDIA CORPORATION. All Rights Reserved.
//
// SPDX-License-Identifier: BSD-3-Clause

#include "unit_test.h"
#include "amg_config.h"
#include "aggregation/coarseAgenerators/low_deg_coarse_A_generator.h"
#include <algorithm>
#include <cstdlib>
#include <map>
#include <utility>
#include <vector>

namespace amgx
{

// The low-degree Galerkin product builds each coarse row in a per-warp hash set in shared
// memory: one lane's insertion is another lane's lookup. Run under Compute Sanitizer racecheck
// and synccheck as well as the normal launcher: a missing warp barrier need not change the result.
DECLARE_UNITTEST_BEGIN(LowDegWarpSync);

typedef std::map<std::pair<int, int>, std::vector<double> > Blocks;

static void accumulate(Blocks &blocks, int i, int j, const double *values, int bsq)
{
    std::vector<double> &block = blocks[std::make_pair(i, j)];
    block.resize(bsq, 0.);

    for (int k = 0; k < bsq; ++k) { block[k] += values[k]; }
}

void check_case(int b, int n, bool separate_diagonal, int determinism_flag)
{
    const int bsq = b * b;
    std::vector<std::vector<int> > columns(n);
    int nnz = 0;

    for (int i = 0; i < n; ++i)
    {
        for (int j = 0; j < n; ++j)
        {
            // Rows of different lengths, with couplings far enough to reach other aggregates.
            const int d = std::abs(i - j);

            if ((d == 0 && !separate_diagonal) || d == 1 || (d == 7 && std::min(i, j) % 2 == 0) || (d == 30 && std::min(i, j) % 5 == 0))
            {
                columns[i].push_back(j);
            }
        }

        nnz += int(columns[i].size());
    }

    Matrix_h host;
    host.addProps(CSR);

    if (separate_diagonal) { host.addProps(DIAG); }

    host.resize(n, n, nnz, b, b, 1);
    std::vector<double> block(bsq);
    // Aggregates of four scattered rows (37 is prime to n): coarse rows gather distant fine rows.
    std::vector<int> aggregate(n);
    const int num_aggregates = (n + 3) / 4;

    for (int i = 0; i < n; ++i) { aggregate[i] = (37 * i) % n / 4; }

    Blocks expected;
    int position = 0;
    host.row_offsets[0] = 0;

    for (int i = 0; i < n; ++i)
    {
        for (int slot_col = -1; slot_col < int(columns[i].size()); ++slot_col)
        {
            const bool diagonal = slot_col < 0;

            if (diagonal && !separate_diagonal) { continue; }

            const int j = diagonal ? i : columns[i][slot_col];
            const int slot = diagonal ? nnz + i : position++;

            if (!diagonal) { host.col_indices[slot] = j; }

            for (int k = 0; k < bsq; ++k)
            {
                // Multiples of 1/8: every sum is exact whatever the order of the atomic updates.
                block[k] = 0.125 * ((3 * i + 5 * j + k) % 17 - 8);
                host.values[slot * bsq + k] = ValueTypeA(block[k]);
            }

            accumulate(expected, aggregate[i], aggregate[j], block.data(), bsq);
        }

        host.row_offsets[i + 1] = position;
    }

    host.computeDiagonal();
    host.set_initialized(1);
    // R: the fine rows of each aggregate.
    IVector_h r_rows(num_aggregates + 1, 0), r_cols(n), aggregates_h(n);

    for (int i = 0; i < n; ++i)
    {
        aggregates_h[i] = aggregate[i];
        ++r_rows[aggregate[i] + 1];
    }

    for (int a = 0; a < num_aggregates; ++a) { r_rows[a + 1] += r_rows[a]; }

    std::vector<int> fill(r_rows.begin(), r_rows.end() - 1);

    for (int i = 0; i < n; ++i) { r_cols[fill[aggregate[i]]++] = i; }

    Matrix_d A = host, Ac;
    IVector aggregates = aggregates_h, R_row_offsets = r_rows, R_col_indices = r_cols;
    AMG_Config cfg;
    cfg.parseParameterString(determinism_flag == 2 ? "determinism_flag=2" : "determinism_flag=1");
    aggregation::LowDegCoarseAGenerator<TConfig> generator(cfg, "default");
    generator.computeAOperator(A, Ac, aggregates, R_row_offsets, R_col_indices, num_aggregates);
    cudaCheckError();
    Matrix_h coarse = Ac;
    Blocks actual;

    for (int a = 0; a < num_aggregates; ++a)
    {
        std::vector<double> values(bsq);

        for (int k = coarse.row_offsets[a]; k < coarse.row_offsets[a + 1]; ++k)
        {
            for (int e = 0; e < bsq; ++e) { values[e] = double(coarse.values[k * bsq + e]); }

            accumulate(actual, a, coarse.col_indices[k], values.data(), bsq);
        }

        if (coarse.hasProps(DIAG))
        {
            for (int e = 0; e < bsq; ++e) { values[e] = double(coarse.values[coarse.diag[a] * bsq + e]); }

            accumulate(actual, a, a, values.data(), bsq);
        }
    }

    this->PrintOnFail("block=%d rows=%d separate_diagonal=%d determinism_flag=%d: %zu coarse blocks, expected %zu\n",
                      b, n, separate_diagonal, determinism_flag, actual.size(), expected.size());
    UNITTEST_ASSERT_TRUE(actual == expected);
}

void run()
{
    // fill_A_kernel_1x1, _NxN (2, 3, 5 and, deterministic, 4) and _4x4, one coarse row per warp.
    for (int b = 1; b <= 5; ++b)
        for (int n : {9, 61, 400})
            for (bool separate_diagonal : {false, true})
            {
                check_case(b, n, separate_diagonal, 1);

                if (b == 4) { check_case(b, n, separate_diagonal, 2); }
            }
}

DECLARE_UNITTEST_END(LowDegWarpSync);

#define AMGX_CASE_LINE(CASE) LowDegWarpSync<TemplateMode<CASE>::Type> LowDegWarpSync_##CASE;
AMGX_FORALL_BUILDS_DEVICE(AMGX_CASE_LINE)
#undef AMGX_CASE_LINE

} // namespace amgx
