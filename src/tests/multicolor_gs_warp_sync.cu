// SPDX-FileCopyrightText: 2026 NVIDIA CORPORATION. All Rights Reserved.
//
// SPDX-License-Identifier: BSD-3-Clause

#include "unit_test.h"
#include "solvers/solver.h"
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <vector>

namespace amgx
{

// MULTICOLOR_GS exchanges block entries between the lanes of a block row when it inverts the
// diagonal blocks (block sizes 2 to 5) and when it smooths 4x4 blocks. Run under Compute
// Sanitizer racecheck and synccheck as well as the normal launcher: a missing exchange barrier
// need not change the result.
DECLARE_UNITTEST_BEGIN(MulticolorGSWarpSync);

// D = L U with unit lower L and power-of-two pivots: Gauss-Jordan without pivoting inverts it
// exactly in single precision (at most 18 significant bits in any intermediate for b <= 5).
static double diagonal_entry(int b, int k, int i, int j)
{
    double sum = 0;

    for (int m = 0; m <= std::min(i, j); ++m)
    {
        const double l = m == i ? 1. : ((i + 2 * m + k) % 3 ? ((i + m + k) % 2 ? -0.5 : 0.5) : 0.);
        const double u = m == j ? double(1 << ((m + k) % 3)) : ((m * j + k) % 2 ? -0.25 : 0.25) * (1 + (m + j + k) % 2);
        sum += l * u;
    }

    return sum;
}

static void invert(int b, std::vector<double> &a)
{
    for (int row = 0; row < b; ++row)
    {
        const double diag = 1. / a[row * b + row];
        std::vector<double> next(a);

        for (int i = 0; i < b; ++i)
            for (int j = 0; j < b; ++j)
            {
                next[i * b + j] = i == row ? (j == row ? diag : a[i * b + j] * diag)
                                  : j == row ? -(a[i * b + j] * diag)
                                  : a[i * b + j] - a[i * b + row] * (a[row * b + j] * diag);
            }

        a = next;
    }
}

// Block row i couples to i +- 1, and to i +- 5 when the lower of the two is a multiple of three,
// so the rows sharing a warp have different numbers of nonzero blocks.
static bool coupled(int i, int j)
{
    return std::abs(i - j) == 1 || (std::abs(i - j) == 5 && std::min(i, j) % 3 == 0);
}

void check_case(int b, int n, bool couplings, bool separate_diagonal)
{
    const int bsq = b * b;
    std::vector<std::vector<int> > columns(n);
    int nnz = 0;

    for (int i = 0; i < n; ++i)
    {
        for (int j = 0; j < n; ++j)
            if ((i == j && !separate_diagonal) || (couplings && coupled(i, j)))
            {
                columns[i].push_back(j);
            }

        nnz += int(columns[i].size());
    }

    Matrix_h host;
    host.addProps(CSR);

    if (separate_diagonal) { host.addProps(DIAG); }

    host.resize(n, n, nnz, b, b, 1);
    std::vector<double> dense(size_t(n) * n * bsq, 0.); // block (i, j) at (i * n + j) * bsq
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

            for (int r = 0; r < b; ++r)
                for (int c = 0; c < b; ++c)
                {
                    // Off-diagonal blocks: small, nonsymmetric, distinct per block row.
                    const double v = i == j ? diagonal_entry(b, i % 3, r, c) : 0.125 * ((i + 2 * j + 3 * r + c) % 5 - 2);
                    host.values[slot * bsq + r * b + c] = ValueTypeA(v);
                    dense[(size_t(i) * n + j) * bsq + r * b + c] = v;
                }
        }

        host.row_offsets[i + 1] = position;
    }

    host.computeDiagonal();
    host.set_initialized(1);
    Matrix_d matrix = host;
    Vector_h rhs(n * b), guess(n * b);
    rhs.set_block_dimy(b);
    guess.set_block_dimy(b);

    for (int i = 0; i < n * b; ++i)
    {
        // Without couplings x = D^-1 e_c: each block row returns one column of its inverse.
        rhs[i] = couplings ? ValueTypeB(0.25 * ((3 * i) % 7 - 3)) : ValueTypeB(i % b == (i / b / 3) % b);
        guess[i] = couplings ? ValueTypeB(0.5 * ((5 * i) % 9 - 4)) : ValueTypeB(0);
    }

    AMG_Config cfg;
    cfg.parseParameterString("solver=MULTICOLOR_GS, max_iters=1, monitor_residual=0, relaxation_factor=1.0, symmetric_GS=0, "
                             "coloring_level=1, matrix_coloring_scheme=MIN_MAX, max_uncolored_percentage=0.0, determinism_flag=1");
    Solver<TConfig> *smoother = SolverFactory<TConfig>::allocate(cfg, "default", "solver");
    UNITTEST_ASSERT_TRUE(smoother != NULL);
    smoother->setup(matrix, false);
    Vector_d b_d = rhs, x_d = guess;
    smoother->solve(b_d, x_d, false);
    cudaCheckError();
    Vector_h x = x_d;
    IVector_h colors = matrix.getMatrixColoring().getRowColors();
    const int num_colors = matrix.getMatrixColoring().getNumColors();
    delete smoother;
    // One forward sweep, colour by colour: x_i += D_i^-1 (b_i - sum_j A_ij x_j).
    std::vector<double> expected(guess.size());

    for (int i = 0; i < n * b; ++i) { expected[i] = double(guess[i]); }

    for (int color = 0; color < num_colors; ++color)
        for (int i = 0; i < n; ++i)
        {
            if (colors[i] != color) { continue; }

            std::vector<double> residual(b), dinv(dense.begin() + (size_t(i) * n + i) * bsq, dense.begin() + (size_t(i) * n + i + 1) * bsq);
            invert(b, dinv);

            for (int r = 0; r < b; ++r)
            {
                residual[r] = double(rhs[i * b + r]);

                for (int j = 0; j < n; ++j)
                    for (int c = 0; c < b; ++c)
                    {
                        residual[r] -= dense[(size_t(i) * n + j) * bsq + r * b + c] * expected[j * b + c];
                    }
            }

            for (int r = 0; r < b; ++r)
                for (int c = 0; c < b; ++c)
                {
                    expected[i * b + r] += dinv[r * b + c] * residual[c];
                }
        }

    const double tolerance = couplings ? (sizeof(ValueTypeB) == 4 ? 1e-5 : 1e-12) : 0.;

    for (int i = 0; i < n * b; ++i)
    {
        this->PrintOnFail("block=%d rows=%d couplings=%d separate_diagonal=%d entry=%d: %g, expected %g\n",
                          b, n, couplings, separate_diagonal, i, double(x[i]), expected[i]);
        UNITTEST_ASSERT_TRUE(std::fabs(double(x[i]) - expected[i]) <= tolerance * (1. + std::fabs(expected[i])));
    }
}

void run()
{
    // In mixed precision only the 4x4 path runs: AMGX smooths the other block sizes with
    // cusparse<t>bsrxmv, which it does not call in mixed precision.
    const bool mixed = sizeof(ValueTypeA) != sizeof(ValueTypeB);

    // 2x2 to 5x5 setup kernels hold 8, 3, 2 and 1 block rows per warp; 4x4 also smooths in place.
    for (int b = 2; b <= 5; ++b)
    {
        if (mixed && b != 4) { continue; }

        for (int n : {1, 11, 97})
        {
            // Without couplings the inverse is checked bit for bit; with them, the sweep.
            check_case(b, n, false, false);
            check_case(b, n, true, false);
            check_case(b, n, true, true);
        }
    }
}

DECLARE_UNITTEST_END(MulticolorGSWarpSync);

#define AMGX_CASE_LINE(CASE) MulticolorGSWarpSync<TemplateMode<CASE>::Type> MulticolorGSWarpSync_##CASE;
AMGX_FORALL_BUILDS_DEVICE(AMGX_CASE_LINE)
#undef AMGX_CASE_LINE

} // namespace amgx
