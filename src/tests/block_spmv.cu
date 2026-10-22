// SPDX-FileCopyrightText: 2026 NVIDIA CORPORATION. All Rights Reserved.
//
// SPDX-License-Identifier: BSD-3-Clause

#include "unit_test.h"
#include "matrix.h"
#include "multiply.h"
#include <algorithm>
#include <limits>
#include <vector>

namespace amgx
{

static void set_dyadic(float &v, double re, double) { v = static_cast<float>(re); }
static void set_dyadic(double &v, double re, double) { v = re; }
static void set_dyadic(cuComplex &v, double re, double im) { v = make_cuComplex(static_cast<float>(re), static_cast<float>(im)); }
static void set_dyadic(cuDoubleComplex &v, double re, double im) { v = make_cuDoubleComplex(re, im); }

// The 3x3 and 4x4 block CSR products of multiply(), with and without a separate diagonal, in
// both block layouts, against an exact CPU product. Neighboring block rows have different
// lengths, some none; the sizes leave partial warps and CTAs, and the largest needs more than one
// grid, so the kernels' grid-stride loops run. Dyadic values keep every sum exact in any order
// and precision, and the output starts as NaN, so a missed write fails too. Run it under Compute
// Sanitizer racecheck and synccheck as well: an unordered exchange between lanes can race
// without changing a result.
DECLARE_UNITTEST_BEGIN(BlockSpmv);

void check(int bsize, int n, bool separate_diagonal, bool row_major)
{
    // Row r has 1 + r % 6 blocks (columns r, r + 3, ...), the first of them its diagonal; without
    // a separate diagonal every seventh row is empty instead.
    std::vector<std::vector<int> > columns(n);
    int nnz = 0;

    for (int row = 0; row < n; ++row)
    {
        if (!separate_diagonal && row % 7 == 0) { continue; }

        for (int d = 0; d <= row % 6; ++d)
        {
            columns[row].push_back((row + 3 * d) % n);
        }

        std::sort(columns[row].begin(), columns[row].end());
        columns[row].erase(std::unique(columns[row].begin(), columns[row].end()), columns[row].end());
        nnz += static_cast<int>(columns[row].size()) - (separate_diagonal ? 1 : 0);
    }

    const int bsize_sq = bsize * bsize;
    Matrix_h host;
    host.addProps(CSR);

    if (separate_diagonal) { host.addProps(DIAG); }

    host.resize(n, n, nnz, bsize, bsize, 1);
    BlockFormat format = row_major ? ROW_MAJOR : COL_MAJOR;
    host.setBlockFormat(format);
    Vector_h input(bsize * n), expected(bsize * n, types::util<ValueTypeB>::get_zero()), poisoned(bsize * n);
    input.set_block_dimy(bsize);
    expected.set_block_dimy(bsize);
    poisoned.set_block_dimy(bsize);
    const double nan = std::numeric_limits<double>::quiet_NaN();

    for (int i = 0; i < bsize * n; ++i)
    {
        set_dyadic(input[i], ((i % 11) - 5) * 0.25, ((i % 3) - 1) * 0.5);
        set_dyadic(poisoned[i], nan, nan);
    }

    int position = 0;
    host.row_offsets[0] = 0;

    for (int row = 0; row < n; ++row)
    {
        for (int col : columns[row])
        {
            const bool diagonal = separate_diagonal && col == row;
            const int slot = diagonal ? nnz + row : position++;

            if (!diagonal) { host.col_indices[slot] = col; }

            for (int i = 0; i < bsize; ++i)
            {
                for (int j = 0; j < bsize; ++j)
                {
                    // Nonsymmetric blocks tell the two layouts apart.
                    ValueTypeA value;
                    set_dyadic(value, ((row + col) % 7 + 1 + 3 * i + j) * 0.125, ((row + 2 * col) % 5 - 2 + i - j) * 0.25);
                    host.values[bsize_sq * slot + (row_major ? bsize * i + j : i + bsize * j)] = value;
                    ValueTypeB widened;
                    types::util<ValueTypeA>::to_uptype(value, widened);
                    expected[bsize * row + i] = expected[bsize * row + i] + widened * input[bsize * col + j];
                }
            }
        }

        host.row_offsets[row + 1] = position;
    }

    host.computeDiagonal();
    host.set_initialized(1);
    Matrix_d matrix = host;
    // Matrix's cross-memory copy does not copy the block format.
    matrix.setBlockFormat(format);
    Vector_d x = input, y = poisoned;
    multiply(matrix, x, y);
    cudaCheckError();
    Vector_h result = y;
    int mismatch = 0;

    while (mismatch < bsize * n && types::util<ValueTypeB>::is_equal(result[mismatch], expected[mismatch]))
    {
        ++mismatch;
    }

    this->PrintOnFail("%dx%d blocks, %d block rows, separate diagonal %d, row major %d: first wrong entry %d\n", bsize, bsize, n, separate_diagonal, row_major, mismatch);
    UNITTEST_ASSERT_TRUE(mismatch == bsize * n);
}

void run()
{
    // Block rows per warp: 10 (3x3), 8 (4x4 without a separate diagonal), 2 (4x4 with one); per
    // CTA: 60, 32, 8. One grid covers at most 4096 CTAs, so the largest size runs a second pass of
    // the grid-stride loop in every kernel.
    for (int bsize = 3; bsize <= 4; ++bsize)
    {
        for (int n : {1, 9, 33, 61, 4096 * 60 + 61})
        {
            for (bool separate_diagonal : {false, true})
            {
                for (bool row_major : {false, true})
                {
                    check(bsize, n, separate_diagonal, row_major);
                }
            }
        }
    }
}

DECLARE_UNITTEST_END(BlockSpmv);

#define AMGX_CASE_LINE(CASE) BlockSpmv<TemplateMode<CASE>::Type> BlockSpmv_##CASE;
AMGX_FORALL_BUILDS_DEVICE(AMGX_CASE_LINE)
AMGX_FORCOMPLEX_BUILDS_DEVICE(AMGX_CASE_LINE)
#undef AMGX_CASE_LINE

} // namespace amgx
