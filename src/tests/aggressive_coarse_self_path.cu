// SPDX-FileCopyrightText: 2011 - 2026 NVIDIA CORPORATION. All Rights Reserved.
//
// SPDX-License-Identifier: BSD-3-Clause

#include <classical/selectors/pmis.h>
#include <types.h>
#include <unit_test.h>

namespace amgx
{

DECLARE_UNITTEST_BEGIN(AggressiveCoarseSelfPath);

void run()
{
    constexpr int rows = 64;
    AMG_Config cfg;
    cfg.parseParameterString("config_version=2, use_opt_kernels=0");
    classical::PMIS_Selector<TConfig> selector(cfg, "default");

    // The average row length switches the legacy kernel from 8 to 32 threads at 16.
    for (int degree : {15, 16})
    {
        Matrix_h host;
        host.addProps(CSR);
        host.resize(rows, rows, rows * degree);
        IVector_h coarse(rows);
        Vector<typename TConfig_h::template setVecPrec<AMGX_vecBool>::Type> strong(rows * degree);
        int entry = 0;
        for (int i = 0; i < rows; ++i)
        {
            coarse[i] = i % 2 == 0 ? i / 2 : FINE;
            host.row_offsets[i] = entry;
            for (int j = 0; j < rows; ++j)
            {
                const int offset = (j + rows - i) % rows;
                if (offset > 7 && offset < rows - 7 && !(degree == 16 && offset == rows / 2))
                {
                    continue;
                }
                host.col_indices[entry] = j;
                strong[entry] = j == (i ^ 1);
                host.values[entry++] = i == j ? 2.0 : (j == (i ^ 1) ? -1.0 : -0.001);
            }
        }
        host.row_offsets[rows] = entry;
        host.set_initialized(1);

        Matrix_d A(host), S2;
        typename classical::Selector<TConfig>::BVector device_strong(strong);
        IVector cf_map(coarse);
        selector.createS2(A, S2, device_strong, cf_map);
        UNITTEST_ASSERT_EQUAL(S2.get_num_rows(), rows / 2);
        // The only strong two-hop path from each coarse point leads back to itself.
        UNITTEST_ASSERT_EQUAL(S2.get_num_nz(), 0);
    }
}

DECLARE_UNITTEST_END(AggressiveCoarseSelfPath);

AggressiveCoarseSelfPath<TemplateMode<AMGX_mode_dDDI>::Type> aggressive_coarse_self_path;

} // namespace amgx
