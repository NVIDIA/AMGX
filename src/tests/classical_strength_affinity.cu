// SPDX-FileCopyrightText: 2011 - 2025 NVIDIA CORPORATION. All Rights Reserved.
//
// SPDX-License-Identifier: BSD-3-Clause

#include "unit_test.h"
#include <matrix_io.h>
#include "test_utils.h"
#include "util.h"
#include "time.h"
#include "classical/strength/affinity.h"

namespace amgx
{

DECLARE_UNITTEST_BEGIN(ClassicalStrengthAffinityTest);

void run()
{
    typedef Vector<typename TConfig::template setVecPrec<AMGX_vecBool>::Type> BVector;
    typedef Vector<typename TConfig::template setVecPrec<AMGX_vecFloat>::Type> FVector;
    AMG_Config cfg;
    cfg.parseParameterString("strength=AFFINITY, determinism_flag=1,strength_threshold=0.55");
    const std::string &cfg_scope = "default";
    Matrix<TConfig> A;
    A.addProps(CSR);
    // generate a strength vector
    Strength<TConfig> *strength;
    strength = StrengthFactory<TConfig>::allocate(cfg, cfg_scope);
    // check the Strength object was created
    this->PrintOnFail("Classical SoC: strength object not created");
    UNITTEST_ASSERT_TRUE(strength != NULL);
    const int num_tests = 10;

    for (int i = 0; i < num_tests; i++)
    {
        // generate a test matrix (must be symmetric for affinity)
        generateMatrixRandomStruct<TConfig>::generate(A, 10000, true, 1, true);
        random_fill(A);
        BVector s_con(A.get_num_nz(), false);
        FVector weights(A.get_num_rows(), 0.0f);
        // get strength & weights
        strength->computeStrongConnectionsAndWeights(A, s_con, weights, -1.);
        // check some connections marked as strong
        bool someStrong = amgx::thrust::reduce(s_con.begin(), s_con.end());
        this->PrintOnFail("Deterministic strength: No strong connections made");
        UNITTEST_ASSERT_TRUE(someStrong == true);
        // use of hash should make result deterministic -- check this
        BVector s_con2(A.get_num_nz(), false);
        FVector weights2(A.get_num_rows(), 0.0f);
        strength->computeStrongConnectionsAndWeights(A, s_con2, weights2, -1.);
        //std::copy(s_con.begin(), s_con.end(), std::ostream_iterator<bool>(std::cout, " "));
        //std::cout << std::endl << std::endl;
        // check
        this->PrintOnFail("Deterministic strength: Different connection strengths");
        UNITTEST_ASSERT_EQUAL(s_con, s_con2);
        this->PrintOnFail("Deterministic strength: Different weights");
        UNITTEST_ASSERT_EQUAL_TOL(weights, weights2, 1e-4);
    }
}

DECLARE_UNITTEST_END(ClassicalStrengthAffinityTest);

//ClassicalStrengthTest <TemplateMode<AMGX_mode_hDDI>::Type>  ClassicalStrengthTest_instance_mode_hDDI;
//ClassicalStrengthTest <TemplateMode<AMGX_mode_hDFI>::Type>  ClassicalStrengthTest_instance_mode_hDFI;
//ClassicalStrengthTest <TemplateMode<AMGX_mode_hFFI>::Type>  ClassicalStrengthTest_instance_mode_hFFI;
ClassicalStrengthAffinityTest <TemplateMode<AMGX_mode_dDDI>::Type>  ClassicalStrengthAffinityTest_instance_mode_dDDI;
ClassicalStrengthAffinityTest <TemplateMode<AMGX_mode_dDFI>::Type>  ClassicalStrengthAffinityTest_instance_mode_dDFI;
ClassicalStrengthAffinityTest <TemplateMode<AMGX_mode_dFFI>::Type>  ClassicalStrengthAffinityTest_instance_mode_dFFI;
} // namespace amgx

