// SPDX-FileCopyrightText: 2011 - 2025 NVIDIA CORPORATION. All Rights Reserved.
//
// SPDX-License-Identifier: BSD-3-Clause

#include "unit_test.h"
#include "version.h"
#include "amgx_c.h"

namespace amgx

{

// parameter is used as test name
DECLARE_UNITTEST_BEGIN(CAPIVersionCheck);

void run()
{
    int major, minor;
    AMGX_get_api_version(&major, &minor);
    UNITTEST_ASSERT_EQUAL(major, __AMGX_API_VERSION_MAJOR);
    UNITTEST_ASSERT_EQUAL(minor, __AMGX_API_VERSION_MINOR);
}

DECLARE_UNITTEST_END(CAPIVersionCheck);


// if you want to be able run this test for all available configs you can write this:
//#define AMGX_CASE_LINE(CASE) SampleTest <TemplateMode<CASE>::Type>  TemplateTest_##CASE;
//  AMGX_FORALL_BUILDS(AMGX_CASE_LINE)
//#undef AMGX_CASE_LINE

// or run for all device configs
//#define AMGX_CASE_LINE(CASE) SampleTest <TemplateMode<CASE>::Type>  TemplateTest_##CASE;
//  AMGX_FORALL_BUILDS_DEVICE(AMGX_CASE_LINE)
//#undef AMGX_CASE_LINE

// or you can specify several desired configs
CAPIVersionCheck<TemplateMode<AMGX_mode_dDDI>::Type>  CAPIVersionCheck_dDDI;


} //namespace amgx
