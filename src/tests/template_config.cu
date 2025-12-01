// SPDX-FileCopyrightText: 2011 - 2025 NVIDIA CORPORATION. All Rights Reserved.
//
// SPDX-License-Identifier: BSD-3-Clause

#include "unit_test.h"

namespace amgx
{
// parameter is used as test name
DECLARE_UNITTEST_BEGIN(TemplateConfigTest);
void run()
{
    {typedef typename TConfig::template setMemSpace<AMGX_host>::Type TConf; UNITTEST_ASSERT_TRUE(TConf::memSpace == AMGX_host);}
    {typedef typename TConfig::template setMemSpace<AMGX_device>::Type TConf; UNITTEST_ASSERT_TRUE(TConf::memSpace == AMGX_device);}
    {typedef typename TConfig::template setVecPrec<AMGX_vecInt>::Type TConf; UNITTEST_ASSERT_TRUE(TConf::vecPrec == AMGX_vecInt);}
    {typedef typename TConfig::template setVecPrec<AMGX_vecUSInt>::Type TConf; UNITTEST_ASSERT_TRUE(TConf::vecPrec == AMGX_vecUSInt);}
    {typedef typename TConfig::template setVecPrec<AMGX_vecUInt>::Type TConf; UNITTEST_ASSERT_TRUE(TConf::vecPrec == AMGX_vecUInt);}
    {typedef typename TConfig::template setVecPrec<AMGX_vecUInt64>::Type TConf; UNITTEST_ASSERT_TRUE(TConf::vecPrec == AMGX_vecUInt64);}
    {typedef typename TConfig::template setVecPrec<AMGX_vecBool>::Type TConf; UNITTEST_ASSERT_TRUE(TConf::vecPrec == AMGX_vecBool);}
    {typedef typename TConfig::template setVecPrec<AMGX_vecDouble>::Type TConf; UNITTEST_ASSERT_TRUE(TConf::vecPrec == AMGX_vecDouble);}
    {typedef typename TConfig::template setVecPrec<AMGX_vecFloat>::Type TConf; UNITTEST_ASSERT_TRUE(TConf::vecPrec == AMGX_vecFloat);}
    {typedef typename TConfig::template setMatPrec<AMGX_matFloat>::Type TConf; UNITTEST_ASSERT_TRUE(TConf::matPrec == AMGX_matFloat);}
    {typedef typename TConfig::template setMatPrec<AMGX_matDouble>::Type TConf; UNITTEST_ASSERT_TRUE(TConf::matPrec == AMGX_matDouble);}
    {typedef typename TemplateMode<AMGX_SET_MODE_VAL(AMGX_VecPrecision, TConfig::mode, AMGX_GET_MODE_VAL(AMGX_MatPrecision, TConfig::mode))>::Type TConf; UNITTEST_ASSERT_TRUE((int)TConf::vecPrec == (int)TConfig::matPrec);}
}
DECLARE_UNITTEST_END(TemplateConfigTest);

#define AMGX_CASE_LINE(CASE) TemplateConfigTest <TemplateMode<CASE>::Type>  TemplateConfigTest_##CASE;
AMGX_FORALL_BUILDS(AMGX_CASE_LINE)
#undef AMGX_CASE_LINE

} //namespace amgx
