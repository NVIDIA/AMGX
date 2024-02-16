// SPDX-FileCopyrightText: 2011 - 2024 NVIDIA CORPORATION. All Rights Reserved.
//
// SPDX-License-Identifier: BSD-3-Clause

#include "unit_test.h"
#include "amg_config.h"
#include "error.h"
#include <fstream>
#include <iostream>


namespace amgx

{

// parameter is used as test name
DECLARE_UNITTEST_BEGIN(ConfigStringParsing);

void run()
{
    AMG_Config cfg;
    UNITTEST_ASSERT_TRUE( AMGX_OK == cfg.parseParameterString("") ); // empty string
    UNITTEST_ASSERT_TRUE( AMGX_OK == cfg.parseParameterString("max_levels=10") );
    UNITTEST_ASSERT_TRUE( AMGX_OK == cfg.parseParameterString("    max_levels = 10,min_coarse_rows = 10 ; \n max_iters \t= 10\n;") );
    UNITTEST_ASSERT_TRUE( AMGX_ERR_CONFIGURATION == cfg.parseParameterString("    max_levels = 10 \n max_iters \t= 10\n") ); // new line is not delimiter
    UNITTEST_ASSERT_TRUE( AMGX_ERR_CONFIGURATION == cfg.parseParameterString("    max_levels =  ,min_coarse_rows = 10") ); // value not specified
    UNITTEST_ASSERT_TRUE( AMGX_ERR_CONFIGURATION == cfg.parseParameterString("    max_levels = 10 min_coarse_rows = 10") ); // no delimiter
    UNITTEST_ASSERT_TRUE( AMGX_ERR_CONFIGURATION == cfg.parseParameterString("    definitely_nonexisting_parameter = 10, min_coarse_rows = 10") ); // bad parameter
    UNITTEST_ASSERT_TRUE( AMGX_ERR_CONFIGURATION == cfg.parseParameterString(" config_version=2,    solver(fgmres = 10, min_coarse_rows = 10") ); // unbalanced character
    UNITTEST_ASSERT_TRUE( AMGX_ERR_CONFIGURATION == cfg.parseParameterString("  config_version=2,  solver(fgmres) = 10, fgmres:preconditioner(jacobi=BLOCK_JACOBI, min_coarse_rows = 10") ); // unbalanced parentheses
    UNITTEST_ASSERT_TRUE( AMGX_OK == cfg.parseParameterString("    config_version=2, undefined_scope:max_iters = 10, min_coarse_rows = 10") ); // undefined scope
    UNITTEST_ASSERT_TRUE( AMGX_OK == cfg.parseParameterString("    max_iters = 10, , min_coarse_rows = 10") ); // empty parameter
    UNITTEST_ASSERT_TRUE( AMGX_OK == cfg.parseParameterString("    max_iters = 10,           , min_coarse_rows = 10") ); // empty parameter
    UNITTEST_ASSERT_TRUE( AMGX_ERR_CONFIGURATION == cfg.parseParameterString("    config_version=2, solver(scope)=FGMRES, preconditioner(scope)=BLOCK_JACOBI, max_iters = 10, min_coarse_rows = 10") ); // two solvers with same scope
    UNITTEST_ASSERT_TRUE( AMGX_ERR_CONFIGURATION == cfg.parseParameterString("    config_version=2, solver(scope2)=FGMRES, scope2::max_iters=10") ); // double colon
    UNITTEST_ASSERT_TRUE( AMGX_ERR_CONFIGURATION == cfg.parseParameterString("    max_iters&=15") ); // invalid symbol
    UNITTEST_ASSERT_TRUE( AMGX_ERR_CONFIGURATION == cfg.parseParameterString("    max_iters==15") ); // double equal
    UNITTEST_ASSERT_TRUE( AMGX_ERR_CONFIGURATION == cfg.parseParameterString("    max_iters(scope3)=15") ); // assigning new scope to non-solver parameter
    UNITTEST_ASSERT_TRUE( AMGX_ERR_CONFIGURATION == cfg.parseParameterString("    max_iters(scope)=15") ); // assigning existing scope to non-solver parameter
    // Using config_version=1 with scopes
    UNITTEST_ASSERT_TRUE( AMGX_ERR_CONFIGURATION == cfg.parseParameterString("config_version=1, solver(scope)=FGMRES, scope:max_iters=1") ); // assigning existing scope to non-solver parameter
//TODO: Check that new scope is only associated with solver
    const char *file_name = "temp.dat";
    std::ofstream fout;
    fout.open(file_name, std::ios::out);
    fout << "     #SOME VERY LONG COMMENTS WITH ILLEGAL CHARACTER &^!@$!@)^*$::( \n  solver=FGMRES" << std::endl; // comments
    UNITTEST_ASSERT_TRUE( AMGX_OK == cfg.parseFile(file_name));
    fout.close();
    fout.open(file_name, std::ios::out);
    fout << "    \t  solver=FGMRES \n max_iters=10" << std::endl; // tab
    UNITTEST_ASSERT_TRUE( AMGX_OK == cfg.parseFile(file_name));
    fout.close();
    fout.open(file_name, std::ios::out);
    fout << "     #           \n  solver=FGMRES" << std::endl; // empty comment line
    UNITTEST_ASSERT_TRUE( AMGX_OK == cfg.parseFile(file_name));
    fout.close();
    fout.open(file_name, std::ios::out);
    fout << "    \n \n  solver=FGMRES \n max_iters=10 \n \n max_levels=10" << std::endl; // blank lines
    UNITTEST_ASSERT_TRUE( AMGX_OK == cfg.parseFile(file_name));
    fout.close();
    // empty parameter file
    fout.open(file_name, std::ios::out);
    fout << "" << std::endl; // blank lines
    UNITTEST_ASSERT_TRUE( AMGX_OK == cfg.parseFile(file_name));
    fout.close();
    // Parsing file and string
    //  OK
    fout.open(file_name, std::ios::out);
    fout << "  config_version=2,  \n \n  solver(solver_name)=FGMRES \n solver_name:max_iters=10 \n \n max_levels=10" << std::endl; // blank lines
    fout.close();
    UNITTEST_ASSERT_TRUE( AMGX_OK == cfg.parseParameterStringAndFile("config_version=2, solver_name:preconditioner=BLOCK_JACOBI", file_name));
    // Different config_versions
    fout.open(file_name, std::ios::out);
    fout << "  config_version=2,  \n \n  solver(solver_name)=FGMRES \n solver_name:max_iters=10 \n \n max_levels=10" << std::endl; // blank lines
    UNITTEST_ASSERT_TRUE( AMGX_ERR_CONFIGURATION == cfg.parseParameterStringAndFile("smoother=BLOCK_JACOBI", file_name));
    fout.close();
}

DECLARE_UNITTEST_END(ConfigStringParsing);

ConfigStringParsing <TemplateMode<AMGX_mode_hDDI>::Type>  ConfigStringParsing_hDDI;


} //namespace amgx
