/* Copyright (c) 2011-2017, NVIDIA CORPORATION. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *  * Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *  * Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *  * Neither the name of NVIDIA CORPORATION nor the names of its
 *    contributors may be used to endorse or promote products derived
 *    from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
 * EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 * PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
 * CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 * EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 * PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
 * OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#include <amgx_c.h>
#include "unit_test.h"
#include <core.h>
#include <error.h>
#include "test_utils.h"
//#include "util.h"
//#include "time.h"

#include <matrix_io.h>
#include <readers.h>

#include <matrix_coloring/min_max.h>
#include <matrix_coloring/round_robin.h>
#include <matrix_coloring/multi_hash.h>

#include <cycles/v_cycle.h>
#include <cycles/w_cycle.h>
#include <cycles/f_cycle.h>
#include <cycles/cg_cycle.h>
#include <cycles/cg_flex_cycle.h>

#include <solvers/algebraic_multigrid_solver.h>
#include <solvers/pcgf_solver.h>
#include <solvers/cg_solver.h>
#include <solvers/pcg_solver.h>
#include <solvers/pbicgstab_solver.h>
#include <solvers/bicgstab_solver.h>
#include <solvers/fgmres_solver.h>
#include <solvers/gauss_seidel_solver.h>
//#include <solvers/jacobi_solver.h>
#include <solvers/jacobi_l1_solver.h>
//#include <solvers/jacobi_nocusp_solver.h>
#include <solvers/kpz_polynomial_solver.h>
#include <solvers/polynomial_solver.h>
#include <solvers/block_jacobi_solver.h>
#include <solvers/multicolor_gauss_seidel_solver.h>
#include <solvers/multicolor_dilu_solver.h>

#include <convergence/absolute.h>
#include <convergence/relative_max.h>
#include <convergence/relative_ini.h>

#include <amg_level.h>
#include <classical/classical_amg_level.h>
#include <aggregation/aggregation_amg_level.h>

#include <classical/interpolators/distance1.h>
#include <classical/interpolators/distance2.h>

#include <classical/selectors/pmis.h>
#include <classical/selectors/cr.h>

#include <energymin/energymin_amg_level.h>
#include <energymin/interpolators/em.h>

#include <classical/strength/ahat.h>
#include <classical/strength/all.h>

#include <aggregation/selectors/dummy.h>
#include <aggregation/selectors/size2_selector.h>
#include <aggregation/selectors/size4_selector.h>
#include <aggregation/selectors/size8_selector.h>

#include <aggregation/coarseAgenerators/hybrid_coarse_A_generator.h>
#include <aggregation/coarseAgenerators/low_deg_coarse_A_generator.h>
#include <aggregation/coarseAgenerators/thrust_coarse_A_generator.h>

namespace amgx
{
// parameter is used as test name
DECLARE_UNITTEST_BEGIN(FactoriesTest);

/*inline void registerParameters() {
  AMG_Config::registerParameter<int>("dummyPar1","dummy parameter of type int",100);
  AMG_Config::registerParameter<double>("dummyPar2","dummy parameter of type double",1e-6);
  AMG_Config::registerParameter<NormType>("dummyPar3","dummy parameter of type NormType",L2);
  AMG_Config::registerParameter<std::string>("dummyPar4","dummy parameter of type std::string","DUMMY");
  AMG_Config::registerParameter<AlgorithmType>("dummyPar5","dummy parameter of type AlgorithmType",CLASSICAL);
}*/

inline void registerClasses()
{
    // Register AMG Level Factories
    AMG_LevelFactory<TConfig>::registerFactory(CLASSICAL, new Classical_AMG_LevelFactory<TConfig>);
    AMG_LevelFactory<TConfig>::registerFactory(AGGREGATION, new Aggregation_AMG_LevelFactory<TConfig>);
    AMG_LevelFactory<TConfig>::registerFactory(ENERGYMIN, new Energymin_AMG_LevelFactory<TConfig>);
    // Register MatrixColoring schemes
    MatrixColoringFactory<TConfig>::registerFactory("MIN_MAX", new MinMaxMatrixColoringFactory<TConfig>);
    MatrixColoringFactory<TConfig>::registerFactory("ROUND_ROBIN", new RoundRobinMatrixColoringFactory<TConfig>);
    MatrixColoringFactory<TConfig>::registerFactory("MULTI_HASH", new MultiHashMatrixColoringFactory<TConfig>);
    //Register Cycles
    CycleFactory<TConfig>::registerFactory("V", new V_CycleFactory<TConfig>);
    CycleFactory<TConfig>::registerFactory("F", new F_CycleFactory<TConfig>);
    CycleFactory<TConfig>::registerFactory("W", new W_CycleFactory<TConfig>);
    CycleFactory<TConfig>::registerFactory("CG", new CG_CycleFactory<TConfig>);
    CycleFactory<TConfig>::registerFactory("CGF", new CG_Flex_CycleFactory<TConfig>);
    //Register Solvers
    SolverFactory<TConfig>::registerFactory("AMG", new AlgebraicMultigrid_SolverFactory<TConfig>);
    SolverFactory<TConfig>::registerFactory("PCGF", new PCGF_SolverFactory<TConfig>);
    SolverFactory<TConfig>::registerFactory("CG", new CG_SolverFactory<TConfig>);
    SolverFactory<TConfig>::registerFactory("PCG", new PCG_SolverFactory<TConfig>);
    SolverFactory<TConfig>::registerFactory("PBICGSTAB", new PBiCGStab_SolverFactory<TConfig>);
    SolverFactory<TConfig>::registerFactory("BICGSTAB", new BiCGStab_SolverFactory<TConfig>);
    SolverFactory<TConfig>::registerFactory("FGMRES", new FGMRES_SolverFactory<TConfig>);
    SolverFactory<TConfig>::registerFactory("JACOBI_L1", new JacobiL1SolverFactory<TConfig>);
    SolverFactory<TConfig>::registerFactory("GS", new GaussSeidelSolverFactory<TConfig>);
    SolverFactory<TConfig>::registerFactory("POLYNOMIAL", new polynomial_solver::PolynomialSolverFactory<TConfig>);
    SolverFactory<TConfig>::registerFactory("KPZ_POLYNOMIAL", new KPZPolynomialSolverFactory<TConfig>);
    SolverFactory<TConfig>::registerFactory("BLOCK_JACOBI", new block_jacobi_solver::BlockJacobiSolverFactory<TConfig>);
    SolverFactory<TConfig>::registerFactory("MULTICOLOR_GS", new multicolor_gauss_seidel_solver::MulticolorGaussSeidelSolverFactory<TConfig>);
    SolverFactory<TConfig>::registerFactory("MULTICOLOR_DILU", new multicolor_dilu_solver::MulticolorDILUSolverFactory<TConfig>);
    //Register Aggregation Selectors
    aggregation::SelectorFactory<TConfig>::registerFactory("SIZE_2", new aggregation::size2_selector::Size2SelectorFactory<TConfig>);
    aggregation::SelectorFactory<TConfig>::registerFactory("SIZE_4", new aggregation::size4_selector::Size4SelectorFactory<TConfig>);
    aggregation::SelectorFactory<TConfig>::registerFactory("SIZE_8", new aggregation::size8_selector::Size8SelectorFactory<TConfig>);
    aggregation::SelectorFactory<TConfig>::registerFactory("DUMMY", new aggregation::DUMMY_SelectorFactory<TConfig>);
    //Register Energymin Selectors
    classical::SelectorFactory<TConfig>::registerFactory("CR", new classical::CR_SelectorFactory<TConfig>);
    //Register Aggregation Coarse Generators
    aggregation::CoarseAGeneratorFactory<TConfig>::registerFactory("LOW_DEG", new aggregation::LowDegCoarseAGeneratorFactory<TConfig>);
    aggregation::CoarseAGeneratorFactory<TConfig>::registerFactory("HYBRID", new aggregation::HybridCoarseAGeneratorFactory<TConfig>);
    aggregation::CoarseAGeneratorFactory<TConfig>::registerFactory("THRUST", new aggregation::ThrustCoarseAGeneratorFactory<TConfig>);
    //Register Energymin Interpolators
    energymin::InterpolatorFactory<TConfig>::registerFactory("EM", new energymin::EM_InterpolatorFactory<TConfig>);
    ConvergenceFactory<TConfig>::registerFactory("ABSOLUTE",     new AbsoluteConvergenceFactory<TConfig>);
    ConvergenceFactory<TConfig>::registerFactory("RELATIVE_INI_CORE",     new RelativeIniConvergenceFactory<TConfig>);
}

inline void registerClassesClassical()
{
    //Register Classical Interpolators
    InterpolatorFactory<TConfig>::registerFactory("D1", new Distance1_InterpolatorFactory<TConfig>);
    InterpolatorFactory<TConfig>::registerFactory("D2", new Distance2_InterpolatorFactory<TConfig>);
    //Register Classical Selectors
    classical::SelectorFactory<TConfig>::registerFactory("PMIS", new classical::PMIS_SelectorFactory<TConfig>);
    //Register Classical Strength
    StrengthFactory<TConfig>::registerFactory("AHAT", new Strength_Ahat_StrengthFactory<TConfig>);
    StrengthFactory<TConfig>::registerFactory("ALL", new Strength_All_StrengthFactory<TConfig>);
}

inline void unregisterClasses()
{
    AMG_LevelFactory<TConfig>::unregisterFactory(CLASSICAL);
    AMG_LevelFactory<TConfig>::unregisterFactory(AGGREGATION);
    AMG_LevelFactory<TConfig>::unregisterFactory(ENERGYMIN);
// Unegister MatrixColoring schemes
    MatrixColoringFactory<TConfig>::unregisterFactory("MIN_MAX");
    MatrixColoringFactory<TConfig>::unregisterFactory("ROUND_ROBIN");
    MatrixColoringFactory<TConfig>::unregisterFactory("MULTI_HASH");
    //Unegister Cycles
    CycleFactory<TConfig>::unregisterFactory("V");
    CycleFactory<TConfig>::unregisterFactory("F");
    CycleFactory<TConfig>::unregisterFactory("W");
    CycleFactory<TConfig>::unregisterFactory("CG");
    CycleFactory<TConfig>::unregisterFactory("CGF");
    //Unegister Solvers
    SolverFactory<TConfig>::unregisterFactory("PCGF");
    SolverFactory<TConfig>::unregisterFactory("CG");
    SolverFactory<TConfig>::unregisterFactory("PCG");
    SolverFactory<TConfig>::unregisterFactory("AMG");
    SolverFactory<TConfig>::unregisterFactory("PBICGSTAB");
    SolverFactory<TConfig>::unregisterFactory("BICGSTAB");
    SolverFactory<TConfig>::unregisterFactory("FGMRES");
    SolverFactory<TConfig>::unregisterFactory("JACOBI_L1");
    SolverFactory<TConfig>::unregisterFactory("GS");
    SolverFactory<TConfig>::unregisterFactory("POLYNOMIAL");
    SolverFactory<TConfig>::unregisterFactory("KPZ_POLYNOMIAL");
    SolverFactory<TConfig>::unregisterFactory("BLOCK_JACOBI");
    SolverFactory<TConfig>::unregisterFactory("MULTICOLOR_GS");
    SolverFactory<TConfig>::unregisterFactory("MULTICOLOR_DILU");
    //Unegister Aggregation Selectors
    aggregation::SelectorFactory<TConfig>::unregisterFactory("SIZE_2");
    aggregation::SelectorFactory<TConfig>::unregisterFactory("SIZE_4");
    aggregation::SelectorFactory<TConfig>::unregisterFactory("SIZE_8");
    aggregation::SelectorFactory<TConfig>::unregisterFactory("DUMMY");
    //Unegister Energymin Selectors
    classical::SelectorFactory<TConfig>::unregisterFactory("CR");
    //Unegister Aggregation Coarse Generators
    aggregation::CoarseAGeneratorFactory<TConfig>::unregisterFactory("LOW_DEG");
    aggregation::CoarseAGeneratorFactory<TConfig>::unregisterFactory("HYBRID");
    aggregation::CoarseAGeneratorFactory<TConfig>::unregisterFactory("THRUST");
    //Unegister Energymin Interpolators
    energymin::InterpolatorFactory<TConfig>::unregisterFactory("EM");
    ConvergenceFactory<TConfig>::unregisterFactory("ABSOLUTE");
    ConvergenceFactory<TConfig>::unregisterFactory("RELATIVE_INI_CORE");
    //MatrixIO<TConfig>::unregisterReaders();
}

inline void unregisterClassesClassical()
{
    //Unegister Classical Interpolators
    InterpolatorFactory<TConfig>::unregisterFactory("D1");
    InterpolatorFactory<TConfig>::unregisterFactory("D2");
    //Unegister Classical Selectors
    classical::SelectorFactory<TConfig>::unregisterFactory("PMIS");
    //Unegister Classical Strength
    StrengthFactory<TConfig>::unregisterFactory("AHAT");
    StrengthFactory<TConfig>::unregisterFactory("ALL");
}

AMGX_ERROR  test_finalize()
{
    try
    {
        unregisterClasses();
        unregisterClassesClassical();
    }
    catch (amgx_exception e)
    {
        return AMGX_ERR_CORE;
    }

    return AMGX_OK;
}

AMGX_ERROR  test_initialize()
{
    try
    {
        registerClasses();
        registerClassesClassical();
    }
    catch (amgx_exception e)
    {
        return AMGX_ERR_CORE;
    }

    return AMGX_OK;
}

void run()
{
    AMGX_finalize();
    UnitTest::amgx_intialized = false;
    AMGX_ERROR errorCode;
    errorCode = test_initialize();
    UNITTEST_ASSERT_EQUAL(errorCode, AMGX_OK);
    errorCode = test_initialize();
    UNITTEST_ASSERT_EQUAL(errorCode, AMGX_ERR_CORE);
    errorCode = test_finalize();
    UNITTEST_ASSERT_EQUAL(errorCode, AMGX_OK);
    errorCode = test_initialize();
    UNITTEST_ASSERT_EQUAL(errorCode, AMGX_OK);
    errorCode = test_finalize();
    UNITTEST_ASSERT_EQUAL(errorCode, AMGX_OK);
    errorCode = test_finalize();
    UNITTEST_ASSERT_EQUAL(errorCode, AMGX_ERR_CORE);
    AMGX_initialize();
    UnitTest::amgx_intialized = true;
}

DECLARE_UNITTEST_END(FactoriesTest);

#define AMGX_CASE_LINE(CASE) FactoriesTest <TemplateMode<CASE>::Type>  FactoriesTest_##CASE;
AMGX_FORALL_BUILDS(AMGX_CASE_LINE)
#undef AMGX_CASE_LINE

} //namespace amgx
