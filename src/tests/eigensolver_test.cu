/* Copyright (c) 2013-2017, NVIDIA CORPORATION. All rights reserved.
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

#include <unit_test.h>
#include <matrix_io.h>
#include <test_utils.h>
#include <multiply.h>
#include <blas.h>
#include <norm.h>
#include <sstream>
#include <eigensolvers/single_iteration_eigensolver.h>

namespace amgx
{

template <typename T>
struct Epsilon
{};

template <>
struct Epsilon<float>
{
    static float value() { return 1.e-4; }
};

template <>
struct Epsilon<double>
{
    static double value() { return 1.e-8; }
};

DECLARE_UNITTEST_BEGIN(EigenSolverTest_Base);

template <typename TConfig>
void test_solve(Matrix_h &B, AMG_Config &cfg,
                ValueTypeB &eigenvalue, Vector_h &eigenvector)
{
    Matrix<TConfig> A(B);
    Vector<TConfig> x(A.get_num_rows());
    fillRandom< Vector<TConfig> >::fill(x);
    EigenSolver<TConfig> *solver = EigenSolverFactory<TConfig>::allocate(cfg, "default", "eig_solver");
    UNITTEST_ASSERT_TRUE(solver != 0);
    solver->setup(A);
    UNITTEST_ASSERT_TRUE(solver->solve(x) == AMGX_ST_CONVERGED);
    eigenvalue = solver->get_eigenvalues().front();
    eigenvector = solver->get_eigenvectors().front();
    double tolerance = cfg.getParameter<double>("eig_tolerance", "default");
    ValueTypeB epsilon = Epsilon<ValueTypeB>::value();
    ValueTypeB eigenvector_norm = get_norm(B, eigenvector, L2);
    // Multiply epsilon by 10 since the L2 norm implementation provided by thrust is not sufficiently accurate.
    UNITTEST_ASSERT_EQUAL_TOL_DESC("Eigenvector is not normalized", eigenvector_norm, 1, 10 * epsilon);
    // Residual should be close to 0.
    Vector_h rhs = eigenvector;
    scal(rhs, eigenvalue);
    Vector_h residual(eigenvector.size());
    axmb(B, eigenvector, rhs, residual);
    Vector_h zeros(eigenvector.size(), ValueTypeB(0));
    UNITTEST_ASSERT_EQUAL_TOL_DESC("Residual is too large.", residual, zeros, tolerance);
}

void test(const std::string &filename, AMG_Config &cfg)
{
    this->randomize(42);
    Matrix_h A;
    Vector_h x;
    Vector_h b;
    UNITTEST_ASSERT_TRUE(MatrixIO<TConfig_h>::readSystem(filename.c_str(), A, b, x) == AMGX_OK);
    ValueTypeB eigenvalue;
    Vector_h eigenvector;
    test_solve<TConfig>(A, cfg, eigenvalue, eigenvector);
}

void test_compare_cpu(const std::string &filename, AMG_Config &cfg)
{
    this->randomize(42);
    Matrix_h A;
    Vector_h x;
    Vector_h b;
    UNITTEST_ASSERT_TRUE(MatrixIO<TConfig_h>::readSystem(filename.c_str(), A, b, x) == AMGX_OK);
    ValueTypeB rhs_eigenvalue;
    Vector_h rhs_eigenvector;
    test_solve<TConfig_h>(A, cfg, rhs_eigenvalue, rhs_eigenvector);
    ValueTypeB lhs_eigenvalue;
    Vector_h lhs_eigenvector;
    test_solve<TConfig>(A, cfg, lhs_eigenvalue, lhs_eigenvector);
    double tolerance = cfg.getParameter<double>("eig_tolerance", "default");
    UNITTEST_ASSERT_EQUAL_TOL_DESC("Comparison: eigenvalues are too different.", lhs_eigenvalue, rhs_eigenvalue, tolerance);
}

DECLARE_UNITTEST_END(EigenSolverTest_Base);

DECLARE_UNITTEST_BEGIN_EXTD(EigenSolverTest_PowerIteration, EigenSolverTest_Base<T_Config>)
void run()
{
    AMG_Config cfg;
    cfg.parseParameterString("eig_solver=POWER_ITERATION;"
                             "eig_max_iters=1000;"
                             "eig_tolerance=1e-4;"
                             "eig_which=largest;");
    EigenSolverTest_Base<T_Config>::test_compare_cpu(UnitTest::get_configuration().data_folder + "Public/Florida/poisson2D.mtx.bin", cfg);
}
DECLARE_UNITTEST_END(EigenSolverTest_PowerIteration)
EigenSolverTest_PowerIteration<TemplateMode<AMGX_mode_dDDI>::Type> EigenSolverTest_PowerIteration_dDDI;

DECLARE_UNITTEST_BEGIN_EXTD(EigenSolverTest_InverseIteration, EigenSolverTest_Base<T_Config>)
void run()
{
    AMG_Config cfg;
    cfg.parseParameterString("solver=FGMRES;"
                             "preconditioner=NOSOLVER;"
                             "tolerance=1e-6;"
                             "max_iters=100;"
                             "monitor_residual=1;"
                             "eig_solver=INVERSE_ITERATION;"
                             "eig_max_iters=500;"
                             "eig_tolerance=1e-4;"
                             "eig_which=smallest;");
    EigenSolverTest_Base<T_Config>::test_compare_cpu(UnitTest::get_configuration().data_folder + "Public/Florida/poisson2D.mtx.bin", cfg);
}
DECLARE_UNITTEST_END(EigenSolverTest_InverseIteration)
EigenSolverTest_InverseIteration<TemplateMode<AMGX_mode_dDDI>::Type> EigenSolverTest_InverseIteration_dDDI;

/*
  DECLARE_UNITTEST_BEGIN_EXTD(EigenSolverTest_Arnoldi, EigenSolverTest_Base<T_Config>)
  void run()
  {
AMG_Config cfg;
cfg.parseParameterString("eig_solver=ARNOLDI;"
           "eig_max_iters=64;"
           "eig_tolerance=1e-3;"
           "eig_which=largest;"
                               "eig_eigenvector=1;"
                               "eig_eigenvector_solver=default");
EigenSolverTest_Base<T_Config>::test(UnitTest::get_configuration().data_folder + "Public/Florida/poisson2D.mtx", cfg);
  }
  DECLARE_UNITTEST_END(EigenSolverTest_Arnoldi)
  EigenSolverTest_Arnoldi<TemplateMode<AMGX_mode_dDDI>::Type> EigenSolverTest_Arnoldi_dDDI;

  DECLARE_UNITTEST_BEGIN_EXTD(EigenSolverTest_Lanczos, EigenSolverTest_Base<T_Config>)
  void run()
  {
AMG_Config cfg;
cfg.parseParameterString("eig_solver=LANCZOS;"
           "eig_max_iters=64;"
           "eig_tolerance=1e-4;"
           "eig_which=largest;"
                               "eig_eigenvector=1;"
                               "eig_eigenvector_solver=default");
EigenSolverTest_Base<T_Config>::test(UnitTest::get_configuration().data_folder + "Internal/poisson/poisson7x2x4x4.mtx", cfg);
  }
  DECLARE_UNITTEST_END(EigenSolverTest_Lanczos)
  EigenSolverTest_Lanczos<TemplateMode<AMGX_mode_dDDI>::Type> EigenSolverTest_Lanczos_dDDI;

  DECLARE_UNITTEST_BEGIN_EXTD(EigenSolverTest_LOBPCG, EigenSolverTest_Base<T_Config>)
  void run()
  {
AMG_Config cfg;
cfg.parseParameterString("eig_solver=LOBPCG;"
           "eig_max_iters=50;"
           "eig_tolerance=1e-4;"
           "eig_which=largest;"
                               "eig_eigenvector=1;"
                               "eig_eigenvector_solver=default");
EigenSolverTest_Base<T_Config>::test(UnitTest::get_configuration().data_folder + "Internal/poisson/poisson7x2x4x4.mtx", cfg);
  }
  DECLARE_UNITTEST_END(EigenSolverTest_LOBPCG)
  EigenSolverTest_LOBPCG<TemplateMode<AMGX_mode_dDDI>::Type> EigenSolverTest_LOBPCG_dDDI;

  DECLARE_UNITTEST_BEGIN_EXTD(EigenSolverTest_JacobiDavidson, EigenSolverTest_Base<T_Config>)
  void run()
  {
AMG_Config cfg;
cfg.parseParameterString("eig_solver=JACOBI_DAVIDSON;"
           "eig_max_iters=128;"
           "eig_tolerance=1e-4;"
           "eig_which=largest;"
                               "eig_eigenvector=1;"
                               "eig_eigenvector_solver=default;"
                               "solver=FGMRES;"
                               "gmres_n_restart=10;"
                               "preconditioner=NOSOLVER;"
                               "max_iters=20;");
EigenSolverTest_Base<T_Config>::test(UnitTest::get_configuration().data_folder + "Internal/poisson/poisson7x2x4x4.mtx", cfg);
  }
  DECLARE_UNITTEST_END(EigenSolverTest_JacobiDavidson)
  EigenSolverTest_JacobiDavidson<TemplateMode<AMGX_mode_dDDI>::Type> EigenSolverTest_JacobiDavidson_dDDI;

  DECLARE_UNITTEST_BEGIN_EXTD(EigenSolverTest_SubspaceIteration, EigenSolverTest_Base<T_Config>)
  void run()
  {
AMG_Config cfg;
cfg.parseParameterString("eig_solver=SUBSPACE_ITERATION;"
           "eig_max_iters=32;"
           "eig_tolerance=1e-2;"
           "eig_which=largest;"
                               "eig_wanted_count=2;"
                               "eig_eigenvector=1;"
                               "eig_eigenvector_solver=default;");
EigenSolverTest_Base<T_Config>::test(UnitTest::get_configuration().data_folder + "Internal/poisson/poisson7x2x4x4.mtx", cfg);
  }
  DECLARE_UNITTEST_END(EigenSolverTest_SubspaceIteration)
  EigenSolverTest_SubspaceIteration<TemplateMode<AMGX_mode_dDDI>::Type> EigenSolverTest_SubspaceIteration_dDDI;
*/

}
