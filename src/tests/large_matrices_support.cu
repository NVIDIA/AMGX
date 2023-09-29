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

#include "unit_test.h"
#include "core.h"
#include "amg_config.h"
#include "test_utils.h"
#include "util.h"
#include "cutil.h"
#include "amg_solver.h"
#include "resources.h"

#include "aggregation/coarseAgenerators/coarse_A_generator.h"
#include "aggregation/selectors/agg_selector.h"
#include "matrix_coloring/matrix_coloring.h"
#include "matrix_coloring/min_max.h"
#include "solvers/solver.h"

#include "classical/selectors/selector.h"
#include "classical/interpolators/interpolator.h"
#include "classical/strength/strength.h"

#include <cusp/print.h>
#include <cusp/gallery/poisson.h>

namespace amgx

{

DECLARE_UNITTEST_BEGIN(LargeMatricesSupport);

typedef typename TConfig_h::template setVecPrec<(AMGX_VecPrecision)AMGX_GET_MODE_VAL(AMGX_MatPrecision, TConfig::mode)>::Type vvec_h;
typedef typename TConfig::template setVecPrec<AMGX_vecInt>::Type ivec;
typedef typename TConfig_h::template setVecPrec<AMGX_vecInt>::Type ivec_h;

// setup restriction on HOST
void fillRowOffsetsAndColIndices(const int num_aggregates,
                                 Vector<ivec_h> aggregates,
                                 const int R_num_cols,
                                 Vector<ivec_h> &R_row_offsets,
                                 Vector<ivec_h> &R_col_indices)
{
    for (int i = 0; i < num_aggregates + 1; i++)
    {
        R_row_offsets[i] = 0;
    }

    // Count number of neighbors for each row
    for (int i = 0; i < R_num_cols; i++)
    {
        int I = aggregates[i];
        R_row_offsets[I]++;
    }

    R_row_offsets[num_aggregates] = R_num_cols;

    for (int i = num_aggregates - 1; i >= 0; i--)
    {
        R_row_offsets[i] = R_row_offsets[i + 1] - R_row_offsets[i];
    }

    /* Set column indices. */
    for (int i = 0; i < R_num_cols; i++)
    {
        int I = aggregates[i];
        int Ip = R_row_offsets[I]++;
        R_col_indices[Ip] = i;
    }

    /* Reset r[i] to start of row memory. */
    for (int i = num_aggregates - 1; i > 0; i--)
    {
        R_row_offsets[i] = R_row_offsets[i - 1];
    }

    R_row_offsets[0] = 0;
}

void test_coarsers(Matrix<T_Config> &A, AMG_Config &cfg, const std::string &cfg_scope)
{
    Matrix<T_Config> Ac;
    int num_aggregates = A.get_num_rows();
    Vector<ivec_h> h_aggregates;
    h_aggregates.resize( A.get_num_rows() );

    for ( int i = 0; i < h_aggregates.size(); i++ )
    {
        h_aggregates[i] = i;
    }

    Vector<ivec_h> h_R_row_offsets;
    Vector<ivec_h> h_R_col_indices;
    h_R_row_offsets.resize( num_aggregates + 1 );
    h_R_col_indices.resize( A.get_num_rows() );
    fillRowOffsetsAndColIndices( num_aggregates, h_aggregates, A.get_num_rows(), h_R_row_offsets, h_R_col_indices );
    Vector<ivec> aggregates = h_aggregates;
    Vector<ivec> R_row_offsets = h_R_row_offsets;
    Vector<ivec> R_col_indices = h_R_col_indices;
    cudaCheckError();
    typename aggregation::CoarseAGeneratorFactory<T_Config>::Iterator iter = aggregation::CoarseAGeneratorFactory<T_Config>::getIterator();
    aggregation::CoarseAGenerator<TConfig> *generator;

    while (!aggregation::CoarseAGeneratorFactory<T_Config>::isIteratorLast(iter))
    {
        generator = NULL;
        generator = iter->second->create(cfg, cfg_scope);
        PrintOnFail("%s : Matrix properties: blocksize = %d, diag_prop = %d\n", iter->first.c_str(), A.get_block_dimy(), (A.hasProps(DIAG) ? 1 : 0));
        UNITTEST_ASSERT_TRUE_DESC("Generator is not created\n", generator != NULL);
        UNITTEST_ASSERT_EXCEPTION_START;
        PrintOnFail("%s : Matrix properties: blocksize = %d, diag_prop = %d\n", iter->first.c_str(), A.get_block_dimy(), (A.hasProps(DIAG) ? 1 : 0));
        generator->computeAOperator(A, Ac, aggregates, R_row_offsets, R_col_indices, num_aggregates);
        UNITTEST_ASSERT_TRUE_DESC("Coarser matrix contains nans\n", !containsNan<ValueTypeA>(Ac.values.raw(), Ac.values.size()));
        UNITTEST_ASSERT_EXCEPTION_END_NOT_IMPLEMENTED;

        if (generator != NULL) { delete generator; }

        ++iter;
        UNITTEST_ASSERT_TRUE(true);
    }
}

void test_selectors(Matrix<T_Config> &A, AMG_Config &cfg, const std::string &cfg_scope)
{
    typename aggregation::SelectorFactory<T_Config>::Iterator iter = aggregation::SelectorFactory<T_Config>::getIterator();
    aggregation::Selector<TConfig> *selector;
    IVector vec, vec1;
    int num;

    while (!aggregation::SelectorFactory<T_Config>::isIteratorLast(iter))
    {
        selector = NULL;
        PrintOnFail("processing: %s\n", iter->first.c_str());
        selector = iter->second->create(cfg, cfg_scope);
        PrintOnFail("Selector creation\n");
        UNITTEST_ASSERT_TRUE(selector != NULL);
        UNITTEST_ASSERT_EXCEPTION_START;
        PrintOnFail("%s : Matrix properties: blocksize = %d, diag_prop = %d\n", iter->first.c_str(), A.get_block_dimy(), (A.hasProps(DIAG) ? 1 : 0));
        selector->setAggregates(A, vec, vec1, num);
        UNITTEST_ASSERT_EXCEPTION_END_NOT_IMPLEMENTED;

        if (selector != NULL) { delete selector; }

        ++iter;
        UNITTEST_ASSERT_TRUE(true);
    }
}

void test_matrix_coloring(Matrix<T_Config> &A, AMG_Config &cfg, const std::string &cfg_scope)
{
    MatrixColoring<TConfig> *color;
    typename MatrixColoringFactory<T_Config>::Iterator iter = MatrixColoringFactory<T_Config>::getIterator();

    while (!MatrixColoringFactory<T_Config>::isIteratorLast(iter))
    {
        color = NULL;
        UNITTEST_ASSERT_EXCEPTION_START;
        PrintOnFail("%s : Matrix properties: blocksize = %d, diag_prop = %d\n", iter->first.c_str(), A.get_block_dimy(), (A.hasProps(DIAG) ? 1 : 0));
        color = iter->second->create(cfg, cfg_scope);
        UNITTEST_ASSERT_TRUE(color != NULL);
        A.colorMatrix(cfg, cfg_scope);
        int num_colors = A.getMatrixColoring().getNumColors();
        UNITTEST_ASSERT_TRUE(num_colors != 0);
        UNITTEST_ASSERT_EXCEPTION_END_NOT_IMPLEMENTED;

        if (color != NULL) { delete color; }

        ++iter;
        UNITTEST_ASSERT_TRUE(true);
    }
}

void test_solvers(Matrix<T_Config> &A, AMG_Config &cfg, const std::string &cfg_scope)
{
    Vector<T_Config> b (A.get_num_rows()*A.get_block_dimy()), x (A.get_num_rows()*A.get_block_dimy());
    thrust_wrapper::fill<T_Config::memSpace>(b.begin(), b.end(), 1);
    Solver<TConfig> *solver;
    typename SolverFactory<T_Config>::Iterator iter = SolverFactory<T_Config>::getIterator();

    while (!SolverFactory<T_Config>::isIteratorLast(iter))
    {
        solver = NULL;
        thrust_wrapper::fill<T_Config::memSpace>(x.begin(), x.end(), static_cast<ValueTypeB>(1.0));
        UNITTEST_ASSERT_EXCEPTION_START;
        PrintOnFail("%s : Matrix properties: blocksize = %d, diag_prop = %d\n", iter->first.c_str(), A.get_block_dimy(), (A.hasProps(DIAG) ? 1 : 0));
        solver = iter->second->create(cfg, cfg_scope);

        if (solver != NULL)
        {
            solver->setup(A, false);
            solver->set_max_iters(1);
            solver->solve(b, x, false);
            UNITTEST_ASSERT_TRUE_DESC("Smoother result contains nans\n", !containsNan<ValueTypeB>(x.raw(), x.size()));
        }

        UNITTEST_ASSERT_EXCEPTION_END_NOT_IMPLEMENTED;

        if (solver != NULL) { delete solver; }

        ++iter;
        UNITTEST_ASSERT_TRUE(true);
    }
}

void generatePoissonForTest(Matrix<TConfig > &Aout, int block_size, bool diag_prop, int points, int x, int y, int z = 1)
{
    Matrix<TConfig_h > Ac;
    {
        Matrix<TConfig_h > A;
        A.set_initialized(0);
        A.addProps(CSR);
        MatrixCusp<TConfig_h, cusp::csr_format> wA(&A);

        switch (points)
        {
            case 5:
                cusp::gallery::poisson5pt(wA, x, y);
                break;

            case 7:
                cusp::gallery::poisson7pt(wA, x, y, z);
                break;

            case 9:
                cusp::gallery::poisson9pt(wA, x, y);
                break;

            case 27:
                cusp::gallery::poisson27pt(wA, x, y, z);
                break;
        }

        A.set_initialized(1);
        Ac.convert( A, ( diag_prop ? DIAG : 0 ) | CSR, block_size, block_size );
        Ac.set_initialized(1);
    }
    Aout = Ac;
}

void test_levels(Resources *res, Matrix<T_Config> &A)
{
    Vector<T_Config> b (A.get_num_rows()*A.get_block_dimy()), x (A.get_num_rows()*A.get_block_dimy());
    thrust_wrapper::fill<T_Config::memSpace>(b.begin(), b.end(), 1);
    thrust_wrapper::fill<T_Config::memSpace>(x.begin(), x.end(), 1);
    int bsize = A.get_block_dimy();
    b.set_block_dimx(1);
    b.set_block_dimy(bsize);
    x.set_block_dimy(1);
    x.set_block_dimx(bsize);
    AMGX_STATUS solve_status;
    {
        AMG_Configuration cfg;
        AMGX_ERROR err = AMGX_OK;
        UNITTEST_ASSERT_TRUE( cfg.parseParameterString("algorithm=CLASSICAL, smoother=MULTICOLOR_DILU, presweeps=1, postsweeps=1, matrix_coloring_scheme=MIN_MAX, determinism_flag=1, max_levels=2, max_iters=1, norm=L1") == AMGX_OK);
        AMG_Solver<TConfig> amg(res, cfg);
        err = amg.setup(A);

        if (err != AMGX_ERR_NOT_SUPPORTED_TARGET && err != AMGX_ERR_NOT_SUPPORTED_BLOCKSIZE && err != AMGX_ERR_NOT_IMPLEMENTED)
        {
            PrintOnFail("Classical algorithm: Matrix properties: blocksize = %d, diag_prop = %d\n", A.get_block_dimy(), (A.hasProps(DIAG) ? 1 : 0));
            UNITTEST_ASSERT_EQUAL(err, AMGX_OK);
            err = amg.solve( b, x, solve_status, true);

            if (err != AMGX_ERR_NOT_SUPPORTED_TARGET && err != AMGX_ERR_NOT_SUPPORTED_BLOCKSIZE && err != AMGX_ERR_NOT_IMPLEMENTED)
            {
                PrintOnFail("Classical algorithm: Matrix properties: blocksize = %d, diag_prop = %d\n", A.get_block_dimy(), (A.hasProps(DIAG) ? 1 : 0));
                UNITTEST_ASSERT_EQUAL(err, AMGX_OK);
                PrintOnFail("Classical algorithm: Matrix properties: blocksize = %d, diag_prop = %d\n", A.get_block_dimy(), (A.hasProps(DIAG) ? 1 : 0));
                UNITTEST_ASSERT_TRUE_DESC("Level solve result contains nans\n", !containsNan<ValueTypeB>(x.raw(), x.size()));
            }
        }
    }
    thrust_wrapper::fill<T_Config::memSpace>(x.begin(), x.end(), 1);
    {
        AMG_Configuration cfg;
        AMGX_ERROR err = AMGX_OK;
        UNITTEST_ASSERT_TRUE( cfg.parseParameterString("algorithm=AGGREGATION, smoother=MULTICOLOR_DILU, presweeps=1, postsweeps=1, selector=SIZE_4, coarseAgenerator=LOW_DEG, matrix_coloring_scheme=MIN_MAX, determinism_flag=1, max_levels=2, max_iters=1, norm=L1") == AMGX_OK);
        AMG_Solver<TConfig> amg(res, cfg);
        err = amg.setup(A);

        if (err != AMGX_ERR_NOT_SUPPORTED_TARGET && err != AMGX_ERR_NOT_SUPPORTED_BLOCKSIZE && err != AMGX_ERR_NOT_IMPLEMENTED)
        {
            PrintOnFail("Aggregation algorithm: Matrix properties: blocksize = %d, diag_prop = %d\n", A.get_block_dimy(), (A.hasProps(DIAG) ? 1 : 0));
            UNITTEST_ASSERT_EQUAL(err, AMGX_OK);
            err = amg.solve( b, x, solve_status, true);

            if (err != AMGX_ERR_NOT_SUPPORTED_TARGET && err != AMGX_ERR_NOT_SUPPORTED_BLOCKSIZE && err != AMGX_ERR_NOT_IMPLEMENTED)
            {
                PrintOnFail("Aggregation algorithm: Matrix properties: blocksize = %d, diag_prop = %d\n", A.get_block_dimy(), (A.hasProps(DIAG) ? 1 : 0));
                UNITTEST_ASSERT_EQUAL(err, AMGX_OK);
                PrintOnFail("Aggregation algorithm: Matrix properties: blocksize = %d, diag_prop = %d\n", A.get_block_dimy(), (A.hasProps(DIAG) ? 1 : 0));
                UNITTEST_ASSERT_TRUE_DESC("Level solve result contains nans\n", !containsNan<ValueTypeB>(x.raw(), x.size()));
            }
        }
    }
}

void test_strength(Matrix<T_Config> &A, AMG_Config &cfg, const std::string &cfg_scope, StrengthFactory<TConfig> **good )
{
    //allocate necessary memory
    typedef Vector<typename T_Config::template setVecPrec<AMGX_vecInt>::Type> IVector;
    typedef Vector<typename T_Config::template setVecPrec<AMGX_vecBool>::Type> BVector;
    typedef Vector<typename T_Config::template setVecPrec<AMGX_vecFloat>::Type> FVector;
    FVector weights(A.get_num_rows(), 0.0);
    BVector s_con(A.get_num_nz(), false);
    IVector cf_map(A.get_num_rows(), 0);
    IVector scratch(A.get_num_rows(), 0); //scratch memory of size num_rows
    //compute strong connections and weights
    double max_row_sum = cfg.getParameter<double>("max_row_sum", cfg_scope);
    Strength<T_Config> *strength;
    typename StrengthFactory<T_Config>::Iterator iter = StrengthFactory<T_Config>::getIterator();

    while (!StrengthFactory<T_Config>::isIteratorLast(iter))
    {
        strength = NULL;
        UNITTEST_ASSERT_EXCEPTION_START;
        PrintOnFail("%s : Matrix properties: blocksize = %d, diag_prop = %d\n", iter->first.c_str(), A.get_block_dimy(), (A.hasProps(DIAG) ? 1 : 0));
        strength = iter->second->create(cfg, cfg_scope);
        UNITTEST_ASSERT_TRUE(strength != NULL);

        if (strength != NULL)
        {
            strength->computeStrongConnectionsAndWeights(A, s_con, weights, max_row_sum);
            UNITTEST_ASSERT_TRUE_DESC("Strength result contains nans\n", !containsNan<float>(weights.raw(), weights.size()));
            *good = iter->second;
        }

        UNITTEST_ASSERT_EXCEPTION_END_NOT_IMPLEMENTED;

        if (strength != NULL) { delete strength; }

        ++iter;
        UNITTEST_ASSERT_TRUE(true);
    }
}

void test_selectors(Matrix<T_Config> &A, AMG_Config &cfg, const std::string &cfg_scope, StrengthFactory<TConfig> *strengthf, classical::SelectorFactory<TConfig> **good )
{
    //allocate necessary memory
    typedef Vector<typename T_Config::template setVecPrec<AMGX_vecInt>::Type> IVector;
    typedef Vector<typename T_Config::template setVecPrec<AMGX_vecBool>::Type> BVector;
    typedef Vector<typename T_Config::template setVecPrec<AMGX_vecFloat>::Type> FVector;
    FVector weights(A.get_num_rows(), 0.0);
    BVector s_con(A.get_num_nz(), false);
    IVector cf_map(A.get_num_rows(), 0);
    IVector scratch(A.get_num_rows(), 0); //scratch memory of size num_rows
    //compute strong connections and weights
    double max_row_sum = cfg.getParameter<double>("max_row_sum", cfg_scope);
    Strength<T_Config> *strength = strengthf->create(cfg, cfg_scope);
    strength->computeStrongConnectionsAndWeights(A, s_con, weights, max_row_sum);
    classical::Selector<T_Config> *selector;
    typename classical::SelectorFactory<T_Config>::Iterator iter = classical::SelectorFactory<T_Config>::getIterator();

    while (!classical::SelectorFactory<T_Config>::isIteratorLast(iter))
    {
        selector = NULL;
        UNITTEST_ASSERT_EXCEPTION_START;
        PrintOnFail("%s : Matrix properties: blocksize = %d, diag_prop = %d\n", iter->first.c_str(), A.get_block_dimy(), (A.hasProps(DIAG) ? 1 : 0));
        selector = iter->second->create();
        UNITTEST_ASSERT_TRUE(strength != NULL);

        if (selector != NULL)
        {
            selector->markCoarseFinePoints(A, weights, s_con, cf_map, scratch);

            for (int i = 0; i < A.get_num_rows(); i++)
            {
                UNITTEST_ASSERT_TRUE(cf_map[i] != UNASSIGNED);
            }

            *good = iter->second;
        }

        UNITTEST_ASSERT_EXCEPTION_END_NOT_IMPLEMENTED;

        if (selector != NULL) { delete selector; }

        ++iter;
        UNITTEST_ASSERT_TRUE(true);
    }
}

void test_interpolators(Resources *res, Matrix<T_Config> &A, AMG_Config &cfg, const std::string &cfg_scope, StrengthFactory<TConfig> *strengthf, classical::SelectorFactory<TConfig> *selectorf )
{
    //allocate necessary memory
    typedef Vector<typename T_Config::template setVecPrec<AMGX_vecInt>::Type> IVector;
    typedef Vector<typename T_Config::template setVecPrec<AMGX_vecBool>::Type> BVector;
    typedef Vector<typename T_Config::template setVecPrec<AMGX_vecFloat>::Type> FVector;
    Matrix<TConfig> P;
    FVector weights(A.get_num_rows(), 0.0);
    BVector s_con(A.get_num_nz(), false);
    IVector cf_map(A.get_num_rows(), 0);
    IVector scratch(A.get_num_rows(), 0); //scratch memory of size num_rows
    //compute strong connections and weights
    double max_row_sum = cfg.getParameter<double>("max_row_sum", cfg_scope);
    Strength<T_Config> *strength = strengthf->create(cfg, cfg_scope);
    classical::Selector<T_Config> *selector = selectorf->create();
    strength->computeStrongConnectionsAndWeights(A, s_con, weights, max_row_sum);
    selector->markCoarseFinePoints(A, weights, s_con, cf_map, scratch);
    Interpolator<T_Config> *interpolator;
    typename InterpolatorFactory<T_Config>::Iterator iter = InterpolatorFactory<T_Config>::getIterator();
    AMG_Configuration scfg;
    AMG_Solver<TConfig> amg(res, scfg);

    while (!InterpolatorFactory<T_Config>::isIteratorLast(iter))
    {
        interpolator = NULL;
        UNITTEST_ASSERT_EXCEPTION_START;
        PrintOnFail("%s : Matrix properties: blocksize = %d, diag_prop = %d\n", iter->first.c_str(), A.get_block_dimy(), (A.hasProps(DIAG) ? 1 : 0));
        interpolator = iter->second->create(cfg, cfg_scope);
        UNITTEST_ASSERT_TRUE(strength != NULL);

        if (interpolator != NULL)
        {
            interpolator->generateInterpolationMatrix(A, cf_map, s_con, scratch, P, &amg);
        }

        UNITTEST_ASSERT_EXCEPTION_END_NOT_IMPLEMENTED;

        if (interpolator != NULL) { delete interpolator; }

        ++iter;
        UNITTEST_ASSERT_TRUE(true);
    }
}


void run()
{
    cudaCheckError();
    randomize( 30 );
    set_forge_ahead(true);
    int nrows_array[] = {8500000, 8500000, 8500002, 4250000}; // limited by gpu ram

    for (int bsize = 1; bsize < 1; ++bsize)
    {
        AMG_Config cfg;
        cfg.parseParameterString("determinism_flag=1");
        const std::string &cfg_scope = "default";
        Resources res;        // default resources
        {
            MatrixA A;
            generateMatrixRandomStruct<TConfig>::generateExact(A, nrows_array[bsize], true, bsize, false, 2);
            random_fill(A);
// aggregation
            test_coarsers(A, cfg, cfg_scope);
            test_selectors(A, cfg, cfg_scope);
            test_matrix_coloring(A, cfg, cfg_scope);
            A.colorMatrix(cfg, cfg_scope);
            test_solvers(A, cfg, cfg_scope);
// classical
//TODO: if strength cannot process matrix
            StrengthFactory<TConfig> *good_strength = NULL;
            test_strength(A, cfg, cfg_scope, &good_strength);

            if (good_strength != NULL)
            {
                classical::SelectorFactory<TConfig> *good_selector = NULL;
                test_selectors(A, cfg, cfg_scope, good_strength, &good_selector);

                if (good_selector != NULL)
                {
                    test_interpolators(&res, A, cfg, cfg_scope, good_strength, good_selector );
                }
            }

// levels
            test_levels(&res, A);
        }
        {
            MatrixA A;
            generateMatrixRandomStruct<TConfig>::generateExact(A, nrows_array[bsize], false, bsize, false, 2);
            random_fill(A);
// aggregation
            test_coarsers(A, cfg, cfg_scope);
            test_selectors(A, cfg, cfg_scope);
            test_matrix_coloring(A, cfg, cfg_scope);
            test_solvers(A, cfg, cfg_scope);
// classical
//TODO: if strength cannot process matrix
            StrengthFactory<TConfig> *good_strength = NULL;
            test_strength(A, cfg, cfg_scope, &good_strength);

            if (good_strength != NULL)
            {
                classical::SelectorFactory<TConfig> *good_selector = NULL;
                test_selectors(A, cfg, cfg_scope, good_strength, &good_selector);

                if (good_selector != NULL)
                {
                    test_interpolators(&res, A, cfg, cfg_scope, good_strength, good_selector );
                }
            }

// levels
            test_levels(&res, A);
        }
    }
}

DECLARE_UNITTEST_END(LargeMatricesSupport);


#define AMGX_CASE_LINE(CASE) LargeMatricesSupport <TemplateMode<CASE>::Type>  LargeMatricesSupport_##CASE;
AMGX_FORALL_BUILDS_DEVICE(AMGX_CASE_LINE)
#undef AMGX_CASE_LINE


} //namespace amgx
