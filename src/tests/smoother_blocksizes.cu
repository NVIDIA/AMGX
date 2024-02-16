// SPDX-FileCopyrightText: 2011 - 2024 NVIDIA CORPORATION. All Rights Reserved.
//
// SPDX-License-Identifier: BSD-3-Clause

#include "unit_test.h"
#include "test_utils.h"
#include "util.h"
#include "time.h"
#include "amg_solver.h"
#include "solvers/solver.h"

#include "solvers/multicolor_dilu_solver.h"

#include <iostream>

// this test run same smoother for the same matrix with DIAG=true and DIAG=false. Runs for different blocksizes (1-7). checks if result is ~the same

namespace amgx
{

template< typename T >
struct Epsilon
{};

template<>
struct Epsilon<float>
{
    static __device__ __host__ __forceinline__ float value( ) { return 1.0e-5f; }
};

template<>
struct Epsilon<double>
{
    static __device__ __host__ __forceinline__ double value( ) { return 1.0e-10; }
};

DECLARE_UNITTEST_BEGIN(SmootherBlocksizes);

void check_block_smoothers_random(int bsize, int rows, std::string smoother_string, int n_smoothing_steps, bool &done)
{
    done = false;
    typedef TemplateConfig<AMGX_host, TConfig::vecPrec, TConfig::matPrec, TConfig::indPrec> TConfig_h;
    Matrix<TConfig_h> A_h;
    this->randomize( 5 );
    generateMatrixRandomStruct<TConfig_h>::generateExact( A_h, 16, false, bsize, false, 10);
    random_fill(A_h);

    for (int i = 0; i < A_h.values.size(); i++)
    {
        A_h.values[i] = 0.1;
    }

    A_h.computeDiagonal();
    A_h.set_initialized(1);
    int n_rows = A_h.get_num_rows() * bsize;
    // Force the diagonal values to be nonzeros
    /*for (int i=0;i<A_h.get_num_rows();i++)
    {
      for (int j=0;j<A_h.get_block_size();j++)
      {
        // Make diagonal large
        //A_h.values[A_h.diag[i]*A_h.get_block_size() + j] += 100;
        if (j%A_h.get_block_dimx() == j/A_h.get_block_dimx())
        {
          A_h.values[A_h.diag[i]*A_h.get_block_size() + j] += 1000;
        }
      }
    }*/

    for (int i = 0; i < A_h.get_num_rows(); i++)
    {
        for (int j = A_h.row_offsets[i]; j < A_h.row_offsets[i + 1]; j++)
        {
            printf("Row: %d, Col: %d\n", i, A_h.col_indices[j]);
            double val = 1;

            if (i == A_h.col_indices[j])
            {
                val = 5;
            }

            // Make diagonal large
            for (int k = 0; k < A_h.get_block_size(); k++)
                if (k % A_h.get_block_dimx() == k / A_h.get_block_dimx())
                {
                    A_h.values[j * A_h.get_block_size() + k] = val;
                }
        }
    }

    fflush(stdout);
    Vector<TConfig_h> x_h;
    Vector<TConfig_h> b_h;
    b_h.resize(n_rows);
    b_h.set_block_dimy(A_h.get_block_dimy());
    x_h.resize(n_rows);
    x_h.set_block_dimy(A_h.get_block_dimx());

    for (int i = 0; i < n_rows; i++)
    {
        x_h[i] = 0;//1.0-rand()/RAND_MAX;
        b_h[i] = 1.0;//+rand()/RAND_MAX;
    }

    Matrix<TConfig> A0 = A_h; // copy to device
    Vector<TConfig> b0 = b_h;
    Vector<TConfig> x0 = x_h;
    // Fill b with ones
    AMG_Config cfg;
    // not reordered, no bsrxmv, outside diagonal
    std::string parameter_string = "print_grid_stats=1, max_iters=1, monitor_residual=1, store_res_history=1, print_solve_stats=1, solver=" + smoother_string + ", determinism_flag=1, smoother_weight=1.0, matrix_coloring_scheme=MIN_MAX, max_uncolored_percentage=0.0, num_colors=10, coloring_level=1" ;
    const std::string &cfg_scope = "default";
    cfg.parseParameterString(const_cast<char *>(parameter_string.c_str()));
    this->PrintOnFail("config creation");
    Solver<TConfig> *smoother0 = SolverFactory<T_Config>::allocate(cfg, cfg_scope, "solver");
    this->PrintOnFail("smoother0 creation");
    MatrixIO<TConfig_h>::writeSystemMatrixMarket("A_host.mtx", &A_h, NULL, NULL);
    UNITTEST_ASSERT_TRUE(smoother0 != NULL);
    smoother0->setup(A0, false);
    multicolor_dilu_solver::MulticolorDILUSolver_Base<TConfig> *dilu_smoother = dynamic_cast<multicolor_dilu_solver::MulticolorDILUSolver_Base<TConfig>*>(smoother0);
    typename Matrix<TConfig>::MVector_h einv_h = dilu_smoother->Einv;
    smoother0->set_max_iters(n_smoothing_steps);
    x_h = x0;
    smoother0->solve(b0, x0, false);
    x_h = x0;
    done = true;
    //for (int i=0;i<1;i++)
    int i = -1;

    if (0)
    {
        Matrix<TConfig> A1 = A_h; // copy to device
        Vector<TConfig> b1 = b_h;
        Vector<TConfig> x1 = x_h;
        AMG_Config cfg1;
        std::string parameter_string1;

        if (i == 1)
        {
            Matrix<TConfig_h> A_temp = A_h;
            // outside diagonal
            A_temp.convert(A_h, CSR | DIAG, A_h.get_block_dimx(), A_h.get_block_dimy());
            A1 = A_temp;
            // reordered, bsrxmv,  outside diag
            parameter_string1 = "print_grid_stats=1, monitor_residual=1, store_res_history=1, print_solve_stats=1,solver=" + smoother_string + ", determinism_flag=1, smoother_weight=1.0, matrix_coloring_scheme=MIN_MAX, max_uncolored_percentage=0.0, reorder_cols_by_color=1, coloring_level=1, use_bsrxmv=1";
        }
        else if (i == 0)
        {
            // reordered, bsrxmv, internal diagonal
            parameter_string1 = "print_grid_stats=1, monitor_residual=1, store_res_history=1, print_solve_stats=1, solver=" + smoother_string + ", determinism_flag=1, smoother_weight=1.0, matrix_coloring_scheme=MIN_MAX, max_uncolored_percentage=0.0, reorder_cols_by_color=1, insert_diag_while_reordering=1, coloring_level=1, use_bsrxmv=1";
        }
        else if (i == 2)
        {
            // outside diagonal
            parameter_string1 = "print_grid_stats=1, monitor_residual=1, store_res_history=1, print_solve_stats=1,solver=" + smoother_string + ", determinism_flag=1, smoother_weight=1.0, matrix_coloring_scheme=MIN_MAX, max_uncolored_percentage=0.0,  coloring_level=1" ;
            Matrix<TConfig_h> A_temp = A_h;
            // outside diagonal
            A_temp.convert(A_h, CSR | DIAG, A_h.get_block_dimx(), A_h.get_block_dimy());
            A1 = A_temp;
        }

        cfg1.parseParameterString(const_cast<char *>(parameter_string1.c_str()));
        Solver<TConfig> *smoother1 = SolverFactory<T_Config>::allocate(cfg1, cfg_scope, "solver");
        this->PrintOnFail("smoother creation");
        UNITTEST_ASSERT_TRUE(smoother1 != NULL);
        //matrix_coloring_scheme2->colorMatrix(A2);
        smoother1->setup(A1, false);
        smoother1->set_max_iters(n_smoothing_steps);
        smoother1->solve(b1, x1, false);
        char buf[100];
        sprintf(buf, "Smoother in/outside diag %s bs %d result inequality", smoother_string.c_str(), bsize);
        typedef typename MatPrecisionMap<TConfig::matPrec>::Type Value_type;
        const Value_type epsilon = Epsilon<Value_type>::value();
        UNITTEST_ASSERT_EQUAL_TOL_DESC(buf, x0, x1, epsilon);
        done = true;
    }
}

void run()
{
    bool done;
    int n_smoothing_steps = 1;
    std::stringstream err_msg;
    int mat_size = 1000;

    for (int i = 1; i < 5; i++)
    {
        /*err_msg << "block jacobi, block_size= " << i ;
          check_block_smoothers_random(i, mat_size*i, "BLOCK_JACOBI", n_smoothing_steps,done) ;
        UNITTEST_ASSERT_TRUE_DESC(err_msg.str().c_str(),done);*/
        /*err_msg << "SGS, block_size= " << i ;
          check_block_smoothers_random(i, mat_size, "MULTICOLOR_GS", n_smoothing_steps, done) ;
        UNITTEST_ASSERT_TRUE(done);*/
        err_msg << "DILU, block_size= " << i ;
        check_block_smoothers_random(i, mat_size, "MULTICOLOR_DILU", n_smoothing_steps, done);
        UNITTEST_ASSERT_TRUE_DESC(err_msg.str().c_str(), done);
    }
}

DECLARE_UNITTEST_END(SmootherBlocksizes);

SmootherBlocksizes <TemplateMode<AMGX_mode_dDDI>::Type>  SmootherBlocksizes_instance_mode_dDDI;
SmootherBlocksizes <TemplateMode<AMGX_mode_dDFI>::Type>  SmootherBlocksizes_instance_mode_dDFI;
SmootherBlocksizes <TemplateMode<AMGX_mode_dFFI>::Type>  SmootherBlocksizes_instance_mode_dFFI;

}
