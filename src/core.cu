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

#include <amg.h>
#include <types.h>
#include <misc.h>
#include <sstream>

#include <matrix_io.h>
#include <readers.h>

#include <matrix_coloring/min_max.h>
#include <matrix_coloring/min_max_2ring.h>
#include <matrix_coloring/parallel_greedy.h>
#include <matrix_coloring/round_robin.h>
#include <matrix_coloring/multi_hash.h>
#include <matrix_coloring/uniform.h>
#include <matrix_coloring/greedy_min_max_2ring.h>
#include <matrix_coloring/serial_greedy_bfs.h>
#include <matrix_coloring/greedy_recolor.h>
#include <matrix_coloring/locally_downwind.h>
#include <cycles/v_cycle.h>
#include <cycles/w_cycle.h>
#include <cycles/f_cycle.h>
#include <cycles/cg_cycle.h>
#include <cycles/cg_flex_cycle.h>


#include <solvers/solver.h>
#include <solvers/algebraic_multigrid_solver.h>
#include <solvers/pcgf_solver.h>
#include <solvers/cheb_solver.h>
#include <solvers/cg_solver.h>
#include <solvers/pcg_solver.h>
#include <solvers/idr_solver.h>
#include <solvers/idrmsync_solver.h>
#include <solvers/pbicgstab_solver.h>
#include <solvers/bicgstab_solver.h>
#include <solvers/fgmres_solver.h>
#include <solvers/gmres_solver.h>
//#include <solvers/jacobi_nocusp_solver.h>
//#include <solvers/jacobi_solver.h>
#include <solvers/jacobi_l1_solver.h>
#include <solvers/gauss_seidel_solver.h>
#include <solvers/block_jacobi_solver.h>
#include <solvers/cf_jacobi_solver.h>
#include <solvers/polynomial_solver.h>
#include <solvers/kpz_polynomial_solver.h>
#include <solvers/multicolor_gauss_seidel_solver.h>
#include <solvers/multicolor_dilu_solver.h>
#include <solvers/multicolor_ilu_solver.h>
#include <solvers/fixcolor_gauss_seidel_solver.h>
#include <solvers/dummy_solver.h>
#include <solvers/dense_lu_solver.h>
#include <solvers/kaczmarz_solver.h>
#include <solvers/chebyshev_poly.h>

#include <convergence/absolute.h>
#include <convergence/relative_max.h>
#include <convergence/relative_ini.h>
#include <convergence/combined_rel_ini_abs.h>

#include <classical/classical_amg_level.h>
#include <classical/interpolators/distance1.h>
#include <classical/interpolators/distance2.h>
#include <classical/interpolators/multipass.h>

#include <classical/selectors/pmis.h>
#include <classical/selectors/aggressive_pmis.h>
#include <classical/selectors/hmis.h>
#include <classical/selectors/aggressive_hmis.h>
#include <classical/selectors/dummy_selector.h>
#include <classical/selectors/cr.h>

#include <classical/strength/ahat.h>
#include <classical/strength/all.h>
#include <classical/strength/affinity.h>

#include <aggregation/aggregation_amg_level.h>
#include <aggregation/selectors/dummy.h>
#include <aggregation/selectors/size2_selector.h>
#include <aggregation/selectors/size4_selector.h>
#include <aggregation/selectors/size8_selector.h>
#include <aggregation/selectors/multi_pairwise.h>
#include <aggregation/selectors/geo_selector.h>
#include <aggregation/selectors/parallel_greedy_selector.h>
//#include <aggregation/selectors/serial_greedy.h>
//#include <aggregation/selectors/adaptive.h>

#include <aggregation/coarseAgenerators/hybrid_coarse_A_generator.h>
#include <aggregation/coarseAgenerators/low_deg_coarse_A_generator.h>
#include <aggregation/coarseAgenerators/thrust_coarse_A_generator.h>


#include <energymin/energymin_amg_level.h>
#include <energymin/interpolators/em.h>


#include <scalers/diagonal_symmetric.h>
#include <scalers/binormalization.h>
#include <scalers/nbinormalization.h>

#include <amgx_timer.h>
#include <version.h>

#include <distributed/amgx_mpi.h>
#include <resources.h>
#include <amgx_types/util.h>

namespace amgx
{

std::vector<std::string> getAllSolvers()
{
    std::vector<std::string> vec;
    std::set<std::string> solvers_set;
#define AMGX_CASE_LINE(CASE) \
      {\
        SolverFactory<TemplateMode<CASE>::Type>::Iterator iter = SolverFactory<TemplateMode<CASE>::Type>::getIterator();\
        while(!SolverFactory<TemplateMode<CASE>::Type>::isIteratorLast(iter))\
        {\
          solvers_set.insert(iter->first);\
          ++iter;\
        }\
      }
    AMGX_FORALL_BUILDS(AMGX_CASE_LINE)
#undef AMGX_CASE_LINE
    vec.assign(solvers_set.begin(), solvers_set.end());
    return vec;
}

std::vector<std::string> getAllInterpolators()
{
    std::vector<std::string> vec;
    std::set<std::string> interpolator_set;
#define AMGX_CASE_LINE(CASE) \
      {\
        InterpolatorFactory<TemplateMode<CASE>::Type>::Iterator iter = InterpolatorFactory<TemplateMode<CASE>::Type>::getIterator();\
        while(!InterpolatorFactory<TemplateMode<CASE>::Type>::isIteratorLast(iter))\
        {\
          interpolator_set.insert(iter->first);\
          ++iter;\
        }\
      }
    AMGX_FORALL_BUILDS(AMGX_CASE_LINE)
#undef AMGX_CASE_LINE
    vec.assign(interpolator_set.begin(), interpolator_set.end());
    return vec;
}

std::vector<std::string> getAllConvergence()
{
    std::vector<std::string> vec;
    std::set<std::string> convergence_set;
#define AMGX_CASE_LINE(CASE) \
      {\
        ConvergenceFactory<TemplateMode<CASE>::Type>::Iterator iter = ConvergenceFactory<TemplateMode<CASE>::Type>::getIterator();\
        while(!ConvergenceFactory<TemplateMode<CASE>::Type>::isIteratorLast(iter))\
        {\
          convergence_set.insert(iter->first);\
          ++iter;\
        }\
      }
    AMGX_FORALL_BUILDS(AMGX_CASE_LINE)
#undef AMGX_CASE_LINE
    vec.assign(convergence_set.begin(), convergence_set.end());
    return vec;
}


std::vector<std::string> getAllMatrixColoring()
{
    std::vector<std::string> vec;
    std::set<std::string> coloring_set;
#define AMGX_CASE_LINE(CASE) \
      {\
        MatrixColoringFactory<TemplateMode<CASE>::Type>::Iterator iter = MatrixColoringFactory<TemplateMode<CASE>::Type>::getIterator();\
        while(!MatrixColoringFactory<TemplateMode<CASE>::Type>::isIteratorLast(iter))\
        {\
          coloring_set.insert(iter->first);\
          ++iter;\
        }\
      }
    AMGX_FORALL_BUILDS(AMGX_CASE_LINE)
#undef AMGX_CASE_LINE
    vec.assign(coloring_set.begin(), coloring_set.end());
    return vec;
}

std::vector<std::string> getAllCycles()
{
    std::vector<std::string> vec;
    std::set<std::string> cycles_set;
#define AMGX_CASE_LINE(CASE) \
      {\
        CycleFactory<TemplateMode<CASE>::Type>::Iterator cycle_iter = CycleFactory<TemplateMode<CASE>::Type>::getIterator();\
        while(!CycleFactory<TemplateMode<CASE>::Type>::isIteratorLast(cycle_iter))\
        {\
          cycles_set.insert(cycle_iter->first);\
          ++cycle_iter;\
        }\
      }
    AMGX_FORALL_BUILDS(AMGX_CASE_LINE)
#undef AMGX_CASE_LINE
    vec.assign(cycles_set.begin(), cycles_set.end());
    return vec;
}

std::vector<std::string> getAllCoarseGenerators()
{
    std::vector<std::string> vec;
    std::set<std::string> coarse_gen_set;
#define AMGX_CASE_LINE(CASE) \
      {\
        aggregation::CoarseAGeneratorFactory<TemplateMode<CASE>::Type>::Iterator coarse_gen_iter = aggregation::CoarseAGeneratorFactory<TemplateMode<CASE>::Type>::getIterator();\
        while(!aggregation::CoarseAGeneratorFactory<TemplateMode<CASE>::Type>::isIteratorLast(coarse_gen_iter))\
        {\
          coarse_gen_set.insert(coarse_gen_iter->first);\
          ++coarse_gen_iter;\
        }\
      }
    AMGX_FORALL_BUILDS(AMGX_CASE_LINE)
#undef AMGX_CASE_LINE
    vec.assign(coarse_gen_set.begin(), coarse_gen_set.end());
    return vec;
}

std::vector<std::string> getClassicalSelectors()
{
    std::vector<std::string> vec;
    std::set<std::string> selectors_set;
#define AMGX_CASE_LINE(CASE) \
      {\
        classical::SelectorFactory<TemplateMode<CASE>::Type>::Iterator selector_iter = classical::SelectorFactory<TemplateMode<CASE>::Type>::getIterator();\
        while(!classical::SelectorFactory<TemplateMode<CASE>::Type>::isIteratorLast(selector_iter))\
        {\
          selectors_set.insert(selector_iter->first);\
          ++selector_iter;\
        }\
      }
    AMGX_FORALL_BUILDS(AMGX_CASE_LINE)
#undef AMGX_CASE_LINE
    vec.assign(selectors_set.begin(), selectors_set.end());
    return vec;
}

#if 1
std::vector<std::string> getAllScalers()
{
    std::vector<std::string> vec;
    std::set<std::string> scalers_set;
#define AMGX_CASE_LINE(CASE) \
    {\
      ScalerFactory<TemplateMode<CASE>::Type>::Iterator scaler_iter = ScalerFactory<TemplateMode<CASE>::Type>::getIterator();\
      while(!ScalerFactory<TemplateMode<CASE>::Type>::isIteratorLast(scaler_iter))\
      {\
        scalers_set.insert(scaler_iter->first);\
        ++scaler_iter;\
      }\
    }
    AMGX_FORALL_BUILDS(AMGX_CASE_LINE)
#undef AMGX_CASE_LINE
    vec.assign(scalers_set.begin(), scalers_set.end());
    return vec;
}
#endif

std::vector<std::string> getAggregationSelectors()
{
    std::vector<std::string> vec;
    std::set<std::string> selectors_set;
#define AMGX_CASE_LINE(CASE) \
      {\
        aggregation::SelectorFactory<TemplateMode<CASE>::Type>::Iterator selector_iter = aggregation::SelectorFactory<TemplateMode<CASE>::Type>::getIterator();\
        while(!aggregation::SelectorFactory<TemplateMode<CASE>::Type>::isIteratorLast(selector_iter))\
        {\
          selectors_set.insert(selector_iter->first);\
          ++selector_iter;\
        }\
      }
    AMGX_FORALL_BUILDS(AMGX_CASE_LINE)
#undef AMGX_CASE_LINE
    vec.assign(selectors_set.begin(), selectors_set.end());
    return vec;
}

std::vector<std::string> getAllStrength()
{
    std::vector<std::string> vec;
    std::set<std::string> strength_set;
#define AMGX_CASE_LINE(CASE) \
      {\
        StrengthFactory<TemplateMode<CASE>::Type>::Iterator strength_iter = StrengthFactory<TemplateMode<CASE>::Type>::getIterator();\
        while(!StrengthFactory<TemplateMode<CASE>::Type>::isIteratorLast(strength_iter))\
        {\
          strength_set.insert(strength_iter->first);\
          ++strength_iter;\
        }\
      }
    AMGX_FORALL_BUILDS(AMGX_CASE_LINE)
#undef AMGX_CASE_LINE
    vec.assign(strength_set.begin(), strength_set.end());
    return vec;
}

inline void registerParameters()
{
    std::vector<int> bool_flag_values;
    bool_flag_values.push_back(0);
    bool_flag_values.push_back(1);
    //Register Determinism and Exception Handling Parameters
    AMG_Config::registerParameter<int>("determinism_flag", "a flag that forces the various aggregators and matrix coloring algorithms to be deterministic (0:non-deterministic, 1:deterministic) <0>", 0, bool_flag_values);
    AMG_Config::registerParameter<int>("exception_handling", "a flag that forces internal exception processing instead of returning error codes(1:internal, 0:external)", 0, bool_flag_values);
    //Register System Parameters (memory pools)
    AMG_Config::registerParameter<size_t>("device_mem_pool_size", "size of the device memory pool in bytes", 256 * 1024 * 1024);
    AMG_Config::registerParameter<size_t>("device_mem_pool_max_alloc_size", "maximum size of a single allocation in the device memory pool in bytes", 20 * 1024 * 1024);
    AMG_Config::registerParameter<size_t>("device_alloc_scaling_factor", "over allocation for large buffers (in %% -- a value of X will lead to 100+X%% allocations)", 10);
    AMG_Config::registerParameter<size_t>("device_alloc_scaling_threshold", "buffers smaller than that threshold will NOT be scaled", 16 * 1024);
    AMG_Config::registerParameter<size_t>("device_mem_pool_size_limit", "size of the device memory pool in bytes. 0 - no limit", 0);
    //Register System Parameters (asynchronous framework)
    AMG_Config::registerParameter<int>("num_streams", "number of additional CUDA streams / threads used for async execution", 0);
    AMG_Config::registerParameter<int>("serialize_threads", "flag that enables thread serialization for debugging <0|1>", 0, bool_flag_values);
    AMG_Config::registerParameter<int>("high_priority_stream", "flag that enables high priority CUDA stream <0|1>", 0, bool_flag_values);
    //Register System Parameters (in distributed setting)
    std::vector<std::string> communicator_values;
    communicator_values.push_back("MPI");
    communicator_values.push_back("MPI_DIRECT");
    AMG_Config::registerParameter<std::string>("communicator", "type of communicator <MPI|MPI_DIRECT>", "MPI");
    std::vector<ViewType> viewtype_values;
    viewtype_values.push_back(INTERIOR);
    viewtype_values.push_back(OWNED);
    viewtype_values.push_back(FULL);
    viewtype_values.push_back(ALL);
    AMG_Config::registerParameter<ViewType>("separation_interior", "separation for latency hiding and coloring/smoothing <ViewType>", INTERIOR, viewtype_values);
    AMG_Config::registerParameter<ViewType>("separation_exterior", "limit of calculations for coloring/smoothing <ViewType>", OWNED, viewtype_values);
    AMG_Config::registerParameter<int>("min_rows_latency_hiding", "number of rows at which to disable latency hiding, negative value means latency hiding is completely disabled", -1);
    AMG_Config::registerParameter<int>("exact_coarse_solve", "flag that changes the dense LU coarse solve to solve the exact global problem for Classical AMG preconditioning <0=disable|1=enable>", 0, bool_flag_values);
    AMG_Config::registerParameter<int>("matrix_halo_exchange", "0 - No halo exchange on lower levels, 1 - just diagonal values, 2 - full", 0);
    std::vector<ColoringType> coloring_values;
    coloring_values.push_back(FIRST);
    coloring_values.push_back(SYNC_COLORS);
    coloring_values.push_back(LAST);
    AMG_Config::registerParameter<ColoringType>("boundary_coloring", "handling of boundary coloring for ILU solvers <ColoringType>", SYNC_COLORS, coloring_values);
    AMG_Config::registerParameter<ColoringType>("halo_coloring", "handling of halo coloring for ILU solvers <ColoringType>", LAST, coloring_values);
    AMG_Config::registerParameter<int>("use_sum_stopping_criteria", "Switch to use sum of all nodes rows to decide whether to stop coarsening, this allows some ranks to have 0 rows in classical AMG", 0);
    //Register Data Format Parameters
    AMG_Config::registerParameter<int>("rhs_from_a", "a flag that asks the reader(requires reader support) that if  RHS vector was not found, generate it from the matrix (1: A*e where e=[1,…,1]^T, 0: b=[1,…,1]^T)", 0);
    AMG_Config::registerParameter<int>("complex_conversion", "a flag that asks the reader to convert complex system in the file to the one of the real-valued systems", 0);
    // 0 - no conversion, read as is
    // 1 - K1 formulation
    // 2 - K2 formulation
    // 3 - K3 formulation
    // 4 - K4 formulation
    // 221 - 2x2 K1 block formulation
    // 222 - 2x2 K2 block formulation
    // 223 - 2x2 K3 block formulation
    // 224 - 2x2 K4 block formulation
    // not yet implemented:
    // 14 - K14 formulation
    std::vector<std::string> writer_values;
    writer_values.push_back("matrixmarket");
    writer_values.push_back("binary");
    AMG_Config::registerParameter<std::string>("matrix_writer", "format to write matrix to the disk <matrixmarket|binary>", "matrixmarket", writer_values);
    std::vector<BlockFormat> blockformat_values;
    blockformat_values.push_back(ROW_MAJOR);
    blockformat_values.push_back(COL_MAJOR);
    AMG_Config::registerParameter<BlockFormat>("block_format", "The format of the blocks. ROW_MAJOR: row major format, COL_MAJOR: column major format <ROW_MAJOR>", ROW_MAJOR, blockformat_values);
    AMG_Config::registerParameter<int>("block_convert", "asks the reader to perform conversion to block matrix. <0>: do not perform conversion, <block_dim>: convert to (block_dim)x(block_dim) block matrix", 0);
    //Register Solver/Preconditioner/Smoother Parameters
    std::vector<std::string> solver_values = getAllSolvers();
    AMG_Config::registerParameter<std::string>("solver", "the solving algorithm <AMG|PCG|PCGF|PBICGSTAB|GMRES|FGMRES|JACOBI_L1|BLOCK_JACOBI|GS|MULTICOLOR_GS|MULTICOLOR_ILU|MULTICOLOR_DILU|KACZMARZ|NOSOLVER>", "AMG", solver_values);
    AMG_Config::registerParameter<std::string>("preconditioner", "the preconditioner algorithm <AMG|PCG|PCGF|PBICGSTAB|GMRES|FGMRES|JACOBI_L1|BLOCK_JACOBI|GS|MULTICOLOR_GS|MULTICOLOR_ILU|MULTICOLOR_DILU|NOSOLVER>", "AMG", solver_values);
    AMG_Config::registerParameter<std::string>("coarse_solver", "the solving algorithm <AMG|PCG|PCGF|PBICGSTAB|GMRES|FGMRES|JACOBI_L1|BLOCK_JACOBI|GS|MULTICOLOR_GS|MULTICOLOR_ILU|MULTICOLOR_DILU|NOSOLVER>", "DENSE_LU_SOLVER", solver_values);
    //AMG_Config::registerParameter< std::vector<std::string> >("smoother","the smoothing algorithm <BLOCK_JACOBI>",std::vector<std::string>(1, std::string("BLOCK_JACOBI")));
    AMG_Config::registerParameter<std::string>("smoother", "the smoothing algorithm <BLOCK_JACOBI>", "BLOCK_JACOBI", solver_values);
    //AMG_Config::registerParameter<std::string>("smoother_amg_list","list of smoothers that will be applied to the AMG hierarchy <BLOCK_JACOBI>","BLOCK_JACOBI", solver_values);
    AMG_Config::registerParameter<std::string>("fine_smoother", "the smoothing algorithm <BLOCK_JACOBI>", "BLOCK_JACOBI", solver_values);
    AMG_Config::registerParameter<std::string>("coarse_smoother", "the smoothing algorithm <BLOCK_JACOBI>", "BLOCK_JACOBI", solver_values);
    //[F]GMRES
    AMG_Config::registerParameter<int>("gmres_n_restart", "the number of Krylov vectors used in FGMRES or GMRES solver ", 20);
    AMG_Config::registerParameter<int>("gmres_krylov_dim", "maximum size fo the krylov subspace. Can be smaller than restart, in that case the algorithm minimizes the quasi residual (QGMRES). Set to zero to automatically match the restart <0>", 0);
    //IDR
    AMG_Config::registerParameter<int>("subspace_dim_s", "the number of dimensions of the small system ", 8);
    //DENSE_LU_SOLVER
    AMG_Config::registerParameter<int>("dense_lu_num_rows", "the dense LU solver will be triggered if the matrix size <= dense_lu_num_rows", 128);
    AMG_Config::registerParameter<int>("dense_lu_max_rows", "the dense LU solver will not be triggered if the matrix size >= dense_lu_max_rows > 0 (not used by default)", 0);
    //Richardson's iteration - relaxation parameter
    AMG_Config::registerParameter<double>("relaxation_factor", "the relaxation factor used in a solver", 0.9, 0.0, 2.0);
    //Richardson's iteration - ILU
    AMG_Config::registerParameter<int>("ilu_sparsity_level", "The multicolor_ilu solver sparsity level. 0:ILU0, 1:ILU1, etc <0>", 0);
    //Richardson's iteration - GS
    AMG_Config::registerParameter<int>("symmetric_GS", "Flag to control if GS smoother is symmetric or not <0|1>", 0);
    AMG_Config::registerParameter<int>("jacobi_iters", "the inner iterations for GSINNER", 5);
    AMG_Config::registerParameter<int>("GS_L1_variant", "Flag to control if GS smoother use L1 variant <0|1>", 0);
    //Polynomial
    AMG_Config::registerParameter<int>("kpz_mu", "the constant mu used in the KPZ polynomial smoother", 4);
    AMG_Config::registerParameter<int>("kpz_order", "the order of the KPZ polynomial smoother", 3);
    //Chebyshev polynomial smoother
    AMG_Config::registerParameter<int>("chebyshev_polynomial_order", "the order of the KPZ polynomial smoother", 5);
    AMG_Config::registerParameter<int>("chebyshev_lambda_estimate_mode", "the order of the KPZ polynomial smoother", 0, 0, 2);
    AMG_Config::registerParameter<double>("cheby_max_lambda", "User guess at maximum eigenvalue of preconditioned operator", 1.0, 0.0, 1.0e20);
    AMG_Config::registerParameter<double>("cheby_min_lambda", "User guess at minimum eigenvalue of preconditioned operator", 0.125, 0.0, 1.0e20);
    //Kaczmarz
    AMG_Config::registerParameter<int>("kaczmarz_coloring_needed", "Enforces MC Kaczmarz (0 for naive single warp implementation)", 1);
    //CF-Jacobi smoother
    AMG_Config::registerParameter<int>("cf_smoothing_mode", "Flavour of CF-smoothing. 0=CF for the presmoothing and FC for post smoothing. 1=opposite of 0. Not yet implemented: 2=FCF. 3=CFC", 0);
    //Register AMGLevel Parameters
    std::vector<AlgorithmType> algorithm_values;
    algorithm_values.push_back(CLASSICAL);
    algorithm_values.push_back(AGGREGATION);
    algorithm_values.push_back(ENERGYMIN);
    AMG_Config::registerParameter<AlgorithmType>("algorithm", "the AMG algorithm <CLASSICAL,AGGREGATION,ENERGYMIN>", CLASSICAL, algorithm_values);
    AMG_Config::registerParameter<int>("amg_host_levels_rows", "levels with number of rows below this number will be solved on host. -1 to disable", -1);
    //Register Cycle Parameters
    AMG_Config::registerParameter<std::string>("cycle", "the cycle algorithm <V|W|F|CG|CGF>", "V", getAllCycles());
    AMG_Config::registerParameter<int>("max_levels", "the maximum number of levels", 100);
    AMG_Config::registerParameter<int>("min_fine_rows", "the minimum number of rows in a fine level", 1);
    AMG_Config::registerParameter<int>("min_coarse_rows", "the minimum number of block rows in a level", 2);
    AMG_Config::registerParameter<int>("max_coarse_iters", "the maximum solve iterations at coarsest levels", 100);
    AMG_Config::registerParameter<double>("coarsen_threshold", "Threshold for creating new coarse level", 1.0);
    AMG_Config::registerParameter<int>("presweeps", "the number of presmooth iterations", 1);
    AMG_Config::registerParameter<int>("postsweeps", "the number of postsmooth iterations", 1);
    AMG_Config::registerParameter<int>("finest_sweeps", "finest level sweeps number", -1);
    AMG_Config::registerParameter<int>("coarsest_sweeps", "the number of smoothing iterations at the coarsest level <2>", 2);
    AMG_Config::registerParameter<int>("cycle_iters", "the number of CG iterations per outer iteration", 2); //only for CG and CGF cycles
    AMG_Config::registerParameter<int>("structure_reuse_levels", "controls reuse of AMG hierarchy, 0 - everything rebuilds, 1 - reuse coloring on the finest level, 2 - reuse coloring on the next level, etc.", 0);
    std::vector<int> error_scaling_values;
    error_scaling_values.push_back(0);
    error_scaling_values.push_back(2);
    error_scaling_values.push_back(3);
    AMG_Config::registerParameter<int>("error_scaling", "scales the coarse grid correction vector. 0=No scaling, 1=deprecated scaling method, 2=minimize residual in 2-norm, 3=minimize error in A-norm (matrix should be SPD) <0>", 0, error_scaling_values);
    AMG_Config::registerParameter<int>("reuse_scale", "option to reuse the scale for the <x> next iterations. <0>", 0 );
    AMG_Config::registerParameter<int>("scaling_smoother_steps", "how many smoothing steps should be applied to the error before computing the scale. <2>", 2);
    AMG_Config::registerParameter<int>("intensive_smoothing", "drastically increases smoothing iterations number", 0);
    //Register Interpolator Parameters
    //Aggregation (Coarse Generators)
    std::vector<std::string> coarse_gen_values = getAllCoarseGenerators();
    AMG_Config::registerParameter<std::string>("coarseAgenerator", "the method used to compute the Galerkin product in Agg-AMG <LOW_DEG|THRUST|HYBRID>", "LOW_DEG", coarse_gen_values);
    AMG_Config::registerParameter<std::string>("coarseAgenerator_coarse", "the method used to compute the Galerkin product in Agg-AMG  for coarser levels <LOW_DEG|THRUST|HYBRID>", "LOW_DEG", coarse_gen_values);
    //Classical (Interpolators)
    AMG_Config::registerParameter<std::string>("interpolator", "the interpolation algorithm <D1|D2|MULTIPASS>", "D1", getAllInterpolators());
    //Energymin (Interpolators)
    AMG_Config::registerParameter<std::string>("energymin_interpolator", "the energymin interpolation algorithm <EM>", "EM");
    //Energymin (Selectors)
    AMG_Config::registerParameter<std::string>("energymin_selector", "the energymin selection algorithm <CR>", "CR");
    //Register Selector (SIZE_[2|4|8]) Parameters
    std::vector<std::string> classical_selector_values = getClassicalSelectors(), aggregation_selector_values = getAggregationSelectors(), combined_selectors = classical_selector_values;
    combined_selectors.insert(combined_selectors.end(), aggregation_selector_values.begin(), aggregation_selector_values.end());
    AMG_Config::registerParameter<std::string>("selector", "the coarse grid selection algorithm (Classical: <PMIS|AGGRESSIVE_PMIS|HMIS|AGGRESSIVE_HMIS|DUMMY>, Aggregation: <SIZE_2|SIZE_4|SIZE_8|MULTI_PAIRWISE>)", "PMIS", combined_selectors);
    AMG_Config::registerParameter<int>("aggressive_levels", "the number of levels to use aggressive coarsening for (Classical only)", 0);
    AMG_Config::registerParameter<std::string>("aggressive_selector", "the aggressive coarse grid selection algorithm, DEFAULT is same as \"selector\" (Classical only) <PMIS|HMIS|DEFAULT>", "DEFAULT", classical_selector_values);
    AMG_Config::registerParameter<std::string>("aggressive_interpolator", "the interpolation algorithm for aggressive coarsening (Classical only) <MULTIPASS>", "MULTIPASS", classical_selector_values);
    AMG_Config::registerParameter<int>("handshaking_phases", "number of handshaking phases for aggregation step, valid values are 1 or 2 phases <1>", 1);
    AMG_Config::registerParameter<int>("aggregation_edge_weight_component", "The component in the block matrices to use to compute the edge weights in the aggregation procedure <0>", 0);
    AMG_Config::registerParameter<int>("max_matching_iterations", "the maximum number of 'matching' iterations in the size2_selector, size4_selector and size8_selector algorithms <15>", 15);
    AMG_Config::registerParameter<double>("max_unassigned_percentage", "the maximum percentage of vertices that are left unaggregated in first phase of matching algorithms <0.05>", 0.05);
    //Register Selector (MULTI_PAIRWISE) Parameters
    AMG_Config::registerParameter<int>("weight_formula", "choose the weight formula. 0: wij=0.5*(|a_ij|+|aji|)/max(|a_ii|,|a_jj|). 1: wij=-0.5(a_ij/a_ii + a_ji/a_jj) <0>", 0);
    AMG_Config::registerParameter<int>("aggregation_passes", "for selector=MULTI_PAIRWISE: each pass about doubles the size of each aggregate", 3);
    AMG_Config::registerParameter<int>("filter_weights", "for selector=MULTI_PAIRWISE: set to 1 to remove weak edges before building aggregates. <0>", 0);
    AMG_Config::registerParameter<double>("filter_weights_alpha", "for selector=MULTI_PAIRWISE: a weight is considered weak iff w_ij<alpha*sqrt(max{w_ik}*max{w_jl}). alpha has to be in range (0,1) <0.5>", 0.5, 0.0, 1.0);
    AMG_Config::registerParameter<int>("full_ghost_level", "for selector=MULTI_PAIRWISE: if set to 1, the full galerkin operator is used to compute the ghost level, else only the weight matrix is used. <0>", 0);
    AMG_Config::registerParameter<int>("notay_weights", "for selector=MULTI_PAIRWISE: uses weights that are built using the quality measure proposed by Artem Napov and Yvan Notay in 'An AMG method with guaranteed convergence rate'. EXPERIMENTAL!", 0);
    AMG_Config::registerParameter<int>("ghost_offdiag_limit", "for selector=MULTI_PAIRWISE: limits the number of off-diagonals per row in ghost levels. 0 means no limit. <0>", 0);
    AMG_Config::registerParameter<int>("merge_singletons", "for selector=MULTI_PAIRWISE or SIZE_2: if set to 1 singletons will be merged into their strongest connected neighbor aggregate. set to 2 it will merge in a smart but more expensive way.<1>", 1);
    AMG_Config::registerParameter<int>("serial_matching", "for selector=MULTI_PAIRWISE: To study the impact of misaligned matchings we have the option of using a serial matching algorithm instead. <0>", 0);
    AMG_Config::registerParameter<int>("modified_handshake", "for selector=MULTI_PAIRWISE: if set to 1, the modified handshake algorithm will be applied. It usually needs a few more iterations to converge but produces better matchings. <0>", 0);
    //Register Selector (DUMMY) Parameters
    AMG_Config::registerParameter<int>("aggregate_size", "the size of the aggregates when using the DUMMY selector <2>", 2);
    //Register Classical Strength of Connection and Truncation Parameters
    AMG_Config::registerParameter<std::string>("strength", "the strength of connection algorithm <ahat|all>", "AHAT", getAllStrength());
    AMG_Config::registerParameter<double>("strength_threshold", "the strength threshold", 0.25);
    AMG_Config::registerParameter<double>("max_row_sum", "weaken dependencies for rows with row sum > max_row_sum", 1.1);
    AMG_Config::registerParameter<double>("interp_truncation_factor", "truncate interpolation matrix elements < trunc_factor*max_coef_on_row", 1.1);
    AMG_Config::registerParameter<int>("interp_max_elements", "Truncate interpolation matrix to maximum of this # elements per row", -1);
    AMG_Config::registerParameter<int>("affinity_iterations", "# smoothing iterations ", 4);
    AMG_Config::registerParameter<int>("affinity_vectors", "# of test vetors on finest level ", 4);
    //Register Matrix Coloring Parameters
    AMG_Config::registerParameter<int>("coloring_level", "the coloring level. 0: no_coloring, 1: regular coloring, 2: distance-2 coloring, 3: distance-3 coloring, etc ..  <0> (Note: if using a multicolor scheme, coloring level will be set to 1)", 1);
    AMG_Config::registerParameter<int>("reorder_cols_by_color", "option to reorder the matrix columns by color. 0: no reordering, 1: reorder columns by color <0>", 0);
    AMG_Config::registerParameter<int>("insert_diag_while_reordering", "option to insert the diagonal in the CSR format while reordering. 0:NO, 1:YES <0>", 0);
    AMG_Config::registerParameter<std::string>("matrix_coloring_scheme", "the matrix coloring algorithm <MIN_MAX|MIN_MAX_2RING|PARALLEL_GREEDY|ROUND_ROBIN|MULTI_HASH|UNIFORM|GREEDY_MIN_MAX_2RING|SERIAL_GREEDY_BFS|GREEDY_RECOLOR|LOCALLY_DOWNWIND>", "MIN_MAX", getAllMatrixColoring());
    AMG_Config::registerParameter<int>("max_num_hash", "the number of hash tables used by min_max coloring scheme <7>", 7);
    AMG_Config::registerParameter<int>("num_colors", "the number of colors used by round_robin coloring scheme <10>", 10);
    AMG_Config::registerParameter<double>("max_uncolored_percentage", "the maximum percentage of vertices that are not colored properly (expects a value between 0. and 1. where a value of 0. leads to a perfect coloring) <0.15>", 0.15, 0.0, 1.0);
    AMG_Config::registerParameter<int>("initial_color", "initial color used matrix coloring", 0);
    AMG_Config::registerParameter<int>("use_bsrxmv", "uses expert cusparse api for DILU. reorder_level > 1 required <0>", 0);
    AMG_Config::registerParameter<int>("fine_levels", "number of aggregation levels to be processed with 'fine' algorithms. -1 == all levels <-1>", -1);
    AMG_Config::registerParameter<int>("coloring_try_remove_last_colors", "Tries to remove the N last colors in GREEDY_MIN_MAX_2RING, defaults N=0", 0);
    AMG_Config::registerParameter<std::string>("coloring_custom_arg", "Custom coloring parameter for new algorithms in test", "");
    AMG_Config::registerParameter<int>("print_coloring_info", "Prints some information about the coloring. <0>", 0 );
    AMG_Config::registerParameter<int>("weakness_bound", "control min-max-2ring flexibility", std::numeric_limits<int>::max() );
    AMG_Config::registerParameter<int>("late_rejection", "use late rejection in mim-max-2ring", 0 );
    AMG_Config::registerParameter<int>("geometric_dim", "use by uniform coloring algorithm", 2 );
    //Register Sparse Matrix - Sparse Matrix Multiplication (SPMM) Parameters.
    AMG_Config::registerParameter<int>( "spmm_gmem_size", "Deprecated. DO NOT USE IN NEW CONFIG", 1024 );
    AMG_Config::registerParameter<int>( "spmm_no_sort", "Deprecated. DO NOT USE IN NEW CONFIG", 1 );
    AMG_Config::registerParameter<int>( "spmm_verbose", "AMGX will print a lot of information about SPMM. It's time consuming so don't use it in production code.", 0 );
    AMG_Config::registerParameter<int>( "spmm_max_attempts", "the number of SPMM attempts before we switch to Cusparse.", 6 );
    //Register Stopping Criteria Parameters
    AMG_Config::registerParameter<int>("max_iters", "the maximum solve iterations", 100);
    AMG_Config::registerParameter<int>("monitor_residual", "flag that turns on calculation of the residual at every iteration <0|1>", 0);
    AMG_Config::registerParameter<std::string>("convergence", "the convergence tolerance algorithm <ABSOLUTE|RELATIVE_MAX|RELATIVE_INI>", "ABSOLUTE", getAllConvergence());
    std::vector<NormType> norm_values;
    norm_values.push_back(L1);
    norm_values.push_back(L2);
    norm_values.push_back(LMAX);
    AMG_Config::registerParameter<NormType>("norm", "the norm used for convergence testing <L1|L2|LMAX>", L2, norm_values);
    AMG_Config::registerParameter<int>("use_scalar_norm", "a flag that allows to use a scalar norm (as opposed to block norms) when dealing with block matrices (0: use block norm, 1: force use of scalar norm) <0>", 0);
    AMG_Config::registerParameter<double>("tolerance", "the convergence tolerance", 1e-12);
    AMG_Config::registerParameter<double>("alt_rel_tolerance", "alternative convergence relative tolerance for combined criteria", 1e-12);
    //Register Statistics and Reporting Parameters
    AMG_Config::registerParameter<int>("verbosity_level", "verbosity level for output, 3 - custom print-outs <0|1|2|3>", 3);
    AMG_Config::registerParameter<int>("solver_verbose", "The solver will print information about its parameters, <0|1>", 0);
    AMG_Config::registerParameter<int>("print_config", "flag that allows to print the solver configuration <0|1>", 0);
    AMG_Config::registerParameter<int>("print_solve_stats", "flag that allows to print information about the solver convergence <0|1>", 0);
    AMG_Config::registerParameter<int>("print_grid_stats", "flag that allows to print information about the amg hierarchy <0|1>", 0);
    AMG_Config::registerParameter<int>("print_vis_data", "flag that allows to print information about the solver convergence <0|1>", 0);
    AMG_Config::registerParameter<int>("print_aggregation_info", "flag that allows to print additional information about aggregation AMG hierarchy<0|1>", 0);
    AMG_Config::registerParameter<int>("obtain_timings", "flag that cause the solvers to print total setup and solve times <0|1>", 0);
    AMG_Config::registerParameter<int>("store_res_history", "flag that allows to store the residual history of a solver solver <0|1>", 0);
    AMG_Config::registerParameter<int>("convergence_analysis", "number of levels that will be analysed. 0=no analysis, 1=only finest, 2=finest and second finest etc. <0>", 0);
    // Register Matrix scaling parameters
    std::vector<std::string> scaler_values = getAllScalers();
    scaler_values.push_back("NONE");
    AMG_Config::registerParameter<std::string>("scaling", "the matrix scaling algorithm <NONE|BINORMALIZATION|DIAGONAL_SYMMETRIC> used", "NONE", combined_selectors);
}

template <class T_Config, bool complex>
struct registerClasses;

template<class T_Config>
struct registerClasses<T_Config, true>
{
    static void register_it()
    {
        //Register Data Formats
        MatrixIO<T_Config>::registerReader("MatrixMarket", ReadMatrixMarket<T_Config>::readMatrixMarket);
        MatrixIO<T_Config>::registerReader("MatrixNVAMG", ReadMatrixMarket<T_Config>::readMatrixMarketV2);
        MatrixIO<T_Config>::registerReader("NVAMGBinary", ReadNVAMGBinary<T_Config>::read);
        MatrixIO<T_Config>::registerWriter("matrixmarket", MatrixIO<T_Config>::writeSystemMatrixMarket);
        MatrixIO<T_Config>::registerWriter("binary", MatrixIO<T_Config>::writeSystemBinary);
        //Register Solvers
        //AMG
        SolverFactory<T_Config>::registerFactory("AMG", new AlgebraicMultigrid_SolverFactory<T_Config>);
        AMG_LevelFactory<T_Config>::registerFactory(AGGREGATION, new Aggregation_AMG_LevelFactory<T_Config>);
        CycleFactory<T_Config>::registerFactory("V", new V_CycleFactory<T_Config>);
        //Preconditioners
        SolverFactory<T_Config>::registerFactory("BLOCK_JACOBI", new block_jacobi_solver::BlockJacobiSolverFactory<T_Config>);
        SolverFactory<T_Config>::registerFactory("GMRES", new GMRES_SolverFactory<T_Config>);
        //No Solver
        SolverFactory<T_Config>::registerFactory("NOSOLVER", new Dummy_SolverFactory<T_Config>);
        aggregation::SelectorFactory<T_Config>::registerFactory("SIZE_8", new aggregation::size8_selector::Size8SelectorFactory<T_Config>);
        aggregation::CoarseAGeneratorFactory<T_Config>::registerFactory("LOW_DEG", new aggregation::LowDegCoarseAGeneratorFactory<T_Config>);
        //Register Convergence
        ConvergenceFactory<T_Config>::registerFactory("ABSOLUTE", new AbsoluteConvergenceFactory<T_Config>);
        ConvergenceFactory<T_Config>::registerFactory("RELATIVE_INI_CORE", new RelativeIniConvergenceFactory<T_Config>);
        ConvergenceFactory<T_Config>::registerFactory("RELATIVE_MAX_CORE", new RelativeMaxConvergenceFactory<T_Config>);
        ConvergenceFactory<T_Config>::registerFactory("RELATIVE_INI", new RelativeIniConvergenceFactory<T_Config>);
        ConvergenceFactory<T_Config>::registerFactory("RELATIVE_MAX", new RelativeMaxConvergenceFactory<T_Config>);
        ConvergenceFactory<T_Config>::registerFactory("COMBINED_REL_INI_ABS", new RelativeAbsoluteCombinedConvergenceFactory<T_Config>);
    };
};

template<class T_Config>
struct registerClasses<T_Config, false>
{
    // temporary header in order to not instantiate not implemented algorithms on complex data
    static void register_it()
    {
        //Register Data Formats
        MatrixIO<T_Config>::registerReader("MatrixMarket", ReadMatrixMarket<T_Config>::readMatrixMarket);
        MatrixIO<T_Config>::registerReader("MatrixNVAMG", ReadMatrixMarket<T_Config>::readMatrixMarketV2);
        MatrixIO<T_Config>::registerReader("NVAMGBinary", ReadNVAMGBinary<T_Config>::read);
        MatrixIO<T_Config>::registerWriter("matrixmarket", MatrixIO<T_Config>::writeSystemMatrixMarket);
        MatrixIO<T_Config>::registerWriter("binary", MatrixIO<T_Config>::writeSystemBinary);
        //Register Solvers
        //AMG
        SolverFactory<T_Config>::registerFactory("AMG", new AlgebraicMultigrid_SolverFactory<T_Config>);
        //Krylov
        SolverFactory<T_Config>::registerFactory("CG", new CG_SolverFactory<T_Config>); //not exposed
        SolverFactory<T_Config>::registerFactory("PCG", new PCG_SolverFactory<T_Config>);
        SolverFactory<T_Config>::registerFactory("PCGF", new PCGF_SolverFactory<T_Config>);
        SolverFactory<T_Config>::registerFactory("BICGSTAB", new BiCGStab_SolverFactory<T_Config>);
        SolverFactory<T_Config>::registerFactory("PBICGSTAB", new PBiCGStab_SolverFactory<T_Config>);
        SolverFactory<T_Config>::registerFactory("GMRES", new GMRES_SolverFactory<T_Config>);
        SolverFactory<T_Config>::registerFactory("FGMRES", new FGMRES_SolverFactory<T_Config>);
        SolverFactory<T_Config>::registerFactory("IDR", new idr_solver::IDR_SolverFactory<T_Config>); //not exposed
        SolverFactory<T_Config>::registerFactory("IDRMSYNC", new idrmsync_solver::IDRMSYNC_SolverFactory<T_Config>); //not exposed
        //Chebyshev iteration
        SolverFactory<T_Config>::registerFactory("CHEBYSHEV", new Chebyshev_SolverFactory<T_Config>);
        //Preconditioners
        SolverFactory<T_Config>::registerFactory("BLOCK_JACOBI", new block_jacobi_solver::BlockJacobiSolverFactory<T_Config>);
        SolverFactory<T_Config>::registerFactory("JACOBI_L1", new JacobiL1SolverFactory<T_Config>);
        SolverFactory<T_Config>::registerFactory("CF_JACOBI", new cf_jacobi_solver::CFJacobiSolverFactory<T_Config>);
        SolverFactory<T_Config>::registerFactory("GS", new GaussSeidelSolverFactory<T_Config>);
        SolverFactory<T_Config>::registerFactory("FIXCOLOR_GS", new fixcolor_gauss_seidel_solver::FixcolorGaussSeidelSolverFactory<T_Config>); //not exposed
        SolverFactory<T_Config>::registerFactory("MULTICOLOR_GS", new multicolor_gauss_seidel_solver::MulticolorGaussSeidelSolverFactory<T_Config>);
        SolverFactory<T_Config>::registerFactory("MULTICOLOR_ILU", new multicolor_ilu_solver::MulticolorILUSolverFactory<T_Config>);
        SolverFactory<T_Config>::registerFactory("MULTICOLOR_DILU", new multicolor_dilu_solver::MulticolorDILUSolverFactory<T_Config>);
        SolverFactory<T_Config>::registerFactory("POLYNOMIAL", new polynomial_solver::PolynomialSolverFactory<T_Config>); //not exposed
        SolverFactory<T_Config>::registerFactory("KPZ_POLYNOMIAL", new KPZPolynomialSolverFactory<T_Config>); //not exposed
        SolverFactory<T_Config>::registerFactory("CHEBYSHEV_POLY", new  chebyshev_poly_smoother::ChebyshevPolySolverFactory<T_Config>);
        SolverFactory<T_Config>::registerFactory("KACZMARZ", new KaczmarzSolverFactory<T_Config>); //not exposed
        //Dense LU (performed locally on each partition in distributed setting)
        SolverFactory<T_Config>::registerFactory("DENSE_LU_SOLVER", new dense_lu_solver::DenseLUSolverFactory<T_Config>);
        //No Solver
        SolverFactory<T_Config>::registerFactory("NOSOLVER", new Dummy_SolverFactory<T_Config>);
        //Register AMGLevel Types
        AMG_LevelFactory<T_Config>::registerFactory(CLASSICAL, new Classical_AMG_LevelFactory<T_Config>);
        AMG_LevelFactory<T_Config>::registerFactory(AGGREGATION, new Aggregation_AMG_LevelFactory<T_Config>);
        AMG_LevelFactory<T_Config>::registerFactory(ENERGYMIN, new Energymin_AMG_LevelFactory<T_Config>);
        //Register Cycles
        CycleFactory<T_Config>::registerFactory("V", new V_CycleFactory<T_Config>);
        CycleFactory<T_Config>::registerFactory("F", new F_CycleFactory<T_Config>);
        CycleFactory<T_Config>::registerFactory("W", new W_CycleFactory<T_Config>);
        CycleFactory<T_Config>::registerFactory("CG", new CG_CycleFactory<T_Config>);
        CycleFactory<T_Config>::registerFactory("CGF", new CG_Flex_CycleFactory<T_Config>);
        //Register Selectors
        //Aggregation
        aggregation::SelectorFactory<T_Config>::registerFactory("SIZE_2", new aggregation::size2_selector::Size2SelectorFactory<T_Config>);
        aggregation::SelectorFactory<T_Config>::registerFactory("SIZE_4", new aggregation::size4_selector::Size4SelectorFactory<T_Config>);
        aggregation::SelectorFactory<T_Config>::registerFactory("SIZE_8", new aggregation::size8_selector::Size8SelectorFactory<T_Config>);
        aggregation::SelectorFactory<T_Config>::registerFactory("MULTI_PAIRWISE", new aggregation::multi_pairwise::MultiPairwiseSelectorFactory<T_Config>);
        aggregation::SelectorFactory<T_Config>::registerFactory("DUMMY", new aggregation::DUMMY_SelectorFactory<T_Config>); //not exposed
        aggregation::SelectorFactory<T_Config>::registerFactory("GEO", new aggregation::GEO_SelectorFactory<T_Config>); //not exposed
        aggregation::SelectorFactory<T_Config>::registerFactory("PARALLEL_GREEDY_SELECTOR", new aggregation::ParallelGreedySelectorFactory<T_Config>); //not exposed
        //Classical
        classical::SelectorFactory<T_Config>::registerFactory("PMIS", new classical::PMIS_SelectorFactory<T_Config>);
        classical::SelectorFactory<T_Config>::registerFactory("AGGRESSIVE_PMIS", new classical::Aggressive_PMIS_SelectorFactory<T_Config>);
        classical::SelectorFactory<T_Config>::registerFactory("HMIS", new classical::HMIS_SelectorFactory<T_Config>);
        classical::SelectorFactory<T_Config>::registerFactory("AGGRESSIVE_HMIS", new classical::Aggressive_HMIS_SelectorFactory<T_Config>);
        classical::SelectorFactory<T_Config>::registerFactory("DUMMY", new classical::Dummy_SelectorFactory<T_Config>);
        classical::SelectorFactory<T_Config>::registerFactory("CR", new classical::CR_SelectorFactory<T_Config>);
        //Energymin (Selectors)
        //Register Interpolators
        //Aggregation (Coarse Generators)
        aggregation::CoarseAGeneratorFactory<T_Config>::registerFactory("LOW_DEG", new aggregation::LowDegCoarseAGeneratorFactory<T_Config>);
        aggregation::CoarseAGeneratorFactory<T_Config>::registerFactory("HYBRID", new aggregation::HybridCoarseAGeneratorFactory<T_Config>);
        aggregation::CoarseAGeneratorFactory<T_Config>::registerFactory("THRUST", new aggregation::ThrustCoarseAGeneratorFactory<T_Config>);
        //Classical (Interpolators)
        InterpolatorFactory<T_Config>::registerFactory("D1", new Distance1_InterpolatorFactory<T_Config>);
        InterpolatorFactory<T_Config>::registerFactory("D2", new Distance2_InterpolatorFactory<T_Config>);
        InterpolatorFactory<T_Config>::registerFactory("MULTIPASS", new Multipass_InterpolatorFactory<T_Config>);
        //Energymin (Interpolators)
        energymin::InterpolatorFactory<T_Config>::registerFactory("EM", new energymin::EM_InterpolatorFactory<T_Config>);
        //Register Classical Strength
        StrengthFactory<T_Config>::registerFactory("AHAT", new Strength_Ahat_StrengthFactory<T_Config>);
        StrengthFactory<T_Config>::registerFactory("ALL", new Strength_All_StrengthFactory<T_Config>);
        StrengthFactory<T_Config>::registerFactory("AFFINITY", new Strength_Affinity_StrengthFactory<T_Config>);
        //Register MatrixColoring schemes
        MatrixColoringFactory<T_Config>::registerFactory("MIN_MAX", new MinMaxMatrixColoringFactory<T_Config>);
        MatrixColoringFactory<T_Config>::registerFactory("MIN_MAX_2RING", new Min_Max_2Ring_Matrix_Coloring_Factory<T_Config>);
        MatrixColoringFactory<T_Config>::registerFactory("PARALLEL_GREEDY", new Parallel_Greedy_Matrix_Coloring_Factory<T_Config>);
        MatrixColoringFactory<T_Config>::registerFactory("ROUND_ROBIN", new RoundRobinMatrixColoringFactory<T_Config>);
        MatrixColoringFactory<T_Config>::registerFactory("MULTI_HASH", new MultiHashMatrixColoringFactory<T_Config>);
        MatrixColoringFactory<T_Config>::registerFactory("UNIFORM", new UniformMatrixColoringFactory<T_Config>);
        MatrixColoringFactory<T_Config>::registerFactory("GREEDY_MIN_MAX_2RING", new Greedy_Min_Max_2Ring_Matrix_Coloring_Factory<T_Config>);
        MatrixColoringFactory<T_Config>::registerFactory("SERIAL_GREEDY_BFS", new Serial_Greedy_BFS_MatrixColoring_Factory<T_Config>);
        MatrixColoringFactory<T_Config>::registerFactory("GREEDY_RECOLOR", new Greedy_Recolor_MatrixColoring_Factory<T_Config>);
        MatrixColoringFactory<T_Config>::registerFactory("LOCALLY_DOWNWIND", new LocallyDownwindColoringFactory<T_Config>);
        //Register Convergence
        ConvergenceFactory<T_Config>::registerFactory("ABSOLUTE", new AbsoluteConvergenceFactory<T_Config>);
        ConvergenceFactory<T_Config>::registerFactory("RELATIVE_INI_CORE", new RelativeIniConvergenceFactory<T_Config>);
        ConvergenceFactory<T_Config>::registerFactory("RELATIVE_MAX_CORE", new RelativeMaxConvergenceFactory<T_Config>);
        ConvergenceFactory<T_Config>::registerFactory("RELATIVE_INI", new RelativeIniConvergenceFactory<T_Config>);
        ConvergenceFactory<T_Config>::registerFactory("RELATIVE_MAX", new RelativeMaxConvergenceFactory<T_Config>);
        ConvergenceFactory<T_Config>::registerFactory("COMBINED_REL_INI_ABS", new RelativeAbsoluteCombinedConvergenceFactory<T_Config>);
        //Register Scaler
        ScalerFactory<T_Config>::registerFactory("DIAGONAL_SYMMETRIC", new DiagonalSymmetricScalerFactory<T_Config>);
        ScalerFactory<T_Config>::registerFactory("BINORMALIZATION", new BinormalizationScalerFactory<T_Config>);
        ScalerFactory<T_Config>::registerFactory("NBINORMALIZATION", new NBinormalizationScalerFactory<T_Config>);
    };
};

template<class T_Config>
inline void unregisterClasses()
{
    //Unregister Data Formats
    MatrixIO<T_Config>::unregisterReaders();
    MatrixIO<T_Config>::unregisterWriters();
    //Unregister Solvers
    SolverFactory<T_Config>::unregisterFactories( );
    //Unregister AMGLevel types
    AMG_LevelFactory<T_Config>::unregisterFactories( );
    //Unregister Cycles
    CycleFactory<T_Config>::unregisterFactories( );
    //Unregister Selectors
    aggregation::SelectorFactory<T_Config>::unregisterFactories( );
    classical::SelectorFactory<T_Config>::unregisterFactories( );
    energymin::SelectorFactory<T_Config>::unregisterFactories( );
    //Unregister Interpolators and Coarse Generators
    aggregation::CoarseAGeneratorFactory<T_Config>::unregisterFactories( );
    InterpolatorFactory<T_Config>::unregisterFactories( );
    energymin::InterpolatorFactory<T_Config>::unregisterFactories( );
    //Uregister Classical Strength
    StrengthFactory<T_Config>::unregisterFactories( );
    //Unregister MatrixColoring schemes
    MatrixColoringFactory<T_Config>::unregisterFactories( );
    //Unregister Convergence criterion
    ConvergenceFactory<T_Config>::unregisterFactories( );
    //Unregister Scaler
    ScalerFactory<T_Config>::unregisterFactories( );
}

AMGX_ERROR initialize()
{
    AMGX_CPU_PROFILER( "initialize " );
    cudaError_t rc;
    std::stringstream info;
    info << "AMGX version " << __AMGX_BUILD_ID__ << "\n";
    info << "Built on " << __AMGX_BUILD_DATE__ << ", " << __AMGX_BUILD_TIME__ << "\n";
    int driver_version = 0, runtime_version = 0;
    rc = cudaDriverGetVersion(&driver_version);

    if (rc != cudaSuccess)
    {
        info << "Failed while initializing CUDA driver in cudaDriverGetVersion";
        amgx_output(info.str().c_str(), info.str().length());
        return AMGX_ERR_CORE;
    }

    rc = cudaRuntimeGetVersion(&runtime_version);

    if (rc != cudaSuccess)
    {
        info << "Failed while initializing CUDA runtime in cudaRuntimeGetVersion";
        amgx_output(info.str().c_str(), info.str().length());
        return AMGX_ERR_CORE;
    }

    int driver_version_maj = driver_version / 1000;
    int driver_version_min = (driver_version - (driver_version_maj * 1000)) / 10;
    int runtime_version_maj = runtime_version / 1000;
    int runtime_version_min = (runtime_version - (runtime_version_maj * 1000)) / 10;
    info << "Compiled with CUDA Runtime " << runtime_version_maj << "." << runtime_version_min << ", using CUDA driver " << driver_version_maj << "." << driver_version_min << "\n";
    std::stringstream cuda_rt_version;
    cuda_rt_version << runtime_version_maj << "." << runtime_version_min;
#ifdef AMGX_WITH_MPI
    int mpi_initialized = 0;
    MPI_Initialized(&mpi_initialized); // We want to make sure MPI_Init has been called.
    amgx_distributed_output(info.str().c_str(), info.str().length());

    if ( (MPI_ERRHANDLER_NULL == glbMPIErrorHandler))
    {
        registerDefaultMPIErrHandler();
    }

#else
    amgx_output(info.str().c_str(), info.str().length());
#endif

    try
    {
#define AMGX_CASE_LINE(CASE) registerClasses<TemplateMode<CASE>::Type, false>::register_it();
        AMGX_FORALL_BUILDS(AMGX_CASE_LINE)
#undef AMGX_CASE_LINE
#define AMGX_CASE_LINE(CASE) registerClasses<TemplateMode<CASE>::Type, true>::register_it();
        AMGX_FORCOMPLEX_BUILDS(AMGX_CASE_LINE)
#undef AMGX_CASE_LINE
        registerParameters();
    }
    catch (amgx_exception e)
    {
        std::string buf = "Error initializing amgx core: ";
        amgx_output(buf.c_str(), buf.length());
        amgx_output(e.what(), strlen(e.what()));
        return AMGX_ERR_CORE;
    }

    return AMGX_OK;
}

void finalize()
{
    // just in case
    free_resources();
    AMGX_CPU_PROFILER( "finalize " );
#define AMGX_CASE_LINE(CASE) unregisterClasses<TemplateMode<CASE>::Type>();
    AMGX_FORALL_BUILDS(AMGX_CASE_LINE)
    AMGX_FORCOMPLEX_BUILDS(AMGX_CASE_LINE)
#undef AMGX_CASE_LINE
    AMG_Config::unregisterParameters( );
#ifdef AMGX_WITH_MPI
    int mpi_initialized = 0;
    MPI_Initialized(&mpi_initialized); // We want to make sure MPI_Init has been called.

    freeDefaultMPIErrHandler();

#endif
}

} // namespace amgx
