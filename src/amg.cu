// SPDX-FileCopyrightText: 2011 - 2024 NVIDIA CORPORATION. All Rights Reserved.
//
// SPDX-License-Identifier: BSD-3-Clause

#include <util.h>
#include <amgx_timer.h>

#include <amg.h>
#include <basic_types.h>
#include <types.h>
#include <norm.h>

#include <iostream>
#include <iomanip>
#include <blas.h>
#include <multiply.h>
#include <algorithm>

#include <amg_level.h>
#include <amgx_c.h>
#include <distributed/glue.h>

#include <misc.h>
#include <string>
#include <cassert>
#include <csr_multiply.h>
#include <memory_info.h>

#include <thrust/sort.h>
#include <thrust/remove.h>
#include <thrust/unique.h>
#include <thrust/binary_search.h>
#include <thrust/iterator/constant_iterator.h>
namespace amgx
{

template<AMGX_VecPrecision t_vecPrec, AMGX_MatPrecision t_matPrec, AMGX_IndPrecision t_indPrec>
AMG<t_vecPrec, t_matPrec, t_indPrec>
::AMG(AMG_Config &cfg, const std::string &cfg_scope)
    : fine_h(0), fine_d(0), m_cfg(&cfg), m_cfg_scope(cfg_scope),
      ref_count(1), csr_workspace(NULL), d2_workspace(NULL)
{
    cycle_iters = cfg.getParameter<int>("cycle_iters", cfg_scope);
    norm = cfg.getParameter<NormType>("norm", cfg_scope);
    max_levels = cfg.getParameter<int>( "max_levels", cfg_scope );
    coarsen_threshold = cfg.getParameter<double>("coarsen_threshold", cfg_scope);
    min_fine_rows = cfg.getParameter<int>( "min_fine_rows", cfg_scope );
    min_coarse_rows = cfg.getParameter<int>( "min_coarse_rows", cfg_scope);
    m_amg_consolidation_flag = cfg.getParameter<int>("amg_consolidation_flag", cfg_scope);
    m_consolidation_lower_threshold = cfg.getParameter<int>("matrix_consolidation_lower_threshold", cfg_scope);
    m_consolidation_upper_threshold = cfg.getParameter<int>("matrix_consolidation_upper_threshold", cfg_scope);
    m_sum_stopping_criteria = cfg.getParameter<int>("use_sum_stopping_criteria", cfg_scope);
    m_structure_reuse_levels = cfg.getParameter<int>("structure_reuse_levels", cfg_scope);
    m_amg_host_levels_rows = cfg.getParameter<int>("amg_host_levels_rows", cfg_scope);

    if (m_consolidation_upper_threshold <= m_consolidation_lower_threshold)
    {
        FatalError("Error, matrix_consolidation_lower_threshold must be smaller than matrix_consolidation_upper_threshold", AMGX_ERR_CONFIGURATION);
    }

    std::string solverName, new_scope, tmp_scope;
    cfg.getParameter<std::string>( "coarse_solver", solverName, cfg_scope, new_scope );

    if (solverName.compare("NOSOLVER") == 0)
    {
        coarse_solver_d = NULL;
        coarse_solver_h = NULL;
    }
    else
    {
        coarse_solver_d = SolverFactory<TConfig_d>::allocate(cfg, cfg_scope, "coarse_solver");
        coarse_solver_h = SolverFactory<TConfig_h>::allocate(cfg, cfg_scope, "coarse_solver");
    }

    //NOTE:
    //if dense_lu_num_rows=0 then either you are not using dense solver (it was not selected) or the matrix size for it to be used was set to zero
    //if dense_lu_max_rows=0 then either you are not using dense solver or you don't want to cap the maximum matrix size
    m_dense_lu_num_rows = 0;
    m_dense_lu_max_rows = 0;

    if ( solverName == "DENSE_LU_SOLVER" )
    {
        m_dense_lu_num_rows = cfg.getParameter<int>( "dense_lu_num_rows", cfg_scope );
        m_dense_lu_max_rows = cfg.getParameter<int>( "dense_lu_max_rows", cfg_scope );
    }
}

template<AMGX_VecPrecision t_vecPrec, AMGX_MatPrecision t_matPrec, AMGX_IndPrecision t_indPrec>
void AMG<t_vecPrec, t_matPrec, t_indPrec>::allocate_fine_level()
{
    fine_d = AMG_LevelFactory<TConfig_d>::allocate(this, tmng);
    fine_h = AMG_LevelFactory<TConfig_h>::allocate(this, tmng);
}

// Print the settings used by amg solver
template<AMGX_VecPrecision t_vecPrec, AMGX_MatPrecision t_matPrec, AMGX_IndPrecision t_indPrec>
void AMG<t_vecPrec, t_matPrec, t_indPrec>::printSettings() const
{
    std::cout << std::endl;
    std::cout << "AMG solver settings:" << std::endl;
    std::cout << "cycle_iters = " << cycle_iters << std::endl;
    std::cout << "norm = " << getString(norm) << std::endl;
    std::cout << "presweeps = " << getNumPresweeps() << std::endl;
    std::cout << "postsweeps = " << getNumPostsweeps() << std::endl;
    std::cout << "max_levels = " << max_levels << std::endl;
    std::cout << "coarsen_threshold = " << coarsen_threshold << std::endl;
    std::cout << "min_fine_rows = " << min_fine_rows << std::endl;
    std::cout << "min_coarse_rows = " << min_coarse_rows << std::endl;
    std::cout << "coarse_solver_d: " << this->coarse_solver_d->getName()
              << " with scope name " << this->coarse_solver_d->getScope() << std::endl;
    std::cout << "coarse_solver_h: " << this->coarse_solver_h->getName()
              << " with scope name " << this->coarse_solver_h->getScope() << std::endl;
}

template<AMGX_VecPrecision t_vecPrec, AMGX_MatPrecision t_matPrec, AMGX_IndPrecision t_indPrec>
AMG<t_vecPrec, t_matPrec, t_indPrec>::~AMG()
{
    if (fine_d) { delete fine_d; }

    if (fine_h) { delete fine_h; } // Don't delete both since the hierarchies meet at some point !!!

    delete coarse_solver_d;
    delete coarse_solver_h;

    if ( d2_workspace != NULL && d2_workspace != csr_workspace )
    {
        typedef TemplateConfig<AMGX_device, t_vecPrec, t_matPrec, t_indPrec> TConfig_d;
        CSR_Multiply<TConfig_d>::csr_workspace_delete( d2_workspace );
        csr_workspace = NULL;
    }

    if ( csr_workspace != NULL )
    {
        typedef TemplateConfig<AMGX_device, t_vecPrec, t_matPrec, t_indPrec> TConfig_d;
        CSR_Multiply<TConfig_d>::csr_workspace_delete( csr_workspace );
        csr_workspace = NULL;
    }
}

/**********************************************************
 * Setups the AMG system
 *********************************************************/

void analyze_coloring(device_vector_alloc<int> aggregates_d, device_vector_alloc<int> colors_d);

template< AMGX_VecPrecision t_vecPrec, AMGX_MatPrecision t_matPrec, AMGX_IndPrecision t_indPrec >
class AMG_Setup
{
    public:
        template< typename TConfig_hd >
        static
        AMG_Level<TConfig_hd> *setup( AMG<t_vecPrec, t_matPrec, t_indPrec> *amg,
                                      AMG_Level<TConfig_hd> *&level,
                                      int min_rows, bool hybrid )
        {
            typedef typename TConfig_hd::MemSpace MemorySpace;
            typedef TemplateConfig<AMGX_host,  t_vecPrec, t_matPrec, t_indPrec> TConfig_h;
            typedef TemplateConfig<AMGX_device, t_vecPrec, t_matPrec, t_indPrec> TConfig_d;
            typedef typename Matrix<TConfig_h>::IVector  IVector_h;
            typedef typename Matrix<TConfig_d>::IVector  IVector_d;
            typedef typename Matrix<TConfig_h>::VVector  VVector_h;
            typedef typename Matrix<TConfig_d>::VVector  VVector_d;
            typedef typename Matrix<TConfig_h>::MVector  MVector_h;
            typedef typename Matrix<TConfig_d>::MVector  MVector_d;
            typedef typename Matrix<TConfig_hd>::IVector IVector_hd;
            typedef typename Matrix<TConfig_hd>::VVector VVector_hd;
            typedef typename Matrix<TConfig_hd>::MVector MVector_hd;
            typedef typename MatPrecisionMap<t_matPrec>::Type ValueTypeA;
            typedef typename VecPrecisionMap<t_vecPrec>::Type ValueTypeB;
            static const AMGX_MemorySpace other_memspace = MemorySpaceMap<opposite_memspace<TConfig_hd::memSpace>::memspace>::id;
            typedef TemplateConfig<other_memspace, t_vecPrec, t_matPrec, t_indPrec> TConfig1;
            typedef TConfig1 T_Config1;
            MemorySpace memorySpaceTag;
            // The previous level.
            AMG_Level<TConfig_hd> *prev_level = 0L;
            typedef TemplateConfig<AMGX_host, AMGX_vecInt, t_matPrec, t_indPrec> hvector_type;
            typedef Vector<hvector_type> HVector;
            std::vector<HVector> partition_rows(0);
            HVector num_rows(1);
            int64_t num_rows_global;
            num_rows[0] = num_rows_global = level->getNumRows( );
            int min_partition_rows = num_rows[0];

            if (level->getA().is_matrix_distributed())
            {
                level->getA( ).manager->getComms()->global_reduce(partition_rows, num_rows,
                        level->getA( ), level->tag * 100 + 7);
                num_rows_global = 0;

                for (int i = 0; i < partition_rows.size(); i++)
                {
                    min_partition_rows = std::min(partition_rows[i][0], min_partition_rows);
                    num_rows_global += partition_rows[i][0];
                }
            }

            Solver<TConfig_hd> *coarseSolver = amg->getCoarseSolver( MemorySpace() );
            bool coarseSolverExists = coarseSolver != NULL;

            // Build the remaining / all the levels on the CPU. Note: level_h is NULL if all the setup happened on the GPU.
            while (true)
            {
            nvtxRange test("setup_level");

                //Check if you reached the coarsest level (min_partition_rows  is the number of rows in this partition/rank)
                //NOTE: min_rows = min_coarse_rows if async framework is disabled (min_fine_rows =< min_coarse_rows)
                if (amg->num_levels >= amg->max_levels || min_partition_rows <= min_rows)
                {
                    //Check if the user wishes to use DENSE_LU_SOLVER capping the matrix the size, and the matrix size exceeds the maximum allowed
                    //NOTE: if dense_lu_max_rows=0 then either you are not using dense solver or you don't want to cap the maximum matrix size
                    if ((amg->m_dense_lu_max_rows != 0) && (min_partition_rows > amg->m_dense_lu_max_rows))
                    {
                        amg->setCoarseSolver(NULL, MemorySpace());
                        delete coarseSolver;
                        coarseSolver = NULL;
                        coarseSolverExists = false;
                    }

                    //Check if there is no coarse solver, then setup the smoother to solve the coarsest level
                    if (!coarseSolverExists)
                    {
                        level->setup_smoother();
                    }

                    return level;
                }

                // Allocate next level or use existing one
                int reuse_next_level;
                AMG_Level<TConfig_hd> *nextLevel;

                if (!level->getNextLevel(MemorySpace()) || (amg->m_structure_reuse_levels <= amg->num_levels && amg->m_structure_reuse_levels != -1))
                {
                    if (level->getNextLevel(MemorySpace()))
                    {
                        delete level->getNextLevel(MemorySpace());
                    }

                    reuse_next_level = 0;
                    level->setReuseLevel(false);
                    nextLevel = AMG_LevelFactory<TConfig_hd>::allocate(amg, level->getSmoother()->get_thread_manager());
                    level->setNextLevel( nextLevel );
                }
                else
                {
                    // reuse existing next level
                    reuse_next_level = 1;
                    level->setReuseLevel(true);
                    nextLevel = level->getNextLevel(MemorySpace());
                    /* WARNING: we do not recompute prolongation (P) and restriction (R) when we
                                are reusing the level structure (structure_reuse_levels > 0), but
                                we do need to modify an existing coarse matrix Ac=R*A*P.
                                Instead of calling Ac.set_initialized(0) in every path afterwards,
                                we wil call it here. Notice that in the if part of this statement
                                above when the new level is allocated it creates a new matrix which
                                is not initialized by default (see the matrix constructor):
                                AMG_Level_Factory::allocate -> Classical_AMG_LevelFactory::create ->
                                new Classical_AMG_Level -> new AMG_Level -> new Matrix
                                We are just matching this Ac.set_initialized(0) setting here. */
                    Matrix<TConfig_hd> &Ac = nextLevel->getA();
                    Ac.set_initialized(0);
                }

                nextLevel->setLevelIndex( amg->num_levels );
                level->getA().template setParameter<int>("level", amg->num_levels);
                {
                    // only compute aggregates if we can't reuse existing ones
                    if (!reuse_next_level)
                    {
                        level->createCoarseVertices( );
                    }
                }
                //set the amg_level_index for this matrix
                nextLevel->getA().amg_level_index = amg->num_levels;
                int64_t N = num_rows_global * level->getA().get_block_dimy();
                num_rows[0] = num_rows_global = level->getNumCoarseVertices();

                if (level->getA().is_matrix_distributed())
                {
                    level->getA().manager->getComms()->global_reduce( partition_rows, num_rows,
                            level->getA(), level->tag * 100 + 8 );
                    num_rows_global = 0;

                    for (int i = 0; i < partition_rows.size(); i++)
                    {
                        num_rows_global += partition_rows[i][0];
                    }
                }

                // num_rows[0] contains the total number of rows across all partitions
                int64_t nextN = num_rows_global * level->getA().get_block_dimy();

                if (!level->getA().is_matrix_distributed())
                {
                    min_partition_rows = num_rows[0];
                }
                else
                {
                    int num_parts = level->getA().manager->getComms()->get_num_partitions();
                    float avg_size = num_rows_global / num_parts;

                    if (avg_size < amg->m_consolidation_lower_threshold)
                    {
                        if (level->isClassicalAMGLevel())
                        {
                            FatalError("Consolidation with classical path not supported)", AMGX_ERR_NOT_IMPLEMENTED);
                        }

                        int new_num_parts;
                        bool want_neighbors = false;
                        level->getA().manager->computeDestinationPartitions(amg->m_consolidation_upper_threshold,
                                avg_size, num_parts, new_num_parts, want_neighbors);

                        if (new_num_parts != num_parts)
                        {
                            level->setIsConsolidationLevel(true);
                            // Modify partition_rows so that non-consolidated partitions have 0 rows
                            // Root partitions have total number of rows to consolidate
                            IVector_h row_count_part(num_parts, 0);

                            for (int i = 0; i < num_parts; i++)
                            {
                                row_count_part[level->getA().manager->getDestinationPartitions()[i]] += partition_rows[i][0];
                            }

                            for (int i = 0; i < num_parts; i++)
                            {
                                partition_rows[i][0] = row_count_part[i];
                            }
                        }
                    }

                    if (!amg->m_sum_stopping_criteria)
                    {
                        min_partition_rows = INT_MAX;

                        for (int i = 0; i < partition_rows.size(); i++)
                        {
                            // If aggregation AMG, ignore partitions with 0 rows, since those are caused by consolidation
                            // If classical AMG, include all partitions
                            if ( level->isClassicalAMGLevel() || (!(level->isClassicalAMGLevel()) && partition_rows[i][0] != 0))
                            {
                                min_partition_rows = std::min(partition_rows[i][0], min_partition_rows);
                            }
                        }
                    }
                    else
                    {
                        // use sum instead of min
                        min_partition_rows = 0;

                        for (int i = 0; i < partition_rows.size(); i++)
                        {
                            // If aggregation AMG, ignore partitions with 0 rows, since those are caused by consolidation
                            // If classical AMG, include all partitions
                            if ( level->isClassicalAMGLevel() || (!(level->isClassicalAMGLevel()) && partition_rows[i][0] != 0))
                            {
                                min_partition_rows += partition_rows[i][0];
                            }
                        }
                    }
                }

                // stop here if next level size is < min_rows
                if ( nextN <= amg->coarsen_threshold * N && nextN != N && min_partition_rows >= min_rows )
                {
                    level->createCoarseMatrices();
                    // Resize coarse vectors.
                    int nextSize = level->getNextLevelSize();
                    level->getxc( ).resize( nextSize );
                    level->getxc().set_block_dimy(level->getA( ).get_block_dimy());
                    level->getxc().set_block_dimx(1);
                    level->getxc().tag = nextLevel->tag * 100 + 1;
                    level->getbc( ).resize( nextSize );
                    level->getbc().set_block_dimy(level->getA( ).get_block_dimy());
                    level->getbc().set_block_dimx(1);
                    level->getbc().tag = nextLevel->tag * 100 + 0;
                    int size, offset;
                    level->getA().getOffsetAndSizeForView(FULL, &offset, &size);
                    level->getr().resize( size * level->getA( ).get_block_dimy() );
                    level->getr().set_block_dimy(level->getA( ).get_block_dimy());
                    level->getr().set_block_dimx(1);
                    level->getr().tag = nextLevel->tag * 100 + 2;
                }
                else
                {
                    // delete next level that we just created
                    level->deleteNextLevel( memorySpaceTag );
                }

                if (!level->isCoarsest() || !coarseSolverExists)
                {
                    level->setup_smoother();
                }

                if (level->isCoarsest())
                {
                    break;
                }

                // If consolidation level and not root partition, break;
                if (!level->getA().is_matrix_singleGPU() && level->isConsolidationLevel()
                        && !level->getA().manager->isRootPartition())
                {
                    amg->setCoarseSolver(NULL, MemorySpace());
                    delete coarseSolver;
                    coarseSolver = NULL;
                    coarseSolverExists = false;
                    break;
                }

                nextLevel->setup();
                // Move to the next level.
                prev_level = level;
                level = nextLevel;
                // Increment the level counter.
                amg->num_levels++;
            } //end of while(true)

            return prev_level;
        }

        template< typename TConfig_hd >
        static
        AMG_Level<TConfig_hd> *setup_v2( AMG<t_vecPrec, t_matPrec, t_indPrec> *amg,
                                         AMG_Level<TConfig_hd> *&level,
                                         int min_rows, bool hybrid )
        {
            typedef typename TConfig_hd::MemSpace MemorySpace;
            typedef TemplateConfig<AMGX_host,  t_vecPrec, t_matPrec, t_indPrec> TConfig_h;
            typedef TemplateConfig<AMGX_device, t_vecPrec, t_matPrec, t_indPrec> TConfig_d;
            typedef typename Matrix<TConfig_h>::IVector  IVector_h;
            typedef typename Matrix<TConfig_d>::IVector  IVector_d;
            typedef typename Matrix<TConfig_h>::VVector  VVector_h;
            typedef typename Matrix<TConfig_d>::VVector  VVector_d;
            typedef typename Matrix<TConfig_h>::MVector  MVector_h;
            typedef typename Matrix<TConfig_d>::MVector  MVector_d;
            typedef typename Matrix<TConfig_hd>::IVector IVector_hd;
            typedef typename Matrix<TConfig_hd>::VVector VVector_hd;
            typedef typename Matrix<TConfig_hd>::MVector MVector_hd;
            typedef typename MatPrecisionMap<t_matPrec>::Type ValueTypeA;
            typedef typename VecPrecisionMap<t_vecPrec>::Type ValueTypeB;
            MemorySpace memorySpaceTag;
            // The previous level.
            AMG_Level<TConfig_hd> *prev_level = 0L;
            typedef TemplateConfig<AMGX_host, AMGX_vecInt, t_matPrec, t_indPrec> hvector_type;
            typedef Vector<hvector_type> HVector;
            std::vector<HVector> partition_rows(0);
            HVector num_rows(1);
            int64_t num_rows_global;
            num_rows[0] = num_rows_global = level->getNumRows( );
            int min_partition_rows =  INT_MAX, offset = 0, n = 0, num_parts = 1, num_active_parts = 0;
            float avg_size;

            if (level->getA().is_matrix_distributed())
            {
                num_parts = level->getA().manager->getComms()->get_num_partitions();
                level->getA( ).manager->getComms()->global_reduce(partition_rows, num_rows,
                        level->getA( ), level->tag * 100 + 7);
                num_rows_global = 0;

                for (int i = 0; i < partition_rows.size(); i++)
                {
                    if (partition_rows[i][0] != 0)
                    {
                        min_partition_rows = std::min(partition_rows[i][0], min_partition_rows);
                        num_active_parts++;
                    }

                    num_rows_global += partition_rows[i][0];
                }

                if (min_partition_rows == INT_MAX)
                {
                    min_partition_rows = 0;
                }
            }

            IVector_h row_count_part(num_parts, 0);
            Solver<TConfig_hd> *coarseSolver = amg->getCoarseSolver( MemorySpace() );
            bool coarseSolverExists = coarseSolver != NULL;

            // Build the remaining / all the levels on the CPU. Note: level_h is NULL if all the setup happened on the GPU.
            while (true)
            {
                // Glue matrices of the current level
                avg_size = num_rows_global / num_parts;
                // Allow to glue other levels tha 0 if COARSE_CLA_CONSO is true
#if COARSE_CLA_CONSO

                if (level->getA().is_matrix_distributed() && avg_size < amg->m_consolidation_lower_threshold)
                {
#else

                if (level->getA().is_matrix_distributed() && avg_size < amg->m_consolidation_lower_threshold && level->getLevelIndex() == 0)
                {
#endif
                    // Just remove level->getLevelIndex() == 0 in the previous test to allow coarse level consolidation
#ifdef AMGX_WITH_MPI
                    level->getA().manager->setIsGlued(false);
                    int new_num_parts = glue_level(amg, level, num_active_parts);

                    if (new_num_parts && new_num_parts != num_active_parts)
                    {
                        if (level->getA().manager->global_id() == 0)
                        {
                            std::cout << "Level " << level->getLevelIndex() << " has been consolidated : " << num_active_parts << " --> " << new_num_parts << std::endl;
                        }

                        // this is for coarse level consolidation
                        if (level->getLevelIndex() > 0)
                        {
                            level->setIsConsolidationLevel(true);
                        }

                        level->setup();
                        num_active_parts = new_num_parts;
                        // Modify partition_rows so that non-consolidated partitions have 0 rows
                        // Root partitions have total number of rows to consolidate
                        num_rows[0] = level->getNumRows();
                        level->getA().manager->getComms()->global_reduce( partition_rows, num_rows,
                                level->getA(), level->tag * 100 + 33 );
                        // Update some local arrays and variables
                        num_rows_global = 0;

                        for (int i = 0; i < partition_rows.size(); i++)
                        {
                            num_rows_global += partition_rows[i][0];
                        }

                        for (int i = 0; i < num_parts; i++)
                        {
                            row_count_part[level->getA().manager->getDestinationPartitions()[i]] += partition_rows[i][0];
                        }

                        for (int i = 0; i < num_parts; i++)
                        {
                            partition_rows[i][0] = row_count_part[i];
                        }
                    }
                    else
                    {
                        level->getA().manager->setIsGlued(false);
                    }

#endif
                }

                level->getA().getOffsetAndSizeForView(OWNED, &offset, &n);

                if (!n)
                {
                    // no coarse solver for empty matrices?
                    // maybe we can deal with this in classical amg cycle
                    amg->setCoarseSolver(NULL, MemorySpace());
                    delete coarseSolver;
                    coarseSolver = NULL;
                    coarseSolverExists = false;
                }

                //Check if you reached the coarsest level (min_partition_rows  is the number of rows in this partition/rank)
                //NOTE: min_rows = min_coarse_rows if async framework is disabled (min_fine_rows =< min_coarse_rows)
                if (amg->num_levels >= amg->max_levels || min_partition_rows <= min_rows)
                {
#if 0   //AMGX_ASYNCCPU_PROOF_OF_CONCEPT
                    asyncmanager::singleton()->waitall();
#endif

                    //Check if the user wishes to use DENSE_LU_SOLVER capping the matrix the size, and the matrix size exceeds the maximum allowed
                    //NOTE: if dense_lu_max_rows=0 then either you are not using dense solver or you don't want to cap the maximum matrix size
                    if ((amg->m_dense_lu_max_rows != 0) && (min_partition_rows > amg->m_dense_lu_max_rows))
                    {
                        amg->setCoarseSolver(NULL, MemorySpace());
                        delete coarseSolver;
                        coarseSolver = NULL;
                        coarseSolverExists = false;
                    }

                    //Check if there is no coarse solver, then setup the smoother to solve the coarsest level
                    // If n is 0 then the matrix is consolidated so we don't setup the smoother
                    // We always setup the smoother on finest level
                    if (!coarseSolverExists)
                    {
                        level->setup_smoother();
                    }

                    return level;
                }

                // Allocate next level or use existing one
                int reuse_next_level;
                AMG_Level<TConfig_hd> *nextLevel;

                if (!level->getNextLevel(MemorySpace()) || (amg->m_structure_reuse_levels <= amg->num_levels && amg->m_structure_reuse_levels != -1))
                {
                    if (level->getNextLevel(MemorySpace()))
                    {
                        delete level->getNextLevel(MemorySpace());
                    }

                    reuse_next_level = 0;
                    level->setReuseLevel(false);
                    nextLevel = AMG_LevelFactory<TConfig_hd>::allocate(amg, level->getSmoother()->get_thread_manager());
                    level->setNextLevel( nextLevel );
                }
                else
                {
                    // reuse existing next level
                    reuse_next_level = 1;
                    level->setReuseLevel(true);
                    nextLevel = level->getNextLevel(MemorySpace());
                    /* WARNING: we do not recompute prolongation (P) and restriction (R) when we
                                are reusing the level structure (structure_reuse_levels > 0), but
                                we do need to modify an existing coarse matrix Ac=R*A*P.
                                Instead of calling Ac.set_initialized(0) in every path afterwards,
                                we wil call it here. Notice that in the if part of this statement
                                above when the new level is allocated it creates a new matrix which
                                is not initialized by default (see the matrix constructor):
                                AMG_Level_Factory::allocate -> Classical_AMG_LevelFactory::create ->
                                new Classical_AMG_Level -> new AMG_Level -> new Matrix
                                We are just matching this Ac.set_initialized(0) setting here. */
                    Matrix<TConfig_hd> &Ac = nextLevel->getA();
                    Ac.set_initialized(0);
                }

                nextLevel->setLevelIndex( amg->num_levels );
                level->getA().template setParameter<int>("level", amg->num_levels);
#if 0 //AMGX_ASYNCCPU_PROOF_OF_CONCEPT

                if (async_global::singleton()->using_async_coloring)
                {
                    struct task_setupsmoother : public task
                    {
                        AMG_Level<TConfig_hd> *level;
                        bool coarseSolverExists;

                        int profiler_color() {return 0x00ffff;}
                        std::string name() { return "setup_smoother"; }
                        void run()
                        {
                            // Setup smoother unless coarseSolver exists and reached coarsest level
                            if ( !( level->isCoarsest() && coarseSolverExists ) )
                            {
                                level->setup_smoother();
                            }
                        }
                    };
                    task_setupsmoother *task_setupsmoother_ = new task_setupsmoother;
                    task_setupsmoother_->level = level;
                    task_setupsmoother_->coarseSolverExists = coarseSolverExists;
                    // create the aggregates (aggregation) or coarse points (classical)
                    level->createCoarseVertices( );
                    enqueue_async(asyncmanager::singleton()->main_thread_queue(0), task_setupsmoother_);
                }
                else
#endif
                {
                    // only compute aggregates if we can't reuse existing ones
                    if (!reuse_next_level)
                    {
                        level->createCoarseVertices( );
                    }
                }

                //set the amg_level_index for this matrix
                nextLevel->getA().amg_level_index = amg->num_levels;
                int64_t N = num_rows_global * level->getA().get_block_dimy();
                num_rows[0] = num_rows_global = level->getNumCoarseVertices();

                // Do reduction across all partitions
                if (level->getA().is_matrix_distributed())
                {
                    level->getA().manager->getComms()->global_reduce( partition_rows, num_rows,
                            level->getA(), level->tag * 100 + 8 );
                    num_rows_global = 0;

                    for (int i = 0; i < partition_rows.size(); i++)
                    {
                        num_rows_global += partition_rows[i][0];
                    }
                }

                // num_rows[0] contains the total number of rows across all partitions
                int64_t nextN = num_rows_global * level->getA().get_block_dimy();

                if (!level->getA().is_matrix_distributed())
                {
                    min_partition_rows = num_rows[0];
                }
                else
                {
                    // level->setIsConsolidationLevel(true); // coaese root partions exited some time in classical
                    if (!amg->m_sum_stopping_criteria)
                    {
                        min_partition_rows = INT_MAX;

                        for (int i = 0; i < partition_rows.size(); i++)
                        {
                            // Before we did
                            // If aggregation AMG, ignore partitions with 0 rows, since those are caused by consolidation
                            // If classical AMG, include all partitions
                            if (partition_rows[i][0] != 0)
                            {
                                min_partition_rows = std::min(partition_rows[i][0], min_partition_rows);
                            }
                        }

                        // if we exit the previous loop with min_partition_rows == INT_MAX it means all next size are 0
                        if (min_partition_rows == INT_MAX)
                        {
                            min_partition_rows = 0;
                        }
                    }
                    else
                    {
                        // use sum instead of min
                        min_partition_rows = 0;

                        for (int i = 0; i < partition_rows.size(); i++)
                        {
                            // If aggregation AMG, ignore partitions with 0 rows, since those are caused by consolidation
                            // If classical AMG, include all partitions
                            if (partition_rows[i][0] != 0)
                            {
                                min_partition_rows += partition_rows[i][0];
                            }
                        }
                    }
                }

                // stop here if next level size is < min_rows
                if ( nextN <= amg->coarsen_threshold * N && nextN != N && min_partition_rows >= min_rows )
                {
                    level->createCoarseMatrices();
                    // Resize coarse vectors.
                    int nextSize = level->getNextLevelSize();
                    level->getxc( ).resize( nextSize );
                    level->getxc().set_block_dimy(level->getA( ).get_block_dimy());
                    level->getxc().set_block_dimx(1);
                    level->getxc().tag = nextLevel->tag * 100 + 1;
                    level->getbc( ).resize( nextSize );
                    level->getbc().set_block_dimy(level->getA( ).get_block_dimy());
                    level->getbc().set_block_dimx(1);
                    level->getbc().tag = nextLevel->tag * 100 + 0;
                    int size, offset;
                    level->getA().getOffsetAndSizeForView(FULL, &offset, &size);
                    level->getr().resize( size * level->getA( ).get_block_dimy() );
                    level->getr().set_block_dimy(level->getA( ).get_block_dimy());
                    level->getr().set_block_dimx(1);
                    level->getr().tag = nextLevel->tag * 100 + 2;
                }
                else
                {
                    // delete next level that we just created
                    level->deleteNextLevel( memorySpaceTag );
                }

#if 0 //AMGX_ASYNCCPU_PROOF_OF_CONCEPT

                if (async_global::singleton()->using_async_coloring)
                {
                    //cancel the CPU coloring task if the GPU is idle
                    cudaStreamSynchronize(amgx::thrust::global_thread_handle::get_stream());
                    cudaCheckError();
                    enqueue_async(asyncmanager::singleton()->global_parallel_queue, async_global::singleton()->cancel_cpu_coloring_task);
                    //wait for every spawning task
                    asyncmanager::singleton()->waitall();
                }
                else
#endif

                    // If n is 0 then the matrix is consolidated so we don't setup the smoother
                    if (!level->isCoarsest() || (!coarseSolverExists))
                    {
                        level->setup_smoother();
                    }

                if (level->isCoarsest())
                {
                    break;
                }

                // Barrier (might be removed)
                // ******************************************
                if (level->getA().is_matrix_distributed()) { level->getA().manager->getComms()->barrier(); }

                // ******************************************
                nextLevel->setup();
                nextLevel->getA().setResources(level->getA().getResources());
#if 0 //AMGX_ASYNCCPU_PROOF_OF_CONCEPT

                //  color the matrix ASAP
                if (!nextmin_fine_rowsmin_fine_rowsmin_fine_rowsLevel->getA().is_matrix_setup())
                {
                    nextLevel->getA().setupMatrix(nextLevel->getSmoother(), *amg->m_cfg, false);
                }

#endif
                // Move to the next level.
                prev_level = level;
                level = nextLevel;
                // Increment the level counter.
                amg->num_levels++;
            } //end of while(true)

#if 0 //AMGX_ASYNCCPU_PROOF_OF_CONCEPT
            cudaStreamSynchronize(amgx::thrust::global_thread_handle::threadStream[getCurrentThreadId()]);
            cudaCheckError();
            amgx::thrust::global_thread_handle::threadStream[getCurrentThreadId()] = 0;
#endif
            return prev_level;
        }

        template< typename TConfig_hd >
        static
        int glue_level(AMG<t_vecPrec, t_matPrec, t_indPrec> *amg, AMG_Level<TConfig_hd> *&level, int num_active_parts)
        {
#ifdef AMGX_WITH_MPI
            if (level->getA().manager->getComms() != NULL)
            {
                MPI_Comm A_com, temp_com;
                int new_num_parts, n_global, num_parts, avg;
                bool wantneighbors = true;
                A_com = level->getA().manager->getComms()->get_mpi_comm();

                if (level->getA().manager->part_offsets_h.size() == 0)  // create part_offsets_h & part_offsets
                {
                    create_part_offsets(A_com, level->getA());
                }

                n_global = level->getA().manager->part_offsets_h.back();
                num_parts = level->getA().manager->getComms()->get_num_partitions();
                avg =   n_global / num_parts;
                level->getA().manager->computeDestinationPartitions(amg->m_consolidation_upper_threshold,
                        avg, num_parts, new_num_parts, wantneighbors);

                if (new_num_parts != num_active_parts)
                {
                    // Compute consolidation info
                    compute_glue_info(level->getA());
                    // Compute a temporary splited communicator to glue matrices
                    temp_com = compute_glue_matrices_communicator(level->getA());
                    // glue_matrices does the following : unpack --> glue --> upload --> repack
                    glue_matrices(level->getA(), A_com, temp_com);
                    return new_num_parts;
                }
                else
                {
                    return num_active_parts;
                }
            }
            else
            {
                return 0;
            }
#else
            return 0;
#endif
        }

        template< typename TConfig0, AMGX_MemorySpace MemSpace0, AMGX_MemorySpace MemSpace1 >
        static
        void
        setup( AMG<t_vecPrec, t_matPrec, t_indPrec> *amg, Matrix<TConfig0> &A )
        {
            typedef typename TConfig0::template setMemSpace<MemSpace1>::Type TConfig1;
            typedef typename MemorySpaceMap<MemSpace0>::Type MemorySpace0;
            typedef typename MemorySpaceMap<MemSpace1>::Type MemorySpace1;
            MemorySpace0 memorySpaceTag0;
            MemorySpace1 memorySpaceTag1;

            // delete zero level from other memoryspace
            if (amg->getFinestLevel(memorySpaceTag1) != NULL)
            {
                delete amg->getFinestLevel(memorySpaceTag1);
                AMG_Level<TConfig1> *level_0_1 = NULL;
                amg->setFinestLevel(level_0_1);
            }

            int min_fine_rows = amg->min_fine_rows;
            int min_coarse_rows = amg->min_coarse_rows;
            // Make sure the number of fine rows is never smaller than min_coarse_rows.
            min_fine_rows = std::max( min_fine_rows, min_coarse_rows );
            // Reset AMG hierarchy.
            amg->num_levels = 1;
            // Build levels on the first device.
            AMG_Level<TConfig0> *level_0 = amg->getFinestLevel(memorySpaceTag0), *prev_level_0 = 0L;

            // if resetup
            if (level_0->isSetup() && amg->m_structure_reuse_levels == 0)
            {
                delete level_0;
                level_0 = AMG_LevelFactory<TConfig0>::allocate(amg);
                amg->setFinestLevel( level_0 );
            }

            level_0->setA(A);
            level_0->setLevelIndex( 0 );
            level_0->setup();

            if (level_0->isClassicalAMGLevel() && amg->m_amg_consolidation_flag == 1 && level_0->getA().is_matrix_distributed())
            {
#ifdef AMGX_WITH_MPI

                if (amg->m_consolidation_lower_threshold == 0 ) // m_consolidation_lower_threshold is unset
                {
                    int root = 0;
                    int max = 0, min = 0;
                    MPI_Comm comm = level_0->getA().manager->getComms()->get_mpi_comm();

                    if (level_0->getA().manager->global_id() == 0 )
                    {
                        size_t avail, total;
                        cudaMemGetInfo (&avail, &total);
                        size_t used = level_0->bytes(); // Memory used by the finest level.
                        size_t hierarchy = 6 * used; // Estimation of the size of the hierarchy
                        size_t overhead = 1000000000; // 1GB of storage for other AMGX stuff
                        // The Strength factor represents how many time a matrix like the one we localy have can fit into this GPU
                        // This is based on the one we have on the finest level on rank 0 and considering the total hierarchy can be 6x larger
                        double strength = (static_cast<double>(total - overhead)) / hierarchy;

                        //    The sum of memory required by coarse levels should be (approximately) smaller or equal than 6x the memory requiered by the finest level.
                        //    This assume a good load balencing
                        //    We should check when we glue matrices that we are not going out of memory.
                        if (strength > 1.0)
                        {
                            int rows = level_0->getNumRows();
                            max = (strength * rows) / 6; // We divide by 6 because we increase the size of the following coarse levels by increasing the size of the current matrix

                            if (max > 0)
                            {
                                min = max - 1;
                            }
                            else
                            {
                                max = 1;
                                min = 0;
                            }
                        }
                        else
                        {
                            max = 1;
                            min = 0;
                        }
                    }

                    MPI_Bcast( &max, 1, MPI_INT, root, comm );
                    MPI_Bcast( &min, 1, MPI_INT, root, comm );
                    amg->m_consolidation_lower_threshold = min;
                    amg->m_consolidation_upper_threshold = max;
                }

                if (amg->m_consolidation_lower_threshold > 0)
                {
                    prev_level_0 =  setup_v2<TConfig0>( amg, level_0, min_fine_rows, min_fine_rows > min_coarse_rows );    // entering in gluing path
                }
                else
#endif
                {
                    prev_level_0 =  setup<TConfig0>( amg, level_0, min_fine_rows, min_fine_rows > min_coarse_rows ); // no glue because the matrix is too big
                }
            }
            else
            {
                prev_level_0 =  setup<TConfig0>( amg, level_0, min_fine_rows, min_fine_rows > min_coarse_rows ); // usual path / aggregation consolidation path
            }

            // Move to the other memory space if needed.
            if ( min_fine_rows == min_coarse_rows )
            {
                Solver<TConfig0> *coarseSolver = amg->getCoarseSolver( memorySpaceTag0 );

                if ( coarseSolver )
                {
                    coarseSolver->setup( level_0->getA(), false );
                }
            }
            else
            {
                AMG_Level<TConfig1> *level_1 = AMG_LevelFactory<TConfig1>::allocate(amg);
                amg->setFinestLevel( level_1 );
                level_1->getA( ).copy( level_0->getA( ) );
                level_1->setLevelIndex( level_0->getLevelIndex( ) );
                level_1->setup();

                // Make that level the next one in the hierarchy.
                if ( prev_level_0 )
                {
                    prev_level_0->setNextLevel( level_1 );
                    assert( prev_level_0->getNextLevel( memorySpaceTag0 ) == level_0 );
                    prev_level_0->deleteNextLevel( memorySpaceTag0 );
                }

                // Build the hierarchy.
                setup<TConfig1>( amg, level_1, min_coarse_rows, false ); 
                // Build the coarse solver.
                Solver<TConfig1> *coarseSolver = amg->getCoarseSolver( memorySpaceTag1 );

                if ( coarseSolver )
                {
                    coarseSolver->setup( level_1->getA(), false );
                }
            }

            // Used only for device modes without hybrid mode. After reaching level where numrows <= amg_host_levels_rows
            // it creates copy of the hierarchy starting with this level.
            // This is experimental feauture intended to measure scaling of the solve part when coarse levels are on the host.
            if (amg->m_amg_host_levels_rows > 0)
            {
                AMG_Level<TConfig0> *d_cur_lvl = amg->getFinestLevel(memorySpaceTag0);
                AMG_Level<TConfig1> *h_cur_lvl = NULL, *h_prev_lvl = NULL;
                AMG_Level<TConfig0> *last_dev_lvl = NULL;
                AMG_Level<TConfig1> *first_host_lvl = NULL;

                while (d_cur_lvl != NULL)
                {
                    if (d_cur_lvl->getNumRows() <= amg->m_amg_host_levels_rows)
                    {
                        break;
                    }

                    last_dev_lvl = d_cur_lvl;
                    d_cur_lvl = d_cur_lvl->getNextLevel( memorySpaceTag0 );
                }

                if (d_cur_lvl != NULL)
                {
                    while (d_cur_lvl != NULL)
                    {
                        h_cur_lvl = AMG_LevelFactory<TConfig1>::allocate(amg, amg->tmng);
                        h_cur_lvl->transfer_from(d_cur_lvl);
                        h_cur_lvl->setup();

                        if (amg->getCoarseSolver(memorySpaceTag0) != NULL)
                        {
                            //remove coarse solver on the device
                            delete amg->getCoarseSolver(memorySpaceTag0);
                            amg->setCoarseSolver(NULL, memorySpaceTag0);
                            // it should exist for the host, but check nevertheless
                            Solver<TConfig1> *coarseSolver = amg->getCoarseSolver( memorySpaceTag1 );
                            bool coarseSolverExists = coarseSolver != NULL;

                            if (!coarseSolverExists)
                            {
                                FatalError("Need to recrreate coarse solver got the host", AMGX_ERR_NOT_IMPLEMENTED);
                            }
                        }
                        else
                        {
                            h_cur_lvl->setup_smoother();
                        }

                        if (first_host_lvl == NULL)
                        {
                            first_host_lvl = h_cur_lvl;
                        }

                        if (h_prev_lvl != NULL)
                        {
                            h_prev_lvl->setNextLevel(h_cur_lvl);
                        }

                        h_prev_lvl = h_cur_lvl;
                        h_cur_lvl = NULL;
                        d_cur_lvl = d_cur_lvl->getNextLevel(memorySpaceTag0);
                    }

                    // cleanup unnecessary device hierarchy part
                    delete last_dev_lvl->getNextLevel(memorySpaceTag0);
                    // link last device level to the first host level
                    last_dev_lvl->setNextLevel(first_host_lvl);
                    last_dev_lvl->resetNextLevel(memorySpaceTag0);
                    // tell amg that there are host levels
                    amg->setFinestLevel( first_host_lvl );
                }
            }

            MemoryInfo::updateMaxMemoryUsage();
        }
};

/**********************************************************
 * Solves the AMG system
 *********************************************************/
template< class T_Config >
class AMG_Solve
{
        typedef T_Config TConfig;
        static const AMGX_VecPrecision vecPrec = TConfig::vecPrec;
        static const AMGX_MatPrecision matPrec = TConfig::matPrec;
        static const AMGX_IndPrecision indPrec = TConfig::indPrec;
        typedef typename TConfig::MemSpace MemorySpace;

        typedef Matrix<TConfig> Matrix_hd;
        typedef Vector<TConfig> Vector_hd;
        typedef Vector<TemplateConfig<AMGX_host, vecPrec, matPrec, indPrec> > Vector_h;
        typedef T_Config TConfig_hd;
        typedef AMG<vecPrec, matPrec, indPrec> AMG_Class;

    public:

        static void solve_iteration( AMG_Class *amg, Vector_hd &b, Vector_hd &x)
        {
            cudaStreamSynchronize(0);
            cudaCheckError();
            nvtxRange amg_si("amg_solve_iteration");
            MemorySpace memorySpaceTag;
            AMG_Level<TConfig_hd> *fine = amg->getFinestLevel( memorySpaceTag );
            assert(fine != NULL);
            CycleFactory<TConfig>::generate( amg, fine, b, x );
            fine->unsetInitCycle();
            // Note: this sometimes takes too much time on host making GPU idle. 
            // Solve is not that important for memory - main mem usage comes from setup.
            // Disabling this call for now
            //MemoryInfo::updateMaxMemoryUsage();
            cudaStreamSynchronize(0);
            cudaCheckError();
        }

};
// Setup the hierarchy to solve on host/device.
template<AMGX_VecPrecision t_vecPrec, AMGX_MatPrecision t_matPrec, AMGX_IndPrecision t_indPrec>
void AMG<t_vecPrec, t_matPrec, t_indPrec>::setup( Matrix_h &A )
{
    if ( m_dense_lu_num_rows > 0 )
    {
        min_coarse_rows = m_dense_lu_num_rows / A.get_block_dimy();
    }

    // read reuse structure levels option from config in case it has been changed
    // this allows fine control over the reuse of hierarchies if setup/solve is called multiple times
    m_structure_reuse_levels = m_cfg->getParameter<int>("structure_reuse_levels", m_cfg_scope);
    AMG_Setup<t_vecPrec, t_matPrec, t_indPrec>::template setup<TConfig_h, AMGX_host, AMGX_device>( this, A );

    // Don't need the workspace anymore
    if ( d2_workspace != NULL && d2_workspace != csr_workspace )
    {
        typedef TemplateConfig<AMGX_device, t_vecPrec, t_matPrec, t_indPrec> TConfig_d;
        CSR_Multiply<TConfig_d>::csr_workspace_delete( d2_workspace );
        csr_workspace = NULL;
    }

    if ( csr_workspace != NULL )
    {
        typedef TemplateConfig<AMGX_device, t_vecPrec, t_matPrec, t_indPrec> TConfig_d;
        CSR_Multiply<TConfig_d>::csr_workspace_delete( csr_workspace );
        csr_workspace = NULL;
    }
}

template<AMGX_VecPrecision t_vecPrec, AMGX_MatPrecision t_matPrec, AMGX_IndPrecision t_indPrec>
void AMG<t_vecPrec, t_matPrec, t_indPrec>::setup( Matrix_d &A )
{
    if ( m_dense_lu_num_rows > 0 )
    {
        min_coarse_rows = m_dense_lu_num_rows / A.get_block_dimy();
    }

    // read reuse structure levels option from config in case it has been changed
    // this allows fine control over the reuse of hierarchies if setup/solve is called multiple times
    m_structure_reuse_levels = m_cfg->getParameter<int>("structure_reuse_levels", m_cfg_scope);
    AMG_Setup<t_vecPrec, t_matPrec, t_indPrec>::template setup<TConfig_d, AMGX_device, AMGX_host>( this, A );

    // Don't need the workspace anymore
    if ( d2_workspace != NULL && d2_workspace != csr_workspace )
    {
        typedef TemplateConfig<AMGX_device, t_vecPrec, t_matPrec, t_indPrec> TConfig_d;
        CSR_Multiply<TConfig_d>::csr_workspace_delete( d2_workspace );
        csr_workspace = NULL;
    }

    if ( csr_workspace != NULL )
    {
        typedef TemplateConfig<AMGX_device, t_vecPrec, t_matPrec, t_indPrec> TConfig_d;
        CSR_Multiply<TConfig_d>::csr_workspace_delete( csr_workspace );
        csr_workspace = NULL;
    }
}

// Setup the hierarchy to solve on host.
template<AMGX_VecPrecision t_vecPrec, AMGX_MatPrecision t_matPrec, AMGX_IndPrecision t_indPrec>
void AMG<t_vecPrec, t_matPrec, t_indPrec>::setup( AMG_Level<TConfig_h> *level )
{
    AMG_Setup<t_vecPrec, t_matPrec, t_indPrec>::template setup<TConfig_h>( this, level, 2, false );
}

template<AMGX_VecPrecision t_vecPrec, AMGX_MatPrecision t_matPrec, AMGX_IndPrecision t_indPrec >
void AMG<t_vecPrec, t_matPrec, t_indPrec>::setup( AMG_Level<TConfig_d> *level )
{
    AMG_Setup<t_vecPrec, t_matPrec, t_indPrec>::template setup<TConfig_d>( this, level, 2, false );
}

template< AMGX_VecPrecision t_vecPrec, AMGX_MatPrecision t_matPrec, AMGX_IndPrecision t_indPrec >
void
AMG<t_vecPrec, t_matPrec, t_indPrec>::solve_init( Vector_d &b, Vector_d &x, bool xIsZero)
{
    if (xIsZero)
    {
        fine_d->setInitCycle();
    }
}

template< AMGX_VecPrecision t_vecPrec, AMGX_MatPrecision t_matPrec, AMGX_IndPrecision t_indPrec >
void
AMG<t_vecPrec, t_matPrec, t_indPrec>::solve_init( Vector_h &b, Vector_h &x, bool xIsZero)
{
    if (xIsZero)
    {
        fine_h->setInitCycle();
    }
}

template< AMGX_VecPrecision t_vecPrec, AMGX_MatPrecision t_matPrec, AMGX_IndPrecision t_indPrec >
void
AMG<t_vecPrec, t_matPrec, t_indPrec>::solve_iteration( Vector_d &b, Vector_d &x)
{
    AMGX_CPU_PROFILER( "AMG::solve_iteration " );
    AMG_Solve<TConfig_d>::solve_iteration( this, b, x);
}

template< AMGX_VecPrecision t_vecPrec, AMGX_MatPrecision t_matPrec, AMGX_IndPrecision t_indPrec >
void
AMG<t_vecPrec, t_matPrec, t_indPrec>::solve_iteration( Vector_h &b, Vector_h &x)
{
    AMGX_CPU_PROFILER( "AMG::solve_iteration " );
    AMG_Solve<TConfig_h>::solve_iteration( this, b, x);
}


template <AMGX_VecPrecision t_vecPrec, AMGX_MatPrecision t_matPrec, AMGX_IndPrecision t_indPrec>
void AMG<t_vecPrec, t_matPrec, t_indPrec>::getGridStatisticsString(std::stringstream &ss)
{
    AMG_Level<TConfig_d> *level_d = this->fine_d;
    AMG_Level<TConfig_h> *level_h = this->fine_h;
    int64_t total_rows = 0;
    int64_t total_nnz = 0;
    float total_size = 0;
    ss << "AMG Grid:\n";
    ss << "         Number of Levels: " << this->num_levels << std::endl;
    ss << std::setw(15) << "LVL";
    ss << std::setw(13) << "ROWS" 
       << std::setw(18) << "NNZ"
       << std::setw(7) << "PARTS"
       << std::setw(10) << "SPRSTY" 
       << std::setw(15) << "Mem (GB)" << std::endl;
    ss << "        ----------------------------------------------------------------------\n";

    while (level_d != NULL)
    {
        int has_diag = level_d->getA( ).hasProps(DIAG) ? 1 : 0;
        int64_t num_rows = (int)(level_d->getA( ).get_num_rows() * level_d->getA( ).get_block_dimy());
        int64_t nnz = (int)((level_d->getA( ).get_num_nz()
                             + has_diag * level_d->getA( ).get_num_rows()) * level_d->getA( ).get_block_dimy()
                            * level_d->getA( ).get_block_dimx());
        int64_t num_parts = level_d->getA().is_matrix_singleGPU() ? 1 : level_d->getA().manager->getComms()->get_num_partitions();
        float size = level_d->bytes(true) / 1024.0 / 1024 / 1024;

        // If aggregation AMG, skip this if # of neighbors = 0, since we're consolidating
        // If classical AMG, we need to enter here since ranks are allowed to have 0 rows (or no neighbors)
        if ( !level_d->getA().is_matrix_singleGPU() ||
                (level_d->isClassicalAMGLevel() && level_d->getA().is_matrix_distributed()) )
        {
            level_d->getA().manager->global_reduce_sum(&num_rows);
            level_d->getA().manager->global_reduce_sum(&nnz);
            level_d->getA().manager->global_reduce_sum(&size);
        }

        total_rows += num_rows;
        total_nnz += nnz;
        total_size += size;
        double sparsity = nnz / (double) ( num_rows * num_rows);
        ss  << std::setw(12) << level_d->getLevelIndex( ) << "(D)"
            << std::setw(13) << num_rows
            << std::setw(18) << nnz
            << std::setw(7) << num_parts
            << std::setw(10) << std::setprecision(3) << sparsity
            << std::setw(15) << size
            << std::setprecision(6) << std::endl;
        level_d = level_d->getNextLevel( device_memory( ) );
    }

    while (level_h != NULL)
    {
        int has_diag = level_h->getA( ).hasProps(DIAG) ? 1 : 0;
        int64_t num_rows = (int)(level_h->getA( ).get_num_rows() * level_h->getA( ).get_block_dimy());
        int64_t nnz = (int)((level_h->getA( ).get_num_nz()
                             + has_diag * level_h->getA( ).get_num_rows()) * level_h->getA( ).get_block_dimy()
                            * level_h->getA( ).get_block_dimx());
        float size = level_h->bytes(true) / 1024.0 / 1024 / 1024;

        // If aggregation AMG, skip this if # of neighbors = 0, since we're consolidating
        // If classical AMG, we need to enter here since ranks are allowed to have 0 rows (or no neighbors)
        if ( !level_h->getA().is_matrix_singleGPU() ||
                (level_h->isClassicalAMGLevel() && level_h->getA().is_matrix_distributed()) )
        {
            level_h->getA().manager->global_reduce_sum(&num_rows);
            level_h->getA().manager->global_reduce_sum(&nnz);
            level_h->getA().manager->global_reduce_sum(&size);
        }

        total_rows += num_rows;
        total_nnz += nnz;
        total_size += size;
        double sparsity = nnz / (double) ( num_rows * num_rows);
        ss  << std::setw(12) << level_h->getLevelIndex( ) << "(H)"
            << std::setw(13) << num_rows
            << std::setw(18) << nnz
            << std::setw(10) << std::setprecision(3) << sparsity
            << std::setw(15) << size
            << std::setprecision(6) << std::endl;
        level_h = level_h->getNextLevel( host_memory( ) );
    }

    int64_t fine_rows;
    int64_t fine_nnz;

    if (this->fine_h)
    {
        fine_rows = this->fine_h->getA( ).get_num_rows()   * this->fine_h->getA( ).get_block_dimy();
        fine_nnz  = this->fine_h->getA( ).get_block_dimy() * this->fine_h->getA( ).get_block_dimx()
                    * ( this->fine_h->getA( ).get_num_nz()
                        + (this->fine_h->getA( ).hasProps(DIAG) ? this->fine_h->getA( ).get_num_rows() : 0) ) ;

        if (this->fine_h->getA().is_matrix_distributed())
        {
            this->fine_h->getA().manager->global_reduce_sum(&fine_rows);
            this->fine_h->getA().manager->global_reduce_sum(&fine_nnz);
        }
    }
    else
    {
        fine_rows = this->fine_d->getA( ).get_num_rows()   * this->fine_d->getA( ).get_block_dimy() ;
        fine_nnz  = this->fine_d->getA( ).get_block_dimy() * this->fine_d->getA( ).get_block_dimx()
                    * ( this->fine_d->getA( ).get_num_nz()
                        + (this->fine_d->getA( ).hasProps(DIAG) ? this->fine_d->getA( ).get_num_rows() : 0) );

        if (this->fine_d->getA().is_matrix_distributed())
        {
            this->fine_d->getA().manager->global_reduce_sum(&fine_rows);
            this->fine_d->getA().manager->global_reduce_sum(&fine_nnz);
        }
    }

    ss << "         ----------------------------------------------------------------------\n";
    ss << "         Grid Complexity: " << total_rows / (double) fine_rows << std::endl;
    ss << "         Operator Complexity: " << total_nnz / (double) fine_nnz << std::endl;
    ss << "         Total Memory Usage: " << total_size << " GB" << std::endl;
    ss << "         ----------------------------------------------------------------------\n";
}

template <AMGX_VecPrecision t_vecPrec, AMGX_MatPrecision t_matPrec, AMGX_IndPrecision t_indPrec>
void AMG<t_vecPrec, t_matPrec, t_indPrec>::printGridStatistics( )
{
    std::stringstream ss;
    this->getGridStatisticsString(ss);
    amgx_output(ss.str().c_str(), static_cast<int>(ss.str().length()));
}

template <AMGX_VecPrecision t_vecPrec, AMGX_MatPrecision t_matPrec, AMGX_IndPrecision t_indPrec>
void AMG<t_vecPrec, t_matPrec, t_indPrec>::getGridStatisticsString2(std::stringstream &ss)
{
    AMG_Level<TConfig_d> *level_d = this->fine_d;
    AMG_Level<TConfig_h> *level_h = this->fine_h;
    int total_rows = 0;
    int total_nnz = 0;
    float total_size = 0;
    ss << " multigrid levels:\n";

    while (level_d != NULL)
    {
        int has_diag = level_d->getA( ).hasProps(DIAG) ? 1 : 0;
        total_rows += (int)(level_d->getA( ).get_num_rows() * level_d->getA( ).get_block_dimy());
        total_nnz += (int)((level_d->getA( ).get_num_nz() + has_diag * level_d->getA( ).get_num_rows()) * level_d->getA( ).get_block_dimy() * level_d->getA( ).get_block_dimx());
        float size = level_d->bytes() / 1024.0 / 1024 / 1024;
        total_size += size;
        ss << std::setw(5) << level_d->getLevelIndex( ) << " "
           << std::setw(5) << level_d->getA( ).get_num_rows() << std::endl;
        level_d = level_d->getNextLevel( device_memory( ) );
    }

    while (level_h != NULL)
    {
        int has_diag = level_h->getA( ).hasProps(DIAG) ? 1 : 0;
        total_rows += (int)(level_h->getA( ).get_num_rows() * level_h->getA( ).get_block_dimy());
        total_nnz += (int)((level_h->getA( ).get_num_nz() + has_diag * level_h->getA( ).get_num_rows()) * level_h->getA( ).get_block_dimy() * level_h->getA( ).get_block_dimx());
        float size = level_h->bytes() / 1024.0 / 1024 / 1024;
        total_size += size;
        ss << std::setw(5) << level_h->getLevelIndex( ) << " "
           << std::setw(5) << level_h->getA( ).get_num_rows() << std::endl;
        level_h = level_h->getNextLevel( host_memory( ) );
    }
}

template <AMGX_VecPrecision t_vecPrec, AMGX_MatPrecision t_matPrec, AMGX_IndPrecision t_indPrec>
void AMG<t_vecPrec, t_matPrec, t_indPrec>::printGridStatistics2( )
{
    std::stringstream ss;
    this->getGridStatisticsString2(ss);
    amgx_output(ss.str().c_str(), static_cast<int>(ss.str().length()));
}

using std::scientific;
using std::fixed;

// print a line of length l, starting at character s
void printLine(const int l, const int s)
{
    std::stringstream ss;
    ss << std::setw(s) << " ";

    for (int i = 0; i < l; i++)
    {
        ss << "-";
    }

    ss << std::endl;
    amgx_output(ss.str().c_str(), static_cast<int>(ss.str().length()));
}

template <AMGX_VecPrecision t_vecPrec, AMGX_MatPrecision t_matPrec, AMGX_IndPrecision t_indPrec>
void AMG<t_vecPrec, t_matPrec, t_indPrec>::printCoarsePoints()
{
#ifdef DEBUG
    typedef std::vector<int> iVec;
    typedef std::vector<int>::iterator iVecIter;
    ofstream coarsePoints("coarse_points.dat");
    iVec originalRows;
    AMG_Level<TConfig_d> *level_d = fine_d;

    while ( level_d != NULL )
    {
        originalRows = level_d->getOriginalRows();
        level_d = level_d->next_d;

        if (level_d == NULL)
        {
            break;
        }

        coarsePoints << level_d->level_id << " " << level_d->getNumRows() << std::endl;

        for (iVecIter it = originalRows.begin(); it != originalRows.end(); ++it)
        {
            coarsePoints << *it << std::endl;
        }
    }

    AMG_Level<TConfig_h> *level_h = fine_h;

    while ( level_h != NULL )
    {
        originalRows = level_h->getOriginalRows();
        level_h = level_h->next_h;

        if (level_h == NULL)
        {
            break;
        }

        coarsePoints << level_h->level_id << " " << level_h->getNumRows() << std::endl;

        for (iVecIter it = originalRows.begin(); it != originalRows.end(); ++it)
        {
            coarsePoints << *it << std::endl;
        }
    }

    coarsePoints.close();
#endif
}

template <AMGX_VecPrecision t_vecPrec, AMGX_MatPrecision t_matPrec, AMGX_IndPrecision t_indPrec>
void AMG<t_vecPrec, t_matPrec, t_indPrec>::printConnections()
{
#ifdef DEBUG
    ofstream connFile("connections.dat");
    AMG_Level<TConfig_d> *level_d = fine_d;
    Matrix_d ATemp_d;

    while (level_d != NULL)
    {
        connFile << level_d->level_id << " " << level_d->getNumRows() << std::endl;
        ATemp_d = level_d->getA();

        for (int i = 0; i < ATemp_d.get_num_rows(); i++)
        {
            // get the row offset & num rows
            int offset = ATemp_d.row_offsets[i];
            int numEntries = ATemp_d.row_offsets[i + 1] - offset;
            // # of connections is numEntries - 1 (ignoring diagonal)
            // this->numConnections.push_back(numEntries-1);
            connFile << numEntries - 1 << " ";

            // loop over non-zeros and add non-diagonal terms
            for (int j = offset; j < offset + numEntries; j++)
            {
                int columnIndex = ATemp_d.column_indices[j];

                if (i != columnIndex)
                {
                    // this->connections.push_back(columnIndex);
                    connFile << columnIndex << " ";
                }
            }

            connFile << std::endl;
        }

        level_d = level_d->next_d;
    }

    AMG_Level<TConfig_h> *level_h = fine_h;
    Matrix_h ATemp_h;

    while (level_h != NULL)
    {
        connFile << level_h->level_id << " " << level_h->getNumRows() << std::endl;
        ATemp_d = level_h->getA();

        for (int i = 0; i < ATemp_h.get_num_rows(); i++)
        {
            // get the row offset & num rows
            int offset = ATemp_h.row_offsets[i];
            int numEntries = ATemp_h.row_offsets[i + 1] - offset;
            // # of connections is numEntries - 1 (ignoring diagonal)
            // this->numConnections.push_back(numEntries-1);
            connFile << numEntries - 1 << " ";

            // loop over non-zeros and add non-diagonal terms
            for (int j = offset; j < offset + numEntries; j++)
            {
                int columnIndex = ATemp_h.column_indices[j];

                if (i != columnIndex)
                {
                    // this->connections.push_back(columnIndex);
                    connFile << columnIndex << " ";
                }
            }

            connFile << std::endl;
        }

        level_h = level_h->next_h;
    }

#endif
}

/****************************************
 * Explict instantiations
 ***************************************/
// real valued case
template class AMG<AMGX_vecDouble, AMGX_matDouble, AMGX_indInt>;
template class AMG<AMGX_vecFloat, AMGX_matFloat, AMGX_indInt>;
template class AMG<AMGX_vecDouble, AMGX_matFloat, AMGX_indInt>;

// complex valued case
template class AMG<AMGX_vecComplex, AMGX_matComplex, AMGX_indInt>;
template class AMG<AMGX_vecDoubleComplex, AMGX_matComplex, AMGX_indInt>;
template class AMG<AMGX_vecDoubleComplex, AMGX_matDoubleComplex, AMGX_indInt>;

} // namespace amgx

