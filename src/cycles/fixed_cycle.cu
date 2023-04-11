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

#include <amg_level.h>
#include <cycles/fixed_cycle.h>
#include <blas.h>
#include <cycles/v_cycle.h>
#include <cycles/w_cycle.h>
#include <cycles/f_cycle.h>
#include <cycles/cg_cycle.h>
#include <cycles/cg_flex_cycle.h>
#include <thrust/inner_product.h>
#include <cycles/convergence_analysis.h>
#include <sstream>
#include <util.h>
#include <distributed/glue.h>

#include <amgx_types/util.h>

namespace amgx
{

template< class T_Config, template<AMGX_VecPrecision t_vecPrec, AMGX_MatPrecision t_matPrec, AMGX_IndPrecision t_indPrec> class CycleDispatcher >
void FixedCycle<T_Config, CycleDispatcher>::cycle( AMG_Class *amg, AMG_Level<T_Config> *level, VVector &b, VVector &x )
{
    AMGX_CPU_COND_MARKER(level->isFinest(), "CYCLE", "Start new cycle");
    typedef typename VVector::value_type ValueType;
    typedef typename TConfig::MemSpace MemorySpace;
    Matrix<T_Config> &A = level->getA();
    Solver<T_Config> *smoother = level->getSmoother();
    VVector &bc = level->getbc();
    bc.set_block_dimx(1);
    bc.set_block_dimy(A.get_block_dimy());
    VVector &xc = level->getxc();
    xc.set_block_dimx(1);
    xc.set_block_dimy(A.get_block_dimx());
    VVector &r = level->getr();
    int levelnum = A.template getParameter <int>("level");
    int *smoothing_direction = nullptr;

    if (!(A.hasParameter("smoothing_direction")) )
    {
        smoothing_direction = new int;
        A.template setParameterPtr <int> ("smoothing_direction", smoothing_direction);
    }
    else
    {
        smoothing_direction = A.template getParameterPtr <int> ("smoothing_direction");
    }

    A.setView(OWNED);

    if (this->isASolvable(A))
    {
        this->solveExactly(A, x, b);
        return;
    }
    else
    {
        //Pre smooth
        level->Profile.tic("Smoother");
        bool xIsZero = false;

        if (level->isInitCycle())
        {
            xIsZero = true;
        }

        *smoothing_direction = 0;
        {
            AMGX_CPU_PROFILER( "FixedCycle::cycle_@presmooth" );
            int n_presweeps;

            if ( level->isCoarsest() && amg->getCoarseSolver(MemorySpace()) != NULL) // Coarsest level, with coarse solver
            {
                n_presweeps = 0;
            }
            else if (level->isCoarsest()) // coarsest level, no coarse solver
            {
                n_presweeps = amg->getNumCoarsestsweeps();
            }
            else if (level->isFinest() && amg->getNumFinestsweeps() != -1)
            {
                n_presweeps = amg->getNumPresweeps() == 0 ? 0 : amg->getNumFinestsweeps();
            }
            else
            {
                n_presweeps = amg->getNumPresweeps();

                if (amg->getNumPresweeps() != 0 && amg->getIntensiveSmoothing())
                {
                    n_presweeps = max(n_presweeps + levelnum - 2, 0);
                }
            }

            if ( n_presweeps > 0 )
            {
                smoother->setTolerance( 0.);
                smoother->set_max_iters( n_presweeps );
                smoother->solve( b, x, xIsZero );
            }
            else if ( xIsZero )
            {
                fill( x, types::util<ValueType>::get_zero());
            }

            level->unsetInitCycle();
        }
        level->Profile.toc("Smoother");

        if ( level->isCoarsest() && amg->getCoarseSolver(MemorySpace()) != NULL)
            // Only one level with coarse solver
        {
            level->launchCoarseSolver( amg, b, x );
        }
        else if (level->isCoarsest()) // Now at coarsest level, performed coarsest_sweeps so return
        {
            return;
        }
        else // Create data necessary for next coarser cycle
        {
            r.set_block_dimy(b.get_block_dimy());
            r.set_block_dimx(1);
            int offset, size;
            A.getOffsetAndSizeForView(OWNED, &offset, &size);
            //compute residual
            level->Profile.tic("ComputeResidual");
            axmb(A, x, b, r, offset, size);
            level->Profile.toc("ComputeResidual");
            //apply restriction
            // in classical the current level is consolidated while in aggregation this is the next one.
            // Hence, in classical, given a level L, if we want to consolidate L+1 vectors (ie coarse vectors of L) we have to look at L+1 flags.
            bool consolidation_flag = false;
            bool isRootPartition_flag = false;

            if (level->isClassicalAMGLevel() && !A.is_matrix_singleGPU())  // In classical consolidation we want to use A.is_matrix_distributed(), this might be an issue when n=1
            {
                consolidation_flag = level->getNextLevel(MemorySpace())->isConsolidationLevel();
                isRootPartition_flag = level->getNextLevel(MemorySpace())->getA().manager->isRootPartition();
            }
            else if (!level->isClassicalAMGLevel() && !A.is_matrix_singleGPU())
            {
                consolidation_flag = level->isConsolidationLevel();
                isRootPartition_flag = A.manager->isRootPartition();
            }

            level->Profile.tic("restrictRes");
            level->restrictResidual(r, bc);
            level->Profile.toc("restrictRes");

            // we have to be very carreful with !A.is_matrix_singleGPU() by A.is_matrix_distributed().
            // In classical consolidation we want to use A.is_matrix_distributed() in order to consolidateVector / unconsolidateVector
            if (!A.is_matrix_singleGPU()  && consolidation_flag)
            {
                level->consolidateVector(bc);
                level->consolidateVector(xc);
            }

            // This should work
            if ( !( !A.is_matrix_singleGPU() && consolidation_flag && !isRootPartition_flag))
            {
                //mark the next level guess for initialization
                level->setNextInitCycle( );
                static const AMGX_VecPrecision vecPrec = T_Config::vecPrec;
                static const AMGX_MatPrecision matPrec = T_Config::matPrec;
                static const AMGX_IndPrecision indPrec = T_Config::indPrec;

                //WARNING: coarse solver might be called inside generateNextCycles routine
                if ( level->isNextCoarsest( ))
                {
                    //if the next level is the coarsest then don't dispatch an entire cycle, instead just launch a single Vfixed cycle.
                    //std::cout << "launching coarsest" << std::endl;
                    level->generateNextCycles( amg, bc, xc, V_CycleDispatcher<vecPrec, matPrec, indPrec>( ) );
                }
                else
                {
                    //solve the next level using the cycle that was passed in
                    level->generateNextCycles( amg, bc, xc, CycleDispatcher<vecPrec, matPrec, indPrec>( ) );
                }
            }

            if (!A.is_matrix_singleGPU() && consolidation_flag)
            {
                level->unconsolidateVector(xc);
            }

            //prolongate correction
            level->prolongateAndApplyCorrection(xc, bc, x, r);
            level->Profile.toc("proCorr");
            //post smooth
            *smoothing_direction = 1;
            level->Profile.tic("Smoother");
            {
                AMGX_CPU_PROFILER( "FixedCycle::cycle_@postmooth" );
                int n_postsweeps;

                if (level->isFinest() && amg->getNumFinestsweeps() != -1)
                {
                    n_postsweeps = amg->getNumPostsweeps() == 0 ? 0 : amg->getNumFinestsweeps();
                }
                else
                {
                    n_postsweeps = amg->getNumPostsweeps();

                    if (amg->getNumPostsweeps() != 0 && amg->getIntensiveSmoothing())
                    {
                        n_postsweeps = max(n_postsweeps + levelnum - 2, 0);
                    }
                }

                if ( amg->m_cfg->AMG_Config::getParameter<int>( "error_scaling", amg->m_cfg_scope ) > 3 )
                {
                    n_postsweeps = 0;
                }

                if ( n_postsweeps > 0 )
                {
                    smoother->set_max_iters( n_postsweeps );
                    smoother->setTolerance( 0.);
                    smoother->solve( b, x, false );
                }
            }
            level->Profile.toc("Smoother");

            if ( (!A.is_matrix_singleGPU()) && (!level->isClassicalAMGLevel()) && consolidation_flag )
            {
                // Note: We need to use the manager/communicator from THIS level
                //       since the manager/communicator for the NEXT level is one for the
                //       reduced set of partitions after consolidation!
                if (!level->isRootPartition())
                {
                    // bc is consolidated, data is sent from non-root to root partition
                    level->getA().manager->getComms()->send_vector_wait_all(bc);
                }
                else
                {
                    // xc is consolidated and then un-consolidated again,
                    // only the MPI send-requests from the latter step need to be waited for 
                    level->getA().manager->getComms()->send_vector_wait_all(xc);
                }
            }

        }
    } //

    AMGX_CPU_COND_MARKER(level->isFinest(), "CYCLE", "End cycle");
}

/****************************************
 * Explict instantiations
 ***************************************/

#define AMGX_CASE_LINE(CASE) template class FixedCycle<TemplateMode<CASE>::Type, V_CycleDispatcher>;
AMGX_FORALL_BUILDS(AMGX_CASE_LINE)
AMGX_FORCOMPLEX_BUILDS(AMGX_CASE_LINE)
#undef AMGX_CASE_LINE

#define AMGX_CASE_LINE(CASE) template class FixedCycle<TemplateMode<CASE>::Type, W_CycleDispatcher>;
AMGX_FORALL_BUILDS(AMGX_CASE_LINE)
AMGX_FORCOMPLEX_BUILDS(AMGX_CASE_LINE)
#undef AMGX_CASE_LINE

#define AMGX_CASE_LINE(CASE) template class FixedCycle<TemplateMode<CASE>::Type, F_CycleDispatcher>;
AMGX_FORALL_BUILDS(AMGX_CASE_LINE)
AMGX_FORCOMPLEX_BUILDS(AMGX_CASE_LINE)
#undef AMGX_CASE_LINE

#define AMGX_CASE_LINE(CASE) template class FixedCycle<TemplateMode<CASE>::Type, CG_CycleDispatcher>;
AMGX_FORALL_BUILDS(AMGX_CASE_LINE)
AMGX_FORCOMPLEX_BUILDS(AMGX_CASE_LINE)
#undef AMGX_CASE_LINE

#define AMGX_CASE_LINE(CASE) template class FixedCycle<TemplateMode<CASE>::Type, CG_Flex_CycleDispatcher>;
AMGX_FORALL_BUILDS(AMGX_CASE_LINE)
AMGX_FORCOMPLEX_BUILDS(AMGX_CASE_LINE)
#undef AMGX_CASE_LINE
} // namespace amgx
