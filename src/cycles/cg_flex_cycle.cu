// SPDX-FileCopyrightText: 2013 - 2025 NVIDIA CORPORATION. All Rights Reserved.
//
// SPDX-License-Identifier: BSD-3-Clause

#include <cycles/cg_flex_cycle.h>
#include <blas.h>
#include <multiply.h>

#include <amgx_types/util.h>

namespace amgx
{

template<class T_Config>
struct DispatchAuxCGF
{
    static const AMGX_VecPrecision vecPrec = T_Config::vecPrec;
    static const AMGX_MatPrecision matPrec = T_Config::matPrec;
    static const AMGX_IndPrecision indPrec = T_Config::indPrec;
    static void dispatch_aux( AMG<vecPrec, matPrec, indPrec> *amg, AMG_Level<T_Config> *level, Vector<T_Config> &b, Vector<T_Config> &x )
    {
        typedef typename Vector<T_Config>::value_type ValueTypeB;
        int N = (int)b.size();
        //create temperary vectors
        Vector<T_Config> y(N);
        Vector<T_Config> z(N);
        Vector<T_Config> r(N);
        Vector<T_Config> d(N);
        Vector<T_Config> p(N);
        y.tag = 9989 * 100 + 1;
        z.tag = 9989 * 100 + 2;
        r.tag = 9989 * 100 + 3;
        p.tag = 9989 * 100 + 4;
        y.set_block_dimy(level->getA().get_block_dimy());
        y.set_block_dimx(1);
        z.set_block_dimy(level->getA().get_block_dimy());
        z.set_block_dimx(1);
        r.set_block_dimy(level->getA().get_block_dimy());
        r.set_block_dimx(1);
        d.set_block_dimy(level->getA().get_block_dimy());
        d.set_block_dimx(1);
        p.set_block_dimy(level->getA().get_block_dimy());
        p.set_block_dimx(1);

        //TODO account for X being 0's
        //not doing this optimization at the moment
        if (level->isInitCycle())
        {
            fill(x, types::util<ValueTypeB>::get_zero());
            level->unsetInitCycle();
        }

        // y = Ax
        multiply(level->getA(), x, y);
        // r = b - A*x
        axpby(b, y, r, types::util<ValueTypeB>::get_one(), types::util<ValueTypeB>::get_minus_one());
        // z = M*r
        level->setInitCycle();
        CG_Flex_Cycle<T_Config> cycle_init( amg, level, r, z );
        // p = z
        copy(z, p);
        int k = 0;

        while (true)
        {
            // y = Ap
            multiply(level->getA(), p, y);
            // rz = <r^H, z>
            ValueTypeB rz = dotc(r, z);
            // alpha = <r,z>/<y,p>
            ValueTypeB alpha =  rz / dotc(y, p);
            // x = x + alpha * p
            axpy(p, x, alpha);

            if (++k == amg->getCycleIters())
            {
                break;
            }

            //d=r
            copy(r, d);
            // r = r - alpha * y
            axpy(y, r, alpha * types::util<ValueTypeB>::get_minus_one());
            //TODO:  if norm(r)<tolerance break
            //d=r-d
            axpby(r, d, d, types::util<ValueTypeB>::get_one(), types::util<ValueTypeB>::get_minus_one());
            // z = M*r
            level->setInitCycle();
            CG_Flex_Cycle<T_Config> cycle( amg, level, r, z );
            // zd = <z, d>
            ValueTypeB zd = dotc(z, d);
            // beta <- <z_{i+1},d}>/<r,z>
            ValueTypeB beta = zd / rz;
            // p += z + beta*p
            axpby(z, p, p, types::util<ValueTypeB>::get_one(), beta);
        }
    }
};

template<AMGX_VecPrecision t_vecPrec, AMGX_MatPrecision t_matPrec, AMGX_IndPrecision t_indPrec >
void CG_Flex_CycleDispatcher<t_vecPrec, t_matPrec, t_indPrec>::dispatch( AMG_Class *amg, AMG_Level<TConfig_h> *level, Vector<TConfig_h> &b, Vector<TConfig_h> &x ) const
{
    DispatchAuxCGF<TemplateConfig<AMGX_host, t_vecPrec, t_matPrec, t_indPrec> >::dispatch_aux( amg, level, b, x );
}

template<AMGX_VecPrecision t_vecPrec, AMGX_MatPrecision t_matPrec, AMGX_IndPrecision t_indPrec >
void CG_Flex_CycleDispatcher<t_vecPrec, t_matPrec, t_indPrec>::dispatch( AMG_Class *amg, AMG_Level<TConfig_d> *level, Vector<TConfig_d> &b, Vector<TConfig_d> &x ) const
{
    DispatchAuxCGF<TemplateConfig<AMGX_device, t_vecPrec, t_matPrec, t_indPrec> >::dispatch_aux( amg, level, b, x );
}

/****************************************
 * Explict instantiations
 ***************************************/
template class CG_Flex_CycleDispatcher<AMGX_vecDouble, AMGX_matDouble, AMGX_indInt>;
template class CG_Flex_CycleDispatcher<AMGX_vecFloat, AMGX_matFloat, AMGX_indInt>;
template class CG_Flex_CycleDispatcher<AMGX_vecDouble, AMGX_matFloat, AMGX_indInt>;

template class CG_Flex_CycleDispatcher<AMGX_vecComplex, AMGX_matComplex, AMGX_indInt>;
template class CG_Flex_CycleDispatcher<AMGX_vecDoubleComplex, AMGX_matComplex, AMGX_indInt>;
template class CG_Flex_CycleDispatcher<AMGX_vecDoubleComplex, AMGX_matDoubleComplex, AMGX_indInt>;

} // namespace amgx
