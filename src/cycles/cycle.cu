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

#include <types.h>
#include <gaussian_elimination.h>
#include <norm.h>
#include <blas.h>
#include <cycles/cycle.h>

#include <assert.h>
#include <amgx_types/math.h>

namespace amgx
{
using namespace std;


// Methods to check if Ax=b is easily invertible
template <class T_Config>
bool Cycle_Base<T_Config>::isASolvable(const Matrix<T_Config> &A)
{
    // currently we have to disable this because in the distributed solver we have to launch coarse solver to exchange halos
    // otherwise in some cases nodes with > 1 coarse rows will be stuck waiting on halos from the node with 1 coarse row
    return false;
    /*return (A.get_num_rows()==1) && (A.get_block_dimy()==A.get_block_dimx());*/
}

template<typename ValueTypeA, typename ValueTypeB>
__global__
void direct_1x1_solve(const ValueTypeA *values, const ValueTypeB *rhs, ValueTypeB *x)

{
    // it is always 1 thread because 1x1, but whatever
    const int tid = blockDim.x * blockIdx.x + threadIdx.x;

    if (tid == 0) { x[tid] = rhs[tid] / values[tid]; }
}

template <AMGX_VecPrecision t_vecPrec, AMGX_MatPrecision t_matPrec, AMGX_IndPrecision t_indPrec>
void Cycle<TemplateConfig<AMGX_host, t_vecPrec, t_matPrec, t_indPrec> >::solveExactly_1x1(const Matrix<T_Config> &A, Vector<T_Config> &x, Vector<T_Config> &b)
{
    x[0] = b[0] / A.values[0];
}


template <AMGX_VecPrecision t_vecPrec, AMGX_MatPrecision t_matPrec, AMGX_IndPrecision t_indPrec>
void Cycle<TemplateConfig<AMGX_device, t_vecPrec, t_matPrec, t_indPrec> >::solveExactly_1x1(const Matrix<T_Config> &A, Vector<T_Config> &x, Vector<T_Config> &b)
{
    // only 1 thread needed though
    direct_1x1_solve <<< 1, 32>>>(
        A.values.raw(),
        b.raw(),
        x.raw());
    cudaCheckError();
}

template <AMGX_VecPrecision t_vecPrec, AMGX_MatPrecision t_matPrec, AMGX_IndPrecision t_indPrec>
void Cycle<TemplateConfig<AMGX_host, t_vecPrec, t_matPrec, t_indPrec> >::solveExactly_4x4(const Matrix<T_Config> &A, Vector<T_Config> &x, Vector<T_Config> &b)
{
    GaussianElimination<T_Config>::gaussianElimination(A, x, b);
}


template <AMGX_VecPrecision t_vecPrec, AMGX_MatPrecision t_matPrec, AMGX_IndPrecision t_indPrec>
void Cycle<TemplateConfig<AMGX_device, t_vecPrec, t_matPrec, t_indPrec> >::solveExactly_4x4(const Matrix<T_Config> &A, Vector<T_Config> &x, Vector<T_Config> &b)
{
    GaussianElimination<T_Config>::gaussianElimination(A, x, b);
}

template <class T_Config>
void Cycle_Base<T_Config>::solveExactly(const Matrix<T_Config> &A, VVector &x, VVector &b)
{
    if (A.get_block_size() == 1)
    {
        solveExactly_1x1(A, x, b);
    }
    else if (A.get_block_dimx() == 4 && A.get_block_dimy() == 4)
    {
        solveExactly_4x4(A, x, b);
    }
    else
    {
        FatalError("Unsupported dimension for aggregation amg level", AMGX_ERR_NOT_SUPPORTED_BLOCKSIZE);
    }
}

template<class T_Config>
std::map<std::string, CycleFactory<T_Config>*> &
CycleFactory<T_Config>::getFactories( )
{
    static std::map<std::string, CycleFactory<T_Config>*> s_factories;
    return s_factories;
}

template<class T_Config>
void CycleFactory<T_Config>::registerFactory(string name, CycleFactory<T_Config> *f)
{
    std::map<std::string, CycleFactory<T_Config>*> &factories = getFactories( );
    typename map<string, CycleFactory<T_Config> *>::const_iterator it = factories.find(name);

    if (it != factories.end())
    {
        string error = "CycleFactory '" + name + "' has already been registered\n";
        FatalError(error.c_str(), AMGX_ERR_CORE);
    }

    factories[name] = f;
}

template<class T_Config>
void CycleFactory<T_Config>::unregisterFactory(std::string name)
{
    std::map<std::string, CycleFactory<T_Config>*> &factories = getFactories( );
    typename std::map<std::string, CycleFactory<T_Config> *>::iterator it = factories.find(name);

    if (it == factories.end())
    {
        std::string error = "CycleFactory '" + name + "' has not been registered\n";
        FatalError(error.c_str(), AMGX_ERR_CORE);
    }

    CycleFactory<T_Config> *factory = it->second;
    assert( factory != NULL );
    delete factory;
    factories.erase(it);
}

template<class T_Config>
void CycleFactory<T_Config>::unregisterFactories( )
{
    std::map<std::string, CycleFactory<T_Config>*> &factories = getFactories( );
    typename map<std::string, CycleFactory<T_Config> *>::iterator it = factories.begin( );

    for ( ; it != factories.end( ) ; )
    {
        CycleFactory<T_Config> *factory = it->second;
        assert( factory != NULL );
        it++;
        delete factory;
    }

    factories.clear( );
}

template<class T_Config>
Cycle<T_Config> *CycleFactory<T_Config>::allocate(AMG_Class *amg, AMG_Level<T_Config> *level, VVector &b, VVector &c)
{
    std::map<std::string, CycleFactory<T_Config>*> &factories = getFactories( );
    std::string cycle_name = amg->m_cfg->AMG_Config::getParameter<string>("cycle", amg->m_cfg_scope);
    typename map<string, CycleFactory<T_Config> *>::const_iterator it = factories.find(cycle_name);

    if (it == factories.end())
    {
        string error = "CycleFactory '" + cycle_name + "' has not been registered\n";
        FatalError(error.c_str(), AMGX_ERR_CORE);
    }

    return it->second->create(amg, level, b, c);
};

template<class T_Config>
void CycleFactory<T_Config>::generate(AMG_Class *amg, AMG_Level<T_Config> *level, VVector &b, VVector &c)
{
    Cycle<T_Config> *cycle = allocate( amg, level, b, c );
    delete cycle;
}

/****************************************
 * Explict instantiations
 ***************************************/

template class Cycle_Base<TConfigGeneric_d>;
template class Cycle_Base<TConfigGeneric_h>;

template class Cycle<TConfigGeneric_d>;
template class Cycle<TConfigGeneric_h>;

template class CycleFactory<TConfigGeneric_d>;
template class CycleFactory<TConfigGeneric_h>;

} // namespace amgx
