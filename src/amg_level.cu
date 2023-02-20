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

#include <amg_level.h>
#include <basic_types.h>
#include <blas.h>
#include <types.h>

#include <solvers/solver.h>
#include <string>

namespace amgx
{

template <class T_Config>
AMG_Level<T_Config>::~AMG_Level()
{
    if (smoother != 0) { delete smoother; }

    if ( next_h ) { delete next_h; }

    if ( next_d ) { delete next_d; }

    if (this->A == this->Aoriginal && this->Aoriginal != NULL) { delete this->A; }
}

template <class T_Config>
AMG_Level<T_Config>::AMG_Level(AMG_Class *amg, ThreadManager *tmng) : smoother(0), amg(amg), next_h(0), next_d(0), init(false), tag(0), is_setup(0), m_amg_level_name("AMGLevelNameNotSet"), m_is_reuse_level(false), m_is_consolidation_level(false), m_next_level_size(0)
{
    Aoriginal = new Matrix<TConfig>();
    A = Aoriginal;
    this->smoother = SolverFactory<T_Config>::allocate(*(amg->m_cfg), amg->m_cfg_scope, "smoother", tmng);
}

template <class T_Config>
void AMG_Level<T_Config>::transfer_from(AMG_Level<T_Config1> *ref_lvl)
{
    // almost deep copy with few exceptions
    this->getA().copy(ref_lvl->getA());
    this->originalRow = ref_lvl->getOriginalRows();
    this->Profile = ref_lvl->Profile;
    this->tag = ref_lvl->tag;
    this->is_setup = ref_lvl->is_setup;
    this->bc = ref_lvl->bc;
    this->r = ref_lvl->r;
    this->xc = ref_lvl->xc;
    this->amg = ref_lvl->amg;
    this->next_d = NULL;
    this->next_h = NULL;
    this->level_id = ref_lvl->level_id;
    this->m_next_level_size = ref_lvl->m_next_level_size;
    this->init = ref_lvl->init;
    this->m_amg_level_name = ref_lvl->m_amg_level_name;
    this->m_is_consolidation_level = ref_lvl->m_is_consolidation_level;
    this->m_is_reuse_level = ref_lvl->m_is_reuse_level;
    this->m_is_root_partition = ref_lvl->m_is_root_partition;
    this->m_destination_part = ref_lvl->m_destination_part;
    this->m_num_parts_to_consolidate = ref_lvl->m_num_parts_to_consolidate;
    this->transfer_level(ref_lvl);
}

template <class T_Config>
void AMG_Level<T_Config>::setup()
{
    ViewType separation_interior = amg->m_cfg->AMG_Config::getParameter<ViewType>("separation_interior", "default");
    ViewType separation_exterior = amg->m_cfg->AMG_Config::getParameter<ViewType>("separation_exterior", "default" );

    if (separation_interior > separation_exterior) { FatalError("Interior separation cannot be wider than the exterior separation", AMGX_ERR_CONFIGURATION); }

    if (separation_interior & INTERIOR == 0) { FatalError("Interior separation must include interior nodes", AMGX_ERR_CONFIGURATION); }

    this->getA().setExteriorView(separation_exterior);
    int offset, size;
    this->getA().getOffsetAndSizeForView(separation_exterior, &offset, &size);

    if (!this->getA().isLatencyHidingEnabled(*this->amg->m_cfg))
    {
        this->getA().setInteriorView(separation_exterior);
    }
    else
    {
        this->getA().setInteriorView(separation_interior);
    }

    this->is_setup = 1;
}

template <class T_Config>
void AMG_Level<T_Config>::setup_smoother()
{
    typedef typename TConfig::MemSpace MemorySpace;
    (*this).Profile.tic("SmootherIni");
    // Initialize the smoother
    /*if( this->smoother != NULL )
      delete this->smoother;*/
    smoother->tag = this->tag * 100 + 0;
    ThreadManager *tmng = smoother->get_thread_manager();
#ifdef AMGX_WITH_MPI

    if ( this->getA().is_matrix_distributed() && this->getA().manager != NULL)
    {
        int offset, n;
        this->getA().getOffsetAndSizeForView(FULL, &offset, &n);

        //if ( this->getA().manager->isGlued() && !this->getA().manager->isRootPartition() )  {
        if (!n &&  this->isClassicalAMGLevel())
        {
            // Skip the solve in gluing path by looking at this flag
            // XXXX: actually setup is skipped. Check if solve can/need to be skipped too
            smoother->setGluedSetup(true);
        }
    }

#endif

    // deferred execution: just push work to queue
    if (tmng == NULL)
    {
        smoother->setup(*A, false);
    }
    else
    {
        AsyncTask *func = new AsyncSolverSetupTask<T_Config>(smoother, A);
        tmng->push_work(func, false);
    }

    (*this).Profile.toc("SmootherIni");
}


template <class T_Config>
void AMG_Level<T_Config>::launchCoarseSolver( AMG_Class *amg, VVector &b, VVector &x)
{
    typedef typename TConfig::MemSpace MemorySpace;
    Solver<TConfig> *coarseSolver = amg->getCoarseSolver( MemorySpace( ) );

    if (this->isInitCycle())
    {
        coarseSolver->solve( b, x, true);
    }
    else
    {
        coarseSolver->solve( b, x, false);
    }
}


template <class T_Config>
std::vector<int> AMG_Level<T_Config>::getOriginalRows()
{
    return originalRow;
}

template<class TConfig>
void AMG_LevelFactory<TConfig>::registerFactory(AlgorithmType name, AMG_LevelFactory<TConfig> *f)
{
    std::map<AlgorithmType, AMG_LevelFactory<TConfig>*> &factories = getFactories( );
    typename std::map<AlgorithmType, AMG_LevelFactory<TConfig> *>::const_iterator it = factories.find(name);

    if (it != factories.end())
    {
        std::string error = "AMG_LevelFactory '" + std::string(getString(name)) + "' has already been registered\n";
        FatalError(error.c_str(), AMGX_ERR_CORE);
    }

    factories[name] = f;
}

template<class TConfig>
void AMG_LevelFactory<TConfig>::unregisterFactory(AlgorithmType name)
{
    std::map<AlgorithmType, AMG_LevelFactory<TConfig>*> &factories = getFactories( );
    typename std::map<AlgorithmType, AMG_LevelFactory<TConfig> *>::iterator it = factories.find(name);

    if (it == factories.end())
    {
        std::string error = "AMG_LevelFactory '" + std::string(getString(name)) + "' has not been registered\n";
        FatalError(error.c_str(), AMGX_ERR_CORE);
    }

    AMG_LevelFactory<TConfig> *factory = it->second;
    assert( factory != NULL );
    delete factory;
    factories.erase(it);
}

template<class TConfig>
void AMG_LevelFactory<TConfig>::unregisterFactories( )
{
    std::map<AlgorithmType, AMG_LevelFactory<TConfig>*> &factories = getFactories( );
    typename std::map<AlgorithmType, AMG_LevelFactory<TConfig> *>::iterator it = factories.begin( );

    for ( ; it != factories.end( ) ; )
    {
        AMG_LevelFactory<TConfig> *factory = it->second;
        assert( factory != NULL );
        it++;
        delete factory;
    }

    factories.clear( );
}

template<class TConfig>
AMG_Level<TConfig> *AMG_LevelFactory<TConfig>::allocate(AMG_Class *amg, ThreadManager *tmng)
{
    std::map<AlgorithmType, AMG_LevelFactory<TConfig>*> &factories = getFactories( );
    AlgorithmType amg_level = amg->m_cfg->AMG_Config::getParameter<AlgorithmType>("algorithm", amg->m_cfg_scope);
    typename std::map<AlgorithmType, AMG_LevelFactory<TConfig> *>::const_iterator it = factories.find(amg_level);

    if (it == factories.end())
    {
        std::string error = "AMG_LevelFactory '" + std::string(getString(amg_level)) + "' has not been registered\n";
        FatalError(error.c_str(), AMGX_ERR_CORE);
    }

    return it->second->create(amg, tmng);
};

template<class TConfig>
std::map<AlgorithmType, AMG_LevelFactory<TConfig>*> &
AMG_LevelFactory<TConfig>::getFactories( )
{
    static std::map<AlgorithmType, AMG_LevelFactory<TConfig>*> s_factories;
    return s_factories;
}


/****************************************
 * Explict instantiations
 ***************************************/
template class AMG_LevelFactory<TConfigGeneric_d>;
template class AMG_LevelFactory<TConfigGeneric_h>;

template class AMG_Level<TConfigGeneric_d>;
template class AMG_Level<TConfigGeneric_h>;

} // namespace amgx
