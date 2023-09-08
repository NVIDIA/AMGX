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

#include <assert.h>
#include <convergence/convergence.h>


namespace amgx
{

// Constructor
template<class TConfig>
Convergence<TConfig>::Convergence(AMG_Config &cfg, const std::string &cfg_scope) : m_convergence_name("ConvergenceNameNotSet"), m_cfg(&cfg), m_cfg_scope(cfg_scope)
{
    setTolerance(cfg.getParameter<double>("tolerance", cfg_scope));
}

template<class TConfig>
void Convergence<TConfig>::convergence_init()
{
}

template<class TConfig>
AMGX_STATUS Convergence<TConfig>::convergence_update_and_check(const PODVec_h &nrm, const PODVec_h &nrm_ini)
{
    FatalError("Convergence::converge_and_update_check not implemented for this type", AMGX_ERR_NOT_IMPLEMENTED);
    return AMGX_ST_ERROR;
}



template<class TConfig>
void ConvergenceFactory<TConfig>::registerFactory(std::string name, ConvergenceFactory<TConfig> *f)
{
    std::map<std::string, ConvergenceFactory<TConfig>*> &factories = getFactories( );
    typename std::map<std::string, ConvergenceFactory<TConfig> *>::const_iterator it = factories.find(name);

    if (it != factories.end())
    {
        std::string error = "ConvergenceFactory '" + name + "' has already been registered\n";
        FatalError(error.c_str(), AMGX_ERR_CORE);
    }

    factories[name] = f;
}

template<class TConfig>
void ConvergenceFactory<TConfig>::unregisterFactory(std::string name)
{
    std::map<std::string, ConvergenceFactory<TConfig>*> &factories = getFactories( );
    typename std::map<std::string, ConvergenceFactory<TConfig> *>::iterator it = factories.find(name);

    if (it == factories.end())
    {
        std::string error = "ConvergenceFactory '" + name + "' has not been registered\n";
        FatalError(error.c_str(), AMGX_ERR_CORE);
    }

    ConvergenceFactory<TConfig> *factory = it->second;
    assert( factory != NULL );
    delete factory;
    factories.erase(it);
}

template<class TConfig>
void ConvergenceFactory<TConfig>::unregisterFactories( )
{
    std::map<std::string, ConvergenceFactory<TConfig>*> &factories = getFactories( );
    typename std::map<std::string, ConvergenceFactory<TConfig> *>::iterator it = factories.begin( );

    for ( ; it != factories.end( ) ; )
    {
        ConvergenceFactory<TConfig> *factory = it->second;
        assert( factory != NULL );
        it++;
        delete factory;
    }

    factories.clear( );
}

template<class TConfig>
Convergence<TConfig> *ConvergenceFactory<TConfig>::allocate(AMG_Config &cfg, const std::string &current_scope)
{
    std::map<std::string, ConvergenceFactory<TConfig>*> &factories = getFactories( );
    std::string conv = cfg.getParameter<std::string>("convergence", current_scope);
    typename std::map<std::string, ConvergenceFactory<TConfig> *>::const_iterator it = factories.find(conv);

    if (it == factories.end())
    {
        std::string error = "ConvergenceFactory '" + conv + "' has not been registered\n";
        FatalError(error.c_str(), AMGX_ERR_CORE);
    }

    Convergence<TConfig> *convergence = it->second->create(cfg, current_scope);
    convergence->setName(conv);
    return convergence;
};

template<class TConfig>
std::map<std::string, ConvergenceFactory<TConfig>*> &
ConvergenceFactory<TConfig>::getFactories( )
{
    static std::map<std::string, ConvergenceFactory<TConfig>*> s_factories;
    return s_factories;
}

/****************************************
 * Explict instantiations
 ***************************************/
#define AMGX_CASE_LINE(CASE) template class ConvergenceFactory<TemplateMode<CASE>::Type>;
AMGX_FORALL_BUILDS(AMGX_CASE_LINE)
AMGX_FORCOMPLEX_BUILDS(AMGX_CASE_LINE)
#undef AMGX_CASE_LINE

#define AMGX_CASE_LINE(CASE) template class Convergence<TemplateMode<CASE>::Type>;
AMGX_FORALL_BUILDS(AMGX_CASE_LINE)
AMGX_FORCOMPLEX_BUILDS(AMGX_CASE_LINE)
#undef AMGX_CASE_LINE

} // end namespace

