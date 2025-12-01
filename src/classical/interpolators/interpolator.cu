// SPDX-FileCopyrightText: 2011 - 2025 NVIDIA CORPORATION. All Rights Reserved.
//
// SPDX-License-Identifier: BSD-3-Clause

#include <classical/interpolators/interpolator.h>
#include <basic_types.h>

#include <assert.h>

namespace amgx
{

template<class TConfig>
std::map<std::string, InterpolatorFactory<TConfig>*> &
InterpolatorFactory<TConfig>::getFactories( )
{
    static std::map<std::string, InterpolatorFactory<TConfig>*> s_factories;
    return s_factories;
}

template<class TConfig>
void InterpolatorFactory<TConfig>::registerFactory(std::string name, InterpolatorFactory<TConfig> *f)
{
    std::map<std::string, InterpolatorFactory<TConfig>*> &factories = getFactories( );
    typename std::map<std::string, InterpolatorFactory<TConfig> *>::const_iterator it = factories.find(name);

    if (it != factories.end())
    {
        std::string error = "InterpolatorFactory '" + name + "' has already been registered\n";
        FatalError(error.c_str(), AMGX_ERR_CORE);
    }

    factories[name] = f;
}

template<class TConfig>
void InterpolatorFactory<TConfig>::unregisterFactory(std::string name)
{
    std::map<std::string, InterpolatorFactory<TConfig>*> &factories = getFactories( );
    typename std::map<std::string, InterpolatorFactory<TConfig> *>::iterator it = factories.find(name);

    if (it == factories.end())
    {
        std::string error = "InterpolatorFactory '" + name + "' has not been registered\n";
        FatalError(error.c_str(), AMGX_ERR_CORE);
    }

    InterpolatorFactory<TConfig> *factory = it->second;
    assert( factory != NULL );
    delete factory;
    factories.erase(it);
}

template<class TConfig>
void InterpolatorFactory<TConfig>::unregisterFactories( )
{
    std::map<std::string, InterpolatorFactory<TConfig>*> &factories = getFactories( );
    typename std::map<std::string, InterpolatorFactory<TConfig> *>::iterator it = factories.begin( );

    for ( ; it != factories.end( ) ; )
    {
        InterpolatorFactory<TConfig> *factory = it->second;
        assert( factory != NULL );
        it++;
        delete factory;
    }

    factories.clear( );
}

template<class TConfig>
Interpolator<TConfig> *InterpolatorFactory<TConfig>::allocate(AMG_Config &cfg, const std::string &cfg_scope)
{
    std::map<std::string, InterpolatorFactory<TConfig>*> &factories = getFactories( );
    std::string interpolator = cfg.getParameter<std::string>("interpolator", cfg_scope);
    typename std::map<std::string, InterpolatorFactory<TConfig> *>::const_iterator it = factories.find(interpolator);

    if (it == factories.end())
    {
        std::string error = "InterpolatorFactory '" + interpolator + "' has not been registered\n";
        FatalError(error.c_str(), AMGX_ERR_CORE);
    }

    return it->second->create(cfg, cfg_scope);
};

/****************************************
 * Explict instantiations
 ***************************************/
#define AMGX_CASE_LINE(CASE) template class Interpolator<TemplateMode<CASE>::Type>;
AMGX_FORALL_BUILDS(AMGX_CASE_LINE)
AMGX_FORCOMPLEX_BUILDS(AMGX_CASE_LINE)
#undef AMGX_CASE_LINE

#define AMGX_CASE_LINE(CASE) template class InterpolatorFactory<TemplateMode<CASE>::Type>;
AMGX_FORALL_BUILDS(AMGX_CASE_LINE)
AMGX_FORCOMPLEX_BUILDS(AMGX_CASE_LINE)
#undef AMGX_CASE_LINE
} // namespace amgx
