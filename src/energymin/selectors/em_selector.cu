// SPDX-FileCopyrightText: 2011 - 2025 NVIDIA CORPORATION. All Rights Reserved.
//
// SPDX-License-Identifier: BSD-3-Clause

#include <types.h>
#include <energymin/selectors/em_selector.h>

#include <assert.h>

namespace amgx
{
namespace energymin
{

template<class TConfig>
std::map<std::string, SelectorFactory<TConfig>*> &
SelectorFactory<TConfig>
::getFactories( )
{
    static std::map<std::string, SelectorFactory<TConfig>*> s_factories;
    return s_factories;
}

template<class TConfig>
void SelectorFactory<TConfig>
::registerFactory(std::string name, SelectorFactory<TConfig> *f)
{
    std::map<std::string, SelectorFactory<TConfig>*> &factories = getFactories( );
    typename std::map<std::string, SelectorFactory<TConfig> *>::const_iterator it = factories.find(name);

    if (it != factories.end())
    {
        std::string error = "SelectorFactory '" + name + "' has already been registered\n";
        FatalError(error.c_str(), AMGX_ERR_CORE);
    }

    factories[name] = f;
}

template<class TConfig>
void SelectorFactory<TConfig>
::unregisterFactory(std::string name)
{
    std::map<std::string, SelectorFactory<TConfig>*> &factories = getFactories( );
    typename std::map<std::string, SelectorFactory<TConfig> *>::iterator it = factories.find(name);

    if (it == factories.end())
    {
        std::string error = "SelectorFactory '" + name + "' has not been registered\n";
        FatalError(error.c_str(), AMGX_ERR_CORE);
    }

    SelectorFactory<TConfig> *factory = it->second;
    assert( factory != NULL );
    delete factory;
    factories.erase(it);
}

template<class TConfig>
void SelectorFactory<TConfig>
::unregisterFactories( )
{
    std::map<std::string, SelectorFactory<TConfig>*> &factories = getFactories( );
    typename std::map<std::string, SelectorFactory<TConfig> *>::iterator it = factories.begin( );

    for ( ; it != factories.end( ); )
    {
        SelectorFactory<TConfig> *factory = it->second;
        assert( factory != NULL );
        it++;
        delete factory;
    }

    factories.clear( );
}

template<class TConfig>
Selector<TConfig> *SelectorFactory<TConfig>
::allocate(AMG_Config &cfg, const std::string &cfg_scope)
{
    std::map<std::string, SelectorFactory<TConfig>*> &factories = getFactories( );
    std::string selector = cfg.getParameter<std::string>("energymin_selector", cfg_scope);
    typename std::map<std::string, SelectorFactory<TConfig> *>::const_iterator it = factories.find(selector);

    if (it == factories.end())
    {
        std::string error = "SelectorFactory '" + selector + "' has not been registered\n";
        FatalError(error.c_str(), AMGX_ERR_CORE);
    }

    return it->second->create();
};



/****************************************
 * Explicit instantiations
 ***************************************/
#define AMGX_CASE_LINE(CASE) template class Selector<TemplateMode<CASE>::Type>;
AMGX_FORALL_BUILDS(AMGX_CASE_LINE)
AMGX_FORCOMPLEX_BUILDS(AMGX_CASE_LINE)
#undef AMGX_CASE_LINE

#define AMGX_CASE_LINE(CASE) template class SelectorFactory<TemplateMode<CASE>::Type>;
AMGX_FORALL_BUILDS(AMGX_CASE_LINE)
AMGX_FORCOMPLEX_BUILDS(AMGX_CASE_LINE)
#undef AMGX_CASE_LINE
} // namespace energymin

} // namespace amgx
