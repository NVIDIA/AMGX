// SPDX-FileCopyrightText: 2011 - 2024 NVIDIA CORPORATION. All Rights Reserved.
//
// SPDX-License-Identifier: BSD-3-Clause

#include <classical/strength/strength.h>
#include <fstream>
#include <cutil.h>
#include <types.h>

#include <assert.h>

namespace amgx
{
template<class TConfig>
std::map<std::string, StrengthFactory<TConfig>*> &
StrengthFactory<TConfig>::getFactories( )
{
    static std::map<std::string, StrengthFactory<TConfig>*> s_factories;
    return s_factories;
}

template<class TConfig>
void StrengthFactory<TConfig>::registerFactory(std::string name, StrengthFactory<TConfig> *f)
{
    std::map<std::string, StrengthFactory<TConfig>*> &factories = getFactories( );
    typename std::map<std::string, StrengthFactory<TConfig> *>::const_iterator it = factories.find(name);

    if (it != factories.end())
    {
        std::string error = "StrengthFactory '" + name + "' has already been registered\n";
        FatalError(error.c_str(), AMGX_ERR_CORE);
    }

    factories[name] = f;
}

template<class TConfig>
void StrengthFactory<TConfig>::unregisterFactory(std::string name)
{
    std::map<std::string, StrengthFactory<TConfig>*> &factories = getFactories( );
    typename std::map<std::string, StrengthFactory<TConfig> *>::iterator it = factories.find(name);

    if (it == factories.end())
    {
        std::string error = "StrengthFactory '" + name + "' has not been registered\n";
        FatalError(error.c_str(), AMGX_ERR_CORE);
    }

    StrengthFactory<TConfig> *factory = it->second;
    assert( factory != NULL );
    delete factory;
    factories.erase(it);
}

template<class TConfig>
void StrengthFactory<TConfig>::unregisterFactories( )
{
    std::map<std::string, StrengthFactory<TConfig>*> &factories = getFactories( );
    typename std::map<std::string, StrengthFactory<TConfig> *>::iterator it = factories.begin( );

    for ( ; it != factories.end( ) ; )
    {
        StrengthFactory<TConfig> *factory = it->second;
        assert( factory != NULL );
        it++;
        delete factory;
    }

    factories.clear( );
}

template<class TConfig>
Strength<TConfig> *StrengthFactory<TConfig>::allocate(AMG_Config &cfg, const std::string &cfg_scope)
{
    std::map<std::string, StrengthFactory<TConfig>*> &factories = getFactories( );
    std::string strength = cfg.getParameter<std::string>("strength", cfg_scope);
    typename std::map<std::string, StrengthFactory<TConfig> *>::const_iterator it = factories.find(strength);

    if (it == factories.end())
    {
        std::string error = "StrengthFactory '" + strength + "' has not been registered\n";
        FatalError(error.c_str(), AMGX_ERR_CORE);
    }

    return it->second->create(cfg, cfg_scope);
};

/****************************************
 * Explict instantiations
 ***************************************/
#define AMGX_CASE_LINE(CASE) template class Strength<TemplateMode<CASE>::Type>;
AMGX_FORALL_BUILDS(AMGX_CASE_LINE)
AMGX_FORCOMPLEX_BUILDS(AMGX_CASE_LINE)
#undef AMGX_CASE_LINE

#define AMGX_CASE_LINE(CASE) template class StrengthFactory<TemplateMode<CASE>::Type>;
AMGX_FORALL_BUILDS(AMGX_CASE_LINE)
AMGX_FORCOMPLEX_BUILDS(AMGX_CASE_LINE)
#undef AMGX_CASE_LINE

} // namespace amgx
