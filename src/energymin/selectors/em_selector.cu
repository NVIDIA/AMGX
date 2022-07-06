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
#include <energymin/selectors/em_selector.h>

#include <assert.h>

namespace amgx
{
using namespace std;
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
::registerFactory(string name, SelectorFactory<TConfig> *f)
{
    std::map<std::string, SelectorFactory<TConfig>*> &factories = getFactories( );
    typename map<string, SelectorFactory<TConfig> *>::const_iterator it = factories.find(name);

    if (it != factories.end())
    {
        string error = "SelectorFactory '" + name + "' has already been registered\n";
        FatalError(error.c_str(), AMGX_ERR_CORE);
    }

    factories[name] = f;
}

template<class TConfig>
void SelectorFactory<TConfig>
::unregisterFactory(std::string name)
{
    std::map<std::string, SelectorFactory<TConfig>*> &factories = getFactories( );
    typename map<string, SelectorFactory<TConfig> *>::iterator it = factories.find(name);

    if (it == factories.end())
    {
        string error = "SelectorFactory '" + name + "' has not been registered\n";
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
    typename map<std::string, SelectorFactory<TConfig> *>::iterator it = factories.begin( );

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
    string selector = cfg.getParameter<string>("energymin_selector", cfg_scope);
    typename map<string, SelectorFactory<TConfig> *>::const_iterator it = factories.find(selector);

    if (it == factories.end())
    {
        string error = "SelectorFactory '" + selector + "' has not been registered\n";
        FatalError(error.c_str(), AMGX_ERR_CORE);
    }

    return it->second->create();
};



/****************************************
 * Explicit instantiations
 ***************************************/
template class Selector<TConfigGeneric_d>;
template class Selector<TConfigGeneric_h>;

template class SelectorFactory<TConfigGeneric_d>;
template class SelectorFactory<TConfigGeneric_h>;

} // namespace energymin

} // namespace amgx
