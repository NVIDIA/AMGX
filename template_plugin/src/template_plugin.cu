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

#include <amg_config.h>
#include <types.h>

#include <solvers/template_solver.h>

namespace amgx
{
namespace template_plugin
{
inline void registerParameters()
{
    //Register parameters here
    AMG_Config::registerParameter<int>("template_parm", "some parameter with a default value of 100", 100);
}

template<class T_Config>
inline void registerClasses()
{
    //Register classes here
    SolverFactory<T_Config>::registerFactory("TEMPLATE_SOLVER", new TemplateSolverFactory<T_Config>);
}

template<class T_Config>
inline void unregisterClasses()
{
    SolverFactory<T_Config>::unregisterFactory("TEMPLATE_SOLVER");
}

AMGX_ERROR initialize()
{
    //Call registration of classes and parameters here
    try
    {
#define AMGX_CASE_LINE(CASE) registerClasses<TemplateMode<CASE>::Type>();
        AMGX_FORALL_BUILDS(AMGX_CASE_LINE)
#undef AMGX_CASE_LINE
        registerParameters();
    }
    catch (amgx_exception e)
    {
        std::string buf = "Error initializing template plugin core: ";
        amgx_output(buf.c_str(), buf.length());
        amgx_output(e.what(), strlen(e.what()) );
        return AMGX_ERR_PLUGIN;
    }

    return AMGX_OK;
}

void finalize()
{
    //cleanup as necessary
#define AMGX_CASE_LINE(CASE) unregisterClasses<TemplateMode<CASE>::Type>();
    AMGX_FORALL_BUILDS(AMGX_CASE_LINE)
#undef AMGX_CASE_LINE
}
} // namespace template_plugin
} // namespace amgx
