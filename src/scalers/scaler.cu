// SPDX-FileCopyrightText: 2011 - 2024 NVIDIA CORPORATION. All Rights Reserved.
//
// SPDX-License-Identifier: BSD-3-Clause

#include <types.h>
#include <scalers/scaler.h>

namespace amgx
{

/*******************************
 * DEVICE SPECIFIC FUNCTIONS
 ******************************/

template <AMGX_VecPrecision t_vecPrec, AMGX_MatPrecision t_matPrec, AMGX_IndPrecision t_indPrec>
void Scaler<TemplateConfig<AMGX_device, t_vecPrec, t_matPrec, t_indPrec> >::setup(Matrix_d &A)
{
}

template <AMGX_VecPrecision t_vecPrec, AMGX_MatPrecision t_matPrec, AMGX_IndPrecision t_indPrec>
void Scaler<TemplateConfig<AMGX_device, t_vecPrec, t_matPrec, t_indPrec> >::scaleMatrix(Matrix_d &A, ScaleDirection scaleOrUnscale)
{
}

template <AMGX_VecPrecision t_vecPrec, AMGX_MatPrecision t_matPrec, AMGX_IndPrecision t_indPrec>
void Scaler<TemplateConfig<AMGX_device, t_vecPrec, t_matPrec, t_indPrec> >::scaleVector(Vector<TConfig_d> &v, ScaleDirection scaleOrUnscale, ScaleSide leftOrRight )
{
}


/*******************************
 * HOST SPECIFIC FUNCTIONS
 ******************************/
template <AMGX_VecPrecision t_vecPrec, AMGX_MatPrecision t_matPrec, AMGX_IndPrecision t_indPrec>
void Scaler<TemplateConfig<AMGX_host, t_vecPrec, t_matPrec, t_indPrec> >::setup(Matrix_h &A)
{
}

template <AMGX_VecPrecision t_vecPrec, AMGX_MatPrecision t_matPrec, AMGX_IndPrecision t_indPrec>
void Scaler<TemplateConfig<AMGX_host, t_vecPrec, t_matPrec, t_indPrec> >::scaleMatrix(Matrix_h &A, ScaleDirection scaleOrUnscale)
{
}

template <AMGX_VecPrecision t_vecPrec, AMGX_MatPrecision t_matPrec, AMGX_IndPrecision t_indPrec>
void Scaler<TemplateConfig<AMGX_host, t_vecPrec, t_matPrec, t_indPrec> >::scaleVector(Vector<TConfig_h> &v, ScaleDirection scaleOrUnscale, ScaleSide leftOrRight)
{
}

// ScalerFactory definitions
template <class TConfig>
std::map<std::string, ScalerFactory<TConfig>*> &
ScalerFactory<TConfig>::getFactories( )
{
    static std::map<std::string, ScalerFactory<TConfig>*> s_factories;
    return s_factories;
}

template<class TConfig>
void ScalerFactory<TConfig>::registerFactory(std::string name, ScalerFactory<TConfig> *f)
{
    std::map<std::string, ScalerFactory<TConfig>*> &factories = getFactories( );
    typename std::map<std::string, ScalerFactory<TConfig> *>::const_iterator it = factories.find(name);

    if (it != factories.end())
    {
        std::string error = "ScalerFactory '" + name + "' has already been registered\n";
        FatalError(error.c_str(), AMGX_ERR_CORE);
    }

    factories[name] = f;
}

template<class TConfig>
void ScalerFactory<TConfig>::unregisterFactory(std::string name)
{
    std::map<std::string, ScalerFactory<TConfig>*> &factories = getFactories( );
    typename std::map<std::string, ScalerFactory<TConfig> *>::iterator it = factories.find(name);

    if (it == factories.end())
    {
        std::string error = "ScalerFactory '" + name + "' has not been registered\n";
        FatalError(error.c_str(), AMGX_ERR_CORE);
    }

    ScalerFactory<TConfig> *factory = it->second;
    assert( factory != NULL );
    delete factory;
    factories.erase(it);
}

template<class TConfig>
void ScalerFactory<TConfig>::unregisterFactories( )
{
    std::map<std::string, ScalerFactory<TConfig>*> &factories = getFactories( );
    typename std::map<std::string, ScalerFactory<TConfig> *>::iterator it = factories.begin( );

    for ( ; it != factories.end( ) ; )
    {
        ScalerFactory<TConfig> *factory = it->second;
        assert( factory != NULL );
        it++;
        delete factory;
    }

    factories.clear( );
}

template<class TConfig>
Scaler<TConfig> *ScalerFactory<TConfig>::allocate(AMG_Config &cfg, const std::string &cfg_scope)
{
    std::map<std::string, ScalerFactory<TConfig>*> &factories = getFactories( );
    std::string scaler = cfg.getParameter<std::string>("scaling", cfg_scope);
    typename std::map<std::string, ScalerFactory<TConfig> *>::const_iterator it = factories.find(scaler);

    if (it == factories.end())
    {
        std::string error = "ScalerFactory '" + scaler + "' has not been registered\n";
        FatalError(error.c_str(), AMGX_ERR_CORE);
    }

    return it->second->create(cfg, cfg_scope);
};

/****************************************
 * Explict instantiations
 ***************************************/
#define AMGX_CASE_LINE(CASE) template class Scaler<TemplateMode<CASE>::Type>;
AMGX_FORALL_BUILDS(AMGX_CASE_LINE)
AMGX_FORCOMPLEX_BUILDS(AMGX_CASE_LINE)
#undef AMGX_CASE_LINE

#define AMGX_CASE_LINE(CASE) template class ScalerFactory<TemplateMode<CASE>::Type>;
AMGX_FORALL_BUILDS(AMGX_CASE_LINE)
AMGX_FORCOMPLEX_BUILDS(AMGX_CASE_LINE)
#undef AMGX_CASE_LINE

} // namespace amgx

