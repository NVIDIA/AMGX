// SPDX-FileCopyrightText: 2011 - 2025 NVIDIA CORPORATION. All Rights Reserved.
//
// SPDX-License-Identifier: BSD-3-Clause

#include <types.h>
#include <aggregation/coarseAgenerators/coarse_A_generator.h>
#include <thrust/remove.h>
#include <thrust/iterator/transform_iterator.h>
#include <thrust/iterator/constant_iterator.h>
#include <thrust/count.h>
#include <thrust_wrapper.h>

#include <thrust/extrema.h>

#include <assert.h>
#include <matrix.h>

namespace amgx
{

namespace aggregation
{

// ---------------------------------------------------------------------
// Method to print the distribution of number of nonzeros in matrix Ac
// ---------------------------------------------------------------------
template <class T_Config>
void CoarseAGenerator<T_Config>::printNonzeroStats(const typename Matrix<T_Config>::IVector &Ac_row_offsets, const int num_aggregates)
{
    // Printing the number of nonzeros per row
    Vector<typename TConfig::template setVecPrec<AMGX_vecBool>::Type> temporary(num_aggregates, 0);
    int max_nonzero = *amgx::thrust::max_element(Ac_row_offsets.begin(), Ac_row_offsets.end()) + 1;
    amgx_printf("\nnew level, max number of nonzeros per row = %d\n", max_nonzero);
    double *breakdown = new double[max_nonzero];

    for (int i = 0; i < max_nonzero; i++)
    {
        thrust_wrapper::transform<T_Config::memSpace>(Ac_row_offsets.begin(), Ac_row_offsets.end(), amgx::thrust::make_constant_iterator(i + 1), temporary.begin(), amgx::thrust::less<int>());
        breakdown[i] = 1.0 * (amgx::thrust::count(temporary.begin(), temporary.end(), true)) / num_aggregates;
        amgx_printf("Percentage of rows with less than %d nonzeros is %d\n", (i + 1), breakdown[i]);
    }

    delete[] breakdown;
}

template<class T_Config>
std::map<std::string, CoarseAGeneratorFactory<T_Config>*> &
CoarseAGeneratorFactory<T_Config>::getFactories( )
{
    static std::map<std::string, CoarseAGeneratorFactory<T_Config>*> s_factories;
    return s_factories;
}

template<class T_Config>
void CoarseAGeneratorFactory<T_Config>::registerFactory(std::string name, CoarseAGeneratorFactory<T_Config> *f)
{
    std::map<std::string, CoarseAGeneratorFactory<T_Config>*> &factories = getFactories( );
    typename std::map<std::string, CoarseAGeneratorFactory<T_Config> *>::const_iterator it = factories.find(name);

    if (it != factories.end())
    {
        std::string error = "CoarseAGeneratorFactory '" + name + "' has already been registered\n";
        FatalError(error.c_str(), AMGX_ERR_CORE);
    }

    factories[name] = f;
}

template<class T_Config>
void CoarseAGeneratorFactory<T_Config>::unregisterFactory(std::string name)
{
    std::map<std::string, CoarseAGeneratorFactory<T_Config>*> &factories = getFactories( );
    typename std::map<std::string, CoarseAGeneratorFactory<T_Config> *>::iterator it = factories.find(name);

    if (it == factories.end())
    {
        std::string error = "CoarseAGeneratorFactory '" + name + "' has not been registered\n";
        FatalError(error.c_str(), AMGX_ERR_CORE);
    }

    CoarseAGeneratorFactory<T_Config> *factory = it->second;
    assert( factory != NULL );
    delete factory;
    factories.erase(it);
}

template<class T_Config>
void CoarseAGeneratorFactory<T_Config>::unregisterFactories( )
{
    std::map<std::string, CoarseAGeneratorFactory<T_Config>*> &factories = getFactories( );
    typename std::map<std::string, CoarseAGeneratorFactory<T_Config> *>::iterator it = factories.begin( );

    for ( ; it != factories.end( ) ; )
    {
        CoarseAGeneratorFactory<T_Config> *factory = it->second;
        assert( factory != NULL );
        it++;
        delete factory;
    }

    factories.clear( );
}

template<class T_Config>
CoarseAGenerator<T_Config> *CoarseAGeneratorFactory<T_Config>::allocate(AMG_Config &cfg, const std::string &cfg_scope)
{
    std::map<std::string, CoarseAGeneratorFactory<T_Config>*> &factories = getFactories( );
    int agg_lvl_change = cfg.AMG_Config::template getParameter<int>("fine_levels", cfg_scope);
    std::string generator;
    generator = cfg.getParameter<std::string>("coarseAgenerator", cfg_scope);
    typename std::map<std::string, CoarseAGeneratorFactory<T_Config> *>::const_iterator it = factories.find(generator);

    if (it == factories.end())
    {
        std::string error = "CoarseAGeneratorFactory '" + generator + "' has not been registered\n";
        FatalError(error.c_str(), AMGX_ERR_CORE);
    }

    return it->second->create(cfg, cfg_scope);
};

// ---------------------------------
// Explict instantiations
// ---------------------------------
#define AMGX_CASE_LINE(CASE) template class CoarseAGenerator<TemplateMode<CASE>::Type>;
AMGX_FORALL_BUILDS(AMGX_CASE_LINE)
AMGX_FORCOMPLEX_BUILDS(AMGX_CASE_LINE)
#undef AMGX_CASE_LINE

#define AMGX_CASE_LINE(CASE) template class CoarseAGeneratorFactory<TemplateMode<CASE>::Type>;
AMGX_FORALL_BUILDS(AMGX_CASE_LINE)
AMGX_FORCOMPLEX_BUILDS(AMGX_CASE_LINE)
#undef AMGX_CASE_LINE

}
}
