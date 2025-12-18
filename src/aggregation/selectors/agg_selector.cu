// SPDX-FileCopyrightText: 2011 - 2025 NVIDIA CORPORATION. All Rights Reserved.
//
// SPDX-License-Identifier: BSD-3-Clause

#include <types.h>
#include <aggregation/selectors/agg_selector.h>
#include <transpose.h>

#include <amgx_types/util.h>

#include <assert.h>

namespace amgx
{

namespace aggregation
{

template<class T_Config>
void Selector<T_Config>::renumberAndCountAggregates(IVector &aggregates, IVector &aggregates_global, const IndexType num_block_rows, IndexType &num_aggregates)
{
    // renumber aggregates
    IVector scratch(num_block_rows + 1);

    if ( num_block_rows != aggregates.size() )
    {
        // we are in a distributed environment
        aggregates_global.resize(aggregates.size());
        amgx::thrust::copy(aggregates.begin(), aggregates.begin() + num_block_rows, aggregates_global.begin());
    }

    // set scratch[aggregates[i]] = 1
    thrust::fill(amgx::thrust::make_permutation_iterator(scratch.begin(), aggregates.begin()),
                 amgx::thrust::make_permutation_iterator(scratch.begin(), aggregates.begin() + num_block_rows), 1);
    // do prefix sum on scratch
    thrust_wrapper::exclusive_scan<T_Config::memSpace>(scratch.begin(), scratch.end(), scratch.begin());
    // aggregates[i] = scratch[aggregates[i]]
    amgx::thrust::copy(amgx::thrust::make_permutation_iterator(scratch.begin(), aggregates.begin()),
                 amgx::thrust::make_permutation_iterator(scratch.begin(), aggregates.begin() + num_block_rows),
                 aggregates.begin());
    // update number of aggregates
    num_aggregates = scratch[scratch.size() - 1];
    cudaCheckError();
}


template<typename IndexType>
__global__
void countRowSizes( const IndexType *R_row_offsets, IndexType *size_array, int num_aggregates)
{
    int tid = threadIdx.x + blockDim.x * blockIdx.x;

    while ( tid < num_aggregates )
    {
        int size = R_row_offsets[tid + 1] - R_row_offsets[tid];
        atomicAdd( &size_array[size], 1 );
        tid += gridDim.x * blockDim.x;
    }
}

//prints aggregate info:
//counts aggregates grouped by size
template<class TConfig>
void Selector<TConfig>::printAggregationInfo(const IVector &aggregates, const IVector &aggregates_global, const IndexType num_aggregates) const
{
    //build restriction row offsets
    Matrix<TConfig> P;
    P.set_initialized(0);
    P.addProps(CSR);
    P.delProps(COO);
    P.delProps(DIAG);
    P.setColsReorderedByColor(false);
    P.resize( 0, 0, 0, 1, 1); //make sure the matrix is empty
    Matrix<TConfig> R;
    R.set_initialized(0);
    R.addProps(CSR);
    R.delProps(COO);
    R.delProps(DIAG);
    R.setColsReorderedByColor(false);
    R.resize( 0, 0, 0, 1, 1);
    int num_rows = aggregates.size();
    //set correct sizes.
    P.row_offsets.resize( num_rows + 1 );
    P.values.resize( num_rows, types::util<typename MVector::value_type>::get_one());
    P.col_indices.resize( num_rows );
    //setup offset array.
    thrust_wrapper::sequence<TConfig::memSpace>( P.row_offsets.begin(), P.row_offsets.end() );
    //swap in aggregates
    amgx::thrust::copy( aggregates.begin(), aggregates.end(), P.col_indices.begin() );
    cudaCheckError();
    //inform P about its size
    P.resize( num_rows, num_aggregates, num_rows, 1, 1, false ); //declare scalar, otherwise transpose won't work
    //P is ready
    P.set_initialized(1);
    //do the transpose
    R.set_initialized(0);
    transpose( P, R ); //note: this won't change P
    //ok now we have row offsets.
    //we are ready to count the row sizes
    IVector size_array;
    size_array.resize(num_rows + 1, 0 );
    const int threads_per_block = 256;
    const int num_blocks = std::min( (int)AMGX_GRID_MAX_SIZE, (int)(num_rows - 1) / threads_per_block + 1 );
    //count
    countRowSizes <<< num_blocks, threads_per_block, 0, 0>>>( R.row_offsets.raw(), size_array.raw(), num_aggregates );
    cudaCheckError();
    cudaDeviceSynchronize();
    cudaCheckError();
    //copy to host and print
    amgx::thrust::host_vector<int> size_array_host;
    size_array_host.resize(size_array.size() );
    amgx::thrust::copy( size_array.begin(), size_array.end(), size_array_host.begin() );
    std::cout << "number of nodes " << num_rows << std::endl;
    std::cout << "number of aggregates by size" << std::endl;

    for (int i = 1; i < size_array_host.size(); i++)
    {
        if ( size_array_host[i] > 0 )
        {
            std::cout << "size " << i << ": " << size_array_host[i] << std::endl;
        }
    }

    std::cout << "total " << (num_aggregates - size_array_host[0]) << std::endl;
}

template<class TConfig>
void Selector<TConfig>::assertAggregates( const IVector &aggregates, int numAggregates )
{
    int *res_sizes = new int[numAggregates];

    for ( int i = 0; i < numAggregates; i++ )
    {
        res_sizes[i] = 0;
    }

    int *h_aggregates = new int[aggregates.size()];
    cudaMemcpy( h_aggregates, aggregates.raw(), sizeof(int)*aggregates.size(), cudaMemcpyDeviceToHost );
    cudaCheckError();

    for ( int i = 0; i < aggregates.size(); i++ )
    {
        if ( h_aggregates[i] < 0 || h_aggregates[i] >= numAggregates )
        {
            std::cout << "Error: aggregates[" << i << "]=" << h_aggregates[i] << " but numAggregates=" << numAggregates << std::endl;
        }
        else
        {
            res_sizes[h_aggregates[i]]++;
        }
    }

    for ( int i = 0; i < numAggregates; i++ )
    {
        if ( res_sizes[i] == 0 )
        {
            std::cout << "Error: aggregate " << i << " is empty" << std::endl;
        }
    }

    std::cout << "assertAggregates done." << std::endl;
}

template<class TConfig>
void Selector<TConfig>::assertRestriction( const IVector &R_row_offsets, const IVector &R_col_indices, const IVector &aggregates )
{
    int *r_ia = new int[R_row_offsets.size()];
    int *r_ja = new int[R_col_indices.size()];
    int *agg = new int[aggregates.size()];
    int *used_col = new int[aggregates.size()];

    for ( int i = 0; i < aggregates.size(); i++ )
    {
        used_col[i] = 0;
    }

    cudaError_t cuda_rc = cudaMemcpy( r_ia, R_row_offsets.raw(), sizeof(int)*R_row_offsets.size(), cudaMemcpyDeviceToHost );
    if (cuda_rc != cudaSuccess)
    {
        FatalError("cudaMemcpy R_row_offsets D2H failed in agg_selector", AMGX_ERR_CUDA_FAILURE);
    }
    cuda_rc = cudaMemcpy( r_ja, R_col_indices.raw(), sizeof(int)*R_col_indices.size(), cudaMemcpyDeviceToHost );
    if (cuda_rc != cudaSuccess)
    {
        FatalError("cudaMemcpy R_col_indices D2H failed in agg_selector", AMGX_ERR_CUDA_FAILURE);
    }
    cuda_rc = cudaMemcpy( agg, aggregates.raw(), sizeof(int)*aggregates.size(), cudaMemcpyDeviceToHost );
    if (cuda_rc != cudaSuccess)
    {
        FatalError("cudaMemcpy aggregates D2H failed in agg_selector", AMGX_ERR_CUDA_FAILURE);
    }

    for ( int i = 0; i < R_row_offsets.size() - 1; i++ )
    {
        for ( int ii = r_ia[i]; ii < r_ia[i + 1]; ii++ )
        {
            int j = r_ja[ii];
            used_col[j]++;

            if ( used_col[j] > 1 )
            {
                std::cout << "column " << j << " is present at least " << used_col[j] << " times" << std::endl;
            }

            if ( j < 0 || j >= aggregates.size() )
            {
                std::cout << "Error: j out of bounds, j = " << j << " and numRows = " << aggregates.size() << std::endl;
            }
            else if  ( agg[j] != i )
            {
                std::cout << "Error: agg[" << j << "] = " << agg[j] << " != " << i << std::endl;
            }
        }
    }

    std::cout << "assert restriction done" << std::endl;
}



template<class T_Config>
std::map<std::string, SelectorFactory<T_Config>*> &
SelectorFactory<T_Config>::getFactories( )
{
    static std::map<std::string, SelectorFactory<T_Config>*> s_factories;
    return s_factories;
}

template<class T_Config>
void SelectorFactory<T_Config>::registerFactory(std::string name, SelectorFactory<T_Config> *f)
{
    std::map<std::string, SelectorFactory<T_Config>*> &factories = getFactories( );
    typename std::map<std::string, SelectorFactory<T_Config> *>::const_iterator it = factories.find(name);

    if (it != factories.end())
    {
        std::string error = "SelectorFactory '" + name + "' has already been registered\n";
        FatalError(error.c_str(), AMGX_ERR_CORE);
    }

    factories[name] = f;
}

template<class T_Config>
void SelectorFactory<T_Config>::unregisterFactory(std::string name)
{
    std::map<std::string, SelectorFactory<T_Config>*> &factories = getFactories( );
    typename std::map<std::string, SelectorFactory<T_Config> *>::iterator it = factories.find(name);

    if (it == factories.end())
    {
        std::string error = "SelectorFactory '" + name + "' has not been registered\n";
        FatalError(error.c_str(), AMGX_ERR_CORE);
    }

    SelectorFactory<T_Config> *factory = it->second;
    assert( factory != NULL );
    delete factory;
    factories.erase(it);
}

template<class T_Config>
void SelectorFactory<T_Config>::unregisterFactories( )
{
    std::map<std::string, SelectorFactory<T_Config>*> &factories = getFactories( );
    typename std::map<std::string, SelectorFactory<T_Config> *>::iterator it = factories.begin( );

    for ( ; it != factories.end( ) ; )
    {
        SelectorFactory<T_Config> *factory = it->second;
        assert( factory != NULL );
        it++;
        delete factory;
    }

    factories.clear( );
}

template<class T_Config>
Selector<T_Config> *SelectorFactory<T_Config>::allocate(AMG_Config &cfg, const std::string &current_scope)
{
    std::map<std::string, SelectorFactory<T_Config>*> &factories = getFactories( );
    int agg_lvl_change = cfg.AMG_Config::template getParameter<int>("fine_levels", current_scope);
    std::string selector;
    selector = cfg.getParameter<std::string>("selector", current_scope);
    typename std::map<std::string, SelectorFactory<T_Config> *>::const_iterator it = factories.find(selector);

    if (it == factories.end())
    {
        std::string error = "SelectorFactory '" + selector + "' has not been registered\n";
        FatalError(error.c_str(), AMGX_ERR_CORE);
    }

    return it->second->create(cfg, current_scope);
};



//****************************************
// Explict instantiations
// **************************************
#define AMGX_CASE_LINE(CASE) template class Selector<TemplateMode<CASE>::Type>;
AMGX_FORALL_BUILDS(AMGX_CASE_LINE)
AMGX_FORCOMPLEX_BUILDS(AMGX_CASE_LINE)
#undef AMGX_CASE_LINE

#define AMGX_CASE_LINE(CASE) template class SelectorFactory<TemplateMode<CASE>::Type>;
AMGX_FORALL_BUILDS(AMGX_CASE_LINE)
AMGX_FORCOMPLEX_BUILDS(AMGX_CASE_LINE)
#undef AMGX_CASE_LINE

}

}
