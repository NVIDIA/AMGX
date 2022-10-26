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

#include <matrix_coloring/matrix_coloring.h>
#include <blas.h>
#include <basic_types.h>
#include <error.h>
#include <types.h>
#include <thrust/transform_reduce.h>
#include <thrust/functional.h>
#include <thrust/device_ptr.h>
#include <thrust/sort.h>
#include <thrust/count.h>
#include <thrust/binary_search.h>
#include <assert.h>
#include <sm_utils.inl>
#include <algorithm>

#include <amgx_types/util.h>
#include <amgx_types/math.h>

using namespace std;
namespace amgx
{
/***************************************
 * Source Definitions
 ***************************************/

template<class TConfig>
MatrixColoring<TConfig>::MatrixColoring(AMG_Config &cfg, const std::string &cfg_scope) :
    m_num_colors(0), m_row_colors(0), m_sorted_rows_by_color(0), m_offsets_rows_per_color(0), m_ref_count(1), m_boundary_coloring(SYNC_COLORS), m_halo_coloring(FIRST)
{
    m_coloring_level = cfg.getParameter<int>("coloring_level", cfg_scope);
    m_boundary_coloring = cfg.getParameter<ColoringType>("boundary_coloring", cfg_scope);
    m_halo_coloring = cfg.getParameter<ColoringType>("halo_coloring", cfg_scope);
}

template<class TConfig>
MatrixColoring<TConfig>::~MatrixColoring()
{
}

__global__ void findSeparation(INDEX_TYPE *rows_by_color, INDEX_TYPE *offsets_by_color, INDEX_TYPE *separation, INDEX_TYPE boundary, INDEX_TYPE num_colors, INDEX_TYPE num_rows)
{
    int block_offset = blockIdx.x * (blockDim.x / 32) * 31; //each warp does 31 rows, 1 on the left side is redundant
    int lane = threadIdx.x % 32;
    int element = block_offset + (threadIdx.x / 32) * 31 + lane - 1;

    while (element < num_rows)
    {
        int color = 0;
        int row = -1;

        if (element != -1)
        {
            row = rows_by_color[element];

            while ((color < num_colors) && ((element < offsets_by_color[color]) || (element >= offsets_by_color[color + 1]))) { color ++; }

            if ((element == offsets_by_color[color]) && (row >= boundary)) { separation[color] = element; } //special case when first row of color is immediately a boundary node

            if ((element == offsets_by_color[color + 1] - 1) && (row < boundary)) { separation[color] = element + 1; } //special case when I am the last, but I am still not a boundary
        }

        unsigned int result = utils::ballot(row >= boundary, utils::activemask());

        //if (result>0) printf("%x\n", result);
        if (lane > 0 && row >= boundary && ((result >> (lane - 1)) & 1) == 0) { separation[color] = element; }

        element += gridDim.x * (blockDim.x / 32) * 31;
    }
}

//prints how many edges fail to obey coloring property
//if the optional aggregates parameter is specified, it also measures the downwind coloring property:
//for each incoming edge (j,i), where i and j share the same aggregate, holds: color(j) < color(i)
template <class TConfig>
void MatrixColoring<TConfig>::assertColoring( Matrix<TConfig> &A, IVector &aggregates )
{
    IndexType numRows = A.get_num_rows();
    IndexType nnz = A.get_num_nz();
    IndexType blocksize = A.get_block_dimx() * A.get_block_dimy();
    IVector &coloring = this->m_row_colors;
    bool check_downwind = aggregates.size() == A.get_num_rows();
    //allocate host memory
    IndexType *ia = new IndexType[numRows + 1];
    IndexType *ja = new IndexType[nnz];
    ValueType *aa = new ValueType[nnz * blocksize];
    IndexType *color = new IndexType[numRows];
    //copy to host
    cudaMemcpy( ia, A.row_offsets.raw(), sizeof(IndexType) * (numRows + 1), cudaMemcpyDeviceToHost );
    cudaMemcpy( ja, A.col_indices.raw(), sizeof(IndexType)*nnz, cudaMemcpyDeviceToHost );
    cudaMemcpy( aa, A.values.raw(), sizeof(ValueType)*blocksize * nnz, cudaMemcpyDeviceToHost );
    cudaMemcpy( color, coloring.raw(), sizeof(IndexType)*numRows, cudaMemcpyDeviceToHost );
    IndexType *agg = new IndexType[numRows];

    if ( check_downwind )
    {
        cudaMemcpy( agg, aggregates.raw(), sizeof(IndexType)*numRows, cudaMemcpyDeviceToHost );
    }

    //count how many nodes have a color
    IndexType *color_used = new IndexType[numRows];

    for (IndexType i = 0; i < numRows; i++)
    {
        color_used[i] = 0;
    }

    for (IndexType i = 0; i < numRows; i++)
    {
        if ( color[i] >= 0 && color[i] < numRows )
        {
            color_used[color[i]]++;
        }
        else
        {
            std::cout << "color out of range: color[" << i << "] = " << color[i] << std::endl;
        }
    }

    // count violations of these two properties:
    // 1. locally downwind: for incoming edges (j,i) in same aggregate: color(j) < color(i)
    // 2. valid coloring: for neighbors j: color(j) != color(i)
    int violation_1 = 0;
    int property_1 = 0;
    int violation_2 = 0;
    int property_2 = 0;
    int inner_edges = 0;

    //note: property 1 cannot be enforeced all the time. Each cycle for example will violate it regardless of the coloring.
    for (IndexType i = 0; i < numRows; i++)
    {
        for (IndexType ii = ia[i]; ii < ia[i + 1]; ii++)
        {
            IndexType j = ja[ii];

            if ( j == i )
            {
                continue;
            }

            //check coloring property
            if ( color[j] == color[i] )
            {
                violation_2++;
            }

            property_2++;

            if ( check_downwind && agg[j] == agg[i] )
            {
                //look for transpose edge to decide outgoing or not
                bool outgoing = true;

                for (IndexType jj = ia[j]; jj < ia[j + 1]; jj++)
                {
                    //found
                    if ( ja[jj] == i )
                    {
                        ValueType weight = types::util<ValueType>::get_zero();

                        for (IndexType iii = ii * blocksize; iii < (ii + 1)*blocksize; iii++)
                        {
                            weight = weight + aa[iii] * aa[iii];
                        }

                        ValueType counter_weight = types::util<ValueType>::get_zero();

                        for (IndexType jjj = jj * blocksize; jjj < (jj + 1)*blocksize; jjj++)
                        {
                            counter_weight = counter_weight + aa[jjj] * aa[jjj];
                        }

                        outgoing = types::util<ValueType>::abs(weight) > types::util<ValueType>::abs(counter_weight);
                        break;
                    }
                }

                //outgoing -> check downwind property
                if ( outgoing )
                {
                    if ( color[j] <= color[i] )
                    {
                        violation_1++;
                    }

                    property_1++;
                }

                inner_edges++;
            }
        }
    }

    //tell results
    if ( check_downwind )
    {
        std::cout << 200 * property_1 / double(inner_edges) << "% of all edges inside an aggregate are directed" << std::endl;

        if ( property_1 > 0 )
        {
            std::cout << 100 * violation_1 / double(property_1) << "% of all outgoing edges inside an aggregate are not colored downwind" << std::endl;
        }
    }

    std::cout << 100 * violation_2 / double(property_2) << "% of all edges violated coloring property" << std::endl;
    std::cout << "number of nodes that use this color:" << std::endl;

    for (IndexType i = 0; i < numRows; i++)
        if ( color_used[i] > 0 )
        {
            std::cout << i << ": " << color_used[i] << std::endl;
        }

    //free!
    delete [] ia;
    delete [] ja;
    delete [] aa;
    delete [] agg;
    delete [] color;
    delete [] color_used;
}
template<class TConfig>
void MatrixColoring<TConfig>::createColorArrays(Matrix<TConfig> &A)
{
    ViewType old = A.currentView();
    A.setViewExterior();
    int num_rows = A.get_num_rows();

    //Disabled since currently we are not doing halo exchanges during colored execution
    /*typedef TemplateConfig<AMGX_host,AMGX_vecInt,matPrec,indPrec> hvector_type;
    typedef Vector<hvector_type> HVector;
    if (!A.is_matrix_singleGPU()) {
      HVector num_colors(1);
      std::vector<HVector> partition_num_colors(0);
      num_colors[0] = m_num_colors;
      A.manager->getComms()->global_reduce(partition_num_colors, num_colors, A, 6332);
      int max_partition_colors = 0;
      for (int i = 0; i < partition_num_colors.size(); i++)
          max_partition_colors = std::max(partition_num_colors[i][0],max_partition_colors);
      m_num_colors = max_partition_colors;
    }*/

    if (m_halo_coloring == LAST)
    {
        amgx::thrust::fill(m_row_colors.begin() + num_rows, m_row_colors.end(), m_num_colors);
        cudaCheckError();
    }

    IVector offsets_rows_per_color;

    if (m_offsets_rows_per_color.size() == 0)
    {
        // Sort the vertices based o their color
        m_sorted_rows_by_color.resize(num_rows);
        // Copy row colors
        IVector row_colors(m_row_colors);
        amgx::thrust::sequence(m_sorted_rows_by_color.begin(), m_sorted_rows_by_color.end());
        amgx::thrust::sort_by_key(row_colors.begin(), row_colors.begin() + num_rows, m_sorted_rows_by_color.begin());
        cudaCheckError();
        // Compute the offset for each color
        offsets_rows_per_color.resize(m_num_colors + 1);
        m_offsets_rows_per_color.resize(m_num_colors + 1);
        // Compute interior-exterior separation for every color
        m_offsets_rows_per_color_separation.resize(m_num_colors);
        //m_offsets_rows_per_color_separation_halo.resize(m_num_colors);
        amgx::thrust::lower_bound(row_colors.begin(),
                            row_colors.begin() + num_rows,
                            amgx::thrust::counting_iterator<IndexType>(0),
                            amgx::thrust::counting_iterator<IndexType>(offsets_rows_per_color.size()),
                            offsets_rows_per_color.begin());
        // Copy from device to host
        m_offsets_rows_per_color = offsets_rows_per_color;
        cudaCheckError();
    }
    else
    {
        m_offsets_rows_per_color_separation.resize(m_num_colors);
    }

    cudaCheckError();

    if (!A.is_matrix_singleGPU() && (A.getViewExterior() != A.getViewInterior()))
    {
        A.setViewInterior();
        int separation = A.get_num_rows();

        if (TConfig::memSpace == AMGX_host)
        {
            for (int i = 0; i < m_num_colors; i++)
            {
                m_offsets_rows_per_color_separation[i] =  m_offsets_rows_per_color[i]
                        + (amgx::thrust::lower_bound(m_sorted_rows_by_color.begin() + m_offsets_rows_per_color[i],
                                               m_sorted_rows_by_color.begin() + m_offsets_rows_per_color[i + 1],
                                               separation)
                           - (m_sorted_rows_by_color.begin() + m_offsets_rows_per_color[i]));
            }

            cudaCheckError();
        }
        // this is not a proper search, rather we look at every single element. But it is still a lot faster than the above (~10*)
        else
        {
            IVector separation_offsets_rows_per_color(m_num_colors);
            int size = num_rows;
            int num_blocks = min(4096, (size + 123) / 124);
            findSeparation <<< num_blocks, 128>>>(m_sorted_rows_by_color.raw(), offsets_rows_per_color.raw(), separation_offsets_rows_per_color.raw(), separation, m_num_colors, num_rows);
            amgx::thrust::copy(separation_offsets_rows_per_color.begin(), separation_offsets_rows_per_color.end(), m_offsets_rows_per_color_separation.begin());
            cudaCheckError();

            for (int i = 0; i < m_num_colors; i++)
            {
                if (this->m_offsets_rows_per_color[i] == this->m_offsets_rows_per_color[i + 1])
                {
                    this->m_offsets_rows_per_color_separation[i] = this->m_offsets_rows_per_color[i + 1];
                }
            }
        }
    }
    else
    {
        amgx::thrust::copy(m_offsets_rows_per_color.begin() + 1, m_offsets_rows_per_color.end(), m_offsets_rows_per_color_separation.begin());
        cudaCheckError();
    }

    A.setView(old);
}

template<class TConfig>
std::map<std::string, MatrixColoringFactory<TConfig>*> &
MatrixColoringFactory<TConfig>::getFactories( )
{
    static std::map<std::string, MatrixColoringFactory<TConfig> *> s_factories;
    return s_factories;
}

template<class TConfig>
void MatrixColoringFactory<TConfig>::registerFactory(string name, MatrixColoringFactory<TConfig> *f)
{
    std::map<std::string, MatrixColoringFactory<TConfig> *> &factories = getFactories( );
    typename map<string, MatrixColoringFactory<TConfig> *>::iterator it = factories.find(name);

    if (it != factories.end())
    {
        string error = "MatrixColoringFactory '" + name + "' has already been registered\n";
        FatalError(error.c_str(), AMGX_ERR_CORE);
    }

    factories[name] = f;
}

template<class TConfig>
void MatrixColoringFactory<TConfig>::unregisterFactory(std::string name)
{
    std::map<std::string, MatrixColoringFactory<TConfig>*> &factories = getFactories( );
    typename std::map<std::string, MatrixColoringFactory<TConfig> *>::iterator it = factories.find(name);

    if (it == factories.end())
    {
        std::string error = "MatrixColoringFactory '" + name + "' has not been registered\n";
        FatalError(error.c_str(), AMGX_ERR_CORE);
    }

    MatrixColoringFactory<TConfig> *factory = it->second;
    assert( factory != NULL );
    delete factory;
    factories.erase(it);
}

template<class TConfig>
void MatrixColoringFactory<TConfig>::unregisterFactories( )
{
    std::map<std::string, MatrixColoringFactory<TConfig>*> &factories = getFactories( );
    typename map<string, MatrixColoringFactory<TConfig> *>::iterator it = factories.begin( );

    for ( ; it != factories.end( ) ; )
    {
        MatrixColoringFactory<TConfig> *factory = it->second;
        assert( factory != NULL );
        it++;
        delete factory;
    }

    factories.clear( );
}

template<class TConfig>
MatrixColoring<TConfig> *MatrixColoringFactory<TConfig>::allocate(AMG_Config &cfg, const std::string &cfg_scope)
{
    std::map<std::string, MatrixColoringFactory<TConfig> *> &factories = getFactories( );
    string matrix_coloring_scheme = cfg.getParameter<string>("matrix_coloring_scheme", cfg_scope);
    typename map<string, MatrixColoringFactory<TConfig> *>::const_iterator it = factories.find(matrix_coloring_scheme);

    if (it == factories.end())
    {
        string error = "MatrixColoringFactory '" + matrix_coloring_scheme + "' has not been registered\n";
        FatalError(error.c_str(), AMGX_ERR_CORE);
    }

    return it->second->create(cfg, cfg_scope);
};

/****************************************
 * Explict instantiations
 ***************************************/
#define AMGX_CASE_LINE(CASE) template class MatrixColoring<TemplateMode<CASE>::Type>;
AMGX_FORALL_BUILDS(AMGX_CASE_LINE)
AMGX_FORCOMPLEX_BUILDS(AMGX_CASE_LINE)
#undef AMGX_CASE_LINE

#define AMGX_CASE_LINE(CASE) template class MatrixColoringFactory<TemplateMode<CASE>::Type>;
AMGX_FORALL_BUILDS(AMGX_CASE_LINE)
AMGX_FORCOMPLEX_BUILDS(AMGX_CASE_LINE)
#undef AMGX_CASE_LINE
}

